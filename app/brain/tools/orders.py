"""Orders and rides through the task runtime (app/tasks): start, inputs, cancel, status, feedback, boundaries."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

from app.brain.tools.base import FOR_WHOM, ToolRefused, TurnCtx, tool
from app.brain.tools.places import (
    _delivery_place,
    _pickup_words_asked,
    _pin_place,
    _ride_end_place,
    _ride_pin_first,
    _save_pin,
    _shared_pin,
    maps_link_pin,
)
from app.care import store
from app.core import clock


def _tools_pkg():
    """The tools package: tests replace task_agent and _follow on it, so they are read from there."""
    import sys

    return sys.modules["app.brain.tools"]


# ── tasks: orders and rides, carried through by the task runtime ────────────────

_agent = None


def task_agent():
    """The browser agent used for stops and cancels during a turn (the sim swaps in a fake)."""
    global _agent
    if _agent is None:
        from app.tasks.browser_use import BrowserUseCloud

        _agent = BrowserUseCloud()
    return _agent


def set_task_agent(agent) -> None:
    global _agent
    _agent = agent


async def _care_limits(ctx: TurnCtx) -> tuple[list[str], list[str]]:
    rows = await store.facts(ctx.session, ctx.family_id, ctx.elder_id, domains=["allergy", "no_order"], statuses=("active",))
    allergies = [r.value.get("allergen") or r.key.split(":", 1)[1] for r in rows if r.domain == "allergy"]
    never = [r.value.get("item") or r.key.split(":", 1)[1].replace("_", " ") for r in rows if r.domain == "no_order"]
    return allergies, never


async def _task(ctx: TurnCtx, task_id: str):
    from app.tasks.models import Task

    try:
        tid = uuid.UUID(task_id)
    except ValueError as exc:
        raise ToolRefused("Use the task id shown in ACTIVE TASKS.") from exc
    # populate_existing: the turn already loaded this task for ACTIVE TASKS; re-read it under the lock so a tick's
    # newer state (timeout, new cart, price change) is not overwritten with the old copy.
    task = await ctx.session.get(Task, tid, with_for_update=True, populate_existing=True)
    if not task or task.family_id != ctx.family_id:
        raise ToolRefused("No such task for this family.")
    return task


@tool(
    "start_task",
    "Start an order or a ride on one of the family's own accounts. It runs in the background until it is placed or "
    "cancelled: an order looks the items up (on the store they named, the stores they named, or with no store named the "
    "usual stores of that kind) and a task update brings what every store found, numbered, with your suggestion marked, for "
    "them to pick; then it logs in if the store needs it (the login code is asked from the person the family set for codes), "
    "builds the cart and comes back with ONE confirm: items and total. Do not ask which product or store before the options "
    "come. A ride is the same: the fares from the app they named (or the apps they named; with no app named, or 'the "
    "cheapest', every ride app) come back numbered for them to pick, then after the login the one confirm with the fare. "
    "Cash only. Allergies and the never-order list are checked first. Tell the person in a few words that you are on it; you "
    "get a task update when someone has to answer. For 'the usual' or 'same as last time', look at past_orders first and use "
    "the exact item names and service from there. If they did not name a store for an order, leave service out and give "
    "category (groceries: Blinkit, Instamart, Zepto; medicines: Apollo, 1mg, PharmEasy; food: Swiggy, Zomato); if they named "
    "several stores or apps to look on, give them as services. The first call for an order only returns the delivery address "
    "to confirm: ask them, and call start_task again after they say yes (with area if they named another saved place).",
    {
        "service": {"type": "string", "enum": ["swiggy", "instamart", "zepto", "blinkit", "zomato", "apollo", "1mg", "pharmeasy", "uber", "ola", "rapido"],
                    "description": "Only when they named one store or ride app (or for 'the usual')"},
        "services": {"type": "array", "items": {"type": "string", "enum": ["swiggy", "instamart", "zepto", "blinkit", "zomato", "apollo", "1mg", "pharmeasy", "uber", "ola", "rapido"]},
                     "description": "Several stores or ride apps they named to look on ('Blinkit aur Zepto dono pe dekho', 'Uber and Ola'): every one of them is searched and reported"},
        "category": {"type": "string", "enum": ["grocery", "medicine", "food", "ride"], "description": "No store or app named (or 'all' / 'the cheapest'): compare the usual ones of that kind. medicine = any medicine, tablet, strip, syrup, ointment or supplement (never grocery)"},
        "kind": {"type": "string", "enum": ["order", "ride"]},
        "goal": {"type": "string", "description": "One line, e.g. 'Atta and toor dal for Amma' or 'Cab to Dr Iyer's clinic'"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string", "description": "The product as a store would list it, in English/brand words (you translate: 'doodh' → 'milk', 'anda' → 'eggs')"},
                    "qty": {"type": "integer"},
                    "must_match": {"type": "array", "items": {"type": "string"}, "description": "Traits they insisted on that must be exact: flavour, variant, brand, size, strength (e.g. ['peri peri'], ['Amul', '1 litre'])"},
                    "restricted": {"type": "string", "enum": ["none", "tobacco", "gutka", "vape", "alcohol", "illegal"],
                                   "description": "Your judgement: is this tobacco (cigarettes or their brands, bidi, hookah), gutka or pan masala, a vape, alcohol, or an illegal drug? none otherwise. Saheli never orders those"},
                    "max_price": {"type": "number", "description": "Rupees, when they gave a price limit ('100 tak', 'under 100', '80-100' → 100)"},
                    "cheapest": {"type": "boolean", "description": "They want the cheapest one ('sasta wala')"},
                    "restaurant": {"type": "string", "description": "Food: the restaurant they named, if any ('Haldiram se')"}},
                    "required": ["name"]},
                  "description": "For orders: one entry per product (never two products in one name), read from their words in any language"},
        "pickup": {"type": "string"},
        "drop": {"type": "string"},
        "vehicle": {"type": "string", "description": "auto, mini, sedan, bike, or omit for cheapest car"},
        "area": {"type": "string", "description": "Delivery area or pincode if known"},
        "for_whom": FOR_WHOM,
        "pickup_shared": {"type": "boolean", "description": "Ride: the pickup is the WhatsApp location they shared (this message or a recent one)"},
        "drop_shared": {"type": "boolean", "description": "Ride: the drop is the WhatsApp location they shared"},
        "pickup_link": {"type": "string", "description": "Ride: a Google Maps link they sent for the pickup, exactly as sent"},
        "drop_link": {"type": "string", "description": "Ride: a Google Maps link they sent for the drop, exactly as sent "
                      "(someone who cannot find WhatsApp's location button often sends one)"},
        "at": {"type": "string", "description": "Only when they want it at a later time ('kal subah 6 baje cab', 'order milk at 7 tomorrow'): "
               "that time as ISO 8601 in IST, e.g. 2026-10-16T06:00 (work it out from NOW). Leave out for now."},
        "at_means": {"type": "string", "enum": ["arrive", "book"], "description": "With at: arrive (default) = the cab should come / the "
                     "order should arrive by then; book = they want it booked or placed exactly then"},
        "landing": {"type": "string", "description": "Ride only, when a flight's landing time is known (ISO 8601 in IST) and they did not "
                    "name the pickup time. The cab is set for 45 minutes after landing, for the bags. Leave out when they named a time (use at)."},
    },
    ["kind", "goal", "for_whom"],
)
async def start_task(ctx: TurnCtx, a: dict) -> dict:
    from app.tasks import runtime

    refused = await _restricted(ctx, a)
    if refused:
        return refused  # never searched, never a task (founder 2026-10-13: "it should not even try to order")

    if a.get("kind") not in ("order", "ride"):
        raise ToolRefused("Give kind: order (groceries, food, medicines) or ride.")
    if not str(a.get("goal") or "").strip():
        # the model left the goal out (lang eval 2026-10-12: a KeyError, not an answer): the items name it, else ask for it
        named = ", ".join(str(i["name"]) for i in a.get("items") or [] if isinstance(i, dict) and i.get("name"))
        if not named:
            raise ToolRefused("Give goal: in a few words, what to order or where the ride goes.")
        a = {**a, "goal": named}
    who = _person(ctx, a)
    if not who and (a["kind"] != "order" or ctx.is_system):
        _for_whom(ctx, a)  # refused: they are asked whom it is for
    a = {**a, "for_whom": who}
    kind_of = {s: ("ride" if c == "ride" else "order") for c, stores in runtime.COMPARE_STORES.items() for s in stores}
    named = list(dict.fromkeys(s for s in a.get("services") or [] if kind_of.get(s) == a["kind"]))
    if len(named) == 1 and not a.get("service"):
        a = {**a, "service": named[0]}
    a = {**a, "services": named if len(named) > 1 else None}
    if a["kind"] == "order" and not ctx.is_system:
        ask = await _confirm_address_first(ctx, a)
        if ask:
            return ask
        if not a["for_whom"]:
            _for_whom(ctx, a)  # the address was answered, not whom it is for: asked now
        if not a.get("service") and not a.get("services"):
            # The store(s) they named stay with the order through the address question (sim 2026-10-11: "Blinkit se …",
            # then "haan ghar" → the second call left the store out and three stores were searched).
            got = await _store_named_with_address(ctx)
            if isinstance(got, list) and len(got) > 1:
                a = {**a, "services": got}
            elif got:
                a = {**a, "service": got if isinstance(got, str) else got[0]}
    if a.get("kind") == "ride" and a.get("landing") and not a.get("at"):
        from app.care import ways
        from app.tasks import later

        land = later.parse_at(a["landing"])
        if land is None:
            raise ToolRefused("landing must be a date and time like 2026-10-16T00:30 (IST).")
        a = {**a, "at": ways.pickup_after_landing(land).isoformat(), "at_means": "arrive", "landing_at": land.isoformat()}
    if a.get("at"):
        # Set for later (founder 2026-10-12): looked up now, then ONE question with everything and the most they will pay;
        # booked at its time without asking (app/tasks/later.py).
        from app.tasks import later

        at = later.parse_at(a["at"])
        if at is None:
            raise ToolRefused("at must be a date and time like 2026-10-16T06:00 (IST); work it out from NOW and their words.")
        if at <= clock.now():
            raise ToolRefused(f"That time ({later.hm(at)}) has passed; ask them which time they mean.")
        means = a.get("at_means") if a.get("at_means") in ("arrive", "book") else "arrive"
        if later.book_time(at, means) - clock.now() >= later.MIN_AHEAD:
            a = {**a, "later": {"at": at.isoformat(), "means": means, "book_at": later.book_time(at, means).isoformat()}}
    if a["kind"] == "ride":
        # The ride's ends as places: a saved place they meant (an AI decision) goes to the app as its full address (live
        # 2026-10-12: a saved home went to the app as words, the app set the city and the ride was booked from there).
        for end in ("pickup", "drop"):
            # a Google Maps link they sent is an exact point, like a shared WhatsApp location (founder 2026-10-12: "I can't
            # find the location in WhatsApp"; his link went to the app as text)
            if a.get(f"{end}_link") and not a.get(f"{end}_place"):
                pin = await maps_link_pin(str(a[f"{end}_link"]))
                if pin is None:
                    raise ToolRefused(f"That Google Maps link for the {end} could not be read. Ask them in one short line to share "
                                      "the place with the Send location button, or to say its address.")
                a = {**a, f"{end}_place": _pin_place(pin), end: _pin_place(pin)["full"]}
        ends = [e for e in ("pickup", "drop") if a.get(e) and not a.get(f"{e}_place")]
        found = await asyncio.gather(*(_ride_end_place(ctx, str(a[e]), a["for_whom"]) for e in ends))
        a = {**a, **{f"{e}_place": f for e, f in zip(ends, found) if f}}
        saved = {e: dict(a[f"{e}_place"]) for e in ("pickup", "drop") if a.get(f"{e}_place")}
        for end in ("pickup", "drop"):
            if a.get(f"{end}_shared"):
                pin = await _shared_pin(ctx)
                if pin is None:
                    raise ToolRefused("No location was shared yet: ask for it (ask_location), or start the ride with their words.")
                if not (str(pin.get("name") or "").strip() or str(pin.get("address") or "").strip()):
                    # "my current location": its address in words (reverse geocoded), for the app and for checking what it sets
                    try:
                        rev = await ctx.host.call("ride_place", {"lat": float(pin["lat"]), "lng": float(pin["lng"])},
                                                  family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id)
                    except Exception:  # noqa: BLE001 — the point alone still goes to the app
                        rev = {}
                    if (rev or {}).get("address"):
                        pin = {**pin, "address": str(rev["address"])[:300]}
                if end == "pickup" and not saved.get("pickup"):
                    # the place they named when the pin was asked for ("from home"), even if this call names the pin itself
                    words = await _pickup_words_asked(ctx)
                    found = await _ride_end_place(ctx, words, a["for_whom"]) if words else None
                    if found:
                        saved["pickup"] = found
                a = {**a, f"{end}_place": _pin_place(pin), end: _pin_place(pin)["full"]}
                if end == "pickup" and saved.get("pickup") and not a.get("later"):
                    a = {**a, "pin_saved_to": await _save_pin(ctx, saved["pickup"], pin, a["for_whom"])}
        ask = await _ride_pin_first(ctx, a)
        if ask:
            return ask
    pin_note = ({"pin_saved": f"The location they shared is now {a['pin_saved_to']}'s exact point for next time: say so in a few words."}
                if a.get("pin_saved_to") else {})
    place = None
    if a["kind"] == "order" and a.get("area"):
        # Also after the address was confirmed: a place they name now must be a saved one (it went to the default place).
        place = await _delivery_place(ctx, a, keep_miss=True)
        problem = _named_place_problem(a, place)
        if problem:
            return problem
    if a["kind"] == "order" and not ctx.is_system:
        # whose home it goes to, against whom it is for (asked once; their answer goes ahead)
        place = place if place and place.get("addressId") else await _delivery_place(ctx, a)
        ask = await _not_their_place(ctx, a, place)
        if ask:
            return ask
    if a.get("services"):
        # Several stores or apps they named: exactly those are searched, and every one is reported (founder 2026-10-11).
        return {**(await _start_compare(ctx, a, tuple(a["services"]), place=place, named=True)), **pin_note}
    if not a.get("service"):
        cat = "ride" if a["kind"] == "ride" else a.get("category") or ""
        stores = runtime.COMPARE_STORES.get(cat) if (cat == "ride") == (a["kind"] == "ride") else None
        if not stores:
            raise ToolRefused("Say which store (service), or give category for an order with no store named.")
        return {**(await _start_compare(ctx, a, stores, place=place)), **pin_note}
    for t in await runtime.live_tasks(ctx.session, ctx.family_id):
        if t.service == a["service"] and t.kind == a["kind"]:
            same = runtime.same_store_again(t)
            if same is not None:
                return same  # (live 2026-10-13: "try again on Blinkit" got "already running", a promise, then silence)
            await runtime.replace_look_up(ctx.session, t, by=ctx.speaker.get("id") or ctx.elder_id)
    task = await _create_task(ctx, a, a["service"], place=place)
    return {**pin_note, "task_id": str(task.id), "status": "started", "update_coming": True,
            "next": ("A task update brings what the store has, numbered with your suggestion, for them to pick; then the one "
                     "confirm with the total. Nothing is ordered before that." if a["kind"] == "order" else
                     "A task update brings the fares, numbered, for them to pick; then the one confirm with the fare. Nothing is "
                     "booked before that.")}


ADDRESS_OK_FOR = timedelta(minutes=10)  # one order's back-and-forth; a later order asks again (live 2026-10-09: 28 min later it skipped)


async def _confirm_address_first(ctx: TurnCtx, a: dict) -> dict | None:
    """The delivery address is confirmed before any search, as the order flow did before Brain v2 (founder 2026-10-09:
    "it used to confirm address before going ahead, then search and give options, then OTP, then place order").
    The first start_task of an order records the ask and returns the address; a call in a later turn by the same person
    (their answer) goes ahead."""
    speaker = ctx.speaker.get("id") or ctx.elder_id
    since = clock.now() - ADDRESS_OK_FOR
    asked = [e for e in await store.events(ctx.session, ctx.family_id, ctx.elder_id, since=since, kinds=["order_address_asked"])
             if (e.payload or {}).get("speaker") == speaker]
    earlier = [e.payload or {} for e in asked if (e.payload or {}).get("ref") != ctx.message_ref]
    if asked and not earlier:  # asked in this same turn already: they have not answered yet
        return {"status": "waiting for the address answer", "next": "Ask them to confirm the address; start nothing yet."}
    if earlier and not await _moved(ctx, a, earlier[-1]):
        return None
    who = a["for_whom"]
    # whom it is for not known yet: the address of the person asking, and whom it is for asked in the same line
    place = await _delivery_place(ctx, {**a, "for_whom": who or speaker}, keep_miss=True)
    problem = _named_place_problem(a, place)
    if problem:
        return problem
    where = (f"{place.get('nickname') or 'Home'}: {place.get('full')}" if place.get("full") else
             f"pincode {place.get('pincode')}" if place.get("pincode") else
             f"near {a['area']}" if a.get("area") else "the address saved in the store account")
    names = _names(ctx)
    owners = [o for o in place.get("defaultFor") or [] if o and o != who]
    elsewhere = bool(who and owners) and who not in (place.get("defaultFor") or []) and await _own_home_elsewhere(ctx, a, place)
    await store.record_event(ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="order_address_asked",
                             summary=f"Asked to confirm the delivery address for: {a['goal']}", actor_id=speaker,
                             payload={"speaker": speaker, "ref": ctx.message_ref, "where": where, "service": a.get("service"),
                                      "services": a.get("services"), "addressId": place.get("addressId"), "for_whom": who})
    whom = who or "the person they say"
    same = (f" (service {a['service']}, the same items, for_whom {whom})" if a.get("service") else
            f" (services {', '.join(a['services'])}, the same items, for_whom {whom})" if a.get("services") else
            f" (the same items, for_whom {whom})")
    for_line = ("" if who == speaker else
                f" Say it is for {names.get(who) or 'them'}, so they can correct it if it is for someone else.")
    if not who:
        others = [names.get(p) or "a family member" for p in names if p != speaker]
        for_line = f" In the same line, ask whom it is for: them or {' or '.join(others)}."
    if elsewhere:
        for_line = (f" {place.get('nickname') or 'This place'} is the home of {_and(owners, names)}, not {names.get(who) or 'their'}'s: "
                    f"ask in the same line whether the order is for {names.get(who) or 'them'} or for {_and(owners, names)}.")
    medicine = a.get("category") == "medicine" or a.get("service") in ("apollo", "1mg", "pharmeasy")
    happened = (" The first message asks what happened (pain, a cough, the stomach), one short question, in their language. "
                "The address question is the next message, with a line that is only --- between them." if medicine else "")
    return {"status": "not started yet", "confirm_address": where, "for_whom": names.get(who) or who or "not known yet: ask",
            "ask_what_happened": medicine,
            "next": "Nothing is searched yet. Ask them to confirm this delivery address (say the place name "
                    f"and the area, not the full line).{for_line}{happened} When they say yes, call start_task again with the same order{same}; if "
                    "they name another saved place, pass it as area; if it is for someone else, pass that person as for_whom."}


def _names(ctx: TurnCtx) -> dict[str, str]:
    return {p.get("id"): (p.get("name") or "").split(" ")[0] for p in [ctx.elder, *ctx.members] if p.get("id")}


def _and(ids: list[str], names: dict[str, str]) -> str:
    return " and ".join(names.get(i) or "a family member" for i in ids)


async def _moved(ctx: TurnCtx, a: dict, asked: dict) -> bool:
    """The address was asked before whom it is for was known (at the place of the person asking); their answer names
    someone whose own default place is another one: that place is confirmed once before any search."""
    if asked.get("for_whom") or not a.get("for_whom") or a.get("area") or not asked.get("addressId"):
        return False
    place = await _delivery_place(ctx, a)
    return bool(place.get("addressId")) and place["addressId"] != asked["addressId"]


async def _own_home_elsewhere(ctx: TurnCtx, a: dict, place: dict) -> bool:
    """The person it is for has a home of their own on record (a default place for them) and it is not this place. Someone
    with no place of their own (living with the family) is not asked whose home it is."""
    own = await _delivery_place(ctx, {**a, "area": None})
    return (bool(own.get("addressId")) and a["for_whom"] in (own.get("defaultFor") or [])
            and own["addressId"] != place.get("addressId"))


async def _restricted(ctx: TurnCtx, a: dict) -> dict | None:
    """Tobacco, gutka, vape, alcohol or an illegal drug: refused before anything starts. Two AI judgements, each enough to
    refuse: the brain's own on each item (restricted), and a separate check of the item names (matcher.restricted_kinds);
    code acts on their answers. The refusal and the alternatives come from app.care.restricted."""
    from app.care import restricted
    from app.tasks import matcher

    if a.get("kind") != "order":
        return None
    items = [i for i in a.get("items") or [] if isinstance(i, dict)]
    names = [str(i.get("name") or "") for i in items] or ([str(a.get("goal") or "")] if a.get("goal") else [])
    said = [str(i.get("restricted") or "").lower() for i in items]
    # the brain's own judgement refuses at once; else the separate check (a few seconds at most; if it cannot answer, the
    # store look-up's own check still stops it before anything is shown)
    checked = [] if any(k in matcher.KINDS for k in said) else await matcher.restricted_kinds(names) or []
    kind = next((k for k in [*said, *checked] if k and k in matcher.KINDS), None)
    if not kind:
        return None
    await store.record_event(ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="order_refused",
                             summary=f"Not ordered ({kind}): {', '.join(n for n in names if n)[:160]}", actor_id=ctx.speaker.get("id"),
                             payload={"kind": kind, "by_brain": kind in said, "by_check": kind in checked})
    return {"status": "not started", **restricted.refusal(kind)}


def _person(ctx: TurnCtx, a: dict) -> str | None:
    """The household member for_whom names; with only one person in the household, them."""
    names = _names(ctx)
    who = str(a.get("for_whom") or "").strip()
    return who if who in names else next(iter(names)) if len(names) == 1 else None


def _for_whom(ctx: TurnCtx, a: dict) -> str:
    """The person an order or a ride is for, from the brain's structured field (founder 2026-10-12): never a default."""
    names = _names(ctx)
    who = str(a.get("for_whom") or "").strip()
    if who not in names:
        listed = "; ".join(f"{pid} ({name or 'no name'})" for pid, name in names.items())
        raise ToolRefused("Say whom this is for: for_whom = the id of the person who will receive or use it (" + listed + "). Read it "
                          "from their words and the conversation; never assume the care recipient. If it is not clear, ask them in a "
                          "few words first.")
    return who


async def _not_their_place(ctx: TurnCtx, a: dict, place: dict) -> dict | None:
    """An order going to the home of other people, not of the person it is for (data: whose default place it is): asked once
    (founder 2026-10-12: his own biscuits to his own home were recorded as his mother's). Their answer goes ahead."""
    who, owners = a["for_whom"], [o for o in place.get("defaultFor") or [] if o]
    if not place.get("addressId") or not owners or who in owners or not await _own_home_elsewhere(ctx, a, place):
        return None
    speaker = ctx.speaker.get("id") or ctx.elder_id
    asked = [e for e in await store.events(ctx.session, ctx.family_id, ctx.elder_id, since=clock.now() - ADDRESS_OK_FOR,
                                           kinds=["order_address_asked", "order_for_whom_asked"])
             if (e.payload or {}).get("speaker") == speaker and (e.payload or {}).get("addressId") == place["addressId"]]
    if any((e.payload or {}).get("ref") != ctx.message_ref for e in asked):
        return None  # asked about this place already: this is their answer
    names = _names(ctx)
    if not asked:
        await store.record_event(ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="order_for_whom_asked",
                                 summary=f"Asked whom the order is for: {a['goal']}", actor_id=speaker,
                                 payload={"speaker": speaker, "ref": ctx.message_ref, "addressId": place["addressId"], "for_whom": who})
    return {"status": "not started yet", "check_for_whom": True,
            "next": f"Nothing is searched yet. {place.get('nickname') or 'That place'} is the home of {_and(owners, names)}, not "
                    f"{names.get(who) or 'the person it is for'}'s. Ask in one short line whether this order is for "
                    f"{names.get(who) or 'them'} (delivered there) or for {_and(owners, names)}; then call start_task again with the same "
                    "order and place and for_whom set to the person they say."}


async def _store_named_with_address(ctx: TurnCtx) -> str | list[str] | None:
    """The store (or the stores) given with this person's order when its address was asked (in an earlier turn, minutes
    ago), if any."""
    speaker = ctx.speaker.get("id") or ctx.elder_id
    asked = [e for e in await store.events(ctx.session, ctx.family_id, ctx.elder_id, since=clock.now() - ADDRESS_OK_FOR,
                                           kinds=["order_address_asked"])
             if (e.payload or {}).get("speaker") == speaker and (e.payload or {}).get("ref") != ctx.message_ref]
    if not asked:
        return None
    p = asked[-1].payload or {}
    return p.get("services") or p.get("service")


def _named_place_problem(a: dict, place: dict) -> dict | None:
    """A place they named that is not saved, or saved places that could not be read: ask, start nothing (an order must never
    go to some other address)."""
    if not a.get("area"):
        return None
    if place.get("lookup_failed"):
        return {"status": "not started yet", "next": "The saved places could not be read just now, so nothing was started. Say so "
                                                     "in one line and ask them to try again in a minute."}
    if place.get("matched") is False:
        saved = "; ".join(place.get("saved") or []) or "none yet"
        return {"status": "not started yet", "unknown_place": a["area"], "saved_places": saved,
                "next": f"'{a['area']}' is not a saved place (saved: {saved}). Ask which saved place, or for the new full address "
                        "with pincode and what to call it; save it with save_place, then call start_task again with area set to its name."}
    return None


async def _start_compare(ctx: TurnCtx, a: dict, stores: tuple[str, ...], place: dict | None = None, named: bool = False) -> dict:
    """The same order looked up on several stores at once (no login); one task update brings every option."""
    from app.tasks import runtime
    from app.tasks.skills import SKILLS

    busy = {}
    for t in await runtime.live_tasks(ctx.session, ctx.family_id):
        if t.kind != a["kind"] or t.service not in stores:
            continue
        if runtime.same_store_again(t) is None:  # a finished look-up with nothing to pick: looked at again
            await runtime.replace_look_up(ctx.session, t, by=ctx.speaker.get("id") or ctx.elder_id)
        else:
            busy[t.service] = t
    free = [s for s in stores if s not in busy]
    if not free:
        raise ToolRefused(f"Every one of these already has {'a ride' if a['kind'] == 'ride' else 'an order'} running: "
                          + "; ".join(runtime.describe(busy[s]) for s in stores))
    gid = uuid.uuid4().hex[:10]
    place = place if place and place.get("addressId") else await _delivery_place(ctx, a)
    started = [await _create_task(ctx, a, s, extra={"compare": gid, **({"named": True} if named else {})}, place=place) for s in free]
    out: dict = {"looking_on": [SKILLS[s]["label"] for s in free], "task_ids": [str(t.id) for t in started], "update_coming": True,
                 "next": ("A task update brings the fares from every app, numbered with your suggestion marked, usually within a "
                          "minute: they pick, then after the login the one confirm with the fare. Nothing is booked before they say "
                          "yes to it." if a["kind"] == "ride" else
                          "A task update brings what every store found, numbered with your suggestion marked, usually within a "
                          "minute or two: they pick, then the one confirm with the total (a store login code may be asked first). "
                          "Nothing is ordered before they say yes to it.")}
    if len(free) < len(stores):
        out["already_running"] = [runtime.describe(busy[s]) for s in stores if s in busy]
    return out


async def _create_task(ctx: TurnCtx, a: dict, service: str, extra: dict | None = None, place: dict | None = None):
    from app.specialists.agents import specialist_for
    from app.specialists.contract import build_limits
    from app.tasks import runtime

    if not place or not place.get("addressId"):
        place = await _delivery_place(ctx, a)
    who = a.get("for_whom") or _for_whom(ctx, a)
    # the care recipient's own spending limit needs someone to approve; a self-care person has nobody (and is not stopped)
    carers = any(m.get("id") and m.get("id") != ctx.elder_id for m in ctx.members)
    limits = await build_limits(
        ctx.session, family_id=ctx.family_id, subject_id=who, kind=a["kind"], agent=specialist_for(service).name,
        requester_is_elder=ctx.speaker_is_elder and carers, place=place,
    )
    usual = None
    if a["kind"] == "ride" and not a.get("vehicle"):
        # the type this person took on this app before (founder 2026-10-12: "ask which type of cab I want"): marked as the
        # suggestion among the fares, never booked without their pick
        usual = next(((f.value or {}).get("vehicle") for f in await store.facts(ctx.session, ctx.family_id, who, domains=["preference"])
                      if f.key == f"ride_vehicle:{service}" and f.status == "active"), None)
    try:
        return await runtime.create(
            ctx.session, family_id=ctx.family_id, subject_id=who, requested_by=ctx.speaker.get("id") or ctx.elder_id,
            service=service, kind=a["kind"], goal=a["goal"],
            details={**{k: a[k] for k in ("items", "pickup", "drop", "vehicle", "area", "pickup_place", "drop_place", "later", "landing_at") if a.get(k)},
                     "for_set": True, **({"usual_vehicle": usual} if usual else {}), **(extra or {})}, limits=limits,
        )
    except runtime.TaskRefused as exc:
        raise ToolRefused(f"{exc}. Tell them kindly and offer something safe.") from exc


@tool(
    "task_input",
    "Give a running task what it is waiting for: the person's confirm of the cart and total (confirm: yes/no), the login "
    "code (otp, from whoever the update says has it), the option they picked when options were listed (go: its number), "
    "approval of a cancellation fee (fee: yes/no, older rides only), which ride fare to book (choice: its number as listed), which alternative to get for an item "
    "that is out of stock (swap: the alternative's name, or 'no'), the family approver's answer (approve: yes/no; only an "
    "approver's answer counts), or, at the confirm, a change they ask for: change (target = which asked item, items = what "
    "replaces it: another product, brand, size or quantity; several products are several entries), add (items), remove "
    "(target), more (target: they want to see that item's options; the list comes back in a task update), keep (value yes: a "
    "caregiver says keep trying when asked about a slow order), resend (the login code did not come: the store is asked for "
    "a new one; never promise a new code without calling this), only (while several stores are looked at, they named one of "
    "them: task_id = that store's task; the others are dropped; never say you are checking one store without calling this), "
    "later (their answer to the one question for a booking set for later: value yes with max_amount, or no_max; or value no), "
    "fare (a booking set for later costs more than their limit now: value yes to book it at the new amount, or no). "
    "You read their words; pass data, never their raw words. The new cart and total come back for the one confirm.",
    {
        "task_id": {"type": "string"},
        "kind": {"type": "string", "enum": ["go", "otp", "confirm", "fee", "choice", "swap", "approve", "more", "change", "add", "remove", "keep", "resend", "only",
                                            "later", "fare"]},
        "max_amount": {"type": "number", "description": "later: the most they will pay in rupees (their number, or the suggested one when they just say yes)"},
        "no_max": {"type": "boolean", "description": "later: they said any amount is fine (no limit)"},
        "value": {"type": "string", "description": "confirm / approve / fee / keep: exactly 'yes' or 'no' (you decide from their words, any language); go: the number(s) of the option(s) they picked, as listed (e.g. '2' or '1, 5'), 'yes' for the suggested one, or 'no' for none; choice (a ride's fare): its number as listed, or 'yes' for the one they picked before; otp: the digits; swap: the option as listed"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string", "description": "The product as a store would list it, in English/brand words (you translate: 'doodh' → 'milk', 'anda' → 'eggs')"},
                    "qty": {"type": "integer"},
                    "must_match": {"type": "array", "items": {"type": "string"}, "description": "Traits they insisted on that must be exact: flavour, variant, brand, size, strength (e.g. ['peri peri'], ['Amul', '1 litre'])"},
                    "max_price": {"type": "number", "description": "Rupees, when they gave a price limit ('100 tak', 'under 100', '80-100' → 100)"},
                    "cheapest": {"type": "boolean", "description": "They want the cheapest one ('sasta wala')"},
                    "restricted": {"type": "string", "enum": ["none", "tobacco", "gutka", "vape", "alcohol", "illegal"],
                                   "description": "Your judgement, as for start_task: tobacco, gutka or pan masala, vape, alcohol or an illegal drug; else none"},
                    "restaurant": {"type": "string", "description": "Food: the restaurant they named, if any ('Haldiram se')"}},
                    "required": ["name"]}, "description": "change / add / more: the new item(s), one entry per product"},
        "target": {"type": "integer", "description": "change / remove / more: the number of the asked item it is about (from 'asked: 1. … 2. …' in the task)"},
    },
    ["task_id", "kind"],
)
async def task_input(ctx: TurnCtx, a: dict) -> dict:
    from app.tasks import runtime

    task = await _task(ctx, a["task_id"])
    if a.get("items") and task.kind == "order":
        refused = await _restricted(ctx, {"kind": "order", "items": a["items"]})
        if refused:
            return {**refused, "task": runtime.describe(task), "next": "The order goes on without it: nothing was added or changed."}
    outcome = await runtime.provide_input(
        ctx.session, task, kind=a["kind"], value=str(a.get("value") or ""), by=ctx.speaker.get("id") or "", by_is_elder=ctx.speaker_is_elder,
        items=a.get("items"), target=a.get("target"), max_amount=a.get("max_amount"), no_max=bool(a.get("no_max")),
    )
    return {"result": outcome, "task": runtime.describe(task), **(_tell_who_cancelled(ctx, task) if task.status == "cancelled" else {})}


@tool(
    "whats_pending",
    "Everything open for the family in one list: orders, rides, approvals, reminders, family chores, questions, and a "
    "medicine or other change still marked pending; for each, who must act next and what, and anything stuck. A pending "
    "change is listed so you can ask. Do not approve or reject it here. Use for 'kya baaki hai?', 'what's pending?', "
    "'what do I need to do?', 'anything left today?'. "
    "mine=true lists only what the speaker must do.",
    {"mine": {"type": "boolean"}},
    [],
)
async def whats_pending(ctx: TurnCtx, a: dict) -> dict:
    from app.care import work

    rows = await work.items(ctx.session, ctx.family_id)
    me = ctx.speaker.get("id")
    if a.get("mine"):
        rows = [r for r in rows if r["owner"] == me]
    names = {m.get("id"): m.get("name") for m in [ctx.elder, *ctx.members]}
    return {"open": [{"title": r["title"], "who": "Saheli" if r["owner"] == work.SAHELI else names.get(r["owner"], r["owner"]),
                      "next": r["next_action"], "due": r["due_at"], "stuck": r["stuck"], "id": r["id"]} for r in rows],
            "count": len(rows)}


@tool(
    "boundaries",
    "The family's limits for orders and rides: what needs approval (amounts, categories, who may order or book) and who "
    "approves. Use for 'what are the limits?', before promising an order will go through, and before changing them.",
    {},
    [],
)
async def boundaries_read(ctx: TurnCtx, a: dict) -> dict:
    from app.care import boundaries

    policy = await boundaries.get(ctx.session, ctx.family_id)
    roster = await store.roster(ctx.session, ctx.family_id)
    names = {m.get("id"): m.get("name") for m in [ctx.elder, *ctx.members]}
    return {"policy": policy, "plain": boundaries.describe(policy, names),
            "approvers": [names.get(x, x) for x in boundaries.approvers(policy, roster)],
            "can_change": boundaries.can_manage(policy, roster, ctx.speaker.get("id") or "")}


@tool(
    "set_boundaries",
    "Change the family's limits (only an approver can; never the care recipient). Give only what changes: "
    "elder_order_limit / elder_ride_limit (₹; the care recipient's own orders or rides above this need approval), "
    "anyone_over (₹; anything above needs approval; 0 turns it off), monthly_cap (₹; 0 off), approval_categories "
    "(subset of grocery, food, pharmacy, ride), approvers (member ids), members ({id: {can_order, can_ride}}), login_codes "
    "({person id: 'self' or a caregiver's member id}: who gives store login codes for that person's orders; the store then "
    "logs in with that person's number). Repeat back the new limits in one line after.",
    {
        "login_codes": {"type": "object", "description": "{person id: 'self' | member id}"},
        "elder_order_limit": {"type": "integer", "minimum": 0}, "elder_ride_limit": {"type": "integer", "minimum": 0},
        "anyone_over": {"type": "integer", "minimum": 0}, "monthly_cap": {"type": "integer", "minimum": 0},
        "approval_categories": {"type": "array", "items": {"type": "string", "enum": ["grocery", "food", "pharmacy", "ride"]}},
        "approvers": {"type": "array", "items": {"type": "string"}},
        "members": {"type": "object"},
    },
    [],
)
async def set_boundaries(ctx: TurnCtx, a: dict) -> dict:
    from app.care import boundaries

    policy = await boundaries.get(ctx.session, ctx.family_id)
    roster = await store.roster(ctx.session, ctx.family_id)
    me = ctx.speaker.get("id") or ""
    if (ctx.speaker_is_elder and not ctx.self_care) or not boundaries.can_manage(policy, roster, me):
        raise ToolRefused("Only the family's approver can change the limits. Say so kindly.")
    ids = {m.get("id") for m in [ctx.elder, *ctx.members]}
    if any(x not in ids for x in a.get("approvers") or []) or any(x not in ids for x in (a.get("members") or {})):
        raise ToolRefused("Use member ids from HOUSEHOLD.")
    codes = a.get("login_codes") or {}
    if not isinstance(codes, dict) or any(k not in ids or (v != "self" and v not in ids) for k, v in codes.items()):
        raise ToolRefused("login_codes: {person id: 'self' or a member id} with ids from HOUSEHOLD.")
    if a.get("approvers") and ctx.elder_id in a["approvers"]:
        raise ToolRefused("The care recipient cannot be the approver of their own orders.")
    new = await boundaries.save(ctx.session, ctx.family_id, a, by=me)
    names = {m.get("id"): m.get("name") for m in [ctx.elder, *ctx.members]}
    return {"saved": boundaries.describe(new, names)}


@tool(
    "cancel_task",
    "Cancel a task: an order or ride not placed or booked yet is stopped (nothing is ordered). A PLACED order or a BOOKED "
    "ride (or one being placed right now): when a caregiver, or someone ordering for themselves, asks, the store or ride app "
    "is asked to cancel it (a free cancel goes through without asking again, a fee is asked once with its amount, the result "
    "comes in a task update, and if the store will not cancel it they are sent to its app); when the care recipient asks, "
    "it cannot be cancelled from here: the caregiver is told right away by the task system. Always call this when they say "
    "cancel (never decide the answer yourself, never alert or message the caregiver about it yourself). Say exactly what the "
    "result says; never say cancelled unless the result or an update says it is cancelled.",
    {"task_id": {"type": "string"}, "reason": {"type": "string"}},
    ["task_id", "reason"],
)
async def cancel_task(ctx: TurnCtx, a: dict) -> dict:
    from app.tasks import runtime

    task = await _task(ctx, a["task_id"])
    out = {"result": await runtime.request_cancel(ctx.session, _tools_pkg().task_agent(), task, by=ctx.speaker.get("id") or "", reason=a["reason"])}
    ask = (task.details or {}).get("cancel_ask") or {}
    if ask.get("elder") and ask.get("state") in ("told_carers", "unclear", "pending"):
        # The task system tells the caregivers (one "[Order needs you]" each, with what to do): not a second message from
        # this turn (sim 2026-10-12: the brain also messaged the son, who got two). Messaged already in this turn: that was it.
        carers = set(ask.get("carers") or [])
        ctx.told_by_tasks |= carers
        if carers and carers <= ctx.messaged and ask.get("state") != "pending" and not ask.get("carers_told"):
            task.details = {**(task.details or {}), "cancel_ask": {**ask, "carers_told": clock.now().isoformat()}}
    return {**out, **_tell_who_cancelled(ctx, task)}


@tool(
    "done_in_app",
    "They cancelled or changed one of the family's orders or rides THEMSELVES in the store's or ride's app (not through you): "
    "record it on the task. what: cancelled | changed; detail: what they changed. Nothing is run on the store; a cancel takes "
    "it out of the month's spend and stops its tracking and after-order questions. Use this, not cancel_task, when they say "
    "they already did it in the app.",
    {"task_id": {"type": "string"}, "what": {"type": "string", "enum": ["cancelled", "changed"]}, "detail": {"type": "string"}},
    ["task_id", "what"],
)
async def done_in_app(ctx: TurnCtx, a: dict) -> dict:
    from app.tasks import runtime
    from app.tasks.skills import SKILLS

    task = await _task(ctx, a["task_id"])
    me = ctx.speaker.get("id") or ""
    out = {"result": await runtime.done_in_app(ctx.session, task, what=a["what"], by=me, detail=a.get("detail") or "")}
    if task.status == "cancelled" and me and me != task.requested_by and not ctx.is_system:
        # as when a caregiver cancels through Saheli (founder 2026-10-10): the person who asked for it hears who cancelled it
        who = next((m.get("name") for m in [ctx.elder, *ctx.members] if m.get("id") == me), "a family member")
        out["tell"] = (f"Tell person {task.requested_by} kindly in one line with send_message that {who} cancelled this "
                       f"{task.kind} themselves in the {SKILLS[task.service]['label']} app.")
    return out


def _tell_who_cancelled(ctx: TurnCtx, task) -> dict:
    """Founder 2026-10-10: when a caregiver cancels an order the care recipient asked for, tell the care recipient who."""
    me = ctx.speaker.get("id") or ""
    if not task.cancel_requested or ctx.is_system or not me or me == task.requested_by:
        return {}
    who = next((m.get("name") for m in [ctx.elder, *ctx.members] if m.get("id") == me), "a family member")
    return {"tell": f"Tell person {task.requested_by} kindly in one line with send_message that {who} asked to cancel this order "
                    f"({task.goal}); nothing was ordered unless it said placed."}


@tool(
    "task_status",
    "How the family's recent orders and rides went, in plain words, including what the browser did on the website (each run, "
    "its steps, the sites visited and the rough cost). Use for 'order ka kya hua?', 'what did the agent do?', 'why did it fail?'. "
    "Leave out task_id for the latest few.",
    {"task_id": {"type": "string"}},
    [],
)
async def task_status(ctx: TurnCtx, a: dict) -> dict:
    from sqlalchemy import select

    from app.tasks import runtime, sandbox
    from app.tasks.models import Task

    if a.get("task_id"):
        rows = [await _task(ctx, a["task_id"])]
    else:
        rows = list((await ctx.session.execute(select(Task).where(Task.family_id == ctx.family_id).order_by(Task.created_at.desc()).limit(5))).scalars())
    return {"tasks": [{"id": str(t.id), "summary": runtime.describe(t), "status": t.status, "phase": t.phase,
                       "browser": sandbox.audit_text(t), "when": clock.ist(t.created_at).strftime("%d %b %H:%M")} for t in rows]}


FEEDBACK_ITEM = {
    "type": "object",
    "properties": {
        "item": {"type": "integer", "description": "The product's number in the follow-up (1, 2, …)"},
        "liked": {"type": "string", "enum": ["yes", "no", "mixed"]},
        "usual": {"type": "string", "enum": ["usual", "new"], "description": "usual: something they usually have; new: a first try"},
        "again": {"type": "string", "enum": ["yes", "no"], "description": "Whether to get it again ('agli baar bhi yahi lana' is yes)"},
        "note": {"type": "string", "description": "What they said about it, in a few words (taste, freshness, size, price, brand)"},
    },
    "required": ["item"],
}


@tool(
    "order_feedback",
    "What someone says about an order after it was placed: whether it came (arrived yes / no / partly = something missing or "
    "wrong), the problem in a few words, and how they liked each product (liked yes/no/mixed; usual = something they usually "
    "have, new = a first try; again = whether to get it again; note = their words in short). Use it for their answers to your "
    "'has it come?' and 'how was it?' questions, and whenever they bring it up themselves ('order nahi aaya', 'doodh phata hua "
    "tha', 'khakhra bahut accha tha'). Products are the numbers in the follow-up. It saves everything in the care memory, tells "
    "the caregiver by itself when the order is long overdue or came wrong, and its result says what to tell them and what, if "
    "anything, to ask next. Leave out task_id for their latest order.",
    {
        "task_id": {"type": "string", "description": "The order's task id (in the follow-up or the task update)"},
        "arrived": {"type": "string", "enum": ["yes", "no", "partly"]},
        "problem": {"type": "string", "description": "What was wrong, missing or damaged, in a few words"},
        "items": {"type": "array", "items": FEEDBACK_ITEM},
        "order_again": {"type": "string", "enum": ["yes", "no"], "description": "Order from this restaurant or shop again?"},
        "eaten": {"type": "string", "enum": ["all", "most", "half", "little", "none", "someone_else"],
                  "description": "A restaurant meal: how much of it the person it was for ate, from their answer (never assumed)"},
        "note": {"type": "string", "description": "Anything else they said about the order as a whole, in a few words"},
        "done": {"type": "boolean", "description": "true when they seem done talking about it (short answers, busy, tired)"},
    },
    [],
)
async def order_feedback(ctx: TurnCtx, a: dict) -> dict:
    from sqlalchemy import select

    from app.tasks import aftercare
    from app.tasks.models import Task

    if ctx.is_system:
        # Only what a person says goes in (sim 2026-10-11: on a scheduled turn it decided "still not come" from an answer
        # given ten minutes earlier and told the caregiver; the founder's rule is their complaint, asked again when late).
        raise ToolRefused("Only their own answer goes into order_feedback. Ask them now with send_message, then reply none.")

    if a.get("task_id"):
        task = await _task(ctx, a["task_id"])
    else:
        rows = (await ctx.session.execute(
            select(Task).where(Task.family_id == ctx.family_id, Task.status == "done", Task.kind.in_(("order", "ride")),
                               Task.created_at >= clock.now() - timedelta(days=7)).order_by(Task.created_at.desc()).limit(10)
        )).scalars()
        task = next((t for t in rows if (t.result or {}).get("placed") or (t.result or {}).get("booked")), None)
    if not task or not ((task.result or {}).get("placed") or (task.result or {}).get("booked")):
        raise ToolRefused("No placed order to give feedback on. If it is about something not ordered through you, save it with "
                          "remember (domain preference).")
    rows = aftercare.items_of(task)
    wrong = [it.get("item") for it in a.get("items") or [] if isinstance(it, dict) and it.get("item") not in {r["n"] for r in rows}]
    if wrong and len(rows) > 1:
        # The options listed before the order had their own numbers (sim 2026-10-12: the khakhra, option 3 there, was given
        # as product 3 of a two-product order and its feedback was lost): the order's own numbers, so nothing is lost.
        raise ToolRefused("Products are numbered by this order, not by the options listed before it: "
                          + "; ".join(f"{r['n']}. {r['name']}" for r in rows) + ". Call order_feedback again with these numbers.")
    return await aftercare.record(
        ctx.session, task, by=ctx.speaker.get("id") or ctx.elder_id, source_kind=ctx.source_kind,
        confidence=0.9 if ctx.speaker_is_elder and not ctx.self_care else 1.0, arrived=a.get("arrived"), problem=a.get("problem"), items=a.get("items"),
        order_again=a.get("order_again"), note=a.get("note"), done=bool(a.get("done")), ref=ctx.message_ref, eaten=a.get("eaten"),
    )
