"""Places and pins: saved places, map links, WhatsApp locations, where a ride starts and ends."""

from __future__ import annotations

from datetime import timedelta

from app.brain.tools.base import ToolRefused, TurnCtx, tool
from app.care import store
from app.core import clock


def _tools_pkg():
    """The tools package: tests replace task_agent and _follow on it, so they are read from there."""
    import sys

    return sys.modules["app.brain.tools"]


RIDE_PIN_ASK_FOR = timedelta(minutes=10)
PIN_KEPT_FOR = timedelta(hours=2)
PIN_SAVE_WITHIN_KM = 1.5


def order_subject(ctx: TurnCtx, for_whom: str | None) -> str:
    """Who an order or ride event is about: the person it is for, or the person who asked when that is not known yet.
    Never the care recipient only because the chat is about them."""
    names = {p.get("id") for p in [ctx.elder, *ctx.members] if p.get("id")}
    who = str(for_whom or "").strip()
    return who if who in names else (ctx.speaker.get("id") or ctx.elder_id)


async def asks_from(ctx: TurnCtx, speaker: str, kinds: list[str], since) -> list:
    """Asks this person was given, on whoever the order is for. They used to sit on the care recipient, so a caregiver's
    own order showed up in the care recipient's daily brief."""
    people = [str(p.get("id")) for p in [ctx.elder, *ctx.members] if p.get("id")]
    found = []
    for pid in dict.fromkeys(people):
        found.extend(await store.events(ctx.session, ctx.family_id, pid, since=since, kinds=kinds))
    found = [e for e in found if (e.payload or {}).get("speaker") == speaker]
    found.sort(key=lambda e: (e.at, e.id or 0))
    return found


async def _shared_pin(ctx: TurnCtx) -> dict | None:
    """The WhatsApp location pin of this message, else the one this person shared in the last two hours."""
    if ctx.location and ctx.location.get("lat") is not None:
        return ctx.location
    who = ctx.speaker.get("id") or ctx.elder_id
    for t in reversed(await store.recent_turns(ctx.session, ctx.family_id, who, limit=12)):
        pin = (t.meta or {}).get("location") if t.role == "user" else None
        if pin and t.at and clock.now() - t.at <= PIN_KEPT_FOR:
            return pin
    return None


MAPS_HOSTS = ("maps.app.goo.gl", "goo.gl", "maps.google.com", "google.com", "www.google.com", "google.co.in", "www.google.co.in")


async def _follow(link: str) -> tuple[str, str]:
    """The address a Maps link ends at, and the start of its page (some answer with the point only in the page)."""
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=8.0, headers={"User-Agent": "Mozilla/5.0"}) as c:
        r = await c.get(link)
        return str(r.url), r.text[:300_000]


def _maps_point(text: str) -> dict | None:
    """The map point in a Google Maps address or page: the place's own pin (!3d..!4d..), else the map's centre (@lat,lng),
    else a query point (q=lat,lng); with the place's name when the address has one."""
    import re
    from urllib.parse import unquote_plus

    m = (re.search(r"!3d(-?\d{1,2}\.\d+)!4d(-?\d{1,3}\.\d+)", text) or re.search(r"@(-?\d{1,2}\.\d+),(-?\d{1,3}\.\d+)", text)
         or re.search(r"[?&](?:q|query|ll|center)=(-?\d{1,2}\.\d+)(?:,|%2C)\s*(-?\d{1,3}\.\d+)", text))
    if not m:
        return None
    lat, lng = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lng <= 180) or (lat == 0 and lng == 0):
        return None
    name = re.search(r"/maps/place/([^/@?]+)", text)
    return {"lat": lat, "lng": lng, "name": unquote_plus(name.group(1))[:120] if name else ""}


async def maps_link_pin(link: str) -> dict | None:
    """A Google Maps link (as the person sent it) to its map point, or None when it is not one or cannot be read."""
    from urllib.parse import urlparse

    link = link.strip()
    u = urlparse(link)
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or host not in MAPS_HOSTS or (host == "goo.gl" and not u.path.startswith("/maps")):
        return None
    try:
        final, page = await _tools_pkg()._follow(link)
    except Exception:  # noqa: BLE001 — unreadable: they are asked for the location instead
        return None
    pin = _maps_point(final) or _maps_point(page)
    if pin and page:
        pin = {**pin, **_maps_meta(page, pin.get("name") or "")}
    return pin


def _maps_meta(page: str, name: str) -> dict:
    """The place's name and address from a Maps page's share card ("Name · Address"), when it has one."""
    import html
    import re

    m = re.search(r'<meta[^>]+(?:property|itemprop)="og:title"[^>]+content="([^"]+)"', page) or \
        re.search(r'<meta[^>]+content="([^"]+)"[^>]+(?:property|itemprop)="og:title"', page)
    if not m:
        return {}
    title = html.unescape(m.group(1)).strip()
    if "·" in title:
        head, tail = (x.strip() for x in title.split("·", 1))
        return {"name": name or head, "address": tail}
    return {"name": name or title} if title and "Google Maps" not in title else {}


def _pin_place(pin: dict) -> dict:
    lat, lng = float(pin["lat"]), float(pin["lng"])
    label = ", ".join(x for x in (str(pin.get("name") or "").strip(), str(pin.get("address") or "").strip()) if x)
    return {"nickname": str(pin.get("name") or "").strip() or "the shared location", "full": label or f"{lat:.5f}, {lng:.5f}",
            "lat": lat, "lng": lng, "precision": "user", "pin": True}


def _km(a: dict, b: dict) -> float:
    import math

    p1, p2 = math.radians(float(a["lat"])), math.radians(float(b["lat"]))
    dp, dl = p2 - p1, math.radians(float(b["lng"]) - float(a["lng"]))
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


async def _save_pin(ctx: TurnCtx, place: dict, pin: dict, who: str | None = None) -> str | None:
    """The pin they shared for a saved place they named ("from home") becomes that place's exact point for next time,
    when it is near where the place was thought to be (they may share where they are, not the place); else nothing."""
    if place.get("precision") == "user" or not place.get("nickname"):
        return None
    if place.get("lat") is not None and place.get("lng") is not None and _km(place, pin) > PIN_SAVE_WITHIN_KM:
        return None
    key = f"pin:{place['nickname']}:{float(pin['lat']):.6f},{float(pin['lng']):.6f}"
    if key in ctx.once:
        return place["nickname"]  # saved already in this turn (sim 2026-10-13: start_task called twice, the pin saved twice)
    ctx.once.add(key)
    try:
        got = await ctx.host.call("save_place", {"place": place["nickname"], "lat": float(pin["lat"]), "lng": float(pin["lng"])},
                                  family_id=ctx.family_id, subject_id=who or ctx.elder_id, actor_id=ctx.actor_id)
    except Exception:  # noqa: BLE001 — the ride goes on with the pin either way
        return None
    return place["nickname"] if (got or {}).get("ok") else None


async def _pickup_words_asked(ctx: TurnCtx) -> str:
    """The pickup in their words when the pin was asked for (minutes ago)."""
    speaker = ctx.speaker.get("id") or ctx.elder_id
    asked = await asks_from(ctx, speaker, ["ride_pin_asked"], clock.now() - RIDE_PIN_ASK_FOR)
    return str((asked[-1].payload or {}).get("pickup") or "") if asked else ""


async def _ride_pin_first(ctx: TurnCtx, a: dict) -> dict | None:
    """Founder 2026-10-12: a ride's pickup is asked as a WhatsApp location (its exact point) unless a saved place already
    has one. Asked once per ride request: their next message goes ahead, with the pin, or with the saved address or their
    words when they would rather."""
    if (ctx.is_system or a.get("pickup_shared") or (a.get("pickup_place") or {}).get("precision") == "user"
            or not (a.get("pickup") and a.get("drop"))):
        return None  # a scheduled turn, a pin already, an exact saved point, or a ride still missing its ends
    speaker = ctx.speaker.get("id") or ctx.elder_id
    # per pickup (founder 2026-10-12: the ask for one pickup was taken as answered for the corrected one, his saved home,
    # whose point is only its pincode's)
    key = str((a.get("pickup_place") or {}).get("nickname") or a.get("pickup") or "").strip().lower()
    asked = [e for e in await asks_from(ctx, speaker, ["ride_pin_asked"], clock.now() - RIDE_PIN_ASK_FOR)
             if (e.payload or {}).get("pickup_key", key) == key]
    if any((e.payload or {}).get("ref") != ctx.message_ref for e in asked):
        return None
    ctx.location_request = True
    if not asked:
        await store.record_event(ctx.session, family_id=ctx.family_id, subject_id=order_subject(ctx, a.get("for_whom")),
                                 kind="ride_pin_asked", summary=f"Asked for the pickup as a location: {a['goal']}", actor_id=speaker,
                                 payload={"speaker": speaker, "ref": ctx.message_ref, "for_whom": a.get("for_whom"),
                                          "pickup": str(a.get("pickup") or "")[:200], "pickup_key": key})
    saved = (a.get("pickup_place") or {}).get("nickname")
    why = f" {saved} is saved without its exact point (only its area), so the driver could go to the wrong door." if saved else ""
    return {"status": "not started yet", "ask_location": "pickup",
            "next": "Nothing is searched yet." + why + " In one short line, ask them to share the pickup with the 'Send "
                    "location' button under your message (if they are not there now, they can pick the spot on the map; a Google "
                    "Maps link works too). When it comes, call start_task again with the same ride and pickup_shared true (a "
                    "link: pickup_link). If they would rather use the saved address or their words, call start_task again as it is."}


async def _ride_end_place(ctx: TurnCtx, words: str, rider: str | None = None) -> dict | None:
    """A ride's pickup or drop: the family's saved place when the words mean one (an AI decision), with its full address and
    map point; None for a landmark or an address they spelt out (the app finds it). "Home" is the rider's own home."""
    who = ctx.speaker.get("id") or ctx.elder_id
    rider = rider or ctx.elder_id
    try:
        args = await saved_place_args(ctx.host, words, family_id=ctx.family_id, subject_id=rider, actor_id=who)
        if not args.get("exact"):
            return None  # no saved places, or the AI could not answer: the words go to the app as they are
        got = await ctx.host.call("delivery_place", args, family_id=ctx.family_id, subject_id=rider, actor_id=who)
    except Exception:  # noqa: BLE001 — the words go to the app as they are
        return None
    if not got or not got.get("matched") or not got.get("full"):
        return None
    out = {k: got.get(k) for k in ("nickname", "full", "pincode", "lat", "lng") if got.get(k) is not None}
    return {**out, **({"precision": got["geoPrecision"]} if got.get("geoPrecision") else {})}


async def saved_place_args(ctx_host, words: str, *, family_id: str, subject_id: str, actor_id: str) -> dict:
    """The place lookup's arguments for a person's words: the saved place they mean, decided by AI from the family's saved
    places and sent by its exact name ({"words": name, "exact": true}). Words that are no saved place go as exact too, so
    the backend answers "not saved" and the person is asked. If the AI cannot answer, the words go as before (the backend's
    own lookup). Founder 2026-10-10: no word lists on people's words."""
    if not words:
        return {"words": ""}
    try:
        listed = (await ctx_host.call("list_places", {}, family_id=family_id, subject_id=subject_id, actor_id=actor_id)).get("places") or []
    except Exception:  # noqa: BLE001
        return {"words": words}
    if not listed:
        return {"words": words}
    from app.tasks import matcher

    try:
        name = await matcher.which_place(words, listed)
    except Exception:  # noqa: BLE001 — the model is down: the backend's own lookup
        return {"words": words}
    return {"words": name or words, "exact": True}


async def _delivery_place(ctx: TurnCtx, a: dict, keep_miss: bool = False) -> dict:
    if a["kind"] != "order":
        return {}
    try:
        who = ctx.speaker.get("id") or ctx.elder_id
        # the places (and the default place) of the person it is for: their home, not the care recipient's by default
        to = a.get("for_whom") or ctx.elder_id
        args = await saved_place_args(ctx.host, a.get("area") or "", family_id=ctx.family_id, subject_id=to, actor_id=who)
        got = await ctx.host.call("delivery_place", args, family_id=ctx.family_id, subject_id=to, actor_id=who)
        if keep_miss and got.get("matched") is False:
            return got
        return got if got.get("addressId") else {}
    except Exception:  # noqa: BLE001 — no saved place: the agent uses the account's home address
        return {"lookup_failed": True} if keep_miss else {}


@tool(
    "places",
    "The family's saved delivery places (address book) for the person: each place's name, address, receiver and which one is "
    "their default. Use for 'kaun kaun se address save hain?', 'which address do you have for Beta?', before changing one.",
    {},
    [],
)
async def places(ctx: TurnCtx, a: dict) -> dict:
    return await ctx.host.call("list_places", {}, family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id)


@tool(
    "save_place",
    "Save a delivery place in the family address book (the same one the dashboard shows), or change a saved one. New place: "
    "the full address with house/flat, street, city and 6-digit pincode, and a short name for it (\"Beta's flat\", \"Clinic\"; "
    "the first place is Home); ask for what is missing. Change one: place = its saved name, plus what changes (address, name, "
    "receiver). receiver_name/receiver_phone: who takes the delivery there when it is not the person. make_default: orders go "
    "there unless they say otherwise. Read the name and area back in one line after saving. The person can save their own "
    "places; a caregiver can save any.",
    {
        "address": {"type": "string", "description": "Full address as they said it, with pincode (for a new place or a changed address)"},
        "name": {"type": "string", "description": "What to call the place (new name when renaming)"},
        "place": {"type": "string", "description": "Saved place to change (its name); leave out for a new place"},
        "receiver_name": {"type": "string"},
        "receiver_phone": {"type": "string"},
        "make_default": {"type": "boolean"},
        "use_shared_location": {"type": "boolean", "description": "The place's exact point is the WhatsApp location they shared (for rides and deliveries)"},
        "maps_link": {"type": "string", "description": "A Google Maps link they sent for this place: its exact point (and name/address when the "
                                                       "link has them) is read and saved"},
    },
    [],
)
async def save_place(ctx: TurnCtx, a: dict) -> dict:
    if ctx.is_system:
        raise ToolRefused("Only the family saves places.")
    if not ctx.is_caregiver and not ctx.speaker_is_elder:
        raise ToolRefused("Only the person themselves or a caregiver can save or change places.")
    args = {k: a[k] for k in ("address", "name", "place", "receiver_name", "receiver_phone", "make_default") if a.get(k) not in (None, "")}
    if a.get("use_shared_location"):
        pin = await _shared_pin(ctx)
        if pin is None:
            raise ToolRefused("No location was shared yet: ask for it with ask_location.")
        args = {**args, "lat": float(pin["lat"]), "lng": float(pin["lng"])}
    if a.get("maps_link"):
        # founder 2026-10-13: a Google Maps link anywhere is read for its point and place details
        pin = await maps_link_pin(str(a["maps_link"]))
        if pin is None:
            raise ToolRefused("That link could not be read as a Google Maps place. Ask them to share the location in WhatsApp instead (ask_location).")
        args = {**args, "lat": float(pin["lat"]), "lng": float(pin["lng"])}
        if not args.get("address") and not args.get("place"):
            label = ", ".join(x for x in (str(pin.get("name") or "").strip(), str(pin.get("address") or "").strip()) if x)
            if label:
                args["address"] = label
            if not args.get("name") and pin.get("name"):
                args["name"] = str(pin["name"])[:40]
    return await ctx.host.call("save_place", args, family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id)


@tool(
    "map_link",
    "Read a Google Maps link someone sent (maps.app.goo.gl, goo.gl/maps, google.com/maps): its place name, address and exact "
    "coordinates. Use it whenever a maps link comes in and you need to know where it points (to save it as a place with "
    "save_place maps_link, to tell them what it is, or for a ride: start_task takes the link itself).",
    {"link": {"type": "string"}},
    ["link"],
)
async def map_link(ctx: TurnCtx, a: dict) -> dict:
    pin = await maps_link_pin(str(a["link"]))
    if pin is None:
        return {"read": False, "next": "Not a readable Google Maps link: ask them to share the location in WhatsApp (ask_location)."}
    return {"read": True, "name": pin.get("name"), "address": pin.get("address"), "lat": float(pin["lat"]), "lng": float(pin["lng"])}


@tool(
    "ask_location",
    "Ask them to share a place as a WhatsApp location: your reply goes with WhatsApp's 'Send location' button. For a ride's "
    "pickup when no saved place has its exact point (start_task asks for it itself), a drop that is not a clear landmark, "
    "or a place to save. Write the question in your reply in one short line (they can also pick the spot on the map). On a "
    "scheduled turn use send_message with ask_location instead.",
    {"purpose": {"type": "string", "enum": ["pickup", "drop", "place"]}},
    ["purpose"],
)
async def ask_location(ctx: TurnCtx, a: dict) -> dict:
    if ctx.is_system:
        raise ToolRefused("On a scheduled turn, send_message with ask_location true.")
    ctx.location_request = True
    return {"ok": True, "next": f"Your reply goes with a 'Send location' button; ask for the {a['purpose']} in one short line."}


@tool(
    "remove_place",
    "Remove a saved delivery place. Caregivers only. First ask them to confirm (say the place's name and area); call with "
    "confirmed=true only after they said yes in their reply.",
    {"place": {"type": "string", "description": "The saved place's name"}, "confirmed": {"type": "boolean"}},
    ["place"],
)
async def remove_place(ctx: TurnCtx, a: dict) -> dict:
    if ctx.is_system or not ctx.is_caregiver:
        raise ToolRefused("Only a caregiver can remove a saved place.")
    return await ctx.host.call("remove_place", {"place": a["place"], "confirmed": a.get("confirmed") is True},
                               family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id)
