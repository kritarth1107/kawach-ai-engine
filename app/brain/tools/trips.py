"""Trips and look-ups: web search, ways there, flights (note, accept, read), doctors and lab tests."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from app.brain.tools.base import ABOUT_OWN, ToolRefused, TurnCtx, logger, tool
from app.brain.tools.orders import start_task
from app.care import store
from app.core import clock


def _tools_pkg():
    """The tools package: tests replace task_agent and _follow on it, so they are read from there."""
    import sys

    return sys.modules["app.brain.tools"]


async def _care_place(ctx: TurnCtx) -> dict:
    """Where the care recipient lives (lat/lon/pincode/city) for lab collection and nearby doctors."""
    try:
        got = await ctx.host.call("delivery_place", {"words": ""}, family_id=ctx.family_id, subject_id=ctx.elder_id,
                                  actor_id=ctx.speaker.get("id") or ctx.elder_id)
        return got or {}
    except Exception:  # noqa: BLE001
        return {}


async def _with_browser(fn):
    """A short guest cloud browser for read-only look-ups (nothing logged in, nothing booked), always stopped after."""
    agent = _tools_pkg().task_agent()
    if not hasattr(agent, "open_session"):
        raise ToolRefused("Live look-ups are not available right now; say so plainly.")
    sid = await agent.open_session(None)
    try:
        cdp = await agent.cdp_url(sid)
        if not cdp:
            raise ToolRefused("Live look-ups are not available right now; say so plainly.")
        return await asyncio.wait_for(fn(cdp), timeout=50)
    finally:
        try:
            await agent.stop_session(sid)
        except Exception:  # noqa: BLE001
            logger.warning("guest browser stop failed")


@tool(
    "find_lab_test",
    "Home blood tests: live prices, fasting needs, report times and the earliest home-collection slots from several labs "
    "(Tata 1mg Labs, Healthians, Redcliffe, PharmEasy, Apollo) for the care recipient's address. Takes about 15 seconds; "
    "say you are checking first. Nothing is booked. Read back the 2-3 best options (cheapest, earliest) in plain words.",
    {"test": {"type": "string", "description": "The test or package as they said it: 'HbA1c', 'CBC', 'thyroid', 'full body checkup'"}},
    ["test"],
)
async def find_lab_test(ctx: TurnCtx, a: dict) -> dict:
    from app.tasks import care_search

    place = await _care_place(ctx)
    rows = await _with_browser(lambda cdp: care_search.labs(cdp, a["test"], lat=place.get("lat"), lon=place.get("lng"),
                                                            pincode=place.get("pincode")))
    cheapest = sorted(((o, r["lab"]) for r in rows if r.get("available") for o in r["options"] if o.get("price")),
                      key=lambda x: float(str(x[0]["price"]).strip("₹") or 0))
    return {"labs": rows, "cheapest": [{"lab": lab, **o} for o, lab in cheapest[:3]], "address": place.get("nickname") or place.get("full"),
            "next": "Booking needs the family's login on the chosen lab's site; ask which one they want and tell the caregiver."}


_NUTRIENT_WORDS = frozenset({"sugar", "fat", "protein", "calorie", "calories", "kcal", "carb", "carbs", "sodium", "fibre", "fiber"})


def _asks_nutrient(question: str) -> bool:
    """The brain's English question, not the person's words: is it asking for a nutrient number."""
    return bool(_NUTRIENT_WORDS & {w.strip(".,?!") for w in (question or "").lower().split()})


async def _logged_nutrient(ctx: TurnCtx, question: str) -> bool:
    """True when they already have food logged today and the question asks for a nutrient of it."""
    if not _asks_nutrient(question):
        return False
    from app.nutrition import entries as food_entries

    day = clock.ist_day()
    people = {p for p in (ctx.speaker.get("id"), ctx.elder_id) if p}
    for person in people:
        rows = await food_entries.between(ctx.session, ctx.family_id, person, day, day)
        if any(r.kind == "food" and r.status == "active" for r in rows):
            return True
    return False


# Every web result, whatever it is about (live 18 Oct: a search about Legacies gave away its finale to someone on S1E4).
SEARCH_NOTE = ("From the internet. Tell it simply; name the source only if they ask or it matters. A show, film, book, game "
               "or match they are still watching or have not seen: never say what happens after the point they told you "
               "(an ending, a death, a twist, a cameo, a later season, a result); if you do not know where they are, ask "
               "first. Only on their clear ask for that exact spoiler.")


@tool(
    "web_search",
    "Look something up on the internet when you need facts you do not have and cannot see: what a thing, food, plant or "
    "product is and what it is for; a place's hours or address; weather; festival or holiday dates; train timings; news; "
    "general health information (from reliable sites; never to change a medicine or dose). Ask one clear question in "
    "English. You get a short answer with sources: tell it simply in their language. Not for the family's own information "
    "(memory and the care record), not for the nutrients of food already in the food log (food_log what=entries), and not for "
    "prices or ordering (start_task).",
    {"question": {"type": "string", "description": "One clear question in English, with the place when it matters (e.g. 'Raipur weather tomorrow')"}},
    ["question"],
)
async def web_search(ctx: TurnCtx, a: dict) -> dict:
    from app.brain import websearch

    q = str(a.get("question") or "").strip()
    if not q:
        raise ToolRefused("Ask one clear question.")
    if await _logged_nutrient(ctx, q):
        raise ToolRefused("That nutrient is already in the food log. Call food_log with what=entries and use those numbers. "
                          "Do not look up nutrients of food already logged.")
    today = await store.events(ctx.session, ctx.family_id, ctx.elder_id, since=clock.now() - timedelta(hours=24), kinds=["web_search"])
    if len(today) >= websearch.daily_limit():
        raise ToolRefused("Today's internet look-ups for this family are used up. Answer from what you know, and say plainly if you are not sure.")
    try:
        out = await websearch.search(q)
    except Exception as exc:  # noqa: BLE001
        logger.warning("web search failed: %s", exc)
        # hard-turn A/B 14 Oct: "My search tool isn't working right now, so I can't look up the side effects" to a health question
        raise ToolRefused("The internet look-up did not work just now. Answer from what is well known (a medicine's common side effects "
                          "and precautions as its label gives them, general self-care, warning signs that need a doctor) as general "
                          "information, with when to see the doctor; guess nothing specific to them, and never mention looking it up, "
                          "searching or tools.") from exc
    await store.record_event(ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="web_search",
                             summary=f"Looked up: {q[:160]}", payload={"question": q[:300], "sources": out["sources"]},
                             actor_id=ctx.speaker.get("id") or "saheli")
    return {**out, "note": SEARCH_NOTE}


@tool(
    "ways_there",
    "Other ways to make a trip when a ride app cannot book it, or they want cheaper. Looks up the airport prepaid taxi, "
    "the airport bus, the metro, a city bus, an auto and a cab on their own app. Returns short messages: send each one, "
    "in their language, and add nothing. A fare or a route that was not on the page is not in the messages. Call this "
    "instead of guessing. Nothing is booked or paid.",
    {
        "origin": {"type": "string", "description": "Where the trip starts, as they named it (a terminal only when the city is known)"},
        "destination": {"type": "string", "description": "Where it ends"},
        "at": {"type": "string", "description": "When, ISO 8601 in IST, when known"},
        "elder": {"type": "boolean", "description": "True when the person travelling is the care recipient or an elder"},
    },
    ["origin", "destination"],
)
async def ways_there(ctx: TurnCtx, a: dict) -> dict:
    from app.care import ways
    from app.tasks import later

    origin, dest = str(a.get("origin") or "").strip(), str(a.get("destination") or "").strip()
    if not origin or not dest:
        raise ToolRefused("Give origin and destination, the places they named.")
    at = later.parse_at(a.get("at")) if a.get("at") else None
    today = await store.events(ctx.session, ctx.family_id, ctx.elder_id, since=clock.now() - timedelta(hours=24), kinds=["web_search"])
    if len(today) >= websearch_limit():
        return {"found": False, "messages": [ways.NOT_FOUND]}
    try:
        raw, sources = await ways.fetch(origin, dest, at)
        proposed = await ways.judge(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ways look-up failed: %s", exc)
        return {"found": False, "messages": [ways.NOT_FOUND]}
    options = ways.options_from(proposed, raw, at=at, elder=bool(a.get("elder")))
    await store.record_event(ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="web_search",
                             summary=f"Ways from {origin[:80]} to {dest[:80]}", payload={"sources": sources},
                             actor_id=ctx.speaker.get("id") or "saheli")
    if not options:
        return {"found": False, "messages": [ways.NOT_FOUND], "sources": sources}
    return {"found": True, "options": options, "messages": ways.messages(options), "sources": sources,
            "note": "Send these messages, one each, in their language. Add no fare, route or time that is not here. "
                    "Nothing was booked and nothing was paid."}


@tool(
    "note_trip",
    "Save a journey from a forwarded airline message or from what they said: airline, flight number, from, to, and the "
    "landing time. A public flight-status page can move the landing time only when that time is written on the page. "
    "Then say what was saved and the one next useful step. Do not book a cab and do not check in from this message. "
    "Ask only for a field this refuses.",
    {
        "airline": {"type": "string"},
        "flight": {"type": "string", "description": "The flight number as written, such as 6E-204"},
        "origin": {"type": "string", "description": "Where it takes off, city or airport"},
        "destination": {"type": "string", "description": "Where it lands"},
        "arrives": {"type": "string", "description": "Landing time, ISO 8601 in IST"},
        "departs": {"type": "string", "description": "Departure time, ISO 8601 in IST, when known"},
        "pnr": {"type": "string", "description": "The booking reference, when the message has one"},
        "seat": {"type": "string", "description": "The seat, when the booking shows one"},
        "about": ABOUT_OWN,
    },
    ["airline", "flight", "origin", "destination", "arrives"],
)
async def note_trip(ctx: TurnCtx, a: dict) -> dict:
    from app.care import ways
    from app.care.domains import fact_key
    from app.tasks import later

    arrives = later.parse_at(a.get("arrives"))
    if arrives is None:
        raise ToolRefused("arrives must be a date and time like 2026-10-16T00:30 (IST). Ask for the landing time.")
    flight = str(a.get("flight") or "").strip()
    if not flight:
        raise ToolRefused("Give the flight number.")
    moved = None
    try:
        raw, _sources = await ways.fetch_text(f"{a.get('airline') or ''} {flight} flight status arrival time")
        moved = ways.checked_arrival(await ways.judge_time(raw, flight), raw, flight)
    except Exception as exc:  # noqa: BLE001 — no page text: the forwarded time stands
        logger.warning("flight status not checked: %s", exc)
    if moved:
        arrives = ways.retimed(arrives, moved)
    subject = ctx.own(a.get("about"))
    cab = ways.pickup_after_landing(arrives)
    departs = later.parse_at(a.get("departs")) if a.get("departs") else None
    origin, destination = str(a.get("origin") or "").strip(), str(a.get("destination") or "").strip()
    airline = str(a.get("airline") or "").strip()
    value = {"airline": airline, "carrier": airline, "flight": flight, "number": flight,
             "origin": origin, "from": origin, "destination": destination, "to": destination,
             "arrives": arrives.isoformat(), "arrive": arrives.isoformat(), "cab_at": cab.isoformat(),
             **({"departs": departs.isoformat(), "depart": departs.isoformat()} if departs else {}),
             **({"pnr": str(a["pnr"]).strip()} if str(a.get("pnr") or "").strip() else {}),
             **({"passengers": [str(p) for p in a["passengers"] if str(p).strip()]} if isinstance(a.get("passengers"), list) else {}),
             **({"from_terminal": str(a["from_terminal"]).strip()} if str(a.get("from_terminal") or "").strip() else {}),
             **({"to_terminal": str(a["to_terminal"]).strip()} if str(a.get("to_terminal") or "").strip() else {}),
             **({"flight_status": str(a["flight_status"]).strip()} if str(a.get("flight_status") or "").strip() else {}),
             **({"seat": str(a["seat"]).strip()} if str(a.get("seat") or "").strip() else {})}
    sentence = f"{airline} {flight} from {origin} to {destination}, lands {later.hm(arrives)}".strip()
    w = await store.write_fact(
        ctx.session, family_id=ctx.family_id, subject_id=subject, domain="trip", key=fact_key("trip", flight),
        value=value, text=sentence, source_kind=ctx.source_kind, source_ref=ctx.message_ref,
        stated_by=ctx.speaker.get("id"), confidence=1.0)
    from app.care import features

    reminders = await features.trip_loops(ctx.session, family_id=ctx.family_id, subject_id=subject, f=w.fact, owner_id=ctx.speaker.get("id"))
    if not reminders:
        wake = arrives - timedelta(hours=3)
        if wake > clock.now():
            loop = await store.open_loop(
                ctx.session, family_id=ctx.family_id, subject_id=subject, kind="trip",
                title=f"{airline} {flight} lands {later.hm(arrives)}: offer a cab from the airport. Do not book it.",
                detail={"key": w.fact.key, "when": arrives.isoformat(), "max_wakes": 1, "stage": "cab"},
                owner_id=ctx.speaker.get("id"), wake_at=wake, alert_rule="dashboard", dedupe_key=f"trip:{subject}:{w.fact.key}:cab")
            reminders = [str(loop.id)]
    return {"saved": sentence, "key": w.fact.key, "arrives": arrives.isoformat(), "cab_at": cab.isoformat(),
            "landing_moved": bool(moved), "reminder": reminders[0] if reminders else None, "reminders": reminders,
            "next": "One short message: what you saved, then the one next useful step (a cab, check-in, or telling the family). "
                    "The reminder is already set. Do not book the cab, do not check in, and do not say either is done. "
                    "When they say yes to that offer, call accept_trip. Do not message anyone until they say yes."}


_AIRLINES = {
    "indigo": "indigo", "airindia": "airindia", "air india": "airindia",
    "airindiaexpress": "airindiaexpress", "air india express": "airindiaexpress",
    "akasa": "akasa", "akasa air": "akasa", "spicejet": "spicejet",
}


def _airline_service(name: str) -> str | None:
    key = " ".join((name or "").lower().split())
    return _AIRLINES.get(key) or _AIRLINES.get(key.replace(" ", ""))


def _surname(ctx: TurnCtx, subject_id: str) -> str:
    """The last word of the name on the care record. A single name is not a last name."""
    name = ctx.member_name(subject_id)
    if not name or name == subject_id:
        return ""
    parts = name.split()
    return parts[-1] if len(parts) >= 2 else ""


@tool(
    "accept_trip",
    "They said yes to an offer on a journey already saved. offer is cab or checkin. Call this only after that yes, "
    "never from the message that shared the booking. A cab is looked up and still needs their fare confirm. "
    "Check-in never pays: a free seat only, and it stops if the seat or an add-on costs money.",
    {
        "offer": {"type": "string", "enum": ["cab", "checkin"]},
        "flight": {"type": "string", "description": "The flight number, when more than one journey is saved"},
        "last_name": {"type": "string", "description": "Check-in only: the last name on the ticket"},
        "pnr": {"type": "string", "description": "Check-in only, when it was not on the saved journey"},
        "service": {"type": "string", "description": "Cab only, when they named one app: uber, ola or rapido"},
        "when": {"type": "string", "enum": ["scheduled", "lands"], "description": "Cab only. scheduled pre-books for 45 minutes after landing. lands watches the flight and books only once it has landed."},
        "about": ABOUT_OWN,
    },
    ["offer"],
)
async def accept_trip(ctx: TurnCtx, a: dict) -> dict:
    from app.tasks import later, runtime
    from app.tasks.skills import SKILLS

    offer = a.get("offer")
    if offer not in ("cab", "checkin"):
        raise ToolRefused("offer is cab or checkin, and only after they said yes to that offer.")
    subject = ctx.own(a.get("about"))
    rows = [f for f in await store.facts(ctx.session, ctx.family_id, subject, domains=["trip"], statuses=("active",)) if f.status == "active"]
    flight = str(a.get("flight") or "").strip()
    if flight:
        rows = [f for f in rows if str((f.value or {}).get("flight") or (f.value or {}).get("number") or "") == flight]
    if not rows:
        raise ToolRefused("No journey is saved for them. Ask them to share the booking. Do not book and do not check in.")
    fact = rows[0]
    v = fact.value or {}
    origin = str(v.get("origin") or v.get("from") or "").strip()
    destination = str(v.get("destination") or v.get("to") or "").strip()
    number = str(v.get("flight") or v.get("number") or "").strip()
    if offer == "cab" and a.get("when") == "lands":
        arrives = later.parse_at(v.get("arrive") or v.get("arrives"))
        if arrives is None or not destination:
            raise ToolRefused("The journey has no landing place and no time. Ask where and when. Do not book.")
        wake = arrives - timedelta(hours=2)
        if wake <= clock.now():
            wake = clock.now()
        loop = await store.open_loop(
            ctx.session, family_id=ctx.family_id, subject_id=subject, kind="trip",
            title=(f"{number or 'The flight'} lands at {destination}: watch it from two hours before. "
                   "If it is delayed, keep the new landing. Book the cab only once it has landed, 45 minutes after, "
                   "within what they will pay. Do not book before it lands."),
            detail={"key": fact.key, "when": arrives.isoformat(), "max_wakes": 4, "stage": "lands"},
            owner_id=ctx.speaker.get("id"), wake_at=wake, alert_rule="dashboard",
            dedupe_key=f"trip:{subject}:{fact.key}:lands")
        return {"started": False, "watching": True, "booked": False, "paid": False, "wake_at": wake.isoformat(),
                "reminder": str(loop.id),
                "next": "Say you will watch the flight and book the cab only once it has landed, 45 minutes after landing, "
                        "within what they will pay. Ask in this one message for anything still missing: the pickup, the drop, "
                        "and the most they will pay. Do not book now. Do not message the family."}
    if offer == "cab":
        depart = later.parse_at(v.get("depart") or v.get("departs"))
        arrives = later.parse_at(v.get("arrive") or v.get("arrives"))
        ride: dict = {"kind": "ride", "for_whom": subject, "goal": f"Cab for {number or 'the journey'}"}
        if a.get("service"):
            ride["service"] = a["service"]
        if depart and origin:
            ride["pickup"], ride["drop"] = "home", origin
            ride["at"] = (depart - timedelta(hours=2)).isoformat()
            ride["at_means"] = "arrive"
        elif arrives and destination:
            ride["pickup"], ride["drop"] = destination, "home"
            ride["landing"] = arrives.isoformat()
        else:
            raise ToolRefused("The journey has no place and no time. Ask where and when. Do not book.")
        out = await start_task(ctx, ride)
        out["note"] = "Nothing is booked until they confirm a fare."
        return out
    service = _airline_service(str(v.get("airline") or v.get("carrier") or ""))
    if service not in SKILLS:
        raise ToolRefused("Ask which airline: IndiGo, Air India, Air India Express, Akasa or SpiceJet. Do not open a site until they say.")
    pnr = str(a.get("pnr") or v.get("pnr") or "").strip()
    last = str(a.get("last_name") or "").strip()
    if not pnr or not last:
        return {"started": False, "next": "Ask for the booking reference and the last name on the ticket. "
                                           "Do not open the airline until both are here. Nothing is checked in."}
    label = SKILLS[service]["label"]
    goal = (f"Web check-in on {label} for flight {number}. Booking reference {pnr}, last name {last}. Open web check-in. "
            "Never pay, and never press a card, UPI or wallet button. Take only a free seat, aisle or window, nearer the front. "
            "If there is no free seat, stop and report free_seat=false and checked_in=false. Skip every paid seat, meal and bag. "
            "If check-in is not open, stop and report not_open=true. If the booking is not found, stop and report pnr_found=false. "
            "When it finishes, save the boarding pass and report checked_in=true, seat, paid=false and boarding_pass=true. "
            "Do not say it is done unless checked_in is true.")
    from app.specialists.agents import specialist_for
    from app.specialists.contract import build_limits

    who = subject
    limits = await build_limits(ctx.session, family_id=ctx.family_id, subject_id=who, kind="checkin", agent=specialist_for(service).name,
                                requester_is_elder=False, place=None)
    task = await runtime.create(
        ctx.session, family_id=ctx.family_id, subject_id=who, requested_by=ctx.speaker.get("id") or ctx.elder_id,
        service=service, kind="checkin", goal=goal,
        details={"pnr": pnr, "last_name": last, "flight": number, "for_set": True}, limits=limits)
    return {"started": True, "task_id": str(task.id), "update_coming": True,
            "next": f"One short line: check-in on {label} is started. Do not say it is done. A task update brings the seat or why it stopped. Nothing is paid."}


@tool(
    "read_trip",
    "Read a booking or a public flight time. Nothing on the booking is changed, cancelled or paid. "
    "A booking needs the airline and the booking reference. The last name defaults to the one on the care record for the person it is for. "
    "Ask only when that name is missing or the site rejects it, in one message. A flight number alone uses the public status page. "
    "If the booking is not found, or the page will not open, say so and ask them to forward the airline's message.",
    {
        "airline": {"type": "string", "description": "IndiGo, Air India, Air India Express, Akasa or SpiceJet"},
        "pnr": {"type": "string", "description": "The booking reference, when they gave one"},
        "last_name": {"type": "string", "description": "The last name on the ticket"},
        "flight": {"type": "string", "description": "The flight number, when there is no booking reference"},
        "about": ABOUT_OWN,
    },
    [],
)
async def read_trip(ctx: TurnCtx, a: dict) -> dict:
    import os

    from app.tasks import checkin_desk
    from app.tasks.skills import SKILLS

    who = ctx.own(a.get("about"))
    service = _airline_service(str(a.get("airline") or ""))
    pnr, last, flight = (str(a.get(k) or "").strip() for k in ("pnr", "last_name", "flight"))
    if pnr and not last:
        last = _surname(ctx, who)
    if pnr and not last:
        return {"found": False, "started": False, "next": "Ask for the last name on the ticket, in this one message. Do not ask for anything else. Nothing was opened."}
    if not pnr and not flight:
        return {"found": False, "started": False, "next": "Ask for the booking reference or the flight number, in this one message. Nothing was opened."}
    if not service:
        return {"found": False, "started": False, "next": "Ask which airline, in this one message. Nothing was opened."}
    base = os.environ.get("CHECKIN_DESK", "").strip()
    if base:
        got = await (checkin_desk.read_booking(base, service, pnr=pnr, last_name=last) if pnr
                     else checkin_desk.read_status(base, service, flight=flight))
        if got.get("name_rejected"):
            return {"found": False, "started": False, "changed": False, "paid": False, "name_rejected": True,
                    "next": "The last name was not accepted. Ask for the last name on the ticket, in this one message. Nothing was changed and nothing was paid."}
        if not got.get("found"):
            why = "the airline page would not open" if got.get("blocked") else "it was not found"
            return {"found": False, "started": False, "changed": False, "paid": False, "blocked": bool(got.get("blocked")),
                    "next": f"Say plainly that {why}. Ask them to forward the airline's message. Nothing was changed and nothing was paid."}
        departs = f"{got['date']}T{got['departs']}" if got.get("date") and got.get("departs") else None
        arrives = f"{got['date']}T{got['arrives']}" if got.get("date") and got.get("arrives") else None
        if not arrives or not got.get("flight"):
            return {"found": False, "started": False, "changed": False, "next": "The page did not show a flight time. Say that. Do not invent one."}
        saved = await note_trip(ctx, {"airline": a.get("airline"), "flight": got["flight"], "origin": got.get("origin") or "",
                                      "destination": got.get("destination") or "", "arrives": arrives,
                                      **({"departs": departs} if departs else {}),
                                      **({"pnr": pnr} if pnr else {}),
                                      "passengers": got.get("passengers") or [],
                                      "from_terminal": got.get("from_terminal") or "",
                                      "to_terminal": got.get("to_terminal") or "",
                                      "flight_status": got.get("status") or "",
                                      **({"seat": got["seat"]} if got.get("seat") else {}),
                                      "about": a.get("about")})
        seat = f" Seat {got['seat']}." if got.get("seat") else ""
        return {"found": True, "changed": False, "paid": False, "flight": got["flight"], "departs": departs, "arrives": arrives,
                "from_terminal": got.get("from_terminal"), "to_terminal": got.get("to_terminal"),
                "status": got.get("status"), "passengers": got.get("passengers") or [], "seat": got.get("seat"),
                "saved": saved.get("saved"),
                "next": "Read back the flight, the times, the terminals, whether it is on time, and the passengers."
                        f"{seat} Say nothing was changed and nothing was paid. Do not check in and do not book a cab unless they ask."}
    label = SKILLS[service]["label"]
    if pnr:
        goal = (f"Read the booking on {label}. Open manage booking. Booking reference {pnr}, last name {last}. "
                f"Read the flight number, date, departure, arrival, terminals, on time or delayed, and the passengers. "
                "Do not change the booking. Do not cancel. Do not check in. Do not pay. "
                "If it is not found, stop and report pnr_found=false. Report changed=false and paid=false.")
    else:
        goal = (f"Open the public flight-status page for {flight} on {label}. Do not log in and do not submit a booking. "
                "Read the flight number, date, departure, arrival, terminals and whether it is on time or delayed. "
                "Do not change anything. Do not pay. If the flight is not listed, say it was not found.")
    from app.specialists.agents import specialist_for
    from app.specialists.contract import build_limits
    from app.tasks import runtime

    who = ctx.own(a.get("about"))
    limits = await build_limits(ctx.session, family_id=ctx.family_id, subject_id=who, kind="checkin", agent=specialist_for(service).name,
                                requester_is_elder=False, place=None)
    task = await runtime.create(
        ctx.session, family_id=ctx.family_id, subject_id=who, requested_by=ctx.speaker.get("id") or ctx.elder_id,
        service=service, kind="checkin", goal=goal,
        details={"pnr": pnr, "last_name": last, "flight": flight, "read_only": True, "for_set": True}, limits=limits)
    task.phase = "browse"
    return {"found": False, "started": True, "task_id": str(task.id), "update_coming": True, "changed": False,
            "next": "One short line: you are reading it. Do not say the time until the task update has it. Nothing is changed and nothing is paid."}


def websearch_limit() -> int:
    from app.brain import websearch

    return websearch.daily_limit()


@tool(
    "find_doctor",
    "Doctors near the care recipient for a specialty or a problem ('cardiologist', 'knee pain', 'sugar doctor'): fee, "
    "experience, clinic, next free slot, clinic visit or video, from Practo and Apollo 24|7. Takes about 15 seconds. "
    "Nothing is booked. Read back the 2-3 best options.",
    {"need": {"type": "string"}, "city": {"type": "string", "description": "Only if they asked for another city"}},
    ["need"],
)
async def find_doctor(ctx: TurnCtx, a: dict) -> dict:
    from app.tasks import care_search

    place = await _care_place(ctx)
    city = a.get("city") or care_search.city_of(place)
    if not city:
        return {"error": "no_city", "next": "No home place is saved. Ask which city or area to look in (one short question), "
                                             "then call find_doctor again with city."}
    rows = await _with_browser(lambda cdp: care_search.doctors(cdp, a["need"], city=city, lat=place.get("lat"), lon=place.get("lng")))
    return {"city": city, "sites": rows,
            "next": "To book, the family picks a doctor and a slot; booking needs their login on that site. Offer to note the "
                    "appointment once booked (remember appointment) so reminders and questions-for-the-doctor work."}
