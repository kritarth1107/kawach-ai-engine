"""One read for the person-centred pages (founder-approved design, 2026-10-13: Care plan, Story, Emergency card).

What the record already holds, shaped for people rather than tables: who they are (age, home, how they're called), how
they've been (mood through the last two weeks), what Saheli learned lately with the words it came from, the people
around them, their day (routine facts with a time), and a week of doses for the pillbox. Read only; nothing is guessed:
a field the record does not hold is left out.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import mood, store
from app.care.domains import is_none_key
from app.care.models import CareEvent, CareFact, Turn
from app.core import clock

LEARN_KINDS = ("caregiver_said", "elder_said", "inferred", "dashboard", "prescription", "lab")
PEOPLE_DOMAINS = ("family", "contact", "doctor", "home")


def _hhmm(v) -> str | None:
    """A clock time from a structured routine value (HH:MM); None when the fact holds none."""
    if not isinstance(v, str) or len(v) != 5 or v[2] != ":":
        return None
    h, m = v[:2], v[3:]
    return v if h.isdigit() and m.isdigit() and int(h) < 24 and int(m) < 60 else None


def routine_time(value: dict) -> str | None:
    for k in ("time", "start_time", "at"):
        t = _hhmm((value or {}).get(k))
        if t:
            return t
    times = (value or {}).get("times")
    if isinstance(times, list):
        for t in times:
            if _hhmm(t):
                return t
    return None


def _value_text(value: dict, *keys: str) -> str | None:
    for k in keys:
        v = (value or {}).get(k)
        if isinstance(v, (str, int, float)) and str(v).strip():
            return str(v).strip()
    return None


async def _said(session: AsyncSession, family_id: str, facts: list[CareFact]) -> dict[str, dict]:
    """The person's own words behind each fact: the message it was saved from, or what the re-read heard."""
    refs: dict[str, CareFact] = {f.source_ref: f for f in facts if f.source_ref}
    out: dict[str, dict] = {}
    if refs:
        rows = (await session.execute(
            select(Turn.message_ref, Turn.text, Turn.at, Turn.meta).where(Turn.family_id == family_id, Turn.message_ref.in_(list(refs)))
        )).all()
        for ref, text, at, meta in rows:
            if ref in refs and text and not text.startswith("["):
                out[str(refs[ref].id)] = {"text": text[:280], "at": at.isoformat(), "voice": bool((meta or {}).get("media")), "heard": False}
    for f in facts:
        note = f.note or ""
        if str(f.id) not in out and note.startswith("heard: "):
            heard = note[len("heard: "):].strip().strip('"').strip()
            if heard:
                # the re-read's own pick of words: shown as heard in the chat, never as the message a fact was saved from
                out[str(f.id)] = {"text": heard[:280], "at": f.recorded_at.isoformat(), "voice": False, "heard": True}
    return out


async def _activities(session: AsyncSession, family_id: str, subject_id: str) -> list[dict]:
    from app.care import activities

    return await activities.status(session, family_id, subject_id)


GAPS = ("medicine", "allergy", "condition", "blood_group", "emergency_contact", "doctor", "hospital", "address", "routine",
        "weight", "enjoys", "language")


async def _gaps(session: AsyncSession, family_id: str, subject_id: str, active: list[CareFact], others: int) -> dict:
    """What Saheli does not know yet about them, in order of how much it helps (the dashboard asks for the missing ones).
    A "has none" answer counts as known."""
    domains = {f.domain for f in active}
    keys = {f.key for f in active}
    weighed = (await session.execute(select(func.count()).select_from(CareEvent).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "vital",
        CareEvent.at >= clock.now() - timedelta(days=45), CareEvent.payload["kind"].astext == "weight"))).scalar_one()
    known = {
        "medicine": "medicine" in domains,
        "allergy": "allergy" in domains,
        "condition": "condition" in domains,
        "blood_group": "profile:blood_group" in keys,
        "emergency_contact": others > 0 or any(f.domain in ("contact", "family") and _value_text(f.value, "phone") for f in active),
        "doctor": "doctor" in domains,
        "hospital": "hospital" in domains,
        "address": any(f.domain == "home" and _value_text(f.value, "address") for f in active),
        "routine": bool(domains & {"routine", "activity"}),
        "weight": weighed > 0,
        "enjoys": bool(domains & {"preference", "dish", "occasion"}),
        "language": "language" in domains,
    }
    items = [{"key": k, "known": known[k]} for k in GAPS]
    return {"known": sum(known.values()), "total": len(GAPS), "items": items}


async def week_doses(session: AsyncSession, family_id: str, subject_id: str, days: int = 7) -> list[dict]:
    from app.api.dash import _day_doses  # the one dose list Home and the Day page show

    meds = await store.facts(session, family_id, subject_id, domains=["medicine"], statuses=("active",))
    meds = [m for m in meds if not is_none_key(m.key)]
    today = clock.ist()
    out = []
    for i in range(days - 1, -1, -1):
        day = clock.ist_day(today - timedelta(days=i))
        evs = await store.events(session, family_id, subject_id, day=day, limit=500)
        out.append({"day": day, "doses": _day_doses(meds, list(evs), day)})
    return out


async def build(session: AsyncSession, family_id: str, subject_id: str, *, names: dict[str, str] | None = None) -> dict:
    now = clock.now()
    roster = await store.roster(session, family_id)
    members = [*([roster.elder] if roster and roster.elder else []), *((roster.members or []) if roster else [])]
    names = {**{m.get("id"): m.get("name") for m in members if m.get("id")}, **(names or {})}
    facts = [f for f in await store.facts(session, family_id, subject_id, statuses=("active", "pending"))]
    active = [f for f in facts if f.status == "active"]
    by_key = {f.key: f for f in active}

    def val(key: str, *fields: str) -> str | None:
        f = by_key.get(key)
        return _value_text(f.value, *fields) if f else None

    homes = [f for f in active if f.domain == "home" and _value_text(f.value, "address")]
    lives = by_key.get("home:lives_with")
    person = {
        "name": names.get(subject_id),
        "age": val("profile:age", "age", "value"),
        "callAs": val("naming:address_as", "name"),
        "language": (lang.text if (lang := by_key.get("language:preferred")) else None),
        "livesWith": (lives.value.get("with") if lives and isinstance(lives.value.get("with"), str) else None),
        "city": (lives.value.get("city") if lives and isinstance(lives.value.get("city"), str) else None)
                or next((h.value.get("city") for h in homes if isinstance(h.value.get("city"), str)), None),
        "address": homes[0].value.get("address") if homes else None,
        "mobility": val("profile:mobility", "value", "mobility"),
    }

    # how they've been: the last mood of each day, 14 days
    since = now - timedelta(days=14)
    moods = list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "mood",
                                CareEvent.at >= since).order_by(CareEvent.at)
    )).scalars())
    per_day: dict[str, dict] = {}
    for e in moods:
        j = mood.as_json(e)
        if j and j["score"]:
            per_day[e.day] = {"day": e.day, **j}
    mood_now = await mood.now_for(session, family_id, subject_id)

    # learned lately, with the words it came from
    recent = sorted((f for f in facts if f.source_kind in LEARN_KINDS and f.recorded_at >= now - timedelta(days=14)
                     and f.domain not in ("language",)), key=lambda f: f.recorded_at, reverse=True)[:16]
    said = await _said(session, family_id, recent)
    learned = [{
        "id": str(f.id), "key": f.key, "domain": f.domain, "text": f.text, "status": f.status, "source": f.source_kind,
        "statedBy": names.get(f.stated_by) if f.stated_by else None, "at": f.recorded_at.isoformat(), "said": said.get(str(f.id)),
    } for f in recent]
    week_ago = now - timedelta(days=7)
    stats = {
        "known": len(active),
        "learnedWeek": sum(1 for f in facts if f.recorded_at >= week_ago and f.source_kind in LEARN_KINDS),
        "pending": sum(1 for f in facts if f.status == "pending"),
    }

    # the people around them: household members, then the record's family, contacts, doctors
    people: list[dict] = []
    for m in store.unique_members(members):
        if m.get("id") and m.get("id") != subject_id:
            people.append({"id": m["id"], "name": m.get("name") or "", "relation": m.get("role") or "", "kind": "member"})
    seen = {p["name"].lower() for p in people}
    for f in active:
        if f.domain not in ("family", "contact", "doctor"):
            continue
        name = _value_text(f.value, "name")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        people.append({"name": name, "relation": _value_text(f.value, "relation", "speciality") or ("doctor" if f.domain == "doctor" else ""),
                       "phone": _value_text(f.value, "phone"), "note": f.text, "kind": f.domain, "key": f.key})

    day_items = sorted(({"time": t, "label": f.text, "key": f.key} for f in active if f.domain == "routine" and (t := routine_time(f.value))),
                       key=lambda d: d["time"])
    enjoys = [{"key": f.key, "text": f.text, "domain": f.domain} for f in active if f.domain in ("preference", "dish", "occasion")]
    naming = by_key.get("naming:address_as")
    avoid = [*(f"Never call them \"{w}\"" for w in ((naming.value.get("avoid") or []) if naming else []) if isinstance(w, str) and w),
             *(f.text for f in active if f.domain == "family" and f.key.split(":", 1)[-1].startswith("do_not"))]

    first_turn = (await session.execute(
        select(func.min(Turn.at)).where(Turn.family_id == family_id, Turn.thread_id == subject_id)
    )).scalar_one_or_none()
    return {
        "person": person,
        "mood": {"now": mood_now, "days": list(per_day.values())},
        "stats": stats,
        "learned": learned,
        "people": people[:12],
        "day": day_items,
        "enjoys": enjoys[:8],
        "avoid": avoid[:6],
        "week": await week_doses(session, family_id, subject_id),
        "activities": await _activities(session, family_id, subject_id),
        "gaps": await _gaps(session, family_id, subject_id, active, sum(1 for m in members if m.get("id") and m.get("id") != subject_id)),
        "knownSince": first_turn.isoformat() if first_turn else None,
    }
