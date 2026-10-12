"""Activities and habits: the Care Memory OS's daily life (founder 2026-10-13).

"gym or regular stuff ... just don't keep them as reminders, keep them as good things: regularly ask what they did in the
gym, what time they come back, confirm they came back, ask what they did and follow up on their answers ... make the
memory and health data." A plain reminder ("call Yash at 8:30") stays a reminder.

An activity is a care fact (domain "activity") with a type from the catalogue, days and times. Its kind decides what
Saheli does around each occurrence:
  habit    a gentle nudge before (optional) and a check-in after: did it happen, what did they do, then follow-ups;
  quit     something they are giving up (smoking): an evening check on how the day went, never shaming;
  task     a one-off-ish chore with a time (pay a bill): one nudge at the time, a check later only if it matters;
  routine  meals, wake-up, sleep: no messages of its own; it tells Saheli when to (not) message.
What they tell her is logged (activity_done / activity_skipped events, details kept), readings go to health data, and
streaks are counted. Check-ins are open loops planned a day ahead by the wake job and written by the brain at the time.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.domains import slug
from app.care.models import CareEvent, CareFact, OpenLoop
from app.core import clock

WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# What each category asks after, and what is worth keeping (a hint to the brain; the words are always hers, fresh).
CATEGORY = {
    "fitness": ("what they did (exercises, distance, how long), how hard it felt, any pain, and whether they ate after",
                ["duration_min", "what", "intensity", "distance_km", "steps", "pain", "energy"]),
    "health": ("whether it was done and the reading or result, how they feel", ["done", "reading", "result", "how_feel"]),
    "mind": ("how it went and how they feel now", ["duration_min", "what", "mood"]),
    "spiritual": ("how it went, anything special today (a festival, a visit)", ["done", "where", "with_whom", "mood"]),
    "food": ("what they had and whether it fit their diet", ["what", "amount", "on_diet"]),
    "sleep": ("how long and how well they slept, anything that woke them", ["hours", "quality", "woke_up"]),
    "self_care": ("whether they got to it and how they feel", ["done", "mood"]),
    "social": ("who they met or spoke to and how it went", ["with_whom", "what", "mood"]),
    "work": ("how the day went, stress, anything to remember", ["how_it_went", "stress", "notes"]),
    "home": ("whether it got done and if they need help with anything", ["done", "notes"]),
    "errand": ("whether it got done and anything left over", ["done", "notes"]),
    "care": ("how the person they care for is, and how they themselves are coping", ["how_they_are", "notes"]),
    "quit": ("how the day went with it, any craving and what helped; praise every win, never shame", ["avoided", "cravings", "count", "what_helped"]),
    "hobby": ("what they made or enjoyed", ["what", "duration_min", "mood"]),
}

# (id, label, category, kind, typical minutes)
_ROWS = [
    # fitness: habits
    ("gym", "Gym", "fitness", "habit", 60), ("strength_training", "Strength training", "fitness", "habit", 45),
    ("walk", "Walk", "fitness", "habit", 30), ("morning_walk", "Morning walk", "fitness", "habit", 30),
    ("evening_walk", "Evening walk", "fitness", "habit", 30), ("run", "Run", "fitness", "habit", 30), ("jog", "Jog", "fitness", "habit", 30),
    ("cycling", "Cycling", "fitness", "habit", 45), ("swimming", "Swimming", "fitness", "habit", 45), ("yoga", "Yoga", "fitness", "habit", 40),
    ("pranayama", "Pranayama", "fitness", "habit", 20), ("stretching", "Stretching", "fitness", "habit", 15),
    ("physiotherapy", "Physiotherapy exercises", "fitness", "habit", 30), ("home_workout", "Home workout", "fitness", "habit", 30),
    ("dance", "Dance", "fitness", "habit", 45), ("zumba", "Zumba", "fitness", "habit", 45), ("aerobics", "Aerobics", "fitness", "habit", 45),
    ("pilates", "Pilates", "fitness", "habit", 45), ("badminton", "Badminton", "fitness", "habit", 60), ("tennis", "Tennis", "fitness", "habit", 60),
    ("cricket", "Cricket", "fitness", "habit", 90), ("football", "Football", "fitness", "habit", 90), ("sports", "Sports", "fitness", "habit", 60),
    ("hiking", "Hiking", "fitness", "habit", 120), ("tai_chi", "Tai chi", "fitness", "habit", 30), ("laughter_club", "Laughter club", "fitness", "habit", 30),
    # health
    ("bp_check", "BP check", "health", "habit", 5), ("sugar_check", "Sugar check", "health", "habit", 5),
    ("weight_check", "Weight check", "health", "habit", 5), ("insulin", "Insulin", "health", "task", 5),
    ("eye_drops", "Eye drops", "health", "task", 5), ("inhaler", "Inhaler", "health", "task", 5),
    ("foot_care", "Foot care", "health", "habit", 10), ("wound_dressing", "Wound dressing", "health", "task", 15),
    ("dental_care", "Brushing and dental care", "health", "routine", 5), ("therapy_session", "Therapy session", "health", "habit", 60),
    ("dialysis", "Dialysis", "health", "habit", 240), ("doctor_visit", "Doctor visit", "health", "task", 60),
    ("lab_test", "Lab test", "health", "task", 30), ("hearing_aid", "Hearing aid charging", "health", "task", 5),
    ("water_intake", "Drinking water", "health", "habit", 0), ("sunlight", "Morning sunlight", "health", "habit", 15),
    # mind
    ("meditation", "Meditation", "mind", "habit", 15), ("journaling", "Journaling", "mind", "habit", 15), ("reading", "Reading", "mind", "habit", 30),
    ("puzzle", "Puzzles and brain games", "mind", "habit", 20), ("learning_language", "Learning a language", "mind", "habit", 20),
    ("music_practice", "Music practice", "mind", "habit", 30), ("gratitude", "Gratitude note", "mind", "habit", 5),
    ("screen_break", "Screen break", "mind", "habit", 10), ("breathing_break", "Breathing break", "mind", "habit", 5),
    ("news_reading", "Reading the newspaper", "mind", "routine", 30),
    # spiritual
    ("pooja", "Pooja", "spiritual", "habit", 20), ("prayer", "Prayer", "spiritual", "habit", 15), ("temple_visit", "Temple visit", "spiritual", "habit", 60),
    ("gurudwara_visit", "Gurudwara visit", "spiritual", "habit", 60), ("church", "Church", "spiritual", "habit", 90),
    ("namaz", "Namaz", "spiritual", "habit", 15), ("bhajan_kirtan", "Bhajan / kirtan", "spiritual", "habit", 60),
    ("satsang", "Satsang", "spiritual", "habit", 90), ("fasting", "Fasting", "spiritual", "habit", 0), ("path_recitation", "Path / recitation", "spiritual", "habit", 30),
    # food
    ("breakfast", "Breakfast", "food", "routine", 20), ("lunch", "Lunch", "food", "routine", 30), ("dinner", "Dinner", "food", "routine", 30),
    ("tea_time", "Evening tea", "food", "routine", 15), ("healthy_snack", "Healthy snack", "food", "habit", 10), ("fruit", "Fruit", "food", "habit", 5),
    ("protein_shake", "Protein shake", "food", "habit", 5), ("meal_prep", "Meal prep", "food", "habit", 45), ("cooking", "Cooking", "food", "routine", 60),
    # sleep
    ("wake_up", "Wake up", "sleep", "routine", 0), ("sleep", "Sleep", "sleep", "routine", 0), ("nap", "Afternoon nap", "sleep", "routine", 45),
    ("wind_down", "Wind down before bed", "sleep", "habit", 20),
    # self care
    ("skincare", "Skincare", "self_care", "habit", 10), ("oil_massage", "Oil massage", "self_care", "habit", 20), ("bath", "Bath", "self_care", "routine", 20),
    ("haircare", "Hair care", "self_care", "habit", 15), ("grooming", "Grooming", "self_care", "routine", 15), ("me_time", "Me time", "self_care", "habit", 30),
    # social
    ("call_family", "Call with family", "social", "habit", 20), ("call_friend", "Call a friend", "social", "habit", 20),
    ("video_call_grandchildren", "Video call with the grandchildren", "social", "habit", 20), ("visit_friends", "Visiting friends", "social", "habit", 90),
    ("community_meet", "Community meet", "social", "habit", 90), ("club", "Club", "social", "habit", 90), ("volunteer", "Volunteering", "social", "habit", 120),
    ("kitty_party", "Kitty party", "social", "habit", 120), ("park_bench_friends", "Park with friends", "social", "habit", 60),
    # work and study
    ("work", "Work", "work", "routine", 480), ("office_commute", "Office commute", "work", "routine", 45), ("study", "Study", "work", "habit", 90),
    ("class", "Class", "work", "habit", 60), ("meeting", "Meeting", "work", "task", 60), ("side_project", "Side project", "work", "habit", 60),
    # home
    ("cleaning", "Cleaning", "home", "task", 45), ("laundry", "Laundry", "home", "task", 30), ("gardening", "Gardening", "home", "habit", 30),
    ("plant_watering", "Watering the plants", "home", "habit", 10), ("pet_care", "Pet care", "home", "habit", 20), ("dog_walk", "Dog walk", "home", "habit", 30),
    ("grocery_shopping", "Grocery shopping", "home", "task", 60), ("home_repairs", "Home repairs", "home", "task", 60), ("bills", "Paying bills", "errand", "task", 15),
    # errands
    ("bank", "Bank visit", "errand", "task", 60), ("post_office", "Post office", "errand", "task", 45), ("pharmacy_visit", "Pharmacy visit", "errand", "task", 30),
    ("market", "Market", "errand", "task", 60), ("recharge", "Phone recharge", "errand", "task", 5), ("rent", "Paying rent", "errand", "task", 10),
    # caregiving
    ("check_on_parent", "Checking on a parent", "care", "habit", 15), ("accompany_to_doctor", "Taking someone to the doctor", "care", "task", 120),
    ("caregiver_break", "Caregiver's break", "care", "habit", 30), ("give_medicine_to", "Giving someone their medicine", "care", "task", 5),
    # quitting
    ("no_smoking", "Skipping cigarettes", "quit", "quit", 0), ("quit_tobacco", "Quitting tobacco / gutka", "quit", "quit", 0),
    ("no_alcohol", "No alcohol", "quit", "quit", 0), ("less_sugar", "Less sugar", "quit", "quit", 0),
    ("no_junk_food", "No junk food", "quit", "quit", 0), ("screen_time_limit", "Less screen time", "quit", "quit", 0),
    # hobbies
    ("painting", "Painting", "hobby", "habit", 45), ("knitting", "Knitting", "hobby", "habit", 45), ("sewing", "Sewing", "hobby", "habit", 45),
    ("crafts", "Crafts", "hobby", "habit", 45), ("photography", "Photography", "hobby", "habit", 45), ("music_listening", "Listening to music", "hobby", "routine", 30),
    ("playing_instrument", "Playing an instrument", "hobby", "habit", 30), ("chess", "Chess", "hobby", "habit", 45), ("cards", "Cards", "hobby", "habit", 60),
    ("tv_serial", "TV serial", "hobby", "routine", 30), ("movie", "Movie", "hobby", "routine", 150), ("writing", "Writing", "hobby", "habit", 30),
    ("custom", "Other activity", "hobby", "habit", 30),
]

CATALOG: dict[str, dict] = {r[0]: {"id": r[0], "label": r[1], "category": r[2], "kind": r[3], "minutes": r[4]} for r in _ROWS}
KINDS = ("habit", "quit", "task", "routine")
NUDGE_BEFORE = {"fitness": 30, "spiritual": 0, "social": 0}  # minutes before a habit: a light "leaving soon?" (0: none)


def catalog_line() -> str:
    """The catalogue for the tool description: ids grouped by category."""
    groups: dict[str, list[str]] = {}
    for a in CATALOG.values():
        groups.setdefault(a["category"], []).append(a["id"])
    return "; ".join(f"{c}: {', '.join(ids)}" for c, ids in groups.items())


def _hm(t: str | None) -> int | None:
    if not isinstance(t, str) or len(t) != 5 or t[2] != ":" or not (t[:2].isdigit() and t[3:].isdigit()):
        return None
    h, m = int(t[:2]), int(t[3:])
    return h * 60 + m if h < 24 and m < 60 else None


def _fmt(mins: int) -> str:
    mins %= 24 * 60
    return f"{mins // 60:02d}:{mins % 60:02d}"


def normalise(value: dict) -> dict:
    """A stored activity's value, filled in from the catalogue."""
    v = dict(value or {})
    cat = CATALOG.get(v.get("type") or "") or CATALOG["custom"]
    v["type"] = cat["id"]
    v.setdefault("label", cat["label"])
    v["category"] = v.get("category") or cat["category"]
    v["kind"] = v.get("kind") if v.get("kind") in KINDS else cat["kind"]
    days = v.get("days")
    v["days"] = sorted({int(d) for d in days if isinstance(d, int) or (isinstance(d, str) and d.isdigit()) if 0 <= int(d) <= 6}) if isinstance(days, list) else []
    start = _hm(v.get("start"))
    end = _hm(v.get("end"))
    minutes = int(v.get("minutes") or cat["minutes"] or 0)
    if start is not None and end is None and minutes:
        end = start + minutes
    v["start"] = _fmt(start) if start is not None else None
    v["end"] = _fmt(end) if end is not None else None
    if "nudge_before" not in v or v["nudge_before"] is None:
        v["nudge_before"] = NUDGE_BEFORE.get(v["category"], 0) if v["kind"] == "habit" else 0
    v["check_after"] = bool(v.get("check_after", v["kind"] in ("habit", "quit")))
    v.setdefault("paused", False)
    return v


def days_text(days: list[int]) -> str:
    if not days or len(days) == 7:
        return "every day"
    if days == [0, 1, 2, 3, 4]:
        return "Mon–Fri"
    if days == [0, 1, 2, 3, 4, 5]:
        return "Mon–Sat"
    return ", ".join(WEEK[d] for d in days)


def describe(v: dict) -> str:
    v = normalise(v)
    when = f"{v['start']}–{v['end']}" if v["start"] and v["end"] and v["end"] != v["start"] else (v["start"] or "no set time")
    return f"{v['label']} ({days_text(v['days'])}, {when})" + (" [paused]" if v.get("paused") else "")


def on_day(v: dict, day: datetime) -> bool:
    return not v.get("days") or day.weekday() in v["days"]


async def active(session: AsyncSession, family_id: str, subject_id: str) -> list[CareFact]:
    return [f for f in await store.facts(session, family_id, subject_id, domains=["activity"], statuses=("active",))]


def key_for(label_or_type: str) -> str:
    return f"activity:{slug(label_or_type)[:60]}"


# ── planning check-ins ─────────────────────────────────────────────────────────

async def _loop_exists(session: AsyncSession, family_id: str, dedupe: str) -> bool:
    return (await session.execute(select(OpenLoop.id).where(OpenLoop.family_id == family_id, OpenLoop.dedupe_key == dedupe).limit(1))).first() is not None


async def plan_for(session: AsyncSession, family_id: str, subject_id: str, fact: CareFact, *, now: datetime | None = None) -> int:
    """Open the check-ins of the next occurrences (today's still ahead, and tomorrow's). Once per occurrence and phase, even
    after it was answered or closed."""
    v = normalise(fact.value)
    if v.get("paused") or v["kind"] == "routine":
        return 0
    now = now or clock.now()
    local = clock.ist(now)
    made = 0
    for ahead in (0, 1):
        day = (local + timedelta(days=ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
        if not on_day(v, day):
            continue
        date = day.strftime("%Y-%m-%d")
        start, end = _hm(v["start"]), _hm(v["end"])
        phases: list[tuple[str, int]] = []
        if v["kind"] == "quit":
            phases.append(("after", _hm(v.get("check_at") or "") or 20 * 60))  # the evening, how the day went
        else:
            if start is None:
                continue
            if v["kind"] == "task":
                phases.append(("before", start))
            elif v["nudge_before"]:
                phases.append(("before", start - int(v["nudge_before"])))
            if v["check_after"] and v["kind"] == "habit":
                phases.append(("after", (end if end is not None else start + 30) + 15))
        for phase, at_min in phases:
            at = day + timedelta(minutes=at_min)
            if at <= local:
                continue
            # one per occurrence and phase of this version of the activity (a changed time or a resume plans afresh)
            dedupe = f"act:{subject_id}:{fact.key}:{str(fact.id)[:8]}:{date}:{phase}"
            if await _loop_exists(session, family_id, dedupe):
                continue
            title = f"{v['label']} {('starts at ' + v['start']) if phase == 'before' and v['start'] else 'check-in'} ({days_text(v['days'])})"
            await store.open_loop(
                session, family_id=family_id, subject_id=subject_id, kind="activity", title=title, owner_id=subject_id,
                wake_at=at.astimezone(now.tzinfo) if now.tzinfo else at, dedupe_key=dedupe,
                detail={"activity_key": fact.key, "phase": phase, "date": date, "max_wakes": 1 if phase == "before" else 2,
                        "rewake_minutes": 75, "expire_at": (day + timedelta(hours=23, minutes=30)).isoformat()},
            )
            made += 1
    return made


async def plan_all(sessions: async_sessionmaker, *, now: datetime | None = None) -> int:
    """The wake job: every active activity has its next check-ins planned."""
    made = 0
    async with sessions() as session:
        rows = list((await session.execute(
            select(CareFact).where(CareFact.domain == "activity", CareFact.status == "active", ~CareFact.family_id.startswith("shadow:"))
        )).scalars())
        for f in rows:
            made += await plan_for(session, f.family_id, f.subject_id, f, now=now)
        await session.commit()
    return made


async def drop_future_loops(session: AsyncSession, family_id: str, key: str) -> int:
    """A changed, paused or stopped activity: its planned check-ins go (new ones are planned from the new schedule)."""
    rows = list((await session.execute(
        select(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.kind == "activity", OpenLoop.status == "open")
        .where(OpenLoop.detail["activity_key"].astext == key)
    )).scalars())
    for lp in rows:
        lp.status, lp.closed_note, lp.updated_at = "cancelled", "activity changed", clock.now()
    return len(rows)


# ── what happened ──────────────────────────────────────────────────────────────

async def logs(session: AsyncSession, family_id: str, subject_id: str, *, days: int = 30, key: str | None = None) -> list[CareEvent]:
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id,
                                CareEvent.kind.in_(("activity_done", "activity_skipped")),
                                CareEvent.at >= clock.now() - timedelta(days=days))
    rows = list((await session.execute(q.order_by(CareEvent.at))).scalars())
    return [e for e in rows if not key or (e.payload or {}).get("key") == key]


def streak(v: dict, evs: list[CareEvent], today: datetime) -> int:
    """Scheduled days in a row (back from today, or yesterday if today's is still ahead) with it done."""
    v = normalise(v)
    done_days = {e.day for e in evs if e.kind == "activity_done"}
    n, day = 0, today
    for _ in range(60):
        if on_day(v, day):
            d = day.strftime("%Y-%m-%d")
            if d in done_days:
                n += 1
            elif day.date() != today.date():
                break
        day -= timedelta(days=1)
    return n


async def status(session: AsyncSession, family_id: str, subject_id: str) -> list[dict]:
    """Each activity with today's state, streak and the last thing they said about it (for the brain and the dashboard)."""
    now = clock.ist()
    today = now.strftime("%Y-%m-%d")
    out = []
    for f in await active(session, family_id, subject_id):
        v = normalise(f.value)
        evs = await logs(session, family_id, subject_id, days=60, key=f.key)
        todays = [e for e in evs if e.day == today]
        if todays:
            state = "done" if todays[-1].kind == "activity_done" else "skipped"
        elif not on_day(v, now) or v["kind"] == "routine":
            state = "off" if not on_day(v, now) else "routine"
        else:
            end = _hm(v["end"]) or _hm(v["start"])
            state = "upcoming" if end is None or now.hour * 60 + now.minute < end else "not_heard"
        week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        last = evs[-1] if evs else None
        out.append({
            "key": f.key, "label": v["label"], "type": v["type"], "category": v["category"], "kind": v["kind"], "days": v["days"],
            "start": v["start"], "end": v["end"], "paused": v.get("paused", False), "goal": v.get("goal"), "today": state,
            "streak": streak(v, evs, now), "week_done": sum(1 for e in evs if e.kind == "activity_done" and e.day >= week_ago),
            "last": {"at": last.at.isoformat(), "summary": last.summary, "done": last.kind == "activity_done"} if last else None,
            "text": describe(v),
        })
    return out


def context_block(name: str, rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = [f"  - {r['text']}: today {r['today'].replace('_', ' ')}"
             + (f", streak {r['streak']}" if r["streak"] > 1 else "")
             + (f"; last: {r['last']['summary'][:90]}" if r.get("last") else "") + f" [{r['key']}]" for r in rows]
    return (f"ACTIVITIES OF {name.upper()} (habits are not reminders: check in on them and log what they tell you with "
            "log_activity; routine ones just shape when to message):\n" + "\n".join(lines))


# ── adopting what is already known ─────────────────────────────────────────────

ADOPT_PROMPT = """You turn what a person's care record already says about their routine into activities.
Each line below is a routine fact (text and data). Return the activities it describes, using ONLY these type ids:
{catalog}
For each: type (custom with a label when none fits), label (their words), kind (habit for things to check in on like the gym
or a walk; routine for wake-up, meals, sleep; quit for giving something up; task for a chore at a time), days (0=Mon..6=Sun,
[] for every day), start and end as HH:MM (24h) when the record says them. Only what the record says; no guesses.
Return only JSON: {{"activities": [{{"type": "...", "label": "...", "kind": "...", "days": [], "start": "HH:MM", "end": "HH:MM or null"}}]}}"""


async def adopt(session: AsyncSession, host, family_id: str, subject_id: str) -> list[str]:
    """Existing routine facts with times become activities (founder 2026-10-13: a gym schedule told to Saheli was never
    scheduled). Skips any activity already saved; a habit whose start already has the person's own reminder gets no extra
    nudge before (the check-in after still runs)."""
    import json as _json
    import re as _re

    from app.llm import router

    facts = [f for f in await store.facts(session, family_id, subject_id, domains=["routine"], statuses=("active",))]
    if not facts:
        return []
    lines = "\n".join(f"- {f.text} | {_json.dumps(f.value, ensure_ascii=False)}" for f in facts)
    try:
        reply = await router.complete("extract", system_stable=ADOPT_PROMPT.format(catalog=catalog_line()),
                                      messages=[{"role": "user", "content": [{"type": "text", "text": lines}]}], max_tokens=1500, effort="low")
    except Exception:  # noqa: BLE001
        return []
    m = _re.search(r"\{.*\}", reply.text or "", _re.S)
    try:
        items = (_json.loads(m.group(0)) if m else {}).get("activities") or []
    except _json.JSONDecodeError:
        return []
    reminder_times: list[int] = []
    try:
        listed = await host.call("list_reminders", {"status": "active"}, family_id=family_id, subject_id=subject_id, actor_id=subject_id)
        for r in (listed or {}).get("reminders") or []:
            for t in ((r.get("schedule") or {}).get("times") or []):
                if _hm(t) is not None:
                    reminder_times.append(_hm(t))
    except Exception:  # noqa: BLE001 — no reminders known: the usual nudge
        pass
    made = []
    existing = {f.key for f in await active(session, family_id, subject_id)}
    for it in items[:12]:
        if not isinstance(it, dict):
            continue
        typ = it.get("type") if it.get("type") in CATALOG else "custom"
        label = (it.get("label") or CATALOG[typ]["label"]).strip()
        key = key_for(label if typ == "custom" else typ)
        if key in existing:
            continue
        value = normalise({"type": typ, "label": label, "kind": it.get("kind"), "days": it.get("days") or [], "start": it.get("start"),
                           "end": it.get("end") if isinstance(it.get("end"), str) else None})
        if value["kind"] in ("habit", "task") and not value["start"]:
            continue
        start = _hm(value["start"])
        if value["kind"] == "habit" and start is not None and any(start - 120 <= t <= start for t in reminder_times):
            value["nudge_before"] = 0  # their own reminder already nudges before it
        w = await store.write_fact(session, family_id=family_id, subject_id=subject_id, domain="activity", key=key, value=value,
                                   text=describe(value), source_kind="inferred", note="from the routine they told Saheli", replace=True)
        await plan_for(session, family_id, subject_id, w.fact)
        existing.add(key)
        made.append(key)
    return made
