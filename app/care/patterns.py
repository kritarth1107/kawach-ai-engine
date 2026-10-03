"""Patterns nobody asked about: Saheli notices trends in what was logged and said.

Plain counting over the last two weeks (no model calls, so it is free and cannot invent a pattern):

- a dose slot that keeps getting missed or not confirmed ("evening Metformin: 5 of the last 8")
- misses that bunch up on one weekday ("Sundays")
- doses taken much later than their time
- blood pressure, sugar or weight drifting up or down against the week before
- readings repeatedly above the usual safe range
- the same complaint coming back ("knee pain, 4 times in 10 days")
- low moods piling up
- going quiet: far fewer messages than usual
- awake and writing in the middle of the night
- meals skipped

Each finding says what was seen, with the evidence, and one gentle suggestion. Nothing here diagnoses or
alerts: patterns go to the brain's context (to mention once to a caregiver), the dashboard and the weekly
check-in. Urgent single readings are handled by the red-flag rules, not here.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from statistics import mean, median

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import slug
from app.care.models import CareEvent, Turn
from app.core import clock

logger = logging.getLogger(__name__)

WINDOW_DAYS = 14
WEEKDAYS = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"]
SLOT_NAMES = [(5, "morning"), (11, "afternoon"), (16, "evening"), (20, "night")]
MISSED = ("dose_missed", "dose_skipped", "dose_refused")
SLOT_MATCH = timedelta(hours=4)

SYMPTOM_TERMS = {
    "knee pain": ["knee", "ghutn", "ghutna", "ghutne", "घुटन"],
    "back pain": ["back pain", "kamar", "कमर"],
    "headache": ["headache", "head ache", "sir dard", "sar dard", "sir mein dard", "सिर"],
    "dizziness": ["dizzy", "dizziness", "chakkar", "चक्कर", "vertigo"],
    "breathlessness": ["breath", "saans", "सांस"],
    "chest discomfort": ["chest", "seene", "सीने"],
    "swelling": ["swell", "sujan", "soojan", "सूजन", "puffy", "puffiness"],
    "cough": ["cough", "khansi", "खांसी"],
    "stomach trouble": ["stomach", "pet dard", "acidity", "gas", "constipat", "kabz", "loose motion", "diarrh", "पेट"],
    "tiredness": ["tired", "weak", "thakan", "kamzori", "थकान", "कमजोरी"],
    "poor sleep": ["can't sleep", "cannot sleep", "no sleep", "neend nahi", "sleepless", "insomnia", "नींद नहीं"],
    "burning feet": ["burning", "jalan", "जलन", "tingling", "jhunjhuni"],
    "joint pain": ["joint", "jodon", "jodo"],
    "nausea": ["nausea", "vomit", "ulti", "ji machal"],
}
SKIPPED_MEAL = re.compile(r"\b(skip\w*|missed|did not eat|didn'?t eat|not eat\w*|nahi khaya|nahin khaya|khana nahi|no (?:lunch|dinner|breakfast)|"
                          r"no appetite|bhookh nahi|bhook nahi)\b|नहीं खाया|भूख नहीं", re.I)
LOW_MOOD = re.compile(r"\b(sad|low|lonely|alone|akela|akeli|udaas|upset|worried|anxious|tension|cry\w*|ro rah\w*|depress\w*|hopeless|bored|"
                      r"irritat\w*|gussa|angry|dukhi|pareshan|ghabra\w*)\b|उदास|अकेल|दुखी|परेशान|घबरा", re.I)


@dataclass
class Pattern:
    kind: str
    key: str  # stable: the same pattern next week has the same key
    title: str  # one line a caregiver reads
    detail: str
    suggestion: str
    severity: str = "info"  # info | watch
    evidence: list[str] = field(default_factory=list)
    subject_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def slot_name(minutes: int) -> str:
    h = minutes // 60
    name = "night"
    for start, label in SLOT_NAMES:
        if h >= start:
            name = label
    return "night" if h < 5 else name


def _minutes(hhmm: str) -> int | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(hhmm).strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m and int(m.group(1)) < 24 else None


def _first(name: str) -> str:
    return (slug(name or "").split("_") or [""])[0]


def _num(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]


def _fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


# ── medicines ──────────────────────────────────────────────────────────────────


def dose_patterns(meds: list[dict], events: list, *, now, subject: str, days: int = WINDOW_DAYS) -> list[Pattern]:
    """meds: [{"name", "times": ["HH:MM"], "days": [0-6] or None, "since": datetime}]. events: dose_* CareEvents."""
    out: list[Pattern] = []
    local_now = clock.ist(now)
    by_med: dict[str, list] = defaultdict(list)
    for e in events:
        by_med[_first((e.payload or {}).get("medicine") or e.summary.split(":")[0])].append(e)
    for med in meds:
        first = _first(med["name"])
        times = [t for t in (_minutes(x) for x in med.get("times") or []) if t is not None]
        if not first or not times:
            continue
        evs = by_med.get(first, [])
        for slot in times:
            seen_days, missed_days, late = [], [], []
            for back in range(days, 0, -1):  # whole past days only
                day = (local_now - timedelta(days=back)).date()
                if med.get("since") and day < clock.ist(med["since"]).date():
                    continue
                if med.get("days") and day.weekday() not in med["days"]:
                    continue
                due = clock.ist(now).replace(year=day.year, month=day.month, day=day.day, hour=slot // 60, minute=slot % 60, second=0, microsecond=0)
                near = [e for e in evs if abs(clock.ist(e.at) - due) <= SLOT_MATCH]
                # Assign each event to its closest slot only.
                near = [e for e in near if min(times, key=lambda t: abs((clock.ist(e.at) - due.replace(hour=t // 60, minute=t % 60)).total_seconds())) == slot]
                seen_days.append(day)
                taken = [e for e in near if e.kind == "dose_taken"]
                if taken:
                    delay = (clock.ist(taken[0].at) - due).total_seconds() / 60
                    late.append(delay)
                elif any(e.kind in MISSED for e in near) or not near:
                    missed_days.append(day)
            n = len(seen_days)
            if n < 5:
                continue
            label = f"{med['name']} ({slot_name(slot)}, {_fmt(slot)})"
            k = f"dose:{first}:{_fmt(slot)}"
            if len(missed_days) >= 3 and len(missed_days) / n >= 0.35:
                out.append(Pattern(
                    "dose_slot", k, f"{label} is often missed or not confirmed: {len(missed_days)} of the last {n} days",
                    f"Missed or not confirmed on {', '.join(d.strftime('%d %b') for d in missed_days[-5:])}.",
                    "Ask what gets in the way at that time; a reminder at a better moment or someone checking in may help.",
                    "watch" if len(missed_days) / n >= 0.5 else "info", [d.isoformat() for d in missed_days], subject))
            # Misses on one weekday every week, while the other days do clearly better.
            wd = Counter(d.weekday() for d in missed_days)
            for day_, cnt in wd.items():
                same = sum(1 for d in seen_days if d.weekday() == day_)
                others = n - same
                other_rate = (len(missed_days) - cnt) / others if others else 1.0
                if cnt >= 2 and cnt == same and other_rate <= 0.5:
                    out.append(Pattern(
                        "dose_weekday", f"{k}:wd{day_}", f"{label} slips on {WEEKDAYS[day_]}: missed every {WEEKDAYS[day_][:-1]} in the last two weeks",
                        f"Other days: {len(missed_days) - cnt} of {others} missed.",
                        f"Something about {WEEKDAYS[day_]} (visits, outings, a different routine) may be the reason; worth asking.",
                        "info", [d.isoformat() for d in missed_days if d.weekday() == day_], subject))
            if len(late) >= 4 and median(late) >= 75:
                out.append(Pattern(
                    "dose_late", f"{k}:late", f"{label} is usually taken late: about {int(median(late))} minutes after its time",
                    f"Over the last {len(late)} doses that were confirmed.", "Ask whether a later reminder time would suit them better.",
                    "info", [], subject))
    return out


# ── vitals ─────────────────────────────────────────────────────────────────────


def vital_patterns(vitals: list, *, now, subject: str) -> list[Pattern]:
    out: list[Pattern] = []
    recent_cut, base_cut = now - timedelta(days=7), now - timedelta(days=21)
    series: dict[str, list[tuple]] = defaultdict(list)  # kind -> [(at, values)]
    for e in vitals:
        p = e.payload or {}
        kind = str(p.get("kind") or e.summary.split(" ")[0]).lower()
        vals = _num(str(p.get("value") or e.summary))
        if vals:
            series[kind].append((e.at, vals))
    for kind, pts in series.items():
        recent = [v for at, v in pts if at >= recent_cut]
        base = [v for at, v in pts if base_cut <= at < recent_cut]
        if kind == "bp":
            recent, base = [v for v in recent if len(v) >= 2], [v for v in base if len(v) >= 2]
            high = [v for v in recent if v[0] >= 140 or v[1] >= 90]
            if len(recent) >= 3 and len(high) >= 3 and len(high) / len(recent) >= 0.6:
                out.append(Pattern(
                    "vital_high", "vital:bp:high", f"Blood pressure has been high on {len(high)} of the last {len(recent)} readings this week",
                    "Readings: " + ", ".join(f"{int(a)}/{int(b)}" for a, b, *_ in high[-5:]) + ".",
                    "Worth sharing these readings with the doctor; the weekly report has them.", "watch", [], subject))
            if len(recent) >= 3 and len(base) >= 3:
                d = mean(v[0] for v in recent) - mean(v[0] for v in base)
                if abs(d) >= 10:
                    out.append(Pattern(
                        "vital_trend", f"vital:bp:{'up' if d > 0 else 'down'}",
                        f"Blood pressure is {'higher' if d > 0 else 'lower'} than the week before: average top number {mean(v[0] for v in recent):.0f} vs {mean(v[0] for v in base):.0f}",
                        f"{len(recent)} readings this week, {len(base)} before.",
                        "Check whether anything changed (salt, sleep, a missed tablet) and mention it to the doctor if it continues.",
                        "watch" if d > 0 else "info", [], subject))
        elif kind in ("sugar", "glucose"):
            r, b = [v[0] for v in recent], [v[0] for v in base]
            if len(r) >= 3 and len(b) >= 3 and abs(mean(r) - mean(b)) >= 20:
                up = mean(r) > mean(b)
                out.append(Pattern(
                    "vital_trend", f"vital:sugar:{'up' if up else 'down'}",
                    f"Sugar readings are {'higher' if up else 'lower'} than the week before: average {mean(r):.0f} vs {mean(b):.0f}",
                    f"{len(r)} readings this week, {len(b)} before.", "Note any change in food, activity or tablets, and share it with the doctor.",
                    "watch" if up else "info", [], subject))
        elif kind == "weight":
            r, b = [v[0] for v in recent], [v[0] for v in base]
            if r and b and abs(mean(r) - mean(b)) >= 2:
                up = mean(r) > mean(b)
                out.append(Pattern(
                    "vital_trend", f"vital:weight:{'up' if up else 'down'}",
                    f"Weight has gone {'up' if up else 'down'} by about {abs(mean(r) - mean(b)):.1f} kg in two weeks",
                    "", "A change this quick is worth telling the doctor, especially with heart or kidney conditions.", "watch", [], subject))
    return out


# ── life: symptoms, mood, meals, talking, sleep ────────────────────────────────


def symptom_patterns(events: list, *, subject: str) -> list[Pattern]:
    out: list[Pattern] = []
    hits: dict[str, list] = defaultdict(list)
    for e in events:
        text = (e.summary or "").lower()
        for name, words_ in SYMPTOM_TERMS.items():
            if any(w in text for w in words_):
                hits[name].append(e)
    for name, es in hits.items():
        days_ = sorted({e.day for e in es})
        if len(days_) >= 3:
            out.append(Pattern(
                "symptom_repeat", f"symptom:{slug(name)}", f"{name.capitalize()} keeps coming back: on {len(days_)} days in the last two weeks",
                "Said: " + "; ".join(f"{e.day[5:]} {e.summary[:60]}" for e in es[-3:]) + ".",
                "Add it to the questions for the next doctor visit (add_doctor_question).", "watch" if len(days_) >= 4 else "info",
                days_, subject))
    return out


def mood_patterns(events: list, *, now, subject: str) -> list[Pattern]:
    week = [e for e in events if e.at >= now - timedelta(days=7)]
    low = [e for e in week if (e.payload or {}).get("severity") in ("watch", "concern") or LOW_MOOD.search(e.summary or "")]
    days_ = sorted({e.day for e in low})
    if len(days_) >= 3:
        return [Pattern(
            "mood_low", "mood:low", f"Low moods on {len(days_)} days this week",
            "Said: " + "; ".join(f"{e.day[5:]} {e.summary[:60]}" for e in low[-3:]) + ".",
            "A call from family, a visit or a favourite activity may help; tell the doctor if it goes on.", "watch", days_, subject)]
    return []


def meal_patterns(events: list, *, now, subject: str) -> list[Pattern]:
    week = [e for e in events if e.at >= now - timedelta(days=7) and SKIPPED_MEAL.search(e.summary or "")]
    days_ = sorted({e.day for e in week})
    if len(days_) >= 3:
        return [Pattern(
            "meals_skipped", "meals:skipped", f"Meals skipped or little appetite on {len(days_)} days this week",
            "Said: " + "; ".join(f"{e.day[5:]} {e.summary[:60]}" for e in week[-3:]) + ".",
            "Poor appetite can matter with diabetes and BP tablets; worth mentioning to the doctor.", "watch", days_, subject)]
    return []


def talk_patterns(turn_times: list, *, now, subject: str) -> list[Pattern]:
    out: list[Pattern] = []
    local = [clock.ist(t) for t in turn_times if t >= now - timedelta(days=WINDOW_DAYS + 3)]
    per_day = Counter(t.date() for t in local)
    today = clock.ist(now).date()
    base = [per_day.get(today - timedelta(days=b), 0) for b in range(4, WINDOW_DAYS + 4)]
    recent = [per_day.get(today - timedelta(days=b), 0) for b in range(1, 4)]
    usual = median(base) if base else 0
    if usual >= 3 and sum(recent) <= 0.3 * usual * 3:
        out.append(Pattern(
            "quiet", "talk:quiet", f"Much quieter than usual: {sum(recent)} messages in the last 3 days (usually about {usual:.0f} a day)",
            "", "A call to check how they are may be worth it.", "watch", [], subject))
    nights = sorted({t.date().isoformat() for t in local if t >= clock.ist(now) - timedelta(days=7) and (t.hour < 4 or (t.hour == 4 and t.minute < 30))})
    if len(nights) >= 3:
        out.append(Pattern(
            "night_awake", "talk:night", f"Awake and writing late at night on {len(nights)} nights this week",
            "Messages between midnight and 4:30 am.", "Poor sleep is worth asking about gently, and mentioning to the doctor.", "info", nights, subject))
    return out


# ── all together ───────────────────────────────────────────────────────────────


async def find(session: AsyncSession, family_id: str, subject_id: str, *, days: int = WINDOW_DAYS) -> list[Pattern]:
    now = clock.now()
    since = now - timedelta(days=days + 7)
    meds = []
    for f in await store.facts(session, family_id, subject_id, domains=["medicine"], statuses=("active",)):
        v = f.value or {}
        meds.append({"name": str(v.get("name") or f.key.split(":", 1)[1]), "times": _as_list(v.get("times")), "days": _weekdays(v.get("days")),
                     "since": getattr(f, "valid_from", None)})
    evs = await store.events(session, family_id, subject_id, since=since, limit=20000)
    doses = [e for e in evs if e.kind.startswith("dose_")]
    vitals = [e for e in evs if e.kind == "vital"]
    life = [e for e in evs if e.kind in ("symptom", "mood", "meal", "sleep", "other") and e.at >= now - timedelta(days=days)]
    turn_times = list((await session.execute(
        select(Turn.at).where(Turn.family_id == family_id, Turn.thread_id == subject_id, Turn.role == "user", Turn.at >= since)
    )).scalars())
    found: list[Pattern] = []
    detectors = [
        lambda: dose_patterns(meds, doses, now=now, subject=subject_id, days=days),
        lambda: vital_patterns(vitals, now=now, subject=subject_id),
        lambda: symptom_patterns([e for e in life if e.kind in ("symptom", "sleep", "other")], subject=subject_id),
        lambda: mood_patterns([e for e in life if e.kind == "mood"], now=now, subject=subject_id),
        lambda: meal_patterns([e for e in life if e.kind == "meal"], now=now, subject=subject_id),
        lambda: talk_patterns(turn_times, now=now, subject=subject_id),
    ]
    for run in detectors:
        try:
            found += run()
        except Exception:  # noqa: BLE001 — one odd record must not hide every other pattern
            logger.exception("pattern detector failed family=%s subject=%s", family_id, subject_id)
    return sorted(found, key=lambda p: (p.severity != "watch", p.kind))


def _as_list(v) -> list:
    if v is None or v == "":
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _weekdays(v) -> list[int] | None:
    """Days a medicine is taken (0 = Monday), from however it was saved; None means every day."""
    days_ = []
    for x in _as_list(v):
        if isinstance(x, int) and 0 <= x <= 6:
            days_.append(x)
        elif isinstance(x, str):
            x = x.strip().lower()[:3]
            names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            if x in names:
                days_.append(names.index(x))
            elif x.isdigit() and 0 <= int(x) <= 6:
                days_.append(int(x))
    return days_ or None


async def record_new(session: AsyncSession, family_id: str, subject_id: str) -> list[Pattern]:
    """Find patterns and record each once per week (care_events kind 'pattern'). Returns the ones new this week."""
    week = clock.ist().strftime("%G-W%V")
    new = []
    for p in await find(session, family_id, subject_id):
        fresh = await store.record_event(
            session, family_id=family_id, subject_id=subject_id, kind="pattern", summary=p.title,
            payload=p.to_dict(), ref=f"{p.key}:{week}",
        )
        if fresh:
            new.append(p)
    return new


async def recent(session: AsyncSession, family_id: str, subject_ids: list[str], *, days: int = 7) -> list[dict]:
    """Patterns recorded in the last few days, newest first, for the brain's context and the dashboard."""
    rows = list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id.in_(subject_ids), CareEvent.kind == "pattern",
                                CareEvent.at >= clock.now() - timedelta(days=days)).order_by(CareEvent.at.desc())
    )).scalars())
    from app.care import outcomes

    votes = await outcomes.votes(session, family_id)
    seen, out = set(), []
    for r in rows:
        k = (r.subject_id, (r.payload or {}).get("key"))
        if k in seen:
            continue
        seen.add(k)
        target = feedback_key(r.payload or {})
        v = votes.get(target, {})
        if v.get("down", 0) > v.get("up", 0):
            continue  # the family said this kind of note is not useful: stop raising it
        out.append({**(r.payload or {}), "target": target, "noticed": r.at.isoformat(), "new": r.at >= clock.now() - timedelta(days=2)})
    return out


def feedback_key(p: dict) -> str:
    return f"pattern:{p.get('key')}:subj={p.get('subject_id')}"


def context_block(items: list[dict], names: dict[str, str]) -> str:
    if not items:
        return ""
    lines = ["PATTERNS NOTICED (from the last two weeks of logs; facts, not diagnoses):"]
    for p in items[:6]:
        tag = "NEW, not mentioned yet" if p.get("new") else "known"
        lines.append(f"  - [{tag}] {names.get(p.get('subject_id'), 'them')}: {p['title']}. {p.get('detail', '')} Suggest: {p['suggestion']} "
                     f"(feedback key {p.get('target') or feedback_key(p)})")
    lines.append("  Mention a NEW pattern once to a caregiver, in one or two lines, when it fits the conversation, then call "
                 "offer_buttons kind feedback with its feedback key so they can tap 👍/👎. To the elder only as a gentle offer (for "
                 "example a better reminder time). Never present it as a diagnosis or an emergency.")
    return "\n".join(lines)
