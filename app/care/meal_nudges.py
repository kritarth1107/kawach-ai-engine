"""Habit asks (nutrition plan §N5, founder §N18, and 15 Oct 2026: meals, water and sleep for every user). After a meal,
a glass of water, or the night's sleep that nobody has logged, the hour's regular nudge may ask. One habit ask per quiet
hour, at most six a day. Meal times come from their own routine, then a time learned from about seven days of their own
logs, then a default moved to their day. The same for water and for when to ask about sleep. It never comes within 30
minutes of a medicine question, never for a meal a delivered order covers, and never while a food question is already
open. For an elder who does not write to Saheli, their caregiver still gets one evening ask when nothing of their own
is due (founder F1).

Meal and water answers go to `log_food`; sleep goes to `log_event` (reply_hint). Until the nutrition agent exists, no
habit ask is offered. The admin flag nudge.meals decides off | shadow | on.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import NudgeLedger, Turn
from app.core import clock

MAIN = ("breakfast", "lunch", "dinner")
SLOTS = (*MAIN, "snack")
DEFAULT_AT = {"lunch": 13 * 60 + 30, "snack": 16 * 60 + 30, "dinner": 20 * 60 + 30}  # breakfast: an hour after they wake
ASK_AFTER, ASK_UNTIL = 45, 180  # minutes after the meal's time
PER_DAY, GAP = 6, timedelta(hours=1)  # meals, water and sleep share one ask per quiet hour
HABIT_KINDS = ("meal", "water", "sleep")
WATER_AT = (11 * 60, 16 * 60, 21 * 60)
WATER_FOR = 90  # minutes the water ask stays open
SLEEP_AFTER, SLEEP_FOR = 30, 60  # from half an hour after waking, for an hour (before breakfast is asked)
WATER_KEY = "learned:nudges_water_times"
SLEEP_KEY = "learned:nudges_sleep_ask"
STABLE_DAYS, STABLE_WITHIN = 7, 45
DOSE_GAP = timedelta(minutes=30)
RECAP_BEFORE_BED, RECAP_WINDOW = 90, 120  # minutes
TO_RECAP, BACK_TO_MEALS, TO_PAUSE = 3, 3, 5  # unanswered per-meal asks → recap; answered recaps → per meal; unanswered recaps → pause
QUIET_ELDER = timedelta(days=30)  # an elder with no message of their own this long does not write to Saheli
MODE_KEY = "learned:nudges_meals_mode"
TIMES_KEY = "learned:nudges_meals_times"


def hooks():
    """The nutrition agent's nudge hooks and settings, or None while it is not there yet."""
    import importlib

    try:
        pkg = importlib.import_module("app.nutrition")
        return getattr(pkg, "nudge_hooks", None) or importlib.import_module("app.nutrition.nudge_hooks"), \
            getattr(pkg, "settings", None) or importlib.import_module("app.nutrition.settings")
    except ImportError:
        return None


def _hm(v) -> int | None:
    try:
        h, m = str(v).strip()[:5].split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _clock(mins: int) -> str:
    h, m = divmod(mins % (24 * 60), 60)
    return f"{h % 12 or 12}{f':{m:02d}' if m else ''} {'pm' if h >= 12 else 'am'}"


async def _wake(session: AsyncSession, family_id: str, person: str) -> int:
    from app.brain.wake import own_night

    night = await own_night(session, family_id, person)
    return night[1] if night else 8 * 60


async def times(session: AsyncSession, family_id: str, person: str, acts: list[dict]) -> dict[str, int]:
    """Each meal's usual time (minutes of the day): their routine, then the learned time (about seven days of logs),
    then a default moved to their own day. A meal before they are half an hour up is not theirs."""
    out: dict[str, int] = {}
    for a in acts:
        if a.get("type") in SLOTS and _hm(a.get("start")) is not None:
            out[a["type"]] = _hm(a["start"])  # type: ignore[assignment]
    learned = await store.active_fact(session, family_id, person, TIMES_KEY)
    for slot in SLOTS:
        t = _hm(((learned.value or {}) if learned else {}).get(slot))
        if slot not in out and t is not None:
            out[slot] = t
    hk = hooks()
    if hk and hasattr(hk[0], "usual_meal_times"):
        try:
            usual = await hk[0].usual_meal_times(session, family_id, person)
        except Exception:  # noqa: BLE001 — a missing figure falls through to the default
            usual = None
        for slot in SLOTS:
            t = _hm((usual or {}).get(slot))
            if slot not in out and (usual or {}).get("stable", {}).get(slot) and t is not None:
                out[slot] = t
    wake = await _wake(session, family_id, person)
    out.setdefault("breakfast", wake + 60)
    for slot, t in DEFAULT_AT.items():
        out.setdefault(slot, t)
    return {s: t for s, t in out.items() if t >= wake + 30}


async def water_times(session: AsyncSession, family_id: str, person: str) -> list[int]:
    """When to ask about water: times learned from their own logs, else the defaults that fall after they are up."""
    learned = await store.active_fact(session, family_id, person, WATER_KEY)
    wake = await _wake(session, family_id, person)
    got = [t for t in (_hm(v) for v in (((learned.value or {}) if learned else {}).get("times") or [])) if t is not None]
    times_ = got or list(WATER_AT)
    return [t for t in times_ if t >= wake + 30]


async def sleep_at(session: AsyncSession, family_id: str, person: str) -> int:
    """When to ask about last night: the learned time, else half an hour after they usually wake."""
    learned = await store.active_fact(session, family_id, person, SLEEP_KEY)
    wake = await _wake(session, family_id, person)
    t = _hm(((learned.value or {}) if learned else {}).get("at"))
    return t if t is not None and t >= wake else wake + SLEEP_AFTER


async def mode(session: AsyncSession, family_id: str, person: str, setting: dict) -> str:
    """per_meal | recap | paused: their own setting first, else what Saheli learned (a visible fact), else per meal."""
    ask = str(setting.get("ask") or "adaptive")
    if ask in ("per_meal", "recap"):
        return ask
    f = await store.active_fact(session, family_id, person, MODE_KEY)
    got = str(((f.value or {}) if f else {}).get("mode") or "per_meal")
    return got if got in ("per_meal", "recap", "paused") else "per_meal"


async def recap_at(session: AsyncSession, family_id: str, person: str) -> int:
    """An hour and a half before their bedtime (a bedtime after 11 pm counts as 11 pm)."""
    from app.brain.wake import own_night

    night = await own_night(session, family_id, person)
    bed = night[0] if night and night[0] >= 18 * 60 else 23 * 60
    return min(bed, 23 * 60) - RECAP_BEFORE_BED


async def silent_elder(session: AsyncSession, family_id: str, elder_id: str, now: datetime) -> bool:
    last = await session.scalar(select(func.max(Turn.at)).where(Turn.family_id == family_id, Turn.thread_id == elder_id, Turn.role == "user"))
    return last is None or now - last >= QUIET_ELDER


async def _water_today(nh, session: AsyncSession, family_id: str, person: str, day: str) -> bool:
    prog = await nh.progress(session, family_id, person, day=day)
    return float((prog.get("totals") or {}).get("water_ml") or 0) > 0


async def _sleep_today(session: AsyncSession, family_id: str, person: str, day: str) -> bool:
    for e in await store.events(session, family_id, person, day=day, kinds=["sleep", "activity_done", "routine"]):
        payload = e.payload or {}
        key = str(payload.get("key") or "")
        if e.kind == "sleep" or payload.get("type") == "sleep" or "sleep" in key:
            return True
    return False


async def _own_habit(session: AsyncSession, p, sent: list[NudgeLedger], now: datetime, acts: list[dict], nh, settings,
                     today: str, m_now: int, local: datetime) -> dict | None:
    """The one habit question this person is due: a meal (including their own, for a caregiver), else water, else sleep."""
    fid, pid = p.family_id, p.id
    s = await settings.get(session, fid, pid)
    meals_off = s.get("tracking") == "off" or s.get("ask") == "off"
    how = "paused" if meals_off else await mode(session, fid, pid, s)
    pending = [] if meals_off else await nh.pending_confirmations(session, fid, pid)
    open_food = any(c.get("kind") == "question" for c in pending)
    if how == "recap" and not open_food:
        at = await recap_at(session, fid, pid)
        meal_asked = [r for r in sent if r.kind == "meal" and clock.ist_day(r.sent_at or r.planned_at) == today]
        if not meal_asked and at <= m_now <= at + RECAP_WINDOW:
            meal_at = await times(session, fid, pid, acts)
            missing = [slot for slot, t in meal_at.items() if t < m_now and not await nh.meal_logged(session, fid, pid, slot, day=today)]
            if missing:
                return {"kind": "meal", "about": f"meal:{today}:recap", "fld": "meal:day",
                        "why": f"a short recap of what they ate today ({', '.join(missing)} not told yet): one easy question, nothing about medicines"}
    if how == "per_meal" and not open_food:
        meal_at = await times(session, fid, pid, acts)
        ordered = [t for t in ((c.get("since") and _since(c["since"])) for c in pending if c.get("kind") == "order_meal") if t]
        due = []
        for slot in SLOTS:
            t = meal_at.get(slot)
            if t is None or not t + ASK_AFTER <= m_now <= t + ASK_UNTIL:
                continue
            if any(r.about_ref == f"meal:{today}:{slot}" for r in sent):
                continue
            start = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=t - 120)
            if any(o >= start for o in ordered):
                continue  # a delivered meal from an order: aftercare asks "did you eat it?"
            if await nh.meal_logged(session, fid, pid, slot, day=today):
                continue
            due.append({"kind": "meal", "about": f"meal:{today}:{slot}", "fld": f"meal:{slot}",
                        "why": f"what they had for {slot} (usually around {_clock(t)}), nothing logged yet: one easy question about what "
                               "and roughly how much; nothing about medicines"})
        if due:
            return due[-1]  # the latest meal only
    if not await _water_today(nh, session, fid, pid, today):
        for t in reversed(await water_times(session, fid, pid)):
            if not t <= m_now <= t + WATER_FOR:
                continue
            about = f"water:{today}:{t // 60:02d}{t % 60:02d}"
            if any(r.about_ref == about for r in sent):
                continue
            return {"kind": "water", "about": about, "fld": "water:count",
                    "why": f"how much water they have had today (usually around {_clock(t)}), nothing logged yet: one easy question, "
                           "glasses or ml; nothing about medicines"}
    if not await _sleep_today(session, fid, pid, today):
        at = await sleep_at(session, fid, pid)
        about = f"sleep:{today}"
        if at <= m_now <= at + SLEEP_FOR and not any(r.about_ref == about for r in sent):
            return {"kind": "sleep", "about": about, "fld": "sleep:when",
                    "why": "when they slept and when they woke: one easy question; the answer is their sleep, nothing about medicines"}
    return None


async def options(session: AsyncSession, p, rows: list[NudgeLedger], now: datetime, *, acts: list[dict], meals_mode: str | None,
                  elder: dict | None = None) -> list[dict]:
    """The one habit question worth asking now ({kind, why, about, fld}); [] when none. rows: their ledger, newest first.
    Everyone is asked about their own meals, water and sleep. elder is the family's elder, for a caregiver's evening ask
    about someone who does not write, and only when nothing of the caregiver's own is due."""
    if meals_mode not in ("shadow", "on"):
        return []
    hk = hooks()
    if hk is None:
        return []
    nh, settings = hk
    fid = p.family_id
    sent = [r for r in rows if r.status in ("sent", "shadow")]
    today = clock.ist_day(now)
    asked = [r for r in sent if r.kind in HABIT_KINDS and clock.ist_day(r.sent_at or r.planned_at) == today]
    if len(asked) >= PER_DAY or (asked and now - (asked[0].sent_at or asked[0].planned_at) < GAP):
        return []
    if any(r.kind == "doses" and now - (r.sent_at or r.planned_at) < DOSE_GAP for r in sent):
        return []
    local = clock.ist(now)
    m_now = local.hour * 60 + local.minute
    own = await _own_habit(session, p, sent, now, acts, nh, settings, today, m_now, local)
    if own:
        return [own]
    if p.cared or not elder or not elder.get("id"):
        return []
    if not await silent_elder(session, fid, elder["id"], now):
        return []
    s = await settings.get(session, fid, elder["id"])
    if s.get("tracking") == "off" or s.get("ask") == "off":
        return []
    at = await recap_at(session, fid, p.id)
    if not at <= m_now <= at + RECAP_WINDOW or (await nh.progress(session, fid, elder["id"])).get("meals_logged"):
        return []
    name = elder.get("name") or "them"
    return [{"kind": "meal", "about": f"meal:{today}:for:{elder['id']}", "fld": "meal:day",
             "why": f"what {name} ate today (they don't write to you themselves; nothing is logged for them yet): one easy "
                    f"question to their caregiver; the answer is about {name} (person {elder['id']}), nothing about medicines"}]


GOAL_FROM, GOAL_UNTIL = 15 * 60, 19 * 60 + 30  # afternoon into early evening


async def goal_option(session: AsyncSession, p, rows: list[NudgeLedger], now: datetime, *, meals_mode: str | None) -> dict | None:
    """One line toward a target they already have, only after two meals are logged, and only naming foods they already eat.
    A maximum ('no more than') is never turned into 'you had too much'."""
    if meals_mode not in ("shadow", "on") or not p.cared:
        return None
    hk = hooks()
    if hk is None:
        return None
    nh, settings = hk
    s = await settings.get(session, p.family_id, p.id)
    if s.get("tracking") == "off" or s.get("ask") == "off":
        return None
    today = clock.ist_day(now)
    sent = [r for r in rows if r.status in ("sent", "shadow")]
    if any(r.kind == "goal" and clock.ist_day(r.sent_at or r.planned_at) == today for r in sent):
        return None
    if any(r.kind == "doses" and now - (r.sent_at or r.planned_at) < DOSE_GAP for r in sent):
        return None
    local = clock.ist(now)
    m_now = local.hour * 60 + local.minute
    if not GOAL_FROM <= m_now <= GOAL_UNTIL:
        return None
    prog = await nh.progress(session, p.family_id, p.id, day=today)
    if int(prog.get("meals_logged") or 0) < 2:
        return None
    open_ = [t for t in (prog.get("targets") or []) if not t.get("met") and (t.get("left") or 0) > 0 and t.get("sense") in ("min", "target")]
    if not open_:
        return None
    order = {"protein_g": 0, "energy_kcal": 1}
    t = sorted(open_, key=lambda x: order.get(str(x.get("nutrient")), 5))[0]
    foods = [str(n).strip() for n in (prog.get("own_foods") or []) if str(n).strip()][:6]
    food_line = (" Name only foods they already eat: " + ", ".join(foods) + ". Name no other food.") if foods else " Do not name any food."
    label = str(t.get("label") or t.get("nutrient") or "target")
    return {"kind": "goal", "about": f"goal:{today}:{t.get('nutrient')}", "fld": f"goal:{t.get('nutrient')}",
            "why": (f"their {label} so far is {t.get('so_far')} of {t.get('target')} {t.get('unit')} (a target already on their record). "
                    "One short line, no diet advice, no 'too much', nothing about medicines." + food_line)}


def _since(v) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=clock.IST)
    except ValueError:
        return None


async def learn_mode(session: AsyncSession, family_id: str, person: str, rows: list[NudgeLedger], now: datetime) -> str | None:
    """Adaptive asking: 3 per-meal asks in a row unanswered → one evening recap; 3 recaps answered → per meal again;
    5 recaps unanswered → paused. Each change is a visible learned fact the family can undo (rows newest first)."""
    from app.care import outcomes_learn

    meals = [r for r in rows if r.kind == "meal" and r.status == "sent" and r.sent_at and now - r.sent_at >= timedelta(hours=3)]
    if not meals:
        return None
    hk = hooks()
    current = await mode(session, family_id, person, await hk[1].get(session, family_id, person)) if hk else "per_meal"
    per_meal = [r for r in meals if r.field != "meal:day"]
    recaps = [r for r in meals if r.field == "meal:day"]
    nxt, text = None, ""
    if current == "per_meal" and len(per_meal) >= TO_RECAP and not any(r.replied_at for r in per_meal[:TO_RECAP]):
        nxt, text = "recap", "Rarely answers questions about each meal; Saheli asks once in the evening what they ate instead."
    elif current == "recap" and len(recaps) >= TO_PAUSE and not any(r.replied_at for r in recaps[:TO_PAUSE]):
        nxt, text = "paused", "Doesn't answer the evening food question; Saheli stops asking, and they can still tell her what they ate."
    elif current in ("recap", "paused") and len(recaps) >= BACK_TO_MEALS and all(r.replied_at for r in recaps[:BACK_TO_MEALS]):
        nxt, text = "per_meal", "Answers the evening food question every day; Saheli asks after each meal again."
    if nxt is None:
        return None
    res = await outcomes_learn.save_learned(session, family_id=family_id, subject_id=person, key=MODE_KEY,
                                            value={"mode": nxt, "since": now.isoformat()}, text=text, now=now)
    return nxt if res in ("created", "superseded") else None


def _hhmm(mins: int) -> str:
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _stable_medians(samples_by_band: list[tuple[str, list[int]]]) -> list[str]:
    """HH:MM for each band whose samples cover 7 days, all within 45 minutes of the median."""
    out = []
    for _name, samples in samples_by_band:
        if len(samples) < STABLE_DAYS:
            continue
        med = int(median(samples))
        if all(abs(s - med) <= STABLE_WITHIN for s in samples):
            out.append(_hhmm(med))
    return out


async def learn_rhythms(session: AsyncSession, family_id: str, person: str) -> dict:
    """Water and sleep times from their own logs, once about seven days agree. Visible learned facts, undoable.
    Meal times stay with the nutrition agent's learn_meal_habits."""
    from app.care import outcomes_learn

    out: dict = {}
    try:
        from app.nutrition.models import NutritionEntry
    except ImportError:
        NutritionEntry = None  # type: ignore[assignment]
    if NutritionEntry is not None:
        since = (clock.ist() - timedelta(days=14)).date().isoformat()
        rows = (await session.execute(select(NutritionEntry.day, NutritionEntry.eaten_at, NutritionEntry.flags, NutritionEntry.kind).where(
            NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person, NutritionEntry.day >= since,
            NutritionEntry.status == "active"))).all()
        by_band: dict[str, dict[str, int]] = {"morning": {}, "afternoon": {}, "evening": {}}
        bands = (("morning", 5 * 60, 12 * 60), ("afternoon", 12 * 60, 17 * 60), ("evening", 17 * 60, 24 * 60))
        for day, at, flags, kind in rows:
            tags = (flags or {}).get("tags") or []
            if kind != "water" and "water" not in tags:
                continue
            minute = clock.ist(at).hour * 60 + clock.ist(at).minute
            for name, lo, hi in bands:
                if lo <= minute < hi and (day not in by_band[name] or minute < by_band[name][day]):
                    by_band[name][day] = minute
        stable = _stable_medians([(name, list(days.values())) for name, days in by_band.items()])
        if stable:
            words = ", ".join(_clock(_hm(t) or 0) for t in stable)
            out["water"] = await outcomes_learn.save_learned(
                session, family_id=family_id, subject_id=person, key=WATER_KEY,
                value={"times": stable}, text=f"Usually drinks water around {words}; Saheli asks then.")
    since_at = clock.now() - timedelta(days=14)
    days: dict[str, int] = {}
    for e in await store.events(session, family_id, person, since=since_at, kinds=["sleep", "activity_done"]):
        payload = e.payload or {}
        key = str(payload.get("key") or "")
        if e.kind != "sleep" and "sleep" not in key and payload.get("type") != "sleep":
            continue
        local = clock.ist(e.at)
        days[getattr(e, "day", None) or clock.ist_day(e.at)] = local.hour * 60 + local.minute
    if len(days) >= STABLE_DAYS:
        med = int(median(days.values()))
        if all(abs(v - med) <= STABLE_WITHIN for v in days.values()):
            out["sleep"] = await outcomes_learn.save_learned(
                session, family_id=family_id, subject_id=person, key=SLEEP_KEY,
                value={"at": _hhmm(med)}, text=f"Usually up around {_clock(med)}; Saheli asks about sleep then.")
    return out
