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

from datetime import date, datetime, timedelta
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
WATER_AFTER, WATER_FOR = 20, 90  # ask a little after the usual glass, and keep the ask open until 90 minutes past it
SLEEP_AFTER, SLEEP_FOR = 30, 60  # from half an hour after waking, for an hour (before breakfast is asked)
WATER_KEY = "learned:nudges_water_times"
SLEEP_KEY = "learned:nudges_sleep_ask"
SKIP_KEY = "learned:nudges_skip"  # {"fields": ["water", ...]} — they asked not to be asked
ASK_HOURS_KEY = "learned:nudges_ask_hours"  # {"avoid_hours": [14], "prefer_hour": 19}
STABLE_DAYS, STABLE_WITHIN = 7, 45
WEEKDAY_NEED, WEEKEND_NEED, OUTLIER = 4, 2, 90  # four agreeing weekdays; two agreeing weekend days; one far day is ignored
DOSE_GAP = timedelta(minutes=30)
RECAP_BEFORE_BED, RECAP_WINDOW = 90, 120  # minutes
TO_RECAP, BACK_TO_MEALS, TO_PAUSE = 3, 3, 5  # unanswered per-meal asks → recap; answered recaps → per meal; unanswered recaps → pause
QUIET_ELDER = timedelta(days=30)  # an elder with no message of their own this long does not write to Saheli
MODE_KEY = "learned:nudges_meals_mode"
TIMES_KEY = "learned:nudges_meals_times"  # a time they stated, or the nutrition agent's flat usual time
CLOCKS_KEY = "learned:nudges_meals_clocks"  # weekday and weekend, from their logs; does not overwrite TIMES_KEY


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


def _side(now=None) -> str:
    return "weekend" if clock.ist(now).weekday() >= 5 else "weekday"


def cluster_minutes(points: list[int], *, need: int) -> int | None:
    """The median once `need` points sit within 45 minutes of it. A day more than 90 minutes off is left out,
    so one odd day or a weekend shift does not stop the rest from settling."""
    if len(points) < need:
        return None
    med = int(median(points))
    core = [p for p in points if abs(p - med) <= OUTLIER]
    if len(core) < need:
        return None
    med = int(median(core))
    close = [p for p in core if abs(p - med) <= STABLE_WITHIN]
    if len(close) < need:
        return None
    return int(median(close))


def split_clocks(samples: list[tuple[str, int]]) -> dict[str, int]:
    """(YYYY-MM-DD, minute) → minutes for the whole week, weekdays, and weekends, each only once it agrees."""
    wd, we = [], []
    for day, minute in samples:
        (we if date.fromisoformat(day).weekday() >= 5 else wd).append(minute)
    out: dict[str, int] = {}
    for key, pts, need in (("at", [m for _, m in samples], WEEKDAY_NEED), ("weekday", wd, WEEKDAY_NEED), ("weekend", we, WEEKEND_NEED)):
        got = cluster_minutes(pts, need=need)
        if got is not None:
            out[key] = got
    return out


def _learned_hhmm(value: dict | None, slot: str, now=None) -> str | None:
    value = value or {}
    if slot in set(value.get("stated") or []) and isinstance(value.get(slot), str):
        return str(value[slot])  # they said this; it applies every day until they undo it
    side = value.get(_side(now))
    if isinstance(side, dict) and side.get(slot):
        return str(side[slot])
    got = value.get(slot)
    return str(got) if isinstance(got, str) else None


def _apply_hours(minute: int, avoid: set[int], prefer: int | None) -> int | None:
    """Move an ask out of an hour they never answer. None drops it when there is nowhere better yet."""
    if minute // 60 not in avoid:
        return minute
    if prefer is not None and prefer not in avoid and 0 <= prefer <= 23:
        return prefer * 60 + (minute % 60)
    return None


async def _wake(session: AsyncSession, family_id: str, person: str) -> int:
    from app.brain.wake import own_night

    night = await own_night(session, family_id, person)
    return night[1] if night else 8 * 60


async def _ask_hours(session: AsyncSession, family_id: str, person: str) -> tuple[set[int], int | None]:
    f = await store.active_fact(session, family_id, person, ASK_HOURS_KEY)
    if not f:
        return set(), None
    value = f.value or {}
    prefer = value.get("prefer_hour")
    return {int(h) for h in (value.get("avoid_hours") or [])}, prefer if isinstance(prefer, int) else None


async def _fields_skipped(session: AsyncSession, family_id: str, person: str) -> set[str]:
    f = await store.active_fact(session, family_id, person, SKIP_KEY)
    return {str(x) for x in (((f.value or {}) if f else {}).get("fields") or [])}


async def times(session: AsyncSession, family_id: str, person: str, acts: list[dict]) -> dict[str, int]:
    """Each meal's usual time (minutes of the day): their routine, then a time they stated or that their logs settled
    on (weekdays and weekends apart), then a default moved to their own day. A meal before they are half an hour up
    is not theirs. An hour they never answer is dropped, or moved to an hour they answer quickly."""
    out: dict[str, int] = {}
    for a in acts:
        if a.get("type") in SLOTS and _hm(a.get("start")) is not None:
            out[a["type"]] = _hm(a["start"])  # type: ignore[assignment]
    clocks = await store.active_fact(session, family_id, person, CLOCKS_KEY)
    said = await store.active_fact(session, family_id, person, TIMES_KEY)
    said_v = (said.value or {}) if said else {}
    value = dict((clocks.value or {}) if clocks else said_v)
    stated = [s for s in (said_v.get("stated") or []) if isinstance(said_v.get(s), str)]
    if stated:
        value["stated"] = stated
        for slot in stated:
            value[slot] = said_v[slot]
    for slot in SLOTS:
        t = _hm(_learned_hhmm(value, slot))
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
    avoid, prefer = await _ask_hours(session, family_id, person)
    moved: dict[str, int] = {}
    for slot, t in out.items():
        if t < wake + 30:
            continue
        at = _apply_hours(t, avoid, prefer)
        if at is not None and at >= wake + 30:
            moved[slot] = at
    return moved


def _listed_times(value: dict | None, now=None) -> list:
    value = value or {}
    side = value.get(_side(now))
    if isinstance(side, list) and side:
        return side
    return value.get("times") or []


async def water_times(session: AsyncSession, family_id: str, person: str) -> list[int]:
    """When to ask about water: times learned from their own logs (weekday or weekend), else the defaults after they are up."""
    learned = await store.active_fact(session, family_id, person, WATER_KEY)
    wake = await _wake(session, family_id, person)
    got = [t for t in (_hm(v) for v in _listed_times((learned.value or {}) if learned else {})) if t is not None]
    avoid, prefer = await _ask_hours(session, family_id, person)
    out = []
    for t in got or list(WATER_AT):
        if t < wake + 30:
            continue
        at = _apply_hours(t, avoid, prefer)
        if at is not None and at >= wake + 30:
            out.append(at)
    return out


async def sleep_at(session: AsyncSession, family_id: str, person: str) -> int | None:
    """When to ask about last night: the learned time, else half an hour after they usually wake. None: that hour is dropped."""
    learned = await store.active_fact(session, family_id, person, SLEEP_KEY)
    wake = await _wake(session, family_id, person)
    value = (learned.value or {}) if learned else {}
    side = value.get(_side())
    t = _hm(side if isinstance(side, str) else None) or _hm(value.get("at"))
    base = t if t is not None and t >= wake else wake + SLEEP_AFTER
    avoid, prefer = await _ask_hours(session, family_id, person)
    return _apply_hours(base, avoid, prefer)


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


def _chat_covers(topics: set[str], *keys: str) -> bool:
    """A check-in topic written today (gap_asked) already asked this."""
    if topics & set(keys):
        return True
    return any(t.startswith(tuple(k + ":" for k in keys)) for t in topics)


def _meal_slot_asked(topics: set[str], slot: str, today: str) -> bool:
    """A chat ask of 'meal', 'meal:day', this slot, or meal:<today>:<slot> already covered it. Another slot did not."""
    return bool(topics & {"meal", "meal:day", f"meal:{slot}", f"meal:{today}:{slot}", f"meal:{today}:day"})


async def _chat_topics(session: AsyncSession, family_id: str, person: str, today: str, now: datetime) -> set[str]:
    """Topics Saheli already asked in conversation today (check-in events), not ones from earlier days."""
    from app.care import checkins

    return {topic for topic, at in (await checkins._asked(session, family_id, person, now)).items()
            if topic and clock.ist_day(at) == today}


async def _own_habit(session: AsyncSession, p, sent: list[NudgeLedger], now: datetime, acts: list[dict], nh, settings,
                     today: str, m_now: int, local: datetime, topics: set[str]) -> dict | None:
    """The one habit question this person is due: a meal (including their own, for a caregiver), else water, else sleep."""
    fid, pid = p.family_id, p.id
    skip = await _fields_skipped(session, fid, pid)
    s = await settings.get(session, fid, pid)
    meals_off = s.get("tracking") == "off" or s.get("ask") == "off"
    how = "paused" if meals_off else await mode(session, fid, pid, s)
    pending = [] if meals_off else await nh.pending_confirmations(session, fid, pid)
    open_food = any(c.get("kind") == "question" for c in pending)
    if how == "recap" and not open_food:
        at = await recap_at(session, fid, pid)
        meal_asked = [r for r in sent if r.kind == "meal" and clock.ist_day(r.sent_at or r.planned_at) == today]
        if not meal_asked and not _chat_covers(topics, "meal") and at <= m_now <= at + RECAP_WINDOW:
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
            if slot in skip:
                continue
            t = meal_at.get(slot)
            if t is None or not t + ASK_AFTER <= m_now <= t + ASK_UNTIL:
                continue
            if any(r.about_ref == f"meal:{today}:{slot}" for r in sent):
                continue
            if _meal_slot_asked(topics, slot, today):
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
    if "water" not in skip and not await _water_today(nh, session, fid, pid, today) and not _chat_covers(topics, "water"):
        for t in reversed(await water_times(session, fid, pid)):
            if not t + WATER_AFTER <= m_now <= t + WATER_FOR:
                continue
            about = f"water:{today}:{t // 60:02d}{t % 60:02d}"
            if any(r.about_ref == about for r in sent):
                continue
            return {"kind": "water", "about": about, "fld": "water:count",
                    "why": f"how much water they have had today (usually around {_clock(t)}), nothing logged yet: one easy question, "
                           "glasses or ml; nothing about medicines"}
    if "sleep" not in skip and not await _sleep_today(session, fid, pid, today) and not _chat_covers(topics, "sleep"):
        at = await sleep_at(session, fid, pid)
        about = f"sleep:{today}"
        if at is not None and at <= m_now <= at + SLEEP_FOR and not any(r.about_ref == about for r in sent):
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
    topics = await _chat_topics(session, fid, p.id, today, now)
    own = await _own_habit(session, p, sent, now, acts, nh, settings, today, m_now, local, topics)
    if own:
        return [own]
    if p.cared or not elder or not elder.get("id") or _chat_covers(topics, "meal"):
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


def _put_split(value: dict, slot: str | None, clocks: dict[str, int]) -> bool:
    """Write one slot (or the sleep clock when slot is None) from a weekday/weekend split. A stated slot is left as they said it."""
    if slot and slot in set(value.get("stated") or []):
        return False
    if slot is None and value.get("stated"):
        return False
    changed = False
    if slot:
        if "weekday" in clocks:
            value[slot] = _hhmm(clocks["weekday"])
            changed = True
        elif "at" in clocks:
            value[slot] = _hhmm(clocks["at"])
            changed = True
        for side in ("weekday", "weekend"):
            if side not in clocks:
                continue
            got = value.get(side)
            if not isinstance(got, dict):
                got = {}
            got[slot] = _hhmm(clocks[side])
            value[side] = got
            changed = True
        return changed
    if "weekday" in clocks:
        value["at"] = _hhmm(clocks["weekday"])
        value["weekday"] = value["at"]
        changed = True
    elif "at" in clocks:
        value["at"] = _hhmm(clocks["at"])
        changed = True
    if "weekend" in clocks:
        value["weekend"] = _hhmm(clocks["weekend"])
        changed = True
    return changed


async def learn_rhythms(session: AsyncSession, family_id: str, person: str) -> dict:
    """Meal, water and sleep times from their own logs. Four agreeing weekdays settle a weekday time; two agreeing
    weekend days settle the weekend. One day more than 90 minutes off is ignored. A time they stated is kept.
    Visible learned facts, undoable."""
    from app.care import outcomes_learn

    out: dict = {}
    try:
        from app.nutrition.models import NutritionEntry
    except ImportError:
        NutritionEntry = None  # type: ignore[assignment]
    if NutritionEntry is not None:
        since = (clock.ist() - timedelta(days=21)).date().isoformat()
        rows = (await session.execute(select(
            NutritionEntry.day, NutritionEntry.eaten_at, NutritionEntry.flags, NutritionEntry.kind, NutritionEntry.meal).where(
            NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person, NutritionEntry.day >= since,
            NutritionEntry.status == "active"))).all()
        by_slot: dict[str, dict[str, int]] = {s: {} for s in SLOTS}
        by_band: dict[str, dict[str, int]] = {"morning": {}, "afternoon": {}, "evening": {}}
        bands = (("morning", 5 * 60, 12 * 60), ("afternoon", 12 * 60, 17 * 60), ("evening", 17 * 60, 24 * 60))
        for day, at, flags, kind, meal in rows:
            minute = clock.ist(at).hour * 60 + clock.ist(at).minute
            tags = (flags or {}).get("tags") or []
            if kind == "water" or "water" in tags:
                for name, lo, hi in bands:
                    if lo <= minute < hi and (day not in by_band[name] or minute < by_band[name][day]):
                        by_band[name][day] = minute
            elif kind == "food" and meal in SLOTS and (day not in by_slot[meal] or minute < by_slot[meal][day]):
                by_slot[meal][day] = minute
        said = await store.active_fact(session, family_id, person, TIMES_KEY)
        prev = await store.active_fact(session, family_id, person, CLOCKS_KEY)
        meal_value = dict((prev.value or {}) if prev else {})
        meal_value["stated"] = [s for s in (((said.value or {}) if said else {}).get("stated") or [])]
        wrote = False
        for slot, days in by_slot.items():
            wrote = _put_split(meal_value, slot, split_clocks(list(days.items()))) or wrote
        if wrote and meal_value != ((prev.value or {}) if prev else {}):
            bits = [f"{slot} around {_clock(_hm(meal_value[slot]) or 0)}" for slot in SLOTS if isinstance(meal_value.get(slot), str)]
            weekend = meal_value.get("weekend") if isinstance(meal_value.get("weekend"), dict) else {}
            if weekend:
                bits.append("weekends " + ", ".join(f"{slot} {_clock(_hm(t) or 0)}" for slot, t in weekend.items()))
            out["meals"] = await outcomes_learn.save_learned(
                session, family_id=family_id, subject_id=person, key=CLOCKS_KEY, value=meal_value,
                text=("Usually has " + "; ".join(bits) + "; Saheli asks after that.") if bits else "Usual meal times, from their own logs.")
        water_prev = await store.active_fact(session, family_id, person, WATER_KEY)
        if not (water_prev and (water_prev.value or {}).get("stated")):
            weekday_times, weekend_times = [], []
            for days in by_band.values():
                clocks = split_clocks(list(days.items()))
                if "weekday" in clocks:
                    weekday_times.append(clocks["weekday"])
                elif "at" in clocks:
                    weekday_times.append(clocks["at"])
                if "weekend" in clocks:
                    weekend_times.append(clocks["weekend"])
            if weekday_times:
                water_value: dict = {"times": [_hhmm(t) for t in sorted(set(weekday_times))],
                                     "weekday": [_hhmm(t) for t in sorted(set(weekday_times))]}
                if weekend_times:
                    water_value["weekend"] = [_hhmm(t) for t in sorted(set(weekend_times))]
                if not water_prev or (water_prev.value or {}) != water_value:
                    words = ", ".join(_clock(t) for t in sorted(set(weekday_times)))
                    out["water"] = await outcomes_learn.save_learned(
                        session, family_id=family_id, subject_id=person, key=WATER_KEY, value=water_value,
                        text=f"Usually drinks water around {words}; Saheli asks then.")
    since_at = clock.now() - timedelta(days=21)
    days: dict[str, int] = {}
    for e in await store.events(session, family_id, person, since=since_at, kinds=["sleep", "activity_done"]):
        payload = e.payload or {}
        key = str(payload.get("key") or "")
        if e.kind != "sleep" and "sleep" not in key and payload.get("type") != "sleep":
            continue
        local = clock.ist(e.at)
        days[getattr(e, "day", None) or clock.ist_day(e.at)] = local.hour * 60 + local.minute
    sleep_prev = await store.active_fact(session, family_id, person, SLEEP_KEY)
    sleep_value = dict((sleep_prev.value or {}) if sleep_prev else {})
    if _put_split(sleep_value, None, split_clocks(list(days.items()))) and sleep_value != ((sleep_prev.value or {}) if sleep_prev else {}):
        med = _hm(sleep_value.get("at")) or 0
        out["sleep"] = await outcomes_learn.save_learned(
            session, family_id=family_id, subject_id=person, key=SLEEP_KEY, value=sleep_value,
            text=f"Usually up around {_clock(med)}; Saheli asks about sleep then.")
    return out


def _hour_span(hours: list[int]) -> str:
    """10 and 11 become 10:00–12:00. Hours are the ones she stops using."""
    parts: list[str] = []
    start = prev = hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        parts.append(f"{start}:00–{prev + 1}:00")
        start = prev = h
    parts.append(f"{start}:00–{prev + 1}:00")
    return " and ".join(parts)


async def learn_reply_hours(session: AsyncSession, family_id: str, person: str) -> str | None:
    """A two-hour window with at least four asks and almost no answer within 45 minutes is dropped. A later message
    the same afternoon is not an answer to that ask. An hour they answer within half an hour, at least three times,
    is where those asks move. The fact is visible and undoable, and says when not to message them."""
    from app.care import outcomes_learn

    soon = outcomes_learn.ANSWER_WITHIN_S / 60
    since = clock.now() - timedelta(days=14)
    rows = (await session.execute(select(NudgeLedger).where(
        NudgeLedger.family_id == family_id, NudgeLedger.person_id == person, NudgeLedger.kind != "reminder_followup",
        NudgeLedger.status == "sent", NudgeLedger.sent_at.is_not(None), NudgeLedger.sent_at >= since))).scalars()
    by: dict[int, list[float | None]] = {}
    windows: dict[int, list[float | None]] = {}
    for r in rows:
        sent_at = r.sent_at
        if sent_at is None:
            continue
        hour = clock.ist(sent_at).hour
        delay = ((r.replied_at - sent_at).total_seconds() / 60) if r.replied_at else None
        by.setdefault(hour, []).append(delay)
        w = outcomes_learn._window(hour)
        if w is not None:
            windows.setdefault(w, []).append(delay)
    quiet = [w for w, delays in windows.items()
             if len(delays) >= outcomes_learn.MIN_TRIES
             and sum(1 for d in delays if d is not None and d <= soon) / len(delays) <= outcomes_learn.LOW]
    avoid = sorted(h for w in quiet for h in range(*outcomes_learn.WINDOWS[w]))
    fast: list[tuple[float, int]] = []
    for hour, delays in by.items():
        got = [d for d in delays if d is not None and d <= 30]
        if len(got) >= 3 and hour not in avoid:
            fast.append((float(median(got)), hour))
    prefer = sorted(fast)[0][1] if fast else None
    if not avoid:
        return None
    value = {"avoid_hours": avoid, "prefer_hour": prefer}
    span = _hour_span(avoid)
    if prefer is None:
        text = f"Doesn't answer from {span}; Saheli stops starting a conversation then."
    else:
        text = f"Doesn't answer from {span}; Saheli asks around {prefer}:00 instead, where replies come quickly."
    prev = await store.active_fact(session, family_id, person, ASK_HOURS_KEY)
    if prev and (prev.value or {}) == value:
        return None
    res = await outcomes_learn.save_learned(session, family_id=family_id, subject_id=person, key=ASK_HOURS_KEY, value=value, text=text)
    return res if res in ("created", "superseded") else None


async def state_habit(session: AsyncSession, family_id: str, person: str, *, what: str, at: str | None = None,
                       skip: bool = False) -> str:
    """They said a usual time, or asked not to be asked. Applied at once, as a fact they can see and undo.
    `what` is a meal slot, water, or sleep. No reading of their words: the caller passes the structured fields."""
    from app.care import outcomes_learn

    if what not in (*SLOTS, "water", "sleep"):
        raise ValueError("what must be a meal, water, or sleep")
    if skip:
        current = await store.active_fact(session, family_id, person, SKIP_KEY)
        fields = set(((current.value or {}) if current else {}).get("fields") or [])
        fields.add(what)
        text = f"Asked not to be asked about {what}."
        return await outcomes_learn.save_learned(
            session, family_id=family_id, subject_id=person, key=SKIP_KEY, value={"fields": sorted(fields)}, text=text)
    if _hm(at) is None:
        raise ValueError("at is a time HH:MM")
    if what == "water":
        key, text = WATER_KEY, f"Usually drinks water around {_clock(_hm(at) or 0)}; they said so."
        current = await store.active_fact(session, family_id, person, key)
        value = dict((current.value or {}) if current else {})
        times = [t for t in (value.get("times") or []) if isinstance(t, str) and t != at]
        value["times"] = sorted([*times, at], key=lambda s: _hm(s) or 0)
        value.pop("weekday", None)
        value.pop("weekend", None)
        value["stated"] = True
    elif what == "sleep":
        key, text = SLEEP_KEY, f"Usually up around {_clock(_hm(at) or 0)}; they said so."
        value = {"at": at, "stated": True}
    else:
        key = TIMES_KEY
        text = f"Usually has {what} around {_clock(_hm(at) or 0)}; they said so."
        current = await store.active_fact(session, family_id, person, key)
        value = dict((current.value or {}) if current else {})
        value[what] = at
        stated = [s for s in (value.get("stated") or []) if s != what]
        value["stated"] = [*stated, what]
    return await outcomes_learn.save_learned(session, family_id=family_id, subject_id=person, key=key, value=value, text=text)
