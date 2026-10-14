"""Meal nudges (nutrition plan §N5, founder §N18, 14 Oct 2026): after a main meal nobody has logged, a regular nudge may ask
what they ate. Adaptive: up to 3 asks a day while they answer; for someone who rarely answers, one evening recap; paused
when even that goes unanswered. Meal times come from their own routine, then what they usually log, then defaults moved
to their own day. A meal ask is that hour's regular nudge, never one more message; it never comes within 30 minutes of a
medicine question, never for a meal a delivered order covers (aftercare asks "did you eat it?"), and never while a food
question is already open. For an elder who does not write to Saheli, their caregiver gets one evening ask (founder F1).

The answer goes to the nutrition agent's `log_food` tool (reply_hint). The nutrition agent (app.nutrition) lands on its
own: until it exists, no meal option is offered. The admin flag nudge.meals decides off | shadow | on.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import NudgeLedger, Turn
from app.core import clock

MAIN = ("breakfast", "lunch", "dinner")
DEFAULT_AT = {"lunch": 13 * 60 + 30, "dinner": 20 * 60 + 30}  # breakfast: an hour after they wake
ASK_AFTER, ASK_UNTIL = 45, 180  # minutes after the meal's time
PER_DAY, GAP = 3, timedelta(hours=3)
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


async def times(session: AsyncSession, family_id: str, person: str, acts: list[dict]) -> dict[str, int]:
    """Each main meal's usual time (minutes of the day): their routine, then the learned time, then a default moved to
    their own day. A meal before they are half an hour up is not theirs (a 1:30 pm riser has no 9 am breakfast and no 1:30 pm
    lunch)."""
    from app.brain.wake import own_night

    out: dict[str, int] = {}
    for a in acts:
        if a.get("type") in MAIN and _hm(a.get("start")) is not None:
            out[a["type"]] = _hm(a["start"])  # type: ignore[assignment]
    learned = await store.active_fact(session, family_id, person, TIMES_KEY)
    for slot in MAIN:
        t = _hm(((learned.value or {}) if learned else {}).get(slot))
        if slot not in out and t is not None:
            out[slot] = t
    night = await own_night(session, family_id, person)
    wake = night[1] if night else 8 * 60
    out.setdefault("breakfast", wake + 60)
    for slot, t in DEFAULT_AT.items():
        out.setdefault(slot, t)
    return {s: t for s, t in out.items() if t >= wake + 30}


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


async def options(session: AsyncSession, p, rows: list[NudgeLedger], now: datetime, *, acts: list[dict], meals_mode: str | None,
                  elder: dict | None = None) -> list[dict]:
    """The meal question worth asking now, as option fields ({kind, why, about, fld}); [] when none. rows: their ledger,
    newest first. elder: the family's elder ({id, name}), for a caregiver's evening ask about someone who does not write."""
    if meals_mode not in ("shadow", "on"):
        return []
    hk = hooks()
    if hk is None:
        return []
    nh, settings = hk
    fid = p.family_id
    sent = [r for r in rows if r.status in ("sent", "shadow")]
    today = clock.ist_day(now)
    asked = [r for r in sent if r.kind == "meal" and clock.ist_day(r.sent_at or r.planned_at) == today]
    if len(asked) >= PER_DAY or (asked and now - (asked[0].sent_at or asked[0].planned_at) < GAP):
        return []
    if any(r.kind == "doses" and now - (r.sent_at or r.planned_at) < DOSE_GAP for r in sent):
        return []
    local = clock.ist(now)
    m_now = local.hour * 60 + local.minute

    if not p.cared:  # a caregiver: only the evening ask about an elder who does not write to Saheli (founder F1)
        if not elder or not elder.get("id") or asked:
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

    s = await settings.get(session, fid, p.id)
    if s.get("tracking") == "off" or s.get("ask") == "off":
        return []
    how = await mode(session, fid, p.id, s)
    if how == "paused":
        return []
    pending = await nh.pending_confirmations(session, fid, p.id)
    if any(c.get("kind") == "question" for c in pending):
        return []  # a food conversation is already open: the agent's own question waits for its answer
    meal_at = await times(session, fid, p.id, acts)
    if how == "recap":
        at = await recap_at(session, fid, p.id)
        if asked or not at <= m_now <= at + RECAP_WINDOW:
            return []
        missing = [slot for slot, t in meal_at.items() if t < m_now and not await nh.meal_logged(session, fid, p.id, slot, day=today)]
        if not missing:
            return []
        return [{"kind": "meal", "about": f"meal:{today}:recap", "fld": "meal:day",
                 "why": f"a short recap of what they ate today ({', '.join(missing)} not told yet): one easy question, nothing about medicines"}]
    ordered = [t for t in ((c.get("since") and _since(c["since"])) for c in pending if c.get("kind") == "order_meal") if t]
    due = []
    for slot in MAIN:
        t = meal_at.get(slot)
        if t is None or not t + ASK_AFTER <= m_now <= t + ASK_UNTIL:
            continue
        if any(r.about_ref == f"meal:{today}:{slot}" for r in sent):
            continue
        start = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=t - 120)
        if any(o >= start for o in ordered):
            continue  # a delivered meal from an order: aftercare asks "did you eat it?"
        if await nh.meal_logged(session, fid, p.id, slot, day=today):
            continue
        due.append({"kind": "meal", "about": f"meal:{today}:{slot}", "fld": f"meal:{slot}",
                    "why": f"what they had for {slot} (usually around {_clock(t)}), nothing logged yet: one easy question about what "
                           "and roughly how much; nothing about medicines"})
    return due[-1:]  # the latest meal only


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
