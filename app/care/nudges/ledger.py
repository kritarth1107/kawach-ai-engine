"""Nudge ledger: the flag cache, what was sent, replies, and what was learned from them."""

from __future__ import annotations

import logging
import os
import re
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, or_, select, text as sql
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from typing import TYPE_CHECKING

from app.care import store
from app.care.models import FamilyRoster, NudgeDue, NudgeLedger, Turn
from app.core import clock

if TYPE_CHECKING:  # planner imports this module, so Person is not a runtime import
    from app.care.nudges.planner import Person

logger = logging.getLogger(__name__)

CARED_EVERY = timedelta(minutes=60)  # people cared for and people caring for themselves
CAREGIVER_EVERY = timedelta(hours=3)  # caregivers and family members who wrote in the last 30 days (D2)
LADDER_MINUTES = {0: 60, 1: 60, 2: 120, 3: 180}  # unanswered in a row → silence before the next (D1)
STOP_AFTER = 4  # then only the next morning's first nudge, until they write
PARALLEL = 5  # people at a time (NUDGE_PARALLEL). One send per person per cadence, not a global cap.
CLAIM_FOR = timedelta(minutes=3)  # a second engine task skips a person claimed this recently
REPLY_WINDOW = timedelta(hours=6)  # a reply this soon after a nudge answers it
SKIP_LOG_EVERY = timedelta(minutes=60)  # the same skip reason is logged at most this often per person
MAX_LEN = {True: 220, False: 300}  # an elder / anyone else
DOSES_AFTER_MIN = 120  # an unmarked item is a nudge option only this long after its time
EARLY_BIRD_MAX, EARLY_BIRD_GAP = 2, timedelta(hours=2)
FOLLOWUP_AFTER = (timedelta(minutes=5), timedelta(minutes=10))  # a reminder this long ago, unanswered, gets one question
MODE_TTL = 60.0
# Learning from how they took it (self-learning, 2026-10-13)
REACTIONS = ("welcome", "neutral", "unwelcome", "fewer", "stop")
STOP_PAUSE = timedelta(days=30)  # "stop messaging me": no regular nudges or follow-ups for 30 days
FEWER_FOR = timedelta(days=14)  # "too many messages": twice the silence before each nudge
UNWELCOME_PAUSE = timedelta(days=30)  # "don't ask me that": that question (or that kind of light talk) rests
LEARN_DAYS = timedelta(days=30)  # reply rates per kind are read over this window
PRIOR = 4.0  # how many nudges' worth of weight the all-family rate carries for a new person
REVIEW_EVERY = 3600.0  # the real nudges are graded at most once an hour, a few at a time
REVIEW_BATCH = 15
WINDOW = timedelta(hours=24) - timedelta(minutes=10)  # WhatsApp's free-message window, with the backend's margin
SENT = ("sent", "shadow")  # a shadow nudge counts as sent for the clock and the ladder in shadow mode
PRIORITY = {"doses": 9, "follow_up": 7, "upcoming": 6, "habit": 6, "collect": 4, "reflect": 4, "know": 3, "delight": 3,
            "early_bird": 5, "reminder_followup": 8, "meal": 7, "water": 7, "sleep": 7, "goal": 5, "stock": 6,
            "checkin": 2, "quiet_note": 2}
HABIT_SHADOW = ("meal", "goal", "water", "sleep")  # nudge.meals: off sends nothing; shadow logs them and does not hold the hour
SLOTS = (("morning", 5 * 60, 12 * 60), ("afternoon", 12 * 60, 17 * 60), ("evening", 17 * 60, 24 * 60))
COLLECT_EVERY = {"feelings": timedelta(days=1), "sleep": timedelta(days=2), "readings": timedelta(days=4),
                 "weight": timedelta(days=20), "reports": timedelta(days=7), "water": timedelta(days=2), "meals": timedelta(days=2)}

_mode: dict = {"value": None, "at": 0.0}


def env_mode() -> str | None:
    v = (os.getenv("NUDGE_REGULAR") or "").strip().lower()
    return v if v in ("off", "shadow", "on") else None


def cached_mode() -> str | None:
    """The mode as last seen (the backend's flag, at most a minute old), for the old check-ins to step aside when on."""
    if env_mode():
        return env_mode()
    return _mode["value"] if _time.monotonic() - _mode["at"] < 10 * MODE_TTL else None


def _remember_mode(gate: dict) -> str:
    v = str((gate or {}).get("mode") or "off")
    _mode.update(value=v if v in ("off", "shadow", "on") else "off", at=_time.monotonic())
    return _mode["value"]


async def ensure_mode(host_for, rosters: list[FamilyRoster]) -> str | None:
    """The flag, read from the backend once a run when the last read is over a minute old. Without it, a run where nobody is
    due (all asleep, talking or within their cadence) never asks, the cached mode expires, and the old check-ins would
    start again while regular nudges are on (found 13 Oct, the night nudges went on)."""
    if _mode["value"] is not None and _time.monotonic() - _mode["at"] < MODE_TTL:
        return _mode["value"]
    for r in rosters[:3]:
        pid = (r.elder or {}).get("id") or next((m.get("id") for m in r.members or [] if m.get("id")), None)
        if not pid:
            continue
        try:
            gate = await host_for(r.family_id).call("nudge_gate", {"to": pid}, family_id=r.family_id, subject_id=pid, actor_id=pid)
        except Exception:  # noqa: BLE001 — another family may answer
            continue
        return _remember_mode(gate or {})
    return None

def _iso(v) -> datetime | None:
    if not v:
        return None
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=clock.IST)
    except ValueError:
        return None


def _when(r: NudgeLedger) -> datetime:
    return r.sent_at or r.planned_at


# Optional chat only. Medicine, stock, meals, water, sleep, habits, appointments and reminder follow-ups are not in this:
# a quiet person still hears those, however many chats they left unanswered.
_OPTIONAL_CHAT = frozenset({"know", "reflect", "delight", "follow_up", "collect"})


def _optional_chat(r: NudgeLedger) -> bool:
    if r.kind not in _OPTIONAL_CHAT:
        return False
    return not (r.kind == "follow_up" and "appointment" in (r.about_ref or ""))

async def ledger_rows(session: AsyncSession, family_id: str, person: str, limit: int = 30) -> list[NudgeLedger]:
    return list((await session.execute(
        select(NudgeLedger).where(NudgeLedger.family_id == family_id, NudgeLedger.person_id == person)
        .order_by(NudgeLedger.planned_at.desc()).limit(limit)
    )).scalars())


async def sent_lines(session: AsyncSession, family_id: str, person: str, since: datetime) -> list[str]:
    """Texts actually sent to this person since `since` (a repeat of one of these does not go out again)."""
    rows = (await session.execute(
        select(NudgeLedger.text).where(NudgeLedger.family_id == family_id, NudgeLedger.person_id == person,
                                       NudgeLedger.status == "sent", NudgeLedger.sent_at >= since, NudgeLedger.text.isnot(None))
        .order_by(NudgeLedger.sent_at.desc()).limit(80)
    )).scalars()
    return [t for t in rows if t]


def _day_start(now: datetime) -> datetime:
    return clock.ist(now).replace(hour=0, minute=0, second=0, microsecond=0)


async def sent_today(session: AsyncSession, person: str, now: datetime) -> list[NudgeLedger]:
    """Nudges sent to this person today, in every family they belong to."""
    return list((await session.execute(
        select(NudgeLedger).where(NudgeLedger.person_id == person, NudgeLedger.status == "sent", NudgeLedger.sent_at >= _day_start(now),
                                  NudgeLedger.kind != "reminder_followup", ~NudgeLedger.family_id.startswith("shadow:"))
    )).scalars())


async def sent_in_other_family(session: AsyncSession, person: str, family_id: str, since: datetime) -> bool:
    """True when another family's stream already reached this person since `since` (one stream per person)."""
    found = await session.scalar(select(NudgeLedger.id).where(
        NudgeLedger.person_id == person, NudgeLedger.family_id != family_id, NudgeLedger.status == "sent",
        NudgeLedger.sent_at >= since, ~NudgeLedger.family_id.startswith("shadow:")).limit(1))
    return found is not None


_PHONE_KEYS = ("phone", "phoneNumber", "whatsapp", "mobile")


def _phone_digits(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else ""


def _blob_phone(blob: object) -> str:
    if not isinstance(blob, dict):
        return ""
    for key in _PHONE_KEYS:
        got = _phone_digits(blob.get(key))
        if got:
            return got
    return ""


async def _person_ids(session: AsyncSession, person: str) -> set[str]:
    """This user id, plus anyone who shares their phone on a roster. No phone stored means just this user."""
    rows = (await session.execute(sql(
        "SELECT elder, members FROM family_rosters WHERE family_id NOT LIKE 'shadow:%' AND ("
        " elder->>'id' = :pid OR EXISTS (SELECT 1 FROM jsonb_array_elements(COALESCE(members, '[]'::jsonb)) m"
        " WHERE m->>'id' = :pid))"
    ), {"pid": person})).all()
    phone = ""
    for elder, members in rows:
        for blob in [elder or {}, *(members or [])]:
            if isinstance(blob, dict) and blob.get("id") == person:
                phone = _blob_phone(blob) or phone
    if not phone:
        return {person}
    found = {person}
    more = (await session.execute(sql(
        "SELECT elder, members FROM family_rosters WHERE family_id NOT LIKE 'shadow:%'"
        " AND (elder::text LIKE :like OR members::text LIKE :like)"
    ), {"like": f"%{phone}%"})).all()
    for elder, members in more:
        for blob in [elder or {}, *(members or [])]:
            if isinstance(blob, dict) and _blob_phone(blob) == phone and blob.get("id"):
                found.add(str(blob["id"]))
    return found


async def stream_owner(session: AsyncSession, person: str) -> tuple[str, str] | None:
    """The one nudge stream for this person: the family where they last wrote (same phone counts as the same person).
    Two turns at the same moment pick the lower family id, so the choice does not flicker."""
    ids = sorted(await _person_ids(session, person))
    row = (await session.execute(
        select(Turn.family_id, Turn.thread_id).where(
            Turn.thread_id.in_(ids), Turn.role == "user", ~Turn.family_id.startswith("shadow:"),
        ).order_by(Turn.at.desc(), Turn.family_id.asc()).limit(1)
    )).first()
    if row is None:
        return None
    return str(row[0]), str(row[1])


async def asked_open(session: AsyncSession, person: str, now: datetime) -> tuple[set[str], list[str]]:
    """Fields and question texts we already asked today and they have not answered since, in any family."""
    start = _day_start(now)
    rows = await sent_today(session, person, now)
    wrote = list((await session.execute(select(Turn.at).where(
        Turn.thread_id == person, Turn.role == "user", Turn.at >= start, ~Turn.family_id.startswith("shadow:")))).scalars())
    fields: set[str] = set()
    texts: list[str] = []
    for r in rows:
        if r.replied_at or (r.sent_at and any(t >= r.sent_at for t in wrote)):
            continue
        if r.field:
            fields.add(r.field)
        if r.text and ("?" in r.text or "？" in r.text):
            texts.append(r.text)
    for text, at in (await session.execute(select(Turn.text, Turn.at).where(
            Turn.thread_id == person, Turn.role == "assistant", Turn.at >= start, ~Turn.family_id.startswith("shadow:")))).all():
        if text and ("?" in text or "？" in text) and not any(t >= at for t in wrote):
            texts.append(text)
    return fields, texts

async def _log(session: AsyncSession, p: Person, now: datetime, *, kind: str, status: str, reason: str = "", text: str = "",
               fld: str | None = None, about: str | None = None, run: int = 0, sent: bool = False,
               cost: float | None = None, context: str | None = None) -> NudgeLedger | None:
    """One ledger row (unique per person per minute: a second run in the same minute adds nothing)."""
    from app.care.nudges.compose import short_ref
    rid = uuid.uuid4()
    lane = "followup" if kind == "reminder_followup" else "regular"  # a follow-up and a regular nudge may share a minute
    if status == "skipped":
        lane += f":skip:{reason[:40]}"  # a skip never takes the minute from a nudge that may still go
    fld, about = short_ref(fld), short_ref(about)  # whatever the caller passed, it fits its column
    row = NudgeLedger(id=rid, family_id=p.family_id, person_id=p.id, kind=kind, slot_key=short_ref(f"{p.id}:{clock.ist(now).strftime('%Y-%m-%dT%H:%M')}:{lane}", 120) or "",
                      status=status, reason=reason[:160], ref=f"nudge:{rid}", text=text, field=fld, about_ref=about, planned_at=now,
                      sent_at=now if sent else None, unanswered_run=run, cost_inr=round(cost, 4) if cost else None,
                      context=context)
    try:
        async with session.begin_nested():
            session.add(row)
            await session.flush()
    except IntegrityError:
        return None
    return row


async def log_skip(session: AsyncSession, p: Person, now: datetime, reason: str, rows: list[NudgeLedger], kind: str = "regular",
                   cost: float | None = None) -> None:
    """A skip is logged at most once an hour per person and reason (an asleep person is checked every run); one that cost
    a model call is always logged, so the ledger holds every rupee."""
    if not cost and any(r.status == "skipped" and r.reason == reason and r.kind == kind and now - r.planned_at < SKIP_LOG_EVERY for r in rows):
        return
    await _log(session, p, now, kind=kind, status="skipped", reason=reason, cost=cost)

# ── replies ───────────────────────────────────────────────────────────────────────────────────────────────────────

SAVE_WITH = (("meal:", "log_food"), ("water:", "log_food"), ("goal:", "log_food"), ("activity:", "log_activity"), ("reading:", "log_vital"), ("weight", "log_vital"), ("doses", "log_dose"),
             ("stock:", "set_stock"), ("know:", "remember"), ("feelings", "log_event mood (and log_event symptom for a problem)"), ("sleep", "log_event"),
             ("reminder:", "log_activity or remember, whichever fits"))


async def pull_due(session: AsyncSession, family_id: str, person_id: str, when: datetime | None = None) -> None:
    """The backend saw a tap and no turn (a Done button, the opener). Look at this person on the next tick.
    A claim another task already holds is left alone, so this cannot cause a second send."""
    when = when or clock.now()
    stmt = pg_insert(NudgeDue).values(
        family_id=family_id, person_id=person_id, next_due=when, anchored_at=when, claimed_until=None,
    ).on_conflict_do_update(
        index_elements=["family_id", "person_id"],
        set_={"next_due": when, "anchored_at": when},
        where=or_(NudgeDue.claimed_until.is_(None), NudgeDue.claimed_until < when),
    )
    await session.execute(stmt)


async def note_reply(session: AsyncSession, family_id: str, person: str) -> None:
    """They wrote: every nudge sent to this person in the last 6 hours, in every family, is answered (the ladder resets)."""
    now = clock.now()
    # One statement: the reply in every family, the other streams wait, and this family's next look. A turn is counted.
    await session.execute(sql(
        "WITH answered AS ("
        " UPDATE nudge_ledger SET replied_at = :now"
        " WHERE person_id = :pid AND status = 'sent' AND family_id NOT LIKE 'shadow:%'"
        " AND replied_at IS NULL AND sent_at >= :since"
        "), held AS ("
        " UPDATE nudge_due SET next_due = :nxt, anchored_at = :now, claimed_until = NULL"
        " WHERE person_id = :pid AND family_id <> :fid AND family_id NOT LIKE 'shadow:%'"
        ") INSERT INTO nudge_due (family_id, person_id, next_due, anchored_at, claimed_until)"
        " VALUES (:fid, :pid, :nxt, :now, NULL)"
        " ON CONFLICT (family_id, person_id) DO UPDATE SET"
        " next_due = EXCLUDED.next_due, anchored_at = EXCLUDED.anchored_at, claimed_until = NULL"
    ), {"now": now, "fid": family_id, "pid": person, "since": now - REPLY_WINDOW, "nxt": now + CARED_EVERY})


async def reply_rates(session: AsyncSession, *, days: int = 7) -> dict:
    """How often a sent nudge was answered, over the last `days`, by kind, hour (IST) and person. No names."""
    since = clock.now() - timedelta(days=max(1, min(days, 30)))
    rows = (await session.execute(sql(
        "SELECT kind, person_id, EXTRACT(HOUR FROM sent_at AT TIME ZONE 'Asia/Kolkata')::int AS hour,"
        " COUNT(*)::int AS sent, COUNT(replied_at)::int AS replied"
        " FROM nudge_ledger WHERE status = 'sent' AND sent_at >= :since AND family_id NOT LIKE 'shadow:%'"
        " GROUP BY kind, person_id, hour"
    ), {"since": since})).all()

    def pack(sent: int, replied: int) -> dict:
        return {"sent": sent, "replied": replied, "rate": round(100 * replied / sent) if sent else None}

    by_kind: dict[str, list[int]] = {}
    by_hour: dict[int, list[int]] = {}
    by_person: dict[str, list[int]] = {}
    sent = replied = 0
    for kind, person, hour, n_sent, n_replied in rows:
        n_sent, n_replied = int(n_sent), int(n_replied)
        sent += n_sent
        replied += n_replied
        for bucket, key in ((by_kind, str(kind or "other")), (by_person, str(person))):
            got = bucket.setdefault(key, [0, 0])
            got[0] += n_sent
            got[1] += n_replied
        hour_n = int(hour)
        got_h = by_hour.setdefault(hour_n, [0, 0])
        got_h[0] += n_sent
        got_h[1] += n_replied
    people = [{"person": pid, **pack(n, r)} for pid, (n, r) in by_person.items()]
    people.sort(key=lambda row: (row["rate"] if row["rate"] is not None else 0, -row["sent"]))
    return {
        **pack(sent, replied),
        "days": days,
        "byKind": [{"kind": k, **pack(n, r)} for k, (n, r) in sorted(by_kind.items(), key=lambda kv: -kv[1][0])],
        "byHour": [{"hour": h, **pack(n, r)} for h, (n, r) in sorted(by_hour.items())],
        "people": people[:50],
    }


async def reply_hint(session: AsyncSession, family_id: str, person: str) -> str:
    """When their last message from Saheli was a nudge that asked for something, a line for the turn: save their answer."""
    last = next((t for t in reversed(await store.recent_turns(session, family_id, person, limit=6)) if t.role == "assistant"), None)
    nudge = ((last.meta or {}).get("nudge") or {}) if last else {}
    if not isinstance(nudge, dict):
        nudge = {}  # only a regular nudge carries its ask here; anything else is not one
    fld = nudge.get("field")
    if last is None or not fld or clock.now() - last.at > REPLY_WINDOW:
        return ""
    tool = next((t for prefix, t in SAVE_WITH if str(fld).startswith(prefix)), "the tool that fits (remember, log_activity, log_vital)")
    about = f" ({nudge['about_ref']})" if nudge.get("about_ref") else ""
    if str(fld).startswith("meal:"):
        slot = str(fld).split(":", 1)[1]
        whom = str(nudge.get("about_ref") or "").partition(":for:")[2]
        return (f"YOUR LAST MESSAGE TO THEM asked what {'person ' + whom if whom else 'they'} ate ({'today' if slot == 'day' else slot}): "
                f"\"{(last.text or '')[:160]}\". If they answered, pass it to log_food (items with quantity and unit as they said"
                + (f"; meal={slot}" if slot != "day" else "") + (f"; for_whom={whom}" if whom else "")
                + "). If the amount is unclear, the tool returns one question with options: ask only that. If it does not answer it, "
                "just answer what they wrote.")
    if str(fld).startswith("water:"):
        return (f"YOUR LAST MESSAGE TO THEM asked how much water: \"{(last.text or '')[:160]}\". If they answered, pass it to log_food "
                "(one item, name water). Unit glass when they said glasses, ml when they said millilitres. No number is one glass: do not ask. "
                "eaten_at is when they drank it, ISO in IST; leave it out only for just now. Each time is its own call. If the tool returns "
                "already=true, say nothing more about it. If they did not answer it, just answer what they wrote.")
    if str(fld).startswith("sleep:"):
        return (f"YOUR LAST MESSAGE TO THEM asked when they slept and woke: \"{(last.text or '')[:160]}\". If they answered, save it with "
                "log_event (kind sleep, the times as they said). If they did not answer it, just answer what they wrote.")
    if str(fld).startswith("stock:"):
        return (f"YOUR LAST MESSAGE TO THEM asked how many are left{about}: \"{(last.text or '')[:160]}\". If they gave a number "
                "on hand, call set_stock with that medicine and count (tablets or doses now). A new strip or bottle with no number: "
                "ask how many are in it, then set_stock. Do not guess a pack size. If they did not answer it, just answer what they wrote.")
    if str(fld).startswith("goal:"):
        return (f"YOUR LAST MESSAGE TO THEM mentioned a target they already have{about}: \"{(last.text or '')[:160]}\". If they said what they "
                "ate, pass it to log_food (items with quantity and unit as they said). Do not suggest a food they did not already eat. If it does "
                "not answer it, just answer what they wrote.")
    return (f"YOUR LAST MESSAGE TO THEM asked about {fld}{about}: \"{(last.text or '')[:160]}\". If what they wrote answers it, save it "
            f"with {tool}; if it does not, just answer what they wrote.")


# ── learning from how they took it (self-learning, 2026-10-13) ────────────────────────────────────────────────────


@dataclass
class Learned:
    stop: bool = False  # they asked Saheli to stop messaging on her own (30 days)
    fewer: bool = False  # they asked for fewer messages (14 days: twice the silence)
    unwelcome: set[str] = field(default_factory=set)  # questions (fields) or kinds of light talk they did not want asked
    rates: dict[str, float] = field(default_factory=dict)  # kind → their reply rate, pulled toward everyone's

    def weight(self, kind: str) -> float:
        """Options of a kind they answer score higher, of a kind they leave unanswered lower (±2)."""
        return round(4 * (self.rates[kind] - 0.5), 3) if kind in self.rates else 0.0


LIGHT_KINDS = ("reflect", "delight", "know", "early_bird")  # unwelcome light talk rests as a kind, not one question
_everyone: dict = {"at": 0.0, "rates": {}}


async def everyone_rates(session: AsyncSession, now: datetime) -> dict[str, float]:
    """Each kind's reply rate across every family (last 30 days), the starting point for someone new; cached an hour."""
    if _time.monotonic() - _everyone["at"] < 3600 and _everyone["rates"]:
        return _everyone["rates"]
    got = (await session.execute(
        select(NudgeLedger.kind, func.count(), func.count(NudgeLedger.replied_at))
        .where(NudgeLedger.status == "sent", NudgeLedger.sent_at >= now - LEARN_DAYS).group_by(NudgeLedger.kind)
    )).all()
    rates = {k: (r + 1) / (n + 2) for k, n, r in got}
    _everyone.update(at=_time.monotonic(), rates=rates)
    return rates


KIND_LABEL = {"follow_up": "follow-ups on what they said", "upcoming": "what's coming up", "habit": "their habits",
              "collect": "health check questions", "reflect": "how-was-your-day questions", "know": "get-to-know questions",
              "delight": "light chat about what they enjoy", "doses": "questions about unmarked medicines",
              "early_bird": "early-morning messages"}
LEARNED = "learned:nudges_"  # every nudge learning is a visible, undoable fact under this prefix (trust plan A5)
TOPICS_KEY = LEARNED + "topics"
TOPICS_MIN_SENT, TOPICS_MIN_KIND = 8, 3


def _until(f, now: datetime) -> bool:
    return (_iso((f.value or {}).get("until")) or now) > now


async def learned(session: AsyncSession, family_id: str, person: str, now: datetime) -> Learned:
    """What Saheli learned about their nudges, read from the visible learned:nudges_* facts only (the family sees each one
    on the dashboard and can undo it; an undone one is simply not here and is not learned again for 60 days)."""
    out = Learned()
    for f in await store.facts(session, family_id, person, domains=["learned"], statuses=("active",)):
        if not f.key.startswith(LEARNED):
            continue
        v = f.value or {}
        if f.key == LEARNED + "pause" and _until(f, now):
            out.stop = True
        elif f.key == LEARNED + "fewer" and _until(f, now):
            out.fewer = True
        elif f.key.startswith(LEARNED + "avoid_") and _until(f, now):
            out.unwelcome |= {x for x in (v.get("field"), v.get("kind")) if x}
        elif f.key == TOPICS_KEY:
            out.rates = {str(k): float(r) for k, r in (v.get("rates") or {}).items() if isinstance(r, (int, float))}
    return out


def _slug(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (x or "").lower()).strip("_")[:60] or "topic"


def _about(kind: str, fld: str | None) -> str:
    if kind in LIGHT_KINDS or not fld:
        return KIND_LABEL.get(kind, kind.replace("_", " "))
    head, _, rest = fld.partition(":")
    return {"know": f"{rest.replace('_', ' ')} (a get-to-know question)", "reading": f"{rest} readings",
            "activity": f"{rest.split(':')[0].replace('_', ' ')}"}.get(head, fld.replace(":", " ").replace("_", " "))


async def save_reaction(session: AsyncSession, row: NudgeLedger, now: datetime) -> str | None:
    """A stop, fewer or unwelcome reaction becomes a learned fact the family sees and can undo."""
    from app.care import outcomes_learn

    day = clock.ist(now).strftime("%d %b")
    if row.reaction == "stop":
        until = now + STOP_PAUSE
        key, value = LEARNED + "pause", {"until": until.isoformat(), "from_nudge": str(row.id)}
        text = f"Asked Saheli not to message on her own ({day}); she starts no conversations until {clock.ist(until):%d %b}."
    elif row.reaction == "fewer":
        until = now + FEWER_FOR
        key, value = LEARNED + "fewer", {"until": until.isoformat(), "from_nudge": str(row.id)}
        text = f"Finds Saheli's messages too frequent ({day}); she waits twice as long between them until {clock.ist(until):%d %b}."
    elif row.reaction == "unwelcome":
        until = now + UNWELCOME_PAUSE
        light = row.kind in LIGHT_KINDS or not row.field
        key = LEARNED + "avoid_" + _slug(row.kind if light else row.field or row.kind)
        value = {"until": until.isoformat(), "kind": row.kind if light else None, "field": None if light else row.field, "from_nudge": str(row.id)}
        text = f"Didn't want to be asked about {_about(row.kind, row.field)} ({day}); Saheli won't bring it up until {clock.ist(until):%d %b}."
    else:
        return None
    return await outcomes_learn.save_learned(session, family_id=row.family_id, subject_id=row.person_id, key=key, value=value, text=text, now=now)


async def save_quiet_why(session: AsyncSession, row: NudgeLedger, why: str, now: datetime) -> str | None:
    """A reason they gave (read by the model, never by a word list) becomes a fact the next nudge uses."""
    if why not in ("travelling", "unwell", "busy", "fewer"):
        return None
    from app.care import outcomes_learn

    subject = row.about_ref if row.kind == "quiet_note" and row.about_ref else row.person_id
    day = clock.ist(now)
    if why == "travelling":
        until = now + timedelta(days=7)
        key, value = LEARNED + "away", {"until": until.isoformat(), "from_nudge": str(row.id)}
        text = f"Away ({day:%d %b}); Saheli will not send a check-in until {clock.ist(until):%d %b}."
    elif why == "unwell":
        until = now + timedelta(days=1)
        key, value = LEARNED + "unwell", {"until": until.isoformat(), "from_nudge": str(row.id)}
        text = f"Not feeling well ({day:%d %b}); Saheli will not send a check-in until {clock.ist(until):%d %b}."
    elif why == "busy":
        old = await store.active_fact(session, row.family_id, subject, LEARNED + "quiet_days")
        raw = ((old.value or {}).get("days") or []) if old else []
        days = sorted({int(x) for x in raw if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())} | {day.weekday()})
        key, value = LEARNED + "quiet_days", {"days": days, "from_nudge": str(row.id)}
        text = f"Often busy on {day:%A}; Saheli skips the check-in that day."
    else:
        until = now + FEWER_FOR
        key, value = LEARNED + "fewer", {"until": until.isoformat(), "from_nudge": str(row.id)}
        text = f"Finds the messages too many ({day:%d %b}); she waits twice as long between them until {clock.ist(until):%d %b}."
    return await outcomes_learn.save_learned(session, family_id=row.family_id, subject_id=subject, key=key, value=value, text=text, now=now)


async def learn_topics(sessions: async_sessionmaker, *, now: datetime | None = None) -> dict:
    """Per person, the kinds of nudge they answer most and least (30 days, smoothed with every family's rate), saved as
    one visible learned fact when it is clear and has changed; the picker weights options from it."""
    from app.care import outcomes_learn

    now = now or clock.now()
    stats = {"people": 0, "saved": 0, "cooling": 0}
    async with sessions() as session:
        got = (await session.execute(
            select(NudgeLedger.family_id, NudgeLedger.person_id, NudgeLedger.kind, func.count(), func.count(NudgeLedger.replied_at))
            .where(NudgeLedger.status == "sent", NudgeLedger.sent_at >= now - LEARN_DAYS, NudgeLedger.kind != "reminder_followup",
                   ~NudgeLedger.family_id.startswith("shadow:"))
            .group_by(NudgeLedger.family_id, NudgeLedger.person_id, NudgeLedger.kind))).all()
        prior = await everyone_rates(session, now)
        by: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
        for fid, pid, kind, n, rep in got:
            by.setdefault((fid, pid), {})[kind] = (n, rep)
        for (fid, pid), kinds in by.items():
            if sum(n for n, _ in kinds.values()) < TOPICS_MIN_SENT:
                continue
            stats["people"] += 1
            rates = {k: round((rep + PRIOR * prior.get(k, 0.5)) / (n + PRIOR), 2) for k, (n, rep) in kinds.items() if n >= TOPICS_MIN_KIND}
            most = sorted((k for k, r in rates.items() if r >= 0.6), key=lambda k: -rates[k])
            least = sorted((k for k, r in rates.items() if r <= 0.2), key=lambda k: rates[k])
            if not most and not least:
                continue
            old = await store.active_fact(session, fid, pid, TOPICS_KEY)
            if old and (old.value or {}).get("answered") == most and (old.value or {}).get("rarely") == least:
                continue
            text = ("Answers " + ", ".join(KIND_LABEL.get(k, k) for k in most) + " most" if most else "") \
                + ("; " if most and least else "") \
                + ("rarely answers " + ", ".join(KIND_LABEL.get(k, k) for k in least) if least else "") \
                + ". Saheli leans on what they answer."
            res = await outcomes_learn.save_learned(session, family_id=fid, subject_id=pid, key=TOPICS_KEY, now=now, text=text[0].upper() + text[1:],
                                                    value={"rates": rates, "answered": most, "rarely": least,
                                                           "evidence": {k: {"sent": n, "answered": rep} for k, (n, rep) in kinds.items()}})
            stats["saved"] += res in ("created", "superseded")
            stats["cooling"] += res == "cooling"
        await session.commit()
    return stats


REACT = """You read how a person took one WhatsApp message that Saheli (a care companion) sent on her own, from what they
wrote back. Judge the meaning in any language, script or mix; when unsure, say neutral. Return only JSON
{"reaction": "welcome" | "neutral" | "unwelcome" | "fewer" | "stop", "why": "travelling" | "fewer" | "unwell" | "busy" | "none"}:
- welcome: they engaged gladly (answered warmly, shared more, thanked her)
- neutral: a short answer, or they moved on to something else; no sign of being bothered
- unwelcome: they did not want THIS question or topic (annoyed by it, asked why she asks, told her not to ask that)
- fewer: they find her messages too many or too frequent
- stop: they want her to stop messaging them on her own
- why: travelling if they are away or with someone else, unwell if they are not feeling well, busy if this sort of day
  is too full, fewer if the messages themselves are too many, none if they gave no such reason"""


async def read_reactions(sessions: async_sessionmaker, *, now: datetime | None = None, limit: int = 20) -> dict:
    """Each answered nudge gets how they took it, read by a model from their reply (never by word lists). Only the newest
    nudge they answered in one go is read; older unanswered ones answered by the same reply count as neutral."""
    from app.care.nudges.compose import _cost, parse
    from app.llm import router

    now = now or clock.now()
    stats: dict = {"read": 0, **{r: 0 for r in REACTIONS}}
    async with sessions() as session:
        rows = list((await session.execute(
            select(NudgeLedger).where(NudgeLedger.status == "sent", NudgeLedger.replied_at.is_not(None), NudgeLedger.reaction.is_(None),
                                      NudgeLedger.replied_at >= now - timedelta(days=2))
            .order_by(NudgeLedger.sent_at.desc()).limit(limit * 3)
        )).scalars())
        seen: set[tuple] = set()
        for row in rows:
            key = (row.family_id, row.person_id, row.replied_at)
            if key in seen:
                row.reaction = "neutral"
                continue
            seen.add(key)
            if stats["read"] >= limit:
                continue
            replies = [t for t in await store.recent_turns(session, row.family_id, row.person_id, limit=30)
                       if t.role == "user" and row.sent_at and t.at >= row.sent_at][:2]
            said = " / ".join((t.text or "").strip()[:400] for t in replies if (t.text or "").strip())
            if not said:
                row.reaction = "neutral"  # a tap, a sticker or a voice note with no words: answered, nothing more to read
                continue
            try:
                rep = await router.complete("judge_fast", system_stable=REACT, effort="low", max_tokens=300, timeout_s=30, messages=[
                    {"role": "user", "content": [{"type": "text", "text": f"SAHELI'S MESSAGE:\n{row.text}\n\nTHEIR REPLY:\n{said}"}]}])
            except Exception:  # noqa: BLE001 — read it next time
                continue
            parsed = parse(rep.text or "")
            got = str(parsed.get("reaction") or "neutral").strip().lower()
            why = str(parsed.get("why") or "none").strip().lower()
            row.reaction = got if got in REACTIONS else "neutral"
            row.cost_inr = round((row.cost_inr or 0.0) + _cost(rep), 4)
            stats["read"] += 1
            stats[row.reaction] += 1
            await save_reaction(session, row, now)
            await save_quiet_why(session, row, why, now)
        await session.commit()
    return stats


GRADE = """You grade one WhatsApp message that Saheli (a warm care companion) started on her own after a quiet hour, the way a
careful care manager would. You get what Saheli knew (the context she was given) and the message.

Fail it for any of these (list each in issues):
- made_up: it states a fact, number, name, time, event or reading that is not in the context.
- medical_advice: it tells them what to take, stop or change, or diagnoses anything.
- dose_reminder: it asks whether they took a medicine, or reminds them of a dose or its time. Their reminders do that.
  A doses option asking about an unmarked item already listed in the context is not this. A stock option asking only
  how many are left, naming that medicine, is not this.
- not_personal: it would fit anyone ("How are you today?") instead of naming something real from the context.
- wrong_language: not in the language, script or dialect the context says to write in.
- too_long: longer than the limit in the context, or more than one question.
- repeats: it repeats or contradicts the last messages or the last nudges in the context.
- tone: cold, pushy, alarming, lecturing, or wrong for who they are (an elder, a caregiver).
Return only JSON: {"pass": true|false, "issues": ["made_up", ...], "note": "one line"}"""

_review: dict = {"at": None}  # None: never graded in this process (monotonic time starts near 0 in a fresh container)


async def review(sessions: async_sessionmaker, *, now: datetime | None = None, limit: int = REVIEW_BATCH) -> dict:
    """Real nudges (sent, or written in shadow) are graded by a stronger model against the context they were written from.
    A made-up fact or medical advice is logged as an error (the owner reads production errors after every change); every
    grade stays on its row, so the good ones become examples to learn from."""
    from app.care.nudges.compose import _cost, parse
    from app.llm import router

    now = now or clock.now()
    stats: dict = {"graded": 0, "pass": 0, "issues": {}}
    async with sessions() as session:
        rows = list((await session.execute(
            select(NudgeLedger).where(NudgeLedger.status.in_(SENT), NudgeLedger.grade.is_(None), NudgeLedger.context.is_not(None),
                                      NudgeLedger.planned_at >= now - timedelta(hours=36))
            .order_by(NudgeLedger.planned_at.desc()).limit(limit)
        )).scalars())
        for row in rows:
            try:
                rep = await router.complete("judge", system_stable=GRADE, effort="low", max_tokens=1500, timeout_s=120, messages=[
                    {"role": "user", "content": [{"type": "text", "text": f"CONTEXT SAHELI HAD:\n{row.context}\n\nHER MESSAGE:\n{row.text}"}]}])
            except Exception:  # noqa: BLE001 — graded next time
                continue
            got = parse(rep.text or "")
            if "pass" not in got:
                continue
            issues = [str(i)[:40] for i in (got.get("issues") or [])][:8]
            ok = bool(got.get("pass")) and not issues
            row.grade = {"pass": ok, "issues": issues, "note": str(got.get("note") or "")[:300], "at": now.isoformat()}
            row.cost_inr = round((row.cost_inr or 0.0) + _cost(rep), 4)
            stats["graded"] += 1
            stats["pass"] += 1 if ok else 0
            for i in issues:
                stats["issues"][i] = stats["issues"].get(i, 0) + 1
            if {"made_up", "medical_advice"} & set(issues):
                logger.error("nudge review: %s in a %s nudge (status %s) family=%s ledger=%s note=%s", "/".join(sorted(set(issues))),
                             row.kind, row.status, row.family_id[:8], row.id, row.grade["note"][:160])
        await session.commit()
    return stats


async def learn(sessions: async_sessionmaker, *, now: datetime | None = None) -> dict:
    """After each nudge run: read how answered nudges were taken, and once an hour grade a few real nudges."""
    from app.care import nudges as api

    out: dict = {}
    try:
        out["reactions"] = await read_reactions(sessions, now=now)
    except Exception:  # noqa: BLE001
        logger.exception("nudge reactions failed")
    state = api._review
    if state["at"] is None or _time.monotonic() - state["at"] >= REVIEW_EVERY:
        state["at"] = _time.monotonic()
        try:
            out["review"] = await review(sessions, now=now)
        except Exception:  # noqa: BLE001
            logger.exception("nudge review failed")
        try:
            out["topics"] = await learn_topics(sessions, now=now)
        except Exception:  # noqa: BLE001
            logger.exception("nudge topic learning failed")
        try:
            out["meals"] = await learn_meals(sessions, now=now)
        except Exception:  # noqa: BLE001
            logger.exception("meal learning failed")
    return out


async def learn_meals(sessions: async_sessionmaker, *, now: datetime | None = None, families: set[str] | None = None) -> dict:
    """Hourly: each person's usual meal times and portions (the nutrition agent's learn_meal_habits) and the adaptive
    asking mode (meal_nudges.learn_mode), all as visible learned facts. families: only these (a simulation)."""
    from app.care import meal_nudges

    from app.care.nudges.planner import people

    now = now or clock.now()
    stats = {"modes": 0, "habits": 0}
    hk = meal_nudges.hooks()
    async with sessions() as session:
        q = select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:"))
        if families is not None:
            q = q.where(FamilyRoster.family_id.in_(sorted(families)))
        rosters = list((await session.execute(q)).scalars())
        for r in rosters:
            for p in await people(session, r):
                if hk is not None:
                    try:
                        await hk[0].learn_meal_habits(session, r.family_id, p.id)
                        stats["habits"] += 1
                    except Exception:  # noqa: BLE001 — one person's habits never stop the rest
                        logger.exception("meal habits failed family=%s", r.family_id)
                try:
                    await meal_nudges.learn_rhythms(session, r.family_id, p.id)
                    await meal_nudges.learn_reply_hours(session, r.family_id, p.id)
                except Exception:  # noqa: BLE001
                    logger.exception("habit rhythms failed family=%s", r.family_id)
                if await meal_nudges.learn_mode(session, r.family_id, p.id, await ledger_rows(session, r.family_id, p.id, limit=60), now):
                    stats["modes"] += 1
        await session.commit()
    return stats
