"""Regular nudges (Kavach NUdge's plan, Part C of the quality plan; founder 2026-10-13).

Saheli keeps the conversation alive: everyone who talks to her gets a personal message after each hour of chat silence
while they are awake (caregivers every 3 hours), even with no medicines, schedules or reminders. Each one is built from
their memory, what is coming up, what Saheli still needs to know and the last 5 messages either way, and it never
interrupts a live conversation. Five minutes after an unanswered reminder that is not about a medicine, one short data
question about it. Late risers get at most 2 soft "whenever you're up" messages in the morning.

Code decides who, when and the options (structured data only, never reading people's words); one worker-model call writes
the message; code checks it before it goes. The admin flag nudge.regular decides: off, nothing here (the old check-ins
run); shadow, planned, written and logged (status shadow) but nothing sent; on, sent, and the old check-ins stop.
Every planned nudge, skip, shadow and reply is a row in nudge_ledger.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import exists, func, or_, select, text as sql, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, FamilyRoster, NudgeDue, NudgeLedger, Turn
from app.core import clock

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
            "early_bird": 5, "reminder_followup": 8, "meal": 7, "water": 7, "sleep": 7, "goal": 5, "stock": 6}
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


# ── who and when ──────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Person:
    family_id: str
    id: str
    name: str
    role: str
    cared: bool  # the care record is about them (hourly), else a caregiver (every 3 h)
    elder: bool  # the care recipient (short messages)


@dataclass
class Clock:
    last_in: datetime | None  # their last message (engine turn, or the backend's last inbound)
    last_out: datetime | None  # Saheli's last message to them of any kind (a shadow nudge counts in shadow mode)
    run: int  # regular nudges in a row with no reply
    nudged_today: bool  # a regular nudge since their morning
    rows: list = field(default_factory=list)  # their recent ledger rows, newest first
    gate: dict = field(default_factory=dict)

    @property
    def last_activity(self) -> datetime | None:
        times = [t for t in (self.last_in, self.last_out) if t]
        return max(times) if times else None


def cadence(cared: bool, run: int) -> timedelta | None:
    """Silence needed before the next regular nudge; None after STOP_AFTER unanswered (only the morning one then)."""
    if run >= STOP_AFTER:
        return None
    base = CARED_EVERY if cared else CAREGIVER_EVERY
    return max(base, timedelta(minutes=LADDER_MINUTES.get(run, 180)))


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


async def people(session: AsyncSession, roster: FamilyRoster) -> list[Person]:
    from app.care.checkins import nudge_people

    elder_id = (roster.elder or {}).get("id")
    by_id = {m.get("id"): m for m in [roster.elder or {}, *(roster.members or [])] if m.get("id")}
    out = []
    for pid, cared in await nudge_people(session, roster):
        m = by_id.get(pid, {})
        out.append(Person(roster.family_id, pid, str(m.get("name") or ""), str(m.get("role") or ""), cared,
                          elder=pid == elder_id and "caregiver" not in str(m.get("role") or "").lower()))
    return out


async def morning_start(session: AsyncSession, family_id: str, person: str, now: datetime) -> datetime:
    """When their day started today: their own wake-up time when known, else 08:30."""
    from app.brain.wake import own_night

    night = await own_night(session, family_id, person)
    local = clock.ist(now)
    wake_min = night[1] if night else 8 * 60 + 30
    start = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=wake_min)
    return start if start <= local else start - timedelta(days=1)


async def engine_times(session: AsyncSession, family_id: str, person: str) -> tuple[datetime | None, datetime | None]:
    """(their last message, Saheli's last message) as the engine saw them: turns in their thread and reminders sent."""
    last_user = await session.scalar(select(func.max(Turn.at)).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user"))
    last_bot = await session.scalar(select(func.max(Turn.at)).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "assistant"))
    last_rem = await session.scalar(select(func.max(CareEvent.at)).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == person, CareEvent.kind == "reminder_sent"))
    outs = [t for t in (last_bot, last_rem) if t]
    return last_user, (max(outs) if outs else None)


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


async def read_clock(session: AsyncSession, family_id: str, person: str, gate: dict, now: datetime, mode: str) -> Clock:
    last_user, last_bot = await engine_times(session, family_id, person)
    last_in = max([t for t in (last_user, _iso(gate.get("last_inbound_at"))) if t], default=None)
    rows = await ledger_rows(session, family_id, person)
    counted = SENT if mode == "shadow" else ("sent",)
    regular = [r for r in rows if r.status in counted and r.kind != "reminder_followup"]
    outs = [t for t in (last_bot, _iso(gate.get("last_outbound_at"))) if t] + [_when(r) for r in rows if r.status in counted
                                                                                 and not (r.kind == "meal" and r.status == "shadow" and mode == "on")]
    last_out = max(outs, default=None)
    run = sum(1 for r in regular if last_in is None or _when(r) > last_in)
    morning = await morning_start(session, family_id, person, now)
    return Clock(last_in=last_in, last_out=last_out, run=run, nudged_today=any(_when(r) >= morning for r in regular), rows=rows, gate=gate)


async def early_bird_window(session: AsyncSession, family_id: str, person: str, now: datetime, rows: list[NudgeLedger]) -> bool:
    """A late riser (usual wake-up after 09:00): a soft message is allowed between max(08:30, bedtime + 5 h) and 30 minutes
    before their wake-up, at most 2 a night, 2 hours apart."""
    from app.brain.wake import own_night

    night = await own_night(session, family_id, person)
    if not night or night[1] <= 9 * 60:
        return False
    sleep, wake_min = night
    local = clock.ist(now)
    m = local.hour * 60 + local.minute
    asleep_for = (m - sleep) % (24 * 60)
    if m < 8 * 60 + 30 or asleep_for < 5 * 60 or m > wake_min - 30:
        return False
    since = now - timedelta(minutes=asleep_for)
    birds = [r for r in rows if r.kind == "early_bird" and r.status in SENT and _when(r) >= since]
    return len(birds) < EARLY_BIRD_MAX and not (birds and now - max(_when(r) for r in birds) < EARLY_BIRD_GAP)


async def woke_early(session: AsyncSession, family_id: str, person: str, now: datetime, last_user: datetime | None) -> bool:
    """They wrote this morning, before their usual wake-up but after the first 5 hours of their night."""
    from app.brain.wake import own_night

    night = await own_night(session, family_id, person)
    if not night or last_user is None:
        return False
    local = clock.ist(now)
    asleep_for = (local.hour * 60 + local.minute - night[0]) % (24 * 60)
    return last_user >= now - timedelta(minutes=asleep_for) + timedelta(hours=5)


@dataclass
class Due:
    person: Person
    clock: Clock
    early_bird: bool = False
    reason: str = ""  # why not, when not due
    mode: str | None = None  # off | shadow | on, once the backend was asked (None: not asked, a cheap skip)
    light: str = ""  # opener | morning | lunch | evening: a reduced line, still due


def light_slot(now: datetime, wake_min: int | None) -> str | None:
    """Their morning (from their own wake, else 08:30, for 90 minutes), around lunch, or early evening."""
    local = clock.ist(now)
    m = local.hour * 60 + local.minute
    wake = 8 * 60 + 30 if wake_min is None else wake_min
    if wake <= m < wake + 90:
        return "morning"
    if 12 * 60 + 30 <= m < 14 * 60 + 30:
        return "lunch"
    if 17 * 60 <= m < 18 * 60 + 30:
        return "evening"
    return None


_LIGHT_WHY = {
    "morning": "a warm good morning from something you know about them; one easy question; not a dose; not anything already asked today",
    "lunch": "a light hello around lunch, from something you know about them; one easy question; not a dose; not anything already asked today",
    "evening": "a warm early-evening line about how their day is going, from something you know; one easy question; not a dose; not anything already asked today",
    "opener": "the good morning that waits behind the notice: warm, from something you know, one easy question, not a dose, not anything already asked today",
}


async def due(session: AsyncSession, host, p: Person, now: datetime, mode: str | None, *, busy: bool) -> Due:
    """Is a regular nudge due for this person now, and if not, why (the plan's §1.3). The cheap checks (their night, a
    message within the base cadence, an order moving) come before any backend call."""
    from app.brain.wake import quiet_for

    rows = await ledger_rows(session, p.family_id, p.id)
    last_user, last_bot = await engine_times(session, p.family_id, p.id)
    empty = Clock(last_user, last_bot, 0, False, rows=rows)
    night, _ = await quiet_for(session, p.family_id, p.id, now)
    if night and await woke_early(session, p.family_id, p.id, now, last_user):
        night = False  # a late riser who answered this morning is awake: the hourly cadence runs from now (§1.5)
    bird = bool(night and p.cared and await early_bird_window(session, p.family_id, p.id, now, rows))
    if night and not bird:
        return Due(p, empty, reason="asleep")
    lr = await learned(session, p.family_id, p.id, now)
    if lr.stop:
        return Due(p, empty, reason="asked_stop", mode=mode or cached_mode())
    from app.care import outcomes_learn

    if await outcomes_learn.avoid_now(session, p.family_id, p.id, now=now):
        return Due(p, empty, reason="learned_hour", mode=mode or cached_mode())  # they rarely answer at this hour (learned, visible)
    base = (CARED_EVERY if p.cared else CAREGIVER_EVERY) * (2 if lr.fewer else 1)
    # a meal question written in its own shadow (nudge.meals) went to nobody: it never holds back the hour's real nudge
    engine_last = max([t for t in (last_user, last_bot, *(_when(r) for r in rows if r.status in SENT and not (r.kind in HABIT_SHADOW and r.status == "shadow")))
                       if t], default=None)
    if engine_last and now - engine_last < base:
        return Due(p, empty, reason="talking" if last_user and now - last_user < base else "cadence")
    if busy:
        return Due(p, empty, reason="busy", mode=mode or cached_mode())
    if mode is None and _mode["value"] == "off" and _time.monotonic() - _mode["at"] < MODE_TTL:
        return Due(p, empty, reason="off", mode="off")  # the flag was read a moment ago: no backend call per person
    try:
        gate = await host.call("nudge_gate", {"to": p.id}, family_id=p.family_id.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id)
    except Exception as exc:  # noqa: BLE001 — fail closed: no nudge without the backend's word
        return Due(p, empty, reason=f"gate_error:{type(exc).__name__}")
    gate = gate or {}
    m = mode or _remember_mode(gate)
    if m == "off":
        return Due(p, empty, reason="off", mode=m)
    ck = await read_clock(session, p.family_id, p.id, gate, now, m)
    if gate.get("paused"):
        return Due(p, ck, reason="paused", mode=m)
    if gate.get("active_flow"):
        return Due(p, ck, reason="busy", mode=m)
    if await sent_in_other_family(session, p.id, p.family_id, now - CARED_EVERY):
        return Due(p, ck, reason="other_family", mode=m)  # one stream per person, whichever family runs first
    # Their own message in the last 24 hours, by either side's record: a free message to someone who never wrote (or not
    # lately) fails at WhatsApp, even when the backend has no record of their chat yet and calls it open.
    closed = not gate.get("window_open", False) or ck.last_in is None or now - ck.last_in >= WINDOW
    if closed:
        if p.cared:
            from app.brain.wake import own_night

            night = await own_night(session, p.family_id, p.id)
            if light_slot(now, night[1] if night else None) == "morning":
                today = await sent_today(session, p.id, now)
                if not any(r.field == "hello" or (r.reason or "").startswith("waiting:") for r in today):
                    return Due(p, ck, light="opener", mode=m)  # one utility template; the greeting waits behind it
        return Due(p, ck, reason="window_closed", mode=m)
    if gate.get("reason"):
        return Due(p, ck, reason=str(gate["reason"])[:60], mode=m)
    if ck.run >= 2:  # the caregiver is being told: the person themselves still gets a light line, a caregiver does not
        try:
            esc = await host.call("escalation_status", {"to": p.id}, family_id=p.family_id.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id) or {}
        except Exception:  # noqa: BLE001 — the ladder's own rules still hold
            esc = {}
        if esc.get("state") in ("pending", "deferred", "sent"):
            if not p.cared:
                return Due(p, ck, reason="escalating", mode=m)
            from app.brain.wake import own_night

            night = await own_night(session, p.family_id, p.id)
            slot = light_slot(now, night[1] if night else None)
            today = await sent_today(session, p.id, now)
            wake = night[1] if night else None
            if slot and len(today) < 3 and not any(light_slot(_when(r), wake) == slot for r in today):
                return Due(p, ck, light=slot, mode=m)
            return Due(p, ck, reason="gentle", mode=m)
    last = ck.last_activity
    if bird:
        if last is not None and now - last < CARED_EVERY:
            return Due(p, ck, reason="talking", mode=m)
        return Due(p, ck, early_bird=True, mode=m)
    need = cadence(p.cared, ck.run)
    if need is not None and lr.fewer:
        need *= 2
    if need is None:
        if ck.nudged_today:
            return Due(p, ck, reason="ladder", mode=m)
        need = base  # the next morning's first nudge
    if last is not None and now - last < need:
        return Due(p, ck, reason="talking" if ck.last_in and now - ck.last_in < need else "cadence", mode=m)
    return Due(p, ck, mode=m)


# ── what to say: context and options (structured, no reading of people's words) ───────────────────────────────────


@dataclass
class Option:
    id: str
    kind: str
    why: str
    about_ref: str | None = None
    field: str | None = None
    score: float = 0.0


@dataclass
class Context:
    person: Person
    text: str  # what the composer sees
    known: str  # what any number, name or time in the message must come from
    recent: list[str]  # the last messages either way (newest last)
    saheli_recent: list[str]  # Saheli's own recent lines, newest last (for the opening and repeat checks)
    profile: dict | None
    options: list[Option]
    max_len: int
    expects_reply: bool = True
    ladder: int = 0
    cost: float = 0.0  # INR spent writing and checking it (every try), logged on its ledger row
    avoid: list[str] = field(default_factory=list)  # names they asked never to be called (trust plan A3)
    never: list[str] = field(default_factory=list)  # what the family said is wrong (rejected facts), never used or asked
    memory: str = ""  # what the no-invention check counts as known (not Saheli's own earlier lines)
    theirs: list[str] = field(default_factory=list)  # what they wrote lately
    sent_texts: list[str] = field(default_factory=list)  # lines sent to them in the last few days: do not send one again
    medicines: list[str] = field(default_factory=list)  # names on their record: a regular nudge may not ask about them
    today_asks: list[str] = field(default_factory=list)  # questions we already asked today and they have not answered


def _medicine_reminder(reminder: dict) -> bool:
    return str(reminder.get("aboutKey") or "").startswith("medicine:")


def medicine_names(facts, reminders: list[dict] | None = None) -> list[str]:
    """Names and doses saved on the care record, and the name of a reminder linked to a medicine. Not a list of drugs:
    only what this person's record already holds."""
    names: list[str] = []
    for f in facts:
        if getattr(f, "domain", None) != "medicine":
            continue
        value = f.value or {}
        for key in ("name", "dose"):
            got = str(value.get(key) or "").strip()
            if len(got) >= 3:
                names.append(got)
        slug = str(getattr(f, "key", "") or "").split(":", 1)[-1].replace("_", " ").strip()
        if len(slug) >= 4:
            names.append(slug)
    for r in reminders or []:
        if not _medicine_reminder(r):
            continue
        slug = str(r.get("aboutKey") or "").split(":", 1)[-1].replace("_", " ").strip()
        if len(slug) >= 4:
            names.append(slug)
    return list(dict.fromkeys(names))


def _mentions(text: str, name: str) -> bool:
    hay, needle = text.casefold(), name.casefold()
    if len(needle) < 3 or needle not in hay:
        return False
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return False
        before = hay[i - 1] if i else " "
        after = hay[i + len(needle)] if i + len(needle) < len(hay) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        start = i + 1


def near_dose(times, now: datetime, *, before: int = 45, after: int = 30) -> bool:
    """True when now is close to one of this medicine's dose times (45 minutes before, 30 after)."""
    m = clock.ist(now).hour * 60 + clock.ist(now).minute
    for t in times or []:
        hm = _hm(t)
        if hm is None:
            continue
        delta = (m - hm + 12 * 60) % (24 * 60) - 12 * 60
        if -before <= delta <= after:
            return True
    return False


async def stock_ask(session: AsyncSession, person: Person, facts: list, now: datetime) -> tuple[str, str, str, str] | None:
    """One medicine with no count, at a quiet hour that is not its dose time. One a day. Two sent asks with no reply
    and that medicine is left alone. The answer is a count, never a dose."""
    from app.care import features

    today = clock.ist_day(now)
    sent = (await session.execute(select(NudgeLedger.field, NudgeLedger.replied_at, NudgeLedger.sent_at).where(
        NudgeLedger.family_id == person.family_id, NudgeLedger.person_id == person.id, NudgeLedger.status == "sent",
        NudgeLedger.field.like("stock:%")))).all()
    if any(at and clock.ist_day(at) == today for _field, _replied, at in sent):
        return None
    silent: dict[str, int] = {}
    for asked, replied, _at in sent:
        if asked and replied is None:
            silent[asked] = silent.get(asked, 0) + 1
    known = {r["key"] for r in await features.stock(session, person.family_id, person.id) if r.get("stock") is not None}
    meds = [f for f in facts if getattr(f, "domain", None) == "medicine" and getattr(f, "status", "active") == "active"
            and not str(f.key).endswith(":none") and f.key not in known]
    for f in sorted(meds, key=lambda x: x.key):
        fld = f"stock:{f.key}"
        if silent.get(fld, 0) >= 2:
            continue
        times = [t for t in ((f.value or {}).get("times") or []) if isinstance(t, str)]
        if near_dose(times, now):
            continue
        name = (f.value or {}).get("name") or str(f.key).split(":", 1)[-1].replace("_", " ")
        why = (f"how many tablets of {name} are left in the strip: one question, only the count; "
               "do not mention the dose, the time, or whether they took it")
        return "stock", why, fld, fld
    return None


def _hm(v) -> int | None:
    try:
        h, m = str(v).strip()[:5].split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


def _said(t: str) -> str:
    return _norm(re.sub(r"(?i)^reminder:\s*", "", t or ""))


def _repeated(text: str, earlier: list[str]) -> str | None:
    """The earlier line this repeats, word for word or nearly, whatever they answered."""
    from app.brain import guards

    n = _said(text)
    if len(n) < 8:
        return None
    for old in earlier:
        o = _said(old)
        if not o:
            continue
        if n == o or guards.similar(text, old) >= 0.85:
            return old
    return None


async def last_messages(session: AsyncSession, family_id: str, person: str, gate: dict, n: int = 5) -> list[tuple[datetime, str, str]]:
    """The last n messages either way in their own thread: engine turns plus what the backend sent them itself (a reminder,
    the morning schedule), oldest first, a message both sides recorded only once."""
    turns = [t for t in await store.recent_turns(session, family_id, person, limit=20) if (t.text or "").strip() and t.text.strip().lower() != "none"]
    rows = [(t.at, "them" if t.role == "user" else "Saheli", t.text.strip()) for t in turns]
    for o in gate.get("last_outbound_preview") or []:
        at = _iso(o.get("at"))
        text = str(o.get("text") or "").strip()
        if not at or not text:
            continue
        key = _norm(text)[:60]
        if any(who == "Saheli" and _norm(t)[:60] == key and abs((at - when).total_seconds()) < 180 for when, who, t in rows):
            continue
        rows.append((at, "Saheli", text))
    rows.sort(key=lambda r: r[0])
    return rows[-n:]


async def build_context(session: AsyncSession, host, p: Person, ck: Clock, now: datetime, *, early_bird: bool = False,
                        only: list[Option] | None = None, extra: str = "", light: str = "") -> Context:
    from app.brain.loop import writing_profiles
    from app.care import activities, checkins, language, naming
    from app.care import mood as care_mood
    from app.care import rejected as rejected_facts

    fid = p.family_id
    facts = [f for f in await store.facts(session, fid, p.id) if f.status == "active"]
    by_domain: dict[str, list] = {}
    for f in facts:
        by_domain.setdefault(f.domain, []).append(f)
    speech = next((language.normalise(f.value) for f in by_domain.get("language", []) if isinstance(f.value, dict)), None)
    names = await naming.for_person(session, fid, p.id)
    wrong = await rejected_facts.rejected(session, fid, p.id, now=now)
    local = clock.ist(now)
    recent = await last_messages(session, fid, p.id, ck.gate)
    nudges = [r for r in ck.rows if r.status in SENT][:10]

    acts = await activities.status(session, fid, p.id)
    schedule: dict = {}
    if p.cared:
        try:
            schedule = await host.call("get_today_schedule", {}, family_id=fid.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id) or {}
        except Exception:  # noqa: BLE001
            schedule = {}
    try:
        rems = (await host.call("list_reminders", {}, family_id=fid.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id) or {}).get("reminders") or []
    except Exception:  # noqa: BLE001
        rems = []

    open_fields, today_asks = await asked_open(session, p.id, now)
    options = only if only is not None else await candidates(session, p, ck, now, facts=facts, acts=acts, schedule=schedule, reminders=rems,
                                                              early_bird=early_bird, open_fields=open_fields, light=light)
    options = sorted(options, key=lambda o: -o.score)[:5]

    m_now = local.hour * 60 + local.minute
    coming = []
    for i in (schedule.get("items") or []):
        t = _hm(i.get("time"))
        if t is not None and t > m_now and i.get("status") in ("upcoming", "due", None):
            coming.append(f"{i.get('time')} {i.get('title')} (its reminder goes on its own: do not remind or advise about it)")
    for r in rems:
        if _medicine_reminder(r):
            continue  # a dose reminder goes at its own time, never as "coming up" for a regular nudge
        for t in r.get("times") or []:
            if (_hm(t) or -1) > m_now:
                coming.append(f"{t} reminder: {r.get('text')}")
    for a in acts:
        if a["today"] == "upcoming" and a.get("start"):
            coming.append(f"{a['start']} {a['label']}" + (f" (streak {a['streak']})" if a["streak"] > 1 else ""))
    for f in by_domain.get("appointment", []):
        when = _iso((f.value or {}).get("when"))
        if when and now <= when <= now + timedelta(hours=24):
            coming.append(f"appointment: {f.text}")
    mood = await care_mood.last(session, fid, p.id, since=now - timedelta(hours=24))
    symptoms = await checkins._symptoms_since(session, fid, p.id, ck.last_out, now)
    known_facts = [f.text for f in facts if f.domain not in ("language", "learned", "medicine")][:24]
    notes = [n.body_md.strip()[:400] for n in (await store.notes(session, fid, [p.id]))[:2] if (n.body_md or "").strip()]
    max_len = MAX_LEN[p.elder]
    lines = [
        f"PERSON: {p.name or 'them'}" + (f" (call them {names['name']})" if names["name"] else "")
        + f"; {'the person cared for' if p.elder else ('caring for themselves' if p.cared else p.role or 'family member')}",
        *([f"NEVER CALL THEM: {', '.join(names['avoid'])}"] if names["avoid"] else []),
        *([rejected_facts.block(wrong)] if wrong else []),
        f"WRITE IN: {language.write_in(speech) if speech else 'the language and script they write to you in (their messages below); English if none'}",
        f"NOW: {local.strftime('%A %d %B, %H:%M')} IST",
        "LAST MESSAGES (oldest first):\n" + ("\n".join(f"  {clock.ist(at).strftime('%d %b %H:%M')} {who}: {t[:220]}" for at, who, t in recent) or "  (none yet)"),
        "WHAT YOU KNOW ABOUT THEM:\n" + ("\n".join(f"  - {t[:180]}" for t in known_facts) or "  (little yet)"),
        *(["NOTES:\n" + "\n".join(f"  - {n}" for n in notes)] if notes else []),
        "COMING UP (next 24 h):\n" + ("\n".join(f"  - {c}" for c in coming[:8]) or "  (nothing on record)"),
        *([f"MOOD TODAY: {mood.summary}"] if mood else []),
        *(["SINCE YOU LAST WROTE THEY MENTIONED: " + "; ".join(e.summary for e in symptoms[-3:])] if symptoms else []),
        "YOUR LAST NUDGES (do not repeat their kind or opening):\n"
        + ("\n".join(f"  - {r.kind}: {r.text[:120]}{' (answered)' if r.replied_at else ''}" for r in nudges) or "  (none)"),
        "OPTIONS (pick one id):\n" + "\n".join(f"  - {o.id} [{o.kind}] {o.why}" + (f" (field: {o.field})" if o.field else "") for o in options),
        f"LIMITS: at most {max_len} characters; at most one question"
        + ("; expects_reply false: no question they must answer, a soft 'whenever you're up' tone" if early_bird else "")
        + (f"; they have not answered your last {ck.run} messages: be lighter and easier to answer" if ck.run >= 2 else "")
        + ("; one easy question, not a dose, and not a question already asked today" if light else ""),
    ]
    if extra:
        lines.append(extra)
    profiles = await writing_profiles(session, fid, [{"id": p.id}])
    earlier = list(dict.fromkeys([*await sent_lines(session, fid, p.id, now - timedelta(days=3)),
                                  *[r.text for r in await sent_today(session, p.id, now - timedelta(days=3)) if r.text]]))
    saheli = [t for _, who, t in recent if who == "Saheli"] + [r.text for r in reversed(nudges) if r.text] + earlier[:8]
    known = "\n".join([*known_facts, *notes, *coming, *(t for _, _, t in recent), *(o.why for o in options), local.strftime("%A %d %B %H:%M")])
    memory = "\n".join([*known_facts, *notes, *coming, *(o.why for o in options), *([mood.summary] if mood else []),
                        *(e.summary for e in symptoms[-3:]), f"Now: {local.strftime('%A %d %B %H:%M')} IST"])
    return Context(person=p, text="\n".join(lines), known=known, recent=[t for _, _, t in recent], saheli_recent=saheli[-15:],
                   profile=profiles.get(p.id), options=options, max_len=max_len, expects_reply=not early_bird, ladder=ck.run,
                   avoid=names["avoid"], never=rejected_facts.terms(wrong), memory=memory,
                   theirs=[t for _, who, t in recent if who == "them"], sent_texts=earlier,
                   medicines=medicine_names(facts, rems), today_asks=today_asks)


async def candidates(session: AsyncSession, p: Person, ck: Clock, now: datetime, *, facts: list, acts: list[dict], schedule: dict,
                     reminders: list[dict], early_bird: bool = False, open_fields: set[str] | None = None, light: str = "") -> list[Option]:
    """What is worth writing about now (the plan's §3.4), each scored by how due it is and how much it helps, minus a
    penalty for the same kind among the last two nudges and for a field asked within its cooldown."""
    from app.care import checkins

    fid = p.family_id
    local = clock.ist(now)
    m_now = local.hour * 60 + local.minute
    rows = [r for r in ck.rows if r.status in SENT]
    last_kinds = [r.kind for r in rows[:2]]
    asked_field = {}
    for r in rows:
        if r.field and r.field not in asked_field:
            asked_field[r.field] = _when(r)
    for topic, at in (await checkins._asked(session, fid, p.id, now)).items():  # what the old check-ins asked counts too
        fld = {"follow_up": "follow_up"}.get(topic, topic.split(":")[0] if topic.startswith("doses:") else topic)
        if fld not in asked_field or at > asked_field[fld]:
            asked_field[fld] = at
    today = clock.ist_day(now)
    out: list[Option] = []
    lr = await learned(session, fid, p.id, now)

    def add(kind: str, why: str, *, about: str | None = None, fld: str | None = None, boost: float = 0.0, cooldown: timedelta | None = None):
        about, fld = short_ref(about), short_ref(fld)
        if fld and cooldown and fld in asked_field and now - asked_field[fld] < cooldown:
            return
        if about and any(r.about_ref == about and clock.ist_day(_when(r)) == today for r in rows):
            return
        if (fld and fld in lr.unwelcome) or kind in lr.unwelcome:
            return  # they did not want this asked: it rests for 30 days
        if fld and open_fields and fld in open_fields and not light:
            return  # asked today, in this family or another, and they have not answered
        score = PRIORITY.get(kind, 3) + boost - (3 if kind in last_kinds else 0) + lr.weight(kind)
        out.append(Option(id=f"o{len(out) + 1}", kind=kind, why=why, about_ref=about, field=fld, score=score))

    if light:
        day = clock.ist(now).strftime("%Y-%m-%d")
        add("follow_up", _LIGHT_WHY[light], fld="hello" if light == "opener" else f"gentle:{light}", about=f"light:{light}:{day}")
        return out
    if early_bird:
        plan = [f"{a['start']} {a['label']}" for a in acts if a["today"] == "upcoming" and a.get("start")][:3]
        add("early_bird", "a soft no-rush good-morning for when they wake" + (f"; today: {', '.join(plan)}" if plan else ""), boost=2)
        return out

    if p.cared:
        # Only items at least 2 hours past their time (the backend's own missed-dose follow-up has asked by then), once a
        # day, naming each item and its time (live 2026-10-13: an evening "your morning medications?" fit anyone).
        due_items = [i for i in checkins.unmarked_due(schedule, now) if (_hm(i.get("time")) or 24 * 60) <= m_now - DOSES_AFTER_MIN]
        if due_items:
            ids = ",".join(sorted(str(i.get("scheduleId")) for i in due_items))
            listed = "; ".join(f"{i.get('title')} at {i.get('time')}" for i in due_items[:6])
            add("doses", f"today's items past their time that nobody marked: {listed} (ask once whether they were done, naming "
                "each item and its time as listed here, never 'your medicines' in general)",
                about=f"doses:{ids}", fld="doses", boost=2, cooldown=timedelta(hours=20))
    # A medicine reminder is not an option: it goes at its own time (a nudge once asked if a dose had been taken).
    for r in reminders:
        if _medicine_reminder(r):
            continue
        for t in r.get("times") or []:
            mins = _hm(t)
            if mins is not None and 0 < mins - m_now <= 240:
                add("upcoming", f"their reminder at {t}: {r.get('text')}", about=f"reminder:{r.get('reminderId')}:{t}")
    for a in acts:
        if a["kind"] in ("habit", "quit"):
            if a["today"] == "upcoming" and a.get("start") and 0 < (_hm(a["start"]) or 0) - m_now <= 240:
                add("upcoming", f"{a['label']} at {a['start']} today" + (f", streak {a['streak']}" if a["streak"] > 1 else ""),
                    about=a["key"], fld=f"activity:{a['type']}:plan", cooldown=timedelta(hours=20))
            elif a["today"] in ("not_heard",) or a["kind"] == "quit":
                add("habit", f"{a['text']}: today {a['today'].replace('_', ' ')}" + (f", streak {a['streak']}" if a["streak"] > 1 else ""),
                    about=a["key"], fld=f"activity:{a['type']}:done", cooldown=timedelta(hours=20), boost=1 if a["streak"] > 1 else 0)
    for f in facts:
        if f.domain == "appointment":
            when = _iso((f.value or {}).get("when"))
            if when and timedelta(0) < when - now <= timedelta(hours=30):
                add("upcoming", f"appointment: {f.text}", about=f.key, boost=1)
            elif when and timedelta(0) < now - when <= timedelta(days=2):
                add("follow_up", f"how the appointment went: {f.text}", about=f.key, boost=1)
    if await checkins._told_lately(session, fid, p.id, now):
        add("follow_up", "something they told you in the last few days (a plan, a worry, a person, an event) that deserves a follow-up now; "
            "see LAST MESSAGES and WHAT YOU KNOW", fld="follow_up", cooldown=timedelta(hours=10))
    for lp in (await store.live_loops(session, fid, [p.id]))[:3]:
        if lp.wake_at is None or lp.wake_at - now > timedelta(hours=6):
            add("follow_up", f"an open thread: {lp.title}", about=f"loop:{lp.id}")
    if p.cared:
        symptoms = await checkins._symptoms_since(session, fid, p.id, None, now)
        if symptoms:
            add("collect", f"how {symptoms[-1].summary} is now", about=f"symptom:{symptoms[-1].id}", fld="feelings", boost=2)
        # A data question only with something of theirs to tie it to (a bare "how did you sleep?" fits anyone)
        anchor = personal_anchor(facts, acts, now)
        if anchor:
            add("collect", f"how they are feeling today, tied to {anchor}", fld="feelings", cooldown=COLLECT_EVERY["feelings"])
        # (no standalone sleep question: written from a bedtime it still came out as "how did you sleep?", which fits anyone)
        for kind in await checkins._readings_needed(session, fid, p.id):
            add("collect", f"today's {'blood pressure' if kind == 'bp' else 'sugar'} reading, asked as ONE question (\"what did it show "
                f"today?\"), if they checked it", fld=f"reading:{kind}", cooldown=COLLECT_EVERY["readings"])
        weight = await checkins._last_vital(session, fid, p.id, "weight")
        if weight is None or now - weight.at >= COLLECT_EVERY["weight"]:
            add("collect", "their weight, whenever they next check it (no pressure)", fld="weight", cooldown=COLLECT_EVERY["weight"])
        add("collect", "any doctor visit, test or new report this week", fld="reports", cooldown=COLLECT_EVERY["reports"])
    item = await checkins.next_to_know(session, fid, p.id, {f: at for f, at in asked_field.items() if f.startswith("know:")}, now)
    if item:
        add("know", f"one light question to learn {item[1]} (save the answer: {item[2]})", fld=f"know:{item[0]}", cooldown=checkins.KNOW_AGAIN)
    for name, start, end in SLOTS:
        if start <= m_now < end:
            done = any(r.kind == "reflect" and r.field == f"reflect:{name}" and clock.ist_day(_when(r)) == today for r in rows)
            if not done:
                what = {"morning": "their plan for the morning", "afternoon": "how their energy is this afternoon",
                        "evening": "the best part of their day"}[name]
                add("reflect", what, fld=f"reflect:{name}")
    likes = [f.text for f in facts if f.domain in ("preference", "dish", "occasion")][:4]
    if likes:
        add("delight", "something light from what they enjoy: " + "; ".join(likes), fld="delight", cooldown=timedelta(hours=8))
    from app.care import meal_nudges

    roster = await store.roster(session, fid)
    elder = (roster.elder or {}) if roster else None
    meals_mode = str(ck.gate.get("meals_mode") or os.getenv("NUDGE_MEALS") or "off").lower()
    for m in await meal_nudges.options(session, p, ck.rows, now, acts=acts, elder=elder, meals_mode=meals_mode):
        add(m["kind"], m["why"], about=m["about"], fld=m["fld"])
    g = await meal_nudges.goal_option(session, p, ck.rows, now, meals_mode=meals_mode)
    if g:
        add(g["kind"], g["why"], about=g["about"], fld=g["fld"])
    ask = await stock_ask(session, p, facts, now)
    if ask and not light and not early_bird:
        add(ask[0], ask[1], about=ask[2], fld=ask[3])
    return out


def short_ref(ref: str | None, limit: int = 160) -> str | None:
    """A reference that fits its ledger column (live 2026-10-13: "doses:<every unmarked schedule id>" overflowed 200
    characters and that nudge was lost). A long one keeps its start and a hash, so the same set compares equal."""
    if ref is None or len(ref) <= limit:
        return ref
    return f"{ref[:limit - 17]}#{hashlib.sha1(ref.encode()).hexdigest()[:16]}"


def personal_anchor(facts: list, acts: list[dict], now: datetime) -> str:
    """Something of theirs to tie a data question to: what they did in the last two days, else a condition they live with."""
    recent = [a for a in acts if a.get("last") and a["last"].get("done") and (_iso(a["last"].get("at")) or now) >= now - timedelta(days=2)]
    if recent:
        return f"their {recent[0]['label'].lower()} ({recent[0]['last']['summary'][:60]})"
    cond = next((f.text for f in facts if f.domain == "condition"), None)
    return f"what they live with: {cond[:80]}" if cond else ""


# ── write, check, send ────────────────────────────────────────────────────────────────────────────────────────────

COMPOSE = """You write ONE short WhatsApp message from Saheli, a warm care companion, to one person after a quiet spell. You keep
the conversation alive the way a caring friend who remembers things would.

You get who they are, the time, the last messages between you, what you know about them, what is coming up, your last
nudges, and OPTIONS (things worth writing about now). Pick ONE option and write the message for it.

Rules:
- Personal: it names something real from what you were given (their plan, habit, something they said, something coming
  up, the people or things they love). A message that would fit anyone ("How are you today?", "How did you sleep?") is not
  allowed: even a data question about feelings or a reading carries something of theirs, and the personal detail named in
  the option you pick goes into the message itself (yesterday's walk, their knee, what they told you).
- Short: one or two lines within the character limit. One question at most, and one means one (not two joined, not a
  question plus "tell me"). No lists, no lecture.
- The name: exactly the one in PERSON ("call them …"), written in their script, or no name at all; never a different
  word for them.
- Medicines: never remind them of a dose, a time or how to take it, and never ask whether they took one, even if they
  just set that reminder or it is in the last messages. Their reminders do that. Only a "doses" option may ask whether
  today's past items were done. A "stock" option asks only how many of that one medicine are left. Name it. Do not
  mention the dose, the time, or whether they took it.
- Fresh: a different opening and angle from the last messages and your last nudges. Never repeat a line they already got
  in the last few days, never ask a question already asked today, and never contradict the last messages; if the last
  chat ended on an open thread, pick it up.
- Their language, script and dialect exactly as WRITE IN says. English stays English. Never switch language. The name
  they like; emoji only if their messages use them. Everyday words, the way a person texts now: no "Reminder:" label,
  no poster line, no slogan.
- Safe: only what you were given (never invent a number, name, time, event or reading, and never put a day or time on
  something unless the context gives it: not "yesterday's gym" when it does not say yesterday); no medical advice; nothing
  alarming; nothing about another person that they would not want shared. Warm, never abrupt.
- Do not greet by name at the start, and no "Reminder:" label.
Return only JSON: {"option_id": "...", "text": "...", "field": "<the data your question asks for, from the option, or null>",
"expects_reply": true}

Good (generic examples; write in their language): "Gym at 7 today — legs or cardio? 💪" · "How did the call with the bank go?" ·
"Day 4 without sweets! What helped most today?"
Bad: "Hello! How are you today?" (not personal) · "Take your medicines, drink water and walk." (a list, a lecture) ·
"Your BP was 150 yesterday, that is dangerous." (invented, alarming)"""


def parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def compose(ctx: Context, problems: list[str] | None = None) -> dict:
    from app.llm import router

    text = ctx.text + (("\n\nYour last try had these problems; write it again without them:\n- " + "\n- ".join(problems)) if problems else "")
    reply = await router.complete("worker", system_stable=COMPOSE, effort="low", max_tokens=1500, timeout_s=40,
                                  messages=[{"role": "user", "content": [{"type": "text", "text": text}]}])
    ctx.cost += _cost(reply)
    return parse(reply.text or "")


def _cost(reply) -> float:
    from app.llm import spend

    return spend.cost_inr(reply.model or "", reply.usage or {})


def check(draft: dict, ctx: Context) -> list[str]:
    """What is wrong with a written nudge before it goes (the plan's §3.6); empty means it may go."""
    from app.brain import guards

    text = str(draft.get("text") or "").strip()
    if not text:
        return ["no message was written"]
    problems = []
    if draft.get("option_id") not in {o.id for o in ctx.options}:
        problems.append("option_id must be one of the OPTIONS ids")
    if len(text) > int(ctx.max_len * 1.15):
        problems.append(f"too long: {len(text)} characters, the limit is {ctx.max_len}")
    if text.count("?") + text.count("？") > 1:
        problems.append("more than one question")
    if _said(text) == "reminder" or text.lower().lstrip().startswith("reminder:"):
        problems.append('no "Reminder:" label; write it the way a person would text')
    again = _repeated(text, ctx.sent_texts)
    if again:
        problems.append(f"this line already went to them in the last few days: \"{again[:80]}\"")
    dup = guards.duplicate_message(text, [(t, False) for t in ctx.saheli_recent], near=0.6)
    if dup:
        problems.append(f"too close to an earlier message: \"{dup[:80]}\"")
    asked = guards.question_again(text, ctx.today_asks)
    if asked:
        problems.append(f"this question was already asked today: \"{asked[:80]}\"")
    first = " ".join(_norm(text).split()[:3])
    if first and any(" ".join(_norm(t).split()[:3]) == first for t in ctx.saheli_recent[-3:]):
        problems.append("opens the same way as one of your last messages")
    problems += [x for x in guards.language_problems(text, ctx.profile, who="they") if not x.startswith("do not mix scripts")]
    local = clock.ist()
    problems += guards.ungrounded(text, known=ctx.known, now_minutes=local.hour * 60 + local.minute)
    from app.care import naming

    for name in naming.uses_avoided(text, ctx.avoid):
        problems.append(f"they asked never to be called \"{name}\": use the name in PERSON or none")
    chosen = next((o for o in ctx.options if o.id == draft.get("option_id")), None)
    if (chosen is None or chosen.kind not in ("doses", "stock")) and any(_mentions(text, name) for name in ctx.medicines):
        problems.append("a regular nudge does not ask about a medicine or its dose; their reminders do that")
    low = text.lower()
    for term in ctx.never:
        if term.lower() in low:
            problems.append(f"\"{term}\" is something the family said is wrong: never use or ask about it")
    return problems


async def grounded(ctx: Context, text: str) -> list[str]:
    """The shared no-invention check (trust plan C4): every specific in the nudge must come from what Saheli knows or what
    they wrote, never from her own earlier lines; her own health advice is a problem too."""
    from app.brain import grounding

    return grounding.problems(await grounding.check(text, memory=ctx.memory, their_messages=ctx.theirs, tool_results=[]))


async def _log(session: AsyncSession, p: Person, now: datetime, *, kind: str, status: str, reason: str = "", text: str = "",
               fld: str | None = None, about: str | None = None, run: int = 0, sent: bool = False,
               cost: float | None = None, context: str | None = None) -> NudgeLedger | None:
    """One ledger row (unique per person per minute: a second run in the same minute adds nothing)."""
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


VERIFY = """You check one WhatsApp message Saheli is about to send, against the context she had. Reply with problems only for:
- invented: it states anything not in the context, including saying something already happened when the context does not
  say so (a walk "done" that is still ahead, "yesterday's" anything without a date), or a number, name or time not given;
- medical: it tells them what to take, stop or change, or how to take a medicine, or it asks whether they took a dose
  or reminds them of a dose or its time (their own reminders do that; only a doses option may ask about an unmarked item;
  a stock option asking only how many are left is not a dose reminder);
- two_questions: it asks more than one thing (two questions joined, or a question plus "tell me");
- not_personal: nothing in it is specific to this person (it would fit anyone).
Return only JSON: {"ok": true|false, "problems": ["invented: ...", ...]}"""


async def verify(ctx: Context, text: str) -> list[str]:
    """A fast second look by a model at what code cannot see (an event said to have happened, advice); fails open."""
    from app.llm import router

    try:
        rep = await router.complete("judge_fast", system_stable=VERIFY, effort="low", max_tokens=800, timeout_s=30, messages=[
            {"role": "user", "content": [{"type": "text", "text": f"CONTEXT:\n{ctx.text}\n\nMESSAGE:\n{text}"}]}])
    except Exception:  # noqa: BLE001 — no second look available: the code checks stand
        return []
    ctx.cost += _cost(rep)
    got = parse(rep.text or "")
    return [] if got.get("ok", True) else [str(x)[:160] for x in (got.get("problems") or ["the reviewer found a problem"])][:4]


async def write_and_check(ctx: Context) -> tuple[dict | None, list[str]]:
    problems: list[str] = []
    for _ in range(3):  # the first try and up to 2 more (the plan's §3.5)
        try:
            draft = await compose(ctx, problems or None)
        except Exception as exc:  # noqa: BLE001
            return None, [f"compose_failed:{type(exc).__name__}"]
        text = str(draft.get("text") or "")
        problems = check(draft, ctx) or await grounded(ctx, text) or await verify(ctx, text)
        if not problems:
            return draft, []
    return None, problems


async def send(sessions: async_sessionmaker, host, p: Person, ctx: Context, draft: dict, *, kind_default: str, mode: str,
               now: datetime, run: int, about: str | None = None, recheck=None, meals_mode: str | None = None) -> str:
    """Send (or, in shadow mode, only log) a checked nudge. The person's clock is read again under the family's lock right
    before it goes: if they wrote or anything else went out meanwhile, it is dropped (never mid-chat)."""
    from app.brain import guards
    from app.brain.loop import _lock_family
    from app.learn import scoring

    opt = next((o for o in ctx.options if o.id == draft.get("option_id")), ctx.options[0] if ctx.options else None)
    kind = opt.kind if opt else kind_default
    # The option's own field, never the writer's words for it (live 2026-10-13: "whether he ended up ordering pizza",
    # "cigarettes_smoked_today"): cooldowns, what was unwelcome and which tool saves the answer all key on it.
    fld = opt.field if opt else None
    about = about or (opt.about_ref if opt else None)
    text = str(draft.get("text") or "").strip()
    if kind in HABIT_SHADOW and meals_mode != "on":
        mode = "shadow"  # meal, water, sleep and goal asks share nudge.meals: in shadow they are written and logged, never sent
    async with sessions() as session:
        await _lock_family(session, p.family_id)
        if recheck is not None and not await recheck(session):
            await _log(session, p, now, kind=kind, status="skipped", reason="talking", text=text, fld=fld, about=about, run=run, cost=ctx.cost,
                       context=ctx.text)
            await session.commit()
            return "talking"
        greeted, mid = guards.opening_state(await store.recent_turns(session, p.family_id, p.id, limit=12), clock.now(), day_of=clock.ist_day)
        text = guards.tidy_opening(text, greeted_today=greeted, mid_conversation=mid)
        if mode != "on":
            await _log(session, p, now, kind=kind, status="shadow", text=text, fld=fld, about=about, run=run, cost=ctx.cost,
                       context=ctx.text)
            await session.commit()
            return "shadow"
        row = await _log(session, p, now, kind=kind, status="planned", text=text, fld=fld, about=about, run=run, cost=ctx.cost,
                       context=ctx.text)
        if row is None:
            await session.commit()
            return "duplicate"
        saved = ((ctx.profile or {}).get("saved") or {})
        opener = fld == "hello"
        payload: dict = {"to": p.id, "text": text, "kind": "nudge", "ref": row.ref}
        if saved.get("voice") == "always":
            payload["voice"] = True  # their setting: a voice note, not a typed line alone
        if opener:
            first = ((p.name or "there").split() or ["there"])[0] or "there"
            template: dict = {"purpose": "update", "params": [first]}
            if saved.get("language"):
                template["lang"] = saved["language"]
            payload["template"] = template  # kavach_saheli_update only; the text waits until they tap or reply
        try:
            # labelled for the backend's delivery ledger: a nudge counts toward "3 unanswered → the caregiver hears" (B2)
            res = await host.call("send_whatsapp", payload, family_id=p.family_id.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id)
        except Exception as exc:  # noqa: BLE001
            row.status, row.reason = "failed", f"send:{type(exc).__name__}"
            await session.commit()
            return "failed"
        row.wamid = str((res or {}).get("wamid") or "")[:120] or None
        if (res or {}).get("via") == "template" and opener:
            row.status, row.sent_at, row.reason = "sent", clock.now(), "waiting:update"
            await session.commit()
            return "waiting"  # the template went; the greeting is queued, not said in the chat yet
        if not (res or {}).get("delivered", True) or (res or {}).get("via") == "template":
            # not said yet: the chat was closed (only the opener template went, or it waits for their next message)
            row.status = "failed"
            row.reason = ("waiting:opener" if (res or {}).get("via") == "template" else str((res or {}).get("reason") or "not_delivered"))[:160]
            await session.commit()
            return "failed"
        turn_id = await store.add_turn(session, family_id=p.family_id, thread_id=p.id, role="assistant", text=text, message_ref=row.ref,
                                       meta={"proactive": True, "ref": row.ref, "nudge": {"id": str(row.id), "kind": kind, "field": fld, "about_ref": about}})
        await scoring.record(session, family_id=p.family_id, thread_id=p.id, turn_id=turn_id, kind="proactive", situation="regular_nudge",
                             speaker_role="system", lang="", user_text="", text=text, tools=[])
        row.status, row.sent_at = "sent", clock.now()
        await session.commit()
    return "sent"


# ── who is due this tick ──────────────────────────────────────────────────────────────────────────────────────────


def _next_minute(now: datetime, minute: int) -> datetime:
    """The next time the clock shows this minute of the day, in the same zone as now."""
    local = clock.ist(now)
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    target = day + timedelta(minutes=int(minute))
    if target <= local:
        target += timedelta(days=1)
    return target


async def _next_look(session: AsyncSession, person: Person, now: datetime, *, early_bird: bool) -> datetime:
    """The next moment this person might be awake enough for a nudge: their wake, or a late riser's early-bird start."""
    from app.brain.wake import own_night

    night = await own_night(session, person.family_id, person.id)
    minute = night[1] if night else 8 * 60 + 30
    if early_bird and person.cared and night and night[1] > 9 * 60:
        minute = max(8 * 60 + 30, (night[0] + 5 * 60) % (24 * 60))
    return _next_minute(now, minute)


async def next_check(session: AsyncSession, person: Person, d: Due, now: datetime) -> datetime:
    """When to read this person again. Sooner is safe (due() still decides); later must not skip a message they would get."""
    if not d.reason:
        step = cadence(person.cared, d.clock.run + 1) or timedelta(hours=12)
        return now + step
    if d.reason in ("cadence", "talking") and d.clock.last_activity:
        lr = await learned(session, person.family_id, person.id, now)
        need = cadence(person.cared, d.clock.run) or (CARED_EVERY if person.cared else CAREGIVER_EVERY)
        if lr.fewer:
            need *= 2
        nxt = d.clock.last_activity + need
        return nxt if nxt > now else now + timedelta(minutes=5)
    if d.reason == "asleep":
        birds = [_when(r) for r in d.clock.rows if r.kind == "early_bird" and r.status in SENT]
        if len(birds) < 2 and birds:
            nxt = max(birds) + EARLY_BIRD_GAP
            if nxt > now:
                return nxt  # the next soft morning message, still this night
        return await _next_look(session, person, now, early_bird=len(birds) < 2)
    if d.reason == "ladder":
        return await _next_look(session, person, now, early_bird=False)
    if d.reason == "window_closed":
        return await _next_look(session, person, now, early_bird=False)  # a tap pulls this forward; until then the next morning
    if d.reason == "asked_stop":
        return now + timedelta(minutes=5)  # an undo has no turn either; the fact is read on the next look
    if d.reason == "learned_hour":
        local = clock.ist(now).replace(minute=0, second=0, microsecond=0)
        return local + timedelta(hours=1)
    if d.reason == "other_family":
        return now + CARED_EVERY
    return now + timedelta(minutes=5)


async def mark_next(session: AsyncSession, family_id: str, person_id: str, when: datetime, *, at: datetime | None = None) -> None:
    """Remember the next look, and the moment we decided. A message from them after that is looked at straight away.
    Does not commit: the caller owns the transaction."""
    anchor = at or when
    stmt = pg_insert(NudgeDue).values(
        family_id=family_id, person_id=person_id, next_due=when, anchored_at=anchor, claimed_until=None,
    ).on_conflict_do_update(
        index_elements=["family_id", "person_id"],
        set_={"next_due": when, "anchored_at": anchor, "claimed_until": None},
    )
    await session.execute(stmt)


async def claim_person(session: AsyncSession, family_id: str, person_id: str, now: datetime, *, only_due: bool) -> bool:
    """Take this person until CLAIM_FOR from now. False when another task already holds them, or (only_due) their time
    has not come. Commits so the other task can see the claim."""
    until = now + CLAIM_FOR
    conds = [
        NudgeDue.family_id == family_id, NudgeDue.person_id == person_id,
        or_(NudgeDue.claimed_until.is_(None), NudgeDue.claimed_until < now),
    ]
    if only_due:
        spoke = exists(select(Turn.id).where(
            Turn.family_id == NudgeDue.family_id, Turn.thread_id == NudgeDue.person_id, Turn.role == "user",
            NudgeDue.anchored_at.is_not(None), Turn.at > NudgeDue.anchored_at,
        ))
        conds.append(or_(NudgeDue.next_due <= now, spoke))
    got = (await session.execute(
        update(NudgeDue).where(*conds).values(claimed_until=until).returning(NudgeDue.person_id)
    )).first()
    if got:
        await session.commit()
        return True
    held = await session.scalar(select(NudgeDue.person_id).where(
        NudgeDue.family_id == family_id, NudgeDue.person_id == person_id))
    if held:
        return False
    session.add(NudgeDue(family_id=family_id, person_id=person_id, next_due=now, claimed_until=until))
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return False
    return True


async def load_work(session: AsyncSession, now: datetime, families: set[str] | None) -> list[tuple[FamilyRoster, Person]]:
    """People whose next look has arrived, plus every person in a family that has never been indexed."""
    if families is not None and not families:
        return []
    spoke = exists(select(Turn.id).where(
        Turn.family_id == NudgeDue.family_id, Turn.thread_id == NudgeDue.person_id, Turn.role == "user",
        NudgeDue.anchored_at.is_not(None), Turn.at > NudgeDue.anchored_at,
    ))
    q = select(NudgeDue.family_id, NudgeDue.person_id).where(
        or_(NudgeDue.next_due <= now, spoke),
        or_(NudgeDue.claimed_until.is_(None), NudgeDue.claimed_until < now),
        ~NudgeDue.family_id.startswith("shadow:"),
    )
    fresh = select(FamilyRoster).where(
        ~FamilyRoster.family_id.startswith("shadow:"),
        ~exists(select(NudgeDue.family_id).where(NudgeDue.family_id == FamilyRoster.family_id)),
    )
    if families is not None:
        ids = sorted(families)
        q = q.where(NudgeDue.family_id.in_(ids))
        fresh = fresh.where(FamilyRoster.family_id.in_(ids))
    due_ids = (await session.execute(q)).all()
    fresh_rosters = list((await session.execute(fresh)).scalars())
    rosters = {r.family_id: r for r in fresh_rosters}
    fams = {fid for fid, _ in due_ids} - set(rosters)
    if fams:
        for r in (await session.execute(select(FamilyRoster).where(FamilyRoster.family_id.in_(sorted(fams))))).scalars():
            rosters[r.family_id] = r
    cached: dict[str, list[Person]] = {}

    async def persons(r: FamilyRoster) -> list[Person]:
        if r.family_id not in cached:
            cached[r.family_id] = await people(session, r)
        return cached[r.family_id]

    work: list[tuple[FamilyRoster, Person]] = []
    seen: set[tuple[str, str]] = set()
    for fid, pid in due_ids:
        r = rosters.get(fid)
        if r is None:
            continue
        match = next((p for p in await persons(r) if p.id == pid), None)
        if match and (fid, pid) not in seen:
            seen.add((fid, pid))
            work.append((r, match))
    for r in fresh_rosters:
        for p in await persons(r):
            key = (r.family_id, p.id)
            if key not in seen:
                seen.add(key)
                work.append((r, p))
    return work


# ── the run ───────────────────────────────────────────────────────────────────────────────────────────────────────


async def run(sessions: async_sessionmaker, host_for, *, now: datetime | None = None, mode: str | None = None,
              families: set[str] | None = None) -> dict:
    """Every few minutes: each person a regular nudge is due for gets one (or a shadow row). Returns counts by outcome.
    families: only these (a simulation runs each family on its own clock)."""
    from app.care import checkins
    from app.tasks import runtime

    now = now or clock.now()
    stats: dict = {"people": 0, "due": 0, "sent": 0, "shadow": 0, "skipped": {}, "mode": mode or env_mode() or cached_mode()}
    async with sessions() as session:
        work = await load_work(session, now, families)
    stats["people"] = len(work)
    if mode is None and not env_mode():
        sample = [r for r, _ in work]
        if not sample:
            async with sessions() as session:
                q = select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:"))
                if families is not None:
                    q = q.where(FamilyRoster.family_id.in_(sorted(families)))
                sample = list((await session.execute(q.limit(3))).scalars())
        known = await ensure_mode(host_for, sample)  # even when nobody is due, so the old check-ins stay off
        stats["mode"] = known or stats["mode"]
        if stats["mode"] == "off":
            return stats
    sem = asyncio.Semaphore(int(os.getenv("NUDGE_PARALLEL") or PARALLEL))
    person_locks: dict[str, asyncio.Lock] = {}

    async def one(r: FamilyRoster, p: Person) -> None:
        async with sem:
            lock = person_locks.setdefault(p.id, asyncio.Lock())
            async with lock:  # the same person in two families is one stream, even in one run
                host = host_for(r.family_id)
                claimed = False
                decided: Due | None = None
                outcome = "error"
                try:
                    async with sessions() as session:
                        if not await claim_person(session, r.family_id, p.id, now, only_due=True):
                            return  # another task already has this person, or their time was moved
                        claimed = True
                        live = await runtime.live_tasks(session, r.family_id)
                        d = await due(session, host, p, now, mode or env_mode(), busy=checkins.moving_for(live, p.id, now))
                        decided = d
                        if d.mode:
                            stats["mode"] = d.mode
                        if d.mode == "off":
                            outcome = "off"
                            return
                        if d.reason:
                            outcome = d.reason
                            stats["skipped"][d.reason] = stats["skipped"].get(d.reason, 0) + 1
                            if d.mode and d.mode != "off" and d.reason not in ("cadence", "asleep", "talking"):
                                await log_skip(session, p, now, d.reason, d.clock.rows)
                                await session.commit()
                            return
                        m = d.mode or "off"
                        stats["due"] += 1
                        ctx = await build_context(session, host, p, d.clock, now, early_bird=d.early_bird, light=d.light)
                    if not ctx.options:
                        outcome = "no_option"
                        stats["skipped"]["no_option"] = stats["skipped"].get("no_option", 0) + 1
                        return
                    draft, problems = await write_and_check(ctx)
                    if draft is None:
                        outcome = "guard"
                        async with sessions() as session:
                            await log_skip(session, p, now, "guard:" + (problems[0][:60] if problems else "?"), d.clock.rows, cost=ctx.cost)
                            await session.commit()
                        stats["skipped"]["guard"] = stats["skipped"].get("guard", 0) + 1
                        return
                    seen = d.clock.last_activity

                    async def still_quiet(session: AsyncSession) -> bool:
                        last_user, last_bot = await engine_times(session, p.family_id, p.id)
                        newest = max([t for t in (last_user, last_bot) if t], default=None)
                        return newest is None or seen is None or newest <= seen

                    out = await send(sessions, host, p, ctx, draft, kind_default="early_bird" if d.early_bird else "follow_up", mode=m,
                                     now=now, run=d.clock.run, recheck=still_quiet,
                                     meals_mode=str(d.clock.gate.get("meals_mode") or os.getenv("NUDGE_MEALS") or "off").lower())
                    outcome = out
                    stats[out] = stats.get(out, 0) + 1
                except Exception:  # noqa: BLE001 — one person's problem never stops the others
                    logger.exception("regular nudge failed family=%s person=%s", r.family_id, p.id)
                    stats["skipped"]["error"] = stats["skipped"].get("error", 0) + 1
                finally:
                    if claimed:
                        try:
                            async with sessions() as session:
                                if outcome == "shadow" and (decided is None or decided.mode != "shadow"):
                                    when = now + timedelta(minutes=5)  # a meals question logged in shadow did not use the hour
                                elif outcome in ("sent", "shadow", "waiting"):
                                    when = await next_check(session, p, decided or Due(p, Clock(None, None, 0, False)), now)
                                elif decided is not None and outcome == decided.reason:
                                    when = await next_check(session, p, decided, now)
                                else:
                                    when = now + timedelta(minutes=5)  # a guard miss or a failed send is tried again next tick
                                await mark_next(session, r.family_id, p.id, when, at=now)
                                await session.commit()
                        except Exception:  # noqa: BLE001 — the claim expires on its own
                            logger.exception("nudge next-due not saved family=%s person=%s", r.family_id, p.id)

    await asyncio.gather(*(one(r, p) for r, p in work))
    return stats


# ── reminder follow-ups (§1.6, §3.9) ──────────────────────────────────────────────────────────────────────────────


def _followup_field(act: dict | None, pay: dict) -> str:
    """Where the answer is saved. A water habit is structured (catalogue type water_intake), so the count goes to log_food."""
    if act and act.get("type") == "water_intake":
        return "water:count"
    if act:
        return f"activity:{act['type']}:{'count' if act.get('kind') == 'quit' else 'done'}"
    return f"reminder:{pay.get('reminderId')}"


async def reminder_followups(sessions: async_sessionmaker, host_for, *, now: datetime | None = None, mode: str | None = None) -> dict:
    """A reminder that is not about a medicine, delivered 5–10 minutes ago, with no message from them and nothing else sent
    since: one short data question tied to what the reminder is for (the model judges the match from the reminder and its
    linked activity). After 3 unanswered follow-ups in a row, only every other reminder gets one."""
    from app.brain.wake import quiet_for
    from app.care import activities, checkins
    from app.tasks import runtime

    now = now or clock.now()
    stats: dict = {"reminders": 0, "sent": 0, "shadow": 0, "skipped": {}}
    async with sessions() as session:
        evs = list((await session.execute(
            select(CareEvent).where(CareEvent.kind == "reminder_sent", CareEvent.at >= now - FOLLOWUP_AFTER[1], CareEvent.at <= now - FOLLOWUP_AFTER[0],
                                    ~CareEvent.family_id.startswith("shadow:"))
        )).scalars())
    sem = asyncio.Semaphore(int(os.getenv("NUDGE_PARALLEL") or PARALLEL))

    async def one_followup(ev):
        async with sem:
            await _one_followup(ev)

    async def _one_followup(ev):
        pay = ev.payload or {}
        if not pay.get("custom") or str(pay.get("aboutKey") or "").startswith("medicine:"):
            return  # medicine reminders keep their own missed-dose follow-up (D3)
        stats["reminders"] += 1
        about = f"reminder:{pay.get('reminderId')}:{pay.get('slot')}:{pay.get('dateKey')}"
        host = host_for(ev.family_id)
        try:
            async with sessions() as session:
                roster = await store.roster(session, ev.family_id)
                if roster is None:
                    return
                person = next((p for p in await people(session, roster) if p.id == ev.subject_id), None)
                if person is None:
                    return
                rows = await ledger_rows(session, ev.family_id, person.id)
                if any(r.kind == "reminder_followup" and r.about_ref == about for r in rows):
                    return
                m = mode or env_mode()
                try:
                    gate = await host.call("nudge_gate", {"to": person.id}, family_id=ev.family_id, subject_id=person.id, actor_id=person.id) or {}
                except Exception:  # noqa: BLE001
                    return
                m = m or _remember_mode(gate)
                if m == "off":
                    return
                reason = ""
                last_user, last_bot = await engine_times(session, ev.family_id, person.id)
                inbound = max([t for t in (last_user, _iso(gate.get("last_inbound_at"))) if t], default=None)
                outbound = [t for t in (last_bot, _iso(gate.get("last_outbound_at"))) if t]
                live = await runtime.live_tasks(session, ev.family_id)
                if (await learned(session, ev.family_id, person.id, now)).stop:
                    reason = "asked_stop"
                elif inbound and inbound > ev.at:
                    reason = "answered"
                elif any(t > ev.at + timedelta(seconds=90) for t in outbound) or any(r.status in SENT and _when(r) > ev.at for r in rows):
                    reason = "talking"
                elif gate.get("paused") or gate.get("active_flow") or checkins.moving_for(live, person.id, now):
                    reason = "busy"
                elif any(r.status == "sent" and r.kind != "reminder_followup" and r.sent_at and now - r.sent_at < CARED_EVERY for r in rows):
                    reason = "cadence"  # a nudge already went this hour: the follow-up does not make a second message
                elif await sent_in_other_family(session, person.id, ev.family_id, now - CARED_EVERY):
                    reason = "other_family"
                elif not gate.get("window_open", False) or inbound is None or now - inbound >= WINDOW:
                    reason = "window_closed"
                elif (await quiet_for(session, ev.family_id, person.id, now))[0]:
                    reason = "asleep"
                elif followup_ladder_skips(rows):
                    reason = "ladder"  # every other reminder after 3 unanswered follow-ups: this one skips, the next may go
                if reason:
                    stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
                    if reason in ("ladder", "busy", "window_closed", "asked_stop"):
                        await _log(session, person, now, kind="reminder_followup", status="skipped", reason=reason, about=about)
                        await session.commit()
                    return
                act = None
                key = str(pay.get("aboutKey") or "")
                if key.startswith("activity:"):
                    act = next((a for a in await activities.status(session, ev.family_id, person.id) if a["key"] == key), None)
                what = (pay.get("preview") or ev.summary or "").strip()
                why = (f"the reminder you sent {int((now - ev.at).total_seconds() // 60)} minutes ago: \"{what[:160]}\" ({ev.summary[:120]}). "
                       "Ask ONE short data question tied to what it is for (how many so far today, going or not and the plan, how "
                       "many glasses…), easy to answer with a number or a word")
                if act:
                    why += f"; it is about their {act['kind']} \"{act['label']}\"" + (f", streak {act['streak']}" if act["streak"] > 1 else "")
                opt = Option(id="o1", kind="reminder_followup", why=why, about_ref=about,
                             field=_followup_field(act, pay), score=PRIORITY["reminder_followup"])
                ck = await read_clock(session, ev.family_id, person.id, gate, now, m)
                ctx = await build_context(session, host, person, ck, now, only=[opt])
            async with sessions() as session:
                if not await claim_person(session, ev.family_id, person.id, now, only_due=False):
                    return  # the other engine task is already asking about this reminder
            draft, problems = await write_and_check(ctx)
            if draft is None:
                stats["skipped"]["guard"] = stats["skipped"].get("guard", 0) + 1
                async with sessions() as session:
                    await _log(session, person, now, kind="reminder_followup", status="skipped", reason="guard:" + (problems[0][:60] if problems else "?"),
                               about=about, cost=ctx.cost)
                    await session.execute(update(NudgeDue).where(
                        NudgeDue.family_id == ev.family_id, NudgeDue.person_id == person.id).values(claimed_until=None))
                    await session.commit()
                return
            seen = max([ev.at, *(t for t in outbound if t <= ev.at + timedelta(seconds=90))])

            async def still_quiet(session: AsyncSession, seen=seen, fid=ev.family_id, pid=person.id) -> bool:
                lu, lb = await engine_times(session, fid, pid)
                return not (lu and lu > seen) and not (lb and lb > seen + timedelta(seconds=90))

            out = await send(sessions, host, person, ctx, draft, kind_default="reminder_followup", mode=m, now=now, run=0, about=about,
                             recheck=still_quiet)
            stats[out] = stats.get(out, 0) + 1
            async with sessions() as session:
                if out in ("sent", "shadow", "waiting"):
                    step = CARED_EVERY if person.cared else CAREGIVER_EVERY
                    await mark_next(session, ev.family_id, person.id, now + step, at=now)
                else:
                    await session.execute(update(NudgeDue).where(
                        NudgeDue.family_id == ev.family_id, NudgeDue.person_id == person.id).values(claimed_until=None))
                await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("reminder follow-up failed family=%s", ev.family_id)
            stats["skipped"]["error"] = stats["skipped"].get("error", 0) + 1
    await asyncio.gather(*(one_followup(ev) for ev in evs))
    return stats


def followup_ladder_skips(rows: list[NudgeLedger]) -> bool:
    """After 3 unanswered follow-ups in a row, only every other reminder gets one (rows newest first)."""
    fups = [r for r in rows if r.kind == "reminder_followup" and (r.status in SENT or r.reason == "ladder")]
    streak = 0
    for r in fups:
        if r.status in SENT:
            if r.replied_at:
                break
            streak += 1
    return streak >= 3 and bool(fups) and fups[0].status in SENT


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
    """They wrote: every nudge sent to them in the last 6 hours with no reply yet is answered (the ladder resets)."""
    now = clock.now()
    # One statement: the reply and the next look. A turn is counted, and this must not add a second round trip.
    await session.execute(sql(
        "WITH answered AS ("
        " UPDATE nudge_ledger SET replied_at = :now"
        " WHERE family_id = :fid AND person_id = :pid AND status = 'sent'"
        " AND replied_at IS NULL AND sent_at >= :since"
        ") INSERT INTO nudge_due (family_id, person_id, next_due, anchored_at, claimed_until)"
        " VALUES (:fid, :pid, :nxt, :now, NULL)"
        " ON CONFLICT (family_id, person_id) DO UPDATE SET"
        " next_due = EXCLUDED.next_due, anchored_at = EXCLUDED.anchored_at, claimed_until = NULL"
    ), {"now": now, "fid": family_id, "pid": person, "since": now - REPLY_WINDOW, "nxt": now + CARED_EVERY})


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
{"reaction": "welcome" | "neutral" | "unwelcome" | "fewer" | "stop"}:
- welcome: they engaged gladly (answered warmly, shared more, thanked her)
- neutral: a short answer, or they moved on to something else; no sign of being bothered
- unwelcome: they did not want THIS question or topic (annoyed by it, asked why she asks, told her not to ask that)
- fewer: they find her messages too many or too frequent
- stop: they want her to stop messaging them on her own"""


async def read_reactions(sessions: async_sessionmaker, *, now: datetime | None = None, limit: int = 20) -> dict:
    """Each answered nudge gets how they took it, read by a model from their reply (never by word lists). Only the newest
    nudge they answered in one go is read; older unanswered ones answered by the same reply count as neutral."""
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
            got = str(parse(rep.text or "").get("reaction") or "neutral").strip().lower()
            row.reaction = got if got in REACTIONS else "neutral"
            row.cost_inr = round((row.cost_inr or 0.0) + _cost(rep), 4)
            stats["read"] += 1
            stats[row.reaction] += 1
            await save_reaction(session, row, now)
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
    out: dict = {}
    try:
        out["reactions"] = await read_reactions(sessions, now=now)
    except Exception:  # noqa: BLE001
        logger.exception("nudge reactions failed")
    if _review["at"] is None or _time.monotonic() - _review["at"] >= REVIEW_EVERY:
        _review["at"] = _time.monotonic()
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
