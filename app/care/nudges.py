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

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, FamilyRoster, NudgeLedger, Turn
from app.core import clock

logger = logging.getLogger(__name__)

CARED_EVERY = timedelta(minutes=60)  # people cared for and people caring for themselves
CAREGIVER_EVERY = timedelta(hours=3)  # caregivers and family members who wrote in the last 30 days (D2)
LADDER_MINUTES = {0: 60, 1: 60, 2: 120, 3: 180}  # unanswered in a row → silence before the next (D1)
STOP_AFTER = 4  # then only the next morning's first nudge, until they write
MAX_SENDS = 20  # per run
PARALLEL = 5  # people at a time (NUDGE_PARALLEL; tests share one connection and run one at a time)
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
            "early_bird": 5, "reminder_followup": 8}
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


async def read_clock(session: AsyncSession, family_id: str, person: str, gate: dict, now: datetime, mode: str) -> Clock:
    last_user, last_bot = await engine_times(session, family_id, person)
    last_in = max([t for t in (last_user, _iso(gate.get("last_inbound_at"))) if t], default=None)
    rows = await ledger_rows(session, family_id, person)
    counted = SENT if mode == "shadow" else ("sent",)
    regular = [r for r in rows if r.status in counted and r.kind != "reminder_followup"]
    outs = [t for t in (last_bot, _iso(gate.get("last_outbound_at"))) if t] + [_when(r) for r in rows if r.status in counted]
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
    base = (CARED_EVERY if p.cared else CAREGIVER_EVERY) * (2 if lr.fewer else 1)
    engine_last = max([t for t in (last_user, last_bot, *(_when(r) for r in rows if r.status in SENT)) if t], default=None)
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
    # Their own message in the last 24 hours, by either side's record: a free message to someone who never wrote (or not
    # lately) fails at WhatsApp, even when the backend has no record of their chat yet and calls it open.
    if not gate.get("window_open", False) or ck.last_in is None or now - ck.last_in >= WINDOW:
        return Due(p, ck, reason="window_closed", mode=m)
    if gate.get("reason"):
        return Due(p, ck, reason=str(gate["reason"])[:60], mode=m)
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


def _hm(v) -> int | None:
    try:
        h, m = str(v).strip()[:5].split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


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
                        only: list[Option] | None = None, extra: str = "") -> Context:
    from app.brain.loop import writing_profiles
    from app.care import activities, checkins, language
    from app.care import mood as care_mood

    fid = p.family_id
    facts = [f for f in await store.facts(session, fid, p.id) if f.status == "active"]
    by_domain: dict[str, list] = {}
    for f in facts:
        by_domain.setdefault(f.domain, []).append(f)
    speech = next((language.normalise(f.value) for f in by_domain.get("language", []) if isinstance(f.value, dict)), None)
    naming = next((f.value for f in by_domain.get("naming", []) if f.key == "naming:address_as"), {}) or {}
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

    options = only if only is not None else await candidates(session, p, ck, now, facts=facts, acts=acts, schedule=schedule, reminders=rems,
                                                              early_bird=early_bird)
    options = sorted(options, key=lambda o: -o.score)[:5]

    m_now = local.hour * 60 + local.minute
    coming = []
    for i in (schedule.get("items") or []):
        t = _hm(i.get("time"))
        if t is not None and t > m_now and i.get("status") in ("upcoming", "due", None):
            coming.append(f"{i.get('time')} {i.get('title')} (its reminder goes on its own: do not remind or advise about it)")
    for r in rems:
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
    known_facts = [f.text for f in facts if f.domain not in ("language",)][:24]
    notes = [n.body_md.strip()[:400] for n in (await store.notes(session, fid, [p.id]))[:2] if (n.body_md or "").strip()]
    max_len = MAX_LEN[p.elder]
    lines = [
        f"PERSON: {p.name or 'them'}" + (f" (call them {naming.get('name')})" if naming.get("name") else "")
        + f"; {'the person cared for' if p.elder else ('caring for themselves' if p.cared else p.role or 'family member')}",
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
        + (f"; they have not answered your last {ck.run} messages: be lighter and easier to answer" if ck.run >= 2 else ""),
    ]
    if extra:
        lines.append(extra)
    profiles = await writing_profiles(session, fid, [{"id": p.id}])
    saheli = [t for _, who, t in recent if who == "Saheli"] + [r.text for r in reversed(nudges) if r.text]
    known = "\n".join([*known_facts, *notes, *coming, *(t for _, _, t in recent), *(o.why for o in options), local.strftime("%A %d %B %H:%M")])
    return Context(person=p, text="\n".join(lines), known=known, recent=[t for _, _, t in recent], saheli_recent=saheli[-15:],
                   profile=profiles.get(p.id), options=options, max_len=max_len, expects_reply=not early_bird, ladder=ck.run)


async def candidates(session: AsyncSession, p: Person, ck: Clock, now: datetime, *, facts: list, acts: list[dict], schedule: dict,
                     reminders: list[dict], early_bird: bool = False) -> list[Option]:
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
        score = PRIORITY.get(kind, 3) + boost - (3 if kind in last_kinds else 0) + lr.weight(kind)
        out.append(Option(id=f"o{len(out) + 1}", kind=kind, why=why, about_ref=about, field=fld, score=score))

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
    # Today's medicines are not options: their own reminders go at their time (a nudge once added "with lunch" to a dose).
    for r in reminders:
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
- Medicines: never remind them of a dose, a time or how to take it, and never advise; their reminders do that. Only a
  "doses" option may ask whether today's past items were done.
- Fresh: a different opening and angle from the last messages and your last nudges. Never repeat or contradict the last
  messages; if the last chat ended on an open thread, pick it up.
- Their language, script and dialect exactly as WRITE IN says; the name they like; emoji only if their messages use them.
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
    dup = guards.duplicate_message(text, [(t, False) for t in ctx.saheli_recent], near=0.6)
    if dup:
        problems.append(f"too close to an earlier message: \"{dup[:80]}\"")
    first = " ".join(_norm(text).split()[:3])
    if first and any(" ".join(_norm(t).split()[:3]) == first for t in ctx.saheli_recent[-3:]):
        problems.append("opens the same way as one of your last messages")
    problems += [x for x in guards.language_problems(text, ctx.profile, who="they") if not x.startswith("do not mix scripts")]
    local = clock.ist()
    problems += guards.ungrounded(text, known=ctx.known, now_minutes=local.hour * 60 + local.minute)
    return problems


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
- medical: it tells them what to take, stop or change, or how to take a medicine;
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
        problems = check(draft, ctx) or await verify(ctx, str(draft.get("text") or ""))
        if not problems:
            return draft, []
    return None, problems


async def send(sessions: async_sessionmaker, host, p: Person, ctx: Context, draft: dict, *, kind_default: str, mode: str,
               now: datetime, run: int, about: str | None = None, recheck=None) -> str:
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
        try:
            res = await host.call("send_whatsapp", {"to": p.id, "text": text}, family_id=p.family_id.removeprefix("shadow:"),
                                  subject_id=p.id, actor_id=p.id)
        except Exception as exc:  # noqa: BLE001
            row.status, row.reason = "failed", f"send:{type(exc).__name__}"
            await session.commit()
            return "failed"
        if not (res or {}).get("delivered", True) or (res or {}).get("via") == "template":
            row.status, row.reason = "failed", str((res or {}).get("reason") or "not_delivered")[:160]
            await session.commit()
            return "failed"
        turn_id = await store.add_turn(session, family_id=p.family_id, thread_id=p.id, role="assistant", text=text, message_ref=row.ref,
                                       meta={"proactive": True, "ref": row.ref, "nudge": {"id": str(row.id), "kind": kind, "field": fld, "about_ref": about}})
        await scoring.record(session, family_id=p.family_id, thread_id=p.id, turn_id=turn_id, kind="proactive", situation="regular_nudge",
                             speaker_role="system", lang="", user_text="", text=text, tools=[])
        row.status, row.sent_at = "sent", clock.now()
        await session.commit()
    return "sent"


# ── the run ───────────────────────────────────────────────────────────────────────────────────────────────────────


async def run(sessions: async_sessionmaker, host_for, *, now: datetime | None = None, mode: str | None = None) -> dict:
    """Every few minutes: each person a regular nudge is due for gets one (or a shadow row). Returns counts by outcome."""
    from app.care import checkins
    from app.tasks import runtime

    now = now or clock.now()
    stats: dict = {"people": 0, "due": 0, "sent": 0, "shadow": 0, "skipped": {}, "mode": mode or env_mode() or cached_mode()}
    async with sessions() as session:
        rosters = list((await session.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")))).scalars())
    work: list[tuple[FamilyRoster, Person]] = []
    for r in rosters:
        async with sessions() as session:
            work += [(r, p) for p in await people(session, r)]
    stats["people"] = len(work)
    if mode is None and not env_mode():
        known = await ensure_mode(host_for, rosters)  # the old check-ins step aside on this, even when nobody is due
        stats["mode"] = known or stats["mode"]
    sem = asyncio.Semaphore(int(os.getenv("NUDGE_PARALLEL") or PARALLEL))
    sends = {"n": 0}

    async def one(r: FamilyRoster, p: Person) -> None:
        async with sem:
            host = host_for(r.family_id)
            try:
                async with sessions() as session:
                    live = await runtime.live_tasks(session, r.family_id)
                    d = await due(session, host, p, now, mode or env_mode(), busy=checkins.moving_for(live, p.id, now))
                    if d.mode:
                        stats["mode"] = d.mode
                    if d.mode == "off":
                        return
                    if d.reason:
                        stats["skipped"][d.reason] = stats["skipped"].get(d.reason, 0) + 1
                        if d.mode and d.mode != "off" and d.reason not in ("cadence", "asleep", "talking"):
                            await log_skip(session, p, now, d.reason, d.clock.rows)
                            await session.commit()
                        return
                    m = d.mode or "off"
                    if sends["n"] >= MAX_SENDS:
                        stats["skipped"]["run_full"] = stats["skipped"].get("run_full", 0) + 1
                        return
                    sends["n"] += 1
                    stats["due"] += 1
                    ctx = await build_context(session, host, p, d.clock, now, early_bird=d.early_bird)
                if not ctx.options:
                    stats["skipped"]["no_option"] = stats["skipped"].get("no_option", 0) + 1
                    return
                draft, problems = await write_and_check(ctx)
                if draft is None:
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
                                 now=now, run=d.clock.run, recheck=still_quiet)
                stats[out] = stats.get(out, 0) + 1
            except Exception:  # noqa: BLE001 — one person's problem never stops the others
                logger.exception("regular nudge failed family=%s person=%s", r.family_id, p.id)
                stats["skipped"]["error"] = stats["skipped"].get("error", 0) + 1

    await asyncio.gather(*(one(r, p) for r, p in work))
    return stats


# ── reminder follow-ups (§1.6, §3.9) ──────────────────────────────────────────────────────────────────────────────


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
    for ev in evs:
        pay = ev.payload or {}
        if not pay.get("custom") or str(pay.get("aboutKey") or "").startswith("medicine:"):
            continue  # medicine reminders keep their own missed-dose follow-up (D3)
        stats["reminders"] += 1
        about = f"reminder:{pay.get('reminderId')}:{pay.get('slot')}:{pay.get('dateKey')}"
        host = host_for(ev.family_id)
        try:
            async with sessions() as session:
                roster = await store.roster(session, ev.family_id)
                if roster is None:
                    continue
                person = next((p for p in await people(session, roster) if p.id == ev.subject_id), None)
                if person is None:
                    continue
                rows = await ledger_rows(session, ev.family_id, person.id)
                if any(r.kind == "reminder_followup" and r.about_ref == about for r in rows):
                    continue
                m = mode or env_mode()
                try:
                    gate = await host.call("nudge_gate", {"to": person.id}, family_id=ev.family_id, subject_id=person.id, actor_id=person.id) or {}
                except Exception:  # noqa: BLE001
                    continue
                m = m or _remember_mode(gate)
                if m == "off":
                    continue
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
                    continue
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
                             field=f"activity:{act['type']}:{'count' if act['kind'] == 'quit' else 'done'}" if act else f"reminder:{pay.get('reminderId')}",
                             score=PRIORITY["reminder_followup"])
                ck = await read_clock(session, ev.family_id, person.id, gate, now, m)
                ctx = await build_context(session, host, person, ck, now, only=[opt])
            draft, problems = await write_and_check(ctx)
            if draft is None:
                stats["skipped"]["guard"] = stats["skipped"].get("guard", 0) + 1
                async with sessions() as session:
                    await _log(session, person, now, kind="reminder_followup", status="skipped", reason="guard:" + (problems[0][:60] if problems else "?"),
                               about=about, cost=ctx.cost)
                    await session.commit()
                continue
            seen = max([ev.at, *(t for t in outbound if t <= ev.at + timedelta(seconds=90))])

            async def still_quiet(session: AsyncSession, seen=seen, fid=ev.family_id, pid=person.id) -> bool:
                lu, lb = await engine_times(session, fid, pid)
                return not (lu and lu > seen) and not (lb and lb > seen + timedelta(seconds=90))

            out = await send(sessions, host, person, ctx, draft, kind_default="reminder_followup", mode=m, now=now, run=0, about=about,
                             recheck=still_quiet)
            stats[out] = stats.get(out, 0) + 1
        except Exception:  # noqa: BLE001
            logger.exception("reminder follow-up failed family=%s", ev.family_id)
            stats["skipped"]["error"] = stats["skipped"].get("error", 0) + 1
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

SAVE_WITH = (("activity:", "log_activity"), ("reading:", "log_vital"), ("weight", "log_vital"), ("doses", "log_dose"),
             ("know:", "remember"), ("feelings", "log_event mood (and log_event symptom for a problem)"), ("sleep", "log_event"),
             ("reminder:", "log_activity or remember, whichever fits"))


async def note_reply(session: AsyncSession, family_id: str, person: str) -> None:
    """They wrote: every nudge sent to them in the last 6 hours with no reply yet is answered (the ladder resets)."""
    now = clock.now()
    await session.execute(update(NudgeLedger).where(
        NudgeLedger.family_id == family_id, NudgeLedger.person_id == person, NudgeLedger.status == "sent",
        NudgeLedger.replied_at.is_(None), NudgeLedger.sent_at >= now - REPLY_WINDOW).values(replied_at=now))


async def reply_hint(session: AsyncSession, family_id: str, person: str) -> str:
    """When their last message from Saheli was a nudge that asked for something, a line for the turn: save their answer."""
    last = next((t for t in reversed(await store.recent_turns(session, family_id, person, limit=6)) if t.role == "assistant"), None)
    nudge = ((last.meta or {}).get("nudge") or {}) if last else {}
    fld = nudge.get("field")
    if last is None or not fld or clock.now() - last.at > REPLY_WINDOW:
        return ""
    tool = next((t for prefix, t in SAVE_WITH if str(fld).startswith(prefix)), "the tool that fits (remember, log_activity, log_vital)")
    about = f" ({nudge['about_ref']})" if nudge.get("about_ref") else ""
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


async def learned(session: AsyncSession, family_id: str, person: str, now: datetime) -> Learned:
    """What their own replies taught: a stop or fewer they asked for, questions they did not want, and per kind how often
    they answer (a Beta mean with everyone's rate as the prior, so one unanswered nudge does not bury a kind)."""
    rows = (await session.execute(
        select(NudgeLedger.kind, NudgeLedger.field, NudgeLedger.reaction, NudgeLedger.replied_at, NudgeLedger.sent_at)
        .where(NudgeLedger.family_id == family_id, NudgeLedger.person_id == person, NudgeLedger.status == "sent",
               NudgeLedger.sent_at >= now - max(LEARN_DAYS, STOP_PAUSE, UNWELCOME_PAUSE))
    )).all()
    out = Learned()
    newest_reacted = max((r for r in rows if r.reaction and r.replied_at), key=lambda r: r.replied_at, default=None)
    if newest_reacted is not None:  # only their latest word counts: a later warm reply lifts an earlier stop
        if newest_reacted.reaction == "stop" and now - newest_reacted.replied_at < STOP_PAUSE:
            out.stop = True
    out.fewer = any(r.reaction == "fewer" and r.replied_at and now - r.replied_at < FEWER_FOR for r in rows)
    for r in rows:
        if r.reaction == "unwelcome" and r.replied_at and now - r.replied_at < UNWELCOME_PAUSE:
            if r.field:
                out.unwelcome.add(r.field)
            if r.kind in LIGHT_KINDS or not r.field:
                out.unwelcome.add(r.kind)
    if rows:
        prior = await everyone_rates(session, now)
        mine: dict[str, list[int]] = {}
        for r in rows:
            if r.sent_at and now - r.sent_at < LEARN_DAYS and r.kind != "reminder_followup":
                n_r = mine.setdefault(r.kind, [0, 0])
                n_r[0] += 1
                n_r[1] += 1 if r.replied_at else 0
        for kind, (n, rep) in mine.items():
            g = prior.get(kind, 0.5)
            out.rates[kind] = (rep + PRIOR * g) / (n + PRIOR)
    return out


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
        await session.commit()
    return stats


GRADE = """You grade one WhatsApp message that Saheli (a warm care companion) started on her own after a quiet hour, the way a
careful care manager would. You get what Saheli knew (the context she was given) and the message.

Fail it for any of these (list each in issues):
- made_up: it states a fact, number, name, time, event or reading that is not in the context.
- medical_advice: it tells them what to take, stop or change, or diagnoses anything.
- not_personal: it would fit anyone ("How are you today?") instead of naming something real from the context.
- wrong_language: not in the language, script or dialect the context says to write in.
- too_long: longer than the limit in the context, or more than one question.
- repeats: it repeats or contradicts the last messages or the last nudges in the context.
- tone: cold, pushy, alarming, lecturing, or wrong for who they are (an elder, a caregiver).
Return only JSON: {"pass": true|false, "issues": ["made_up", ...], "note": "one line"}"""

_review: dict = {"at": 0.0}


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
    if _time.monotonic() - _review["at"] >= REVIEW_EVERY:
        _review["at"] = _time.monotonic()
        try:
            out["review"] = await review(sessions, now=now)
        except Exception:  # noqa: BLE001
            logger.exception("nudge review failed")
    return out
