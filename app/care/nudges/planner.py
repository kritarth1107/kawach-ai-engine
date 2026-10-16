"""Who is due, and the run that asks each of them once."""

from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import delete, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, FamilyRoster, NudgeDue, NudgeLedger, Turn
from app.core import clock

logger = logging.getLogger(__name__)

from app.care.nudges.ledger import (
    CARED_EVERY, CAREGIVER_EVERY, CLAIM_FOR, EARLY_BIRD_GAP, EARLY_BIRD_MAX, HABIT_SHADOW,
    LADDER_MINUTES, MODE_TTL, PARALLEL,
    SENT,
    STOP_AFTER, WINDOW,
    _iso, _mode, _optional_chat, _remember_mode,
    _when, cached_mode, ensure_mode, env_mode, learned, ledger_rows,
    log_skip, sent_in_other_family, sent_today, stream_owner,
)

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
    chat_run: int = 0  # unanswered optional chat (get-to-know, reflect, delight, small talk); medicine and the schedule are not in this
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


async def read_clock(session: AsyncSession, family_id: str, person: str, gate: dict, now: datetime, mode: str) -> Clock:
    last_user, last_bot = await engine_times(session, family_id, person)
    last_in = max([t for t in (last_user, _iso(gate.get("last_inbound_at"))) if t], default=None)
    # A reply in any family they belong to resets the ladder. The chat lives in one family; the sends may not.
    any_in = await session.scalar(select(func.max(Turn.at)).where(
        Turn.thread_id == person, Turn.role == "user", ~Turn.family_id.startswith("shadow:")))
    if any_in and (last_in is None or any_in > last_in):
        last_in = any_in
    rows = await ledger_rows(session, family_id, person)
    counted = SENT if mode == "shadow" else ("sent",)
    regular = [r for r in rows if r.status in counted and r.kind not in ("reminder_followup", "warm")]
    cross = list((await session.execute(
        select(NudgeLedger).where(
            NudgeLedger.person_id == person, NudgeLedger.status.in_(counted),
            NudgeLedger.kind.notin_(("reminder_followup", "warm")), ~NudgeLedger.family_id.startswith("shadow:"),
        ).order_by(NudgeLedger.planned_at.desc()).limit(30)
    )).scalars())
    outs = [t for t in (last_bot, _iso(gate.get("last_outbound_at"))) if t] + [_when(r) for r in rows if r.status in counted
                                                                                 and not (r.kind == "meal" and r.status == "shadow" and mode == "on")]
    last_out = max(outs, default=None)
    run = sum(1 for r in cross if not r.replied_at and (last_in is None or _when(r) > last_in))
    chat_run = sum(1 for r in cross if _optional_chat(r) and not r.replied_at and (last_in is None or _when(r) > last_in))
    morning = await morning_start(session, family_id, person, now)
    return Clock(last_in=last_in, last_out=last_out, run=run, nudged_today=any(_when(r) >= morning for r in regular),
                 chat_run=chat_run, rows=rows, gate=gate)


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
    hold_chat: bool = False  # optional chat waits; medicine, the schedule and appointments do not
    quiet_why: str = ""  # a reason already on the record: no check-in until it passes
    notice: tuple[str, str, str] | None = None  # a closed-chat appointment: what, time, fact key (key may end with :later)


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
    "warm": "a warm line from something they said or you know about them; no question at all, and nothing they have to answer",
}


def _on_trip(value: dict, now: datetime) -> bool:
    departs = _iso(value.get("departs") or value.get("depart"))
    arrives = _iso(value.get("arrives") or value.get("arrive"))
    if departs and arrives:
        return departs <= now <= arrives
    if departs and departs <= now <= departs + timedelta(days=2):
        return True
    return bool(arrives and now <= arrives and arrives - now <= timedelta(days=1))


async def quiet_reason(session: AsyncSession, p: Person, ck: Clock, now: datetime) -> str:
    """Why they have gone quiet, from the record only. Empty means we do not know, so one easy check-in is allowed."""
    last = max((r for r in ck.rows if r.status in ("sent", "failed") and _optional_chat(r)), key=_when, default=None)
    if last is not None:
        reason = last.reason or ""
        if last.status == "failed" or reason.startswith(("not_delivered", "waiting:", "unread")):
            return "not_reached"
    for f in await store.facts(session, p.family_id, p.id, domains=["trip", "learned"], statuses=("active",)):
        if f.domain == "trip" and _on_trip(f.value or {}, now):
            return "travelling"
        if f.key == "learned:nudges_away" and (_iso((f.value or {}).get("until")) or now) > now:
            return "travelling"
        if f.key == "learned:nudges_unwell" and (_iso((f.value or {}).get("until")) or now) > now:
            return "unwell"
        if f.key == "learned:nudges_quiet_days" and clock.ist(now).weekday() in ((f.value or {}).get("days") or []):
            return "quiet_day"
    from app.care import checkins

    if await checkins._symptoms_since(session, p.family_id, p.id, now - timedelta(hours=24), now):
        return "unwell"
    return ""


def _clock_label(at: datetime) -> str:
    return clock.ist(at).strftime("%I:%M %p").lstrip("0")


async def _appointment_sent(session: AsyncSession, family_id: str, person_id: str, key: str, now: datetime) -> bool:
    since = now - timedelta(hours=20)
    rows = list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == person_id,
                                CareEvent.kind == "appointment_notice", CareEvent.at >= since)
    )).scalars())
    return any((r.payload or {}).get("key") == key for r in rows)


async def appointment_notice(session: AsyncSession, p: Person, now: datetime) -> tuple[str, str, str] | None:
    """An appointment whose time was in the last 15 minutes, or a Later tap whose 15 minutes are up. None when already sent."""
    facts = await store.facts(session, p.family_id, p.id, domains=["appointment"])
    by_key = {f.key: f for f in facts}
    for f in facts:
        when = _iso((f.value or {}).get("when"))
        if when is None:
            continue
        late = now - when
        if timedelta(0) <= late <= timedelta(minutes=15) and not await _appointment_sent(session, p.family_id, p.id, f.key, now):
            return ((f.text or "your appointment").strip()[:60], _clock_label(when), f.key)
    since = now - timedelta(minutes=20)
    later = list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == p.family_id, CareEvent.subject_id == p.id,
                                CareEvent.kind == "appointment_later", CareEvent.at >= since)
    )).scalars())
    for ev in later:
        send_at = _iso((ev.payload or {}).get("send_at"))
        ref = str((ev.payload or {}).get("ref") or "")
        if not send_at or not ref or abs(now - send_at) > timedelta(minutes=2):
            continue
        if await _appointment_sent(session, p.family_id, p.id, f"{ref}:later", now):
            continue
        fact = by_key.get(ref)
        when = _iso((fact.value or {}).get("when")) if fact else None
        label = ((fact.text if fact else None) or "your appointment").strip()[:60]
        return (label, _clock_label(when) if when else _clock_label(now), f"{ref}:later")
    return None


async def send_closed_notice(session: AsyncSession, host, p: Person, notice: tuple[str, str, str], now: datetime) -> bool:
    """The reminder template for an appointment, at its time, when the chat is closed. The buttons record the tap."""
    what, when, key = notice
    ref = key[:-6] if key.endswith(":later") else key
    langs = await store.facts(session, p.family_id, p.id, domains=["language"])
    code = str((langs[0].value or {}).get("language") or "") if langs else ""
    if code == "english":
        code = "en"
    template: dict = {"purpose": "reminder", "params": [what[:60], when[:20]],
                      "quickReplyPayloads": [f"done:appointment:{ref}", f"later:appointment:{ref}", f"skip:appointment:{ref}"]}
    if code:
        template["lang"] = code
    res = await host.call(
        "send_whatsapp",
        {"to": p.id, "text": f"{what} is due at {when}.", "kind": "reminder", "ref": f"appointment:{ref}"[:160],
         "template": template},
        family_id=p.family_id.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id,
    )
    if not (res or {}).get("delivered"):
        return False
    await store.record_event(
        session, family_id=p.family_id, subject_id=p.id, kind="appointment_notice",
        summary=f"{what} at {when}"[:500], payload={"key": key, "ref": ref}, actor_id=p.id, at=now,
    )
    return True


async def record_appointment_tap(session: AsyncSession, family_id: str, person_id: str, action: str, ref: str, *, now: datetime | None = None) -> str:
    """A button on an appointment reminder. Later asks for one more send, 15 minutes on."""
    now = now or clock.now()
    if action not in ("done", "later", "skip") or not ref:
        return "unknown"
    if action == "later":
        prior = list((await session.execute(
            select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == person_id,
                                    CareEvent.kind == "appointment_later", CareEvent.at >= now - timedelta(hours=20))
        )).scalars())
        if any((r.payload or {}).get("ref") == ref for r in prior):
            return "already"  # one Later per appointment
        await store.record_event(
            session, family_id=family_id, subject_id=person_id, kind="appointment_later",
            summary="later", payload={"ref": ref, "send_at": (now + timedelta(minutes=15)).isoformat()}, actor_id=person_id, at=now,
        )
        return "snoozed"
    await store.record_event(
        session, family_id=family_id, subject_id=person_id, kind="appointment_done" if action == "done" else "appointment_skipped",
        summary=action, payload={"ref": ref, "button": action}, actor_id=person_id, at=now,
    )
    return "noted"


async def due(session: AsyncSession, host, p: Person, now: datetime, mode: str | None, *, busy: bool) -> Due:
    """Is a regular nudge due for this person now, and if not, why (the plan's §1.3). The cheap checks (their night, a
    message within the base cadence, an order moving) come before any backend call."""
    from app.brain.wake import quiet_for

    owner = await stream_owner(session, p.id)
    if owner and (owner[0] != p.family_id or owner[1] != p.id):
        return Due(p, Clock(None, None, 0, False), reason="other_family")  # their chat owns the only stream
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
    # a closed chat (nothing from them in 24 hours) is not "talking": a template we sent this morning must not hold an
    # appointment that is due now
    inbound_stale = last_user is None or now - last_user >= WINDOW
    if engine_last and now - engine_last < base and not inbound_stale:
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
                    notice = await appointment_notice(session, p, now)
                    return Due(p, ck, light="opener", mode=m, notice=notice)  # one utility template; the greeting waits behind it
        notice = await appointment_notice(session, p, now)
        return Due(p, ck, reason="window_closed", mode=m, notice=notice)
    if gate.get("reason"):
        return Due(p, ck, reason=str(gate["reason"])[:60], mode=m)
    hold_chat = False
    quiet_why = ""
    if ck.chat_run >= 2:  # two unanswered optional chats: look at the record before another line
        try:
            esc = await host.call("escalation_status", {"to": p.id}, family_id=p.family_id.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id) or {}
        except Exception:  # noqa: BLE001 — the ladder's own rules still hold
            esc = {}
        if esc.get("state") in ("pending", "deferred", "sent") and not p.cared:
            return Due(p, ck, reason="escalating", mode=m)
        quiet_why = await quiet_reason(session, p, ck, now)
        hold_chat = True
    last = ck.last_activity
    if bird:
        if last is not None and now - last < CARED_EVERY:
            return Due(p, ck, reason="talking", mode=m)
        return Due(p, ck, early_bird=True, mode=m)
    if hold_chat:
        need = CARED_EVERY if p.cared else CAREGIVER_EVERY  # quiet chat does not slow medicine or the schedule
    else:
        need = cadence(p.cared, ck.run)
    if need is not None and lr.fewer:
        need *= 2  # fewer messages, never more, when they have asked for less
    if need is None:
        if ck.nudged_today:
            return Due(p, ck, reason="ladder", mode=m)
        need = base  # the next morning's first nudge
    if last is not None and now - last < need:
        return Due(p, ck, reason="talking" if ck.last_in and now - ck.last_in < need else "cadence", mode=m)
    return Due(p, ck, mode=m, hold_chat=hold_chat, quiet_why=quiet_why)


async def caregiver_note_due(session: AsyncSession, p: Person, d: Due, now: datetime) -> bool:
    """An elder quiet for a day, and we do not already know why: one gentle note to the caregiver."""
    if not p.elder or d.reason == "asleep":
        return False
    ck = d.clock
    if ck.chat_run < 2 or ck.last_in is None or now - ck.last_in < timedelta(hours=24):
        return False
    why = d.quiet_why or await quiet_reason(session, p, ck, now)
    return not why


async def note_caregiver(sessions: async_sessionmaker, host, roster: FamilyRoster, elder: Person, now: datetime) -> None:
    """One soft line to each caregiver, once a day. Not an alarm, and not a second escalation."""
    from app.care.nudges.compose import Option, build_context, send, write_and_check
    async with sessions() as session:
        carers = [q for q in await people(session, roster) if q.id != elder.id and "caregiver" in (q.role or "").lower()]
    for carer in carers:
        async with sessions() as session:
            if any(r.kind == "quiet_note" for r in await sent_today(session, carer.id, now)):
                continue
            name = elder.name or "they"
            opt = Option("o1", "quiet_note",
                         f"not an alarm: one gentle line that {name} has been quiet since yesterday; the caregiver can say why",
                         about_ref=elder.id, field="quiet_note")
            ctx = await build_context(session, host, carer, Clock(None, None, 0, False), now, only=[opt],
                                      extra="This is not an alarm. One short line, and they can tell you why whenever they like.")
        draft, _problems = await write_and_check(ctx)
        if draft is None:
            continue
        await send(sessions, host, carer, ctx, draft, kind_default="quiet_note", mode="on", now=now, run=0, about=elder.id)


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


async def next_appointment_at(session: AsyncSession, person: Person, now: datetime) -> datetime | None:
    """The next appointment time still to send, or a Later tap's send time, so a closed chat is looked at then."""
    facts = await store.facts(session, person.family_id, person.id, domains=["appointment"])
    times: list[datetime] = []
    for f in facts:
        when = _iso((f.value or {}).get("when"))
        if when and when > now and not await _appointment_sent(session, person.family_id, person.id, f.key, now):
            times.append(when)
    later = list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == person.family_id, CareEvent.subject_id == person.id,
                                CareEvent.kind == "appointment_later", CareEvent.at >= now - timedelta(minutes=20))
    )).scalars())
    for ev in later:
        send_at = _iso((ev.payload or {}).get("send_at"))
        ref = str((ev.payload or {}).get("ref") or "")
        if send_at and send_at > now and ref and not await _appointment_sent(session, person.family_id, person.id, f"{ref}:later", now):
            times.append(send_at)
    return min(times) if times else None


async def _before_appointment(session: AsyncSession, person: Person, now: datetime, when: datetime) -> datetime:
    """Look again at an appointment's time when that is sooner than the usual next look."""
    appt = await next_appointment_at(session, person, now)
    return appt if appt and now < appt < when else when


async def next_check(session: AsyncSession, person: Person, d: Due, now: datetime) -> datetime:
    """When to read this person again. Sooner is safe (due() still decides); later must not skip a message they would get."""
    if not d.reason:
        if d.hold_chat or d.clock.chat_run >= 1:
            step = CARED_EVERY if person.cared else CAREGIVER_EVERY  # the next look still sees a meal or a dose
        else:
            step = cadence(person.cared, d.clock.run + 1) or timedelta(hours=12)
        return await _before_appointment(session, person, now, now + step)
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
        morning = await _next_look(session, person, now, early_bird=False)  # a tap pulls this forward; until then the next morning
        return await _before_appointment(session, person, now, morning)
    if d.reason == "asked_stop":
        return now + timedelta(minutes=5)  # an undo has no turn either; the fact is read on the next look
    if d.reason == "learned_hour":
        local = clock.ist(now).replace(minute=0, second=0, microsecond=0)
        return local + timedelta(hours=1)
    if d.reason == "other_family":
        return now + CARED_EVERY
    if d.reason == "quiet":
        return await _next_look(session, person, now, early_bird=False)  # the next warm line is tomorrow
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
            owner = await stream_owner(session, p.id)
            if owner and (owner[0] != r.family_id or owner[1] != p.id):
                continue  # their chat is with another family: that stream is the only one
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
    from app.care import checkins, nudges as api
    from app.care.nudges.compose import build_context, send
    from app.tasks import runtime

    now = now or clock.now()
    stats: dict = {"people": 0, "due": 0, "sent": 0, "shadow": 0, "skipped": {}, "mode": mode or env_mode() or cached_mode()}
    async with sessions() as session:
        work = await api.load_work(session, now, families)
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
                        if await caregiver_note_due(session, p, d, now):
                            await note_caregiver(sessions, host, r, p, now)
                        notice_ok = await send_closed_notice(session, host, p, d.notice, now) if d.notice else True
                        if d.mode:
                            stats["mode"] = d.mode
                        if d.mode == "off":
                            outcome = "off"
                            return
                        if d.reason:
                            if d.notice and not notice_ok:
                                outcome = "notice_retry"  # the template did not go: look again in a few minutes, not tomorrow
                                return
                            outcome = d.reason
                            stats["skipped"][d.reason] = stats["skipped"].get(d.reason, 0) + 1
                            if d.mode and d.mode != "off" and d.reason not in ("cadence", "asleep", "talking"):
                                await log_skip(session, p, now, d.reason, d.clock.rows)
                                await session.commit()
                            return
                        if d.notice and notice_ok:
                            await session.commit()  # the appointment template is its own send; the morning line uses the next session
                        m = d.mode or "off"
                        stats["due"] += 1
                        ctx = await build_context(session, host, p, d.clock, now, early_bird=d.early_bird, light=d.light, hold_chat=d.hold_chat, quiet_why=d.quiet_why)
                    if not ctx.options:
                        outcome = "no_option"
                        stats["skipped"]["no_option"] = stats["skipped"].get("no_option", 0) + 1
                        return
                    draft, problems = await api.write_and_check(ctx)
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
                                dropped = False
                                if outcome == "other_family":
                                    owner = await stream_owner(session, p.id)
                                    if owner and owner[0] != r.family_id:
                                        await session.execute(delete(NudgeDue).where(
                                            NudgeDue.family_id == r.family_id, NudgeDue.person_id == p.id))
                                        await session.execute(delete(NudgeLedger).where(
                                            NudgeLedger.family_id == r.family_id, NudgeLedger.person_id == p.id,
                                            NudgeLedger.status == "planned"))
                                        await session.commit()
                                        dropped = True
                                if not dropped:
                                    if outcome == "shadow" and (decided is None or decided.mode != "shadow"):
                                        when = now + timedelta(minutes=5)  # a meals question logged in shadow did not use the hour
                                    elif outcome in ("sent", "shadow", "waiting"):
                                        when = await next_check(session, p, decided or Due(p, Clock(None, None, 0, False)), now)
                                    elif outcome == "no_option" and decided is not None and decided.hold_chat:
                                        when = now + (CARED_EVERY if p.cared else CAREGIVER_EVERY)
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

