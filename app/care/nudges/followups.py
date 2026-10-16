"""One question after an unanswered reminder that is not a medicine."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, NudgeDue, NudgeLedger
from app.core import clock

logger = logging.getLogger(__name__)

from app.care.nudges.compose import Option, build_context, send, write_and_check
from app.care.nudges.ledger import (
    CARED_EVERY, CAREGIVER_EVERY, FOLLOWUP_AFTER, PARALLEL, PRIORITY, SENT, WINDOW,
    _iso, _log, _remember_mode, _when, env_mode, learned, ledger_rows,
    sent_in_other_family,
)
from app.care.nudges.planner import (
    claim_person, engine_times, mark_next, people, read_clock,
)

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

