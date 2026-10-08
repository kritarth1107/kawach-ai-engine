"""Scheduled wake-ups: an open loop is due, so Saheli looks again and decides what to do.

Following Instinct's split, each loop carries its goal (title), when to look again (wake_at) and what
to do if nobody answered (alert_rule). The decision itself is the brain's, on a system turn.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.brain.host import ToolHost
from app.brain.loop import TurnRequest, run_turn
from app.care import store
from app.care.models import OpenLoop, Turn
from app.core import clock

logger = logging.getLogger(__name__)

SYSTEM = {"id": "saheli-scheduler", "name": "Scheduler", "role": "system"}
QUIET_START, QUIET_END = 22, 7  # IST; only no-answer alerts run inside these hours
MAX_WAKES = 3


def in_quiet_hours(at=None) -> bool:
    h = clock.ist(at).hour
    return h >= QUIET_START or h < QUIET_END


def next_morning(at=None):
    local = clock.ist(at)
    morning = local.replace(hour=7, minute=30, second=0, microsecond=0)
    if local.hour >= QUIET_START:
        morning += timedelta(days=1)
    return morning


KIND_PROMPTS = {
    "family_task": (
        "[Scheduled wake-up] Family task {id} is due: \"{title}\". It is assigned to person {owner}. Send {owner} one short "
        "reminder with send_message (who it is for and what to do). Do not message anyone else. If the conversation shows it "
        "is already done, close_loop it instead. Then reply none."
    ),
    "appointment": (
        "[Scheduled wake-up] {title}. Remind the person the appointment is for (send_message, in their language), and tell "
        "the caregiver who saved it in one line if it is someone else. Mention the questions to ask from care_team if any, and "
        "offer to book a cab (start_task ride) only if they want one. Then close_loop {id} and reply none."
    ),
    "followup": (
        "[Scheduled follow-up] {title} Ask person {owner} one short, warm question in their language with send_message, "
        "attaching buttons {{\"kind\": \"{set}\", \"key\": \"{key}\"}}. If the conversation already shows how it turned out, "
        "log_outcome instead of asking. Then close_loop {id} and reply none."
    ),
    "checkin": (
        "[Caregiver check-in] It is the weekly check-in on person {owner} themselves. Send them one short, warm message with "
        "send_message asking how they are doing this week (sleep, stress, their own health). One question only. If PATTERNS "
        "NOTICED lists something about the person they care for, add the most important one in one line, with its suggestion. "
        "Also ask whether there was any fall, hospital visit or medicine change this week, attaching buttons "
        "{{\"kind\": \"outcome\", \"key\": \"{key}\"}}. Then close_loop {id} and reply none."
    ),
    "confirm_skill": (
        "[Skill suggestion] {title} Ask person {owner} once, short and in their language, with send_message whether you should "
        "use this with them (it only changes your tone or timing). Keep this loop open: when they answer later, call "
        "save_family_skill with id {skill} for yes or forget_skill with id {skill} for no, then close_loop {id}. Reply none."
    ),
    "memory_check": (
        "[Memory check] Something in the care record needs a caregiver's answer: \"{title}\". Ask person {owner} that one "
        "question with send_message, short and in their language. When they answer later, fix the record (remember / stop / "
        "confirm_change). Then close_loop {id} and reply none."
    ),
    "refill": (
        "[Scheduled wake-up] {title}. Ask one caregiver from HOUSEHOLD with send_message whether to reorder it, and from which "
        "pharmacy (Apollo, 1mg or PharmEasy), unless the medicine belongs to that caregiver's own self care, then ask them. "
        "If they already answered, act on it: start_task (order) after a yes, close_loop {id} after a no. Then reply none."
    ),
}


async def wake_prompt(session: AsyncSession, loop: OpenLoop, elder_id: str) -> str:
    if loop.kind in KIND_PROMPTS:
        d = loop.detail or {}
        return KIND_PROMPTS[loop.kind].format(id=loop.id, title=loop.title, owner=loop.owner_id or "the caregiver",
                                              set=d.get("set", "outcome"), key=d.get("key", f"loop:{loop.id}"), skill=d.get("skill_id", ""))
    last = (
        await session.execute(
            select(Turn)
            .where(Turn.family_id == loop.family_id, Turn.thread_id == elder_id, Turn.role == "user")
            .order_by(Turn.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    replied = bool(last and last.at > loop.created_at)
    since = f"The elder last wrote at {clock.ist(last.at).strftime('%d %b %H:%M')}" if last else "The elder has not written yet"
    return (
        f"[Scheduled wake-up] Open loop {loop.id} is due: \"{loop.title}\" ({loop.kind}), opened "
        f"{clock.ist(loop.created_at).strftime('%d %b %H:%M')}. If no answer: {loop.alert_rule or 'ask_again'}. "
        f"{since}; {'they have written since the loop was opened' if replied else 'nothing from them since the loop was opened'}. "
        f"This is wake-up {int((loop.detail or {}).get('wakes', 0)) + 1} of {MAX_WAKES}. Decide now."
    )


async def system_turn(sessions: async_sessionmaker, host: ToolHost, family_id: str, prompt: str, ref: str, deliver_to: str | None = None) -> None:
    """Let the brain act on something that happened without a person writing (a due loop, a task update).

    deliver_to: the person this update is for (who asked for the order or ride). If the brain wrote its message as the
    final reply instead of sending it with send_message (that reply goes to no one), it is sent to them here, so an
    update can never vanish (live 2026-10-08 21:41: the Blinkit price and go-ahead question reached no one)."""
    async with sessions() as session:
        roster = await store.roster(session, family_id)
    if not roster:
        logger.warning("system turn skipped, no roster family=%s", family_id)
        return
    async with sessions() as session:
        result = await run_turn(
            session, host,
            TurnRequest(family_id=family_id, elder=roster.elder, speaker=SYSTEM, members=roster.members, text=prompt, message_ref=ref, channel="scheduler"),
        )
    text = (result.reply or "").strip()
    known = {m.get("id") for m in [roster.elder, *roster.members]}
    sent = any(a.get("tool") == "send_message" and a.get("ok") and (a.get("args") or {}).get("to") == deliver_to for a in result.actions)
    if not deliver_to or deliver_to not in known or sent or result.duplicate or not text or text.lower().strip(" .\"'") == "none":
        return
    if any(w in text for w in ("task_input", "task_id", "[Task", "send_message")):
        logger.error("system turn reply not delivered (reads like notes) family=%s ref=%s", family_id, ref)
        return
    try:
        res = await host.call("send_whatsapp", {"to": deliver_to, "text": text}, family_id=family_id.removeprefix("shadow:"),
                              subject_id=roster.elder["id"], actor_id="saheli")
        async with sessions() as session:
            await store.add_turn(session, family_id=family_id, thread_id=deliver_to, role="assistant", text=text,
                                 meta={"proactive": True, "delivered": res.get("delivered"), "fallback_delivery": ref})
            await session.commit()
        logger.warning("system turn reply delivered by fallback family=%s ref=%s", family_id, ref)
    except Exception:  # noqa: BLE001
        logger.exception("fallback delivery failed family=%s ref=%s", family_id, ref)


async def wake_due(sessions: async_sessionmaker, host_for: Callable[[str], ToolHost], *, limit: int = 50) -> dict:
    stats = {"due": 0, "ran": 0, "deferred": 0, "expired": 0, "failed": 0}
    async with sessions() as session:
        due = await store.due_loops(session, before=clock.now(), limit=limit)
        stats["due"] = len(due)
        work = [(l.id, l.family_id) for l in due]
        await session.commit()

    for loop_id, family_id in work:
        async with sessions() as session:
            loop = await session.get(OpenLoop, loop_id, with_for_update=True)
            if not loop or loop.status != "open":
                continue
            wakes = int((loop.detail or {}).get("wakes", 0))
            if wakes >= MAX_WAKES:
                loop.status, loop.closed_note, loop.updated_at = "expired", "no resolution after wake-ups", clock.now()
                await store.record_event(session, family_id=family_id, subject_id=loop.subject_id, kind="loop_expired", summary=loop.title)
                await session.commit()
                stats["expired"] += 1
                continue
            if in_quiet_hours() and loop.alert_rule != "alert_caregiver":
                loop.wake_at = next_morning()
                await session.commit()
                stats["deferred"] += 1
                continue
            roster = await store.roster(session, family_id)
            if not roster:
                loop.wake_at = clock.now() + timedelta(hours=1)
                await session.commit()
                stats["failed"] += 1
                continue
            prompt = await wake_prompt(session, loop, roster.elder["id"])
            # Count the wake before the brain runs, so a crash cannot make it fire forever.
            loop.detail = {**(loop.detail or {}), "wakes": wakes + 1}
            # One-shot reminders (family tasks, appointments, refills) stay open on the dashboard but wake only once.
            once = wakes + 1 >= int((loop.detail or {}).get("max_wakes", MAX_WAKES))
            loop.wake_at = None if once else clock.now() + timedelta(hours=1)
            await session.commit()
        try:
            async with sessions() as session:
                await run_turn(
                    session,
                    host_for(family_id),
                    TurnRequest(
                        family_id=family_id, elder=roster.elder, speaker=SYSTEM, members=roster.members, text=prompt,
                        message_ref=f"wake:{loop_id}:{wakes + 1}", channel="scheduler",
                    ),
                )
            stats["ran"] += 1
        except Exception:  # noqa: BLE001 — one family's failure must not stop the others
            logger.exception("wake failed loop=%s family=%s", loop_id, family_id)
            stats["failed"] += 1
    logger.info("wake run %s", stats)
    return stats
