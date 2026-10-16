"""Scheduled wake-ups: an open loop is due, so Saheli looks again and decides what to do.

Following Instinct's split, each loop carries its goal (title), when to look again (wake_at) and what
to do if nobody answered (alert_rule). The decision itself is the brain's, on a system turn.
"""

from __future__ import annotations

import contextlib

import logging
from datetime import datetime, timedelta
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.brain.host import ToolHost
from app.brain.loop import TurnRequest, run_turn
from app.care import store
from app.care.models import OpenLoop, Turn
from app.core import clock
from app.core.drain import Draining

logger = logging.getLogger(__name__)

SYSTEM = {"id": "saheli-scheduler", "name": "Scheduler", "role": "system"}
QUIET_START, QUIET_END = 21, 8  # IST; only no-answer alerts run inside these hours (founder: nothing at night)
MAX_WAKES = 3


def in_quiet_hours(at=None) -> bool:
    h = clock.ist(at).hour
    return h >= QUIET_START or h < QUIET_END


async def own_night(session: AsyncSession, family_id: str, person: str | None) -> tuple[int, int] | None:
    """Their own night from what they told Saheli (sleep and wake-up times), as minutes of the day; None when not known
    or not believable. Someone who sleeps at 3 and wakes at 1:30 pm is not asleep at 9:15 pm (live 2026-10-13: an
    evening gym check-in would have waited till the next morning)."""
    if not person:
        return None
    from app.care import activities

    acts = {activities.normalise(f.value).get("type"): activities.normalise(f.value) for f in await activities.active(session, family_id, person)}
    sleep = activities._hm((acts.get("sleep") or {}).get("start"))
    wake = activities._hm((acts.get("wake_up") or {}).get("start"))
    if sleep is None or wake is None:
        return None
    length = (wake - sleep) % (24 * 60)
    return (sleep, wake) if 5 * 60 <= length <= 12 * 60 else None


def _inside(minute: int, night: tuple[int, int]) -> bool:
    start, end = night
    return start <= minute < end if start < end else minute >= start or minute < end


async def quiet_for(session: AsyncSession, family_id: str, person: str | None, at=None) -> tuple[bool, datetime | None]:
    """Whether it is night for this person, and when their morning starts (their wake-up plus 30 min, or 08:30)."""
    night = await own_night(session, family_id, person)
    if night is None:
        return (True, next_morning(at)) if in_quiet_hours(at) else (False, None)
    local = clock.ist(at)
    minute = local.hour * 60 + local.minute
    if not _inside(minute, night):
        return False, None
    morning = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=night[1] + 30)
    if morning <= local:
        morning += timedelta(days=1)
    return True, morning


def next_morning(at=None):
    local = clock.ist(at)
    morning = local.replace(hour=8, minute=30, second=0, microsecond=0)
    if local.hour >= QUIET_START:
        morning += timedelta(days=1)
    return morning


KIND_PROMPTS = {
    "plan": (
        "[Follow-up on something they said] Person {owner} said: \"{title}\". It should have happened by now. Ask them warmly "
        "with send_message how it went, one short question in their language, building on what you know. When they answer, "
        "remember what matters (remember, log_event, log_activity) and reply like a friend. If the chat already shows how it "
        "went, do not ask. Then close_loop {id} and reply none."
    ),
    "family_task": (
        "[Scheduled wake-up] Family task {id} is due: \"{title}\". It is assigned to person {owner}. Send {owner} one short "
        "reminder with send_message (who it is for and what to do). Do not message anyone else. If the conversation shows it "
        "is already done, close_loop it instead. Then reply none."
    ),
    "appointment": (
        "[Scheduled wake-up] {title}. Remind the person the appointment is for (send_message, in their language), and tell "
        "the caregiver who saved it in one line if it is someone else. Mention the questions to ask from care_team if any; call "
        "weekly_report (days 30) and, if it shows repeated problems or symptoms, add in a few lines what to tell the doctor "
        "(a repeated problem as one point with how many times and each one's circumstances). Offer to book a cab (start_task "
        "ride) only if they want one. Then close_loop {id} and reply none."
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
    "finish": (
        "[Finish the reply] Person {owner} is waiting. You already said you would reply shortly, and anything you saved "
        "on their last message stays saved: do not log, send or change it again. Write the answer you owe them now, with "
        "send_message to {owner}, in their language. Then close_loop {id} and reply none."
    ),
    "refill": (
        "[Scheduled wake-up] {title}. Offer once, in the language they use with you, to reorder it the usual way. "
        "What: {item}. Where: {where}. How many: {qty}. Cash on delivery only. Ask person {owner} (their own self care, "
        "or a caregiver when it is for the elder). One short question. Only after a yes, call start_task (kind order, "
        "for_whom the person the stock is for, this item and quantity, service {service} when it names a store, otherwise "
        "category {category}). Do not place it: they still get what every store found and one confirm with the amount, and "
        "an elder's order still waits for the caregiver's approval. If the CARE RECORD marks it \"check it is still right\", "
        "first ask whether it is still being taken. If they say no, close_loop {id}. If they already answered, do that and "
        "do not ask again. Then reply none."
    ),
}


async def activity_prompt(session: AsyncSession, loop: OpenLoop) -> str | None:
    """Around one occurrence of a habit (founder 2026-10-13): a light nudge before, a check-in after with follow-ups, an
    evening check for something they are giving up. The words are written now, fresh; the facts come from the record."""
    from app.care import activities as acts

    d = loop.detail or {}
    fact = await store.active_fact(session, loop.family_id, loop.subject_id, d.get("activity_key") or "")
    if not fact or acts.normalise(fact.value).get("paused"):
        return None
    v = acts.normalise(fact.value)
    rows = {r["key"]: r for r in await acts.status(session, loop.family_id, loop.subject_id)}
    st = rows.get(fact.key) or {}
    evs = await acts.logs(session, loop.family_id, loop.subject_id, days=10, key=fact.key)
    if any(e.day == d.get("date") for e in evs):
        return None  # they already told Saheli about this day's
    recent = [e.summary for e in evs[-3:]]
    ctx = (f" Streak: {st.get('streak', 0)} in a row; {st.get('week_done', 0)} this week." + (f" Goal: {v['goal']}." if v.get("goal") else "")
           + (" Last times: " + " | ".join(recent) if recent else ""))
    who = loop.owner_id or loop.subject_id
    if d.get("phase") == "before":
        return (f"[Activity: before] {acts.describe(v)} for person {who} starts at {v['start']}.{ctx} Send them one short, light nudge "
                f"with send_message, fresh words (a cheer, their goal or streak, what to carry), never the same as last time. If the "
                f"conversation shows they are skipping today or already went, do not send. Then close_loop {loop.id} and reply none.")
    asks = acts.CATEGORY.get(v["category"], ("how it went", []))[0]
    if v["kind"] == "quit":
        return (f"[Activity: evening check] Person {who} is working on: {v['label']}.{ctx} Ask them warmly with send_message how "
                f"today went with it (one question). When they answer: log_activity (done = they kept to it), praise every win, "
                f"no shame for a slip, ask what helped or one gentle follow-up. Then close_loop {loop.id} and reply none.")
    return (f"[Activity: after] {acts.describe(v)} for person {who} should be over now.{ctx} With send_message ask them warmly "
            f"whether they are back / it happened and how it went, one question. When they answer, log_activity what they say "
            f"(done or skipped, details, any reading such as weight), then ask one natural follow-up from their answer ({asks}); "
            f"at most two follow-ups in all, then wrap up warmly. If they skipped, be kind and ask if the time should change. "
            f"If they do not answer, reply none and do not open another loop. One question until they answer. Reply none.")


async def wake_prompt(session: AsyncSession, loop: OpenLoop, elder_id: str) -> str | None:
    """The scheduled turn for a due loop; None when there is nothing left to do (the loop is closed)."""
    if loop.kind == "activity":
        return await activity_prompt(session, loop)
    if loop.kind == "delivery":
        from app.tasks import aftercare  # did the order come, and later how was it

        return await aftercare.wake_prompt(session, loop)
    if loop.kind in KIND_PROMPTS:
        d = loop.detail or {}
        return KIND_PROMPTS[loop.kind].format(
            id=loop.id, title=loop.title, owner=loop.owner_id or "the caregiver",
            set=d.get("set", "outcome"), key=d.get("key", f"loop:{loop.id}"), skill=d.get("skill_id", ""),
            item=d.get("item") or "it", where=d.get("where") or "no usual store yet",
            qty=d.get("qty") if d.get("qty") not in (None, "") else "their usual amount",
            service=d.get("service") or "none", category=d.get("category") or "grocery",
        )
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


MUST_REACH = ("[Task update]", "[Login code needed]", "[Order needs you]", "[Approval needed]",
              "[Nutrition update]")  # updates that must reach the person

IN_THEIR_WORDS = """Rewrite this WhatsApp message from Saheli (a caring assistant) for the person: in the language and script
Saheli has been writing to them in (her recent messages are below; else the one they write in; English if there are
none). Keep every number, amount, id, code, name, store and place exactly as they are, keep the line breaks of a numbered
list, add nothing and leave nothing out. Only the message."""


def _silent(text: str) -> bool:
    return not text or text.lower().strip(" .\"'") == "none"


def _sent_to(result, person: str | None) -> bool:
    return bool(result) and any(a.get("tool") == "send_message" and a.get("ok") and (a.get("args") or {}).get("to") == person
                                for a in result.actions)


def _numbers(text: str) -> set[str]:
    """The numbers in a message, in any script's digits (₹१४५ → 145)."""
    import re
    import unicodedata

    plain = "".join(str(unicodedata.digit(ch)) if unicodedata.digit(ch, None) is not None else ch for ch in text or "")
    return set(re.findall(r"\d+", plain))


async def in_their_words(sessions: async_sessionmaker, family_id: str, person: str, text: str) -> str:
    """A message written by code, put in the language the person writes in (an AI rewrite; their own recent messages
    show the language). The text as it is when no model answers."""
    from app.llm import router

    try:
        async with sessions() as session:
            recent = await store.recent_turns(session, family_id, person, limit=12)
        mine = [t.text for t in recent if t.role == "user" and t.text][-4:]
        hers = [t.text for t in recent if t.role == "assistant" and (t.text or "").strip() not in ("", "none")][-3:]
        reply = await router.complete("worker", system_stable=IN_THEIR_WORDS, effort="low", max_tokens=4000, timeout_s=30, messages=[
            {"role": "user", "content": [{"type": "text", "text": "SAHELI'S RECENT MESSAGES TO THEM:\n"
                                          + ("\n".join(f"- {m[:200]}" for m in hers) or "(none)")
                                          + "\n\nTHEIR RECENT MESSAGES:\n" + ("\n".join(f"- {m[:200]}" for m in mine) or "(none)")
                                          + f"\n\nMESSAGE:\n{text}"}]}])
        out = (reply.text or "").strip()
        # a rewrite that lost a number, id, amount or code is not sent (sim 2026-10-12: a Tamil rewrite stopped half-way and
        # dropped the driver, car and PIN): the plain text goes instead
        if not out or len(out) > 4 * len(text) + 200 or not _numbers(text) <= _numbers(out):
            logger.warning("in_their_words: the rewrite lost something; sending the plain text family=%s", family_id)
            return text
        return out
    except Exception as exc:  # noqa: BLE001 — the plain text still goes
        logger.warning("in_their_words failed: %s", exc)
        return text


async def saved_language_line(session, family_id: str, person: str | None) -> str:
    """For a message to one person that Saheli starts herself: write it in their saved language setting."""
    if not person:
        return ""
    from app.care import language

    row = next((f for f in await store.facts(session, family_id, person, domains=["language"]) if f.status == "active"), None)
    line = language.write_in(row.value) if row else ""
    return f"\n\n{line}" if line else ""


async def system_turn(sessions: async_sessionmaker, host: ToolHost, family_id: str, prompt: str, ref: str, deliver_to: str | None = None,
                      still_true=None, fallback: str | None = None) -> None:
    """Let the brain act on something that happened without a person writing (a due loop, a task update).

    deliver_to: the person this update is for (who asked for the order or ride). If the brain wrote its message as the
    final reply instead of sending it with send_message (that reply goes to no one), it is sent to them here (live
    2026-10-08 21:41: the Blinkit price and go-ahead question reached no one).

    still_true: for an update about a task (app.tasks.runtime.TaskUpdate): checked once the family's turn is held; an
    update whose task moved on meanwhile is dropped, so the person never hears two contradicting things.

    An update that must reach the person (MUST_REACH) does (founder 2026-10-12: the login-code ask and the booked ride never
    reached him: the brain answered "none" and the retry was skipped because Saheli had spoken to him in the last two
    minutes). It is skipped only when they were already told this exact state; if the brain sends nothing it is asked
    again; if it still sends nothing (or no model answers), the news written from the task data goes. Only when the person
    wrote after the update was written, and was answered, may the brain decide on the retry that they already heard it."""
    async with sessions() as session:
        roster = await store.roster(session, family_id)
        speak = await saved_language_line(session, family_id, deliver_to) if roster else ""
    if not roster:
        logger.warning("system turn skipped, no roster family=%s", family_id)
        return
    known = {m.get("id") for m in [roster.elder, *roster.members]}
    update = still_true if hasattr(still_true, "already_told") else None
    must = bool(deliver_to and deliver_to in known and prompt.startswith(MUST_REACH))
    if must and update is not None:
        async with sessions() as session:
            if await update.already_told(session, deliver_to):
                logger.info("task update not sent again: person %s was told this state family=%s ref=%s", deliver_to, family_id, ref)
                return
            # (when they wrote after it was written and were answered, the brain decides whether they already heard it)
            booked = await update.booked(session) and not await update.wrote_since(session, family_id, deliver_to)
            if booked:
                names = {m.get("id"): m.get("name") for m in [roster.elder, *roster.members] if m.get("id")}
                news = await update.news(session, deliver_to, names)
        if booked and news:
            # Placed or booked: every detail, written from the task data (the brain once left out the driver, car and PIN).
            words = await in_their_words(sessions, family_id, deliver_to, news[0])
            if await _send(sessions, host, roster, family_id, deliver_to, words, f"{ref}:booked", buttons=news[1]):
                async with sessions() as session:
                    await update.told(session, deliver_to, "composed")
                return

    async def turn(text: str, ref_: str):
        try:
            async with sessions() as session:
                return await run_turn(session, host, TurnRequest(
                    family_id=family_id, elder=roster.elder, speaker=SYSTEM, members=roster.members, text=text, message_ref=ref_,
                    channel="scheduler", still_true=still_true))
        except Draining:
            raise
        except Exception:
            if not must:
                raise
            logger.exception("task update turn failed family=%s ref=%s", family_id, ref_)  # the news still goes below
            return None

    result = await turn(prompt + speak, ref)
    if result is not None and (result.stale or result.duplicate):
        return
    text = (result.reply or "").strip() if result else ""
    sent = _sent_to(result, deliver_to)
    soft = False
    if must and not sent and _silent(text):
        if update is not None:
            async with sessions() as session:
                soft = await update.wrote_since(session, family_id, deliver_to)
        logger.error("task update sent nothing; asking again family=%s ref=%s soft=%s", family_id, ref, soft)
        again = (f"If person {deliver_to} has not heard this yet, write the message for them now, in their language, as your final "
                 "reply (only the message, no notes); if your last answer to them already told them exactly this, reply none."
                 if soft else
                 f"Person {deliver_to} has NOT heard this. Write the message for them now, in their language, as your final reply "
                 "(only the message, no notes).")
        retry = await turn(f"{prompt}{speak}\n\nYour last turn on this update sent NOTHING, so person {deliver_to} heard nothing from it. {again}",
                           f"{ref}:again")
        if retry is not None and retry.stale:
            return
        if retry is not None:
            text, sent = (retry.reply or "").strip(), _sent_to(retry, deliver_to)
    delivered = "brain" if sent else ""
    if not delivered and deliver_to in known and not _silent(text):
        night = False
        if not must:
            async with sessions() as session:
                night, _ = await quiet_for(session, family_id, deliver_to)
        if any(w in text for w in ("task_input", "task_id", "[Task", "[task ", "send_message")):
            logger.error("system turn reply not delivered (reads like notes) family=%s ref=%s", family_id, ref)
        elif night:
            # send_message refuses at their night; its final reply must not slip out by this side door (2026-10-14)
            logger.warning("system turn reply held: their night family=%s ref=%s", family_id, ref)
        elif await _send(sessions, host, roster, family_id, deliver_to, text, ref):
            delivered = "reply"
    if must and not delivered and not soft and update is not None:
        # The last resort: the news written from the task data, in their language when a model answers.
        async with sessions() as session:
            if not await update(session):
                return  # it moved on meanwhile: the next update tells the truth
            names = {m.get("id"): m.get("name") for m in [roster.elder, *roster.members] if m.get("id")}
            news = await update.news(session, deliver_to, names)
        if news:
            plain, buttons = news
            logger.error("task update written from the task data (the brain sent nothing twice) family=%s ref=%s", family_id, ref)
            words = await in_their_words(sessions, family_id, deliver_to, plain)
            if await _send(sessions, host, roster, family_id, deliver_to, words, f"{ref}:news", buttons=buttons):
                delivered = "composed"
    elif must and not delivered and not soft and fallback and fallback.strip() and deliver_to:
        # A saved line with the numbers already in it (a meal confirmation). The brain sent nothing, so this goes directly.
        logger.error("update sent as the written line (the brain sent nothing) family=%s ref=%s", family_id, ref)
        if await _send(sessions, host, roster, family_id, deliver_to, fallback.strip(), f"{ref}:plain"):
            delivered = "plain"
    elif must and not delivered and not soft:
        logger.error("an update for person %s reached nobody (no task to write it from) family=%s ref=%s", deliver_to, family_id, ref)
    if delivered and update is not None:
        async with sessions() as session:
            await update.told(session, deliver_to, delivered)


async def _send(sessions: async_sessionmaker, host: ToolHost, roster, family_id: str, person: str, text: str, ref: str,
                buttons: tuple[str, str] | None = None) -> bool:
    """Send a message to a person directly (a system turn's reply the brain did not send itself, or news from a task)."""
    from app.brain import guards

    async with sessions() as session:
        greeted, mid = guards.opening_state(await store.recent_turns(session, family_id, person, limit=12), clock.now(), day_of=clock.ist_day)
    text = guards.tidy_opening(text, greeted_today=greeted, mid_conversation=mid)
    from app.brain.tools import delivery_kind

    # the backend's delivery ledger: what it is (a check-in counts towards the no-reply alert) and the ref it came from
    payload: dict = {"to": person, "text": text, "kind": delivery_kind(ref), "ref": ref[:160]}
    if buttons:
        from app.brain.loop import writing_profiles
        from app.care import outcomes

        async with sessions() as session:
            profile = (await writing_profiles(session, family_id, [{"id": person}])).get(person)
        payload["buttons"] = outcomes.buttons(buttons[0], buttons[1], profile, text)
    try:
        # The backend only accepts a family member as the actor (as TurnCtx.actor_id does on system turns): "saheli" was
        # refused, the request hung and the update never went out (live 2026-10-09 15:05, Maa's Blinkit price).
        res = await host.call("send_whatsapp", payload, family_id=family_id.removeprefix("shadow:"),
                              subject_id=roster.elder["id"], actor_id=roster.elder["id"])
        async with sessions() as session:
            await store.add_turn(session, family_id=family_id, thread_id=person, role="assistant", text=text,
                                 meta={"proactive": True, "delivered": res.get("delivered"), "fallback_delivery": ref,
                                       **({"wamid": res["wamid"]} if res.get("wamid") else {})})
            await session.commit()
        logger.warning("system turn message delivered directly family=%s ref=%s", family_id, ref)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("direct delivery failed family=%s ref=%s", family_id, ref)
        return False


async def _order_moving_for(session: AsyncSession, loop: OpenLoop) -> bool:
    from app.care.checkins import moving_for
    from app.tasks import runtime

    person = loop.owner_id or loop.subject_id
    return bool(person) and moving_for(await runtime.live_tasks(session, loop.family_id), person, clock.now())


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
            expire_at = (loop.detail or {}).get("expire_at")
            if expire_at and clock.now() >= datetime.fromisoformat(expire_at):
                # a follow-up nobody answered closes by itself, without a last message
                loop.status, loop.closed_note, loop.updated_at = "expired", "no answer; closed by itself", clock.now()
                await store.record_event(session, family_id=family_id, subject_id=loop.subject_id, kind="loop_expired", summary=loop.title)
                await session.commit()
                stats["expired"] += 1
                continue
            if wakes >= MAX_WAKES:
                loop.status, loop.closed_note, loop.updated_at = "expired", "no resolution after wake-ups", clock.now()
                await store.record_event(session, family_id=family_id, subject_id=loop.subject_id, kind="loop_expired", summary=loop.title)
                await session.commit()
                stats["expired"] += 1
                continue
            # A reply they are already waiting for goes out now, including at night and while an order is moving.
            owes = loop.kind == "finish" or loop.alert_rule == "alert_caregiver"
            quiet, morning = (False, None) if owes else await quiet_for(session, family_id, loop.owner_id or loop.subject_id)
            if quiet:
                loop.wake_at = morning or next_morning()
                await session.commit()
                stats["deferred"] += 1
                continue
            roster = await store.roster(session, family_id)
            if not roster:
                loop.wake_at = clock.now() + timedelta(hours=1)
                await session.commit()
                stats["failed"] += 1
                continue
            if loop.kind not in ("delivery", "finish") and loop.alert_rule != "alert_caregiver" and await _order_moving_for(session, loop):
                # An order or ride is under way for them and may wait for their answer: a new question waits, the way
                # check-ins do (live 2026-10-08: "BP check kiya?" right after "send me the login code"). It is not used up.
                from app.care.checkins import MOVING_WAIT

                loop.wake_at = clock.now() + MOVING_WAIT
                await session.commit()
                stats["deferred"] += 1
                continue
            if loop.kind == "activity":
                from app.care import activities as acts

                if not await acts.claim_occurrence(session, loop):
                    await session.commit()
                    continue
                # one question for this occurrence. A loop already planned with a second wake waits until it expires.
                if int((loop.detail or {}).get("wakes") or 0) >= 1:
                    expire = (loop.detail or {}).get("expire_at")
                    loop.wake_at = datetime.fromisoformat(expire) if expire else None
                    await session.commit()
                    continue
            prompt = await wake_prompt(session, loop, roster.elder["id"])
            kind = loop.kind
            loop_subject = loop.subject_id
            if prompt is not None and loop.owner_id:
                prompt += await saved_language_line(session, family_id, loop.owner_id)  # the one person it is for
            if prompt is None:
                loop.status, loop.closed_note, loop.updated_at = "done", "nothing left to ask", clock.now()
                await session.commit()
                continue
            # Count the wake before the brain runs, so a crash cannot make it fire forever. "runs" never goes back (an order's
            # follow-up starts its wakes again when they say it has not come yet): it keeps each wake's turn its own.
            runs = int((loop.detail or {}).get("runs", wakes)) + 1
            d = loop.detail = {**(loop.detail or {}), "wakes": wakes + 1, "runs": runs}
            # One-shot reminders (family tasks, appointments, refills) stay open on the dashboard but wake only once; a loop
            # with expire_at wakes once more then, to close itself.
            cap = 1 if kind == "activity" else int(d.get("max_wakes", MAX_WAKES))
            once = wakes + 1 >= cap
            if once:
                loop.wake_at = datetime.fromisoformat(d["expire_at"]) if d.get("expire_at") else None
            else:
                loop.wake_at = clock.now() + timedelta(minutes=int(d.get("rewake_minutes") or 60))
            loop_trace = loop.trace_id  # a follow-up continues the trail it was opened in (an order's "has it come?")
            await session.commit()
        try:
            from app.core import trace

            with trace.use(loop_trace, task_id=(d.get("task_id") if isinstance(d, dict) else None), family_id=family_id) if loop_trace \
                    else contextlib.nullcontext():
                async with sessions() as session:
                    await run_turn(
                        session,
                        host_for(family_id),
                        TurnRequest(
                            family_id=family_id, elder=roster.elder, speaker=SYSTEM, members=roster.members, text=prompt,
                            about=loop_subject,
                            # an order's follow-up answers their own order: like a task update, it is not held back as a nudge
                            # (sim 2026-10-11: "has it come?" was blocked because they had not answered "placed")
                            message_ref=f"{'task:' if kind == 'delivery' else 'refill:' if kind == 'refill' else ''}wake:{loop_id}:{runs}", channel="scheduler",
                        ),
                    )
            stats["ran"] += 1
        except Draining:
            raise
        except Exception:  # noqa: BLE001 — one family's failure must not stop the others
            logger.exception("wake failed loop=%s family=%s", loop_id, family_id)
            stats["failed"] += 1
    logger.info("wake run %s", stats)
    return stats
