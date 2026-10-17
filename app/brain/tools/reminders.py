"""Reminders, activities and loops: set and change reminders, the day's schedule, open and close loops."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from app.brain import guards
from app.brain.tools.base import ABOUT_OWN, ToolRefused, TurnCtx, logger, tool
from app.brain.tools.memory import _follows_the_chat
from app.brain.tools.orders import _task
from app.care import store
from app.care.domains import slug
from app.care.models import OpenLoop
from app.core import clock

# ── loops ──────────────────────────────────────────────────────────────────────


def wake_from_clock(hhmm: str, now: datetime | None = None) -> datetime | None:
    """Their HH:MM as a wake time: later today, or tomorrow if that minute has passed. None when it is not a clock time."""
    parts = str(hhmm or "").strip().split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    hour, minute = int(parts[0]), int(parts[1])
    if hour > 23 or minute > 59:
        return None
    local = clock.ist(now).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if local <= clock.ist(now):
        local += timedelta(days=1)
    return local


@tool(
    "open_loop",
    "Track something unfinished: a question you asked and need an answer to, a follow-up to do later, "
    "a task in progress. Set at (HH:MM, their clock) to be woken when something they are doing ends, or "
    "check_back_in_minutes. if_no_answer decides what happens then: "
    "ask_again, alert_caregiver (only when the elder not answering is itself a concern, e.g. after a fall or a "
    "missed critical dose), or dashboard. A follow-up that falls in their night waits until morning.",
    {
        "kind": {"type": "string", "enum": ["question", "followup", "task", "watch"]},
        "title": {"type": "string"},
        "at": {"type": "string", "description": "When to look again, their clock HH:MM. Today, or tomorrow if that time has passed."},
        "check_back_in_minutes": {"type": "integer", "minimum": 5, "maximum": 10080},
        "if_no_answer": {"type": "string", "enum": ["ask_again", "alert_caregiver", "dashboard"]},
        "owner": {"type": "string", "description": "Member id of who must act next (default: the person it is about on a "
                                                   "scheduled turn, else the speaker)"},
        "next_action": {"type": "string", "description": "What has to happen next, in a few words"},
        "about": ABOUT_OWN,
    },
    ["kind", "title"],
)
async def open_loop_tool(ctx: TurnCtx, a: dict) -> dict:
    # An activity check already asks once and stays open until they answer. A watch or follow-up opened from that
    # turn asked the same thing again (one wake produced the check, then a second loop asked an hour later).
    if a.get("kind") in ("question", "followup", "watch"):
        parts = (ctx.message_ref or "").split(":")
        if len(parts) >= 2 and parts[0] == "wake":
            try:
                source = await ctx.session.get(OpenLoop, uuid.UUID(parts[1]))
            except ValueError:
                source = None
            if source and source.kind == "activity" and source.status == "open":
                return {"skipped": True, "why": "this activity check is already open; do not ask again until they answer",
                        "loop_id": str(source.id)}
    wake = wake_from_clock(a["at"]) if a.get("at") else None
    if wake is None and a.get("check_back_in_minutes"):
        wake = clock.now() + timedelta(minutes=a["check_back_in_minutes"])
    subject = ctx.own(a.get("about"))
    ids = {m.get("id") for m in [ctx.elder, *ctx.members]}
    # Every open job has an owner who acts next (live: check-in questions were owned by the scheduler, so nobody was)
    owner = a.get("owner") if a.get("owner") in ids else (subject if ctx.is_system else ctx.speaker.get("id"))
    loop = await store.open_loop(
        ctx.session, family_id=ctx.family_id, subject_id=subject, kind=a["kind"], title=a["title"],
        owner_id=owner, wake_at=wake, alert_rule=a.get("if_no_answer"),
        dedupe_key=f"{a['kind']}:{slug(a['title'])[:80]}", detail={"next_action": a["next_action"][:160]} if a.get("next_action") else None,
    )
    return {"loop_id": str(loop.id), "wake_at": clock.ist(wake).strftime("%H:%M") if wake else None}


@tool(
    "gap_asked",
    "Call in the same turn you ask one item from NOT KNOWN YET. A scheduled check-in reads this and will not ask it again today.",
    {"key": {"type": "string", "description": "The key from NOT KNOWN YET, such as water, meal, sleep, preference, or habit:gym"}},
    ["key"],
)
async def gap_asked(ctx: TurnCtx, a: dict) -> dict:
    key = str(a.get("key") or "").strip()[:80]
    if not key:
        raise ToolRefused("key is the item from NOT KNOWN YET")
    from app.care import gaps as gap_list

    who = ctx.speaker.get("id") or ctx.elder_id
    listed = await gap_list.block(ctx.session, ctx.family_id, who, ctx.speaker.get("name") or "them")
    allowed = {line.split(":", 1)[0].removeprefix("- ").strip() for line in listed.splitlines() if line.startswith("- ")}
    if key not in allowed and key.split(":")[0] not in allowed:
        raise ToolRefused("That is not on NOT KNOWN YET. Do not ask it. If they were teasing, reply lightly and do not ask for a name.")
    from app.care.gaps import ledger_topics
    topics = ledger_topics(key)
    for topic in topics:
        await store.record_event(
            ctx.session, family_id=ctx.family_id, subject_id=who, kind="checkin",
            summary=f"Asked in conversation: {topic}", payload={"topic": topic, "sent": True, "via": "conversation"},
            actor_id=ctx.speaker.get("id"),
        )
    return {"noted": key, "topics": topics}


@tool(
    "close_loop",
    "Close an open loop when it is answered, done, or no longer needed.",
    {"loop_id": {"type": "string"}, "outcome": {"type": "string"}},
    ["loop_id", "outcome"],
)
async def close_loop_tool(ctx: TurnCtx, a: dict) -> dict:
    try:
        loop_id = uuid.UUID(a["loop_id"])
    except ValueError as exc:
        raise ToolRefused("loop_id must be the id shown in OPEN LOOPS") from exc
    found = await ctx.session.get(OpenLoop, loop_id)
    if found and found.kind == "delivery" and found.status == "open":
        # An order's follow-up moves on through order_feedback (it came → how was it → done) and closes by itself (sim
        # 2026-10-11: closed after "it came", so "how was it?" was never asked).
        return {"closed": False, "why": "An order's follow-up closes by itself; pass what they said to order_feedback instead."}
    loop = await store.close_loop(ctx.session, loop_id, note=a["outcome"])
    return {"closed": bool(loop)}


SEND_COOLDOWN = timedelta(hours=2)


async def send_problems(ctx: TurnCtx, to: str, text: str) -> list[str]:
    """What is wrong with a message to someone else before it goes out: repeats, timing, language, invented details."""
    problems: list[str] = []
    turns = await store.recent_turns(ctx.session, ctx.family_id, to, limit=20)
    recent = [t for t in turns if t.at >= clock.now() - timedelta(hours=24)]
    sent = []
    for i, t in enumerate(recent):
        if t.role == "assistant":
            sent.append((t.text, any(u.role == "user" for u in recent[i + 1:])))
    # An order/ride update answers something they asked for ("I'll tell you when the cart is ready"), so it is not
    # a nudge: waiting for them to write first would leave the login code or the failure unsaid. Each update is a new
    # state of the same order, so its words look alike (sim 2026-10-10: "delivered" was blocked as a repeat of "placed,
    # arriving in 10 min"); only a near-exact repeat is stopped.
    task_update = ctx.is_system and (ctx.message_ref or "").startswith("task:")
    dup = guards.duplicate_message(text, sent, near=0.9 if task_update else 0.6)
    if dup:
        problems.append(f"you already sent them this today and they have not answered yet: \"{dup[:120]}\". Do not send it again")
    # Only Saheli's own unanswered nudge asks for patience: an order's update or follow-up answers their order (live
    # 2026-10-12: a 08:30 "has it come?" held back Maa's 10:30 check-in), and her reply in a chat they ended is not a
    # message waiting for an answer (live 2026-10-12: "You're welcome!" at 15:34 held back his 16:30 nudges).
    last_out = next((t for t in reversed(recent) if t.role == "assistant" and (t.meta or {}).get("proactive")
                     and (t.meta or {}).get("via") != "reminder"  # a reminder in the chat is not her nudge (its own caps)
                     and not str((t.meta or {}).get("ref") or (t.meta or {}).get("fallback_delivery") or "").startswith("task:")), None)
    answered = last_out is not None and any(t.role == "user" and t.at > last_out.at for t in recent)
    if ctx.is_system and not task_update and last_out and not answered and clock.now() - last_out.at < SEND_COOLDOWN:
        problems.append(f"you messaged them at {clock.ist(last_out.at).strftime('%H:%M')} and they have not answered; give them time "
                        "(reply none now; alert_caregiver only if the loop's rule says so)")
    if ctx.is_system and not task_update:
        from app.brain.wake import quiet_for
        from app.learn import timing

        # Their own night (or the household's), whoever the turn was about: a scheduled turn about the elder once reached a
        # caregiver at 22:00 (world message-budget sim, 2026-10-13). Alerts go through alert_caregiver, not here.
        night, morning = await quiet_for(ctx.session, ctx.family_id, to)
        if night and await _no_answer_alert_loop(ctx):
            night = False  # founder: only a no-answer alert may go out at night (wake_due lets those loops run)
        if night and str(ctx.message_ref or "").startswith(f"brief:{to}:"):
            from app.care import brief as care_brief

            if (await care_brief.setting(ctx.session, ctx.family_id, to)).get("time"):
                night = False  # the caregiver chose this time for their brief themselves
        if night:
            problems.append(f"it is night for them now (their morning starts {clock.ist(morning).strftime('%H:%M') if morning else 'later'}); "
                            "do not message them now, reply none")
        quiet = await timing.quiet_reason(ctx.session, ctx.family_id, to, ctx.situation)
        if quiet:
            problems.append(quiet)
    # The name they asked for, in every message she starts (trust plan A3; 13 Oct: order updates and check-ins to a
    # caregiver used his roster name, not the saved one)
    from app.care import naming

    roster_name = next((str(m.get("name") or "") for m in [ctx.elder, *ctx.members] if m.get("id") == to), "")
    problems += naming.address_problems(text, roster_name=roster_name, who=await naming.for_person(ctx.session, ctx.family_id, to))
    to_role = next((str(m.get("role") or "") for m in ctx.members if m.get("id") == to), "")
    if to == ctx.elder_id and "caregiver" not in to_role.lower() and not (ctx.message_ref or "").startswith("task:"):
        problems += guards.question_problems(text)  # the person cared for: one question at a time (an order's options excepted)
    # Wrong script blocks the message; a few English words in it are only advice (never worth not passing it on).
    problems += [x for x in guards.language_problems(text, ctx.profiles.get(to), who="they") if not x.startswith("do not mix scripts")]
    now = clock.ist()
    known = "\n".join(ctx.known)
    problems += guards.ungrounded(text, known=known, fresh=ctx.user_text, meds=ctx.meds, now_minutes=now.hour * 60 + now.minute)
    if str(ctx.message_ref or "").startswith(("brief:", "foodrecap:")):
        for phrase in guards.brief_mood_claims(text, known):
            problems.append(f"the brief says '{phrase}', which is not in the day's data: do not say how they feel, or that they are fine, "
                            "stable, peaceful, or free of symptoms, unless the data says it")
    return problems


async def _grounded_send(ctx: TurnCtx, to: str, text: str) -> str:
    """No invention in a message she starts or passes on (trust plan C4.5; 13 Oct live: the nightly brief said Maa "went to
    bed at 9:15 after a goodnight message" from a 21:15 message that said "bhindi"; send_message had no such check). The
    sources are memory and this turn's context (the scheduled prompt's data included), the recipient's own messages and this
    turn's tool results, never her own earlier messages. The first time, the message is sent back with the exact items; a
    second attempt to the same person loses those sentences instead, so a brief or an order update still goes."""
    from app.brain import grounding
    from app.quality import sample

    used = [a["tool"] for a in ctx.actions if a.get("ok")]
    force = sample.always_live(voice_unsure=ctx.voice_unsure, alerts=bool(ctx.alerts), health=False, tools=used, situation=ctx.situation)
    if force:
        ctx.judge_band = sample.ALWAYS
    elif ctx.judge_band == sample.SKIP:
        return text
    elif ctx.judge_band not in (sample.SAMPLE, sample.ALWAYS):
        ctx.judge_band = sample.SAMPLE if await sample.ordinary_room(ctx.session, ctx.family_id) > 0 else sample.SKIP
    if ctx.judge_band == sample.SKIP:
        return text
    theirs = [t.text for t in await store.recent_turns(ctx.session, ctx.family_id, to, limit=20) if t.role == "user" and t.text]
    memory = [p for p in ctx.known[:5] if p]  # family block, record, turn context, dynamic block, this turn's message or prompt
    if ctx.prompt and ctx.prompt not in memory:
        memory.append(ctx.prompt)
    claims = await grounding.check(text, memory="\n".join(memory), their_messages=[*theirs, ctx.user_text],
                                   tool_results=ctx.tool_texts)
    if not claims:
        return text
    tries = ctx.send_checks.get(to, 0)
    ctx.send_checks[to] = tries + 1
    if tries == 0:
        raise ToolRefused("Not sent: " + "; ".join(grounding.problems(claims)) + ". Write it again without those, then send it.")
    kept = grounding.strip(text, claims)
    logger.warning("ungrounded sentences removed from a message family=%s claim_count=%d", ctx.family_id[:8], len(claims))
    if not kept:
        raise ToolRefused("Not sent: nothing in it is supported by memory, this chat or a tool result.")
    return kept


def delivery_kind(ref: str | None) -> str:
    """The backend ledger's kind for a message Saheli starts: a scheduled check-in, the nightly brief, or anything else
    (news from a task, a relay, a follow-up she promised). Only check-ins and nudges count towards the no-reply alert."""
    r = ref or ""
    if r.startswith("checkin:"):
        return "checkin"
    if r.startswith("brief:"):
        return "brief"
    return "message"


async def _no_answer_alert_loop(ctx: TurnCtx) -> bool:
    """This scheduled turn runs an open loop whose rule is to tell a caregiver when there is no answer."""
    ref = ctx.message_ref or ""
    if not ref.startswith("wake:"):
        return False
    from app.care.models import OpenLoop

    try:
        loop_id = uuid.UUID(ref.split(":")[1])
    except (ValueError, IndexError):
        return False
    loop = await ctx.session.get(OpenLoop, loop_id)
    return bool(loop and loop.alert_rule == "alert_caregiver")


async def _approval_just_asked(ctx: TurnCtx, person: str) -> bool:
    """An order or ride of this family started waiting for this person's approval a moment ago (the task system asks them)."""
    from app.tasks import runtime

    for t in await runtime.live_tasks(ctx.session, ctx.family_id):
        ap = (t.details or {}).get("approval_needed") or {}
        if t.input_needed == "approve" and person in (ap.get("approvers") or []) and ap.get("asked_at"):
            try:
                if clock.now() - datetime.fromisoformat(ap["asked_at"]) < timedelta(minutes=3):
                    return True
            except ValueError:
                continue
    return False


def first_line(text: str, limit: int = 90) -> str:
    """The first sentence of Saheli's message: the short line of an alert or update template."""
    import re

    one = " ".join(re.sub(r"[*_~`]", "", text or "").split())
    first = (re.split(r"(?<=[.!?।॥])\s", one, maxsplit=1)[0] or one).rstrip(".।")
    return first if len(first) <= limit else first[: limit - 1].rsplit(" ", 1)[0] + "…"


async def _order_for(ctx: TurnCtx, person: str):
    """The latest order this person asked for or is the subject of, or None."""
    from sqlalchemy import or_, select

    from app.tasks.models import Task

    return (await ctx.session.execute(
        select(Task).where(Task.family_id == ctx.family_id, or_(Task.requested_by == person, Task.subject_id == person))
        .order_by(Task.updated_at.desc()).limit(1)
    )).scalar_one_or_none()


def _first_name(ctx: TurnCtx, person_id: str | None) -> str:
    name = next((str(m.get("name") or "") for m in [ctx.elder, *ctx.members] if m.get("id") == person_id), "")
    return (name.split() or ["-"])[0]


async def _order_notice(ctx: TurnCtx, person: str) -> dict:
    """The approved one-line notice for a closed chat (Meta, 1 name: who the order is for). An approval waiting on this
    person uses kavach_alert_approval; every other order update uses kavach_alert_order. The full update waits."""
    t = await _order_for(ctx, person)
    who = _first_name(ctx, (t.subject_id if t else None) or ctx.elder.get("id"))
    approvers = (((t.details or {}) if t else {}).get("approval_needed") or {}).get("approvers") or []
    if t is not None and t.input_needed == "approve" and person in approvers:
        return {"purpose": "alert_approval", "params": [who]}
    return {"purpose": "alert_order", "params": [who]}


async def _refill_who(ctx: TurnCtx, ref: str) -> str:
    """Who a reorder offer is for: the person whose stock is running out, named on the approved order notice."""
    try:
        loop_id = uuid.UUID(ref.split(":")[2])
    except (IndexError, ValueError):
        return _first_name(ctx, ctx.elder.get("id"))
    loop = await ctx.session.get(OpenLoop, loop_id)
    if loop is None:
        return _first_name(ctx, ctx.elder.get("id"))
    return _first_name(ctx, loop.subject_id)


def _clock_label(at=None) -> str:
    t = clock.ist(at)
    return t.strftime("%I:%M %p").lstrip("0")


async def template_hint(ctx: TurnCtx, to: str, text: str, reminder: dict | None = None) -> dict | None:
    """What reaches someone whose WhatsApp chat is closed (no message from them in 24 hours; founder 2026-10-12): the
    backend sends this Utility template, and the message itself waits for their next message. The nightly brief, an
    order's update, a reorder offer, and news for a caregiver have one; a check-in, a question to the person cared
    for, and a message someone asked Saheli to pass on do not (they are not sent while the chat is closed)."""
    if not ctx.is_system:
        # Someone asked Saheli to send this (live 2026-10-13: a caregiver asked for his mother's missed reminders and the
        # message could not reach her, her chat had been closed for weeks): it goes as the reminder notice, and the full
        # message waits for her tap or reply.
        saved = ((ctx.profiles.get(to) or {}).get("saved") or {}).get("language")
        what = str((reminder or {}).get("what") or "").strip()[:60]
        if not what:
            sender = (str(ctx.speaker.get("name") or "").split() or ["your family"])[0]
            what = f"a message from {sender}"
        when = str((reminder or {}).get("at") or "").strip()[:20] or _clock_label()
        return {"purpose": "reminder", "params": [what, when], **({"lang": saved} if saved else {})}
    ref = ctx.message_ref or ""
    if ref.startswith("checkin:"):
        return None
    saved = ((ctx.profiles.get(to) or {}).get("saved") or {}).get("language")
    lang = {"lang": saved} if saved else {}
    who = (str(ctx.elder.get("name") or "").split() or ["-"])[0]
    if ref.startswith("brief:"):
        return {"purpose": "summary", "params": [who], **lang}
    if ref.startswith("task:"):
        return {**(await _order_notice(ctx, to)), **lang}
    if ref.startswith("refill:"):
        return {"purpose": "alert_order", "params": [await _refill_who(ctx, ref)], **lang}
    if ref.startswith("wake:") and to == ctx.elder_id:
        hint = await _wake_reminder(ctx, ref)
        if hint:
            return {**hint, **lang}
    if to != ctx.elder_id:
        return {"purpose": "alert", "params": [who, first_line(text)], **lang}
    return None


async def _wake_reminder(ctx: TurnCtx, ref: str) -> dict | None:
    """A habit's before-nudge, as the reminder template, so a closed chat still gets it at that time."""
    from app.care import activities as acts

    try:
        loop_id = uuid.UUID(ref.split(":")[1])
    except (IndexError, ValueError):
        return None
    loop = await ctx.session.get(OpenLoop, loop_id)
    if loop is None or loop.kind != "activity":
        return None
    detail = loop.detail or {}
    fact = await store.active_fact(ctx.session, ctx.family_id, loop.subject_id, str(detail.get("activity_key") or ""))
    return acts.closed_window_reminder(detail, fact.value if fact else {})


@tool(
    "send_message",
    "Send a WhatsApp message to someone in the family: pass a message on, or, on a scheduled wake-up, start the "
    "follow-up you promised. Your normal reply already goes to the person you are talking to, so do not use this for "
    "that — except when they ask for a voice note of particular words ('send a voice note saying …', 'send it to me, "
    "I will forward it'): then call this with voice true, to them or to another family member. voice speaks those "
    "exact words as a WhatsApp voice note. A song is sing, not this. Changing whether replies are spoken is voice_replies.",
    {
        "to": {"type": "string", "description": "Person id from HOUSEHOLD"},
        "text": {"type": "string"},
        "voice": {"type": "boolean", "description": "true: also send this text as a spoken WhatsApp voice note (to the person you are replying to, or to another family member)"},
        "buttons": {"type": "object", "description": "Optional one-tap answers: {kind: outcome|visit|feedback|order, key: the key given in the prompt (for order: the task id)}",
                    "properties": {"kind": {"type": "string", "enum": ["outcome", "visit", "feedback", "order", "later"]}, "key": {"type": "string"}}},
        "ask_location": {"type": "boolean", "description": "The message asks them to share a location: it goes with WhatsApp's 'Send location' button"},
        "reminder": {"type": "object", "description": "When the message reminds them of something (a medicine, a task): what and when, "
                     "e.g. {what: 'morning medicines', at: '8:00 AM'} (at empty = now). Used for the short notice when their chat is closed.",
                     "properties": {"what": {"type": "string"}, "at": {"type": "string"}}},
    },
    ["to", "text"],
)
async def send_message(ctx: TurnCtx, a: dict) -> dict:
    ids = {m.get("id") for m in ctx.members} | {ctx.elder_id}
    if a["to"] not in ids:
        raise ToolRefused("Send only to people in HOUSEHOLD, by id.")
    if a["to"] == ctx.speaker.get("id") and not a.get("voice"):
        raise ToolRefused("Your reply already goes to the speaker. To send them these words as a voice note, call send_message with voice true.")
    if not await _follows_the_chat(ctx, a.get("text") or ""):
        raise ToolRefused("This does not follow the last messages. Ask what they mean, or send only what those messages are about.")
    if a["to"] in ctx.told_by_tasks:
        raise ToolRefused("Not sent: the task system is telling this person about the order and the cancel right now, with what "
                          "to do. Do not message them about it.")
    if not ctx.is_system and await _approval_just_asked(ctx, a["to"]):
        # sim 2026-10-12: Maa tapped Yes, the order needed her son's OK, and he got two requests (the brain's and the task's)
        raise ToolRefused("Not sent: the approval request goes to this person from the task system right now, with the order and "
                          "the amount. Do not message them about it; tell the person you are talking to that it was sent.")
    problems = await send_problems(ctx, a["to"], a["text"])
    if problems:
        raise ToolRefused("Not sent: " + "; ".join(problems) + ".")
    a = {**a, "text": await _grounded_send(ctx, str(a["to"]), str(a["text"]))}
    greeted, mid = guards.opening_state(await store.recent_turns(ctx.session, ctx.family_id, a["to"], limit=12), clock.now(), day_of=clock.ist_day)
    a = {**a, "text": guards.dialect_tidy(guards.whatsapp_format(guards.tidy_opening(a["text"], greeted_today=greeted, mid_conversation=mid)),
                                          ctx.profiles.get(a["to"]))}
    payload = {"to": a["to"], "text": a["text"]}
    if a.get("voice"):
        payload["voice"] = True
    if (a.get("buttons") or {}).get("kind") == "later":
        # only on the question of a booking set for later (sim 2026-10-12: they went on a ride's fare confirm for now)
        from app.tasks import later, runtime

        try:
            t = await _task(ctx, str(a["buttons"].get("key") or "").partition(":")[0])
        except ToolRefused:
            t = None
        if t is None or t.input_needed != "later":
            a = {**a, "buttons": None}
        else:
            a = {**a, "buttons": {"kind": "later", "key": f"{t.id}:{later.suggest_max(runtime._later_amount(t)) or ''}"}}
    if (a.get("buttons") or {}).get("kind") == "order":
        # only on the one confirm with its cart (sim 2026-10-14: yes / change / cancel went on the options message, before
        # any cart or total existed)
        from app.tasks import runtime

        try:
            t = await _task(ctx, str(a["buttons"].get("key") or "").partition(":")[0])
        except ToolRefused:
            t = None
        fp = str(((t.details or {}) if t else {}).get("cart_fp") or "")[:8]
        if t is None or t.input_needed != "confirm" or not fp:
            a = {**a, "buttons": None}
        else:
            a = {**a, "buttons": {"kind": "order", "key": f"{t.id}:{fp}"}}
    if a.get("ask_location"):
        payload["location_request"] = True
    elif (a.get("buttons") or {}).get("kind") and (a.get("buttons") or {}).get("key"):
        from app.care import outcomes

        payload["buttons"] = outcomes.buttons(a["buttons"]["kind"], a["buttons"]["key"], ctx.profiles.get(a["to"]), a["text"])
    template = await template_hint(ctx, a["to"], a["text"], a.get("reminder") if isinstance(a.get("reminder"), dict) else None)
    if template:
        payload["template"] = template
    # the backend's delivery ledger (trust plan B1.1): what this message is, so a check-in counts towards the no-reply
    # escalation and the wamid it returns links WhatsApp's receipts (delivered, read) back to this turn
    payload["kind"] = delivery_kind(ctx.message_ref if ctx.is_system else None)
    if ctx.is_system and ctx.message_ref:
        payload["ref"] = ctx.message_ref[:160]
    res = await ctx.host.call(
        "send_whatsapp", payload,
        family_id=ctx.family_id.removeprefix("shadow:"), subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )
    if res.get("reason") == "outside_window":
        # nothing reached them: not part of their chat, and it does not count as said (a check-in is tried again later)
        note = ("Not sent: their WhatsApp chat is closed (no message from them in 24 hours), and nothing can "
                "reach them until they write.")
        if payload.get("voice"):
            note += " The voice note was not sent either."
        return {**res, "note": note}
    ctx.messaged.add(a["to"])
    waiting = res.get("via") == "template"
    turn_id = await store.add_turn(
        ctx.session, family_id=ctx.family_id, thread_id=a["to"], role="assistant", text=a["text"],
        meta={"proactive": True, "delivered": res.get("delivered"), **({"via": "template", "waiting": True} if waiting else {}),
              **({"ref": ctx.message_ref} if ctx.is_system and ctx.message_ref else {}),
              **({"wamid": res["wamid"]} if res.get("wamid") else {})},
    )
    if waiting:
        res = {**res, "note": "Their WhatsApp chat was closed (no message from them in 24 hours): they got a short notice, "
                              "and this message reaches them when they write or tap it."}
    try:
        from app.learn import scoring, situations

        to_role = "elder" if a["to"] == ctx.elder_id else "caregiver"
        from app.core import trace as turn_trace

        secrets = set().union(*[turn_trace.secrets_in(x.get("args")) for x in ctx.actions]) if ctx.actions else set()
        why = turn_trace.scrub(ctx.prompt if ctx.is_system else ctx.user_text, secrets)  # the scheduler's prompt, or the request to pass it on
        trace = [{"role": "user", "text": why[:4000]}] if why else []
        if ctx.tool_texts:
            trace.append({"role": "tool", "results": [{"name": t.split(":", 1)[0], "content": turn_trace.scrub(t[:4000], secrets)}
                                                      for t in ctx.tool_texts[-12:]]})
        await scoring.record(ctx.session, family_id=ctx.family_id, thread_id=a["to"], turn_id=turn_id,
                             kind="proactive" if ctx.is_system else "relay", situation=ctx.situation if ctx.is_system else "relay",
                             speaker_role=to_role, lang=situations.lang_of(ctx.profiles.get(a["to"])), user_text=why, text=a["text"],
                             tools=["send_message"], playbook_version=ctx.playbook_version, arm=ctx.arm, trace=trace or None,
                             judge_band=ctx.judge_band or None)
    except Exception:  # noqa: BLE001 — learning must never stop a message
        logger.exception("learning log failed (send_message)")
    if payload.get("voice"):
        heard = res.get("voice")
        if heard == "sent":
            extra = "These words also went as a WhatsApp voice note."
        elif heard in ("none", "failed", "stale"):
            extra = "The text was sent. The voice note could not be made this time; do not say they heard it."
        else:
            extra = "These words were sent to go as a WhatsApp voice note as well."
        res = {**res, "note": (str(res["note"]) + " " if res.get("note") else "") + extra}
    return res
