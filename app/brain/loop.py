"""One brain turn: build context from Care Memory, think with tools, check the reply, remember it."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from datetime import timedelta

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain import guards, policy, tools
from app.brain.host import ToolHost
from app.brain.persona import PERSONA, REPLY_FORMAT
from app.care import digest, importer, store
from app.care.domains import SLOTS
from app.care.models import Turn
from app.core import clock
from app.llm import router

logger = logging.getLogger(__name__)

MAX_STEPS = 10
# Thinking depth for brain turns. Scenarios pass at low; raise per deployment if quality needs it.
BRAIN_EFFORT = os.getenv("BRAIN_EFFORT", "low")
HISTORY_TURNS = 40
COMPACT_AFTER = 80


@dataclass
class TurnRequest:
    family_id: str
    elder: dict
    speaker: dict
    members: list[dict]
    text: str
    message_ref: str | None = None
    images: list[dict] = field(default_factory=list)  # [{"mime", "data"}] base64
    channel: str = "whatsapp"


@dataclass
class TurnResult:
    reply: str
    actions: list[dict]
    alerts: list[dict]
    model: str = ""
    duplicate: bool = False
    shadow_writes: list[dict] = field(default_factory=list)
    ms: int = 0


async def _lock_family(session: AsyncSession, family_id: str) -> None:
    """One turn at a time per family, across instances; released at commit."""
    await session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": family_id})


async def medicine_times(session: AsyncSession, req: TurnRequest) -> dict[str, set[int]]:
    rows = await store.facts(session, req.family_id, req.elder["id"], domains=["medicine"], statuses=("active",))
    if req.speaker["id"] != req.elder["id"]:
        rows += await store.facts(session, req.family_id, req.speaker["id"], domains=["medicine"], statuses=("active",))
    return guards.med_times([(str(f.value.get("name") or f.key.split(":", 1)[1]), f.text + " " + " ".join(map(str, f.value.get("times") or []))) for f in rows])


async def writing_profiles(session: AsyncSession, family_id: str, people: list[dict]) -> dict[str, dict]:
    """How each person writes, from their last messages to Saheli."""
    out = {}
    for p in people:
        turns = [t for t in await store.recent_turns(session, family_id, p["id"], limit=16) if t.role == "user"][-8:]
        prof = guards.profile([t.text for t in turns])
        if prof:
            out[p["id"]] = prof
    return out


async def family_block(session: AsyncSession, req: TurnRequest) -> tuple[str, str, list[str]]:
    """The cached block: household, care record, notes. Also returns text for the reply guard and avoid-words."""
    rows = await store.facts(session, req.family_id, req.elder["id"])
    note_rows = await store.notes(session, req.family_id, [req.elder["id"], "family", req.speaker["id"]])
    record = digest.care_record(req.elder.get("name", "care recipient"), rows)
    block = "\n\n".join(
        p for p in (digest.household(req.elder, req.members, req.speaker["id"]), record, digest.notes_block(note_rows)) if p
    )
    avoid: list[str] = []
    for f in rows:
        if f.domain == "naming" and f.status == "active":
            avoid += [w for w in (f.value.get("avoid") or []) if isinstance(w, str)]
    known = record + "\n" + "\n".join(n.body_md for n in note_rows)
    member_names = {w.lower() for m in req.members for w in (m.get("name") or "").split()} | {w.lower() for w in (req.elder.get("name") or "").split()}
    unreachable = sorted({
        n.split()[0] for f in rows if f.domain in ("home", "contact", "doctor") and f.status == "active"
        for n in [str(f.value.get("name") or "")] if n and n.split()[0].lower() not in member_names and len(n.split()[0]) > 2
    })
    return block, known, avoid + [f"__unreachable__{u}" for u in unreachable]


async def turn_context(session: AsyncSession, req: TurnRequest) -> tuple[str, str]:
    now = clock.ist()
    day_events = await store.events(session, req.family_id, req.elder["id"], day=clock.ist_day())
    loops = await store.live_loops(session, req.family_id, [req.elder["id"], req.speaker["id"]])
    hits = await store.recall(session, req.family_id, [req.elder["id"], "family"], req.text, limit=8) if req.text.strip() else []
    from app.tasks import runtime as task_runtime

    tasks = await task_runtime.live_tasks(session, req.family_id)
    parts = [
        f"NOW: {now.strftime('%A %d %B %Y, %H:%M')} IST",
        f"SPEAKING: {req.speaker.get('name')} ({'the care recipient' if req.speaker['id'] == req.elder['id'] else req.speaker.get('role', 'family')}), via {req.channel}",
        digest.ledger(day_events),
        digest.loops(loops),
        ("ACTIVE TASKS (orders and rides running in the background):\n" + "\n".join(f"  - {task_runtime.describe(t)}" for t in tasks))
        if tasks else "ACTIVE TASKS: none.",
    ]
    if hits:
        parts.append("POSSIBLY RELEVANT MEMORY:\n" + "\n".join(f"  [{clock.ist(h.when).strftime('%d %b')}] {h.text}" for h in hits))
    if req.speaker["id"] != req.elder["id"] and req.speaker.get("role") != "system":
        own = await store.facts(session, req.family_id, req.speaker["id"], statuses=("active", "pending"))
        if own:
            own_today = await store.events(session, req.family_id, req.speaker["id"], day=clock.ist_day())
            parts.append(
                f"YOUR OWN CARE (the speaker's self care; use about={req.speaker['id']} for anything about them):\n"
                + digest.care_record(req.speaker.get("name") or "the speaker", own)
                + ("\n" + digest.ledger(own_today) if own_today else "")
            )
    if req.speaker["id"] != req.elder["id"]:
        elder_turns = await store.recent_turns(session, req.family_id, req.elder["id"], limit=12)
        if elder_turns:
            parts.append(
                f"RECENT WITH {req.elder.get('name', 'the elder').upper()}:\n"
                + "\n".join(f"  {clock.ist(t.at).strftime('%d %b %H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text[:300]}" for t in elder_turns)
            )
    known = "\n".join(h.text for h in hits) + "\n" + "\n".join(e.summary for e in day_events)
    return "\n\n".join(parts), known


def writing_block(req: TurnRequest, profiles: dict[str, dict]) -> str:
    people = [req.elder] + [m for m in req.members if m.get("id") != req.elder.get("id")]
    lines = [f"  {p.get('name')}: {guards.describe(profiles.get(p['id']))}" for p in people if p.get("id")]
    return "HOW EACH PERSON WRITES (write to each person this way, in replies and in send_message):\n" + "\n".join(lines)


async def history(session: AsyncSession, req: TurnRequest) -> list[dict]:
    summ = await store.summary(session, req.family_id, req.speaker["id"])
    turns = await store.recent_turns(
        session, req.family_id, req.speaker["id"], after_id=summ.covers_until_turn if summ else 0, limit=HISTORY_TURNS
    )
    msgs: list[dict] = []
    if summ and summ.summary:
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"(Earlier in our conversation, summarized: {summ.summary})"}]})
        msgs.append({"role": "assistant", "text": "Noted.", "tool_calls": []})
    for t in turns:
        if t.message_ref and t.message_ref == req.message_ref:
            continue
        stamp = clock.ist(t.at).strftime("%d %b %H:%M")
        if t.role == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": f"[{stamp}] {t.text}"}]})
        else:
            msgs.append({"role": "assistant", "text": t.text, "tool_calls": []})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


CLAIM_WINDOW = timedelta(hours=3)

# Flood limits: a stuck sender, an echo loop between bots, or a wake-up storm must not run up model spend.
PERSON_BURST, PERSON_WINDOW = 25, timedelta(minutes=10)
SYSTEM_BURST, SYSTEM_WINDOW = 30, timedelta(hours=1)
PAUSE_NOTE = {
    "devanagari": "एक साथ बहुत सारे मैसेज आ रहे हैं, मैं कुछ मिनट रुक रही हूँ। ज़रूरी हो तो परिवार को फ़ोन कीजिए 🙏",
    "indic": "Bahut saare message ek saath aa rahe hain, main kuch minute ruk rahi hoon. Zaroori ho to family ko call kijiye 🙏",
    "english": "I'm getting a lot of messages at once, so I'll pause for a few minutes. If it's urgent, please call your family. 🙏",
}


async def flood_reply(session: AsyncSession, req: TurnRequest) -> str | None:
    """A canned reply (no model call) when this sender is over the limit; None to answer normally."""
    system = req.speaker.get("role") == "system"
    window, burst = (SYSTEM_WINDOW, SYSTEM_BURST) if system else (PERSON_WINDOW, PERSON_BURST)
    if not system and guards.RED_FLAG_WORDS.search(req.text or ""):
        return None  # never throttle someone describing an emergency
    count = (await session.execute(
        select(func.count()).select_from(Turn).where(
            Turn.family_id == req.family_id, Turn.thread_id == req.speaker["id"], Turn.role == "user", Turn.at >= clock.now() - window)
    )).scalar_one()
    if count <= burst:
        return None
    logger.error("flood limit family=%s thread=%s count=%s window=%s", req.family_id, req.speaker["id"], count, window)
    if system:
        return "none"
    if count > burst + 1:
        return "🙏"
    p = guards.profile([req.text]) or {}
    key = "devanagari" if p.get("script") == "devanagari" else ("indic" if p.get("roman") == "indic" else "english")
    return PAUSE_NOTE[key]


async def recently_messaged(session: AsyncSession, req: TurnRequest) -> set[str]:
    """People Saheli sent a message to in the last few hours (so 'I have told Asha' can be true)."""
    since = clock.now() - CLAIM_WINDOW
    rows = await session.execute(
        select(Turn.thread_id).where(Turn.family_id == req.family_id, Turn.role == "assistant", Turn.at >= since, Turn.thread_id != req.speaker["id"])
    )
    return set(rows.scalars())


async def guard_problems(session: AsyncSession, req: TurnRequest, ctx: tools.TurnCtx, final: str, *, avoid: list[str], unreachable: list[str]) -> list[str]:
    """Everything wrong with a reply before the person reads it (policy + guards)."""
    known = "\n".join(ctx.known)
    problems = policy.reply_problems(final, known_text=known, avoid_words=avoid, user_text=req.text, unreachable=unreachable)
    if guards.leaked_reasoning(final):
        problems.append("the reply contains your private notes (thoughts, the time, who is speaking); write only the message to the person")
    fresh = req.text + "\n" + "\n".join(a.get("args", {}).__repr__() for a in ctx.actions)
    now = clock.ist()
    problems += guards.ungrounded(
        final, known=known, fresh=fresh, meds=ctx.meds, now_minutes=now.hour * 60 + now.minute,
        allowed_times={int(v[:2]) * 60 + int(v[3:]) for v in SLOTS.values()},
    )
    problems += guards.language_problems(final, ctx.profiles.get(req.speaker["id"]), who="they")
    earlier = [t.text for t in await store.recent_turns(session, req.family_id, req.speaker["id"], limit=20)
               if t.role == "assistant" and t.at >= clock.now() - timedelta(hours=12)][-8:]
    problems += guards.repeats(final, earlier)
    me = req.speaker["id"]
    others = {}
    for m in [req.elder, *req.members]:
        first = ((m.get("name") or "").split() or [""])[0].lower()
        if m.get("id") and m["id"] != me and len(first) > 2:
            others[first] = m["id"]
    messaged = {a["args"].get("to") for a in ctx.actions if a["tool"] == "send_message" and a.get("ok")}
    if any(a["tool"] == "alert_caregiver" and a.get("ok") for a in ctx.actions):
        messaged |= {m["id"] for m in req.members if m.get("id") != req.elder["id"]}
    messaged |= await recently_messaged(session, req)
    from app.tasks import runtime as task_runtime

    ordering_ok = any(a["tool"] in ("start_task", "task_input") and a.get("ok") for a in ctx.actions) or bool(await task_runtime.live_tasks(session, req.family_id))
    problems += guards.false_claims(final, others=others, messaged=messaged, ordering_ok=ordering_ok)
    seen, out = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


async def run_turn(session: AsyncSession, host: ToolHost, req: TurnRequest) -> TurnResult:
    started = time.monotonic()
    await _lock_family(session, req.family_id)

    inserted = await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="user", text=req.text,
        speaker_id=req.speaker["id"], message_ref=req.message_ref, meta={"channel": req.channel, "images": len(req.images)},
    )
    if req.message_ref and inserted is None:
        prior = (
            await session.execute(
                select(Turn).where(Turn.family_id == req.family_id, Turn.meta["reply_to"].astext == req.message_ref)
            )
        ).scalar_one_or_none()
        await session.commit()
        return TurnResult(reply=prior.text if prior else "", actions=[], alerts=[], duplicate=True)

    canned = await flood_reply(session, req)
    if canned is not None:
        await store.add_turn(session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=canned,
                             meta={"reply_to": req.message_ref, "model": "none", "throttled": True})
        await session.commit()
        return TurnResult(reply=canned, actions=[], alerts=[], model="none", ms=int((time.monotonic() - started) * 1000))

    if req.speaker.get("role") != "system":
        await store.save_roster(session, req.family_id, req.elder, req.members)
    if not await importer.already_imported(session, req.family_id, req.elder["id"]):
        try:
            async with session.begin_nested():
                await importer.import_family(
                    session, host, family_id=req.family_id,
                    backend_family_id=req.family_id.removeprefix("shadow:"), elder_id=req.elder["id"],
                )
        except Exception:  # noqa: BLE001 — a failed import must not cost the person their reply
            logger.exception("care import failed family=%s", req.family_id)

    fam_block, known_record, avoid = await family_block(session, req)
    unreachable = [w.removeprefix("__unreachable__") for w in avoid if w.startswith("__unreachable__")]
    avoid = [w for w in avoid if not w.startswith("__unreachable__")]
    dynamic, known_turn = await turn_context(session, req)
    msgs = await history(session, req)
    content = [{"type": "text", "text": f"[{clock.ist().strftime('%d %b %H:%M')}] {req.text}"}]
    content += [{"type": "image", "mime": i["mime"], "data": i["data"]} for i in req.images]
    msgs.append({"role": "user", "content": content})

    is_system = req.speaker.get("role") == "system"
    profiles = await writing_profiles(session, req.family_id, [req.elder, *[m for m in req.members if m.get("id") != req.elder["id"]]])
    if not is_system:
        cur = guards.profile([req.text])
        if cur and req.speaker["id"] not in profiles:
            profiles[req.speaker["id"]] = cur
    dynamic += "\n\n" + writing_block(req, profiles)
    tool_texts: list[str] = []
    ctx = tools.TurnCtx(
        session=session, host=host, family_id=req.family_id, elder=req.elder, speaker=req.speaker,
        members=req.members, message_ref=req.message_ref, user_text="" if is_system else req.text,
        known=[fam_block, known_record, known_turn, dynamic, req.text, *[_msg_text(m) for m in msgs]],
        meds=await medicine_times(session, req), profiles=profiles,
    )
    ctx.known.append("")  # tool results are appended as they come
    stable = [PERSONA + "\n\n" + REPLY_FORMAT, fam_block]
    guarded = False
    reply: router.LLMReply | None = None
    for step in range(MAX_STEPS):
        last = step == MAX_STEPS - 1
        reply = await router.complete(
            "brain",
            system_stable=stable,
            system_dynamic=dynamic + ("\n\nYou have used all your steps: reply to the person now with what you have." if last else ""),
            messages=msgs,
            # On the last step there are no tools, so the person always gets a reply.
            tools=None if last else tools.specs(),
            max_tokens=6000,
            effort=BRAIN_EFFORT,
        )
        msgs.append(reply.as_message())
        if not reply.tool_calls:
            text_now = (reply.text or "").strip()
            if guarded or is_system or not text_now or last:
                break
            problems = await guard_problems(session, req, ctx, text_now, avoid=avoid, unreachable=unreachable)
            if not problems:
                break
            guarded = True
            logger.warning("reply guard family=%s problems=%s", req.family_id, problems)
            msgs.append({"role": "user", "content": [{"type": "text", "text": (
                "(Check before sending, the person has not seen this yet: " + "; ".join(problems) + ". If you said you would do "
                "something, do it now with the tool; otherwise drop the claim. Then write only the corrected message to the person.)"
            )}]})
            continue
        results = []
        searches = sum(1 for a in ctx.actions if a["tool"] in ("recall", "search_records"))
        for call in reply.tool_calls:
            if call.name in ("recall", "search_records") and searches >= 4:
                # Searching memory again rarely finds what four searches did not: answer with what you have.
                results.append({"id": call.id, "name": call.name, "is_error": True,
                                "content": json.dumps({"refused": "You have searched enough. If it is not there, say plainly you do not have it and ask."})})
                continue
            out, is_error = await tools.run(ctx, call.name, call.args)
            tool_texts.append(out)
            ctx.known.append(out)
            results.append({"id": call.id, "name": call.name, "content": out, "is_error": is_error})
        msgs.append({"role": "tool", "results": results})

    final = (reply.text if reply else "").strip()
    if guarded and not final:
        # The fix came back empty: fall back to the last non-empty text the model wrote.
        final = next((m.get("text", "").strip() for m in reversed(msgs) if m.get("role") == "assistant" and (m.get("text") or "").strip()), "")

    await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=final,
        meta={"reply_to": req.message_ref, "model": reply.model if reply else "", "actions": [a["tool"] for a in ctx.actions]},
    )
    await maybe_compact(session, req.family_id, req.speaker["id"])
    await session.commit()
    return TurnResult(
        reply=final,
        actions=ctx.actions,
        alerts=ctx.alerts,
        model=reply.model if reply else "",
        shadow_writes=getattr(host, "would_have", []),
        ms=int((time.monotonic() - started) * 1000),
    )


async def maybe_compact(session: AsyncSession, family_id: str, thread_id: str) -> None:
    """Fold the oldest turns into the thread summary once the raw history gets long."""
    summ = await store.summary(session, family_id, thread_id)
    after = summ.covers_until_turn if summ else 0
    count = (
        await session.execute(
            select(func.count()).select_from(Turn).where(Turn.family_id == family_id, Turn.thread_id == thread_id, Turn.id > after)
        )
    ).scalar_one()
    if count <= COMPACT_AFTER:
        return
    old = await store.recent_turns(session, family_id, thread_id, after_id=after, limit=count)
    fold = old[: count - HISTORY_TURNS]
    transcript = "\n".join(f"{clock.ist(t.at).strftime('%d %b %H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text}" for t in fold)
    try:
        out = await router.complete(
            "extract",
            system_stable="You keep the running summary of a conversation between Saheli, a care companion, and one person. "
            "Merge the previous summary and the new messages into one summary under 250 words: what matters for care, "
            "promises made, questions still open, their mood and situation, people mentioned. Plain text.",
            messages=[{"role": "user", "content": [{"type": "text", "text": f"Previous summary:\n{summ.summary if summ else '(none)'}\n\nNew messages:\n{transcript}"}]}],
            max_tokens=1200,
            effort="low",
        )
        await store.set_summary(session, family_id, thread_id, out.text.strip(), fold[-1].id)
    except router.AllModelsFailed:
        logger.warning("compaction skipped family=%s thread=%s", family_id, thread_id)


def _msg_text(m: dict) -> str:
    if m.get("role") == "assistant":
        return m.get("text") or ""
    return " ".join(c.get("text", "") for c in m.get("content") or [] if c.get("type") == "text")


def result_json(r: TurnResult) -> dict:
    return json.loads(json.dumps(r.__dict__, default=str))
