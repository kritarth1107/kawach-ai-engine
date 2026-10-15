"""One brain turn: build context from Care Memory, think with tools, check the reply, remember it."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

from datetime import datetime, timedelta
from typing import Awaitable, Callable

from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain import grounding, guards, policy, tools
from app.brain.host import ToolHost
from app.brain.persona import PERSONA, REPLY_FORMAT
from app.care import digest, freshness, importer, links, store
from app.care.domains import SLOTS
from app.care.models import Turn
from app.core import clock, trace
from app.llm import router

logger = logging.getLogger(__name__)

MAX_STEPS = 10
# Thinking depth for brain turns. Scenarios pass at low; raise per deployment if quality needs it.
BRAIN_EFFORT = os.getenv("BRAIN_EFFORT", "low")
# Hard turns (health, a correction, an upset person) think harder (trust plan C6). Off (same as BRAIN_EFFORT) until the
# paid A/B on the test world shows medium is at least 5 points better at p90 <= 15 s (founder decision D3).
BRAIN_EFFORT_HARD = os.getenv("BRAIN_EFFORT_HARD", BRAIN_EFFORT)
HISTORY_TURNS = 40
# Small talk and acks, from the situation tagger's existing labels. A greeting does not re-read the whole thread,
# think hard, or wait on the reply-path judge. A tool call still runs; the reply after it thinks at the normal depth.
LIGHT_SITUATIONS = frozenset({"thanks", "chit_chat"})
LIGHT_HISTORY = 8
LIGHT_MAX_TOKENS = 1500
COMPACT_AFTER = 80
# How long a turn may take (seconds). A person's message: the backend stops waiting at 120 s (aiPostBrainTurn), so past
# that a reply is wasted while the family stays locked. A scheduled turn: under Cloud Run's 300 s request limit.
TURN_BUDGET_S = {"message": float(os.getenv("BRAIN_TURN_BUDGET_S", "110")), "system": 240.0}
# One model call. Live 15 Oct afternoon: a call that should be ~3 s sat for 54–78 s inside a 110 s budget, and the
# person waited the whole time. 20 s is past a normal call (morning p90 12.5 s) and short of that stall.
CALL_CAP_S = float(os.getenv("BRAIN_CALL_CAP_S", "20"))
# Of that, how long it may wait for the family's previous turn to finish.
LOCK_WAIT_S = {"message": 90.0, "system": 150.0}
MIN_CALL_S = 8.0  # less left than this: no new model call is started


@dataclass
class TurnRequest:
    family_id: str
    elder: dict
    speaker: dict
    members: list[dict]
    text: str
    message_ref: str | None = None
    images: list[dict] = field(default_factory=list)  # [{"mime", "data"}] base64: image, video, audio or PDF they sent
    media_note: str = ""  # media they sent that could not be opened
    channel: str = "whatsapp"
    modality: str = "text"  # text | voice (text is then the transcript of a voice note)
    voice_confidence: float | None = None  # 0..1 when the speech engine reported one
    voice_language: str | None = None
    # A system turn about something that can move on while it waits for this family's turn (a task update): checked once
    # the turn is held; False → nothing is said (founder 2026-10-11: one true status).
    still_true: Callable[[AsyncSession], Awaitable[bool]] | None = None
    # reminders due in the next hours for the speaker and the person cared for (from the backend, with the message):
    # [{"id", "person", "text", "day", "time", "repeat", "item"}]
    reminders_soon: list[dict] = field(default_factory=list)
    # a WhatsApp location pin in this message {"lat", "lng", "name", "address"} (founder 2026-10-12: a ride's pickup)
    location: dict | None = None
    onboarding: bool = False  # the speaker has not finished setting up: Saheli runs the setup chat
    # messages that could not reach the speaker while their WhatsApp chat was closed (they got a template notice):
    # [{"purpose", "text", "at"}] from the backend, given in this turn
    waiting: list[dict] = field(default_factory=list)

    @property
    def voice(self) -> bool:
        return self.modality == "voice"

    @property
    def voice_unsure(self) -> bool:
        """A voice note the speech engine was not sure about (or that came out very short): check before acting on it."""
        return self.voice and ((self.voice_confidence is not None and self.voice_confidence < VOICE_SURE)
                               or len((self.text or "").split()) <= 1)


@dataclass
class TurnResult:
    reply: str
    actions: list[dict]
    alerts: list[dict]
    model: str = ""
    duplicate: bool = False
    shadow_writes: list[dict] = field(default_factory=list)
    ms: int = 0
    messages: list[str] = field(default_factory=list)  # WhatsApp bubbles, answer first; a waiting brief is its own later bubble
    buttons: list[dict] = field(default_factory=list)  # one-tap WhatsApp answers sent with the reply
    trace_id: str | None = None  # the why-trail this turn is filed under (app.core.trace)
    turn_id: int | None = None  # Saheli's reply turn
    stale: bool = False  # an update that was no longer true when its turn came: nothing was said
    location_request: bool = False  # the reply goes with WhatsApp's "Send location" button


class FamilyBusy(router.AllModelsFailed):
    """The family's turn did not come in time: another turn held it. Answered like a model outage (503: the backend's
    holding line and retry), never queued for minutes behind a stuck turn (live 12 Oct 20:33: one turn hung, held the
    family for 10 minutes and five messages waited behind it)."""


async def _lock_family(session: AsyncSession, family_id: str, wait_s: float | None = None) -> None:
    """One turn at a time per family, across instances; released at commit. wait_s bounds the wait (the queue keeps
    its order: Postgres hands the lock to waiters first come, first served)."""
    if not wait_s:
        await session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": family_id})
        return
    before = (await session.execute(text("SELECT current_setting('lock_timeout')"))).scalar_one()
    await session.execute(text("SELECT set_config('lock_timeout', :t, true)"), {"t": f"{int(wait_s * 1000)}ms"})
    try:
        await session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": family_id})
    except DBAPIError as exc:
        orig = getattr(exc, "orig", None)
        if getattr(orig, "sqlstate", None) == "55P03" or getattr(orig, "pgcode", None) == "55P03" or "lock timeout" in str(exc):
            await session.rollback()
            logger.error("family busy for %.0fs: turn not taken family=%s", wait_s, family_id)
            raise FamilyBusy(f"family busy for {wait_s:.0f}s") from exc
        raise
    await session.execute(text("SELECT set_config('lock_timeout', :t, true)"), {"t": before})


class TurnData:
    """What one turn reads from memory, fetched once and shared by the context, the guards and the tools' checks
    (the turn used to read the same facts, threads and tasks several times)."""

    def __init__(self, session: AsyncSession, req: TurnRequest) -> None:
        self.session, self.req = session, req
        self._facts: dict[str, list] = {}
        self._turns: dict[str, list] | None = None
        self._tasks: list | None = None
        self.history_full = True  # set by history(): did the thread fill a whole history window?

    async def facts(self, subject: str) -> list:
        if subject not in self._facts:
            self._facts[subject] = await store.facts(self.session, self.req.family_id, subject)
        return self._facts[subject]

    async def turns(self, thread: str) -> list:
        """The last HISTORY_TURNS turns of a thread; every household thread is read in one query."""
        if self._turns is None:
            people = {self.req.speaker["id"], self.req.elder["id"], *(m.get("id") for m in self.req.members if m.get("id"))}
            self._turns = await store.recent_turns_many(self.session, self.req.family_id, list(people), limit=HISTORY_TURNS)
        if thread not in self._turns:
            self._turns[thread] = await store.recent_turns(self.session, self.req.family_id, thread, limit=HISTORY_TURNS)
        return self._turns[thread]

    async def tasks(self) -> list:
        if self._tasks is None:
            from app.tasks import runtime as task_runtime

            self._tasks = await task_runtime.shown_tasks(self.session, self.req.family_id)
        return self._tasks


async def medicine_times(data: TurnData) -> dict[str, set[int]]:
    req = data.req
    rows = [f for f in await data.facts(req.elder["id"]) if f.domain == "medicine" and f.status == "active"]
    if req.speaker["id"] != req.elder["id"] and req.speaker.get("role") != "system":
        rows += [f for f in await data.facts(req.speaker["id"]) if f.domain == "medicine" and f.status == "active"]
    return guards.med_times([(str(f.value.get("name") or f.key.split(":", 1)[1]), f.text + " " + " ".join(map(str, f.value.get("times") or []))) for f in rows])


async def writing_profiles(session: AsyncSession, family_id: str, people: list[dict], data: TurnData | None = None) -> dict[str, dict]:
    """How each person writes (their last messages to Saheli) plus their saved language, dialect and script (care record)."""
    from app.care import language

    out = {}
    for p in people:
        recent = (await data.turns(p["id"]))[-16:] if data else await store.recent_turns(session, family_id, p["id"], limit=16)
        turns = [t for t in recent if t.role == "user"][-8:]
        # A voice-note transcript is the speech engine's guess. It must not teach how they write, and it must not
        # override a saved language (live 2026-10-14: English speech transcribed as Devanagari, then every reply in Hindi).
        typed = [t for t in turns if not (t.meta or {}).get("voice")]
        prof = guards.profile([t.text for t in typed]) or {}
        facts = await data.facts(p["id"]) if data else await store.facts(session, family_id, p["id"])
        saved = next((f for f in facts if f.domain == "language" and f.status == "active"), None)
        saved_speech = language.normalise(saved.value) if saved else {}
        if saved:
            prof = {"script": prof.get("script"), "roman": prof.get("roman"), "saved": saved_speech}
        latest = turns[-1] if turns else None
        if latest and (latest.meta or {}).get("voice") and not saved_speech.get("language"):
            heard = guards.script_of(latest.text)
            learned = prof.get("script")
            # A transcript answers this turn in its script only when we have not already learned another way they write.
            # English they typed is not replaced by a Devanagari transcript (live 2026-10-14).
            if heard and heard != "latin" and learned in (None, heard):
                prof = {**prof, "now": heard, "transcript": True}
            elif typed:
                prof["now"] = guards.script_of(typed[-1].text)
        elif typed and (prof or saved):
            prof["now"] = guards.script_of(typed[-1].text)  # what they typed wins over the saved script for this reply
        if prof and turns:
            mine = [t.text for t in recent if t.role == "assistant" and (t.text or "").strip() not in ("", "none")][-5:]
            if replied := guards.usual_script(mine):
                prof["replied"] = replied  # the script Saheli has been writing to them in (keeps a short reply in it)
        if prof:
            out[p["id"]] = prof
    return out


async def _rejected(session: AsyncSession, family_id: str, subject_id: str) -> list:
    from app.care import rejected as care_rejected

    return await care_rejected.rejected(session, family_id, subject_id)


async def family_block(session: AsyncSession, req: TurnRequest, data: TurnData | None = None) -> tuple[str, str, list[str]]:
    """The cached block: household, care record, notes. Also returns text for the reply guard and avoid-words."""
    rows = await data.facts(req.elder["id"]) if data else await store.facts(session, req.family_id, req.elder["id"])
    note_rows = await store.notes(session, req.family_id, [req.elder["id"], "family", req.speaker["id"]])
    stale = await freshness.stale_facts(session, req.family_id, req.elder["id"], rows)
    from app.care import rejected as care_rejected

    wrong = await care_rejected.rejected(session, req.family_id, req.elder["id"])
    record = digest.care_record(req.elder.get("name", "care recipient"), rows, stale,
                                await links.links_for(session, req.family_id, req.elder["id"]), rejected=wrong)
    from app.care import naming as care_naming

    who_naming = await care_naming.for_people(session, req.family_id, [req.elder["id"], *[str(m.get("id")) for m in req.members if m.get("id")]])
    block = "\n\n".join(
        p for p in (digest.household(req.elder, req.members, req.speaker["id"], who_naming), record, digest.notes_block(note_rows)) if p
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
    return block, known, avoid + [f"__unreachable__{u}" for u in unreachable] + [f"__rejected__{t}" for t in care_rejected.terms(wrong)]


SYSTEM_ID = "saheli-scheduler"


async def turn_context(session: AsyncSession, req: TurnRequest, data: TurnData | None = None) -> tuple[str, str]:
    data = data or TurnData(session, req)
    now = clock.ist()
    day_events = await store.events(session, req.family_id, req.elder["id"], day=clock.ist_day())
    # A waiting brief is not shown here. The conversation would rewrite it into this reply (and invent a day). It is
    # sent afterwards as its own WhatsApp message, in the words the brief writer already produced.
    loops = [lp for lp in await store.live_loops(session, req.family_id, [req.elder["id"], req.speaker["id"]]) if lp.kind != "brief"]
    from app.care import memory_index

    subjects = list(dict.fromkeys([req.elder["id"], req.speaker.get("id") or req.elder["id"], "family"]))
    hits = await memory_index.search(session, req.family_id, subjects, req.text, limit=8) if req.text.strip() else []
    from app.tasks import runtime as task_runtime

    tasks = await data.tasks()
    dose = await dose_answer_block(data, req)
    parts = [
        f"NOW: {now.strftime('%A %d %B %Y, %H:%M')} IST",
        f"SPEAKING: {req.speaker.get('name')} ({'the care recipient' if req.speaker['id'] == req.elder['id'] else req.speaker.get('role', 'family')}), via {req.channel}",
        voice_block(req),
        digest.ledger(day_events),
        digest.loops(loops),
        ("ACTIVE TASKS (orders and rides running in the background):\n" + "\n".join(f"  - {task_runtime.describe(t)}" for t in tasks))
        if tasks else "ACTIVE TASKS: none.",
        reminders_block(req),
        *([dose] if dose else []),
        onboarding_block(req),
        waiting_block(req),
    ]
    if req.speaker.get("role") != "system":
        from app.care import nudges

        hint = await nudges.reply_hint(session, req.family_id, req.speaker["id"])
        if hint:
            parts.append(hint)
        still = await questions_still_open(session, req.speaker["id"])
        if still:
            parts.append("ALREADY ASKED TODAY, still unanswered. Do not ask any of these again:\n"
                         + "\n".join(f"- {t[:180]}" for t in still[:8]))
    try:  # the nutrition agent's questions still waiting for an answer (log_food answer_to)
        from app.nutrition import jobs as food_jobs

        async with session.begin_nested():
            asked = await food_jobs.open_for(session, req.family_id, list({req.elder["id"], req.speaker["id"]}))
        food_q = food_jobs.context_block(asked, {m.get("id"): m.get("name") for m in [req.elder, *req.members] if m.get("id")})
        if food_q:
            parts.append(food_q)
    except Exception:  # noqa: BLE001 — never costs the turn
        logger.exception("food questions block failed family=%s", req.family_id)
    from app.care import activities as care_activities

    for pid, pname in {req.elder["id"]: req.elder.get("name"), req.speaker["id"]: req.speaker.get("name")}.items():
        if pid == SYSTEM_ID:
            continue
        block = care_activities.context_block(pname or "them", await care_activities.status(session, req.family_id, pid))
        if block:
            parts.append(block)
    if req.speaker.get("role") != "system" and req.speaker.get("id"):
        from app.care import gaps

        missing = await gaps.block(session, req.family_id, req.speaker["id"], req.speaker.get("name") or "them")
        if missing:
            parts.append(missing)
    if req.speaker["id"] != req.elder["id"]:
        # A caregiver asking "what's pending?" gets who acts next on every job (the elder's turns stay short).
        from app.care import work

        names = {m.get("id"): m.get("name") for m in [req.elder, *req.members]}
        parts.append(work.brief(await work.items(session, req.family_id), names, max_lines=6))
    from app.care import baselines as care_baselines
    from app.care import patterns as care_patterns

    for pid, pname in {req.elder["id"]: req.elder.get("name"), req.speaker["id"]: req.speaker.get("name")}.items():
        line = care_baselines.usual_line(pname or "them", await care_baselines.get(session, req.family_id, pid))
        if line:
            parts.append(line)

    noticed = await care_patterns.recent(session, req.family_id, list({req.elder["id"], req.speaker["id"]}))
    if noticed:
        names = {m.get("id"): m.get("name") for m in [req.elder, *req.members]}
        parts.append(care_patterns.context_block(noticed, names))
    from app.care import skillbook

    to = req.elder if req.speaker["id"] == req.elder["id"] or req.speaker.get("role") == "system" else req.speaker
    from app.learn import situations as _situations

    is_sys = req.speaker.get("role") == "system"
    moment = _situations.tag(text=req.text, role="system" if is_sys else ("elder" if req.speaker["id"] == req.elder["id"] else "caregiver"),
                             tools=[], prompt=req.text if is_sys else "")
    # family skills shape tone only; in an emergency or a rule-breaking message they are left out entirely, and they stay
    # out while an emergency is still going on (a red flag or an alert in the last hour), for a photo with no words,
    # and for a scheduled turn about an alert
    safety = moment in ("emergency", "injection") or bool(guards.RED_FLAG_WORDS.search(req.text or ""))
    safety = safety or (bool(req.images) and not (req.text or "").strip()) or (is_sys and "alert" in (req.text or "").lower())
    if not safety:
        hour_ago = clock.now() - timedelta(hours=1)
        safety = any(e.kind in ("alert_whatsapp", "alert_dashboard", "red_flag") and e.at >= hour_ago for e in day_events)
        for thread in {req.elder["id"], req.speaker["id"]}:
            recent = [t for t in await data.turns(thread) if t.role == "user" and t.at >= hour_ago]
            safety = safety or any(guards.RED_FLAG_WORDS.search(t.text or "") for t in recent)
    how = await skillbook.context_block(session, req.family_id, to["id"], (to.get("name") or "").split(" ")[0], safety=safety)
    if how:
        parts.append(how)
    if hits:
        parts.append("POSSIBLY RELEVANT MEMORY (a line marked EARLIER is history, not true now; where a memory disagrees with the "
                     "CARE RECORD, the record is right):\n" + "\n".join(f"  [{clock.ist(h.when).strftime('%d %b')}] {h.text}" for h in hits))
    if req.speaker["id"] != req.elder["id"] and req.speaker.get("role") != "system":
        own = await data.facts(req.speaker["id"])
        if own:
            own_today = await store.events(session, req.family_id, req.speaker["id"], day=clock.ist_day())
            parts.append(
                f"YOUR OWN CARE (the speaker's self care; use about={req.speaker['id']} for anything about them):\n"
                + digest.care_record(req.speaker.get("name") or "the speaker", own,
                                     await freshness.stale_facts(session, req.family_id, req.speaker["id"], own),
                                     await links.links_for(session, req.family_id, req.speaker["id"]),
                                     rejected=await _rejected(session, req.family_id, req.speaker["id"]))
                + ("\n" + digest.ledger(own_today) if own_today else "")
            )
    if req.speaker["id"] != req.elder["id"]:
        elder_turns = (await data.turns(req.elder["id"]))[-12:]
        if elder_turns:
            parts.append(
                f"RECENT WITH {req.elder.get('name', 'the elder').upper()}:\n"
                + "\n".join(f"  {clock.ist(t.at).strftime('%d %b %H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text[:300]}" for t in elder_turns)
            )
    parts.append(
        "WHATSAPP SHAPE: count the separate things they just told you. One short WhatsApp message for each, with a line that is "
        "only --- between them. A meal, a reading, and something at home are three messages, and two facts in one of them fails. "
        "A blank line does not split a message. Never write one sentence that names two facts and then asks to add a person: "
        "that sentence is the failure. Each fact is its own message, and the question about adding someone comes after, as its "
        "own message, only if you still need it. Do not call it stressful or a hassle, and do not say whether a number is good or bad."
    )
    known = "\n".join(h.text for h in hits) + "\n" + "\n".join(e.summary for e in day_events)
    return "\n\n".join(parts), known


def onboarding_block(req: TurnRequest) -> str:
    """Setup on WhatsApp (founder 2026-10-12: "if a new user sends hi, start their onboarding directly"): a chat, one
    question at a time, language first, then whether it is for them or someone they look after (the approved design of
    the dashboard's onboarding chat)."""
    if not req.onboarding or req.speaker.get("role") == "system":
        return ""
    me = req.speaker["id"]
    cared = req.elder["id"] != me
    who = (f"The person they look after is already added: {req.elder.get('name')} (id {req.elder['id']}); the care questions are about "
           "them (about = their id)." if cared else
           "Nobody else is added yet: until they say it is for someone else, it is for themselves (about = their own id).")
    return (
        "SETUP NOT FINISHED: this person is new on WhatsApp or never finished setting up. Run the setup as a warm chat: one "
        "short question per message, in their language, saving each answer with its tool before the next question. "
        f"{who}\n"
        "  1. If they just greeted you: welcome them as Saheli from Kavach in one or two lines (daily check-ins and reminders for "
        "a parent or for themselves, medicines, orders and rides, always asking first), then ask which language they want to "
        "talk in, offering their own bhasha or dialect too (language_preference, about their own id).\n"
        "  2. Their name, if you only have 'Friend' or no name (my_name).\n"
        "  3. Is this for themselves, or for someone they look after (a parent or another family member)?\n"
        "  4. For someone else: their name and what the family calls them, how they are related, and their WhatsApp number with "
        "country code, then add_person; then that person's language or dialect (language_preference about their id).\n"
        "  5. The care basics with setup_progress (about the person cared for, or themselves): conditions, allergies, medicines "
        "with times, their day (wake, meals, sleep), doctor, emergency contact, what they enjoy.\n"
        "  6. When the essentials are saved, or they want to stop: finish_setup, then tell them in a few lines what is set up, "
        "what you will do (reminders at the saved times, check-ins, the nightly brief), the dashboard (app.kavach.care, sign in "
        "with this WhatsApp number), and, if someone was added, the link to share so that person sends you a hi from their own "
        "WhatsApp (you can write to them only after that).\n"
        "If they ask for something else, help with that first, then offer to go on with the setup. If they already use Kavach "
        "with another number, tell them their family can add this number on the dashboard, and set up nothing new. Never ask "
        "for something already saved, never invent details.")


def waiting_block(req: TurnRequest) -> str:
    """Messages that waited while the speaker's WhatsApp chat was closed (founder 2026-10-12: WhatsApp's 24-hour window).
    They got a short notice (a template) instead; this is their first message since, so Saheli gives them now."""
    rows = [w for w in (req.waiting or []) if isinstance(w, dict) and str(w.get("text") or "").strip()]
    if not rows or req.speaker.get("role") == "system":
        return ""
    given = [w for w in rows if w.get("delivered")]
    rows = [w for w in rows if not w.get("delivered")]
    out = []
    if given:
        lines = [f"  - {_waiting_at(w)} ({w.get('purpose') or 'message'}): {str(w.get('text'))[:1500]}" for w in given[:8]]
        out.append("JUST GIVEN TO THE SPEAKER (their WhatsApp chat was closed, so these waited; the system sent them word for "
                   "word right before this message): do not repeat them. They may be answering one of them; if what they wrote "
                   "is about one, answer that.\n" + "\n".join(lines))
    if not rows:
        return "\n\n".join(out)
    lines = []
    for w in rows[:8]:
        try:
            at = clock.ist(datetime.fromisoformat(str(w.get("at")).replace("Z", "+00:00"))).strftime("%d %b %H:%M")
        except ValueError:
            at = "earlier"
        lines.append(f"  - {at} ({w.get('purpose') or 'message'}): {str(w.get('text'))[:1500]}")
    out.append("WAITING FOR THE SPEAKER (their WhatsApp chat was closed, so these could not reach them; they got a short notice "
               "and have now written): give these in this reply, first, in their language, as they stand now (check the care "
               "record, tasks and events: say what changed since, and leave out what no longer applies), then answer what "
               "they wrote.\n" + "\n".join(lines))
    return "\n\n".join(out)


def _waiting_at(w: dict) -> str:
    try:
        return clock.ist(datetime.fromisoformat(str(w.get("at")).replace("Z", "+00:00"))).strftime("%d %b %H:%M")
    except ValueError:
        return "earlier"


def dose_answer_line(last_text: str, meds: list[tuple[str, str, str, list[str]]]) -> str:
    """Saheli's own last message named a saved medicine. The person's words are not read here.

    meds are (name, care-record key, dose, times)."""
    low = (last_text or "").lower()
    hit = [m for m in meds if len(m[0]) >= 4 and m[0].lower() in low]
    if not hit:
        return ""
    lines = []
    for name, key, dose, times in hit[:3]:
        when = ", ".join(times) or "its saved time"
        lines.append(f"{name}{(' ' + dose) if dose else ''} at {when} (change_reminder id {key})")
    return (
        "THE LAST MESSAGE YOU SENT was a dose reminder for " + "; ".join(lines) + ". If they are answering that reminder, "
        "call the tool in this turn. Do not ask which thing they mean, and do not call set_reminder. A new time: "
        "change_reminder, action change, that id, times the new HH:MM. Do not say it changed until change_reminder returns. "
        "They finished it: log_dose outcome taken for that medicine now, at its saved time. Do not ask whether they mean "
        "this medicine. If they are talking about something else, answer that and leave this dose.")


async def dose_answer_block(data: TurnData, req: TurnRequest) -> str:
    if req.speaker.get("role") == "system":
        return ""
    turns = await data.turns(req.speaker["id"])
    last = next((t for t in reversed(turns) if t.role == "assistant" and (t.text or "").strip()), None)
    if last is None:
        return ""
    meds: list[tuple[str, str, str, list[str]]] = []
    seen: set[str] = set()
    for pid in (req.speaker["id"], req.elder["id"]):
        for fact in await data.facts(pid):
            if fact.domain != "medicine" or fact.status != "active":
                continue
            name = str((fact.value or {}).get("name") or "").strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            value = fact.value or {}
            meds.append((name, fact.key, str(value.get("dose") or "").strip(), [str(t) for t in (value.get("times") or [])]))
    return dose_answer_line(last.text, meds)


def reminders_block(req: TurnRequest) -> str:
    """Reminders due in the next hours, so one the person says is already done is skipped instead of going out anyway
    (live 2026-10-12: "i had the medicine" at 00:12, and the reminder for it still came at 00:15)."""
    rows = [r for r in (req.reminders_soon or []) if isinstance(r, dict) and r.get("id")]
    if not rows:
        return ""
    names = {m.get("id"): m.get("name") for m in [req.elder, *req.members]}
    lines = []
    for r in rows[:12]:
        who = "the speaker" if r.get("person") == req.speaker["id"] else (names.get(r.get("person")) or "them")
        item = f"; for {r['item']}" if r.get("item") else ""
        lines.append(f"  [{r['id']}] {r.get('day')} {r.get('time')} for {who}: \"{str(r.get('text') or '')[:140]}\" ({r.get('repeat') or 'daily'}{item})")
    return ("REMINDERS DUE SOON (they go out with these exact words at these times): if someone says they already did what one "
            "is for, or it no longer applies, skip that time with change_reminder (action skip, its id, date and time) and say "
            "it will not come; a repeating one carries on after that.\n" + "\n".join(lines))


VOICE_SURE = 0.6


def voice_block(req: TurnRequest) -> str:
    """How to answer a voice note: it is a transcript (may be wrong), and the reply is spoken back as a voice note."""
    if req.channel == "call":
        return ("THIS IS A LIVE PHONE CALL. What they said is a live transcript (it can have wrong words). Your reply is spoken "
                "at once: one or two short spoken sentences, one question at a time, no lists, no symbols, numbers said as "
                "words. Never say you will message them; you are talking now. If it might be an emergency, act as usual.")
    if not req.voice:
        return ""
    lines = [
        "THIS MESSAGE WAS A VOICE NOTE. The text is a machine transcript and can have wrong words, especially names, "
        "medicines and numbers. Reply in the language HOW EACH PERSON WRITES gives. The transcript's script is not a "
        "request to switch. Your reply is sent as a voice note (read aloud) and as text, so write it to be heard: "
        "short spoken sentences, no emoji, no lists, no links, no symbols; say times and doses simply ('subah aath baje', "
        "'aadhi goli').",
    ]
    if req.voice_unsure:
        lines.append(
            "THE TRANSCRIPT IS UNSURE. If you cannot tell what they meant, kindly ask them to say it again. Before you save or "
            "change any medicine, dose, allergy or time from it, repeat back what you heard and ask them to confirm. An "
            "emergency is the exception: if it might be one, act on it as usual."
        )
    return "\n".join(lines)


def writing_block(req: TurnRequest, profiles: dict[str, dict]) -> str:
    people = [req.elder] + [m for m in req.members if m.get("id") != req.elder.get("id")]
    lines = [f"  {p.get('name')}: {guards.describe(profiles.get(p['id']))}" for p in people if p.get("id")]
    return "HOW EACH PERSON WRITES (write to each person this way, in replies and in send_message):\n" + "\n".join(lines)


async def history(session: AsyncSession, req: TurnRequest, data: TurnData | None = None) -> list[dict]:
    summ = await store.summary(session, req.family_id, req.speaker["id"])
    after = summ.covers_until_turn if summ else 0
    if data:
        turns = [t for t in await data.turns(req.speaker["id"]) if t.id > after]
        data.history_full = len(turns) >= HISTORY_TURNS
    else:
        turns = await store.recent_turns(session, req.family_id, req.speaker["id"], after_id=after, limit=HISTORY_TURNS)
    msgs: list[dict] = []
    if summ and summ.summary:
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"(Earlier in our conversation, summarized: {summ.summary})"}]})
        msgs.append({"role": "assistant", "text": "Noted.", "tool_calls": []})
    for t in turns:
        if t.message_ref and t.message_ref == req.message_ref:
            continue
        if t.role != "user" and (t.meta or {}).get("waiting"):
            continue  # it waited while their chat was closed: the WAITING block gives it now, as it stands
        stamp = clock.ist(t.at).strftime("%d %b %H:%M")
        if t.role == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": f"[{stamp}] {t.text}"}]})
        else:
            msgs.append({"role": "assistant", "text": t.text, "tool_calls": []})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


# Families whose backend care record was already imported (checked in the database once per process).
_IMPORTED: dict[tuple[str, str], bool] = {}
_IMPORTED_MAX = 20000


def _remember_imported(family_id: str, elder_id: str) -> None:
    if len(_IMPORTED) >= _IMPORTED_MAX:
        _IMPORTED.pop(next(iter(_IMPORTED)))
    _IMPORTED[(family_id, elder_id)] = True


# What makes a turn hard, judged only from data and the brain's own tool choices (never from the person's words): a low
# mood the mood reader logged today, a symptom or red flag in the last day, a correction of the record in the last 15
# minutes, or, during the turn, a tool call that changes a health fact, the name, logs a symptom/mood or alerts someone.
HARD_TOOLS = {"alert_caregiver", "confirm_change", "log_vital", "add_doctor_question", "undo_change", "forget"}
HARD_DOMAINS = {"medicine", "allergy", "condition", "vital_target", "diet", "naming"}


def hard_call(call) -> bool:
    args = call.args if isinstance(getattr(call, "args", None), dict) else {}
    if call.name in HARD_TOOLS or args.get("correction") or args.get("wrong"):
        return True
    if call.name in ("remember", "stop") and args.get("domain") in HARD_DOMAINS:
        return True
    return call.name == "log_event" and args.get("kind") in ("symptom", "mood")


async def hard_start(session: AsyncSession, req: "TurnRequest") -> bool:
    from sqlalchemy import select as _select

    from app.care.models import CareEvent, CareFact

    now = clock.now()
    people = {req.speaker["id"], req.elder["id"]}
    events = (await session.execute(_select(CareEvent.kind, CareEvent.payload).where(
        CareEvent.family_id == req.family_id, CareEvent.subject_id.in_(people), CareEvent.at >= now - timedelta(hours=24),
        CareEvent.kind.in_(("symptom", "red_flag", "mood"))))).all()
    for kind, payload in events:
        if kind in ("symptom", "red_flag"):
            return True
        score = (payload or {}).get("score")
        if kind == "mood" and isinstance(score, (int, float)) and score <= 2:
            return True
    recent = (await session.execute(_select(CareFact.id).where(
        CareFact.family_id == req.family_id, CareFact.recorded_at >= now - timedelta(minutes=15),
        CareFact.status.in_(("pending", "retracted", "stopped", "superseded"))).limit(1))).first()
    return recent is not None


# Hard moments get the strongest model when one is configured (MODEL_ROUTES "brain_hard": e.g. Opus or Gemini Pro);
# everything else stays on the fast one. BRAIN_ROLE lets a training run point Saheli at a tuned model ("brain_tuned").
HEALTH_TOOLS = {"web_search", "log_vital", "log_outcome", "add_doctor_question"}


def health_turn(actions: list[dict]) -> bool:
    """A turn that looked health information up or recorded a symptom, a fall, a reading or a health outcome (from the tools
    it called, never from people's words): its reply must say when to see a doctor (the no-invention check asks). Hard-turn
    A/B 14 Oct: dizziness, loose motions and a high sugar reading were answered without a search and without a doctor line."""
    return any(a.get("tool") in HEALTH_TOOLS or (a.get("tool") == "log_event" and str((a.get("args") or {}).get("kind")) in ("symptom", "fall"))
               for a in actions if a.get("ok", True))


HARD_SITUATIONS = {"emergency", "grief", "med_question", "injection", "caregiver_stressed", "low_mood"}


def role_for(situation: str) -> str:
    base = os.getenv("BRAIN_ROLE", "brain")
    if base == "brain" and situation in HARD_SITUATIONS and "brain_hard" in router.configured_roles():
        return "brain_hard"
    return base


def effort_for(situation: str, *, hard: bool) -> str:
    """Thinking depth. A hard moment (recent health, a correction) stays on BRAIN_EFFORT_HARD. Small talk and acks
    think as little as the model allows. Everything else stays on BRAIN_EFFORT."""
    if hard:
        return BRAIN_EFFORT_HARD
    if situation in LIGHT_SITUATIONS:
        return "minimal"
    return BRAIN_EFFORT


def light_history(msgs: list[dict]) -> list[dict]:
    """The recent turns only. The care record is already in the prompt; a greeting does not need the whole thread."""
    if len(msgs) <= LIGHT_HISTORY:
        return msgs
    head: list[dict] = []
    rest = msgs
    first = msgs[0]
    if first.get("role") == "user":
        blob = first.get("content") or []
        text = blob[0].get("text", "") if blob and isinstance(blob[0], dict) else ""
        if text.startswith("(Earlier in our conversation"):
            head, rest = msgs[:2], msgs[2:]
    tail = rest[-LIGHT_HISTORY:]
    while tail and tail[0].get("role") != "user":
        tail = tail[1:]
    return head + tail


def _mechanical(problems: list[str]) -> bool:
    """True when every problem is one the code can shorten away (too long, or a note as the opener). Which of two
    questions matters, an invention, the wrong language or a missing name still goes back to the model."""
    if not problems:
        return False
    return all(p.startswith("do not open with bookkeeping") or p.startswith("this message is ") for p in problems)


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
    if system and (req.message_ref or "").startswith("task:"):
        return None  # an order/ride update ("do not order again") must always reach the family
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
    return guards.canned(guards.profile([req.text]), PAUSE_NOTE)


async def recently_messaged(session: AsyncSession, req: TurnRequest) -> set[str]:
    """People Saheli sent a message to in the last few hours (so 'I have told Asha' can be true)."""
    since = clock.now() - CLAIM_WINDOW
    rows = await session.execute(
        select(Turn.thread_id).where(Turn.family_id == req.family_id, Turn.role == "assistant", Turn.at >= since, Turn.thread_id != req.speaker["id"])
    )
    return set(rows.scalars())


async def questions_still_open(session: AsyncSession, person: str) -> list[str]:
    """Today's questions, from a nudge or from here, that they have not answered in their own words."""
    from app.care.models import NudgeLedger

    start = clock.ist(clock.now()).replace(hour=0, minute=0, second=0, microsecond=0)
    asked = [(at, text) for text, at in (await session.execute(select(Turn.text, Turn.at).where(
        Turn.thread_id == person, Turn.role == "assistant", Turn.at >= start, ~Turn.family_id.startswith("shadow:")))).all()]
    asked += [(at, text) for text, at in (await session.execute(select(NudgeLedger.text, NudgeLedger.sent_at).where(
        NudgeLedger.person_id == person, NudgeLedger.status == "sent", NudgeLedger.sent_at >= start,
        ~NudgeLedger.family_id.startswith("shadow:")))).all() if at]
    wrote = [(at, text) for text, at in (await session.execute(select(Turn.text, Turn.at).where(
        Turn.thread_id == person, Turn.role == "user", Turn.at >= start, ~Turn.family_id.startswith("shadow:")))).all()]
    return guards.unanswered(asked, wrote)


async def guard_problems(session: AsyncSession, req: TurnRequest, ctx: tools.TurnCtx, final: str, *, avoid: list[str], unreachable: list[str],
                         data: TurnData | None = None, wrong_terms: list[str] | None = None) -> list[str]:
    """Everything wrong with a reply before the person reads it (policy + guards)."""
    known = "\n".join(ctx.known)
    problems = policy.reply_problems(final, known_text=known, avoid_words=avoid, user_text=req.text, unreachable=unreachable)
    from app.care import naming

    for t in naming.uses_avoided(final, list(wrong_terms or [])):
        if not naming.uses_avoided(req.text, [t]):  # unless they brought it up themselves in this message
            problems.append(f"the reply mentions '{t}', which the family said is wrong; do not say, assume or use it")
    who = await naming.for_person(session, req.family_id, req.speaker["id"])
    if req.speaker["id"] == req.elder["id"]:
        who = {**who, "avoid": []}  # the elder's never-use names are already in avoid (policy.reply_problems)
    problems += naming.address_problems(final, roster_name=req.speaker.get("name") or "", who=who)
    if guards.leaked_reasoning(final):
        problems.append("the reply contains your private notes (thoughts, the time, who is speaking); write only the message to the person")
    if guards.code_in_reply(final):
        problems.append("the reply contains code or a technical/programming answer: you never answer coding, programming or technical "
                        "development questions. Say in one short line that it is outside what you do (health, routines, schedules, "
                        "family care) and that a search engine or a general AI assistant can help; a word with an everyday meaning "
                        "(Python, Java) gets only that meaning")
    fresh = req.text + "\n" + "\n".join(a.get("args", {}).__repr__() for a in ctx.actions)
    now = clock.ist()
    problems += guards.ungrounded(
        final, known=known, fresh=fresh, meds=ctx.meds, now_minutes=now.hour * 60 + now.minute,
        allowed_times={int(v[:2]) * 60 + int(v[3:]) for v in SLOTS.values()},
    )
    problems += guards.saved_for_problems(final, ctx.actions)
    problems += guards.language_problems(final, ctx.profiles.get(req.speaker["id"]), who="they")
    problems += guards.dialect_problems(final, ctx.profiles.get(req.speaker["id"]))
    if not ctx.is_system and not ctx.buttons:
        problems += guards.question_problems(final)
        problems += guards.opener_problems(final)
        if not health_turn(ctx.actions):
            problems += guards.length_problems(final, cap=2 if ctx.speaker_is_elder else 3)
    problems += guards.food_line_problems(final, ctx.known)
    problems += guards.logged_nutrient_problems(final, ctx.known)
    if guards.asks_language_switch(req.text) and not any(a["tool"] == "language_preference" and a.get("ok") for a in ctx.actions):
        problems.append("they asked you to talk in another language or dialect: call language_preference with it first, then reply in it")
    recent = (await data.turns(req.speaker["id"]))[-20:] if data else await store.recent_turns(session, req.family_id, req.speaker["id"], limit=20)
    earlier = [t.text for t in recent
               if t.role == "assistant" and t.at >= clock.now() - timedelta(hours=12)][-8:]
    problems += guards.repeats(final, earlier)
    problems += guards.timed_promise(final, ctx.actions)
    if (req.text or "").strip() and not ctx.is_system:
        still = await questions_still_open(session, req.speaker["id"])
        if hit := guards.question_again(final, still):
            problems.append(f"this question was already asked today (\"{hit[:80]}\"); wait until they answer")
    if not (req.text or "").strip():
        # A question already asked today, by a nudge or here, waits until they write something new (founder 2026-10-15).
        from app.care.models import NudgeLedger

        start = clock.ist(clock.now()).replace(hour=0, minute=0, second=0, microsecond=0)
        me = req.speaker["id"]
        asked = [t for t in (await session.execute(select(Turn.text).where(
            Turn.thread_id == me, Turn.role == "assistant", Turn.at >= start, ~Turn.family_id.startswith("shadow:")))).scalars() if t]
        asked += [t for t in (await session.execute(select(NudgeLedger.text).where(
            NudgeLedger.person_id == me, NudgeLedger.status == "sent", NudgeLedger.sent_at >= start,
            ~NudgeLedger.family_id.startswith("shadow:")))).scalars() if t]
        if hit := guards.question_again(final, asked):
            problems.append(f"this question was already asked today (\"{hit[:80]}\"); wait until they answer")
    me = req.speaker["id"]
    others = {}
    for m in [req.elder, *req.members]:
        first = ((m.get("name") or "").split() or [""])[0].lower()
        if m.get("id") and m["id"] != me and len(first) > 2:
            others[first] = m["id"]
    messaged = {a["args"].get("to") for a in ctx.actions if a["tool"] == "send_message" and a.get("ok")}
    if any(al.get("whatsapp") for al in ctx.alerts):  # a downgraded (dashboard-only) alert reached nobody
        messaged |= {m["id"] for m in req.members if m.get("id") != req.elder["id"]}
    messaged |= await recently_messaged(session, req)
    from app.tasks import runtime as task_runtime

    ordering_ok = any(a["tool"] in ("start_task", "task_input", "order_feedback") and a.get("ok") for a in ctx.actions) or bool(
        await data.tasks() if data else await task_runtime.shown_tasks(session, req.family_id))
    prices_ok = ordering_ok or any(a["tool"] in ("past_orders", "spending") and a.get("ok") for a in ctx.actions)
    problems += guards.false_claims(final, others=others, messaged=messaged, ordering_ok=ordering_ok, prices_ok=prices_ok)
    seen, out = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# Tools that change something; once one has run, a failed turn is kept rather than retried elsewhere.
WRITE_TOOLS = {
    "remember", "stop", "forget", "note", "confirm_change", "fact_still_true", "set_boundaries", "log_dose", "log_vital", "log_event", "set_reminder", "open_loop",
    "close_loop", "send_message", "alert_caregiver", "start_task", "task_input", "cancel_task", "done_in_app", "set_stock",
    "add_doctor_question", "assign_family_task", "health_record",
    "log_food", "change_food", "nutrition_target", "pantry", "nutrition_settings", "nutrition_report",
}
PARTIAL = {
    "devanagari": "मैंने आपकी बात नोट कर ली है 🙏 एक मिनट, मैं अभी जवाब देती हूँ।",
    "mr": "तुमचं म्हणणं नोंदवून ठेवलं आहे 🙏 एक मिनिट, मी लगेच उत्तर देते.",
    "bengali": "আপনার কথা লিখে রেখেছি 🙏 এক মিনিট, আমি এখনই উত্তর দিচ্ছি।",
    "tamil": "நீங்கள் சொன்னதைக் குறித்துக்கொண்டேன் 🙏 ஒரு நிமிடம், நான் இப்போதே பதில் சொல்கிறேன்.",
    "telugu": "మీరు చెప్పింది రాసుకున్నాను 🙏 ఒక్క నిమిషం, నేను ఇప్పుడే జవాబు ఇస్తాను.",
    "kannada": "ನೀವು ಹೇಳಿದ್ದನ್ನು ಬರೆದುಕೊಂಡಿದ್ದೇನೆ 🙏 ಒಂದು ನಿಮಿಷ, ನಾನು ಈಗಲೇ ಉತ್ತರಿಸುತ್ತೇನೆ.",
    "malayalam": "നിങ്ങൾ പറഞ്ഞത് കുറിച്ചുവെച്ചു 🙏 ഒരു നിമിഷം, ഞാൻ ഇപ്പോൾ തന്നെ മറുപടി തരാം.",
    "gujarati": "તમારી વાત નોંધી લીધી છે 🙏 એક મિનિટ, હું હમણાં જ જવાબ આપું છું.",
    "gurmukhi": "ਮੈਂ ਤੁਹਾਡੀ ਗੱਲ ਨੋਟ ਕਰ ਲਈ ਹੈ 🙏 ਇੱਕ ਮਿੰਟ, ਮੈਂ ਹੁਣੇ ਜਵਾਬ ਦਿੰਦੀ ਹਾਂ।",
    "odia": "ଆପଣଙ୍କ କଥା ଲେଖି ରଖିଛି 🙏 ଗୋଟିଏ ମିନିଟ୍, ମୁଁ ଏବେ ଉତ୍ତର ଦେଉଛି।",
    "indic": "Maine aapki baat note kar li hai 🙏 Ek minute, main abhi jawab deti hoon.",
    "english": "I've noted what you told me 🙏 One moment, I'll reply shortly.",
}


SCOPE = {
    "devanagari": "यह मेरे काम से बाहर है 🙏 मैं आपकी सेहत, दिनचर्या, शेड्यूल और परिवार की देखभाल में मदद करती हूँ। इसके लिए कोई सर्च इंजन या दूसरा AI असिस्टेंट बेहतर रहेगा।",
    "indic": "Yeh mere kaam se bahar hai 🙏 Main aapki sehat, routine, schedule aur parivaar ki dekhbhaal mein madad karti hoon. Iske liye koi search engine ya doosra AI assistant behtar rahega.",
    "english": "That's outside what I do 🙏 I help with your health, routines, schedules and family care. A search engine or a general AI assistant is better for that.",
}
ACK = {"devanagari": "जी, ठीक है 🙏", "mr": "हो, ठीक आहे 🙏", "indic": "Ji, theek hai 🙏", "english": "Okay, noted 🙏",
       "bengali": "ঠিক আছে 🙏", "tamil": "சரி 🙏", "telugu": "సరే 🙏", "kannada": "ಸರಿ 🙏", "malayalam": "ശരി 🙏",
       "gujarati": "ઠીક છે 🙏", "gurmukhi": "ਠੀਕ ਹੈ 🙏", "odia": "ଠିକ ଅଛି 🙏"}


def ack_reply(p: dict | None) -> str:
    return guards.canned(p, ACK)


def partial_reply(p: dict | None) -> str:
    return guards.canned(p, PARTIAL)


async def finish_later(session: AsyncSession, req: TurnRequest) -> None:
    """The model died after something was already saved. A retry of the whole turn would save it again, so a short loop
    wakes in two minutes and answers what is left. Nothing was saved: the caller gets 503 and the backend holds the turn."""
    who = req.speaker.get("id") or ""
    if not who:
        return
    await store.open_loop(
        session, family_id=req.family_id, subject_id=who, kind="finish", title="Finish the reply you owe them",
        owner_id=who, wake_at=clock.now() + timedelta(minutes=2), alert_rule="dashboard",
        dedupe_key=f"finish:{(req.message_ref or who)[:120]}",
        detail={"next_action": "Answer their last message. What already succeeded stays saved; do not do it again.",
                "max_wakes": 3, "rewake_minutes": 2},
    )


async def run_turn(session: AsyncSession, host: ToolHost, req: TurnRequest) -> TurnResult:
    """One turn, filed under its trace (D-13): a person's message under its own message id; a turn about a task (an
    update the task runtime asked Saheli to pass on) under the task's trace, with the turn's ref as `via`; anything
    else under a new id. Afterwards one trace_turns row records it (scrubbed), for the admin console's "Why?"."""
    outer = trace.current()
    tid = outer or req.message_ref or trace.new("turn")
    via = (req.message_ref if req.message_ref != outer else None) if outer else None
    at = clock.now()
    from app.care import why

    with (trace.use(tid, task_id=trace.task_id(), via=via or trace.via(), family_id=req.family_id), trace.collect_spend() as calls,
          trace.collect_timing() as timing):
        result = await _run_turn(session, host, req)
        result.trace_id = tid
        if not result.duplicate and not req.family_id.startswith("shadow:"):
            await why.record_turn_safely(
                session, trace_id=tid, family_id=req.family_id, thread_id=req.speaker.get("id") or "",
                kind="system" if req.speaker.get("role") == "system" else "message", channel=req.channel, message_ref=req.message_ref,
                task_id=trace.task_id(), turn_id=result.turn_id, at=at, ms=result.ms, text=req.text, reply=result.reply, timing=timing,
                actions=result.actions, calls=calls)
            await session.commit()
    return result


async def _run_turn(session: AsyncSession, host: ToolHost, req: TurnRequest) -> TurnResult:
    started = time.monotonic()
    kind = "system" if req.speaker.get("role") == "system" else "message"
    await _lock_family(session, req.family_id, wait_s=LOCK_WAIT_S[kind])
    trace.note_step("lock", (time.monotonic() - started) * 1000)
    context_at = time.monotonic()
    if req.still_true is not None and not await req.still_true(session):
        # The task this update is about moved on while it waited for the family's turn (the person answered, changed or
        # cancelled it meanwhile): said now, it would contradict what they were just told.
        await session.commit()
        return TurnResult(reply="none", actions=[], alerts=[], model="none", stale=True)

    inserted = await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="user", text=req.text,
        speaker_id=req.speaker["id"], message_ref=req.message_ref,
        meta={"channel": req.channel, "images": len(req.images), "media": [i["mime"] for i in req.images] or None, **({"voice": True, "voice_confidence": req.voice_confidence} if req.voice else {}),
              **({"location": req.location} if req.location else {})},
    )
    if inserted is not None and req.speaker.get("role") != "system":
        from app.care import nudges

        try:  # a regular nudge they had not answered is answered now (its ladder resets)
            async with session.begin_nested():
                await nudges.note_reply(session, req.family_id, req.speaker["id"])
        except Exception:  # noqa: BLE001 — never stops the turn
            logger.exception("nudge reply note failed family=%s", req.family_id)
    if req.message_ref and inserted is None:
        prior = (
            await session.execute(
                select(Turn).where(Turn.family_id == req.family_id, Turn.meta["reply_to"].astext == req.message_ref)
            )
        ).scalar_one_or_none()
        if prior is not None:
            await session.commit()
            return TurnResult(reply=prior.text, actions=[], alerts=[], duplicate=True)
        # The line was saved and the process died before the reply. Continue, so the reply is not lost.
        logger.warning("unfinished turn retried family=%s ref=%s", req.family_id, req.message_ref)

    from app.care import outcomes

    parsed_button = outcomes.parse_button(req.text)
    if parsed_button and parsed_button[0] == "od":
        # A tap on an order's confirm buttons: yes / cancel act at once; change (or a yes that needs explaining) goes to the brain.
        speaker_prof = (await writing_profiles(session, req.family_id, [req.speaker])).get(req.speaker["id"])
        if not (speaker_prof or {}).get("saved") and guards.canned_key(speaker_prof) in ("english", "indic"):
            # A few Roman words ("blinkit se butter", "haan ghar") do not say the language: answer the tap in the script the
            # confirm itself was written in (sim 2026-10-11: a Hindi confirm got "Okay, placing the order now").
            recent = await store.recent_turns(session, req.family_id, req.speaker["id"], limit=6)
            shown = next((guards.script_of(t.text) for t in reversed(recent) if t.role == "assistant" and (t.text or "").strip() not in ("", "none")), None)
            if shown and shown != "latin":
                speaker_prof = {**(speaker_prof or {}), "script": shown}
        tapped, for_brain = await outcomes.handle_order_button(session, family_id=req.family_id, speaker_id=req.speaker["id"],
                                                               elder_id=req.elder["id"], code=parsed_button[1], task_id=parsed_button[2],
                                                               profile=speaker_prof)
        if tapped:
            await store.add_turn(session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=tapped,
                                 meta={"reply_to": req.message_ref, "model": "none", "button": True})
            await session.commit()
            if not getattr(host, "shadow", False):
                try:  # place it now, not at the next minute's run
                    from app.api.brain import kick_tasks

                    kick_tasks()
                except Exception:  # noqa: BLE001
                    logger.exception("kick tasks failed")
            return TurnResult(reply=tapped, actions=[{"tool": "order_button", "args": {"id": req.text}, "ok": True}], alerts=[], model="none",
                              ms=int((time.monotonic() - started) * 1000))
        req.text = for_brain or req.text
    elif parsed_button and parsed_button[0] == "lt":
        # A tap on the question of a booking set for later: the answer is data; the brain tells them what was set.
        req.text = await outcomes.handle_later_button(session, family_id=req.family_id, speaker_id=req.speaker["id"],
                                                      elder_id=req.elder["id"], code=parsed_button[1], key=parsed_button[2])
    elif parsed_button:
        # A tapped Saheli button: record it and answer without a model call.
        speaker_prof = (await writing_profiles(session, req.family_id, [req.speaker])).get(req.speaker["id"])
        tapped = await outcomes.handle_button(session, family_id=req.family_id, elder_id=req.elder["id"], speaker_id=req.speaker["id"],
                                              text=req.text, profile=speaker_prof)
        if tapped:
            await store.add_turn(session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=tapped,
                                 meta={"reply_to": req.message_ref, "model": "none", "button": True})
            await session.commit()
            return TurnResult(reply=tapped, actions=[{"tool": "button", "args": {"id": req.text}, "ok": True}], alerts=[], model="none",
                              ms=int((time.monotonic() - started) * 1000))

    canned = await flood_reply(session, req)
    if canned is not None:
        await store.add_turn(session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=canned,
                             meta={"reply_to": req.message_ref, "model": "none", "throttled": True})
        await session.commit()
        return TurnResult(reply=canned, actions=[], alerts=[], model="none", ms=int((time.monotonic() - started) * 1000))

    if req.speaker.get("role") != "system":
        await store.save_roster(session, req.family_id, req.elder, req.members)
    if (req.family_id, req.elder["id"]) not in _IMPORTED:
        if await importer.already_imported(session, req.family_id, req.elder["id"]):
            _remember_imported(req.family_id, req.elder["id"])
        else:
            try:
                async with session.begin_nested():
                    await importer.import_family(
                        session, host, family_id=req.family_id,
                        backend_family_id=req.family_id.removeprefix("shadow:"), elder_id=req.elder["id"],
                    )
                _remember_imported(req.family_id, req.elder["id"])
            except Exception:  # noqa: BLE001 — a failed import must not cost the person their reply; it is tried again next turn
                logger.exception("care import failed family=%s", req.family_id)

    data = TurnData(session, req)
    fam_block, known_record, avoid = await family_block(session, req, data)
    unreachable = [w.removeprefix("__unreachable__") for w in avoid if w.startswith("__unreachable__")]
    wrong_terms = [w.removeprefix("__rejected__") for w in avoid if w.startswith("__rejected__")]
    avoid = [w for w in avoid if not w.startswith(("__unreachable__", "__rejected__"))]
    dynamic, known_turn = await turn_context(session, req, data)
    data.history_full = True
    msgs = await history(session, req, data)
    said = f"(voice note, transcript) {req.text}" if req.voice else req.text
    if req.images:
        kinds = ", ".join(sorted({i["mime"].split("/")[0].replace("application", "document") for i in req.images}))
        said = f"(they sent a {kinds}, attached; look at it) {said}".rstrip()
    if req.media_note:
        said = f"({req.media_note}) {said}".rstrip()
    content = [{"type": "text", "text": f"[{clock.ist().strftime('%d %b %H:%M')}] {said}"}]
    content += [{"type": "image", "mime": i["mime"], "data": i["data"]} for i in req.images]
    msgs.append({"role": "user", "content": content})
    turn_start = len(msgs) - 1

    is_system = req.speaker.get("role") == "system"
    profiles = await writing_profiles(session, req.family_id, [req.elder, *[m for m in req.members if m.get("id") != req.elder["id"]]], data)
    if not is_system and not req.voice:
        # A transcript's script is not how they write. Only a message they typed can fill in a profile.
        cur = guards.profile([req.text])
        if cur and req.speaker["id"] not in profiles:
            profiles[req.speaker["id"]] = cur
        elif cur and profiles[req.speaker["id"]].get("script") is None:  # saved language, first message seen now
            profiles[req.speaker["id"]] = {**cur, "saved": profiles[req.speaker["id"]].get("saved")}
    dynamic += "\n\n" + writing_block(req, profiles)
    # Learned style from other families (only this situation's lessons and examples; under the fixed rules).
    from app.learn import playbook as learned
    from app.learn import situations

    role = "system" if is_system else ("elder" if req.speaker["id"] == req.elder["id"] else "caregiver")
    lang = situations.lang_of(profiles.get(req.speaker["id"]))
    situation = situations.tag(text=req.text, role=role, tools=[], prompt=req.text if is_system else "")
    pb, arm = await learned.for_family(session, req.family_id)
    pb_block = learned.block(pb, situation, lang)
    if pb_block:
        dynamic += "\n\n" + pb_block
    tool_texts: list[str] = []
    ctx = tools.TurnCtx(
        session=session, host=host, family_id=req.family_id, elder=req.elder, speaker=req.speaker,
        members=req.members, message_ref=req.message_ref, user_text="" if is_system else req.text, prompt=req.text if is_system else "",
        tool_texts=tool_texts, known=[fam_block, known_record, known_turn, dynamic, req.text, *[_msg_text(m) for m in msgs]],
        meds=await medicine_times(data), profiles=profiles,
        situation=situation, playbook_version=pb.version if pb else 0, arm=arm, voice_unsure=req.voice_unsure,
        voice=req.voice,
        location=req.location, images=list(req.images or []),
    )
    ctx.known.append("")  # tool results are appended as they come
    stable = [PERSONA + "\n\n" + REPLY_FORMAT, fam_block]
    brain_role = role_for(situation)
    # A photo or video stills: the vision model looks. The text brain cannot open them and used to say so.
    if req.images and "vision" in router.configured_roles():
        brain_role = "vision"
    hard = False
    if not is_system:
        try:
            hard = await hard_start(session, req)
        except Exception:  # noqa: BLE001 — the turn goes on at the usual effort
            logger.exception("hard-turn check failed family=%s", req.family_id)
    effort = BRAIN_EFFORT if is_system else effort_for(situation, hard=hard)
    if situation in LIGHT_SITUATIONS and not is_system and turn_start > LIGHT_HISTORY:
        kept = light_history(msgs[:turn_start])
        msgs = kept + msgs[turn_start:]
        turn_start = len(kept)
    guarded = False
    guard_rounds = 0  # a rewrite is checked again (trust plan C4); after two, unsupported sentences are removed by code
    # What a reply may draw on for the no-invention check: memory and this turn's context, what the PEOPLE wrote (never
    # Saheli's own earlier replies), and this turn's tool results (tool_texts, filled as tools run).
    # A scheduled turn's prompt carries the data it is about (the day's digest for the nightly brief, the order update):
    # that is a source too.
    grounding_memory = "\n".join(p for p in (fam_block, known_record, known_turn, dynamic, req.text if is_system else "") if p)
    their_msgs = [_msg_text(m) for m in msgs if m.get("role") == "user"] + ([req.text] if req.text and not is_system else [])
    degraded = ""
    tool_text_retried = False
    reply: router.LLMReply | None = None
    trace.note_step("context", (time.monotonic() - context_at) * 1000)
    for step in range(MAX_STEPS):
        last = step == MAX_STEPS - 1
        # The turn's time budget: a model call never runs past it (the person has stopped waiting, and the family stays
        # locked meanwhile). Spent: the same as no model answering (nothing done yet → 503; something done → said so).
        left = TURN_BUDGET_S[kind] - (time.monotonic() - started)
        try:
            if left < MIN_CALL_S:
                raise router.AllModelsFailed(f"turn budget of {TURN_BUDGET_S[kind]:.0f}s spent")
            cap = min(left, CALL_CAP_S)
            # The Singapore call is a second budget only when that flag is on. Off: one call, then stop.
            bound = cap + (CALL_CAP_S if router.sg_enabled() else 0)
            try:
                reply = await asyncio.wait_for(router.complete(
                    brain_role,
                    system_stable=stable,
                    system_dynamic=dynamic + ("\n\nYou have used all your steps: reply to the person now with what you have." if last else ""),
                    messages=msgs,
                    # On the last step there are no tools, so the person always gets a reply.
                    tools=None if last else tools.specs(),
                    max_tokens=LIGHT_MAX_TOKENS if effort == "minimal" else 6000,
                    effort=effort,
                    essential=not is_system,  # past the hard cap only people's messages get answers
                    timeout_s=cap,
                ), timeout=bound)
            except TimeoutError as exc:
                logger.error("turn budget spent family=%s step=%d", req.family_id, step)
                raise router.AllModelsFailed(f"turn budget of {TURN_BUDGET_S[kind]:.0f}s spent") from exc
        except router.AllModelsFailed:
            if not any(a.get("ok") and a["tool"] in WRITE_TOOLS for a in ctx.actions):
                raise  # nothing happened yet: the caller may safely try another path (503, then the holding line)
            # Something was already saved or sent: keep it (a retry elsewhere would do it twice). Say we'll finish, and do.
            logger.error("brain failed mid-turn family=%s after %s", req.family_id, [a["tool"] for a in ctx.actions])
            degraded = "none" if is_system else partial_reply(ctx.profiles.get(req.speaker["id"]))
            if not is_system:
                await finish_later(session, req)
            reply = None
            break
        msgs.append(reply.as_message())
        if not reply.tool_calls:
            text_now = (reply.text or "").strip()
            if text_now and not last and not tool_text_retried and guards.leaked_tool_call(text_now, set(tools.names())):
                # A tool call written as text did nothing and would reach the person as code: send it back once.
                tool_text_retried = True
                logger.error("tool call written as text family=%s", req.family_id)
                msgs.append({"role": "user", "content": [{"type": "text", "text": (
                    "(Your last answer was a tool call written as plain text: NOTHING was done, and the person would have seen code. "
                    "If the action is still needed, call the tool through the tool interface now. Then write only the message for the person.)"
                )}]})
                continue
            if is_system or not text_now or last:
                break
            checked_at = time.monotonic()
            # Length and a bookkeeping opener are cut here. A second brain call for that was 5–17 s on live turns.
            # Two questions, an invention, the wrong language or a missing name still go back to the model.
            problems = await guard_problems(session, req, ctx, text_now, avoid=avoid, unreachable=unreachable, data=data,
                                            wrong_terms=wrong_terms)
            if _mechanical(problems):
                clipped = guards.clip_reply(text_now, cap=2 if ctx.speaker_is_elder else 3)
                if clipped.strip():
                    reply.text = clipped
                    text_now = clipped.strip()
                    guarded = True
                    problems = []
            claims: list = []
            if not problems:
                claims = await grounding.check(text_now, memory=grounding_memory, their_messages=their_msgs,
                                               tool_results=tool_texts, actions=True, doctor_note=health_turn(ctx.actions))
                problems += grounding.problems(claims)
            trace.note_step("check", (time.monotonic() - checked_at) * 1000)
            if not problems:
                break
            guarded = True
            guard_rounds += 1
            logger.warning("reply guard family=%s problems=%s", req.family_id, problems)
            strippable = [c for c in claims if c.source not in ("doctor_note", "unwanted_apology", "followup_missing")]
            if guard_rounds >= 2 and (strippable or guards.opener_problems(text_now)):
                # A second draft that still invents, advises, or opens as a note is not sent on for another wording.
                # A missing "see a doctor" line has nothing to cut, so that one gets one more rewrite.
                reply.text = guards.without_bookkeeping(grounding.strip(text_now, strippable)) or ""
                break
            if guard_rounds >= 3:
                break
            try:
                from app.learn import corrections

                await corrections.note_guard(session, family_id=req.family_id, thread_id=req.speaker["id"], draft=text_now,
                                             problems=problems, user_text=req.text)
            except Exception:  # noqa: BLE001 — learning never blocks a reply
                logger.exception("guard note failed")
            msgs.append({"role": "user", "content": [{"type": "text", "text": (
                "(Check before sending, the person has not seen this yet: " + "; ".join(problems) + ". If you said you would do "
                "something, do it now with the tool; otherwise drop the claim. Then write only the corrected message to the person.)"
            )}]})
            continue
        if not is_system and reply.tool_calls and effort == "minimal":
            effort = BRAIN_EFFORT  # a tool ran: the reply after it is no longer small talk
        if effort != BRAIN_EFFORT_HARD and not is_system and any(hard_call(c) for c in reply.tool_calls):
            effort = BRAIN_EFFORT_HARD  # the rest of this turn (the reply after the health change) thinks harder
            logger.info("hard turn family=%s effort=%s", req.family_id, effort)
        results = []
        searches = sum(1 for a in ctx.actions if a["tool"] in ("recall", "search_records"))
        for call in reply.tool_calls:
            if call.name in ("recall", "search_records") and searches >= 4:
                # Searching memory again rarely finds what four searches did not: answer with what you have.
                results.append({"id": call.id, "name": call.name, "is_error": True,
                                "content": json.dumps({"refused": "You have searched enough. If it is not there, say plainly you do not have it and ask."})})
                continue
            ran_at = time.monotonic()
            out, is_error = await tools.run(ctx, call.name, call.args)
            trace.note_timed("tool", call.name, (time.monotonic() - ran_at) * 1000)
            tool_texts.append(f"{call.name}{' (failed)' if is_error else ''}: {out}")
            ctx.known.append(out)
            results.append({"id": call.id, "name": call.name, "content": out, "is_error": is_error})
        msgs.append({"role": "tool", "results": results})
        if (not is_system and req.speaker.get("id")
                and any(r["name"] == "log_food" and not r["is_error"] for r in results)
                and not any(a.get("ok") and a["tool"] == "gap_asked" for a in ctx.actions)):
            from app.care import gaps

            note = gaps.refresh_note(await gaps.block(session, req.family_id, req.speaker["id"], req.speaker.get("name") or "them"))
            if note:
                msgs.append({"role": "user", "content": [{"type": "text", "text": note}]})
        if any(c.name == "remember" and (c.args or {}).get("domain") == "naming" for c in reply.tool_calls) and any(
                r["name"] == "remember" and not r["is_error"] for r in results):
            msgs.append({"role": "user", "content": [{"type": "text", "text":
                "They corrected how they are addressed. Say one brief sorry in their language, then the name they asked for. "
                "Do not use an avoided word."}]})

    final = guards.strip_tool_chrome((reply.text if reply else degraded).strip())
    if final and final != "none" and (guards.leaked_reasoning(final) or guards.leaked_tool_call(final, set(tools.names()))):
        # Before any tidy: joining lines can hide "thought" on its own line. The model's notes never reach a person.
        logger.error("leaked reasoning blocked family=%s", req.family_id)
        final = "none" if is_system else ""
        guarded = True
    if final and final != "none" and reply is not None and (guard_rounds >= 2 or is_system):
        # A second rewrite (or a scheduled message, which is not rewritten) that still states something memory, the chat
        # and the tools do not support: those sentences are removed (trust plan C4).
        checked_at = time.monotonic()
        still = await grounding.check(final, memory=grounding_memory, their_messages=their_msgs, tool_results=tool_texts, actions=True,
                                      doctor_note=health_turn(ctx.actions))
        trace.note_step("check", (time.monotonic() - checked_at) * 1000)
        if still:
            logger.warning("ungrounded sentences removed family=%s claims=%s", req.family_id, [c.quote for c in still])
            final = grounding.strip(final, still) or ("none" if is_system else "")
            guarded = True
        opened = guards.without_bookkeeping(final)
        if opened != final:
            final = opened
            guarded = True
    if final and final != "none" and not is_system and guards.code_in_reply(final):
        logger.error("code in reply blocked family=%s", req.family_id)  # the rewrite kept it: the person gets the scope line
        final = guards.canned(ctx.profiles.get(req.speaker["id"]), SCOPE)
    if final and final != "none" and (guards.leaked_reasoning(final) or guards.leaked_tool_call(final, set(tools.names()))):
        # A tidy that joined lines can hide a note. Check again.
        logger.error("leaked reasoning blocked family=%s", req.family_id)
        final = "none" if is_system else ""
        guarded = True
    if guarded and not final and not is_system:
        # The rewrite came back empty. The earlier text failed the checks, so it is not sent: a short, true acknowledgement is.
        final = ack_reply(ctx.profiles.get(req.speaker["id"]))
    if not is_system and guards.food_line_problems(final, ctx.known):
        # The rewrite still dropped the saved numbers (a water question, or praise). Send the line the tool already gave.
        final = guards.saved_food_line(ctx.known)
        guarded = True
    elif not is_system and guards.logged_nutrient_problems(final, ctx.known):
        final = guards.logged_nutrient_line(ctx.known)
        guarded = True
    if final and final != "none" and not is_system:
        greeted, mid = guards.opening_state(await data.turns(req.speaker["id"]), clock.now(), day_of=clock.ist_day)
        name = (req.speaker.get("name") or "").strip()
        final = guards.dialect_tidy(guards.whatsapp_format(guards.tidy_opening(
            final, greeted_today=greeted, mid_conversation=mid, names=[name, *name.split()[:1]])), ctx.profiles.get(req.speaker["id"]))

    after_at = time.monotonic()
    bubbles = guards.whatsapp_messages(final) if final and final != "none" and not is_system else []
    stored = "\n\n".join(bubbles) if bubbles else final
    turn_id = await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=stored,
        meta={"reply_to": req.message_ref, "model": reply.model if reply else "", "actions": [a["tool"] for a in ctx.actions]},
    )
    if not is_system:
        await _learn_from_turn(session, req, ctx, final, turn_id, role, lang, trace=turn_trace(msgs, turn_start, final))
    # Folding the thread is another model call. A turn that is already slow sends the reply first; the next one folds.
    if data.history_full and (time.monotonic() - started) < 8:
        await maybe_compact(session, req.family_id, req.speaker["id"])
    await session.commit()
    trace.note_step("after", (time.monotonic() - after_at) * 1000)
    order_tap = bool(parsed_button and parsed_button[0] == "od")  # a Cancel tap on a placed order: its caregiver is told now
    if not getattr(host, "shadow", False) and (order_tap or any(a["tool"] in ("start_task", "task_input", "cancel_task") and a.get("ok")
                                                                for a in ctx.actions)):
        # Move the order now instead of at the next minute's scheduler run (live 2026-10-09: 47 s wait before the look-ups).
        try:
            from app.api.brain import kick_tasks

            kick_tasks()
        except Exception:  # noqa: BLE001 — the scheduler still runs every minute
            logger.exception("kick tasks failed")
    briefs: list[str] = []
    if not is_system and req.speaker.get("id"):
        from app.care import brief as care_brief

        briefs = await care_brief.deliver_waiting(session, req.family_id, req.speaker["id"], req.elder.get("name") or "them")
    send = [*bubbles, *briefs]
    if not send and stored and stored != "none":
        send = [stored]
    shown = send[0] if send else stored
    if ctx.buttons and shown and shown != "none" and getattr(ctx, "buttons_spec", None):
        from app.care import outcomes

        # in the script of the reply they go with (a Tamil confirm with Tamil buttons, not English ones)
        ctx.buttons = outcomes.buttons(*ctx.buttons_spec, ctx.profiles.get(req.speaker["id"]), shown)
    return TurnResult(
        reply=shown,
        messages=send,
        actions=ctx.actions,
        alerts=ctx.alerts,
        model=reply.model if reply else "",
        shadow_writes=getattr(host, "would_have", []),
        ms=int((time.monotonic() - started) * 1000),
        buttons=ctx.buttons if final and final != "none" else [],
        turn_id=turn_id,
        location_request=bool(ctx.location_request and final and final != "none" and not ctx.buttons),
    )


TRACE_RESULT_CHARS = 4000  # the judge grades against these results: at 1500 a record's medicine list was cut and read as made up (13 Oct)


def turn_trace(msgs: list[dict], start: int, final: str) -> list[dict]:
    """This turn as the model saw it, from the person's message on: tool calls, trimmed results, the final reply. Login
    codes and secrets never enter it (app.core.trace.scrub: a code passed to a task is masked everywhere it appears)."""
    known: set[str] = set()
    for m in msgs[start:]:
        for c in m.get("tool_calls") or [] if m.get("role") == "assistant" else []:
            known |= trace.secrets_in(c.args)
    out: list[dict] = []
    for m in msgs[start:]:
        if m.get("role") == "user":
            out.append({"role": "user", "text": trace.scrub(_msg_text(m)[:2000], known)})
        elif m.get("role") == "assistant":
            calls = [{"name": c.name, "args": trace.mask(c.args, known)} for c in m.get("tool_calls") or []]
            if calls:
                out.append({"role": "model", "calls": calls, **({"text": trace.scrub(m.get("text"), known)} if m.get("text") else {})})
        elif m.get("role") == "tool":
            out.append({"role": "tool", "results": [{"name": r.get("name"), "content": trace.scrub(str(r.get("content"))[:TRACE_RESULT_CHARS], known),
                                                     "error": bool(r.get("is_error"))} for r in m.get("results") or []]})
    out.append({"role": "model", "text": trace.scrub(final, known)})
    return out


async def _learn_from_turn(session: AsyncSession, req: TurnRequest, ctx: tools.TurnCtx, final: str, turn_id: int | None, role: str, lang: str,
                           trace: list | None = None) -> None:
    """Log the reply for scoring, and note anything asked that Saheli could not do. Never costs the person their reply."""
    from app.learn import gaps, scoring, situations

    try:
        used = [a["tool"] for a in ctx.actions if a.get("ok")]
        ctx.situation = situations.tag(text=req.text, role=role, tools=used)
        await scoring.record(session, family_id=req.family_id, thread_id=req.speaker["id"], turn_id=turn_id, kind="reply",
                             situation=ctx.situation, speaker_role=role, lang=lang, user_text=req.text, text=final, tools=used,
                             playbook_version=ctx.playbook_version, arm=ctx.arm, trace=trace)
        how = gaps.detect(user_text=req.text, reply=final, actions=ctx.actions)
        if how:
            names = [req.elder.get("name", "")] + [m.get("name", "") for m in req.members]
            await gaps.record(session, family_id=req.family_id, user_text=req.text, how=how, names=names)
    except Exception:  # noqa: BLE001
        logger.exception("learning log failed family=%s", req.family_id)


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
