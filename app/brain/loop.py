"""One brain turn: build context from Care Memory, think with tools, check the reply, remember it."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

from datetime import timedelta
from typing import Awaitable, Callable

from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain import confirm, grounding, guards, photos, policy, tools
from app.brain.context import (
    HISTORY_TURNS,
    LIGHT_HISTORY,
    PROMPT_TURN_CHARS as PROMPT_TURN_CHARS,
    SYSTEM_ID as SYSTEM_ID,
    TurnData,
    VOICE_SURE,
    dose_answer_line as dose_answer_line,
    family_block,
    history,
    light_history,
    medicine_times,
    onboarding_block as onboarding_block,
    questions_still_open,
    reminders_block as reminders_block,
    turn_context,
    voice_block as voice_block,
    waiting_block as waiting_block,
    writing_block,
    writing_profiles,
)
from app.brain.host import ToolHost
from app.brain import persona as persona_text
from app.brain.persona import REPLY_FORMAT
from app.care import importer, store
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
# Small talk and acks, from the situation tagger's existing labels. A greeting does not re-read the whole thread,
# think hard, or wait on the reply-path judge. A tool call still runs; the reply after it thinks at the normal depth.
LIGHT_SITUATIONS = frozenset({"thanks", "chit_chat"})
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
    # A scheduled turn about one person (a loop's subject: their gym check-in, their order's follow-up): what it saves with
    # no about is theirs, not the care recipient's (live 16 Oct: a caregiver's gym follow-up was opened on his mother).
    about: str | None = None
    # reminders due in the next hours for the speaker and the person cared for (from the backend, with the message):
    # [{"id", "person", "text", "day", "time", "repeat", "item"}]
    reminders_soon: list[dict] = field(default_factory=list)
    # a WhatsApp location pin in this message {"lat", "lng", "name", "address"} (founder 2026-10-12: a ride's pickup)
    location: dict | None = None
    onboarding: bool = False  # the speaker has not finished setting up: Saheli runs the setup chat
    # messages that could not reach the speaker while their WhatsApp chat was closed (they got a template notice):
    # [{"purpose", "text", "at"}] from the backend, given in this turn
    waiting: list[dict] = field(default_factory=list)
    # the WhatsApp message id this one quotes (a reply to their earlier photo): its kept reading comes back with it
    reply_to: str | None = None

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
    """Scheduled turns, one at a time per family, released at commit. A person's message does not take this: their
    own messages share a speaker lock, and another person in the family does not wait (app.brain.batch)."""
    if not wait_s:
        await session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": family_id})
        session.sync_session.info["family_xact_lock"] = family_id
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
    session.sync_session.info["family_xact_lock"] = family_id




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
    problems += guards.medicine_order_problems(final, ctx.known)
    if getattr(ctx, "pending_gap", "") and not any(a.get("ok") and a["tool"] == "gap_asked" for a in ctx.actions):
        problems.append("a meal was saved and water is not logged today: ask one short question about water and call "
                        "gap_asked with key water. No other question")
    if any(a.get("ok") and a["tool"] == "gap_asked" for a in ctx.actions) and "?" not in (final or "") and "？" not in (final or ""):
        problems.append("you called gap_asked: the reply asks that one question, after the save line")
    if guards.asks_language_switch(req.text) and not any(a["tool"] == "language_preference" and a.get("ok") for a in ctx.actions):
        problems.append("they asked you to talk in another language or dialect: call language_preference with it first, then reply in it")
    recent = (await data.turns(req.speaker["id"]))[-20:] if data else await store.recent_turns(session, req.family_id, req.speaker["id"], limit=20)
    earlier = [t.text for t in recent
               if t.role == "assistant" and t.at >= clock.now() - timedelta(hours=12)][-8:]
    problems += guards.repeats(final, earlier)
    week = [(clock.ist(t.at).strftime("%d %b"), t.text) for t in recent
            if t.role == "assistant" and clock.now() - timedelta(days=7) <= t.at < clock.now() - timedelta(hours=12)]
    if copied := guards.old_message_again(final, week):
        problems.append(f"this is the message you already sent on {copied}; write something new about now, from what they "
                        "said and their record, not a line from your earlier messages")
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
# A question. An acknowledgement is not the answer. The tag comes from the situation already on the turn.
ANSWER_SITUATIONS = frozenset({"med_question", "caregiver_update", "refill", "appointment", "order", "relay"})
ACK = {"devanagari": "जी, ठीक है 🙏", "mr": "हो, ठीक आहे 🙏", "indic": "Ji, theek hai 🙏", "english": "Okay, noted 🙏",
       "bengali": "ঠিক আছে 🙏", "tamil": "சரி 🙏", "telugu": "సరే 🙏", "kannada": "ಸರಿ 🙏", "malayalam": "ശരി 🙏",
       "gujarati": "ઠીક છે 🙏", "gurmukhi": "ਠੀਕ ਹੈ 🙏", "odia": "ଠିକ ଅଛି 🙏"}


def ack_reply(p: dict | None) -> str:
    return guards.canned(p, ACK)


def _only_ack(text: str, profile: dict | None) -> bool:
    return (text or "").strip() == ack_reply(profile).strip()


def partial_reply(p: dict | None) -> str:
    return guards.canned(p, PARTIAL)


def _json_result(action: dict) -> dict:
    try:
        data = json.loads(action.get("result") or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _last_draft(msgs: list) -> str:
    """The last reply already written, when a later model call does not come back."""
    for message in reversed(msgs):
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        text = str(message.get("text") or "").strip()
        if text:
            return text
    return ""


def _asked_name(ctx: tools.TurnCtx) -> str:
    """The name the speaker asked to be called, saved in this turn. Empty when this turn did not change how the speaker
    is addressed: a naming save about someone else (how a caregiver calls their mother, "Maa") is not their own name
    (live 16 Oct: a caregiver who wrote "don't speak in Hindi" was told "I'll call you Maa")."""
    me = ctx.speaker.get("id")
    for action in reversed(ctx.actions):
        if not action.get("ok") or action.get("tool") != "remember":
            continue
        args = action.get("args") or {}
        if args.get("domain") != "naming" or action.get("subject") != me:
            continue
        return str((args.get("details") or {}).get("name") or "").strip()
    return ""


def _told(ctx: tools.TurnCtx) -> str:
    for action in reversed(ctx.actions):
        if action.get("ok") and action.get("tool") == "alert_caregiver":
            told = _json_result(action).get("told")
            if told:
                return str(told)
    return ""


async def cover_red_flag(ctx: tools.TurnCtx, situation: str) -> None:
    """A red-flag reading or a red-flag moment is logged and the caregiver is told, even when the draft skipped it."""
    alerted = any(a.get("ok") and a.get("tool") == "alert_caregiver" for a in ctx.actions)
    vital_flag = ""
    for action in ctx.actions:
        if action.get("ok") and action.get("tool") == "log_vital" and _json_result(action).get("red_flag"):
            vital_flag = str(_json_result(action)["red_flag"])
    logged = any(a.get("ok") and a.get("tool") in ("log_event", "log_vital") for a in ctx.actions)
    if situation == "emergency" and not logged:
        await tools.run(ctx, "log_event", {"kind": "symptom", "summary": (ctx.user_text or "symptom")[:160], "severity": "concern"})
    if (vital_flag or situation == "emergency") and not alerted:
        await tools.run(ctx, "alert_caregiver", {
            "reason": "red_flag", "issue": "red-flag",
            "message": (vital_flag or "A red flag was reported.")[:300], "confidence": 0.9,
        })


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
    from app.core.drain import drain

    async with drain.turn(session):
        return await _run_turn_body(session, host, req)


async def _run_turn_body(session: AsyncSession, host: ToolHost, req: TurnRequest) -> TurnResult:
    outer = trace.current()
    tid = outer or req.message_ref or trace.new("turn")
    via = (req.message_ref if req.message_ref != outer else None) if outer else None
    at = clock.now()
    from app.care import why

    with (trace.use(tid, task_id=trace.task_id(), via=via or trace.via(), family_id=req.family_id), trace.collect_spend() as calls,
          trace.collect_timing() as timing):
        try:
            from app.brain import batch

            result = await batch.run(session, host, req)
        finally:
            await store.release_record_locks(session, commit=False)
        result.trace_id = tid
        if not result.duplicate and not req.family_id.startswith("shadow:"):
            await why.record_turn_safely(
                session, trace_id=tid, family_id=req.family_id, thread_id=req.speaker.get("id") or "",
                kind="system" if req.speaker.get("role") == "system" else "message", channel=req.channel, message_ref=req.message_ref,
                task_id=trace.task_id(), turn_id=result.turn_id, at=at, ms=result.ms, text=req.text, reply=result.reply, timing=timing,
                actions=result.actions, calls=calls)
            await session.commit()
    return result


def _tool_say(results: list[dict]) -> str | None:
    """The line a tool already wrote for the person, when every result this step is that kind of line."""
    said: str | None = None
    for r in results:
        if r.get("is_error"):
            return None
        try:
            data = json.loads(r.get("content") or "")
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        say = str(data.get("say") or "").strip()
        if not say:
            return None
        if said is not None:
            return None
        said = say
    return said


def _reply_is_tool_line(draft: str, tool_texts: list[str], profile: dict | None) -> bool:
    """The draft is the tool's own line (or that line in their script). Nothing in it was written by the model."""
    d = (draft or "").strip()
    if not d:
        return False
    for raw in tool_texts:
        body = raw.split(": ", 1)[-1] if ": " in raw else raw
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        say = str(data.get("say") or "").strip()
        if say and d in (say, guards.saved_line_for(profile, say).strip()):
            return True
    return False


async def _judge_claims(ctx: tools.TurnCtx, draft: str, *, memory: str, theirs: list[str], tool_texts: list[str], situation: str,
                        asked: bool = False) -> list:
    """The live judge, on this family's sample for the day and on every crisis, alert, or unsure voice note.

    A draft is checked once. The same words are not sent again. A rewrite is a new draft, so it is checked.
    A draft that is only the tool's line is not checked: the tool wrote it."""
    from app.quality import sample

    if _reply_is_tool_line(draft, tool_texts, ctx.profiles.get(ctx.speaker.get("id"))):
        return []
    used = [a["tool"] for a in ctx.actions if a.get("ok")]
    force = sample.always_live(voice_unsure=ctx.voice_unsure, alerts=bool(ctx.alerts), health=health_turn(ctx.actions),
                               tools=used, situation=situation)
    if not force and not asked and not grounding.needs_check(draft):
        return []
    # The same draft is not sent to the model twice. A rewrite is a new draft and is checked.
    text = (draft or "").strip()
    if text and text == ctx.judged_text:
        return list(ctx.judged_claims)
    if force:
        ctx.judge_band = sample.ALWAYS
    elif ctx.judge_band == sample.SKIP:
        return []
    elif ctx.judge_band not in (sample.SAMPLE, sample.ALWAYS):
        ctx.judge_band = sample.SAMPLE if await sample.ordinary_room(ctx.session, ctx.family_id) > 0 else sample.SKIP
    if ctx.judge_band == sample.SKIP:
        return []
    claims = await grounding.check(draft, memory=memory, their_messages=theirs, tool_results=tool_texts, actions=True,
                                   doctor_note=health_turn(ctx.actions), always=asked,
                                   effort="low" if situation == "chit_chat" else None)
    ctx.judged_text = text
    ctx.judged_claims = list(claims)
    return claims


async def _run_turn(session: AsyncSession, host: ToolHost, req: TurnRequest, *, take_lock: bool = True) -> TurnResult:
    started = time.monotonic()
    kind = "system" if req.speaker.get("role") == "system" else "message"
    if take_lock:
        await _lock_family(session, req.family_id, wait_s=LOCK_WAIT_S[kind])
        trace.note_step("lock", (time.monotonic() - started) * 1000)
    context_at = time.monotonic()
    if req.still_true is not None and not await req.still_true(session):
        # The task this update is about moved on while it waited for the family's turn (the person answered, changed or
        # cancelled it meanwhile): said now, it would contradict what they were just told.
        await session.commit()
        return TurnResult(reply="none", actions=[], alerts=[], model="none", stale=True)

    user_meta = {"channel": req.channel, "images": len(req.images), "media": [i["mime"] for i in req.images] or None,
                 **({"voice": True, "voice_confidence": req.voice_confidence} if req.voice else {}),
                 **({"location": req.location} if req.location else {}), **({"reply_to": req.reply_to} if req.reply_to else {})}
    inserted = await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="user", text=req.text,
        speaker_id=req.speaker["id"], message_ref=req.message_ref, meta=user_meta,
    )
    # A photo is listed (every item it shows) beside the turn's own model call, so the reply is not slower; the reading is
    # kept on this message for later turns (app.brain.photos).
    photo_task = photos.start_read(req.images, req.text) if req.images and req.speaker.get("role") != "system" else None
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
    speaker_turns = await data.turns(req.speaker["id"]) if req.speaker.get("role") != "system" else []
    photo_block = photos.block(speaker_turns)
    if photo_block:
        dynamic += "\n\n" + photo_block
        known_turn += "\n" + photo_block
    pending_before = confirm.pending_from(speaker_turns)
    waiting_yes = confirm.waiting_block(pending_before)
    if waiting_yes:
        dynamic += "\n\n" + waiting_yes
    data.history_full = True
    msgs = await history(session, req, data)
    said = f"(voice note, transcript) {req.text}" if req.voice else req.text
    if req.images:
        kinds = ", ".join(sorted({i["mime"].split("/")[0].replace("application", "document") for i in req.images}))
        said = f"(they sent a {kinds}, attached; look at it) {said}".rstrip()
    quoted_photo = photos.quoted(speaker_turns, req.reply_to) if not req.images else None
    if quoted_photo:
        shown = photos.reading_text(quoted_photo.get("read")) or "it could not be read then"
        said = f"(they are replying to the photo they sent at {quoted_photo['at']}; it showed: {shown}) {said}".rstrip()
    elif photos.quoted_unread(speaker_turns, req.reply_to) and not req.images:
        # A quoted photo whose reading was not kept: she cannot see it now, and says so (never a guess).
        when = photos.quoted_unread(speaker_turns, req.reply_to)["at"]
        said = (f"(they are replying to the photo they sent at {when}; you cannot see it now and what it showed was not kept: "
                f"say so honestly and ask them to send it again or type the names; never guess what it showed) {said}").rstrip()
    elif req.media_note:
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
        confirm_gate=not is_system and req.channel in ("whatsapp", "call") and os.getenv("SAVE_CONFIRM", "on").strip().lower() != "off",
        pending_before=pending_before, photo_turns=speaker_turns,
        voice_low_score=bool(req.voice and req.voice_confidence is not None and req.voice_confidence < VOICE_SURE),
    )
    if pending_before and ctx.confirm_gate:
        ctx.tool_groups.add("confirm")  # answer_pending, only while a read-back waits
    ctx.system_about = req.about if is_system and req.about in {m.get("id") for m in [req.elder, *req.members]} else None
    if not is_system and req.speaker.get("id"):
        earlier = await data.turns(req.speaker["id"])
        last_hers = next((t.text for t in reversed(earlier) if t.role == "assistant" and (t.text or "").strip() not in ("", "none")), "")
        ctx.prior_assistant = bool(last_hers)
        ctx.last_assistant = last_hers
        ctx.held_topic = guards.held_topic(last_hers, req.text)
    ctx.known.append("")  # tool results are appended as they come

    def _groups() -> set[str]:
        # Core every call. The situation tag, an open task, onboarding, a photo, or need_tools adds a group.
        return tools.groups_for(
            situation=situation, open_tasks=bool(data.open_tasks), system=is_system,
            onboarding=bool(req.onboarding), images=bool(req.images), extra=ctx.tool_groups,
        )
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
    photo_read: dict | None = None
    photo_known = False

    async def _photo_into_sources(wait_s: float) -> None:
        # What the photo showed is a source for the checks: names read from it are not inventions (live 17 Oct: three of four
        # strips were cut from the reply as "not in memory or this chat").
        nonlocal photo_read, photo_known
        if photo_known or photo_task is None:
            return
        photo_read = await photos.settle(photo_task, wait_s)
        text = photos.reading_text(photo_read)
        if text:
            photo_known = True
            line = f"(the photo they sent now shows: {text})"
            their_msgs.append(line)
            ctx.known.append(line)
    degraded = ""
    degraded_draft = False  # the degraded reply is her own draft of this turn (a read-back in it was written by her)
    tool_text_retried = False
    reply: router.LLMReply | None = None
    trace.note_step("context", (time.monotonic() - context_at) * 1000)
    if not is_system and not session.sync_session.info.get("family_xact_lock"):
        # The roster row and the saved line stay locked until commit. Commit before the model call so another
        # person in the family is not stuck behind this one for the whole reply.
        await session.commit()
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
                groups = _groups()
                reply = await asyncio.wait_for(router.complete(
                    brain_role,
                    # Rebuilt each step: need_tools adds a group, and that group's persona block, for the next call.
                    system_stable=[persona_text.for_groups(groups) + "\n\n" + REPLY_FORMAT, fam_block],
                    system_dynamic=dynamic + ("\n\nYou have used all your steps: reply to the person now with what you have." if last else ""),
                    messages=msgs,
                    # On the last step there are no tools, so the person always gets a reply.
                    tools=None if last else tools.specs_for(
                        open_tasks=bool(data.open_tasks), system=is_system, situation=situation,
                        onboarding=bool(req.onboarding), images=bool(req.images), extra=ctx.tool_groups,
                    ),
                    max_tokens=LIGHT_MAX_TOKENS if effort == "minimal" else 6000,
                    effort=effort,
                    essential=not is_system,  # past the hard cap only people's messages get answers
                    timeout_s=cap,
                ), timeout=bound)
            except TimeoutError as exc:
                logger.error("turn budget spent family=%s step=%d", req.family_id, step)
                raise router.AllModelsFailed(f"turn budget of {TURN_BUDGET_S[kind]:.0f}s spent") from exc
        except router.AllModelsFailed:
            # Only a reply written in THIS turn: her previous message in the history is not this turn's answer (a "Save it?"
            # read-back was sent again after the yes had saved it, when the next call failed).
            draft = "" if is_system else _last_draft(msgs[turn_start:])
            if draft:
                # A later call ran out of time. The reply already written still goes out (a night of empty replies
                # was five turns that had a draft and then spent the budget).
                logger.error("turn budget spent family=%s; sending the draft already written", req.family_id)
                degraded = draft
                degraded_draft = True
                reply = None
                break
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
            await _photo_into_sources(min(6.0, max(0.0, TURN_BUDGET_S[kind] - (time.monotonic() - started) - MIN_CALL_S)))
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
                asked = "?" in (req.text or "") or "？" in (req.text or "")
                claims = await _judge_claims(ctx, text_now, memory=grounding_memory, theirs=their_msgs, tool_texts=tool_texts,
                                             situation=situation, asked=asked)
                problems += grounding.problems(claims)
            trace.note_step("check", (time.monotonic() - checked_at) * 1000)
            if not problems:
                break
            guarded = True
            guard_rounds += 1
            logger.warning("reply guard family=%s problems=%s", req.family_id, problems)
            profile = ctx.profiles.get(req.speaker["id"])
            if any(c.source in ("unwanted_apology", "tease_as_question") for c in claims):
                reply.text = guards.light_reply(profile)
                ctx.force_light = True
                break
            if any(c.source == "framing_missing" for c in claims):
                reply.text = guards.with_framing(profile, text_now)
                ctx.needs_framing = True
                break
            if ctx.held_topic and not guards.answer_used(ctx.actions) and any(c.source == "followup_missing" for c in claims):
                reply.text = guards.thread_reply(profile, ctx.held_topic)
                break
            strippable = [c for c in claims if c.source not in (
                "doctor_note", "unwanted_apology", "followup_missing", "framing_missing", "question_unanswered")]
            if guard_rounds >= 2 and (strippable or guards.opener_problems(text_now)) and not (
                    any(c.source == "followup_missing" for c in claims) and "?" not in text_now and "？" not in text_now):
                # A second draft that still invents, advises, or opens as a note is not sent on for another wording.
                # A missing question about something they shared gets one more rewrite: cutting the draft does not ask it.
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
        if photo_task is not None and not photo_task.done() and any(
                c.name == "log_food" and ((c.args or {}).get("from_photo") or (c.args or {}).get("label_photo")) for c in reply.tool_calls):
            photo_task.cancel()  # a plate or a label: the food flow reads it; no separate photo read is kept
            photo_task = None
        for call in reply.tool_calls:
            if call.name in ("recall", "search_records") and searches >= 4:
                # Searching memory again rarely finds what four searches did not: answer with what you have.
                results.append({"id": call.id, "name": call.name, "is_error": True,
                                "content": json.dumps({"refused": "You have searched enough. If it is not there, say plainly you do not have it and ask."})})
                continue
            ran_at = time.monotonic()
            if guards.blocks_held_thread(call.name, call.args or {}, ctx.held_topic):
                out = json.dumps({"refused": "They answered your last question. Ask one specific question about that. "
                                              "Do not start setup or ask what to call them."})
                is_error = True
                ctx.actions.append({"tool": call.name, "args": call.args, "ok": False, "refused": "held thread"})
            else:
                out, is_error = await tools.run(ctx, call.name, call.args)
                if not is_system and store.record_locks_held(session):
                    # The write is committed here, so the other person's model call is not waiting on it.
                    await store.release_record_locks(session, commit=True)
            trace.note_timed("tool", call.name, (time.monotonic() - ran_at) * 1000)
            tool_texts.append(f"{call.name}{' (failed)' if is_error else ''}: {out}")
            ctx.known.append(out)
            results.append({"id": call.id, "name": call.name, "content": out, "is_error": is_error})
        msgs.append({"role": "tool", "results": results})
        needs_model = False
        if (not is_system and req.speaker.get("id")
                and any(r["name"] == "log_food" and not r["is_error"] for r in results)
                and not any(a.get("ok") and a["tool"] == "gap_asked" for a in ctx.actions)):
            from app.care import gaps

            note = gaps.refresh_note(await gaps.block(session, req.family_id, req.speaker["id"], req.speaker.get("name") or "them"))
            if note:
                ctx.pending_gap = "water"
                needs_model = True
                msgs.append({"role": "user", "content": [{"type": "text", "text": note}]})
        if (not is_system and any(c.name == "open_loop" and (c.args or {}).get("kind") == "followup" and (c.args or {}).get("at")
                                  for c in reply.tool_calls)
                and not any(a.get("ok") and a["tool"] == "web_search" for a in ctx.actions)):
            needs_model = True
            msgs.append({"role": "user", "content": [{"type": "text", "text":
                "They named what it is and when it ends. Call web_search for it in this turn, then one question about that thing. "
                "The follow-up is already set."}]})
        if any(c.name == "remember" and (c.args or {}).get("domain") == "naming" for c in reply.tool_calls) and any(
                r["name"] == "remember" and not r["is_error"] for r in results):
            needs_model = True
            msgs.append({"role": "user", "content": [{"type": "text", "text":
                "They corrected how they are addressed. Say one brief sorry in their language, then the name they asked for. "
                "Do not use an avoided word."}]})
        if not needs_model and not is_system:
            # The tool already wrote the line. Another brain call would only repeat it, a few seconds later.
            say = _tool_say(results)
            if say:
                reply.text = guards.saved_line_for(ctx.profiles.get(req.speaker.get("id")), say)
                break

    if not is_system:
        await cover_red_flag(ctx, situation)
    # A held save may be confirmed later only if its read-back really went out: any line below that replaces her reply
    # (the fixed reading line, the name line, the emergency line, a fallback) drops what this turn held.
    replaced = reply is None and not degraded_draft
    final = guards.strip_tool_chrome((reply.text if reply else degraded).strip())
    if final and final != "none" and (guards.leaked_reasoning(final) or guards.leaked_tool_call(final, set(tools.names()))):
        # Before any tidy: joining lines can hide "thought" on its own line. The model's notes never reach a person.
        logger.error("leaked reasoning blocked family=%s", req.family_id)
        final = "none" if is_system else ""
        replaced = True
        guarded = True
    if final and final != "none" and reply is not None and (guard_rounds >= 2 or is_system):
        # A second rewrite (or a scheduled message, which is not rewritten) that still states something memory, the chat
        # and the tools do not support: those sentences are removed (trust plan C4).
        checked_at = time.monotonic()
        still = await _judge_claims(ctx, final, memory=grounding_memory, theirs=their_msgs, tool_texts=tool_texts, situation=situation)
        trace.note_step("check", (time.monotonic() - checked_at) * 1000)
        if still:
            logger.warning("ungrounded sentences removed family=%s claims=%s", req.family_id, [c.quote for c in still])
            replaced = replaced or ("?" in final and "?" not in grounding.strip(final, still))
            final = grounding.strip(final, still) or ("none" if is_system else "")
            guarded = True
        opened = guards.without_bookkeeping(final)
        if opened != final:
            final = opened
            guarded = True
    if final and final != "none" and not is_system and guards.code_in_reply(final):
        logger.error("code in reply blocked family=%s", req.family_id)  # the rewrite kept it: the person gets the scope line
        final = guards.canned(ctx.profiles.get(req.speaker["id"]), SCOPE)
        replaced = True
    if final and final != "none" and (guards.leaked_reasoning(final) or guards.leaked_tool_call(final, set(tools.names()))):
        # A tidy that joined lines can hide a note. Check again.
        logger.error("leaked reasoning blocked family=%s", req.family_id)
        final = "none" if is_system else ""
        replaced = True
        guarded = True
    profile = ctx.profiles.get(req.speaker["id"])
    saved = any(a.get("ok") and a["tool"] in WRITE_TOOLS for a in ctx.actions)
    if not is_system and situation in ANSWER_SITUATIONS and not saved and (not final or final == "none" or _only_ack(final, profile)):
        # They asked something and nothing was saved. "Okay, noted" is not the answer.
        try:
            again = await router.complete(
                brain_role, system_stable=[persona_text.for_groups(_groups()) + "\n\n" + REPLY_FORMAT, fam_block],
                system_dynamic=dynamic + "\n\nThey asked a question. The last draft was not an answer. Answer it from the "
                "care record and this turn. If the record does not have it, say you do not have it. Do not reply with only an acknowledgement.",
                messages=msgs, tools=None, max_tokens=600, effort=effort, essential=True, timeout_s=CALL_CAP_S,
            )
            final = guards.strip_tool_chrome((again.text or "").strip())
            replaced = True
        except router.AllModelsFailed:
            final = ""
        if final and final != "none":
            still = await _judge_claims(ctx, final, memory=grounding_memory, theirs=their_msgs, tool_texts=tool_texts, situation=situation)
            if still:
                final = grounding.strip(final, still) or ""
        if _only_ack(final, profile):
            final = ""
    elif guarded and not final and not is_system:
        # The rewrite came back empty. The earlier text failed the checks, so it is not sent: a short, true acknowledgement is.
        final = ack_reply(profile)
        replaced = True
    if not is_system and guards.food_line_problems(final, ctx.known):
        # The rewrite still dropped the saved numbers. Send the tool's line, in their script, and keep a question
        # already in the reply (the water ask). Never replace a Bengali reply with the English "Saved:" line.
        say = guards.saved_line_for(ctx.profiles.get(req.speaker["id"]), guards.saved_food_line(ctx.known))
        asked = [s for s in guards.sentences(final) if "?" in s or "？" in s]
        final = (say + " " + " ".join(asked)).strip() if say else final
        replaced = True
        guarded = True
    elif not is_system and guards.language_problems(final, ctx.profiles.get(req.speaker["id"])):
        say = guards.saved_line_for(ctx.profiles.get(req.speaker["id"]), guards.saved_food_line(ctx.known))
        if say and say != final:
            final = say
            replaced = True
            guarded = True
    elif not is_system and guards.logged_nutrient_problems(final, ctx.known):
        final = guards.logged_nutrient_line(ctx.known)
        replaced = True
        guarded = True
    if final and final != "none" and not is_system:
        greeted, mid = guards.opening_state(await data.turns(req.speaker["id"]), clock.now(), day_of=clock.ist_day)
        name = (req.speaker.get("name") or "").strip()
        final = guards.dialect_tidy(guards.whatsapp_format(guards.tidy_opening(
            final, greeted_today=greeted, mid_conversation=mid, names=[name, *name.split()[:1]])), ctx.profiles.get(req.speaker["id"]))
        naming = any(a.get("ok") and a["tool"] == "remember" and (a.get("args") or {}).get("domain") == "naming" for a in ctx.actions)
        from app.learn import situations
        ctx.situation = situations.tag(text=req.text, role=role, tools=[a["tool"] for a in ctx.actions if a.get("ok")])
        saved_meds: list[tuple[str, str, list[str]]] = []
        seen_meds: set[str] = set()
        for pid in (req.speaker["id"], req.elder["id"]):
            for fact in await data.facts(pid):
                if fact.domain != "medicine" or fact.status != "active":
                    continue
                name = str((fact.value or {}).get("name") or "").strip()
                if not name or name.lower() in seen_meds:
                    continue
                seen_meds.add(name.lower())
                value = fact.value or {}
                saved_meds.append((name, str(value.get("dose") or "").strip(), [str(t) for t in (value.get("times") or [])]))
        settled = guards.settle_reply(
            final, ctx.profiles.get(req.speaker["id"]), situation=ctx.situation, needs_framing=ctx.needs_framing,
            force_light=ctx.force_light, held_topic_text=ctx.held_topic, actions=ctx.actions,
            prior_assistant=ctx.prior_assistant, naming=naming, user_text=req.text,
            medicines=saved_meds, last_assistant=ctx.last_assistant)
        if settled != final:
            replaced = replaced or ("?" in final and "?" not in settled)
            final = settled
            guarded = True
        if req.onboarding:
            hello = guards.first_hello(ctx.profiles.get(req.speaker["id"]), req.text)
            if hello:
                final = hello
                replaced = True
                guarded = True
        profile = ctx.profiles.get(req.speaker["id"])
        told = _told(ctx)
        vital = next((a for a in ctx.actions if a.get("ok") and a.get("tool") == "log_vital"), None)
        vital_data = _json_result(vital) if vital else {}
        asked_name = _asked_name(ctx)
        if asked_name and (asked_name not in (final or "") or guards.leaked_reasoning(final or "")
                           or guards.echoes_prompt(final or "", persona_text.for_groups(_groups()) + "\n" + REPLY_FORMAT)
                           or guards.language_problems(final or "", profile)):
            # Their own new name, and a draft that leaves it out, copies the prompt ("Roman letters", "35 words") or is in
            # the wrong language: the confirmation line instead. A good draft that also answers the rest of the message stays.
            final = guards.call_you(profile, asked_name)
            replaced = True
            guarded = True
        if vital:
            # A reading is noted. The number is not called fine, high or low: that judgement is not hers.
            final = guards.noted_reading(profile, str(vital_data.get("logged") or "the reading"), told if vital_data.get("red_flag") else "")
            replaced = True
            guarded = True
        elif ctx.situation == "emergency" and not any(a.get("reason") == "crisis" for a in ctx.alerts):
            # A non-crisis alert: her own answer stays when it is a short, plain reply. Only an empty, framed or long
            # draft becomes the calm line (16 Oct: the line replaced good answers, and in Hindi script for someone writing
            # Roman Hinglish; the language guard above already checked the draft's script).
            if not (final or "").strip() or guards.has_framing(final) or len(guards.sentences(final)) > 2:
                final = guards.calm_noted(profile, told)
                replaced = True
                guarded = True
        elif ctx.situation == "emergency":
            if not (final or "").strip() or guards.has_framing(final) or len(guards.sentences(final)) > 2:
                final = guards.calm_noted(profile, told)
                replaced = True
            elif told and told not in (final or ""):
                final = f"{final.rstrip()} {told}".strip()
            guarded = True
        sched = next((a for a in ctx.actions if a.get("ok") and a.get("tool") == "today_schedule"), None)
        due = [m for m in ((sched or {}).get("medicines") or _json_result(sched).get("medicines") or [])
               if isinstance(m, dict) and m.get("name")] if sched else []
        # The schedule line replaces her reply only when they asked about a medicine. A "?" on a dinner
        # message is not that question, even if the model also looked the day up (live 16 Oct 22:54).
        asked_schedule = situations.tag(text=req.text, role=role, tools=[]) == "med_question"
        if asked_schedule and due and any(str(m["name"]).lower() not in (final or "").lower() for m in due):
            full = guards.due_today(profile, due)
            if full:
                final = full
                replaced = True
                guarded = True
        dose = next((a for a in ctx.actions if a.get("ok") and a.get("tool") == "log_dose"), None)
        dose_data = _json_result(dose) if dose else {}
        dose_times = [str(t) for t in (dose_data.get("scheduled") or []) if t]
        if dose_times and not any(t in (final or "") for t in dose_times):
            # "On time" is only true when the reply names the time the record holds. A language with no fixed line keeps her reply.
            logged = guards.dose_logged(profile, str(dose_data.get("medicine") or ""), dose_times)
            if logged:
                final = logged
                replaced = True
                guarded = True
        asked = "?" in (req.text or "") or "？" in (req.text or "")
        sources = "\n".join([req.text or "", *tool_texts, *(str(x) for x in ctx.known)])
        sent = any(a.get("ok") and a.get("tool") == "send_message" for a in ctx.actions)
        if asked and (guards.invented_clock(final or "", sources) or guards.quiet_unasked(final or "", situation=ctx.situation or "", sent=sent)):
            # A clock they did not give, or a question about when to message with no time and no question back.
            final = guards.ask_time(profile)
            replaced = True
            guarded = True
        report = next((t for t in tool_texts if t.startswith("weekly_report:")), "")
        if report:
            try:
                summary = str((json.loads(report.split(":", 1)[1]) or {}).get("summary") or "")
            except json.JSONDecodeError:
                summary = ""
            facts = guards.week_facts(summary)
            bodies = [f.split(":", 1)[-1].strip().rstrip(".") for f in facts]
            bodies = [b for b in bodies if b]
            if bodies and any(b[:24] not in (final or "") for b in bodies):
                final = ". ".join(bodies[:3]) + "."
                replaced = True
                guarded = True
        same = next((a for a in ctx.actions if a.get("ok") and a.get("tool") == "remember" and _json_result(a).get("result") == "unchanged"), None)
        if same:
            note = str(_json_result(same).get("note") or "")
            what = note.split("Already on the record as ", 1)[-1].split(".", 1)[0].strip()
            final = guards.already_on_record(profile, what)
            replaced = True
            guarded = True
        reminder = next((a for a in ctx.actions if a.get("ok") and a.get("tool") == "reminder_log"), None)
        if reminder and situation == "med_question":
            attempts = _json_result(reminder).get("reminder_attempts") or []
            if not any(isinstance(row, dict) and row.get("delivered") for row in attempts):
                final = guards.missed_reminder_line(profile)
                replaced = True
                guarded = True

    if not is_system and (not (final or "").strip() or (final or "").strip().lower() == "none"):
        final = guards.missed_reminder_line(ctx.profiles.get(req.speaker["id"])) if situation == "med_question" else ack_reply(ctx.profiles.get(req.speaker["id"]))
        replaced = True

    after_at = time.monotonic()
    bubbles = guards.whatsapp_messages(final) if final and final != "none" and not is_system else []
    stored = "\n\n".join(bubbles) if bubbles else final
    held = [p for p in ctx.pending_saves if p["sig"] not in ctx.confirmed_saves]
    if held and replaced:
        logger.info("held saves dropped: their read-back was not sent family=%s tools=%s", req.family_id, [p["tool"] for p in held])
        held = []
    turn_id = await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=stored,
        meta={"reply_to": req.message_ref, "model": reply.model if reply else "", "actions": [a["tool"] for a in ctx.actions],
              **({"pending_saves": held[:10]} if held else {}),
              **({"declined_saves": ctx.declined_saves[:10]} if ctx.declined_saves else {})},
    )
    if photo_task is not None:
        await _keep_photo(session, photo_task, photo_read, inserted, user_meta, req)
    if not is_system:
        await _learn_from_turn(session, req, ctx, final, turn_id, role, lang, trace=turn_trace(msgs, turn_start, final), guarded=guarded)
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


async def _keep_photo(session: AsyncSession, task, read: dict | None, turn_id: int | None, meta: dict, req: TurnRequest) -> None:
    """Keeps what the photo showed on their message. A read still running after the reply finishes on its own session."""
    mimes = [i["mime"] for i in req.images]
    if read is None and task.done():
        read = await photos.settle(task, 0)
    if read is not None or task.done():
        await photos.keep(session, turn_id, meta, photos.photo_meta(read, ref=req.message_ref, mimes=mimes, caption=req.text))
        return

    async def later() -> None:
        late = await photos.settle(task, photos.READ_TIMEOUT_S)
        try:
            from app.db.session import SessionLocal

            async with SessionLocal() as s2:
                await photos.keep(s2, turn_id, meta, photos.photo_meta(late, ref=req.message_ref, mimes=mimes, caption=req.text))
                await s2.commit()
        except Exception:  # noqa: BLE001 — the reply is already out; the photo is only not listed for later turns
            logger.exception("keeping a photo reading failed family=%s", req.family_id)

    _BACKGROUND.add(t := asyncio.create_task(later()))
    t.add_done_callback(_BACKGROUND.discard)


_BACKGROUND: set = set()

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
                           trace: list | None = None, guarded: bool = False) -> None:
    """Log the reply for scoring, and note anything asked that Saheli could not do. Never costs the person their reply."""
    from app.learn import gaps, scoring, situations

    try:
        used = [a["tool"] for a in ctx.actions if a.get("ok")]
        ctx.situation = situations.tag(text=req.text, role=role, tools=used)
        band = ctx.judge_band or None
        if guarded or ctx.voice_unsure or ctx.alerts or ctx.situation == "emergency" or "alert_caregiver" in used:
            band = "always"
        await scoring.record(session, family_id=req.family_id, thread_id=req.speaker["id"], turn_id=turn_id, kind="reply",
                             situation=ctx.situation, speaker_role=role, lang=lang, user_text=req.text, text=final, tools=used,
                             playbook_version=ctx.playbook_version, arm=ctx.arm, trace=trace, judge_band=band)
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
