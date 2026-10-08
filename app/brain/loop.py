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
    modality: str = "text"  # text | voice (text is then the transcript of a voice note)
    voice_confidence: float | None = None  # 0..1 when the speech engine reported one
    voice_language: str | None = None

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
    buttons: list[dict] = field(default_factory=list)  # one-tap WhatsApp answers sent with the reply


async def _lock_family(session: AsyncSession, family_id: str) -> None:
    """One turn at a time per family, across instances; released at commit."""
    await session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": family_id})


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

            self._tasks = await task_runtime.live_tasks(self.session, self.req.family_id)
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
        prof = guards.profile([t.text for t in turns]) or {}
        facts = await data.facts(p["id"]) if data else await store.facts(session, family_id, p["id"])
        saved = next((f for f in facts if f.domain == "language" and f.status == "active"), None)
        if saved:
            prof = {"script": prof.get("script"), "roman": prof.get("roman"), "saved": language.normalise(saved.value)}
        if prof:
            out[p["id"]] = prof
    return out


async def family_block(session: AsyncSession, req: TurnRequest, data: TurnData | None = None) -> tuple[str, str, list[str]]:
    """The cached block: household, care record, notes. Also returns text for the reply guard and avoid-words."""
    rows = await data.facts(req.elder["id"]) if data else await store.facts(session, req.family_id, req.elder["id"])
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


async def turn_context(session: AsyncSession, req: TurnRequest, data: TurnData | None = None) -> tuple[str, str]:
    data = data or TurnData(session, req)
    now = clock.ist()
    day_events = await store.events(session, req.family_id, req.elder["id"], day=clock.ist_day())
    loops = await store.live_loops(session, req.family_id, [req.elder["id"], req.speaker["id"]])
    from app.care import memory_index

    hits = await memory_index.search(session, req.family_id, [req.elder["id"], "family"], req.text, limit=8) if req.text.strip() else []
    from app.tasks import runtime as task_runtime

    tasks = await data.tasks()
    parts = [
        f"NOW: {now.strftime('%A %d %B %Y, %H:%M')} IST",
        f"SPEAKING: {req.speaker.get('name')} ({'the care recipient' if req.speaker['id'] == req.elder['id'] else req.speaker.get('role', 'family')}), via {req.channel}",
        voice_block(req),
        digest.ledger(day_events),
        digest.loops(loops),
        ("ACTIVE TASKS (orders and rides running in the background):\n" + "\n".join(f"  - {task_runtime.describe(t)}" for t in tasks))
        if tasks else "ACTIVE TASKS: none.",
    ]
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
        parts.append("POSSIBLY RELEVANT MEMORY:\n" + "\n".join(f"  [{clock.ist(h.when).strftime('%d %b')}] {h.text}" for h in hits))
    if req.speaker["id"] != req.elder["id"] and req.speaker.get("role") != "system":
        own = await data.facts(req.speaker["id"])
        if own:
            own_today = await store.events(session, req.family_id, req.speaker["id"], day=clock.ist_day())
            parts.append(
                f"YOUR OWN CARE (the speaker's self care; use about={req.speaker['id']} for anything about them):\n"
                + digest.care_record(req.speaker.get("name") or "the speaker", own)
                + ("\n" + digest.ledger(own_today) if own_today else "")
            )
    if req.speaker["id"] != req.elder["id"]:
        elder_turns = (await data.turns(req.elder["id"]))[-12:]
        if elder_turns:
            parts.append(
                f"RECENT WITH {req.elder.get('name', 'the elder').upper()}:\n"
                + "\n".join(f"  {clock.ist(t.at).strftime('%d %b %H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text[:300]}" for t in elder_turns)
            )
    known = "\n".join(h.text for h in hits) + "\n" + "\n".join(e.summary for e in day_events)
    return "\n\n".join(parts), known


VOICE_SURE = 0.6


def voice_block(req: TurnRequest) -> str:
    """How to answer a voice note: it is a transcript (may be wrong), and the reply is spoken back as a voice note."""
    if not req.voice:
        return ""
    lines = [
        "THIS MESSAGE WAS A VOICE NOTE. The text is a machine transcript and can have wrong words, especially names, "
        "medicines and numbers. Your reply is sent as a voice note (read aloud) and as text, so write it to be heard: "
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


# Hard moments get the strongest model when one is configured (MODEL_ROUTES "brain_hard": e.g. Opus or Gemini Pro);
# everything else stays on the fast one. BRAIN_ROLE lets a training run point Saheli at a tuned model ("brain_tuned").
HARD_SITUATIONS = {"emergency", "grief", "med_question", "injection", "caregiver_stressed", "low_mood"}


def role_for(situation: str) -> str:
    base = os.getenv("BRAIN_ROLE", "brain")
    if base == "brain" and situation in HARD_SITUATIONS and "brain_hard" in router.configured_roles():
        return "brain_hard"
    return base


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
                         data: TurnData | None = None) -> list[str]:
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
    recent = (await data.turns(req.speaker["id"]))[-20:] if data else await store.recent_turns(session, req.family_id, req.speaker["id"], limit=20)
    earlier = [t.text for t in recent
               if t.role == "assistant" and t.at >= clock.now() - timedelta(hours=12)][-8:]
    problems += guards.repeats(final, earlier)
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

    ordering_ok = any(a["tool"] in ("start_task", "task_input") and a.get("ok") for a in ctx.actions) or bool(
        await data.tasks() if data else await task_runtime.live_tasks(session, req.family_id))
    problems += guards.false_claims(final, others=others, messaged=messaged, ordering_ok=ordering_ok)
    seen, out = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# Tools that change something; once one has run, a failed turn is kept rather than retried elsewhere.
WRITE_TOOLS = {
    "remember", "stop", "note", "confirm_change", "log_dose", "log_vital", "log_event", "set_reminder", "open_loop",
    "close_loop", "send_message", "alert_caregiver", "start_task", "task_input", "cancel_task", "set_stock",
    "add_doctor_question", "assign_family_task", "health_record",
}
PARTIAL = {
    "devanagari": "मैंने आपकी बात नोट कर ली है 🙏 अभी पूरा जवाब देने में दिक्कत आ रही है, कुछ मिनट बाद फिर लिखिए।",
    "mr": "तुमचं म्हणणं नोंदवून ठेवलं आहे 🙏 आत्ता पूर्ण उत्तर द्यायला अडचण येत आहे, काही मिनिटांनी पुन्हा लिहा.",
    "bengali": "আপনার কথা লিখে রেখেছি 🙏 এখন পুরো উত্তর দিতে একটু অসুবিধা হচ্ছে, কয়েক মিনিট পরে আবার লিখুন।",
    "tamil": "நீங்கள் சொன்னதைக் குறித்துக்கொண்டேன் 🙏 இப்போது முழு பதில் தர சிரமமாக உள்ளது, சில நிமிடங்கள் கழித்து மீண்டும் எழுதுங்கள்.",
    "telugu": "మీరు చెప్పింది రాసుకున్నాను 🙏 ఇప్పుడు పూర్తి సమాధానం ఇవ్వడంలో ఇబ్బంది ఉంది, కొన్ని నిమిషాల తర్వాత మళ్ళీ రాయండి.",
    "kannada": "ನೀವು ಹೇಳಿದ್ದನ್ನು ಬರೆದುಕೊಂಡಿದ್ದೇನೆ 🙏 ಈಗ ಪೂರ್ತಿ ಉತ್ತರ ಕೊಡಲು ತೊಂದರೆಯಾಗುತ್ತಿದೆ, ಕೆಲವು ನಿಮಿಷಗಳ ನಂತರ ಮತ್ತೆ ಬರೆಯಿರಿ.",
    "malayalam": "നിങ്ങൾ പറഞ്ഞത് കുറിച്ചുവെച്ചു 🙏 ഇപ്പോൾ മുഴുവൻ മറുപടി നൽകാൻ ബുദ്ധിമുട്ടുണ്ട്, കുറച്ച് മിനിറ്റ് കഴിഞ്ഞ് വീണ്ടും എഴുതൂ.",
    "gujarati": "તમારી વાત નોંધી લીધી છે 🙏 હમણાં પૂરો જવાબ આપવામાં તકલીફ થાય છે, થોડી મિનિટ પછી ફરી લખજો.",
    "gurmukhi": "ਮੈਂ ਤੁਹਾਡੀ ਗੱਲ ਨੋਟ ਕਰ ਲਈ ਹੈ 🙏 ਹੁਣੇ ਪੂਰਾ ਜਵਾਬ ਦੇਣ ਵਿੱਚ ਦਿੱਕਤ ਆ ਰਹੀ ਹੈ, ਕੁਝ ਮਿੰਟ ਬਾਅਦ ਫਿਰ ਲਿਖੋ।",
    "odia": "ଆପଣଙ୍କ କଥା ଲେଖି ରଖିଛି 🙏 ଏବେ ପୂରା ଉତ୍ତର ଦେବାରେ ଅସୁବିଧା ହେଉଛି, କିଛି ମିନିଟ ପରେ ପୁଣି ଲେଖନ୍ତୁ।",
    "indic": "Maine aapki baat note kar li hai 🙏 Abhi poora jawab dene mein dikkat aa rahi hai, kuch minute baad phir likhiye.",
    "english": "I've noted what you told me 🙏 I'm having trouble answering fully right now; please write again in a few minutes.",
}


ACK = {"devanagari": "जी, ठीक है 🙏", "mr": "हो, ठीक आहे 🙏", "indic": "Ji, theek hai 🙏", "english": "Okay, noted 🙏",
       "bengali": "ঠিক আছে 🙏", "tamil": "சரி 🙏", "telugu": "సరే 🙏", "kannada": "ಸರಿ 🙏", "malayalam": "ശരി 🙏",
       "gujarati": "ઠીક છે 🙏", "gurmukhi": "ਠੀਕ ਹੈ 🙏", "odia": "ଠିକ ଅଛି 🙏"}


def ack_reply(p: dict | None) -> str:
    return guards.canned(p, ACK)


def partial_reply(p: dict | None) -> str:
    return guards.canned(p, PARTIAL)


async def run_turn(session: AsyncSession, host: ToolHost, req: TurnRequest) -> TurnResult:
    started = time.monotonic()
    await _lock_family(session, req.family_id)

    inserted = await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="user", text=req.text,
        speaker_id=req.speaker["id"], message_ref=req.message_ref,
        meta={"channel": req.channel, "images": len(req.images), **({"voice": True, "voice_confidence": req.voice_confidence} if req.voice else {})},
    )
    if req.message_ref and inserted is None:
        prior = (
            await session.execute(
                select(Turn).where(Turn.family_id == req.family_id, Turn.meta["reply_to"].astext == req.message_ref)
            )
        ).scalar_one_or_none()
        await session.commit()
        return TurnResult(reply=prior.text if prior else "", actions=[], alerts=[], duplicate=True)

    from app.care import outcomes

    if outcomes.parse_button(req.text):
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
    avoid = [w for w in avoid if not w.startswith("__unreachable__")]
    dynamic, known_turn = await turn_context(session, req, data)
    data.history_full = True
    msgs = await history(session, req, data)
    said = f"(voice note, transcript) {req.text}" if req.voice else req.text
    content = [{"type": "text", "text": f"[{clock.ist().strftime('%d %b %H:%M')}] {said}"}]
    content += [{"type": "image", "mime": i["mime"], "data": i["data"]} for i in req.images]
    msgs.append({"role": "user", "content": content})
    turn_start = len(msgs) - 1

    is_system = req.speaker.get("role") == "system"
    profiles = await writing_profiles(session, req.family_id, [req.elder, *[m for m in req.members if m.get("id") != req.elder["id"]]], data)
    if not is_system:
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
        members=req.members, message_ref=req.message_ref, user_text="" if is_system else req.text,
        known=[fam_block, known_record, known_turn, dynamic, req.text, *[_msg_text(m) for m in msgs]],
        meds=await medicine_times(data), profiles=profiles,
        situation=situation, playbook_version=pb.version if pb else 0, arm=arm, voice_unsure=req.voice_unsure,
    )
    ctx.known.append("")  # tool results are appended as they come
    stable = [PERSONA + "\n\n" + REPLY_FORMAT, fam_block]
    brain_role = role_for(situation)
    guarded = False
    degraded = ""
    reply: router.LLMReply | None = None
    for step in range(MAX_STEPS):
        last = step == MAX_STEPS - 1
        try:
            reply = await router.complete(
                brain_role,
                system_stable=stable,
                system_dynamic=dynamic + ("\n\nYou have used all your steps: reply to the person now with what you have." if last else ""),
                messages=msgs,
                # On the last step there are no tools, so the person always gets a reply.
                tools=None if last else tools.specs(),
                max_tokens=6000,
                effort=BRAIN_EFFORT,
                essential=not is_system,  # past the hard cap only people's messages get answers
            )
        except router.AllModelsFailed:
            if not any(a.get("ok") and a["tool"] in WRITE_TOOLS for a in ctx.actions):
                raise  # nothing happened yet: the caller may safely try another path
            # Something was already saved or sent: keep it (a retry elsewhere would do it twice) and say so plainly.
            logger.error("brain failed mid-turn family=%s after %s", req.family_id, [a["tool"] for a in ctx.actions])
            degraded = "none" if is_system else partial_reply(ctx.profiles.get(req.speaker["id"]))
            reply = None
            break
        msgs.append(reply.as_message())
        if not reply.tool_calls:
            text_now = (reply.text or "").strip()
            if guarded or is_system or not text_now or last:
                break
            problems = await guard_problems(session, req, ctx, text_now, avoid=avoid, unreachable=unreachable, data=data)
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

    final = (reply.text if reply else degraded).strip()
    if final and final != "none" and guards.leaked_reasoning(final):
        # Last line of defence: the model's own notes never reach a person, even when the rewrite leaked again or the
        # turn was a scheduled one (those skip the rewrite).
        logger.error("leaked reasoning blocked family=%s", req.family_id)
        final = "none" if is_system else ""
        guarded = True
    if guarded and not final and not is_system:
        # The rewrite came back empty. The earlier text failed the checks, so it is not sent: a short, true acknowledgement is.
        final = ack_reply(ctx.profiles.get(req.speaker["id"]))

    turn_id = await store.add_turn(
        session, family_id=req.family_id, thread_id=req.speaker["id"], role="assistant", text=final,
        meta={"reply_to": req.message_ref, "model": reply.model if reply else "", "actions": [a["tool"] for a in ctx.actions]},
    )
    if not is_system:
        await _learn_from_turn(session, req, ctx, final, turn_id, role, lang, trace=turn_trace(msgs, turn_start, final))
    if data.history_full:  # fewer turns than a full history cannot need folding yet
        await maybe_compact(session, req.family_id, req.speaker["id"])
    await session.commit()
    return TurnResult(
        reply=final,
        actions=ctx.actions,
        alerts=ctx.alerts,
        model=reply.model if reply else "",
        shadow_writes=getattr(host, "would_have", []),
        ms=int((time.monotonic() - started) * 1000),
        buttons=ctx.buttons if final and final != "none" else [],
    )


TRACE_RESULT_CHARS = 1500


def turn_trace(msgs: list[dict], start: int, final: str) -> list[dict]:
    """This turn as the model saw it, from the person's message on: tool calls, trimmed results, the final reply."""
    out: list[dict] = []
    for m in msgs[start:]:
        if m.get("role") == "user":
            out.append({"role": "user", "text": _msg_text(m)[:2000]})
        elif m.get("role") == "assistant":
            calls = [{"name": c.name, "args": c.args} for c in m.get("tool_calls") or []]
            if calls:
                out.append({"role": "model", "calls": calls, **({"text": m.get("text")} if m.get("text") else {})})
        elif m.get("role") == "tool":
            out.append({"role": "tool", "results": [{"name": r.get("name"), "content": str(r.get("content"))[:TRACE_RESULT_CHARS],
                                                     "error": bool(r.get("is_error"))} for r in m.get("results") or []]})
    out.append({"role": "model", "text": final})
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
