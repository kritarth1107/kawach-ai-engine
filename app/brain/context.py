"""What one turn shows the model: the family, the care record, the thread, and the blocks around them.

The turn itself (lock, tool calls, reply checks) stays in loop.py. These names are re-exported from
loop so existing callers keep working.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain import guards, photos
from app.care import digest, freshness, links, store
from app.care.models import Turn
from app.core import clock

if TYPE_CHECKING:
    from app.brain.loop import TurnRequest

logger = logging.getLogger(__name__)

# Older turns in the prompt; the care record keeps the facts, the stored turn stays whole.
HISTORY_TURNS = 40
PROMPT_TURN_CHARS = 500
LIGHT_HISTORY = 8

class TurnData:
    """What one turn reads from memory, fetched once and shared by the context, the guards and the tools' checks
    (the turn used to read the same facts, threads and tasks several times)."""

    def __init__(self, session: AsyncSession, req: TurnRequest) -> None:
        self.session, self.req = session, req
        self._facts: dict[str, list] = {}
        self._turns: dict[str, list] | None = None
        self._tasks: list | None = None
        self.history_full = True  # set by history(): did the thread fill a whole history window?
        self.open_tasks = False  # an order or ride is already running (its tools come back on)

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
    data.open_tasks = bool(tasks)
    dose = await dose_answer_block(data, req)
    tapped_open = (req.text or "").strip() == "open:waiting"
    parts = [
        f"NOW: {now.strftime('%A %d %B %Y, %H:%M')} IST",
        *(["THEIR MESSAGE is a tap on \"read message\" under Saheli's notice, not words of theirs. What waited was just "
           "given (JUST GIVEN below) or nothing waited. Do not repeat it and do not answer it. Write one short, warm line that "
           "fits this time of day (NOW) and their record, or reply none. Never a line from your earlier messages, and "
           "nothing they did not tell you."] if tapped_open else []),
        f"SPEAKING: {req.speaker.get('name')} ({'the care recipient' if req.speaker['id'] == req.elder['id'] else req.speaker.get('role', 'family')}), via {req.channel}",
        voice_block(req),
        digest.ledger(day_events),
        digest.loops(loops),
        ("ACTIVE TASKS (orders and rides running in the background):\n" + "\n".join(f"  - {task_runtime.describe(t)}" for t in tasks))
        if tasks else "ACTIVE TASKS: none.",
        await reminders_block(session, req),
        *([dose] if dose else []),
        await medicines_block(session, req, data),
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

        names = cast(dict[str, str], {m.get("id"): m.get("name") for m in [req.elder, *req.members]})
        parts.append(work.brief(await work.items(session, req.family_id), names, max_lines=6))
    from app.care import baselines as care_baselines
    from app.care import patterns as care_patterns

    for pid, pname in {req.elder["id"]: req.elder.get("name"), req.speaker["id"]: req.speaker.get("name")}.items():
        line = care_baselines.usual_line(pname or "them", await care_baselines.get(session, req.family_id, pid))
        if line:
            parts.append(line)

    noticed = await care_patterns.recent(session, req.family_id, list({req.elder["id"], req.speaker["id"]}))
    if noticed:
        names = cast(dict[str, str], {m.get("id"): m.get("name") for m in [req.elder, *req.members]})
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
        "  1. If they just greeted you: two short lines and stop. Welcome them as Saheli from Kavach (check-ins, medicine "
        "reminders, orders and rides, always asking first), then ask which language they want, including their own bhasha "
        "(language_preference, about their own id). Do not ask the rest of setup in this reply.\n"
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


async def medicines_block(session: AsyncSession, req: TurnRequest, data: TurnData) -> str:
    """Every saved medicine with what is known about it: dose, times, course dates, purpose, and the tablets left the family
    counted (the dashboard's Stock & refills). Saheli does not ask for what is listed here (live 17 Oct: she asked a
    caregiver for a medicine count the dashboard already showed)."""
    from sqlalchemy import select as _select

    from app.care import features
    from app.care.models import CareEvent

    people = [req.elder["id"]]
    if req.speaker.get("role") != "system" and req.speaker["id"] != req.elder["id"]:
        people.insert(0, req.speaker["id"])
    names = {m.get("id"): m.get("name") for m in [req.elder, *req.members]}
    lines: list[str] = []
    for pid in people:
        meds = [f for f in await data.facts(pid) if f.domain == "medicine" and f.status == "active" and (f.value or {}).get("name")]
        if not meds:
            continue
        stock: dict[str, dict] = {}
        counted = (await session.execute(_select(CareEvent.id).where(
            CareEvent.family_id == req.family_id, CareEvent.subject_id == pid, CareEvent.kind == "stock_set").limit(1))).first()
        if counted:
            try:
                async with session.begin_nested():
                    stock = {r["key"]: r for r in await features.stock(session, req.family_id, pid)}
            except Exception:  # noqa: BLE001 — the list still shows without counts
                logger.exception("stock for context failed family=%s", req.family_id)
        who = "the speaker" if pid == req.speaker["id"] else (names.get(pid) or "them")
        lines.append(f"  {who}:")
        for n, f in enumerate(sorted(meds, key=lambda x: x.key)[:15], 1):
            v = f.value or {}
            bits = [" ".join(str(x) for x in (v.get("name"), v.get("strength") or "", v.get("dose") or "") if x).strip()]
            if v.get("salt"):
                bits.append(str(v["salt"]))
            bits.append("at " + ", ".join(map(str, v["times"])) if v.get("times") else "no reminder times")
            if v.get("food_timing"):
                bits.append(str(v["food_timing"]).replace("_", " "))
            if v.get("purpose"):
                bits.append(f"for {v['purpose']}")
            if v.get("ends_on"):
                bits.append(f"course {v.get('starts_on') or ''} to {v['ends_on']}".replace("  ", " "))
            row = stock.get(f.key)
            if row and row.get("stock") is not None:
                bits.append(f"{row['stock']} left" + (f", about {row['daysLeft']} days" if row.get("daysLeft") is not None else "")
                            + (f" (counted {clock.ist(datetime.fromisoformat(row['asOf'])).strftime('%d %b')})" if row.get("asOf") else ""))
            lines.append(f"    {n}. [{f.key}] " + "; ".join(b for b in bits if b))
    if not lines:
        return ""
    return ("MEDICINES ON RECORD (what the dashboard shows; numbered as you would list them). Use these: never ask for a count, "
            "dose, time or stock listed here. To stop or delete one: stop with domain medicine and its name. When they ask to "
            "delete or stop medicines without saying which, list that person's medicines with these numbers plus 'all', and ask "
            "which:\n" + "\n".join(lines))


async def reminders_block(session: AsyncSession, req: TurnRequest) -> str:
    """Reminders due in the next hours, so one the person says is already done is skipped instead of going out anyway
    (live 2026-10-12: "i had the medicine" at 00:12, and the reminder for it still came at 00:15).
    A medicine that was stopped or replaced is left out: its old time is not an answer."""
    from app.care import doses

    ended, names = await doses.ended_medicines(session, req.family_id, [req.elder["id"], req.speaker.get("id") or ""])
    rows = [r for r in (req.reminders_soon or [])
            if isinstance(r, dict) and r.get("id") and str(r.get("item") or "") not in ended
            and not doses.mentions_ended_medicine(str(r.get("text") or ""), names)]
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
        said = t.text or ""
        if len(said) > PROMPT_TURN_CHARS:
            said = said[:PROMPT_TURN_CHARS].rstrip() + "…"
        if t.role == "user":
            # A photo they sent stays in the thread as what it showed (app.brain.photos), not as an empty line.
            seen = photos.history_note(t.meta)
            if seen:
                said = f"{seen[:900]} {said}".rstrip()
            msgs.append({"role": "user", "content": [{"type": "text", "text": f"[{stamp}] {said}"}]})
        else:
            msgs.append({"role": "assistant", "text": said, "tool_calls": []})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


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
