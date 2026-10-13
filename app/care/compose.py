"""Reminder lines written at the moment they go out (founder 2026-10-13: "send new message every time ... it should be
creative, should not pre-generate the message; generate it at the time of sending").

The backend's reminder scheduler asks here just before it sends. The line is written for this person and this moment:
their language and script, how they like messages (their saved style, emoji if they asked), the time of day, what the
reminder is for, and the last lines sent for it, which it must not repeat. A medicine's name and dose stay exact.
If nothing usable comes back, the backend sends the stored text, so a reminder is never lost.
"""

from __future__ import annotations

import json
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.care import language, skillbook, store
from app.core import clock
from app.llm import router

logger = logging.getLogger(__name__)

PROMPT = """You write one WhatsApp reminder from Saheli, a warm care companion, to one person, at the moment it is sent.

Rules:
- Say clearly what the reminder is for. Keep a medicine's name and dose exactly as given.
- Write it fresh: a different opening, wording and angle from every earlier line listed (not a rephrase of one). Vary the
  approach: a light question, encouragement, a small fact, a callback to their goal, humour when it suits them.
- Short: one or two lines, the way a caring friend texts. No greeting by name every time, no "Reminder:" label.
- Their language and script exactly as told. Emoji only if their style says they like them.
- Never shame, nag or sound like a bot.
- Use the context when it fits (Kavach NUdge's plan §1.7): a streak, something from the last messages ("3 hours clean since
  lunch — that walk idea worked?"), but never contradict the last messages and never invent anything not given here.
Return only JSON: {"text": "..."}"""


def _parse(raw: str) -> str:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return ""
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ""
    return str(data.get("text") or "").strip() if isinstance(data, dict) else ""


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


def fresh(text: str, recent: list[str]) -> bool:
    """Not one of the recent lines, and not opening the same way as the last one."""
    n = _norm(text)
    if not n:
        return False
    seen = [_norm(r) for r in recent if r]
    if n in seen:
        return False
    first = " ".join(n.split()[:3])
    return not (seen and first and " ".join(seen[0].split()[:3]) == first)


async def reminder_text(session: AsyncSession, *, family_id: str, subject_id: str, purpose: str, at: str | None = None,
                        recent: list[str] | None = None, kind: str = "reminder", item: str | None = None) -> dict:
    recent = [r for r in (recent or []) if r][:8]
    roster = await store.roster(session, family_id)
    people = [*([roster.elder] if roster and roster.elder else []), *((roster.members or []) if roster else [])]
    person = next((m for m in people if m.get("id") == subject_id), {"id": subject_id, "name": ""})
    facts = await store.facts(session, family_id, subject_id, statuses=("active",))
    speech = next((language.normalise(f.value) for f in facts if f.domain == "language" and isinstance(f.value, dict)), None)
    naming = next((f.value for f in facts if f.key == "naming:address_as"), {}) or {}
    try:
        style = await skillbook.context_block(session, family_id, subject_id, person.get("name") or "them")
    except Exception:  # noqa: BLE001 — style is a nicety
        style = ""
    now = clock.ist()
    lines = [
        f"PERSON: {person.get('name') or 'them'}" + (f" (call them {naming.get('name')})" if naming.get("name") else ""),
        f"LANGUAGE: {language.label(speech) if speech else 'the language they write in; English if unknown'}"
        + (f"; script: {language.script_for(speech)}" if speech and language.script_for(speech) else ""),
        f"NOW: {now.strftime('%A %H:%M')} IST" + (f"; the reminder is for {at}" if at else ""),
        f"FOR: {purpose}" + (f" (medicine: {item}, keep name and dose exact)" if kind == "medicine" and item else ""),
        "EARLIER LINES (do not repeat or rephrase any):\n" + ("\n".join(f"- {r}" for r in recent) if recent else "- none yet"),
    ]
    prefs = [f.text for f in facts if f.domain == "preference"][:8]
    if prefs:
        lines.append("WHAT THEY TOLD SAHELI THEY LIKE:\n" + "\n".join(f"- {p}" for p in prefs))
    try:
        from app.care import activities

        turns = [t for t in await store.recent_turns(session, family_id, subject_id, limit=12) if (t.text or "").strip() not in ("", "none")][-5:]
        if turns:
            lines.append("LAST MESSAGES (oldest first):\n" + "\n".join(
                f"- {clock.ist(t.at).strftime('%H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text[:160]}" for t in turns))
        habits = [a for a in await activities.status(session, family_id, subject_id) if a["kind"] in ("habit", "quit")]
        if habits:
            lines.append("THEIR HABITS:\n" + "\n".join(
                f"- {a['text']}: today {a['today'].replace('_', ' ')}" + (f", streak {a['streak']} days" if a["streak"] > 1 else "") for a in habits[:6]))
    except Exception:  # noqa: BLE001 — context is a nicety; the reminder still goes
        logger.exception("reminder context failed family=%s", family_id)
    if style:
        lines.append(f"HOW THEY LIKE MESSAGES:\n{style[:1200]}")
    text = ""
    for attempt in range(2):
        try:
            reply = await router.complete("worker", system_stable=PROMPT,
                                          messages=[{"role": "user", "content": [{"type": "text", "text": "\n".join(lines)
                                                     + ("\nThe last try repeated an earlier line; write something new." if attempt else "")}]}],
                                          max_tokens=1500, effort="low")  # thinking counts against it
        except Exception:  # noqa: BLE001 — the backend sends the stored text
            logger.exception("compose reminder failed family=%s", family_id)
            break
        text = _parse(reply.text)
        if fresh(text, recent):
            break
        text = ""
    return {"text": text[:600] if text else None, "fresh": bool(text)}
