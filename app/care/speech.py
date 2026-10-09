"""What Saheli would *say* in a voice note, not read out (founder 2026-10-09: "voice should not just read it like a script;
it should feel like a human talking, pauses, emotions").

One fast model call turns the WhatsApp text into a spoken script (short sentences, at most one natural filler, pause
marks) and names the mood, which picks the voice's emotion. The backend's TTS speaks it. Facts cannot drift: every number
must survive and the length stays close, else the text is spoken as it is.
"""

from __future__ import annotations

import json
import logging
import re

from app.llm import router

logger = logging.getLogger(__name__)

MOODS = ("concerned", "reassuring", "cheerful", "gentle", "neutral")
TIMEOUT_S = 4.5

PROMPT = """You turn Saheli's WhatsApp message into what she would SAY in a voice note to this person. Saheli is a warm, caring young Indian woman.
- Same meaning and every fact: medicine names, people's names, times and numbers exactly as written (keep digits as digits).
- Same language or dialect, in the same script as the message. Never translate.
- Sound like a person talking, not reading: short sentences, everyday spoken words, at most one natural filler where it fits
  (अच्छा / देखिए / हाँ जी, or the language's own), no lists, no emoji, no symbols, nothing added that the message does not say.
- Mark 1 to 3 pauses where a person would breathe or think, with [short pause] or [medium pause].
- mood: concerned | reassuring | cheerful | gentle | neutral, from what the message says.
Return only JSON {"script": "...", "mood": "..."}."""

_INDIC_ZERO = (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66)
_DIGITS = {chr(z + i): str(i) for z in _INDIC_ZERO for i in range(10)}
PAUSE = re.compile(r"\s*\[(?:short|medium|long) pause\]\s*")


def plain(text: str) -> str:
    return re.sub(r"\s{2,}", " ", PAUSE.sub(" ", text)).strip()


def _numbers(text: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:[.:]\d+)?", "".join(_DIGITS.get(c, c) for c in text)))


def same_points(text: str, script: str) -> bool:
    """Every number kept and no padding with new content."""
    said = plain(script)
    return _numbers(text) == _numbers(said) and len(text) * 0.6 <= len(said) <= len(text) * 1.8 + 40


def _json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


async def prepare(text: str, language: str) -> dict:
    """{"script", "mood", "prepared"}; on any doubt the text itself (prepared false, mood "")."""
    fallback = {"script": text, "mood": "", "prepared": False}
    if not text.strip() or len(text) > 1500:
        return fallback
    try:
        reply = await router.complete(
            "speech", system_stable=PROMPT, messages=[{"role": "user", "content": [{"type": "text", "text": f"Listener speaks: {language}.\nMessage:\n{text}"}]}],
            max_tokens=1500, effort="low", timeout_s=TIMEOUT_S, essential=False,
        )
    except Exception as exc:  # noqa: BLE001 — the voice note still goes out with the text as written
        logger.warning("speech prepare failed: %s", exc)
        return fallback
    out = _json(reply.text)
    script = str(out.get("script") or "").strip()
    if not script or not same_points(text, script):
        logger.info("speech prepare rejected (facts or length changed)")
        return fallback
    mood = out.get("mood") if out.get("mood") in MOODS else ""
    return {"script": script, "mood": mood, "prepared": True, "model": reply.model}
