"""What Saheli would *say* in a voice note, not read out (founder 2026-10-09: "voice should not just read it like a script;
it should feel like a human talking, pauses, emotions").

One fast model call turns the WhatsApp text into a spoken script (short sentences, at most one natural filler, pause
marks) and names the mood, which picks the voice's emotion. The backend's TTS speaks it. Facts cannot drift: every number
must survive and the length stays close, else the text is spoken as it is.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.llm import router

logger = logging.getLogger(__name__)

MOODS = ("concerned", "reassuring", "cheerful", "gentle", "neutral")
TIMEOUT_S = 6.0

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


async def _ask(route, text: str, language: str) -> tuple[str, str]:
    reply = await router.complete(
        "speech", system_stable=PROMPT, messages=[{"role": "user", "content": [{"type": "text", "text": f"Listener speaks: {language}.\nMessage:\n{text}"}]}],
        max_tokens=1500, effort="low", timeout_s=TIMEOUT_S, essential=False, routes=[route],
    )
    return reply.text, reply.model


def _good(text: str, raw: str) -> dict | None:
    out = _json(raw)
    script = str(out.get("script") or "").strip()
    if not script or not same_points(text, script):
        return None
    return {"script": script, "mood": out.get("mood") if out.get("mood") in MOODS else ""}


async def prepare(text: str, language: str) -> dict:
    """{"script", "mood", "prepared"}; on any doubt or after TIMEOUT_S the text itself (prepared false, mood "").

    Every model on the speech route is asked at once and the first good answer wins: a single call's latency varied from
    2 to 8 s live, and a voice note waits for this."""
    fallback = {"script": text, "mood": "", "prepared": False}
    if not text.strip() or len(text) > 1500:
        return fallback
    pending = {asyncio.ensure_future(_ask(r, text, language)) for r in router.routes_for("speech")}
    deadline = asyncio.get_running_loop().time() + TIMEOUT_S
    try:
        while pending:
            done, pending = await asyncio.wait(pending, timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                                               return_when=asyncio.FIRST_COMPLETED)
            if not done:
                break
            for task in done:
                if task.exception():
                    logger.warning("speech prepare failed on one model: %s", task.exception())
                    continue
                raw, model = task.result()
                good = _good(text, raw)
                if good:
                    return {**good, "prepared": True, "model": model}
                logger.info("speech prepare rejected from %s (facts or length changed)", model)
        return fallback
    finally:
        for task in pending:
            task.cancel()
