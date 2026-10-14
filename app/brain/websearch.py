"""Saheli's internet look-up: one question → a short answer with its sources, from Gemini grounded on Google Search.

For facts Saheli does not have (what a thing is, a place's hours, weather, festival dates, general health information).
Not for the family's own information (memory) or prices and orders (start_task). Each family gets a daily number of
searches; every search is in the day's ledger.

`grounded()` is the shared call (also the nutrition lookup's, app.nutrition.lookup): a prompt in, the text and its
sources out, priced into the day's model spend (tokens plus the search fee).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

PROMPT = ("Answer this question for an elderly person's care companion in India, using current web results. Plain facts "
          "only, at most 6 short sentences; prefer Indian sources and units (₹, Indian places). State the facts directly: no "
          "\"according to…\" or \"sources say\" (the sources are listed separately). For health questions use reliable medical "
          "sources, include when to see a doctor, and never give a dose or tell anyone to change a medicine. Say so when the "
          "results do not answer it.\n\nQuestion: ")
# Gemini counts its thinking in the output limit: at 900 the answer was cut mid-sentence ("the target post-meal level is
# typically below 18"; hard-turn A/B, 14 Oct), and Saheli passed on half a fact.
OUT_TOKENS = 4000
TIMEOUT_S = float(os.getenv("WEB_SEARCH_TIMEOUT_S", "25"))  # usually 7-10 s; a hung search held a turn for minutes


def daily_limit() -> int:
    return int(os.getenv("WEB_SEARCH_PER_FAMILY_DAY", "25"))


_clients: dict[str, genai.Client] = {}


def _gemini(location: str | None = None) -> genai.Client:
    loc = location or os.getenv("WEB_SEARCH_LOCATION", "global")
    if loc not in _clients:
        _clients[loc] = genai.Client(vertexai=True, project=get_settings().gcp_project_id, location=loc)
    return _clients[loc]


@dataclass
class Grounded:
    text: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    inr: float = 0.0


def search_fee_inr() -> float:
    """Google Search grounding is billed per search on top of tokens (about $14 per 1,000 for Gemini 3); a prompt may run more
    than one search, so the estimate is two."""
    return float(os.getenv("GROUNDED_SEARCH_FEE_INR", "2.5"))


async def grounded(prompt: str, *, location: str | None = None, model: str | None = None, max_tokens: int = OUT_TOKENS,
                   temperature: float = 0.2, role: str = "web_search", max_sources: int = 4, timeout_s: float | None = None) -> Grounded:
    """One Gemini call grounded on Google Search. Raises on failure, a timeout or an empty answer. Thinking is kept low:
    Gemini counts it in the output limit, and a cut answer passes on half a fact."""
    import asyncio

    from app.llm import policy_block, spend

    why = policy_block.blocked("gemini")
    if why:  # a policy refusal of the project is never retried (app.llm.policy_block): nothing is sent while it lasts
        raise RuntimeError(f"not sent: the provider refused this project by policy ({why[:80]})")
    name = model or os.getenv("WEB_SEARCH_MODEL", "gemini-3.5-flash")
    try:
        resp = await asyncio.wait_for(_gemini(location).aio.models.generate_content(
            model=name, contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=temperature,
                                               max_output_tokens=max_tokens, thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)),
        ), timeout=timeout_s or TIMEOUT_S)
    except Exception as exc:
        if policy_block.is_policy_refusal(getattr(exc, "code", None) or getattr(exc, "status", None), str(exc)):
            policy_block.note("gemini", str(exc))
        raise
    text = re.sub(r"\s*\[\d+(?:\.\d+)*(?:,\s*\d+(?:\.\d+)*)*\]", "", (resp.text or "")).strip()  # citation markers
    sources: list[dict] = []
    for cand in resp.candidates or []:
        meta = getattr(cand, "grounding_metadata", None)
        for chunk in (getattr(meta, "grounding_chunks", None) or []):
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None) and len(sources) < max_sources:
                sources.append({"title": (getattr(web, "title", "") or "")[:80], "url": web.uri})
    meta = getattr(resp, "usage_metadata", None)
    usage = {"in": getattr(meta, "prompt_token_count", 0) or 0, "out": getattr(meta, "candidates_token_count", 0) or 0,
             "think": getattr(meta, "thoughts_token_count", 0) or 0}
    inr = 0.0
    try:
        inr = await spend.record(role, name, usage, extra_inr=search_fee_inr())
    except Exception:  # noqa: BLE001 — pricing never fails the answer
        logger.warning("grounded search spend not recorded", exc_info=True)
    if not text:
        raise RuntimeError("no answer from the search")
    return Grounded(text=text, sources=sources, model=name, inr=inr)


async def search(question: str) -> dict:
    """{answer, sources: [{title, url}]} or raises on failure."""
    out = await grounded(PROMPT + question.strip()[:400])
    return {"answer": out.text[:1500], "sources": out.sources}
