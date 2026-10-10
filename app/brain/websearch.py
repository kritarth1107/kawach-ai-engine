"""Saheli's internet look-up: one question → a short answer with its sources, from Gemini grounded on Google Search.

For facts Saheli does not have (what a thing is, a place's hours, weather, festival dates, general health information).
Not for the family's own information (memory) or prices and orders (start_task). Each family gets a daily number of
searches; every search is in the day's ledger.
"""

from __future__ import annotations

import logging
import os

from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

PROMPT = ("Answer this question for an elderly person's care companion in India, using current web results. Plain facts "
          "only, at most 6 short sentences; prefer Indian sources and units (₹, Indian places). For health questions use "
          "reliable medical sources and never give a dose or tell anyone to change a medicine. Say so when the results do not "
          "answer it.\n\nQuestion: ")


def daily_limit() -> int:
    return int(os.getenv("WEB_SEARCH_PER_FAMILY_DAY", "25"))


_client: genai.Client | None = None


def _gemini() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=get_settings().gcp_project_id, location=os.getenv("WEB_SEARCH_LOCATION", "global"))
    return _client


async def search(question: str) -> dict:
    """{answer, sources: [{title, url}]} or raises on failure."""
    resp = await _gemini().aio.models.generate_content(
        model=os.getenv("WEB_SEARCH_MODEL", "gemini-3.5-flash"),
        contents=PROMPT + question.strip()[:400],
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.2,
                                           max_output_tokens=900),
    )
    import re

    answer = re.sub(r"\s*\[\d+(?:\.\d+)*(?:,\s*\d+(?:\.\d+)*)*\]", "", (resp.text or "")).strip()  # citation markers
    sources: list[dict] = []
    for cand in resp.candidates or []:
        meta = getattr(cand, "grounding_metadata", None)
        for chunk in (getattr(meta, "grounding_chunks", None) or []):
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None) and len(sources) < 4:
                sources.append({"title": (getattr(web, "title", "") or "")[:80], "url": web.uri})
    if not answer:
        raise RuntimeError("no answer from the search")
    return {"answer": answer[:1500], "sources": sources}
