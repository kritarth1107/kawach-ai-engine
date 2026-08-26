"""LLM-backed metadata extraction for uploaded family documents."""

from __future__ import annotations

import json
import re
from typing import Any

from app.llm.provider import chat_invoke

ALLOWED_KINDS = {"lab", "scan", "prescription", "note", "vitals", "chat_export"}

SYSTEM_PROMPT = """You analyze documents for a family caregiving memory system (Saheli).
Extract ONLY what is explicitly printed in the document text. Do not diagnose or interpret clinically.

Respond with a single JSON object (no markdown fences) using exactly these keys:
- title: short descriptive document title (max 80 characters)
- kind: one of lab, scan, prescription, note, vitals, chat_export
- tags: array of 3-8 short lowercase tags (e.g. "tsh", "diabetes", "bp")
- summary: 1-2 factual sentences about what the document contains
- record_date: date string found in the document (e.g. "8 Aug 2026") or null
- highlights: array of up to 8 key printed values or facts as short strings

If the text is empty or unreadable, use kind "note", empty tags, and summary explaining that."""


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model response")
    return json.loads(text[start : end + 1])


def _normalize_kind(value: str | None) -> str:
    kind = (value or "note").strip().lower()
    if kind == "vitals":
        return "vitals"
    if kind not in ALLOWED_KINDS:
        return "lab" if "lab" in kind or "report" in kind else "note"
    return kind


async def analyze_document_text(
    *,
    title: str,
    raw_text: str,
    file_name: str | None = None,
) -> dict[str, Any]:
    snippet = (raw_text or "").strip()
    if not snippet:
        fallback_title = (file_name or title).rsplit(".", 1)[0][:80] if file_name else title[:80]
        return {
            "title": fallback_title or "Uploaded document",
            "kind": "note",
            "tags": ["upload"],
            "summary": f"Uploaded file {file_name or title} with no extractable text.",
            "record_date": None,
            "highlights": [],
        }

    clipped = snippet[:14_000]
    user = f"Title: {title}\nFile: {file_name or 'unknown'}\n\nDocument text:\n{clipped}"

    try:
        reply = await chat_invoke(SYSTEM_PROMPT, user)
        parsed = _parse_json_payload(reply)
    except Exception:
        return {
            "title": title[:80] or "Health record",
            "kind": "lab" if re.search(r"\b(TSH|HbA1c|glucose|hemoglobin)\b", clipped, re.I) else "note",
            "tags": ["health-record"],
            "summary": clipped[:240] + ("…" if len(clipped) > 240 else ""),
            "record_date": None,
            "highlights": [],
        }

    tags = parsed.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower()[:40] for t in tags if str(t).strip()][:8]

    highlights = parsed.get("highlights") or []
    if not isinstance(highlights, list):
        highlights = []
    highlights = [str(h).strip()[:200] for h in highlights if str(h).strip()][:8]

    record_date = parsed.get("record_date")
    if record_date is not None:
        record_date = str(record_date).strip()[:80] or None

    summary = str(parsed.get("summary") or "").strip()[:500]
    if not summary:
        summary = clipped[:240] + ("…" if len(clipped) > 240 else "")

    detected_title = str(parsed.get("title") or "").strip()[:80]
    if not detected_title:
        detected_title = title[:80] or clipped.split("\n", 1)[0][:80] or "Health record"

    return {
        "title": detected_title,
        "kind": _normalize_kind(parsed.get("kind")),
        "tags": tags,
        "summary": summary,
        "record_date": record_date,
        "highlights": highlights,
    }
