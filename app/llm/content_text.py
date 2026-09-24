"""Safe extraction of visible text from LLM content blocks.

Never stringify whole content objects — Gemini / Vertex may return
lists/dicts with thought_signature that must not leak to WhatsApp.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_IGNORE_KEYS = frozenset(
    {
        "thought_signature",
        "thoughtSignature",
        "signature",
        "thought",
        "thinking",
    }
)


def _text_from_mapping(block: Mapping[str, Any]) -> str:
    for bad in _IGNORE_KEYS:
        if bad in block and "text" not in block and "type" not in block:
            return ""
    # Prefer explicit text parts; ignore thought_signature blobs.
    typ = block.get("type")
    text = block.get("text")
    if typ == "text" and isinstance(text, str):
        return text.strip()
    if isinstance(text, str) and text.strip() and typ not in ("thinking", "thought"):
        return text.strip()
    # Nested parts
    parts = block.get("parts") or block.get("content")
    if parts is not None and parts is not block:
        return stringify_ai_content(parts)
    return ""


def stringify_ai_content(content: object) -> str:
    """Extract user-visible text only. Never str(whole content)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Mapping):
        return _text_from_mapping(content)
    # Objects with .text (e.g. some SDK part types)
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        typ = getattr(content, "type", None)
        if typ not in ("thinking", "thought"):
            return text_attr.strip()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
                continue
            if isinstance(block, Mapping):
                extracted = _text_from_mapping(block)
                if extracted:
                    parts.append(extracted)
                continue
            nested = stringify_ai_content(block)
            if nested:
                parts.append(nested)
        return "\n".join(parts).strip()
    # Last resort: refuse to dump opaque objects / thought signatures.
    return ""
