"""Content hashing for family_memories deduplication."""

from __future__ import annotations

import hashlib
import re


def normalize_for_hash(content: str) -> str:
    return re.sub(r"\s+", " ", content.lower().strip())


def content_hash(content: str) -> str:
    normalized = normalize_for_hash(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
