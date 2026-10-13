"""The name to use for a person and the names never to use, for every writer (trust plan A3).

One place, read by the brain's reply guard, reminder lines, regular nudges, the nightly brief, template parameters and
the backend's fallback text, so "don't call me Maa" holds in every message, not just in chat replies.
"""

from __future__ import annotations

import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store


async def for_person(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    """{"name": str | None, "avoid": [str]} from the person's active naming fact."""
    f = await store.active_fact(session, family_id, subject_id, "naming:address_as")
    v = (f.value if f else None) or {}
    avoid = [w.strip() for w in (v.get("avoid") or []) if isinstance(w, str) and w.strip()]
    raw = v.get("name")
    name = raw.strip() if isinstance(raw, str) and raw.strip() else None
    return {"name": name, "avoid": avoid}


def uses_avoided(text: str, avoid: list[str]) -> list[str]:
    """Avoided names that appear in Saheli's own text (as a whole word, any script, any case)."""
    hits = []
    for w in avoid:
        w = w.strip()
        if w and re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text or "", re.IGNORECASE):
            hits.append(w)
    return hits
