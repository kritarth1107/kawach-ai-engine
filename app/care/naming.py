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


async def for_people(session: AsyncSession, family_id: str, subject_ids: list[str]) -> dict[str, dict]:
    """{subject id: {"name", "avoid"}} for everyone in the household who has a naming fact (one query)."""
    from sqlalchemy import select

    from app.care.models import CareFact

    ids = [s for s in dict.fromkeys(subject_ids) if s]
    if not ids:
        return {}
    rows = (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id.in_(ids),
                                                         CareFact.key == "naming:address_as", CareFact.status == "active"))).scalars()
    out: dict[str, dict] = {}
    for f in rows:
        v = f.value or {}
        avoid = [w.strip() for w in (v.get("avoid") or []) if isinstance(w, str) and w.strip()]
        raw = v.get("name")
        name = raw.strip() if isinstance(raw, str) and raw.strip() else None
        if name or avoid:
            out[f.subject_id] = {"name": name, "avoid": avoid}
    return out


def line(who: dict | None) -> str:
    """How to address a person, for the household block: "call them Kittu; never: Maa"."""
    if not who:
        return ""
    bits = [f"call them {who['name']}"] if who.get("name") else []
    if who.get("avoid"):
        bits.append("never: " + ", ".join(who["avoid"]))
    return "; ".join(bits)


def address_problems(text: str, *, roster_name: str, who: dict | None) -> list[str]:
    """Saheli's own message to this person against their naming fact: an avoided name used, or their roster first name used
    where the family set another name to use (13 Oct: check-ins and order updates opened "Kritarth, …" to a caregiver whose
    saved name is "Kittu"; messages she starts never had the guard). Code on her output and the record, never their words."""
    if not who:
        return []
    out = [f"never call them '{w}'" + (f"; call them {who['name']}" if who.get("name") else "") for w in uses_avoided(text, who.get("avoid") or [])]
    name = who.get("name") or ""
    first = (roster_name or "").split()[0] if (roster_name or "").split() else ""
    if name and len(first) > 1 and first.lower() not in name.lower() and uses_avoided(text, [first]) and not uses_avoided(text, [name]):
        out.append(f"call them {name}, the name they asked for, not {first}")
    return out
