"""What the family said is wrong, kept so it never comes back (trust plan A3).

A fact a caregiver rejected (a pending proposal turned down: status retracted), or one a person stopped as a mistake
(`stop` with wrong: true, note "rejected: …"), or a learned behaviour undone (A5). The list is shown to the brain as
"never say or use", given to the nightly re-read and the duplicate check so it is never proposed again, and checked in
every reply. It comes back only when a person says it again in chat, which makes a new fact from a person.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import CareFact
from app.core import clock

REJECTED_NOTE = "rejected: "
LOOK_BACK = timedelta(days=365)


async def rejected(session: AsyncSession, family_id: str, subject_id: str, *, now=None, limit: int = 30) -> list[CareFact]:
    """Newest first; one row per key (the latest rejection)."""
    since = (now or clock.now()) - LOOK_BACK
    rows = (await session.execute(
        select(CareFact).where(
            CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.recorded_at >= since,
            or_(CareFact.status == "retracted", and_(CareFact.status == "stopped", CareFact.note.like(REJECTED_NOTE + "%"))),
        ).order_by(CareFact.recorded_at.desc())
    )).scalars().all()
    active = {k for (k,) in (await session.execute(
        select(CareFact.key).where(CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.status == "active")
    )).all()}
    out: dict[str, CareFact] = {}
    for r in rows:
        # a key a person has since set again is not rejected any more (the newer active fact wins)
        if r.key in out or (r.key in active and r.status == "retracted"):
            continue
        out[r.key] = r
    return list(out.values())[:limit]


def terms(rows: list[CareFact]) -> list[str]:
    """The structured names in rejected facts (the item, medicine or condition name), for checking Saheli's own reply."""
    found: list[str] = []
    for r in rows:
        v = r.value or {}
        for k in ("name", "item", "allergen", "condition", "medicine", "food", "dish"):
            t = v.get(k)
            if isinstance(t, str) and len(t.strip()) >= 3:
                found.append(t.strip())
        items = v.get("items")
        if isinstance(items, list):
            found += [i.strip() for i in items if isinstance(i, str) and len(i.strip()) >= 3]
    seen: dict[str, str] = {}
    for t in found:
        seen.setdefault(t.lower(), t)
    return list(seen.values())


def block(rows: list[CareFact]) -> str:
    """The brain's "never say or use" lines."""
    if not rows:
        return ""
    lines = [f"  - {r.text}" for r in rows[:15] if r.text]
    return "Never say, assume or use (the family said these are wrong):\n" + "\n".join(lines) if lines else ""
