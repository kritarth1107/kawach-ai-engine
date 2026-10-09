"""How fresh each saved fact is, so Saheli re-checks old facts before acting on them.

A record is only as good as its last confirmation. A medicine nobody has mentioned in three months may have been stopped;
a doctor or address from last year may have changed. Freshness = the later of when the fact was saved and when someone
last confirmed it (said it again: a fact_unchanged event, or a fact_confirmed event from a check-in).

Stale facts are marked in the care record ("last confirmed over 3 months ago, check before acting on it"), the check-ins
ask about the stalest one now and then, and load-bearing actions (a refill order) re-check first.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import CareEvent, CareFact
from app.core import clock

# How long a fact stays fresh without anyone confirming it. Domains not listed never go stale (allergies, conditions,
# naming, language: they do not change quietly, and asking again would annoy).
MAX_AGE = {
    "medicine": timedelta(days=90),
    "routine": timedelta(days=120),
    "doctor": timedelta(days=365),
    "hospital": timedelta(days=365),
    "home": timedelta(days=365),
    "contact": timedelta(days=365),
}
CONFIRM_KINDS = ("fact_unchanged", "fact_confirmed", "fact_superseded", "fact_created")


def _age_label(age: timedelta) -> str:
    """Coarse on purpose: the care record sits in the cached prompt, so the label changes rarely."""
    months = age.days // 30
    if months >= 12:
        return "over a year ago"
    return f"over {months} months ago" if months >= 2 else "over a month ago"


async def last_confirmed(session: AsyncSession, family_id: str, subject_id: str, keys: list[str]) -> dict[str, datetime]:
    if not keys:
        return {}
    rows = (await session.execute(
        select(CareEvent.payload["key"].astext, CareEvent.at).where(
            CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind.in_(CONFIRM_KINDS),
            CareEvent.payload["key"].astext.in_(keys))
    )).all()
    out: dict[str, datetime] = {}
    for key, at in rows:
        if key not in out or at > out[key]:
            out[key] = at
    return out


async def stale_facts(session: AsyncSession, family_id: str, subject_id: str, facts: list[CareFact], *,
                      now: datetime | None = None) -> dict[str, str]:
    """{fact key: "over 4 months ago"} for active facts past their domain's age."""
    now = now or clock.now()
    live = [f for f in facts if f.status == "active" and f.domain in MAX_AGE]
    seen = await last_confirmed(session, family_id, subject_id, [f.key for f in live])
    out = {}
    for f in live:
        fresh_at = max(filter(None, (f.recorded_at, f.valid_from, seen.get(f.key))))
        age = now - fresh_at
        if age >= MAX_AGE[f.domain]:
            out[f.key] = _age_label(age)
    return out


async def stalest(session: AsyncSession, family_id: str, subject_id: str, facts: list[CareFact], *,
                  now: datetime | None = None, domains: tuple[str, ...] = ("medicine", "doctor", "home")) -> CareFact | None:
    """The one fact most in need of a check (medicines first), or None."""
    now = now or clock.now()
    live = [f for f in facts if f.status == "active" and f.domain in domains]
    seen = await last_confirmed(session, family_id, subject_id, [f.key for f in live])
    best, best_over = None, timedelta(0)
    for f in live:
        over = now - max(filter(None, (f.recorded_at, f.valid_from, seen.get(f.key)))) - MAX_AGE[f.domain]
        if f.domain == "medicine":
            over += timedelta(days=365)  # a stale medicine matters more than a stale address
        if over > best_over:
            best, best_over = f, over
    return best


async def confirm(session: AsyncSession, *, family_id: str, subject_id: str, key: str, by: str | None, summary: str = "") -> None:
    """Someone said the fact is still right: it is fresh again."""
    from app.care import store

    await store.record_event(session, family_id=family_id, subject_id=subject_id, kind="fact_confirmed",
                             summary=summary or f"{key} confirmed", payload={"key": key}, actor_id=by)
