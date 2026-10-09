"""Dose outcomes, by day and dose time. A later answer about the same dose replaces the earlier one.

Founder (2026-10-09): "whatever they say should be updated". Maa said she took everything but Folvite, then corrected
herself: nothing today, and yesterday everything but Folvite. The first answer stayed on the dashboard (it read the
oldest event of the day) and yesterday could not be marked at all.

An answer that is replaced keeps its row for the record, with kind dose_corrected and payload was/replaced_by, so every
reader that counts dose_* kinds sees only the latest answer.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import slug
from app.care.models import CareEvent
from app.core import clock

KINDS = ("dose_taken", "dose_skipped", "dose_refused", "dose_missed", "dose_empty_strip")
CORRECTED = "dose_corrected"
MAX_DAYS_BACK = 7


def hhmm(t: object) -> str | None:
    """'8:00 AM', '08:00', '20:30', '9 pm' → 'HH:MM' (None if it is not a time)."""
    m = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*", str(t or ""), re.I)
    if not m:
        return None
    h, mins, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if ampm.startswith("p") and h < 12:
        h += 12
    if ampm.startswith("a") and h == 12:
        h = 0
    return f"{h:02d}:{mins:02d}" if h < 24 and mins < 60 else None


def resolve_day(day: str | None, now: datetime | None = None) -> str | None:
    """'today' / 'yesterday' / 'YYYY-MM-DD' → the IST day, or None if it is in the future or too long ago."""
    now = now or clock.now()
    today = clock.ist_day(now)
    word = (day or "today").strip().lower()
    if word in ("today", "aaj", ""):
        return today
    if word in ("yesterday", "kal"):
        return clock.ist_day(now - timedelta(days=1))
    try:
        d = datetime.strptime(word, "%Y-%m-%d").date()
    except ValueError:
        return None
    oldest = clock.ist(now).date() - timedelta(days=MAX_DAYS_BACK)
    return d.isoformat() if oldest <= d <= clock.ist(now).date() else None


def _first(name: str) -> str:
    return slug(name).split("_")[0]


async def medicine_times(session: AsyncSession, family_id: str, subject_id: str, medicine: str) -> list[str]:
    first = _first(medicine)
    for f in await store.facts(session, family_id, subject_id, domains=["medicine"], statuses=("active",)):
        if first and first == _first(str(f.value.get("name") or f.key.split(":", 1)[-1])):
            return sorted(t for t in (hhmm(x) for x in f.value.get("times") or []) if t)
    return []


def _slot(times: list[str], given: str | None, at_time: str | None) -> str | None:
    """Which dose of the day this is about: the time given, the only time, or (said on the day) the latest dose time
    that has come by then; None when it cannot be told."""
    if given:
        return given
    if len(times) == 1:
        return times[0]
    if times and at_time:
        came = [t for t in times if t <= at_time]
        return came[-1] if came else times[0]
    return None


async def record(
    session: AsyncSession,
    *,
    family_id: str,
    subject_id: str,
    kind: str,
    medicine: str,
    summary: str,
    day: str | None = None,
    time: str | None = None,
    actor_id: str | None = None,
    ref: str | None = None,
    payload: dict | None = None,
) -> int | None:
    """Record a dose outcome for a day (default today) and retire earlier answers about the same dose."""
    now = clock.now()
    today = clock.ist_day(now)
    day = day or today
    times = await medicine_times(session, family_id, subject_id, medicine)
    on_the_day = day == today
    slot = _slot(times, hhmm(time), clock.ist(now).strftime("%H:%M") if on_the_day else None)
    at = None
    if not on_the_day:  # it belongs to that day's record, at the dose's time
        h, m = map(int, (slot or "21:00").split(":"))
        at = datetime.strptime(day, "%Y-%m-%d").replace(hour=h, minute=m, tzinfo=clock.IST)
    new_id = await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind=kind, summary=summary, actor_id=actor_id, ref=ref, at=at,
        payload={**(payload or {}), "medicine": medicine, "day": day, **({"time": slot} if slot else {})},
    )
    if new_id is None:
        return None
    first = _first(medicine)
    earlier = (await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.day == day,
                                CareEvent.kind.in_(KINDS), CareEvent.id != new_id)
    )).scalars()
    for e in earlier:
        p = e.payload or {}
        if not first or _first(str(p.get("medicine") or e.summary.split(":")[0])) != first:
            continue
        theirs = _slot(times, hhmm(p.get("time")), clock.ist(e.at).strftime("%H:%M"))
        if slot != theirs and (slot or times):  # another dose of the day (a twice-daily medicine)
            continue
        e.payload = {**p, "was": e.kind, "replaced_by": new_id}
        e.kind = CORRECTED
    await session.flush()
    return new_id
