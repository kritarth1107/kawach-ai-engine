"""Pending health facts never wait forever (trust plan A4; founder 2026-10-13: "If nobody approves one in 14 days, ask once
more, then archive it. It shouldn't sit there forever.").

A pending fact is a change a weaker source proposed (an elder's own word on a medicine, a re-read, an unclear voice note)
that waits for a caregiver. Its life, run every night:
- day 0: asked as it is today (the confirm_fact question, the dashboard banner);
- day 14 unanswered: ONE final ask to the person who can confirm (a caregiver; a self-care user confirms their own), as a
  wake-up loop at 11:00 that Saheli turns into one message (the approved template when the chat is closed);
- day 17 (3 days after the final ask, founder decision D7) still unanswered: archived. It leaves the brain's context and
  the dashboard lists, stays in memory history ("filed away: nobody confirmed") and comes back with one tap (restore), or
  when a person says it again (a new fact). Archived is not rejected: nobody said it was wrong.
With nobody able to confirm (an elder with no caregiver account), it is archived at day 14 with a note.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store, versions
from app.care.models import CareFact, OpenLoop
from app.core import clock

logger = logging.getLogger(__name__)

FINAL_ASK_AFTER = timedelta(days=14)
ARCHIVE_AFTER_ASK = timedelta(days=3)
ARCHIVED_NOTE = "archived: nobody confirmed"


def _final_key(fact: CareFact) -> str:
    return f"pending-final:{fact.id}"


async def _confirmer(session: AsyncSession, family_id: str, subject_id: str) -> str | None:
    """Who can confirm a change about this person: themselves in self care, else a caregiver."""
    roster = await store.roster(session, family_id)
    members = list((roster.members if roster else None) or [])
    me = next((m for m in members if m.get("id") == subject_id), None)
    if me and "caregiver" in str(me.get("role", "")).lower():
        return subject_id  # self care: their own record
    for m in members:
        if m.get("id") != subject_id and "primary" in str(m.get("role", "")).lower():
            return m.get("id")
    return next((m.get("id") for m in members if m.get("id") != subject_id and "caregiver" in str(m.get("role", "")).lower()), None)


async def archive(session: AsyncSession, fact: CareFact, *, why: str) -> None:
    now = clock.now()
    fact.status, fact.valid_to, fact.note = "archived", now, f"{ARCHIVED_NOTE} ({why})"[:500]
    await session.execute(
        update(OpenLoop).where(OpenLoop.family_id == fact.family_id, OpenLoop.status == "open",
                               OpenLoop.kind.in_(("confirm_fact", "memory_check")),
                               (OpenLoop.detail["fact_id"].astext == str(fact.id)) | (OpenLoop.dedupe_key == _final_key(fact)))
        .values(status="done", updated_at=now, closed_note="archived")
    )
    await session.flush()
    await versions.record_fact(session, fact, "archive")
    await store.record_event(session, family_id=fact.family_id, subject_id=fact.subject_id, kind="fact_archived",
                             summary=f"Filed away, nobody confirmed: {fact.text}"[:300], payload={"key": fact.key, "fact_id": str(fact.id)})


async def restore(session: AsyncSession, fact_id, *, by: str) -> CareFact | None:
    """An archived fact back to pending (one tap on the dashboard, or a caregiver asking for it back)."""
    row = await session.get(CareFact, fact_id, with_for_update=True)
    if not row or row.status != "archived":
        return None
    row.status, row.valid_to, row.recorded_at, row.note = "pending", None, clock.now(), f"restored by {by}"[:500]
    await session.flush()
    await versions.record_fact(session, row, "restore")
    return row


async def run(session: AsyncSession, *, now=None) -> dict:
    """The nightly pass over every pending fact. Idempotent: running it twice asks and archives nothing twice."""
    now = now or clock.now()
    rows = list((await session.execute(select(CareFact).where(CareFact.status == "pending",
                                                              CareFact.recorded_at < now - FINAL_ASK_AFTER))).scalars())
    out = {"final_asks": 0, "archived": 0}
    for f in rows:
        asked = (await session.execute(select(OpenLoop).where(OpenLoop.family_id == f.family_id, OpenLoop.dedupe_key == _final_key(f))
                                       .order_by(OpenLoop.created_at.desc()))).scalars().first()
        who = await _confirmer(session, f.family_id, f.subject_id)
        if asked is None:
            if who is None:
                await archive(session, f, why="no one can confirm it")
                out["archived"] += 1
                continue
            day = clock.ist_day(f.recorded_at)
            await store.open_loop(
                session, family_id=f.family_id, subject_id=f.subject_id, kind="memory_check",
                title=(f"Saheli heard \"{f.text}\" on {day} and nobody has confirmed it. Should it be kept? "
                       "It will be filed away in 3 days if nobody confirms."),
                detail={"problem": f"waiting for a caregiver since {day}: {f.text}", "kind": "pending_final", "fact_id": str(f.id),
                        "key": f.key, "max_wakes": 1},
                owner_id=who, wake_at=clock.ist(now).replace(hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1),
                alert_rule="dashboard", dedupe_key=_final_key(f),
            )
            out["final_asks"] += 1
        elif asked.created_at <= now - ARCHIVE_AFTER_ASK:
            await archive(session, f, why="no answer to the final ask")
            out["archived"] += 1
    return out
