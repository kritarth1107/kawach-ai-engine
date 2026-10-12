"""The Kavach care team is told (founder 2026-10-13): "if any of our users is talking [of suicide], inform us via email
kritarth@kavach.care; any user, alert us immediately".

- crisis: talk of suicide, self-harm, wanting to die or harming someone, even if it may be a joke or a test. A later
  "just kidding" never cancels it: the team decides.
- emergency_alone: a medical red flag with no one else in the family to tell (a person caring for themselves).

The first alert of an episode is emailed at once. Alerts within SEND_GAP of the last email wait and go together in one
update (the wake job sends them), so a crisis over five messages is two or three emails, not five, and never late by
more than a few minutes. Each alert is a care event (kind care_team_alert): sent, or waiting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent
from app.core import clock

logger = logging.getLogger(__name__)

SEND_GAP = timedelta(minutes=3)  # alerts closer than this to the last email wait for one update
KIND = "care_team_alert"


async def _rows(session: AsyncSession, family_id: str, subject_id: str, since) -> list[CareEvent]:
    return list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == KIND,
                                CareEvent.at >= since).order_by(CareEvent.id)
    )).scalars())


def _at(v) -> datetime | None:
    try:
        return datetime.fromisoformat(str(v)) if v else None
    except ValueError:
        return None


def _last_sent(rows: list[CareEvent]) -> datetime | None:
    return max((t for t in (_at((r.payload or {}).get("sent_at")) for r in rows if (r.payload or {}).get("state") == "sent") if t),
               default=None)


def _line(e: CareEvent) -> dict:
    p = e.payload or {}
    return {"at": e.at.isoformat(), "said": p.get("said") or "", "summary": e.summary}


async def _send(session: AsyncSession, host, family_id: str, subject_id: str, rows: list[CareEvent], *, update: bool) -> bool:
    if family_id.startswith("shadow:") or not rows:
        return False
    kind = "crisis" if any((r.payload or {}).get("kind") == "crisis" for r in rows) else "emergency_alone"
    told = max(int((r.payload or {}).get("family_told") or 0) for r in rows)
    try:
        out = await host.call("alert_care_team", {"kind": kind, "lines": [_line(r) for r in rows], "family_told": told, "update": update},
                              family_id=family_id, subject_id=subject_id, actor_id=subject_id)
    except Exception:  # noqa: BLE001 — stays waiting: the wake job tries again
        logger.exception("care team alert failed family=%s", family_id)
        return False
    ok = bool((out or {}).get("sent"))
    for r in rows:
        r.payload = {**(r.payload or {}), "state": "sent" if ok else "waiting", "sent_at": clock.now().isoformat() if ok else None}
    return ok


async def notify(session: AsyncSession, host, *, family_id: str, subject_id: str, kind: str, summary: str, said: str = "",
                 family_told: int = 0, actor_id: str | None = None) -> str:
    """Record the alert and email the team: 'sent' (now), 'waiting' (goes with the next update in a few minutes), or
    'not sent' (shadow)."""
    if family_id.startswith("shadow:"):
        return "not sent"
    now = clock.now()
    ev_id = await store.record_event(session, family_id=family_id, subject_id=subject_id, kind=KIND, summary=summary[:500],
                                     payload={"kind": kind, "said": said[:600], "family_told": family_told, "state": "waiting"},
                                     actor_id=actor_id, ref=f"{KIND}:{subject_id}:{now.isoformat()}")
    recent = await _rows(session, family_id, subject_id, now - timedelta(days=1))
    last_sent = _last_sent(recent)
    if last_sent and now - last_sent < SEND_GAP:
        return "waiting"
    waiting = [r for r in recent if (r.payload or {}).get("state") == "waiting"] or [r for r in recent if r.id == ev_id]
    ok = await _send(session, host, family_id, subject_id, waiting, update=last_sent is not None)
    return "sent" if ok else "waiting"


async def flush(sessions: async_sessionmaker, host_for) -> int:
    """The wake job: alerts that waited (sent in the same few minutes as an email, or a failed send) go as one update."""
    now = clock.now()
    sent = 0
    async with sessions() as session:
        rows = list((await session.execute(
            select(CareEvent).where(CareEvent.kind == KIND, CareEvent.at >= now - timedelta(days=1))
            .where(CareEvent.payload["state"].astext == "waiting").order_by(CareEvent.id)
        )).scalars())
        by_person: dict[tuple[str, str], list[CareEvent]] = {}
        for r in rows:
            if not r.family_id.startswith("shadow:"):
                by_person.setdefault((r.family_id, r.subject_id), []).append(r)
        for (fid, sid), waiting in by_person.items():
            recent = await _rows(session, fid, sid, now - timedelta(days=1))
            last_sent = _last_sent(recent)
            if last_sent and now - last_sent < SEND_GAP:
                continue
            sent += await _send(session, host_for(fid), fid, sid, waiting, update=last_sent is not None)
        await session.commit()
    return sent
