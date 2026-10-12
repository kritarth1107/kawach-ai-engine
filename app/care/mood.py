"""How the person seems, through the day (founder 2026-10-13: "moods in dashboard should be also changed regularly and
show properly"; Home showed "—" all day because its mood came only from the backend's nightly snapshot).

A mood is a care event of kind "mood" with a score (1 very low … 5 very good, the scale the backend's wellbeing days use)
and one word for the dashboard. Saheli logs it in the conversation (log_event); the 30-minute re-read fills in a
conversation where she did not. The same score again within MERGE_GAP is not logged twice; a change always is.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import CareEvent
from app.core import clock

SCALE = {1: "very low", 2: "low", 3: "okay", 4: "good", 5: "very good"}
MERGE_GAP = timedelta(minutes=45)
STALE = timedelta(hours=12)  # the re-read does not log a mood from a conversation older than this (a backlog read late)


def score_of(v) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n in SCALE else None


def _severity(score: int) -> str:
    return "concern" if score == 1 else "watch" if score == 2 else "info"


async def last(session: AsyncSession, family_id: str, subject_id: str, *, since=None) -> CareEvent | None:
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "mood")
    if since is not None:
        q = q.where(CareEvent.at >= since)
    return (await session.execute(q.order_by(CareEvent.at.desc(), CareEvent.id.desc()).limit(1))).scalar_one_or_none()


async def record(session: AsyncSession, *, family_id: str, subject_id: str, score: int | None, word: str, summary: str,
                 severity: str | None = None, actor_id: str | None = None, source: str = "chat", at=None) -> bool:
    """Log a mood (at: when they showed it, now by default). False when it repeats the last one (same score within MERGE_GAP)."""
    now = at or clock.now()
    prev = await last(session, family_id, subject_id, since=now - MERGE_GAP)
    if prev is not None and score is not None and score_of((prev.payload or {}).get("score")) == score:
        return False
    word = (word or "").strip().lower()[:24]
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="mood", summary=summary[:500],
        payload={"severity": severity or (_severity(score) if score else "info"), **({"score": score} if score else {}),
                 **({"word": word} if word else {}), "source": source},
        actor_id=actor_id, at=now,
    )
    return True


def as_json(e: CareEvent | None) -> dict | None:
    if e is None:
        return None
    p = e.payload or {}
    score = score_of(p.get("score"))
    return {"score": score, "word": p.get("word") or (SCALE[score] if score else None), "summary": e.summary,
            "at": e.at.isoformat(), "source": p.get("source") or "chat"}


async def now_for(session: AsyncSession, family_id: str, subject_id: str) -> dict | None:
    """The latest mood heard in the last 24 hours, for Home."""
    return as_json(await last(session, family_id, subject_id, since=clock.now() - timedelta(hours=24)))
