"""The learning loop's scheduled work.

Nightly (inside Saheli's dream, after the families): score replies, grow the anonymised corpus, grade a small
sample, decide on the canary, drop raw logs past retention. Weekly (Sunday night, or POST /v2/jobs/learn-weekly):
learn a new playbook draft, gate it, start its canary.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core import clock
from app.learn import gaps, grader, lessons, scoring, timing
from app.learn.models import LearningExample, PlaybookVersion, ReplyLog

logger = logging.getLogger(__name__)


async def nightly(sessions: async_sessionmaker, *, with_models: bool = True) -> dict:
    out: dict = {}
    steps = [("scored", lambda: scoring.score_pending(sessions)), ("corpus", lambda: scoring.build_corpus(sessions))]
    if with_models:
        steps.append(("graded", lambda: grader.grade_sample(sessions)))
    steps += [("canary", lambda: lessons.evaluate(sessions)), ("pruned", lambda: scoring.prune(sessions))]
    for name, fn in steps:
        try:
            out[name] = await fn()
        except Exception as exc:  # noqa: BLE001 — one step must not stop the others
            logger.exception("learn step %s failed", name)
            out[name] = {"error": str(exc)[:200]}
    if with_models and clock.ist().weekday() == 6:  # Sunday night: learn a new draft
        try:
            out["weekly"] = await lessons.weekly(sessions)
        except Exception as exc:  # noqa: BLE001
            logger.exception("weekly lessons failed")
            out["weekly"] = {"error": str(exc)[:200]}
    timing.reset_cache()
    return out


async def overview(sessions: async_sessionmaker, *, weeks: int = 8) -> dict:
    """Everything the founder's learning view shows."""
    from datetime import timedelta

    since = clock.now() - timedelta(weeks=weeks)
    async with sessions() as session:
        pbs = list((await session.execute(select(PlaybookVersion).order_by(PlaybookVersion.version.desc()).limit(12))).scalars())
        week = func.date_trunc("week", ReplyLog.at)
        trend = (await session.execute(
            select(week, ReplyLog.situation, func.count(), func.avg(ReplyLog.score))
            .where(ReplyLog.at >= since, ReplyLog.score.is_not(None)).group_by(week, ReplyLog.situation).order_by(week)
        )).all()
        totals = (await session.execute(
            select(func.count(), func.count(ReplyLog.scored_at), func.avg(ReplyLog.score)).where(ReplyLog.at >= since)
        )).one()
        corpus = (await session.execute(select(func.count()).select_from(LearningExample))).scalar_one()
        gap_rank = await gaps.ranking(session)
        timing_view = await timing.summary(session)
    return {
        "messages": int(totals[0] or 0), "scored": int(totals[1] or 0), "avgScore": round(float(totals[2]), 3) if totals[2] is not None else None,
        "corpus": int(corpus),
        "playbooks": [{"version": p.version, "status": p.status, "createdAt": p.created_at.isoformat(), "note": p.note,
                       "lessons": p.lessons, "examples": p.examples, "gate": p.gate, "canary": p.canary, "approvedBy": p.approved_by,
                       "liveSince": p.live_since.isoformat() if p.live_since else None} for p in pbs],
        "trend": [{"week": w.date().isoformat(), "situation": s, "n": int(n), "avgScore": round(float(a), 3)} for w, s, n, a in trend],
        "gaps": gap_rank,
        "timing": timing_view,
    }
