"""The learning loop's scheduled work.

Nightly (inside Saheli's dream, after the families): score replies, grow the anonymised corpus, grade a small
sample, decide on the canary, drop raw logs past retention. Weekly (Sunday night, or POST /v2/jobs/learn-weekly):
learn a new playbook draft, gate it, start its canary.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core import clock
from app.learn import gaps, grader, lessons, review, scoring, timing
from app.learn.models import LearningExample, PlaybookVersion, ReplyLog

logger = logging.getLogger(__name__)


async def nightly(sessions: async_sessionmaker, *, with_models: bool = True) -> dict:
    out: dict = {}
    steps = [("scored", lambda: scoring.score_pending(sessions)), ("corpus", lambda: scoring.build_corpus(sessions))]
    if with_models:
        steps.append(("graded", lambda: grader.grade_sample(sessions)))
    steps += [("canary", lambda: lessons.evaluate(sessions)), ("drift", lambda: review.drift(sessions)), ("pruned", lambda: scoring.prune(sessions))]
    for name, fn in steps:
        try:
            out[name] = await fn()
        except Exception as exc:  # noqa: BLE001 — one step must not stop the others
            logger.exception("learn step %s failed", name)
            out[name] = {"error": str(exc)[:200]}
    if with_models and clock.ist().weekday() == 6:  # Sunday night: review the week, then learn a new draft
        for name, fn in (("calibration", lambda: review.calibrate(sessions)), ("rule_proposals", lambda: review.propose_rules(sessions)),
                         ("gap_specs", lambda: review.gap_specs(sessions)), ("weekly", lambda: lessons.weekly(sessions))):
            try:
                out[name] = await fn()
            except Exception as exc:  # noqa: BLE001
                logger.exception("weekly %s failed", name)
                out[name] = {"error": str(exc)[:200]}
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
        from app.learn.models import RuleProposal

        proposals = list((await session.execute(select(RuleProposal).order_by(RuleProposal.created_at.desc()).limit(20))).scalars())
    reports = {k: await review.latest(sessions, k) for k in ("calibration", "drift", "gap_specs")}
    return {
        "backup": await last_backup(sessions),
        "messages": int(totals[0] or 0), "scored": int(totals[1] or 0), "avgScore": round(float(totals[2]), 3) if totals[2] is not None else None,
        "corpus": int(corpus),
        "playbooks": [{"version": p.version, "status": p.status, "scope": p.scope or [], "createdAt": p.created_at.isoformat(), "note": p.note,
                       "lessons": p.lessons, "examples": p.examples, "gate": p.gate, "canary": p.canary, "approvedBy": p.approved_by,
                       "liveSince": p.live_since.isoformat() if p.live_since else None} for p in pbs],
        "trend": [{"week": w.date().isoformat(), "situation": s, "n": int(n), "avgScore": round(float(a), 3)} for w, s, n, a in trend],
        "gaps": gap_rank,
        "timing": timing_view,
        "ruleProposals": [{"id": r.id, "situation": r.situation, "rule": r.rule, "why": r.why, "evidence": r.evidence, "status": r.status,
                           "at": r.created_at.isoformat()} for r in proposals],
        "calibration": reports["calibration"], "drift": reports["drift"], "gapSpecs": reports["gap_specs"],
    }


async def last_backup(sessions: async_sessionmaker) -> dict | None:
    """Newest row the backup job wrote (backup/backup.py creates the table itself)."""
    async with sessions() as session:
        try:
            row = (await session.execute(text(
                "SELECT day, ok, finished_at, bytes, detail FROM backup_runs ORDER BY id DESC LIMIT 1"))).first()
            ok_row = (await session.execute(text(
                "SELECT finished_at FROM backup_runs WHERE ok ORDER BY id DESC LIMIT 1"))).first()
        except Exception:  # noqa: BLE001 — no backup has run yet
            return None
    if not row:
        return None
    return {"day": row[0].isoformat(), "ok": bool(row[1]), "finishedAt": row[2].isoformat() if row[2] else None,
            "bytes": int(row[3] or 0), "error": (row[4] or {}).get("error"),
            "lastOkAt": ok_row[0].isoformat() if ok_row and ok_row[0] else None}
