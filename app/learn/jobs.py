"""The learning loop's scheduled work.

Nightly (inside Saheli's dream, after the families): score replies, grow the anonymised corpus, decide on the canary,
drop raw logs past retention. Every real reply is graded by the full-memory judge inside each family's night
(app.quality.nightly); the scorer and the corpus read its verdict from reply_log.judge_pass. Weekly (Sunday night, or POST /v2/jobs/learn-weekly):
learn a new playbook draft, gate it, start its canary.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core import clock
from app.learn import gaps, lessons, review, scoring, timing
from app.learn.models import LearningExample, PlaybookVersion, ReplyLog

logger = logging.getLogger(__name__)


async def nightly(sessions: async_sessionmaker, *, with_models: bool = True) -> dict:
    out: dict = {}
    steps = [("scored", lambda: scoring.score_pending(sessions)), ("corpus", lambda: scoring.build_corpus(sessions))]
    from app.learn import corrections

    steps += [("canary", lambda: lessons.evaluate(sessions)), ("drift", lambda: review.drift(sessions)), ("pruned", lambda: scoring.prune(sessions)),
              ("corrections", lambda: corrections.nightly(sessions))]
    for name, fn in steps:
        try:
            out[name] = await fn()
        except Exception as exc:  # noqa: BLE001 — one step must not stop the others
            logger.exception("learn step %s failed", name)
            out[name] = {"error": str(exc)[:200]}
    if with_models:
        # §10, every night: failures become rule proposals, safe ones go into the trial automatically (founder Q10), and
        # a new draft is tried when no canary is running (the replay gate, then 20% of families, then everyone).
        for name, fn in (("rule_proposals", lambda: review.propose_rules(sessions)), ("auto_rules", lambda: _auto_rules(sessions)),
                         ("draft", lambda: _draft_if_idle(sessions))):
            try:
                out[name] = await fn()
            except Exception as exc:  # noqa: BLE001
                logger.exception("learn %s failed", name)
                out[name] = {"error": str(exc)[:200]}
    if with_models and clock.ist().weekday() == 6:  # Sunday night: review the week, then learn a new draft
        for name, fn in (("calibration", lambda: review.calibrate(sessions)), ("gap_specs", lambda: review.gap_specs(sessions))):
            try:
                out[name] = await fn()
            except Exception as exc:  # noqa: BLE001
                logger.exception("weekly %s failed", name)
                out[name] = {"error": str(exc)[:200]}
    timing.reset_cache()
    return out


async def _auto_rules(sessions: async_sessionmaker) -> dict:
    async with sessions() as session:
        out = await review.auto_approve(session)
        await session.commit()
    return out


async def _draft_if_idle(sessions: async_sessionmaker) -> dict:
    async with sessions() as session:
        busy = (await session.execute(select(PlaybookVersion.version).where(PlaybookVersion.status.in_(("canary", "draft"))).limit(1))).first()
    if busy:  # one trial at a time; a waiting draft is retried (it may have been waiting for the replay runner)
        waiting = (await _waiting_draft(sessions))
        return await lessons.gate_and_canary(sessions, waiting) if waiting else {"skipped": "a canary is running"}
    return await lessons.weekly(sessions)


async def _waiting_draft(sessions: async_sessionmaker) -> int | None:
    async with sessions() as session:
        row = (await session.execute(select(PlaybookVersion.version).where(PlaybookVersion.status == "draft")
                                     .order_by(PlaybookVersion.version.desc()).limit(1))).first()
    return row[0] if row else None


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
            "families": (row[4] or {}).get("families"), "familiesError": (row[4] or {}).get("familiesError"),
            "lastOkAt": ok_row[0].isoformat() if ok_row and ok_row[0] else None}
