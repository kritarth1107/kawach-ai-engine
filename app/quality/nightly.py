"""The daily score's part of Saheli's night (02:00 dream, app.care.dream).

Per family, after the dream's own steps:
1. the duplicate merge (app.care.dedupe; FACT_MERGE=dryrun lists, on applies), so the memory score sees the record as
   it stands after it (the 02:15 consolidate job then skips the families merged here);
2. the full-memory judge on the day's sample of real Saheli messages (app.quality.judge);
3. checks 2 and 3 and the reply pass rate (app.quality.daily);
4. the memory score (app.quality.memory_score), with unknown truths asked once and failures filed as replay cases;
5. replay cases from the judge's failures.
After every family: check 1 from the delivery ledger, each family's overall and everyone's (app.quality.daily.finish).
The brief goes out later in the day (each caregiver's own time), so it is always after the score.
Every step is isolated: one failing step is recorded and the rest still run.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care.models import CareEvent, CareFact, FamilyRoster
from app.quality import daily, judge, memory_score, replay, snapshot
from app.quality.models import ReplyVerdict

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.getenv("DAILY_SCORE", "on").strip().lower() != "off"


async def merge_groups(session, family_id: str, *, apply: bool) -> dict[str, list[dict]]:
    """The nightly duplicate merge for one family (as app.care.dedupe.merge_family), keeping the groups that remain
    duplicates after it: all of them on a dry run, the report-only ones (medicines) when applied."""
    from app.care import dedupe, versions

    subjects = sorted({sid for (sid,) in (await session.execute(
        select(CareFact.subject_id).where(CareFact.family_id == family_id, CareFact.status == "active").distinct())).all()})
    out: dict[str, list[dict]] = {}
    for sid in subjects:
        groups, pairs = await dedupe.find(session, family_id, sid)
        if not groups and not pairs:
            continue
        if apply:
            with versions.attribution(actor_id="saheli", source="nightly", reason="the same fact saved more than once"):
                await dedupe.apply(session, family_id, sid, groups, pairs)
        left = [g for g in groups if not apply or g.domain in dedupe.REPORT_ONLY]
        if left:
            out[sid] = [{"domain": g.domain, "keep": g.keep, "drop": list(g.drop), "text": g.text} for g in left]
    return out


async def merged_tonight(session, family_id: str, day: str) -> bool:
    """Did tonight's dream already run the duplicate merge for this family?"""
    ev = (await session.execute(select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == "dream",
                                                        CareEvent.ref == f"dream:{day}"))).scalar_one_or_none()
    return bool(ev and ((ev.payload or {}).get("quality") or {}).get("merged"))


async def judge_cases(sessions: async_sessionmaker, roster: FamilyRoster, day: str) -> int:
    """Each failed verdict of the day becomes a replay case (context as it was, the bad reply, what must hold)."""
    from app.learn.models import ReplyLog

    n = 0
    async with sessions() as s:
        vs = list((await s.execute(select(ReplyVerdict).where(ReplyVerdict.family_id == roster.family_id, ReplyVerdict.day == day,
                                                              ReplyVerdict.passed.is_(False)))).scalars())
        for v in vs:
            r = await s.get(ReplyLog, v.reply_id)
            if r is None:
                continue
            snap = await snapshot.take(s, roster, r.thread_id, r.at, message=r.user_text, speaker_role=r.speaker_role, trace=r.trace, kind=r.kind)
            naming = next((f for f in snap["facts"] if f["key"] == "naming:address_as" and f["status"] == "active"), None)
            lang = next((f for f in snap["facts"] if f["domain"] == "language" and f["status"] == "active" and f["subject"] == r.thread_id), None)
            lv = (lang or {}).get("value") or {}
            rules, rubric = replay.from_verdict(v.verdict or {}, avoid=((naming or {}).get("value") or {}).get("avoid") or [], script=lv.get("script"),
                                                language=lv.get("language"))
            ok = await replay.add_snapshot(s, source="judge", source_ref=f"judge:reply:{r.id}", kind=(v.issues or ["judge"])[0], day=day, snap=snap,
                                           message=r.user_text, bad_reply=r.text, assertion=rules, rubric=rubric, family_id=roster.family_id,
                                           subject_id=r.thread_id, thread_id=r.thread_id)
            n += 1 if ok else 0
        await s.commit()
    return n


async def family(sessions: async_sessionmaker, roster: FamilyRoster, day: str) -> dict:
    """The daily score's steps for one family (inside its dream)."""
    from app.care import dedupe

    fid = roster.family_id
    out: dict = {"errors": []}

    async def step(name: str, fn):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 — one step must not stop the rest
            logger.exception("daily score step %s failed family=%s", name, fid)
            out["errors"].append(f"{name}: {str(exc)[:160]}")
            return None

    groups: dict[str, list[dict]] = {}
    mode = dedupe.merge_mode()
    if mode != "off":
        async def merge():
            async with sessions() as s:
                got = await merge_groups(s, fid, apply=mode == "on")
                await s.commit()
                return got

        got = await step("merge", merge)
        if got is not None:
            groups = got
            out["merged"] = True
            out["duplicates_left"] = sum(len(v) for v in groups.values())
    out["judge"] = await step("judge", lambda: judge.judge_replies(sessions, fid, day))
    out["checks"] = await step("checks", lambda: daily.family_checks(sessions, fid, day))

    async def memory():
        async with sessions() as s:
            got = await memory_score.score_family(s, roster, day, groups=groups)
            await s.commit()
        got.pop("per_person", None)
        return got

    out["memory"] = await step("memory", memory)
    out["cases"] = await step("cases", lambda: judge_cases(sessions, roster, day))
    return out


async def finish(sessions: async_sessionmaker, day: str, host=None) -> dict:
    """Check 1 and the totals; then the night's other replay cases (corrections, guard rewrites, thumbs-down) and the
    export of every case to the test bucket for the pre-deploy runner (trust plan C3)."""
    from app.brain.host import LiveHost
    from app.quality import cases

    out = await daily.finish(sessions, host or LiveHost(), day)
    for name, fn in (("cases_from_corrections", lambda: cases.from_corrections(sessions)), ("cases_export", lambda: cases.export(sessions))):
        try:
            out[name] = await fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", name)
            out[name] = {"error": str(exc)[:200]}
    return out
