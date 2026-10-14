"""The daily score's routes (trust plan C1, A2, C2).

/v2/admin (the admin API's identity + ENGINE_ADMIN_SECRET, as every admin route):
  GET  /daily-score?days=30&day=     the live score, its trend, failures, memory scores, the nightly world score
  GET  /eval-scores?days=30          the nightly world scores
  GET  /calibration                  the judge's verdicts to label this month, and the agreement
  POST /calibration/{id}             a person's yes/no on one verdict
/v2/admin, authenticated service account (the nightly workflow's own Google identity, EVAL_POSTER_SA):
  POST /eval-scores                  the night's scores.json; stored in eval_scores; a failing night emails the team
/v2/jobs (X-Kavach-Secret, Cloud Scheduler):
  POST /jobs/morning-report          08:00 IST: the team's email (once a day)
  POST /jobs/daily-score?day=        run the daily score for one day now (the 02:00 dream runs it every night)
"""

from __future__ import annotations

import os
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin import caller_email, verify_admin_caller
from app.core.security import verify_api_secret
from app.db.session import get_db

DB = Annotated[AsyncSession, Depends(get_db)]


async def verify_eval_poster(request: Request) -> str:
    """Only the nightly workflow's service account (a Google ID token for the engine's audience)."""
    allowed = {e.strip().lower() for e in os.environ.get("EVAL_POSTER_SA", "").split(",") if e.strip()}
    if not allowed:
        raise HTTPException(status_code=503, detail="eval score posting is switched off")
    email = caller_email(request.headers.get("authorization"))
    if email not in allowed:
        raise HTTPException(status_code=403, detail="caller not allowed")
    return email


router = APIRouter(prefix="/v2/admin", tags=["quality"], dependencies=[Depends(verify_admin_caller)])
poster = APIRouter(prefix="/v2/admin", tags=["quality"])
jobs = APIRouter(prefix="/v2", tags=["quality"], dependencies=[Depends(verify_api_secret)])


@router.get("/daily-score")
async def daily_score(session: DB, days: int = 30, day: str | None = None) -> dict:
    from app.quality import report

    if day:
        try:
            date.fromisoformat(day)
        except ValueError:
            raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD") from None
    return await report.page(session, days=max(1, min(days, 90)), day=day)


@router.get("/eval-scores")
async def eval_scores(session: DB, days: int = 30) -> dict:
    from app.quality import report

    return await report.world(session, max(1, min(days, 120)))


class EvalIn(BaseModel):
    run_id: str = Field(min_length=1, max_length=80)
    kind: str = Field(default="nightly", pattern="^(nightly|fixture|replay)$")
    day: str = Field(min_length=10, max_length=10)
    passed: bool
    commit: str = Field(default="", max_length=64)
    branch: str = Field(default="", max_length=80)
    scores: dict
    failures: list[str] = Field(default_factory=list)


@poster.post("/eval-scores")
async def post_eval_scores(body: EvalIn, session: DB, caller: Annotated[str, Depends(verify_eval_poster)]) -> dict:
    from app.core import clock
    from app.quality.models import EvalScore

    try:
        d = date.fromisoformat(body.day)
    except ValueError:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD") from None
    values = dict(run_id=body.run_id, kind=body.kind, day=d, passed=body.passed, commit=body.commit, branch=body.branch, scores=body.scores,
                  failures=[f[:300] for f in body.failures][:200], created_at=clock.now())
    stmt = insert(EvalScore).values(**values).on_conflict_do_update(
        index_elements=[EvalScore.run_id, EvalScore.kind], set_={k: v for k, v in values.items() if k not in ("run_id", "kind")}).returning(EvalScore)
    row = (await session.execute(stmt)).scalar_one()
    await session.commit()
    emailed = None
    cases_made = 0
    if body.kind == "nightly" and body.branch in ("master", "main"):
        from app.quality import cases

        cases_made = await cases.from_probes(session, body.scores, body.day)  # every probe failure becomes a replay case (C3)
        await session.commit()
    if not body.passed and body.kind == "nightly" and body.branch in ("master", "main"):
        from app.brain.host import LiveHost
        from app.quality import report

        try:
            emailed = await report.nightly_failed(LiveHost(), row)
        except Exception as exc:  # noqa: BLE001 — stored either way; the workflow still fails the night
            emailed = {"error": str(exc)[:200]}
    return {"stored": True, "id": row.id, "by": caller, "emailed": emailed, "cases": cases_made}


class ReplayResultsIn(BaseModel):
    commit: str = Field(default="", max_length=64)
    branch: str = Field(default="", max_length=80)
    results: list[dict] = Field(default_factory=list, max_length=2000)
    summary: dict = Field(default_factory=dict)


@poster.post("/replay-results")
async def post_replay_results(body: ReplayResultsIn, session: DB, caller: Annotated[str, Depends(verify_eval_poster)]) -> dict:
    """The pre-deploy runner's results (trust plan C3.6): on master, an open case that passed becomes fixed."""
    from app.core import clock
    from app.quality import cases
    from app.quality.models import EvalScore

    out = await cases.apply_results(session, body.results, commit=body.commit, branch=body.branch)
    run_id = str(body.summary.get("run_id") or f"replay-{clock.now().strftime('%Y%m%d%H%M%S')}")[:80]
    values = dict(run_id=run_id, kind="replay", day=clock.ist().date(), passed=bool(body.summary.get("passed", True)), commit=body.commit, branch=body.branch,
                  scores=body.summary, failures=[str(f)[:300] for f in body.summary.get("failures") or []][:200], created_at=clock.now())
    await session.execute(insert(EvalScore).values(**values).on_conflict_do_update(
        index_elements=[EvalScore.run_id, EvalScore.kind], set_={k: v for k, v in values.items() if k not in ("run_id", "kind")}))
    await session.commit()
    emailed = None
    if body.summary.get("unavailable") and body.branch in ("master", "main"):
        # the gate could not test (the provider refused every case): the team hears when the streak starts
        from sqlalchemy import select

        from app.brain.host import LiveHost
        from app.quality import report

        gate = await report.replay_gate(session)
        if gate.get("runs") == 1:
            row = (await session.execute(select(EvalScore).where(EvalScore.run_id == run_id, EvalScore.kind == "replay"))).scalar_one()
            try:
                emailed = await report.replay_unavailable(LiveHost(), row, str(gate.get("since")))
            except Exception as exc:  # noqa: BLE001 — stored either way; the page and the morning email show it
                emailed = {"error": str(exc)[:200]}
    return {**out, "by": caller, "emailed": emailed}


def _pb_json(pb) -> dict | None:
    if pb is None:
        return None
    return {"version": pb.version, "status": pb.status, "lessons": pb.lessons or {}, "examples": pb.examples or {}, "scope": pb.scope or [],
            "note": pb.note or "", "gate": pb.gate or {}}


@poster.get("/learn/waiting-draft")
async def learn_waiting_draft(session: DB, caller: Annotated[str, Depends(verify_eval_poster)]) -> dict:
    """The newest playbook draft waiting for its replay gate (trust plan §10.2 step 2), and the live playbook it is compared
    with: the learn-gate workflow runs both through the replay cases in the test project and posts the result back."""
    from sqlalchemy import select

    from app.learn import playbook
    from app.learn.models import PlaybookVersion

    draft = (await session.execute(select(PlaybookVersion).where(PlaybookVersion.status == "draft")
                                   .order_by(PlaybookVersion.version.desc()).limit(1))).scalars().first()
    live, _canary = await playbook.versions(session)
    return {"draft": _pb_json(draft), "live": _pb_json(live), "by": caller}


class GateIn(BaseModel):
    run_id: str = Field(default="", max_length=80)
    regressions: list[str] = Field(default_factory=list, max_length=500)
    fixed: list[str] = Field(default_factory=list, max_length=500)
    skipped: list[str] = Field(default_factory=list, max_length=500)
    n: int = 0
    cost_inr: float = 0.0
    live: int | None = None
    error: str = Field(default="", max_length=500)


@poster.post("/learn/playbooks/{version}/replay")
async def learn_replay_result(version: int, body: GateIn, caller: Annotated[str, Depends(verify_eval_poster)]) -> dict:
    """The replay gate's result for a waiting draft: no regression → its canary starts (20% of families); a regression → it
    is rejected; an error → it keeps waiting (never a silent pass)."""
    from app.db.session import SessionLocal

    return await apply_gate(SessionLocal, version, body, caller)


async def apply_gate(sessions, version: int, body: GateIn, caller: str) -> dict:
    from app.learn import lessons

    if body.error or not body.n:
        replay = {"status": "unavailable", "why": (body.error or "no case ran")[:200], "run_id": body.run_id}
    else:
        replay = {"status": "failed" if body.regressions else "passed", "regressions": body.regressions, "fixed": body.fixed,
                  "skipped": body.skipped, "n": body.n, "cost_inr": body.cost_inr, "live": body.live, "run_id": body.run_id, "by": caller}
    out = await lessons.gate_and_canary(sessions, version, replay=replay)
    if out.get("skipped"):
        raise HTTPException(status_code=409, detail="that version is not a waiting draft")
    return {"version": version, "status": out.get("status"), "replay": replay["status"]}


@router.get("/calibration")
async def calibration_view(session: DB, batch: str | None = None) -> dict:
    from app.db.session import SessionLocal
    from app.quality import calibration

    return {**await calibration.view(session, batch), "agreement": await calibration.agreement(SessionLocal, batch)}


class LabelIn(BaseModel):
    good: bool
    by: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=500)


@router.post("/calibration/{label_id}")
async def calibration_label(label_id: int, body: LabelIn) -> dict:
    from app.db.session import SessionLocal
    from app.quality import calibration

    out = await calibration.label(SessionLocal, label_id, good=body.good, by=body.by, note=body.note)
    if out is None:
        raise HTTPException(status_code=404, detail="no such verdict to label")
    return out


@jobs.post("/jobs/morning-report")
async def morning_report(day: str | None = None, force: bool = False) -> dict:
    from app.brain.host import LiveHost
    from app.db.session import SessionLocal
    from app.quality import report

    return await report.send(SessionLocal, LiveHost(), day, force=force)


@jobs.post("/jobs/daily-score")
async def daily_score_job(day: str, family: str = "", finish: bool = True) -> dict:
    """The daily score for one IST day now, for every family (or ids starting with family=). Safe to run again: a reply
    already judged is not judged twice, and every row is replaced."""
    from sqlalchemy import select

    from app.care.models import FamilyRoster
    from app.db.session import SessionLocal
    from app.quality import nightly

    try:
        date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD") from None
    async with SessionLocal() as s:
        rosters = [r for r in (await s.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:"),
                                                                         ~FamilyRoster.family_id.startswith("sim-")))).scalars()
                   if r.family_id.startswith(family)]
    out: dict = {"day": day, "families": {}}
    for r in rosters:
        out["families"][r.family_id[:8]] = await nightly.family(SessionLocal, r, day)
    if finish:
        out["finish"] = await nightly.finish(SessionLocal, day)
    return out
