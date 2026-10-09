"""Saheli Brain v2 endpoint. The backend calls it with the raw message and who is speaking."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain.host import LiveHost, ShadowHost
from app.brain.loop import TurnRequest, result_json, run_turn
from app.care import store
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.llm.router import AllModelsFailed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2", tags=["brain"], dependencies=[Depends(verify_api_secret)])


class Person(BaseModel):
    id: str
    name: str = ""
    role: str = ""


class Image(BaseModel):
    mime: str
    data: str


class TurnIn(BaseModel):
    family_id: str
    elder: Person
    speaker: Person
    members: list[Person] = Field(default_factory=list)
    text: str = ""
    message_ref: str | None = None
    images: list[Image] = Field(default_factory=list)
    channel: str = "whatsapp"
    modality: str = Field(default="text", pattern="^(text|voice)$")
    voice_confidence: float | None = Field(default=None, ge=0, le=1)
    voice_language: str | None = Field(default=None, max_length=16)
    # shadow: runs beside the live path on separate memory; writes are recorded, not executed.
    mode: str = Field(default="live", pattern="^(live|shadow)$")


@router.post("/turn")
async def turn(body: TurnIn, session: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    shadow = body.mode == "shadow"
    host = ShadowHost() if shadow else LiveHost()
    req = TurnRequest(
        family_id=("shadow:" if shadow else "") + body.family_id,
        elder=body.elder.model_dump(),
        speaker=body.speaker.model_dump(),
        members=[m.model_dump() for m in body.members],
        text=body.text,
        message_ref=body.message_ref,
        images=[i.model_dump() for i in body.images],
        channel=body.channel,
        modality=body.modality,
        voice_confidence=body.voice_confidence,
        voice_language=body.voice_language,
    )
    try:
        result = await run_turn(session, host, req)
    except AllModelsFailed as exc:
        logger.error("brain down family=%s: %s", body.family_id, exc)
        raise HTTPException(status_code=503, detail="Saheli is unavailable") from exc
    logger.info(
        "brain turn family=%s mode=%s model=%s ms=%s actions=%s alerts=%s",
        body.family_id, body.mode, result.model, result.ms, [a["tool"] for a in result.actions], len(result.alerts),
    )
    return result_json(result)


class EventIn(BaseModel):
    family_id: str
    subject_id: str
    kind: str = Field(max_length=40)
    summary: str = ""
    payload: dict = Field(default_factory=dict)
    ref: str | None = None
    at: datetime | None = None


@router.post("/events")
async def push_event(body: EventIn, session: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    """Things that happened outside a conversation (a reminder sent, a dose marked on the dashboard)."""
    ids = []
    # Shadow memory mirrors the live ledger, so the shadow brain sees the same day.
    for fid in (body.family_id, "shadow:" + body.family_id):
        ids.append(
            await store.record_event(
                session, family_id=fid, subject_id=body.subject_id, kind=body.kind, summary=body.summary,
                payload=body.payload, ref=body.ref, at=body.at,
            )
        )
    await session.commit()
    return {"recorded": ids[0] is not None}


@router.post("/jobs/wake")
async def wake_job() -> dict:
    """Cloud Scheduler, every 5 minutes: run the open loops that are due."""
    from app.brain.wake import wake_due
    from app.db.session import SessionLocal

    return await wake_due(SessionLocal, lambda fid: ShadowHost() if fid.startswith("shadow:") else LiveHost())


@router.post("/jobs/extract")
async def extract_job() -> dict:
    """Cloud Scheduler, every 30 minutes: re-read new conversation turns for facts the brain missed."""
    from app.care.extract import extract_family, families_with_new_turns
    from app.db.session import SessionLocal

    done = {}
    async with SessionLocal() as session:
        families = await families_with_new_turns(session)
    from app.care import versions

    for fid in families:
        try:
            with versions.attribution(actor_id="saheli", source="background", reason="re-read of new messages"):
                async with SessionLocal() as session:
                    done[fid] = await extract_family(session, fid)
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.exception("extract failed family=%s", fid)
            done[fid] = {"error": str(exc)[:200]}
    return {"families": len(families), "results": done}


@router.post("/jobs/consolidate")
async def consolidate_job() -> dict:
    """Nightly: rewrite long memory notes into clean files and expire unanswered confirmations."""
    from sqlalchemy import select as sql_select

    from app.care.extract import consolidate_family, expire_stale_confirmations
    from app.care.models import MemoryNote
    from app.db.session import SessionLocal

    async with SessionLocal() as session:
        fids = [r[0] for r in await session.execute(sql_select(MemoryNote.family_id).distinct())]
        expired = await expire_stale_confirmations(session)
        await session.commit()
    rewritten = 0
    from app.care import versions

    for fid in fids:
        try:
            with versions.attribution(actor_id="saheli", source="nightly", reason="nightly tidy-up of long notes"):
                async with SessionLocal() as session:
                    rewritten += await consolidate_family(session, fid)
                    await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("consolidate failed family=%s", fid)
    return {"families": len(fids), "notes_rewritten": rewritten, "confirmations_expired": expired}


@router.post("/jobs/tasks")
async def tasks_job() -> dict:
    """Cloud Scheduler, every minute: advance running orders and rides."""
    import uuid as _uuid

    from app.brain.tools import task_agent
    from app.brain.wake import system_turn
    from app.db.session import SessionLocal
    from app.tasks.runtime import tick

    def host_for(fid: str):
        return ShadowHost() if fid.startswith("shadow:") else LiveHost()

    async def profile_for(task) -> dict:
        # The family's browser profile for this store, and the number to log in with (the person who asked).
        return await host_for(task.family_id).call(
            "browser_profile", {"partner": task.service},
            family_id=task.family_id.removeprefix("shadow:"), subject_id=task.subject_id, actor_id=task.requested_by,
        )

    async def notify(family_id: str, requested_by: str, prompt: str) -> None:
        try:
            # Name the send explicitly: twice live the brain wrote the update as its final reply (which reaches no one).
            await system_turn(SessionLocal, host_for(family_id), family_id,
                              f"{prompt} (Requested by {requested_by}: tell them with send_message to {requested_by}, in their language, then reply none.)",
                              f"task:{_uuid.uuid4().hex[:12]}", deliver_to=requested_by)
        except Exception:  # noqa: BLE001
            logger.exception("task notify failed family=%s", family_id)

    return await tick(SessionLocal, task_agent(), profile_for=profile_for, notify=notify, host_for=host_for)


@router.post("/jobs/browser-sweep")
async def browser_sweep_job() -> dict:
    """Cloud Scheduler, every 10 min: stop cloud browsers older than 20 min that no live task uses (cost safety)."""
    from app.brain.tools import task_agent
    from app.db.session import SessionLocal
    from app.tasks import sandbox

    return await sandbox.sweep(SessionLocal, task_agent())


@router.get("/agents/metrics")
async def agent_metrics(session: Annotated[AsyncSession, Depends(get_db)], days: int = 7) -> dict:
    """Shopping / pharmacy / rides agents: success rate, time to cart and to placed, cost, channels, failures, alerts."""
    from app.specialists import channels, metrics

    return {**await metrics.rollup(session, days=max(1, min(days, 90))), "channels": await channels.health_table(session)}


@router.get("/llm/spend")
async def llm_spend(days: int = 7) -> dict:
    """Model spend per day, role and model (estimated ₹), with today's total against the caps."""
    from app.db.session import SessionLocal
    from app.llm import spend

    return {"today": round(await spend.spent_today(), 2), "softCap": spend.soft_cap(), "hardCap": spend.hard_cap(),
            "rows": await spend.summary(SessionLocal, days=max(1, min(days, 90)))}


@router.post("/jobs/dream")
async def dream_job(budget_s: float = 240.0) -> dict:
    """Saheli's night for families not done yet tonight (the nightly Cloud Run Job runs the same, without a budget)."""
    from app.care import dream
    from app.db.session import SessionLocal

    return await dream.dream_all(SessionLocal, budget_s=max(10.0, min(budget_s, 280.0)))


@router.post("/jobs/learn-weekly")
async def learn_weekly_job() -> dict:
    """Learn a new playbook draft from the anonymised corpus, gate it, and start its 10% canary."""
    from app.db.session import SessionLocal
    from app.learn import lessons

    return await lessons.weekly(SessionLocal)


@router.get("/learn/overview")
async def learn_overview(weeks: int = 8) -> dict:
    from app.db.session import SessionLocal
    from app.learn import jobs

    return await jobs.overview(SessionLocal, weeks=max(1, min(weeks, 52)))


class PlaybookAction(BaseModel):
    by: str = Field(min_length=1, max_length=64)


@router.post("/learn/playbooks/{version}/approve")
async def learn_approve(version: int, body: PlaybookAction) -> dict:
    from app.db.session import SessionLocal
    from app.learn import lessons

    return await lessons.approve(SessionLocal, version, body.by)


class RuleDecision(BaseModel):
    by: str = Field(min_length=1, max_length=64)
    approve: bool


@router.post("/learn/rules/{rule_id}")
async def learn_rule_decision(rule_id: int, body: RuleDecision) -> dict:
    from app.db.session import SessionLocal
    from app.learn import review

    return await review.decide_rule(SessionLocal, rule_id, approve=body.approve, by=body.by)


@router.post("/learn/playbooks/{version}/block")
async def learn_block(version: int, body: PlaybookAction) -> dict:
    from app.db.session import SessionLocal
    from app.learn import lessons

    return await lessons.block(SessionLocal, version, body.by)


@router.post("/jobs/daily")
async def daily_job() -> dict:
    """Cloud Scheduler, 10:00 IST: medicines running low become a refill loop that wakes Saheli to ask about a reorder."""
    from app.care import features
    from app.core import clock
    from app.db.session import SessionLocal

    opened = 0
    async with SessionLocal() as session:
        from sqlalchemy import select as sql_select

        from app.care.models import OpenLoop

        for fid, sid, row in await features.refill_candidates(session):
            dedupe = f"refill:{sid}:{row['key']}"
            already = (
                await session.execute(
                    sql_select(OpenLoop.id).where(OpenLoop.family_id == fid, OpenLoop.dedupe_key == dedupe, OpenLoop.status == "open")
                )
            ).first()
            if already:
                continue  # asked once already; it stays on the dashboard until someone acts
            await store.open_loop(
                session, family_id=fid, subject_id=sid, kind="refill",
                title=f"Refill {row['name']}: about {row['daysLeft']} days left ({row['stock']} left) for person {sid}",
                detail={"key": row["key"], "days_left": row["daysLeft"], "stock": row["stock"], "max_wakes": 1},
                wake_at=clock.now(), alert_rule="dashboard", dedupe_key=dedupe,
            )
            opened += 1
        await session.commit()
    ended = 0
    try:
        from app.care import records

        def host_for(fid: str):
            return ShadowHost() if fid.startswith("shadow:") else LiveHost()

        async with SessionLocal() as session:
            ended = await records.end_finished_courses(session, host_for, clock.ist_day())
            await session.commit()
    except Exception:  # noqa: BLE001 — refills are already saved; a course check failing must not undo them
        logger.exception("end finished courses failed")
    return {"refills": opened, "courses_ended": ended}  # patterns and baselines are learned in the nightly dream (app/care/dream.py)


async def _plan_checkins() -> dict:
    from sqlalchemy import select as sql_select

    from app.care.models import FamilyRoster
    from app.core import clock
    from app.db.session import SessionLocal
    from app.learn import timing

    planned = 0
    async with SessionLocal() as session:
        rosters = list((await session.execute(sql_select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")))).scalars())
        for r in rosters:
            elder = (r.elder or {}).get("id")
            for m in r.members or []:
                cg = m.get("id")
                if not cg or cg == elder:
                    continue
                at = await timing.next_send_at(session, r.family_id, cg, "checkin", default_hour=18, earliest=clock.now())
                await store.open_loop(
                    session, family_id=r.family_id, subject_id=elder or cg, kind="checkin", title="Weekly caregiver check-in",
                    detail={"key": f"week:{clock.ist_day()}:subj={elder}", "max_wakes": 1}, owner_id=cg, wake_at=at,
                    alert_rule="dashboard", dedupe_key=f"checkin:{cg}:{clock.ist_day()}",
                )
                planned += 1
        await session.commit()
    return {"planned": planned}


async def _roster(SessionLocal, fid: str):
    async with SessionLocal() as session:
        return await store.roster(session, fid)


CHECKIN_PROMPT = (
    "[Caregiver check-in] It is the weekly check-in on the caregivers themselves. For each caregiver in HOUSEHOLD "
    "(not the care recipient), send one short, warm message with send_message asking how they are doing this week "
    "(sleep, stress, their own health). One question only, no lists. When they answer later, log it with log_event "
    "kind mood about their own id. If PATTERNS NOTICED lists something about the person they care for, add the most "
    "important one to that same message in one line, with its suggestion. Also ask, in the same message, whether there was "
    "any fall, hospital visit or medicine change this week, attaching buttons {{\"kind\": \"outcome\", \"key\": \"week:{day}:subj={elder}\"}}. "
    "After sending, reply none."
)


@router.post("/jobs/weekly")
async def weekly_job(plan: bool = False) -> dict:
    """Cloud Scheduler, Sunday: a short check-in with each caregiver about themselves.

    plan=false (Sunday 18:00): send now, one turn per family. plan=true (Sunday 08:00): schedule each caregiver's
    check-in for the hour they have answered best (learned; 18:00 until there is history)."""
    if plan:
        return await _plan_checkins()
    from sqlalchemy import select as sql_select

    from app.brain.wake import system_turn
    from app.care.models import FamilyRoster
    from app.core import clock
    from app.db.session import SessionLocal

    async with SessionLocal() as session:
        fids = [r[0] for r in await session.execute(sql_select(FamilyRoster.family_id).where(~FamilyRoster.family_id.startswith("shadow:")))]
    ran = 0
    for fid in fids:
        try:
            # One check-in per family per day: a scheduler retry is a duplicate turn, not a second message.
            roster = await _roster(SessionLocal, fid)
            prompt = CHECKIN_PROMPT.format(day=clock.ist_day(), elder=(roster.elder or {}).get("id", "") if roster else "")
            await system_turn(SessionLocal, LiveHost(), fid, prompt, f"checkin:{clock.ist_day()}")
            ran += 1
        except Exception:  # noqa: BLE001
            logger.exception("caregiver check-in failed family=%s", fid)
    return {"families": len(fids), "ran": ran}
