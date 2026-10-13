"""Saheli Brain v2 endpoint. The backend calls it with the raw message and who is speaking."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain.host import LiveHost, ShadowHost
from app.brain.loop import TurnRequest, result_json, run_turn
from app.care import store
from app.core import clock
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.llm.router import AllModelsFailed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2", tags=["brain"], dependencies=[Depends(verify_api_secret)])


class Person(BaseModel):
    id: str
    name: str = ""
    role: str = ""
    relation: str = Field(default="", max_length=40)  # who they are to relation_to (the member who added them): "mother", "spouse"
    relation_to: str = Field(default="", max_length=80)


class Image(BaseModel):
    mime: str
    data: str


class ReminderSoon(BaseModel):
    id: str = Field(max_length=80)
    person: str = Field(max_length=80)
    text: str = Field(default="", max_length=400)
    day: str = Field(max_length=10)
    time: str = Field(max_length=5)
    repeat: str = Field(default="daily", max_length=10)
    item: str | None = Field(default=None, max_length=200)


class Location(BaseModel):
    """A WhatsApp location they shared (the backend passes it as data, beside the text)."""

    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    name: str = Field(default="", max_length=200)
    address: str = Field(default="", max_length=400)


class Waiting(BaseModel):
    purpose: str = Field(default="message", max_length=24)
    text: str = Field(max_length=4000)
    at: str = Field(default="", max_length=40)


class TurnIn(BaseModel):
    family_id: str
    elder: Person
    speaker: Person
    members: list[Person] = Field(default_factory=list)
    text: str = ""
    message_ref: str | None = None
    images: list[Image] = Field(default_factory=list)  # any media the model can see: image, video, audio, PDF (base64)
    media_note: str = Field(default="", max_length=300)  # media they sent that could not be opened (too large, unknown type)
    channel: str = "whatsapp"
    modality: str = Field(default="text", pattern="^(text|voice)$")
    voice_confidence: float | None = Field(default=None, ge=0, le=1)
    voice_language: str | None = Field(default=None, max_length=16)
    reminders_soon: list[ReminderSoon] = Field(default_factory=list, max_length=24)  # due in the next hours (backend)
    onboarding: bool = False  # the speaker has not finished setting up (a new WhatsApp sign-up, or never finished)
    location: Location | None = None  # a WhatsApp location pin in this message
    # messages that waited while the speaker's WhatsApp chat was closed (backend; they got a template notice)
    waiting: list[Waiting] = Field(default_factory=list, max_length=8)
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
        media_note=body.media_note,
        channel=body.channel,
        modality=body.modality,
        voice_confidence=body.voice_confidence,
        voice_language=body.voice_language,
        reminders_soon=[r.model_dump() for r in body.reminders_soon],
        location=body.location.model_dump() if body.location else None,
        onboarding=body.onboarding,
        waiting=[w.model_dump() for w in body.waiting],
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
    from app.care import doses

    ids = []
    # A dose marked on the dashboard or with the Done button replaces any earlier answer about that dose (any recent day).
    medicine = body.payload.get("medicine") if body.kind in doses.KINDS else None
    day = doses.resolve_day(body.payload.get("dateKey")) if medicine else None
    # Shadow memory mirrors the live ledger, so the shadow brain sees the same day.
    for fid in (body.family_id, "shadow:" + body.family_id):
        if medicine and day:
            ids.append(await doses.record(
                session, family_id=fid, subject_id=body.subject_id, kind=body.kind, medicine=str(medicine), summary=body.summary,
                day=day, time=body.payload.get("time"), ref=body.ref, payload=body.payload,
            ))
            continue
        ids.append(
            await store.record_event(
                session, family_id=fid, subject_id=body.subject_id, kind=body.kind, summary=body.summary,
                payload=body.payload, ref=body.ref, at=body.at,
            )
        )
        if body.kind == "reminder_sent" and ids[-1] is not None:
            # The reminder is a message in their chat too (Kavach NUdge's plan §3.8): Saheli sees it when they answer it, and
            # the regular nudges' silence clock counts it. Not a brain nudge: the old check-in caps leave it out (via).
            line = str(body.payload.get("preview") or body.summary or "").strip()
            if line:
                await store.add_turn(session, family_id=fid, thread_id=body.subject_id, role="assistant", text=line[:600],
                                     message_ref=f"reminder:{body.ref}" if body.ref else None,
                                     meta={"proactive": True, "via": "reminder", "ref": body.ref, "reminder_id": body.payload.get("reminderId"),
                                           "slot": body.payload.get("slot")})
    await session.commit()
    return {"recorded": ids[0] is not None}


class SpeechIn(BaseModel):
    text: str = Field(max_length=3000)
    language: str = Field(default="Hindi", max_length=80)


class ComposeIn(BaseModel):
    family_id: str
    subject_id: str
    purpose: str = Field(max_length=600)
    at: str | None = None
    recent: list[str] = Field(default_factory=list, max_length=12)
    kind: str = "reminder"
    item: str | None = None


@router.post("/brain/compose-reminder")
async def compose_reminder(body: ComposeIn, session: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    """The backend's reminder scheduler asks for the line just before it sends: written for this moment, never a repeat
    of the earlier lines (founder 2026-10-13). {"text": null} means send the stored text."""
    from app.care import compose

    return await compose.reminder_text(session, family_id=body.family_id, subject_id=body.subject_id, purpose=body.purpose,
                                       at=body.at, recent=body.recent, kind=body.kind, item=body.item)


@router.post("/voice/prepare")
async def voice_prepare(body: SpeechIn) -> dict:
    """The backend's TTS asks how Saheli would say a message in a voice note (spoken script + mood)."""
    from app.care import speech

    return await speech.prepare(body.text, body.language)


_nudge_minute: dict = {"last": None}


async def run_nudges(host_for) -> dict:
    """Regular nudges, then reminder follow-ups; once a minute at most (the wake job and the per-minute job may overlap)."""
    from app.care import nudges
    from app.db.session import SessionLocal

    minute = clock.now().replace(second=0, microsecond=0)
    if _nudge_minute["last"] == minute:
        return {"skipped": "ran this minute"}
    _nudge_minute["last"] = minute
    return {"regular": await nudges.run(SessionLocal, host_for), "followups": await nudges.reminder_followups(SessionLocal, host_for)}


@router.post("/jobs/nudge")
async def nudge_job() -> dict:
    """Cloud Scheduler, every minute (D5): regular nudges and 5-minute reminder follow-ups. The admin flag nudge.regular
    decides off | shadow | on."""

    def host_for(fid: str):
        return ShadowHost() if fid.startswith("shadow:") else LiveHost()

    return await run_nudges(host_for)


@router.post("/jobs/wake")
async def wake_job() -> dict:
    """Cloud Scheduler, every 5 minutes: run the open loops that are due."""
    from app.brain.wake import wake_due
    from app.care import checkins
    from app.db.session import SessionLocal

    def host_for(fid: str):
        return ShadowHost() if fid.startswith("shadow:") else LiveHost()

    try:
        from app.care import activities

        # every activity's next check-ins are planned (a day ahead), before the due loops run
        out_plan = await activities.plan_all(SessionLocal)
    except Exception:  # noqa: BLE001
        logger.exception("activity planning failed")
        out_plan = None
    out = await wake_due(SessionLocal, host_for)
    out["activities_planned"] = out_plan
    try:
        from app.care import careteam

        # alerts to the Kavach care team that waited a few minutes (or a send that failed) go as one update
        out["care_team"] = await careteam.flush(SessionLocal, host_for)
    except Exception:  # noqa: BLE001
        logger.exception("care team flush failed")
    try:
        # Regular nudges and reminder follow-ups (Kavach NUdge's plan), here until the per-minute job is approved (D5)
        out["nudges"] = await run_nudges(host_for)
    except Exception:  # noqa: BLE001
        logger.exception("regular nudges failed")
    try:
        # Saheli's own check-ins (today's unmarked schedule after a chat, how they feel, readings, reports, weight); they
        # step aside while regular nudges are on
        from app.care import nudges

        if nudges.cached_mode() != "on":
            out["checkins"] = await checkins.run(SessionLocal, host_for)
    except Exception:  # noqa: BLE001 — loops above already ran
        logger.exception("check-ins failed")
    try:
        from app.care import brief

        out["briefs"] = await brief.run(SessionLocal, host_for)  # each caregiver's nightly brief, at their time
    except Exception:  # noqa: BLE001
        logger.exception("daily briefs failed")
    return out


EXTRACT_BATCHES = 6  # per family per run: up to 6 x 120 new turns


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
                # a long day's messages are read in a few batches, so no family falls behind (the commit ran after the
                # session had closed from 4 to 13 Oct 2026, so only the nightly re-read of one batch was ever kept)
                for _ in range(EXTRACT_BATCHES):
                    async with SessionLocal() as session:
                        done[fid] = await extract_family(session, fid)
                        await session.commit()
                    if not done[fid].get("more"):
                        break
        except Exception as exc:  # noqa: BLE001
            logger.exception("extract failed family=%s", fid)
            done[fid] = {"error": str(exc)[:200]}
    return {"families": len(families), "results": done}


@router.post("/jobs/adopt-activities")
async def adopt_activities_job() -> dict:
    """Once after the activities release (safe to run again): routine facts with times become activities with check-ins."""
    from sqlalchemy import select as sql_select

    from app.care import activities
    from app.care.models import CareFact
    from app.db.session import SessionLocal

    async with SessionLocal() as session:
        people = sorted({(r[0], r[1]) for r in await session.execute(
            sql_select(CareFact.family_id, CareFact.subject_id).where(CareFact.domain == "routine", CareFact.status == "active",
                                                                     ~CareFact.family_id.startswith("shadow:"), ~CareFact.family_id.startswith("sim-")))})
    done: dict[str, list[str]] = {}
    for fid, sid in people:
        try:
            async with SessionLocal() as session:
                made = await activities.adopt(session, LiveHost(), fid, sid)
                await session.commit()
            done[f"{fid[:8]}/{sid[:8]}"] = made
        except Exception as exc:  # noqa: BLE001
            logger.exception("adopt activities failed family=%s", fid)
            done[f"{fid[:8]}/{sid[:8]}"] = [f"error: {str(exc)[:120]}"]
    return {"people": len(people), "adopted": done}


@router.post("/jobs/recover-readings")
async def recover_readings_job(apply: bool = False) -> dict:
    """Once (safe to run again): readings kept only in facts or notes go to Health Records. apply=false lists them."""
    from sqlalchemy import select as sql_select
    from sqlalchemy import union

    from app.care import readings
    from app.care.models import CareFact, MemoryNote
    from app.db.session import SessionLocal

    real = lambda col: ~col.startswith("shadow:") & ~col.startswith("sim-")  # noqa: E731
    async with SessionLocal() as session:
        people = sorted({(r[0], r[1]) for r in await session.execute(union(
            sql_select(CareFact.family_id, CareFact.subject_id).where(CareFact.status == "active", real(CareFact.family_id)),
            sql_select(MemoryNote.family_id, MemoryNote.subject_id).where(real(MemoryNote.family_id))))})
    found: dict[str, list[dict]] = {}
    for fid, sid in people:
        if not sid or sid == "family":
            continue
        try:
            async with SessionLocal() as session:
                got = await readings.recover(session, fid, sid, apply=apply)
                await session.commit()
            if got:
                found[f"{fid[:8]}/{sid[:8]}"] = got
        except Exception as exc:  # noqa: BLE001
            logger.exception("recover readings failed family=%s", fid)
            found[f"{fid[:8]}/{sid[:8]}"] = [{"error": str(exc)[:120]}]
    return {"people": len(people), "applied": apply, "readings": found}


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


_kicked: asyncio.Task | None = None


def kick_tasks() -> None:
    """Advance orders right away in the background (a turn just started one or gave it an answer). One at a time: a run
    already going keeps looking every few seconds and picks the new work up."""
    global _kicked
    if _kicked is not None and not _kicked.done():
        return
    import contextvars

    from app.core import trace

    # every family's tasks move in this run: none of it belongs to the turn that kicked it (each task files its own trace)
    ctx = contextvars.copy_context()
    ctx.run(trace.detach)
    _kicked = asyncio.get_running_loop().create_task(tasks_job(), context=ctx)


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
        # The family's browser profile for this store, and the number to log in with: the person the family set to give
        # login codes for this person's orders (else whoever asked), so the code reaches the phone of someone who can answer.
        return await host_for(task.family_id).call(
            "browser_profile", {"partner": task.service},
            family_id=task.family_id.removeprefix("shadow:"), subject_id=task.subject_id,
            actor_id=(task.details or {}).get("code_from") or task.requested_by,
        )

    async def notify(family_id: str, requested_by: str, prompt: str) -> None:
        from app.tasks import runtime as _runtime

        try:
            # Name the send explicitly: twice live the brain wrote the update as its final reply (which reaches no one).
            await system_turn(SessionLocal, host_for(family_id), family_id,
                              f"{prompt} (For person {requested_by}, who asked: send it to them now with send_message (to {requested_by}), in their language. After that send, reply none.)",
                              f"task:{_uuid.uuid4().hex[:12]}", deliver_to=requested_by, still_true=_runtime.delivery_check())
        except Exception:  # noqa: BLE001
            logger.exception("task notify failed family=%s", family_id)

    import time

    from sqlalchemy import func, select

    from app.tasks.models import Task

    async def moving() -> bool:
        async with SessionLocal() as s:
            return bool(await s.scalar(select(func.count()).select_from(Task).where(Task.status.in_(("queued", "running")))))

    # Cloud Scheduler calls once a minute. While an order is moving, look again every few seconds, so each step (code
    # asked, cart ready, placed) reaches the person within seconds instead of up to a minute later (live 2026-10-09: the
    # order was placed at 16:33:35 and Saheli said so only after the next minute's tick). Live 2026-10-11: a call that ran
    # 58 s was followed by no call the next minute, and a login code the store sent at 15:45:29 was asked for at 15:46:02.
    # So a call ends well inside its minute, the looks in between are light (no housekeeping), and a call that ends with
    # orders still moving hands over to the background loop until the next minute's call.
    started, agent = time.monotonic(), task_agent()
    background = asyncio.current_task() is _kicked
    stats = await tick(SessionLocal, agent, profile_for=profile_for, notify=notify, host_for=host_for, light=background)
    while await moving() and time.monotonic() - started + TASKS_FAST_EVERY < TASKS_FAST_WINDOW:
        await asyncio.sleep(TASKS_FAST_EVERY)
        more = await tick(SessionLocal, agent, profile_for=profile_for, notify=notify, host_for=host_for, light=True)
        stats = {k: stats.get(k, 0) + more.get(k, 0) for k in {*stats, *more}}
    if not background and await moving():
        kick_tasks()
    return stats


TASKS_FAST_EVERY, TASKS_FAST_WINDOW = 3, 40  # seconds


@router.post("/jobs/browser-sweep")
async def browser_sweep_job() -> dict:
    """Cloud Scheduler, every 10 min: stop cloud browsers older than 20 min that no live task uses (cost safety)."""
    from app.brain.tools import task_agent
    from app.db.session import SessionLocal
    from app.tasks import credits, sandbox

    out = await sandbox.sweep(SessionLocal, task_agent())
    try:
        out["credits"] = await credits.check(SessionLocal, task_agent(), LiveHost())
    except Exception:  # noqa: BLE001 — the sweep's own work is done
        logger.exception("credit check failed")
    return out


@router.get("/agents/care-search")
async def agent_care_search(kind: str = "labs", q: str = "HbA1c", lat: float | None = None, lon: float | None = None,
                            pincode: str | None = None, city: str | None = None) -> dict:
    """Ops check of the lab / doctor look-ups from Cloud Run (guest browser, nothing booked). Without a point it looks at
    the daily store check's city of the day (a public city centre, never a family's address)."""
    import time as _time

    from app.brain.tools import task_agent
    from app.tasks import care_search, store_health

    if lat is None or lon is None:
        ref = store_health.place_for_today()
        lat, lon, pincode, city = ref["lat"], ref["lng"], pincode or ref.get("pincode"), city or ref["city"]
    agent, started = task_agent(), _time.monotonic()
    sid = await agent.open_session(None)
    try:
        cdp = await agent.cdp_url(sid)
        rows = await (care_search.labs(cdp, q, lat=lat, lon=lon, pincode=pincode) if kind == "labs"
                      else care_search.doctors(cdp, q, city=city or "", lat=lat, lon=lon))
    finally:
        await agent.stop_session(sid)
    return {"seconds": round(_time.monotonic() - started, 1), "rows": rows}


@router.get("/agents/credits")
async def agent_credits() -> dict:
    """USD left on the browser service and the alert threshold (admin console)."""
    from app.brain.tools import task_agent
    from app.tasks import credits

    return {"usd": await credits.balance(task_agent()), "warnAt": credits.warn_at()}


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


@router.get("/learn/corrections")
async def learn_corrections(session: Annotated[AsyncSession, Depends(get_db)], days: int = 30, format: str = "json"):
    """Every correction (chat, dashboard edit of a saved fact, reply-guard rewrite, thumbs down), anonymised; jsonl = eval fixtures."""
    from fastapi.responses import PlainTextResponse

    from app.learn import corrections

    rows = await corrections.cases(session, days=max(1, min(days, 180)))
    if format == "jsonl":
        return PlainTextResponse("\n".join(corrections.as_fixture(c) for c in rows) + ("\n" if rows else ""), media_type="application/x-ndjson")
    return {"cases": rows, "count": len(rows)}


@router.get("/learn/tuning-readiness")
async def learn_tuning_readiness(session: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    """Is there enough good, consented, anonymised data to fine-tune a fast model yet, and what is missing."""
    from app.learn import tuning_readiness

    return await tuning_readiness.readiness(session)


@router.get("/learn/flywheel")
async def learn_flywheel(session: Annotated[AsyncSession, Depends(get_db)], days: int = 14) -> dict:
    """Completion, time, reopen rate and cost per task; adherence; reply and check-in answer rates; approvals; delegation."""
    from app.learn import flywheel

    return await flywheel.compute(session, days=max(1, min(days, 90)))


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
    stores: dict = {}
    try:
        # One guest look-up per store: a changed site is found before a family's order hits it (alert to the founder).
        from app.brain.tools import task_agent
        from app.tasks import store_health

        async def alert(subject: str, text: str) -> None:
            from app.tasks.credits import OPS

            await LiveHost().call("ops_alert", {"subject": subject, "text": text}, family_id=OPS, subject_id=OPS, actor_id=OPS)

        stores = await store_health.check(task_agent(), alert)
    except Exception:  # noqa: BLE001
        logger.exception("store health check failed")
    return {"refills": opened, "courses_ended": ended, "stores": stores}  # patterns and baselines are learned in the nightly dream


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
