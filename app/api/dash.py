"""Caregiver dashboard view of Saheli's care memory and tasks.

The backend authorises the caregiver for the family before calling these; ids are Kavach ids.
Edits go through the same care tools the brain uses, so medicine reminders stay in step.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain import tools
from app.brain.host import LiveHost
from app.care import store
from app.care.domains import DOMAINS
from app.care.models import CareEvent, CareFact, OpenLoop
from app.core import clock
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.tasks import runtime
from app.tasks.models import Task
from app.tasks.skills import SKILLS

router = APIRouter(prefix="/v2/dash", tags=["dashboard"], dependencies=[Depends(verify_api_secret)])
DB = Annotated[AsyncSession, Depends(get_db)]


def make_host():
    return LiveHost()


class Actor(BaseModel):
    id: str
    name: str = ""


def fact_json(f: CareFact) -> dict:
    return {
        "id": str(f.id), "domain": f.domain, "key": f.key, "name": f.key.split(":", 1)[-1], "value": f.value, "text": f.text,
        "status": f.status, "source": f.source_kind, "statedBy": f.stated_by, "confirmedBy": f.confirmed_by,
        "confidence": f.confidence, "validFrom": f.valid_from.isoformat(), "validTo": f.valid_to.isoformat() if f.valid_to else None,
        "recordedAt": f.recorded_at.isoformat(), "note": f.note,
    }


def event_json(e: CareEvent) -> dict:
    return {"id": e.id, "kind": e.kind, "at": e.at.isoformat(), "day": e.day, "summary": e.summary, "actorId": e.actor_id, "payload": e.payload}


def loop_json(o: OpenLoop) -> dict:
    return {
        "id": str(o.id), "kind": o.kind, "title": o.title, "status": o.status, "detail": o.detail, "ownerId": o.owner_id,
        "wakeAt": o.wake_at.isoformat() if o.wake_at else None, "rule": o.alert_rule, "createdAt": o.created_at.isoformat(),
    }


def task_json(t: Task) -> dict:
    d = dict(t.details or {})
    d.pop("otp", None)
    return {
        "id": str(t.id), "service": t.service, "serviceLabel": SKILLS[t.service]["label"], "kind": t.kind, "goal": t.goal,
        "details": d, "status": t.status, "phase": t.phase, "inputNeeded": t.input_needed, "cancelRequested": t.cancel_requested,
        "result": t.result, "history": t.history, "requestedBy": t.requested_by, "hasLiveView": bool(t.agent_session),
        "createdAt": t.created_at.isoformat(), "updatedAt": t.updated_at.isoformat(),
    }


@router.get("/{family_id}/{elder_id}/overview")
async def overview(family_id: str, elder_id: str, session: DB, day: str | None = None) -> dict:
    day = day or clock.ist_day()
    from app.care import importer

    if not await importer.already_imported(session, family_id, elder_id):
        try:
            await importer.import_family(session, make_host(), family_id=family_id, backend_family_id=family_id, elder_id=elder_id)
            await session.commit()
        except Exception:  # noqa: BLE001 — show what there is
            await session.rollback()
    facts = await store.facts(session, family_id, elder_id)
    loops = await store.live_loops(session, family_id, [elder_id, "family"])
    events = await store.events(session, family_id, elder_id, day=day, limit=500)
    tasks = list(
        (await session.execute(select(Task).where(Task.family_id == family_id).order_by(Task.created_at.desc()).limit(30))).scalars()
    )
    notes = await store.notes(session, family_id, [elder_id, "family"])
    return {
        "day": day,
        "domains": DOMAINS,
        "facts": [fact_json(f) for f in facts],
        "pending": [fact_json(f) for f in facts if f.status == "pending"],
        "loops": [loop_json(o) for o in loops],
        "events": [event_json(e) for e in events],
        "tasks": [task_json(t) for t in tasks],
        "notes": [{"subjectId": n.subject_id, "slug": n.slug, "title": n.title, "body": n.body_md, "version": n.version, "updatedAt": n.updated_at.isoformat()} for n in notes],
    }


@router.get("/{family_id}/{elder_id}/history")
async def history(family_id: str, elder_id: str, key: str, session: DB) -> dict:
    return {"key": key, "versions": [fact_json(f) for f in await store.fact_history(session, family_id, elder_id, key)]}


@router.get("/{family_id}/{elder_id}/events")
async def events(family_id: str, elder_id: str, session: DB, day: str | None = None, kinds: str | None = None, limit: int = 300) -> dict:
    rows = await store.events(session, family_id, elder_id, day=day, kinds=kinds.split(",") if kinds else None, limit=min(limit, 1000))
    return {"events": [event_json(e) for e in rows]}


def _ctx(session: AsyncSession, family_id: str, elder_id: str, actor: Actor) -> tools.TurnCtx:
    return tools.TurnCtx(
        session=session, host=make_host(), family_id=family_id, elder={"id": elder_id, "name": ""},
        speaker={"id": actor.id, "name": actor.name, "role": "caregiver"}, members=[{"id": elder_id}, {"id": actor.id}],
    )


async def _run(session: AsyncSession, ctx: tools.TurnCtx, name: str, args: dict) -> dict:
    import json

    out, is_error = await tools.run(ctx, name, args)
    if is_error:
        await session.rollback()
        raise HTTPException(status_code=400, detail=json.loads(out).get("refused") or json.loads(out).get("detail") or "failed")
    await store.record_event(
        session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="dashboard_edit", summary=f"{name}: {args.get('sentence') or args.get('name') or ''}",
        actor_id=ctx.actor_id,
    )
    await session.commit()
    return json.loads(out)


class FactIn(BaseModel):
    actor: Actor
    domain: str
    name: str = Field(min_length=1, max_length=120)
    details: dict = Field(default_factory=dict)
    sentence: str = Field(min_length=1, max_length=500)


@router.post("/{family_id}/{elder_id}/facts")
async def save_fact(family_id: str, elder_id: str, body: FactIn, session: DB) -> dict:
    if body.domain not in DOMAINS:
        raise HTTPException(status_code=400, detail="unknown domain")
    ctx = _ctx(session, family_id, elder_id, body.actor)
    return await _run(session, ctx, "remember", {"domain": body.domain, "name": body.name, "details": body.details, "sentence": body.sentence})


class StopIn(BaseModel):
    actor: Actor
    domain: str
    name: str
    reason: str = "removed by caregiver"


@router.post("/{family_id}/{elder_id}/facts/stop")
async def stop_fact(family_id: str, elder_id: str, body: StopIn, session: DB) -> dict:
    ctx = _ctx(session, family_id, elder_id, body.actor)
    return await _run(session, ctx, "stop", {"domain": body.domain, "name": body.name, "reason": body.reason})


class ResolveIn(BaseModel):
    actor: Actor
    key: str
    approve: bool


@router.post("/{family_id}/{elder_id}/facts/resolve")
async def resolve(family_id: str, elder_id: str, body: ResolveIn, session: DB) -> dict:
    ctx = _ctx(session, family_id, elder_id, body.actor)
    return await _run(session, ctx, "confirm_change", {"key": body.key, "approve": body.approve})


class CloseIn(BaseModel):
    actor: Actor
    note: str = "closed from dashboard"


@router.post("/{family_id}/{elder_id}/loops/{loop_id}/close")
async def close_loop(family_id: str, elder_id: str, loop_id: uuid.UUID, body: CloseIn, session: DB) -> dict:
    loop = await session.get(OpenLoop, loop_id)
    if not loop or loop.family_id != family_id:
        raise HTTPException(status_code=404, detail="no such loop")
    await store.close_loop(session, loop_id, note=f"{body.note} ({body.actor.name or body.actor.id})")
    await session.commit()
    return {"closed": True}


class NoteIn(BaseModel):
    actor: Actor
    subject_id: str
    slug: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(max_length=20000)


@router.put("/{family_id}/{elder_id}/notes")
async def save_note(family_id: str, elder_id: str, body: NoteIn, session: DB) -> dict:
    subject = body.subject_id if body.subject_id in (elder_id, "family") else elder_id
    from app.care.redact import scrub_secrets

    n = await store.upsert_note(session, family_id=family_id, subject_id=subject, slug=body.slug, title=body.title, body_md=scrub_secrets(body.body))
    await store.record_event(session, family_id=family_id, subject_id=elder_id, kind="dashboard_edit", summary=f"note: {body.title}", actor_id=body.actor.id)
    await session.commit()
    return {"saved": True, "version": n.version}


async def _task(session: AsyncSession, family_id: str, task_id: uuid.UUID) -> Task:
    t = await session.get(Task, task_id, with_for_update=True)
    if not t or t.family_id != family_id:
        raise HTTPException(status_code=404, detail="no such task")
    return t


class TaskInputIn(BaseModel):
    actor: Actor
    kind: str = Field(pattern="^(otp|confirm|fee|choice)$")
    value: str = Field(min_length=1, max_length=200)


@router.post("/{family_id}/{elder_id}/tasks/{task_id}/input")
async def task_input(family_id: str, elder_id: str, task_id: uuid.UUID, body: TaskInputIn, session: DB) -> dict:
    t = await _task(session, family_id, task_id)
    result = await runtime.provide_input(session, t, kind=body.kind, value=body.value, by=body.actor.id, by_is_elder=False)
    await session.commit()
    return {"result": result, "task": task_json(t)}


class TaskCancelIn(BaseModel):
    actor: Actor
    reason: str = "cancelled from dashboard"


@router.post("/{family_id}/{elder_id}/tasks/{task_id}/cancel")
async def task_cancel(family_id: str, elder_id: str, task_id: uuid.UUID, body: TaskCancelIn, session: DB) -> dict:
    t = await _task(session, family_id, task_id)
    result = await runtime.request_cancel(session, tools.task_agent(), t, by=body.actor.id, reason=body.reason)
    await session.commit()
    return {"result": result, "task": task_json(t)}


@router.get("/{family_id}/{elder_id}/tasks/{task_id}/live")
async def task_live(family_id: str, elder_id: str, task_id: uuid.UUID, session: DB) -> dict:
    t = await _task(session, family_id, task_id)
    if not t.agent_session:
        return {"url": None}
    try:
        return {"url": await tools.task_agent().live_url(t.agent_session)}
    except Exception:  # noqa: BLE001 — the session may already be closed
        return {"url": None}
