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
from app.tasks import runtime, sandbox
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
        "audit": sandbox.audit_text(t),
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


def _ctx(session: AsyncSession, family_id: str, elder_id: str, actor: Actor, *, confirmed: bool = False) -> tools.TurnCtx:
    return tools.TurnCtx(
        session=session, host=make_host(), family_id=family_id, elder={"id": elder_id, "name": ""},
        speaker={"id": actor.id, "name": actor.name, "role": "caregiver"}, members=[{"id": elder_id}, {"id": actor.id}],
        channel="dashboard", confirmed=confirmed,
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


class RecordRememberIn(BaseModel):
    actor: Actor
    document_id: str = Field(min_length=3, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    date: str | None = Field(default=None, max_length=10)
    points: list[str] = Field(default_factory=list, max_length=8)


@router.post("/{family_id}/{elder_id}/records/remember")
async def record_remember(family_id: str, elder_id: str, body: RecordRememberIn, session: DB) -> dict:
    """A health record the person chose to have Saheli remember (after reviewing what was read)."""
    from app.care import records

    out = await records.remember(session, family_id, elder_id, document_id=body.document_id, title=body.title, when=body.date,
                                 points=body.points, actor_id=body.actor.id)
    await session.commit()
    return out


class RecordForgetIn(BaseModel):
    actor: Actor
    document_id: str = Field(min_length=3, max_length=64)
    memory_document_id: str | None = Field(default=None, max_length=64)


@router.post("/{family_id}/{elder_id}/records/forget")
async def record_forget(family_id: str, elder_id: str, body: RecordForgetIn, session: DB) -> dict:
    """A deleted record (or someone else's): Saheli forgets everything she kept from it."""
    from app.care import records

    out = await records.forget(session, family_id, elder_id, document_id=body.document_id, memory_document_id=body.memory_document_id,
                               actor_id=body.actor.id)
    await session.commit()
    return out


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

    n = await store.upsert_note(session, family_id=family_id, subject_id=subject, slug=body.slug, title=body.title, body_md=scrub_secrets(body.body),
                                actor_id=body.actor.id, source="dashboard")
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
    kind: str = Field(pattern="^(go|otp|confirm|fee|choice)$")
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


# ── home summary ───────────────────────────────────────────────────────────────

DOSE_KINDS = {"dose_taken": "taken", "dose_skipped": "skipped", "dose_refused": "skipped", "dose_missed": "missed", "dose_empty_strip": "taken"}


def _mentions(text: str, name: str) -> bool:
    from app.care.domains import slug as _slug

    s = _slug(name).replace("_", " ").split(" ")[0]
    return bool(s) and s in (text or "").lower()


def _hm(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


@router.get("/{family_id}/{elder_id}/home")
async def home(family_id: str, elder_id: str, session: DB) -> dict:
    from datetime import timedelta

    from app.care.models import Turn

    now = clock.ist()
    today = clock.ist_day()
    from app.care import doses as dose_days

    meds = [f for f in await store.facts(session, family_id, elder_id, domains=["medicine"], statuses=("active",))]

    def doses_on(day: str) -> list[dict]:
        """The day's doses: a weekly medicine only on its day (live 2026-10-09: weekly Vitamin D3 showed every day)."""
        return sorted(
            ({"name": f.value.get("name") or f.key.split(":", 1)[1].replace("_", " ").title(), "dose": f.value.get("dose"), "time": t, "key": f.key}
             for f in meds if dose_days.due_on(f.value, day) for t in (f.value.get("times") or [])),
            key=lambda d: d["time"],
        )

    schedule = doses_on(today)
    since = clock.now() - timedelta(days=14)
    evs = await store.events(session, family_id, elder_id, since=since, limit=5000)
    by_day: dict[str, list] = {}
    for e in evs:
        by_day.setdefault(e.day, []).append(e)

    # today's doses
    todays = by_day.get(today, [])
    doses = []
    used: set[int] = set()
    mins_now = now.hour * 60 + now.minute
    for i, d in enumerate(schedule):
        status = "upcoming"
        for e in todays:
            if e.id in used or e.kind not in DOSE_KINDS:
                continue
            if _mentions(e.summary, d["name"]) or _mentions(str(e.payload.get("medicine", "")), d["name"]):
                status = DOSE_KINDS[e.kind]
                used.add(e.id)
                break
        if status == "upcoming":
            reminded = any(e.kind == "reminder_sent" and _mentions(e.summary, d["name"]) and d["time"] in e.summary for e in todays)
            delta = mins_now - _hm(d["time"])
            if delta >= 0 and delta <= 90:
                status = "due"
            elif delta > 90:
                status = "reminded" if reminded else "unmarked"
        doses.append({"id": f"{d['key']}@{d['time']}", "time": d["time"], "name": d["name"], "dose": d.get("dose"), "status": status})

    # 14-day adherence: of the doses due that day, how many were taken (a dose of a medicine not due does not count)
    def day_counts(day: str) -> tuple[int, int]:
        due = doses_on(day)
        taken_evs = [e for e in by_day.get(day, []) if e.kind in ("dose_taken", "dose_empty_strip")]
        taken = 0
        for name in {d["name"] for d in due}:
            slots = sum(1 for d in due if d["name"] == name)
            hits = sum(1 for e in taken_evs if _mentions(e.summary, name) or _mentions(str(e.payload.get("medicine", "")), name))
            taken += min(slots, hits)
        return taken, len(due)

    days, adherence, week_taken, week_due = [], [], 0, 0
    for back in range(13, -1, -1):
        day = clock.ist_day(clock.now() - timedelta(days=back))
        taken, due = day_counts(day)
        days.append(day)
        adherence.append(round(100 * taken / due) if due else 0)
        if back < 7:
            week_taken, week_due = week_taken + taken, week_due + due
    streak = 0
    for pct, day in zip(reversed(adherence[:-1]), reversed(days[:-1])):
        if pct < 100 or not doses_on(day):
            break
        streak += 1

    # vitals
    vitals: dict[str, list] = {}
    for e in evs:
        if e.kind == "vital" and e.payload.get("kind"):
            vitals.setdefault(e.payload["kind"], []).append(e)

    def vital(kind: str) -> dict | None:
        rows = vitals.get(kind) or []
        if not rows:
            return None
        last = rows[-1]
        import re as _re

        nums = [float(_re.findall(r"\d+(?:\.\d+)?", r.payload.get("value", "0"))[0]) for r in rows if _re.findall(r"\d+(?:\.\d+)?", r.payload.get("value", ""))]
        change = None
        if len(nums) >= 2 and nums[-2]:
            pct = round(100 * (nums[-1] - nums[-2]) / nums[-2])
            change = {"pct": abs(pct), "dir": "up" if pct > 0 else "down"} if pct else None
        return {
            "value": last.payload.get("value"), "unit": last.payload.get("unit"), "at": last.at.isoformat(),
            "trend": nums[-7:], "change": change, "redFlag": last.payload.get("red_flag"),
        }

    loops = await store.live_loops(session, family_id, [elder_id, "family"])
    tasks = list((await session.execute(select(Task).where(Task.family_id == family_id).order_by(Task.created_at.desc()).limit(8))).scalars())
    pending = [f for f in await store.facts(session, family_id, elder_id, statuses=("pending",))]
    needs = [{"id": f"fact:{f.key}", "kind": "fact", "key": f.key, "title": f.text, "meta": f.note or "Reported in chat; waits for your OK"} for f in pending]
    needs += [
        {"id": f"task:{t.id}", "kind": "task", "taskId": str(t.id), "input": t.input_needed, "title": f"{SKILLS[t.service]['label']}: {t.goal}",
         "meta": f"Total {t.result.get('total')}" if t.result.get("total") else (t.input_needed or "")}
        for t in tasks if t.status in ("needs_input", "awaiting_confirm")
    ]
    from app.care import features

    from app.care import work as care_work

    for r in await care_work.items(session, family_id):
        if r.get("stuck") and r["source"] == "loop":
            needs.append({"id": r["id"], "kind": "stuck", "title": r["title"], "meta": f"Stuck ({r['stuck']}): {r['next_action']}"})
    for r in await features.stock(session, family_id, elder_id):
        if r["low"]:
            needs.append({"id": f"refill:{r['key']}", "kind": "refill", "key": r["key"], "title": f"{r['name']}: {r['daysLeft']} days left",
                          "meta": f"{r['stock']} left. Reorder from Apollo, 1mg or PharmEasy, or set a new count."})
    appts = [features.appointment_json(f) for f in await store.facts(session, family_id, elder_id, domains=["appointment"], statuses=("active",))]
    upcoming = sorted((a for a in appts if a["upcoming"] and a["when"]), key=lambda a: a["when"])
    next_appt = upcoming[0] if upcoming else None
    if next_appt and next_appt["when"][:10] <= clock.ist_day(clock.now() + timedelta(days=2)):
        needs.append({"id": f"appointment:{next_appt['key']}", "kind": "appointment", "key": next_appt["key"],
                      "title": f"{next_appt['doctor']}, {next_appt['when'][8:10]}/{next_appt['when'][5:7]} at {next_appt['when'][11:16]}",
                      "meta": " · ".join(x for x in (next_appt.get("place"), f"{len(next_appt['questions'])} questions to ask" if next_appt["questions"] else None) if x)})
    last_turn = (
        await session.execute(
            select(Turn).where(Turn.family_id == family_id, Turn.thread_id == elder_id, Turn.role == "user").order_by(Turn.id.desc()).limit(1)
        )
    ).scalar_one_or_none()
    hidden = {"import_done", "memory_extract", "note_rewritten", "probe", "dashboard_edit", "dose_corrected"}
    timeline = [{"id": e.id, "at": e.at.isoformat(), "kind": e.kind, "text": e.summary} for e in reversed(todays) if e.kind not in hidden][:20]
    return {
        "now": now.isoformat(),
        "doses": doses,
        "week": {"taken": week_taken, "scheduled": week_due, "streakDays": streak, "adherence": adherence, "days": days},
        "vitals": {k: vital(k) for k in ("bp", "sugar", "weight", "temperature", "spo2")},
        "needsYou": needs,
        "followUps": [loop_json(o) for o in loops if o.kind not in ("confirm_fact", "refill", "appointment")],
        "tasks": [task_json(t) for t in tasks],
        "timeline": timeline,
        "lastHeardAt": last_turn.at.isoformat() if last_turn else None,
        "nextAppointment": next_appt,
    }


# ── care views (the same data Saheli's tools read on WhatsApp) ─────────────────


@router.get("/{family_id}/{elder_id}/stock")
async def stock_view(family_id: str, elder_id: str, session: DB) -> dict:
    from app.care import features

    return {"medicines": await features.stock(session, family_id, elder_id), "refillWithinDays": features.REFILL_DAYS}


class StockIn(BaseModel):
    actor: Actor
    key: str = Field(min_length=3, max_length=160)
    count: int = Field(ge=0, le=2000)


@router.post("/{family_id}/{elder_id}/stock")
async def stock_set(family_id: str, elder_id: str, body: StockIn, session: DB) -> dict:
    from app.care import features

    f = await features.find_medicine(session, family_id, elder_id, body.key)
    if not f:
        raise HTTPException(status_code=404, detail="no such medicine")
    row = await features.set_stock(session, family_id=family_id, subject_id=elder_id, key=f.key, name=features.med_name(f), count=body.count, actor_id=body.actor.id)
    await session.commit()
    return row


class RefillOrderIn(BaseModel):
    actor: Actor
    service: str = Field(pattern="^(apollo|1mg|pharmeasy)$")
    qty: int = Field(default=1, ge=1, le=20)


@router.post("/{family_id}/{elder_id}/stock/{key}/order")
async def refill_order(family_id: str, elder_id: str, key: str, body: RefillOrderIn, session: DB) -> dict:
    from app.care import features

    f = await features.find_medicine(session, family_id, elder_id, key)
    if not f:
        raise HTTPException(status_code=404, detail="no such medicine")
    name = features.med_name(f)
    for t in await runtime.live_tasks(session, family_id):
        if t.service == body.service and t.kind == "order":
            return {"alreadyRunning": True, "task": task_json(t)}
    from app.specialists.contract import build_limits

    item = f"{name} {f.value.get('dose') or ''}".strip()
    limits = await build_limits(session, family_id=family_id, subject_id=elder_id, kind="order", agent="pharmacy", requester_is_elder=False)
    try:
        task = await runtime.create(
            session, family_id=family_id, subject_id=elder_id, requested_by=body.actor.id, service=body.service, kind="order",
            goal=f"Refill {item}", details={"items": [{"name": item, "qty": body.qty}], "refill_key": f.key}, limits=limits,
        )
    except runtime.TaskRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    live = (await session.execute(select(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.dedupe_key == f"refill:{elder_id}:{f.key}", OpenLoop.status == "open"))).scalars()
    for loop in live:
        await store.close_loop(session, loop.id, note=f"reorder started on {body.service}")
    await session.commit()
    return {"task": task_json(task)}


@router.get("/{family_id}/{elder_id}/emergency")
async def emergency_view(family_id: str, elder_id: str, session: DB) -> dict:
    from app.care import features

    return await features.emergency(session, family_id, elder_id)


@router.get("/{family_id}/{elder_id}/care-team")
async def care_team_view(family_id: str, elder_id: str, session: DB) -> dict:
    from app.care import features

    return await features.care_team(session, family_id, elder_id)


class QuestionIn(BaseModel):
    actor: Actor
    key: str = Field(min_length=3, max_length=160)
    question: str = Field(min_length=1, max_length=400)


@router.post("/{family_id}/{elder_id}/appointments/question")
async def appointment_question(family_id: str, elder_id: str, body: QuestionIn, session: DB) -> dict:
    out = await _run(session, _ctx(session, family_id, elder_id, body.actor), "add_doctor_question", {"appointment": body.key, "question": body.question})
    await session.commit()
    return out


@router.get("/{family_id}/{elder_id}/report")
async def report_view(family_id: str, elder_id: str, session: DB, days: int = 7, name: str = "") -> dict:
    from app.care import features

    r = await features.report(session, family_id, elder_id, days)
    r["narrative"] = await features.narrative(session, family_id, elder_id, name or "the care recipient", r)
    await session.commit()
    return r


class OutcomeIn(BaseModel):
    actor: Actor
    kind: str = Field(pattern="^(fall|hospital_visit|er_visit|doctor_visit|medicine_changed|recovered|all_fine|other)$")
    summary: str = Field(default="", max_length=400)


class FeedbackIn(BaseModel):
    actor: Actor
    target: str = Field(max_length=250)
    vote: str = Field(pattern="^(up|down)$")


class ConsentIn(BaseModel):
    actor: Actor
    granted: bool


class ConsentActor(BaseModel):
    actor: Actor


class ForgetIn(BaseModel):
    actor: Actor
    what: str = Field(min_length=3, max_length=200)


def _page_subjects(elder_id: str, actor: Actor) -> list[str]:
    """Whose memory a page reaches: the person, plus the shared family notes, except on a caregiver's own self-care
    page (their private record never mixes with the family's)."""
    return [elder_id] if actor.id == elder_id else [elder_id, "family"]


@router.get("/{family_id}/{elder_id}/memory-health")
async def memory_health_view(family_id: str, elder_id: str, session: DB) -> dict:
    """Problems in the care record Saheli will ask about, her profile card of this person, and what was forgotten."""
    from app.care import memory_upkeep
    from app.care.models import CareEvent, MemoryNote

    card = (await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id, MemoryNote.subject_id == elder_id,
                                                           MemoryNote.slug == "profile-card"))).scalar_one_or_none()
    forgotten = (await session.execute(select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == elder_id,
                                                               CareEvent.kind == "memory_forgotten")
                                       .order_by(CareEvent.at.desc()).limit(20))).scalars()
    return {
        "issues": await memory_upkeep.health(session, family_id, elder_id),
        "profileCard": card.body_md if card else None,
        "forgotten": [{"id": e.id, "at": e.at.isoformat(), "what": (e.payload or {}).get("what"), "restored": bool((e.payload or {}).get("restored"))}
                      for e in forgotten],
    }


@router.post("/{family_id}/{elder_id}/forget")
async def forget_view(family_id: str, elder_id: str, body: ForgetIn, session: DB) -> dict:
    from app.care import memory_upkeep

    out = await memory_upkeep.forget(session, family_id, _page_subjects(elder_id, body.actor), body.what, by=body.actor.id)
    await session.commit()
    return out


@router.post("/{family_id}/{elder_id}/forgotten/{event_id}/restore")
async def restore_view(family_id: str, elder_id: str, event_id: int, body: ConsentActor, session: DB) -> dict:
    from app.care import memory_upkeep

    out = await memory_upkeep.restore(session, family_id, event_id, by=body.actor.id, subjects=_page_subjects(elder_id, body.actor))
    await session.commit()
    return out


@router.get("/{family_id}/{elder_id}/outcomes")
async def outcomes_view(family_id: str, elder_id: str, session: DB) -> dict:
    from app.care import outcomes

    return {"outcomes": await outcomes.recent(session, family_id, elder_id), "kinds": outcomes.OUTCOMES,
            "consent": await outcomes.consent(session, family_id, elder_id)}


@router.post("/{family_id}/{elder_id}/outcomes")
async def outcomes_add(family_id: str, elder_id: str, body: OutcomeIn, session: DB) -> dict:
    """'Log an event' on the dashboard (same as telling Saheli on WhatsApp)."""
    from app.care import outcomes

    await outcomes.record_outcome(session, family_id=family_id, subject_id=elder_id, kind=body.kind, summary=body.summary.strip(),
                                  source="dashboard", actor_id=body.actor.id)
    await session.commit()
    return {"outcomes": await outcomes.recent(session, family_id, elder_id)}


@router.post("/{family_id}/{elder_id}/feedback")
async def feedback_add(family_id: str, elder_id: str, body: FeedbackIn, session: DB) -> dict:
    from app.care import outcomes

    await outcomes.record_feedback(session, family_id=family_id, subject_id=elder_id, target=body.target, vote=body.vote,
                                   by=body.actor.id, source="dashboard")
    await session.commit()
    return {"ok": True}


@router.get("/{family_id}/{elder_id}/speech")
async def speech_view(family_id: str, elder_id: str, session: DB) -> dict:
    """How this person speaks, from the care record (language, dialect, script); {} when not known. The backend reads it
    when its own copy is empty, so reminders and voice notes follow what Saheli already knows."""
    from app.care import language

    facts = await store.facts(session, family_id, elder_id)
    row = next((f for f in facts if f.domain == "language" and f.status == "active"), None)
    return language.normalise(row.value) if row else {}


@router.get("/{family_id}/{elder_id}/consent")
async def consent_view(family_id: str, elder_id: str, session: DB) -> dict:
    from app.care import outcomes

    return await outcomes.consent(session, family_id, elder_id)


@router.post("/{family_id}/{elder_id}/consent")
async def consent_set(family_id: str, elder_id: str, body: ConsentIn, session: DB) -> dict:
    from app.care import outcomes

    out = await outcomes.set_consent(session, family_id, elder_id, granted=body.granted, by=body.actor.id)
    await session.commit()
    return out


@router.get("/{family_id}/{elder_id}/patterns")
async def patterns_view(family_id: str, elder_id: str, session: DB) -> dict:
    """What Saheli noticed in the last two weeks without being asked (same as the WhatsApp 'patterns' tool)."""
    from app.care import patterns

    return {"patterns": [p.to_dict() for p in await patterns.find(session, family_id, elder_id)], "windowDays": patterns.WINDOW_DAYS}


@router.get("/{family_id}/{elder_id}/wellbeing")
async def wellbeing_view(family_id: str, elder_id: str, session: DB, days: int = 14) -> dict:
    from app.care import features

    return await features.wellbeing(session, family_id, elder_id, max(3, min(days, 60)))


@router.get("/{family_id}/{elder_id}/family-tasks")
async def family_tasks_view(family_id: str, elder_id: str, session: DB) -> dict:
    from app.care import features

    return {"tasks": await features.family_tasks(session, family_id)}


class FamilyTaskIn(BaseModel):
    actor: Actor
    title: str = Field(min_length=2, max_length=300)
    assignee: str = Field(min_length=3, max_length=64)
    due: str | None = None


@router.post("/{family_id}/{elder_id}/family-tasks")
async def family_task_add(family_id: str, elder_id: str, body: FamilyTaskIn, session: DB) -> dict:
    from app.care import features

    due = features.parse_when(body.due)
    loop = await features.add_family_task(session, family_id=family_id, subject_id=elder_id, title=body.title, assignee=body.assignee, due=due, by=body.actor.id)
    await store.record_event(session, family_id=family_id, subject_id=elder_id, kind="dashboard_edit", summary=f"task for {body.assignee}: {body.title}", actor_id=body.actor.id)
    await session.commit()
    return features.family_task_json(loop)


class TaskDoneIn(BaseModel):
    actor: Actor
    note: str = "done"


@router.post("/{family_id}/{elder_id}/family-tasks/{task_id}/done")
async def family_task_done(family_id: str, elder_id: str, task_id: uuid.UUID, body: TaskDoneIn, session: DB) -> dict:
    from app.care import features

    loop = await session.get(OpenLoop, task_id)
    if not loop or loop.family_id != family_id or loop.kind != "family_task":
        raise HTTPException(status_code=404, detail="no such task")
    await store.close_loop(session, task_id, note=body.note[:200])
    await session.commit()
    return features.family_task_json(loop)


@router.get("/{family_id}/{elder_id}/spending")
async def spending_view(family_id: str, elder_id: str, session: DB, month: str | None = None) -> dict:
    import re

    from app.care import features

    if month and not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise HTTPException(status_code=400, detail="month must be YYYY-MM")
    return await features.spending(session, family_id, month)


@router.get("/{family_id}/{elder_id}/work")
async def work_view(family_id: str, elder_id: str, session: DB, all: bool = False) -> dict:
    """Every job the family handed over: state, who acts next, next action, deadline, and anything stuck."""
    from app.care import work

    rows = await work.items(session, family_id, open_only=not all, days=14)
    roster = await store.roster(session, family_id)
    names = {m.get("id"): m.get("name") for m in ([roster.elder, *roster.members] if roster else [])}
    for r in rows:
        r["ownerName"] = "Saheli" if r["owner"] == work.SAHELI else names.get(r["owner"] or "", r["owner"])
    return {"items": rows, "stuck": sum(1 for r in rows if r.get("stuck")), "open": sum(1 for r in rows if r["state"] in ("working", "waiting"))}


class BoundariesIn(BaseModel):
    actor: Actor
    changes: dict


@router.get("/{family_id}/{elder_id}/boundaries")
async def boundaries_view(family_id: str, elder_id: str, session: DB) -> dict:
    """The family's limits for orders and rides, who approves, this month's spend, and orders waiting for approval."""
    from app.care import boundaries
    from app.tasks.models import Task

    policy = await boundaries.get(session, family_id)
    roster = await store.roster(session, family_id)
    names = {m.get("id"): m.get("name") for m in ([roster.elder, *roster.members] if roster else [])}
    waiting = list((await session.execute(select(Task).where(Task.family_id == family_id, Task.status == "awaiting_confirm",
                                                             Task.input_needed == "approve"))).scalars())
    return {
        "policy": policy, "plain": boundaries.describe(policy, names),
        "approvers": [{"id": a, "name": names.get(a, a)} for a in boundaries.approvers(policy, roster) if a != boundaries.ANYONE],
        "members": [{"id": m.get("id"), "name": m.get("name"), "role": m.get("role")} for m in (roster.members if roster else [])],
        "monthSpent": await boundaries.month_spent(session, family_id),
        "categories": list(boundaries.CATEGORIES),
        "waiting": [{"taskId": str(t.id), "title": task_json(t).get("title") or t.goal, **((t.details or {}).get("approval_needed") or {})} for t in waiting],
    }


@router.put("/{family_id}/{elder_id}/boundaries")
async def boundaries_save(family_id: str, elder_id: str, body: BoundariesIn, session: DB) -> dict:
    """Change the limits from the dashboard (only an approver; the backend checks the signed-in caregiver)."""
    from app.care import boundaries

    policy = await boundaries.get(session, family_id)
    roster = await store.roster(session, family_id)
    if not boundaries.can_manage(policy, roster, body.actor.id):
        raise HTTPException(status_code=403, detail="Only the family's approver can change the limits")
    ids = {m.get("id") for m in ([roster.elder, *roster.members] if roster else [])}
    if any(a not in ids for a in body.changes.get("approvers") or []) or elder_id in (body.changes.get("approvers") or []):
        raise HTTPException(status_code=400, detail="Approvers must be family members other than the care recipient")
    new = await boundaries.save(session, family_id, body.changes, by=body.actor.id, source_kind="dashboard")
    await session.commit()
    return {"policy": new}


class SkillIn(BaseModel):
    actor: Actor
    text: str = Field(min_length=3, max_length=600)


class SkillAction(BaseModel):
    actor: Actor
    action: str = Field(pattern="^(approve|edit|remove|restore)$")
    text: str | None = Field(default=None, max_length=600)


@router.get("/{family_id}/{elder_id}/skills")
async def skills_view(family_id: str, elder_id: str, session: DB) -> dict:
    """How this person likes things (family skills), plus what the store agents learned (shared, read-only)."""
    from app.care import skillbook

    rows = await skillbook.family_skills(session, family_id, [elder_id], statuses=("proposed", "active", "stale", "blocked"))
    return {"skills": [skillbook.view(s) for s in rows], "store": await skillbook.store_list(session)}


@router.post("/{family_id}/{elder_id}/skills")
async def skills_add(family_id: str, elder_id: str, body: SkillIn, session: DB) -> dict:
    from app.care import skillbook

    out = await skillbook.save_family(session, family_id, elder_id, body.text, source="caregiver", by=body.actor.id)
    if not out.get("saved"):
        raise HTTPException(status_code=422, detail="; ".join(out.get("problems") or ["not saved"]))
    await session.commit()
    return out


@router.post("/{family_id}/{elder_id}/skills/{skill_id}")
async def skills_decide(family_id: str, elder_id: str, skill_id: int, body: SkillAction, session: DB) -> dict:
    from app.care import skillbook

    s = await skillbook._family_row(session, family_id, skill_id)
    if not s or s.subject_id != elder_id:
        raise HTTPException(status_code=404, detail="skill not found")
    out = await skillbook.decide(session, family_id, skill_id, action=body.action, by=body.actor.id, body=body.text)
    if not out.get("ok"):
        raise HTTPException(status_code=422, detail="; ".join(out.get("problems") or [out.get("error") or "not changed"]))
    await session.commit()
    return out


@router.get("/{family_id}/{elder_id}/logins")
async def logins_view(family_id: str, elder_id: str, session: DB) -> dict:
    """Login state of each store/ride app for this family, from the agents' own runs."""
    return {"logins": await sandbox.logins(session, family_id)}


class UndoIn(BaseModel):
    actor: Actor
    mode: str = Field(default="undo", pattern="^(undo|restore)$")
    reason: str = Field(default="", max_length=300)
    # the caregiver confirmed in the dialog: an undo may end or restart a medicine, allergy or condition at once
    confirm: bool = False


@router.get("/{family_id}/{elder_id}/memory-history")
async def memory_history(family_id: str, elder_id: str, session: DB, kind: str | None = None, target: str | None = None,
                         what: str = "", limit: int = 40, actor: str = "") -> dict:
    """What changed in this person's memory (and the family notes, except on a caregiver's own self-care page): newest
    first, with who, where, why, and undo/restore. Lines someone asked to forget show as '(forgotten)'."""
    from app.care import versions

    if kind and kind not in versions.KINDS:
        raise HTTPException(status_code=400, detail="unknown kind")
    subjects = _page_subjects(elder_id, Actor(id=actor or "-"))
    rows = await versions.changes(session, family_id, subjects, kinds=(kind,) if kind else None, target=target,
                                  words=what, limit=min(max(limit, 1), 100), include_baseline=bool(target))
    hidden = await versions.forgotten_lines(session, family_id, subjects)
    return {"changes": [await versions.view(session, v, hidden=hidden) for v in rows]}


@router.get("/{family_id}/{elder_id}/memory-history/{version_id}/preview")
async def memory_undo_preview(family_id: str, elder_id: str, version_id: int, session: DB, mode: str = "undo", actor: str = "") -> dict:
    """What an undo (or restore) would do, in plain words, before the caregiver confirms. Changes nothing."""
    from app.care import versions

    v = await versions.get(session, family_id, version_id)
    if not v or v.subject_id not in _page_subjects(elder_id, Actor(id=actor or "-")):
        raise HTTPException(status_code=404, detail="no such change")
    return await versions.preview(session, v, "restore" if mode == "restore" else "undo")


@router.post("/{family_id}/{elder_id}/memory-history/{version_id}")
async def memory_undo(family_id: str, elder_id: str, version_id: int, body: UndoIn, session: DB) -> dict:
    """Undo one change, or put an item back as it was at that version (same tool Saheli uses on WhatsApp)."""
    from app.care import versions

    v = await versions.get(session, family_id, version_id)
    if not v or v.subject_id not in _page_subjects(elder_id, body.actor):
        raise HTTPException(status_code=404, detail="no such change")
    ctx = _ctx(session, family_id, elder_id, body.actor, confirmed=body.confirm)
    return await _run(session, ctx, "undo_change", {"id": version_id, "mode": body.mode, "reason": body.reason})
