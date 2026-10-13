"""Admin console routes (/v2/admin), separate from the public /v2 API and secured differently.

Only the admin API service (kavach-admin-api) may call them: its Google identity (ADMIN_API_SA) plus its own secret
(ENGINE_ADMIN_SECRET), never the backend's X-Kavach-Secret. Unset either and the routes answer 503 (switched off).
Conversations, care records and exports are reached only after the admin API checked break-glass permission, a
reason and wrote its audit entry.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db


def _decode(part: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def _audiences() -> set[str]:
    return {a.strip().rstrip("/") for a in os.environ.get("ENGINE_AUDIENCES", "").split(",") if a.strip()}


def caller_email(authorization: str | None, *, on_cloud_run: bool | None = None, verify=None) -> str:
    """The Google identity that called. Cloud Run IAM already checked the token; this re-checks who it is."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="no caller token")
    token = authorization.split(" ", 1)[1].strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="bad caller token")
    on_cloud_run = bool(os.environ.get("K_SERVICE")) if on_cloud_run is None else on_cloud_run
    auds = _audiences()
    stripped = parts[2] in ("", "SIGNATURE_REMOVED_BY_GOOGLE")  # Cloud Run checked it and removed the signature
    if not stripped:
        try:
            if verify is None:
                from google.auth.transport import requests as g_requests
                from google.oauth2 import id_token

                payload = id_token.verify_oauth2_token(token, g_requests.Request(), audience=None)
            else:
                payload = verify(token)
        except Exception:
            raise HTTPException(status_code=401, detail="bad caller token") from None
    elif on_cloud_run:
        # Only behind Cloud Run IAM (the service is deployed --no-allow-unauthenticated): it verified this token.
        payload = _decode(parts[1])
        if float(payload.get("exp", 0)) < time.time():
            raise HTTPException(status_code=401, detail="bad caller token")
    else:
        raise HTTPException(status_code=401, detail="bad caller token")
    if not auds or str(payload.get("aud", "")).rstrip("/") not in auds:  # no audiences configured = closed
        raise HTTPException(status_code=401, detail="bad caller token")
    if payload.get("email_verified") is False:
        raise HTTPException(status_code=403, detail="caller not allowed")
    return str(payload.get("email", "")).lower()


async def verify_admin_caller(request: Request) -> None:
    secret = os.environ.get("ENGINE_ADMIN_SECRET", "")
    allowed = os.environ.get("ADMIN_API_SA", "").lower()
    if len(secret) < 32 or not allowed:
        raise HTTPException(status_code=503, detail="admin routes are switched off")
    if not hmac.compare_digest(request.headers.get("x-admin-secret", ""), secret):
        raise HTTPException(status_code=401, detail="bad admin secret")
    if caller_email(request.headers.get("authorization")) != allowed:
        raise HTTPException(status_code=403, detail="caller not allowed")


router = APIRouter(prefix="/v2/admin", tags=["admin"], dependencies=[Depends(verify_admin_caller)])
DB = Annotated[AsyncSession, Depends(get_db)]


@router.get("/health")
async def admin_health(session: DB) -> dict:
    from app.care.models import FamilyRoster
    from app.llm.provider import llm_provider_label
    from app.rag.embeddings import embedding_provider_label
    from app.care.models import CareEvent

    # the nightly dream's ledger event (app/care/dream.py), not the retired v1 profile render (trust plan A1)
    last_dream = (await session.execute(select(func.max(CareEvent.at)).where(CareEvent.kind == "dream"))).scalar_one_or_none()
    families = (await session.execute(select(func.count()).select_from(FamilyRoster))).scalar_one()
    from app.db.session import SessionLocal
    from app.learn.jobs import last_backup

    try:
        backup = await last_backup(SessionLocal)
    except Exception:  # noqa: BLE001 - the health view must not fail on the backup lookup
        backup = None
    return {"status": "ok", "llm": llm_provider_label(), "embeddings": embedding_provider_label(),
            "lastDream": last_dream.isoformat() if last_dream else None, "families": families, "backup": backup}


@router.get("/spend")
async def admin_spend(days: int = 30) -> dict:
    from app.api.brain import llm_spend

    return await llm_spend(days=days)


@router.get("/models")
async def admin_models() -> dict:
    """Which models each role uses (first = primary, rest = fallbacks), their estimated prices, and today's caps."""
    from app.llm import router as llm_router
    from app.llm import spend
    from app.llm.provider import llm_provider_label
    from app.rag.embeddings import embedding_provider_label

    roles = sorted(set(llm_router.DEFAULT_ROUTES) | llm_router.configured_roles())
    configured = llm_router.configured_roles()
    out = {}
    for role in roles:
        out[role] = {
            "configured": role in configured,
            "routes": [{"provider": r.provider, "model": r.model, "location": r.location,
                        "priceInr": [round(x, 2) for x in spend.price_inr(r.model)]} for r in llm_router.routes_for(role)],
        }
    return {"llm": llm_provider_label(), "embeddings": embedding_provider_label(), "roles": out,
            "today": round(await spend.spent_today(), 2), "softCap": spend.soft_cap(), "hardCap": spend.hard_cap(),
            "usdInr": spend.USD_INR}


@router.get("/metrics")
async def admin_metrics(session: DB, days: int = 7) -> dict:
    from app.api.brain import agent_metrics

    return await agent_metrics(session=session, days=days)


@router.get("/learn")
async def admin_learn(weeks: int = 8) -> dict:
    from app.api.brain import learn_overview

    return await learn_overview(weeks=weeks)


@router.get("/flywheel")
async def admin_flywheel(session: DB, days: int = 14) -> dict:
    """The flywheel page: task completion/time/reopen/cost, conversations, check-ins, per family, plus learning signals
    (correction cases this week, rule proposals waiting, fine-tuning readiness) and the browser credit."""
    from app.api.brain import agent_credits
    from app.learn import corrections, flywheel, tuning_readiness

    out = await flywheel.compute(session, days=max(1, min(days, 90)))
    cases = await corrections.cases(session, days=7, limit=500)
    kinds: dict[str, int] = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    ready = await tuning_readiness.readiness(session)
    try:
        credit = await agent_credits()
    except Exception:  # noqa: BLE001
        credit = {"usd": None, "warnAt": None}
    from sqlalchemy import func, select

    from app.learn.models import RuleProposal

    waiting = (await session.execute(select(func.count()).select_from(RuleProposal).where(RuleProposal.status == "proposed"))).scalar_one()
    return {**out, "learning": {"correctionsThisWeek": kinds, "ruleProposalsWaiting": waiting,
                                "tuning": {k: ready[k] for k in ("ready", "goodReplies", "consented", "families", "missing")}},
            "credit": credit}


class By(BaseModel):
    by: str = Field(min_length=1, max_length=64)


class RuleBy(By):
    approve: bool


@router.post("/learn/playbooks/{version}/{action}")
async def admin_playbook(version: int, action: str, body: By) -> dict:
    from app.api import brain

    if action == "approve":
        return await brain.learn_approve(version, brain.PlaybookAction(by=body.by))
    if action == "block":
        return await brain.learn_block(version, brain.PlaybookAction(by=body.by))
    raise HTTPException(status_code=404, detail="no such action")


@router.post("/learn/rules/{rule_id}")
async def admin_rule(rule_id: int, body: RuleBy) -> dict:
    from app.api import brain

    return await brain.learn_rule_decision(rule_id, brain.RuleDecision(by=body.by, approve=body.approve))


@router.get("/families/{family_id}/conversation")
async def admin_conversation(family_id: str, thread: str, session: DB, limit: int = 80) -> dict:
    from app.care.models import Turn

    rows = (await session.execute(
        select(Turn).where(Turn.family_id == family_id, Turn.thread_id == thread).order_by(Turn.id.desc()).limit(max(1, min(limit, 300)))
    )).scalars().all()
    return {"turns": [{"at": t.at.isoformat(), "role": t.role, "speaker": t.speaker_id, "text": t.text,
                       "voice": bool((t.meta or {}).get("modality") == "voice")} for t in reversed(rows)]}


@router.get("/families/{family_id}/care-record")
async def admin_care_record(family_id: str, subject: str, session: DB) -> dict:
    from app.care import links
    from app.care.models import CareFact
    from app.care.versions import MemoryVersion

    rows = (await session.execute(
        select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject).order_by(CareFact.recorded_at.desc()).limit(400)
    )).scalars().all()
    # the change that made each row, for the "Why?" link
    made = dict((await session.execute(select(MemoryVersion.ref, func.min(MemoryVersion.id)).where(
        MemoryVersion.family_id == family_id, MemoryVersion.subject_id == subject, MemoryVersion.kind == "fact",
        MemoryVersion.ref.in_([str(f.id) for f in rows])).group_by(MemoryVersion.ref))).all()) if rows else {}
    linked = links.by_from(await links.links_for(session, family_id, subject, statuses=("active", "pending")))
    return {"facts": [{"domain": f.domain, "key": f.key, "text": f.text, "status": f.status, "source": f.source_kind,
                       "confidence": f.confidence, "recordedAt": f.recorded_at.isoformat(),
                       "validTo": f.valid_to.isoformat() if f.valid_to else None, "versionId": made.get(str(f.id)),
                       "links": [{"text": links.phrase(ln), "status": ln.status} for ln in linked.get(f.key, [])] if f.status == "active" else []}
                      for f in rows]}


@router.get("/why")
async def admin_why(session: DB, family: str, task: str | None = None, version: int | None = None, loop: str | None = None,
                    event: int | None = None, trace_id: Annotated[str | None, Query(alias="trace")] = None) -> dict:
    """The why-trail (D-13) of a task, a care-record change, an alert or question (open loop), a ledger event or a trace,
    within one family: the message, model, role and cost, tool calls, confirms, outcome and what was learned."""
    from app.care import why

    if not any((task, version, loop, event, trace_id)):
        raise HTTPException(status_code=400, detail="give task, version, loop, event or trace")
    try:
        out = await why.chain(session, family_id=family, task_id=task, version_id=version, loop_id=loop, event_id=event, trace_id=trace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad id") from None
    if out is None:
        raise HTTPException(status_code=404, detail="nothing like that in this family")
    return out


@router.get("/families/{family_id}/why")
async def admin_why_recent(family_id: str, session: DB, days: int = 14) -> dict:
    """What can be asked "why?" about in this family: recent tasks, care-record and link changes, alerts and questions."""
    from app.care import why

    return await why.recent(session, family_id, days=days)


@router.get("/families/{family_id}/export")
async def admin_export(family_id: str, subject: str, session: DB) -> dict:
    """Everything Saheli holds about one person in one family, for that person's own data request."""
    from app.care.models import CareFact, Turn

    from app.care.models import CareLink

    facts = (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject))).scalars().all()
    turns = (await session.execute(select(Turn).where(Turn.family_id == family_id, Turn.thread_id == subject).order_by(Turn.id))).scalars().all()
    rows = (await session.execute(select(CareLink).where(CareLink.family_id == family_id, CareLink.subject_id == subject))).scalars().all()
    return {
        "careRecord": [{"domain": f.domain, "key": f.key, "value": f.value, "text": f.text, "status": f.status, "source": f.source_kind,
                        "recordedAt": f.recorded_at.isoformat(), "validFrom": f.valid_from.isoformat(),
                        "validTo": f.valid_to.isoformat() if f.valid_to else None} for f in facts],
        "links": [{"from": ln.from_ref, "relation": ln.relation, "to": ln.to_ref, "label": ln.label, "status": ln.status, "source": ln.source_kind,
                   "validFrom": ln.valid_from.isoformat(), "validTo": ln.valid_to.isoformat() if ln.valid_to else None} for ln in rows],
        "conversation": [{"at": t.at.isoformat(), "role": t.role, "text": t.text} for t in turns],
    }


@router.get("/live")
async def admin_live(session: DB, minutes: int = 30) -> dict:
    """What is going on right now (founder 2026-10-13: "watch live, every session ongoing"): conversations in the last
    minutes with their pace, cost and mood, orders and rides running, check-ins due within the hour, and alerts of the
    day. No message text here: reading a conversation stays behind break-glass."""
    from datetime import timedelta

    from app.care.models import CareEvent, OpenLoop, Turn
    from app.care.why import TraceTurn
    from app.core import clock
    from app.tasks.models import Task as CareTask

    now = clock.now()
    since = now - timedelta(minutes=max(5, min(minutes, 240)))
    real = lambda col: ~col.startswith("shadow:") & ~col.startswith("sim-")  # noqa: E731
    turns = (await session.execute(select(Turn.family_id, Turn.thread_id, Turn.role, Turn.at).where(Turn.at >= since, real(Turn.family_id))
                                   .order_by(Turn.at))).all()
    convs: dict[tuple[str, str], dict] = {}
    for fid, thread, role, at in turns:
        if thread == "saheli-scheduler":
            continue
        c = convs.setdefault((fid, thread), {"familyId": fid, "personId": thread, "startedAt": at.isoformat(), "messages": 0, "replies": 0,
                                             "lastAt": at.isoformat(), "lastRole": role})
        c["messages" if role == "user" else "replies"] += 1
        c["lastAt"], c["lastRole"] = at.isoformat(), role
    traces = (await session.execute(select(TraceTurn.family_id, TraceTurn.thread_id, TraceTurn.ms, TraceTurn.cost_inr, TraceTurn.models, TraceTurn.kind)
                                    .where(TraceTurn.at >= since, real(TraceTurn.family_id)))).all()
    outreach: dict[str, int] = {}
    for fid, thread, ms, cost, models, kind in traces:
        c = convs.get((fid, thread))
        if kind == "system":
            outreach[fid] = outreach.get(fid, 0) + 1
        if not c:
            continue
        c.setdefault("_ms", []).append(ms or 0)
        c["costInr"] = round(c.get("costInr", 0) + float(cost or 0), 2)
        for m in models or []:
            if isinstance(m, dict) and m.get("role") == "brain" and m.get("model"):
                c["model"] = m["model"]
    if convs:
        moods = (await session.execute(select(CareEvent.family_id, CareEvent.subject_id, CareEvent.payload, CareEvent.at)
                                       .where(CareEvent.kind == "mood", CareEvent.at >= now - timedelta(hours=24),
                                              CareEvent.family_id.in_({k[0] for k in convs})).order_by(CareEvent.at))).all()
        last_mood = {(f, s): p for f, s, p, _ in moods}
    else:
        last_mood = {}
    out_convs = []
    for (fid, thread), c in convs.items():
        ms = c.pop("_ms", [])
        mood = last_mood.get((fid, thread)) or {}
        waiting = c["lastRole"] == "user"
        out_convs.append({**c, "avgReplyMs": int(sum(ms) / len(ms)) if ms else None, "waitingForSaheli": waiting,
                          "waitingSeconds": int((now - datetime_from(c["lastAt"])).total_seconds()) if waiting else None,
                          "mood": {"score": mood.get("score"), "word": mood.get("word")} if mood else None,
                          "active": (now - datetime_from(c["lastAt"])).total_seconds() < 300})
    out_convs.sort(key=lambda c: c["lastAt"], reverse=True)

    tasks = (await session.execute(select(CareTask).where(CareTask.status.in_(("queued", "running", "needs_input", "awaiting_confirm")),
                                                          real(CareTask.family_id)).order_by(CareTask.updated_at.desc()).limit(50))).scalars().all()
    out_tasks = [{"taskId": str(t.id), "familyId": t.family_id, "personId": t.subject_id, "service": t.service, "kind": t.kind,
                  "status": t.status, "phase": t.phase, "inputNeeded": t.input_needed, "startedAt": t.created_at.isoformat(),
                  "updatedAt": t.updated_at.isoformat(), "runs": t.runs, "hasLiveView": bool(t.agent_session),
                  "minutes": int((now - t.created_at).total_seconds() // 60)} for t in tasks]

    due = (await session.execute(select(OpenLoop.family_id, OpenLoop.owner_id, OpenLoop.kind, OpenLoop.wake_at)
                                 .where(OpenLoop.status == "open", OpenLoop.wake_at >= now, OpenLoop.wake_at <= now + timedelta(minutes=60),
                                        real(OpenLoop.family_id)).order_by(OpenLoop.wake_at).limit(50))).all()
    alerts = (await session.execute(select(CareEvent.family_id, CareEvent.subject_id, CareEvent.kind, CareEvent.at)
                                    .where(CareEvent.kind.in_(("alert_whatsapp", "alert_dashboard", "crisis", "emergency_alone")),
                                           CareEvent.at >= now - timedelta(hours=24), real(CareEvent.family_id)).order_by(CareEvent.at.desc()).limit(50))).all()
    return {
        "now": now.isoformat(),
        "windowMinutes": int((now - since).total_seconds() // 60),
        "conversations": out_convs,
        "outreach": [{"familyId": f, "count": n} for f, n in outreach.items()],
        "tasks": out_tasks,
        "comingUp": [{"familyId": f, "personId": o, "kind": k, "at": w.isoformat()} for f, o, k, w in due],
        "alerts": [{"familyId": f, "personId": s, "kind": k, "at": a.isoformat()} for f, s, k, a in alerts],
    }


def datetime_from(iso: str):
    from datetime import datetime as _dt

    return _dt.fromisoformat(iso)


@router.get("/tasks/{task_id}/live-url")
async def admin_task_live_url(task_id: str, session: DB) -> dict:
    """The browser view of a running order or ride (the admin API asks for break-glass and a reason first)."""
    import uuid as _uuid

    from app.brain import tools
    from app.tasks.models import Task as CareTask

    try:
        t = await session.get(CareTask, _uuid.UUID(task_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="bad id") from None
    if not t:
        raise HTTPException(status_code=404, detail="no such task")
    if not t.agent_session:
        return {"url": None}
    try:
        return {"url": await tools.task_agent().live_url(t.agent_session)}
    except Exception:  # noqa: BLE001 — the browser may already be closed
        return {"url": None}
