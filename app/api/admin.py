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

from fastapi import APIRouter, Depends, HTTPException, Request
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
    from app.models.entities import MemoryProfile

    last_dream = (await session.execute(select(func.max(MemoryProfile.rendered_at)))).scalar_one_or_none()
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


@router.get("/metrics")
async def admin_metrics(session: DB, days: int = 7) -> dict:
    from app.api.brain import agent_metrics

    return await agent_metrics(session=session, days=days)


@router.get("/learn")
async def admin_learn(weeks: int = 8) -> dict:
    from app.api.brain import learn_overview

    return await learn_overview(weeks=weeks)


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
    from app.care.models import CareFact

    rows = (await session.execute(
        select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject).order_by(CareFact.recorded_at.desc()).limit(400)
    )).scalars().all()
    return {"facts": [{"domain": f.domain, "key": f.key, "text": f.text, "status": f.status, "source": f.source_kind,
                       "confidence": f.confidence, "recordedAt": f.recorded_at.isoformat(),
                       "validTo": f.valid_to.isoformat() if f.valid_to else None} for f in rows]}


@router.get("/families/{family_id}/export")
async def admin_export(family_id: str, subject: str, session: DB) -> dict:
    """Everything Saheli holds about one person in one family, for that person's own data request."""
    from app.care.models import CareFact, Turn

    facts = (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject))).scalars().all()
    turns = (await session.execute(select(Turn).where(Turn.family_id == family_id, Turn.thread_id == subject).order_by(Turn.id))).scalars().all()
    return {
        "careRecord": [{"domain": f.domain, "key": f.key, "value": f.value, "text": f.text, "status": f.status, "source": f.source_kind,
                        "recordedAt": f.recorded_at.isoformat(), "validFrom": f.valid_from.isoformat(),
                        "validTo": f.valid_to.isoformat() if f.valid_to else None} for f in facts],
        "conversation": [{"at": t.at.isoformat(), "role": t.role, "text": t.text} for t in turns],
    }
