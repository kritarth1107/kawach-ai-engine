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
