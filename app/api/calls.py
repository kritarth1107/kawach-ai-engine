"""Phone calls with Saheli (behind CALLS_ENABLED; needs the founder's telephony accounts, see journal 2026-10-09).

The voice layer (ElevenLabs Agents on an Exotel number) does the listening and speaking; Saheli's own brain answers each
turn, so a call reads and writes the same care record as WhatsApp. The public backend receives the voice layer's
'custom LLM' calls and post-call webhooks (signed) and forwards them here behind the internal secret:
  POST /v2/calls/turn     one turn of a live call → Saheli's reply (spoken by the voice layer)
  POST /v2/calls/summary  after the call: transcript summary and outcome into the ledger
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import verify_api_secret

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/calls", tags=["calls"], dependencies=[Depends(verify_api_secret)])


def enabled() -> bool:
    return os.getenv("CALLS_ENABLED", "off") == "on"


class CallTurn(BaseModel):
    family_id: str
    person_id: str
    call_id: str = Field(max_length=80)
    text: str = Field(max_length=2000)
    turn: int = 0
    purpose: str = Field(default="talk", max_length=40)  # talk | reminder | checkin | red_flag


@router.post("/turn")
async def call_turn(body: CallTurn) -> dict:
    from app.brain.host import LiveHost
    from app.brain.loop import TurnRequest, run_turn
    from app.care import store
    from app.db.session import SessionLocal

    if not enabled():
        raise HTTPException(status_code=503, detail="calls are off")
    async with SessionLocal() as session:
        roster = await store.roster(session, body.family_id)
        if not roster:
            raise HTTPException(status_code=404, detail="no such family")
        people = {m.get("id"): m for m in [roster.elder, *roster.members]}
        speaker = people.get(body.person_id)
        if not speaker:
            raise HTTPException(status_code=403, detail="caller is not in this family")
        text = body.text if body.turn else f"[Phone call started: {body.purpose}] {body.text}".strip()
        res = await run_turn(session, LiveHost(), TurnRequest(
            family_id=body.family_id, elder=roster.elder, speaker=speaker, members=roster.members, text=text,
            message_ref=f"call:{body.call_id}:{body.turn}", channel="call", modality="voice"))
    return {"reply": res.reply, "model": res.model, "ms": res.ms}


class CallSummary(BaseModel):
    family_id: str
    person_id: str
    call_id: str = Field(max_length=80)
    purpose: str = "talk"
    summary: str = Field(default="", max_length=2000)
    minutes: float | None = None
    outcome: str | None = Field(default=None, max_length=200)  # e.g. "took medicine: yes"


@router.post("/summary")
async def call_summary(body: CallSummary) -> dict:
    from app.care import store
    from app.db.session import SessionLocal

    async with SessionLocal() as session:
        await store.record_event(
            session, family_id=body.family_id, subject_id=body.person_id, kind="call",
            summary=f"Phone call ({body.purpose}{f', {body.minutes:.0f} min' if body.minutes else ''}): {body.summary[:400]}",
            payload={"call_id": body.call_id, "purpose": body.purpose, "outcome": body.outcome, "minutes": body.minutes},
            ref=f"call:{body.call_id}")
        await session.commit()
    return {"recorded": True}
