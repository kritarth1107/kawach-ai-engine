import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.saheli_graph import (
    run_saheli_caregiver_chat,
    run_saheli_chat,
    run_saheli_check_in,
    run_saheli_family_share,
    run_saheli_outreach,
)
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.rag.retrieve import get_recent_messages
from app.services.tenant import scope_caregiver_chat_request, scope_chat_request

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(verify_api_secret)])


class ScheduleItemIn(BaseModel):
    title: str
    time: str | None = None
    dosage: str | None = None
    type: str | None = None


class ChatRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    message: str
    conversation_id: uuid.UUID | None = None
    companion_profile: dict | None = None
    care_record_context: str | None = None
    elder_thread_context: str | None = None
    labs_context: str | None = None
    session_context: str | None = None


class CheckInRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    conversation_id: uuid.UUID | None = None
    schedule_items: list[ScheduleItemIn] = Field(default_factory=list)
    care_record_context: str | None = None
    companion_profile: dict | None = None


class OutreachRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    conversation_id: uuid.UUID | None = None
    outreach_kind: str = Field(default="casual", pattern="^(casual|care|mixed)$")
    topic_bucket: str | None = None
    topic_hint: str | None = None
    care_record_context: str | None = None
    companion_profile: dict | None = None
    schedule_items: list[ScheduleItemIn] = Field(default_factory=list)


class FamilyShareRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    share_summary: str = Field(min_length=8, max_length=2000)
    memory_ids: list[uuid.UUID] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


class OutreachResponse(BaseModel):
    reply: str
    conversation_id: str
    topic_bucket: str
    topic_hint: str
    outreach_kind: str


class ChatMessageOut(BaseModel):
    role: str
    content: str
    created_at: str | None = None


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[ChatMessageOut]


def _history_messages(recent: list[tuple]) -> list[ChatMessageOut]:
    out: list[ChatMessageOut] = []
    for row in recent:
        role, content = row[0], row[1]
        created = row[2] if len(row) > 2 else None
        out.append(
            ChatMessageOut(
                role=role,
                content=content,
                created_at=created.isoformat() if created is not None else None,
            )
        )
    return out


@router.get("/history", response_model=ChatHistoryResponse)
async def chat_history(
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
):
    conv = await scope_chat_request(
        db,
        family_id=family_id,
        elder_id=elder_id,
        conversation_id=None,
    )
    recent = await get_recent_messages(
        db,
        family_id=family_id,
        elder_id=elder_id,
        limit=limit,
        conversation_id=conv.id,
    )
    return ChatHistoryResponse(
        conversation_id=str(conv.id),
        messages=_history_messages(recent),
    )


@router.get("/caregiver/history", response_model=ChatHistoryResponse)
async def caregiver_chat_history(
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
):
    conv = await scope_caregiver_chat_request(
        db,
        family_id=family_id,
        elder_id=elder_id,
        conversation_id=None,
    )
    recent = await get_recent_messages(
        db,
        family_id=family_id,
        elder_id=elder_id,
        limit=limit,
        conversation_id=conv.id,
    )
    return ChatHistoryResponse(
        conversation_id=str(conv.id),
        messages=_history_messages(recent),
    )


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    conv = await scope_chat_request(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=body.conversation_id,
    )

    reply = await run_saheli_chat(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=conv.id,
        message=body.message.strip(),
        companion_profile=body.companion_profile,
        care_record_context=body.care_record_context,
    )
    return ChatResponse(reply=reply, conversation_id=str(conv.id))


@router.post("/check-in", response_model=ChatResponse)
async def check_in(body: CheckInRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    conv = await scope_chat_request(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=body.conversation_id,
    )
    reply = await run_saheli_check_in(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=conv.id,
        schedule_items=[item.model_dump() for item in body.schedule_items],
        care_record_context=body.care_record_context,
        companion_profile=body.companion_profile,
    )
    return ChatResponse(reply=reply, conversation_id=str(conv.id))


@router.post("/outreach", response_model=OutreachResponse)
async def outreach(body: OutreachRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    conv = await scope_chat_request(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=body.conversation_id,
    )
    result = await run_saheli_outreach(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=conv.id,
        outreach_kind=body.outreach_kind,
        topic_bucket=body.topic_bucket,
        topic_hint=body.topic_hint,
        care_record_context=body.care_record_context,
        companion_profile=body.companion_profile,
        schedule_items=[item.model_dump() for item in body.schedule_items],
    )
    return OutreachResponse(
        reply=result["reply"],
        conversation_id=str(conv.id),
        topic_bucket=result["topic_bucket"],
        topic_hint=result["topic_hint"],
        outreach_kind=result["outreach_kind"],
    )


@router.post("/family-share", response_model=ChatResponse)
async def family_share(body: FamilyShareRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    await scope_chat_request(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=None,
    )
    reply = await run_saheli_family_share(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        share_summary=body.share_summary.strip(),
        memory_ids=body.memory_ids or None,
    )
    caregiver_conv = await scope_caregiver_chat_request(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=None,
    )
    return ChatResponse(reply=reply, conversation_id=str(caregiver_conv.id))


@router.post("/caregiver", response_model=ChatResponse)
async def caregiver_chat(body: ChatRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    conv = await scope_caregiver_chat_request(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=body.conversation_id,
    )

    reply = await run_saheli_caregiver_chat(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=conv.id,
        message=body.message.strip(),
        care_record_context=body.care_record_context,
        elder_thread_context=body.elder_thread_context,
        labs_context=body.labs_context,
        session_context=body.session_context,
    )
    return ChatResponse(reply=reply, conversation_id=str(conv.id))
