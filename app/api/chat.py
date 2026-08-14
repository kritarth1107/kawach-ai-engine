import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.saheli_graph import run_saheli_caregiver_chat, run_saheli_chat, run_saheli_check_in
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


class CheckInRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    conversation_id: uuid.UUID | None = None
    schedule_items: list[ScheduleItemIn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


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
    )
    return ChatResponse(reply=reply, conversation_id=str(conv.id))


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
    )
    return ChatResponse(reply=reply, conversation_id=str(conv.id))
