import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.caregiver_agent import run_elder_whatsapp_agent, stream_caregiver_agent
from app.agents.saheli_graph import (
    run_saheli_caregiver_chat,
    run_saheli_chat,
    run_saheli_check_in,
    run_saheli_family_share,
    run_saheli_outreach,
)
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.rag.retrieve import get_recent_messages, sync_conversation_history
from app.services.tenant import scope_caregiver_chat_request, scope_chat_request

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(verify_api_secret)])
logger = logging.getLogger(__name__)


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
    order_context: str | None = None
    schedule_context: str | None = None
    channel_context: str | None = None
    use_agent: bool = True
    actor_user_id: str | None = None
    kavach_family_id: str | None = None
    kavach_recipient_user_id: str | None = None


class SyncHistoryMessage(BaseModel):
    external_id: str
    role: str
    content: str
    created_at: str | None = None


class SyncHistoryRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    conversation_id: uuid.UUID | None = None
    thread: str = Field(default="caregiver", pattern="^(elder|caregiver)$")
    messages: list[SyncHistoryMessage] = Field(default_factory=list)


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
    outreach_kind: str = Field(default="casual", pattern="^(casual|care|mixed|memory)$")
    memory_hint: str | None = None
    topic_bucket: str | None = None
    topic_hint: str | None = None
    care_record_context: str | None = None
    companion_profile: dict | None = None
    schedule_items: list[ScheduleItemIn] = Field(default_factory=list)
    # Previous Saheli nudge(s) went unanswered → gentle continuation (sent as a quoted reply).
    followup: dict | None = None


class FamilyShareRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    share_summary: str = Field(min_length=8, max_length=2000)
    memory_ids: list[uuid.UUID] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    order: dict | None = None
    connect: dict | None = None
    order_preview: dict | None = None
    order_flow: dict | None = None
    tool_trace: list[dict] | None = None


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

    care_context = body.care_record_context
    if body.schedule_context:
        care_context = "\n\n".join(filter(None, [body.schedule_context, care_context]))

    if body.use_agent and body.actor_user_id:
        from app.models.entities import Message, MessageRole

        recent = await get_recent_messages(
            db,
            family_id=body.family_id,
            elder_id=body.elder_id,
            limit=50,
            conversation_id=conv.id,
        )
        from app.rag.retrieve import db_messages_to_langchain

        history = db_messages_to_langchain(recent)
        try:
            result = await run_elder_whatsapp_agent(
                db,
                family_id=body.family_id,
                elder_id=body.elder_id,
                message=body.message.strip(),
                care_record_context=care_context,
                schedule_context=body.schedule_context,
                channel_context=body.channel_context,
                order_context=body.order_context,
                companion_profile=body.companion_profile,
                history_messages=history,
                actor_user_id=body.actor_user_id,
                kavach_family_id=body.kavach_family_id,
                kavach_recipient_user_id=body.kavach_recipient_user_id,
            )
        except Exception as exc:
            logger.exception(
                "elder whatsapp agent failed family=%s elder=%s",
                body.family_id,
                body.elder_id,
            )
            raise HTTPException(
                status_code=503,
                detail="Saheli agent unavailable",
            ) from exc
        reply = result.get("reply", "")
        elder_msg = Message(
            conversation_id=conv.id,
            family_id=body.family_id,
            elder_id=body.elder_id,
            role=MessageRole.elder,
            content=body.message.strip(),
        )
        db.add(elder_msg)
        db.add(
            Message(
                conversation_id=conv.id,
                family_id=body.family_id,
                elder_id=body.elder_id,
                role=MessageRole.saheli,
                content=reply,
            )
        )
        await db.flush()
        from app.rag.memory_queue import (
            eager_memory_extract,
            is_high_value_message,
            schedule_memory_extract,
        )

        msg_text = body.message.strip()
        if is_high_value_message(msg_text):
            await eager_memory_extract(
                family_id=body.family_id,
                elder_id=body.elder_id,
                message=msg_text,
                source_message_id=elder_msg.id,
                source_role="elder",
            )
        else:
            schedule_memory_extract(
                family_id=body.family_id,
                elder_id=body.elder_id,
                message=msg_text,
                source_message_id=elder_msg.id,
                source_role="elder",
            )
        await db.commit()
        return ChatResponse(
            reply=reply,
            conversation_id=str(conv.id),
            order=result.get("order"),
            connect=result.get("connect"),
            order_preview=result.get("order_preview"),
            order_flow=result.get("order_flow"),
            tool_trace=result.get("tool_trace"),
        )

    reply = await run_saheli_chat(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=conv.id,
        message=body.message.strip(),
        companion_profile=body.companion_profile,
        care_record_context=care_context,
        schedule_context=body.schedule_context,
        channel_context=body.channel_context,
        order_context=body.order_context,
    )
    return ChatResponse(reply=reply, conversation_id=str(conv.id))


@router.post("/sync-history")
async def sync_history(body: SyncHistoryRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    if body.thread == "caregiver":
        conv = await scope_caregiver_chat_request(
            db,
            family_id=body.family_id,
            elder_id=body.elder_id,
            conversation_id=body.conversation_id,
        )
    else:
        conv = await scope_chat_request(
            db,
            family_id=body.family_id,
            elder_id=body.elder_id,
            conversation_id=body.conversation_id,
        )
    synced = await sync_conversation_history(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=conv.id,
        thread=body.thread,
        messages=[m.model_dump() for m in body.messages],
    )
    return {"conversation_id": str(conv.id), "synced": synced}


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
        memory_hint=body.memory_hint,
        followup=body.followup,
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

    try:
        result = await run_saheli_caregiver_chat(
            db,
            family_id=body.family_id,
            elder_id=body.elder_id,
            conversation_id=conv.id,
            message=body.message.strip(),
            care_record_context=body.care_record_context,
            elder_thread_context=body.elder_thread_context,
            labs_context=body.labs_context,
            session_context=body.session_context,
            order_context=body.order_context,
            use_agent=body.use_agent,
            actor_user_id=body.actor_user_id,
            kavach_family_id=body.kavach_family_id,
            kavach_recipient_user_id=body.kavach_recipient_user_id,
        )
    except Exception as exc:
        logger.exception(
            "caregiver chat failed family=%s elder=%s",
            body.family_id,
            body.elder_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Saheli agent unavailable",
        ) from exc
    return ChatResponse(
        reply=result.get("reply", ""),
        conversation_id=str(conv.id),
        order=result.get("order"),
        connect=result.get("connect"),
        order_preview=result.get("order_preview"),
        tool_trace=result.get("tool_trace"),
    )


@router.post("/caregiver/stream")
async def caregiver_chat_stream(body: ChatRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    conv = await scope_caregiver_chat_request(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        conversation_id=body.conversation_id,
    )

    async def event_generator():
        if body.use_agent and body.actor_user_id:
            async for event in stream_caregiver_agent(
                db,
                family_id=body.family_id,
                elder_id=body.elder_id,
                message=body.message.strip(),
                care_record_context=body.care_record_context,
                elder_thread_context=body.elder_thread_context,
                labs_context=body.labs_context,
                session_context=body.session_context,
                order_context=body.order_context,
                actor_user_id=body.actor_user_id,
                kavach_family_id=body.kavach_family_id,
                kavach_recipient_user_id=body.kavach_recipient_user_id,
            ):
                if event.get("type") == "done":
                    event["conversation_id"] = str(conv.id)
                yield f"data: {json.dumps(event)}\n\n"
            return

        result = await run_saheli_caregiver_chat(
            db,
            family_id=body.family_id,
            elder_id=body.elder_id,
            conversation_id=conv.id,
            message=body.message.strip(),
            care_record_context=body.care_record_context,
            elder_thread_context=body.elder_thread_context,
            labs_context=body.labs_context,
            session_context=body.session_context,
            order_context=body.order_context,
            use_agent=body.use_agent,
            actor_user_id=body.actor_user_id,
            kavach_family_id=body.kavach_family_id,
            kavach_recipient_user_id=body.kavach_recipient_user_id,
        )
        reply = result.get("reply", "")
        chunk_size = 24
        for i in range(0, len(reply), chunk_size):
            yield f"data: {json.dumps({'type': 'token', 'delta': reply[i:i + chunk_size]})}\n\n"
        if result.get("order"):
            yield f"data: {json.dumps({'type': 'tool_result', 'id': 'order', 'order': result['order']})}\n\n"
        if result.get("connect"):
            yield f"data: {json.dumps({'type': 'tool_result', 'id': 'connect', 'connect': result['connect']})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id), 'reply': reply})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
