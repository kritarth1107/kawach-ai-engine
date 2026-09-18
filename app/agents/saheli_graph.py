import random
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompts import (
    OUTREACH_TOPIC_HINTS,
    SAHELI_CARE_RULES,
    build_elder_system_prompt,
    build_family_share_system_prompt,
    build_outreach_system_prompt,
    CAREGIVER_SAHELI_SYSTEM,
)
from app.agents.caregiver_agent import run_caregiver_agent
from app.llm.provider import chat_invoke, chat_invoke_messages
from app.models.entities import Conversation, Elder, Message, MessageRole
from app.rag.memory_extract import process_elder_message_memories
from app.rag.retrieve import (
    db_messages_to_langchain,
    format_family_memories,
    get_elder_thread_context,
    get_recent_messages,
    retrieve_context,
    retrieve_family_memories,
)
from app.services.family import get_caregiver_conversation, get_primary_conversation


class SaheliState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    family_id: str
    elder_id: str
    conversation_id: str
    user_message: str
    rag_context: str
    family_memories: str
    recent_chat: str
    companion_profile: dict[str, Any]
    care_record_context: str
    order_context: str
    reply: str


async def node_retrieve(state: SaheliState, session: AsyncSession) -> dict:
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    query = state["user_message"]

    chunks = await retrieve_context(session, family_id=family_id, elder_id=elder_id, query=query)
    rag_lines = [f"[{c.kind}:{c.source}] {c.content}" for c in chunks]
    rag_context = "\n".join(rag_lines) if rag_lines else "(No matching memory yet.)"

    memories = await retrieve_family_memories(
        session, family_id=family_id, elder_id=elder_id, query=query, limit=10
    )
    family_memories = format_family_memories(memories)

    primary = await get_primary_conversation(session, family_id, elder_id)
    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=16,
        conversation_id=primary.id,
    )
    recent_chat = "\n".join(f"{role}: {content}" for role, content, _ts in recent)

    return {"rag_context": rag_context, "family_memories": family_memories, "recent_chat": recent_chat}


async def node_generate(state: SaheliState, session: AsyncSession) -> dict:
    system = build_elder_system_prompt(
        rag_context=state["rag_context"],
        recent_chat="",
        family_memories=state["family_memories"],
        companion_profile=state.get("companion_profile"),
        care_record_context=state.get("care_record_context") or None,
    )
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    conversation_id = uuid.UUID(state["conversation_id"])
    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=12,
        conversation_id=conversation_id,
    )
    history = db_messages_to_langchain(recent)
    messages = [SystemMessage(content=system), *history, HumanMessage(content=state["user_message"])]
    if state.get("order_context"):
        messages.insert(-1, HumanMessage(content=state["order_context"]))
    response = await chat_invoke_messages(messages)
    reply = response.content if isinstance(response.content, str) else str(response.content)
    return {"reply": reply, "messages": [AIMessage(content=reply)]}


async def node_persist(state: SaheliState, session: AsyncSession) -> dict:
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    conversation_id = uuid.UUID(state["conversation_id"])

    is_legacy_checkin = "aaj ka check-in shuru" in state["user_message"].lower()
    incoming_role = MessageRole.system if is_legacy_checkin else MessageRole.elder

    elder_msg = Message(
        conversation_id=conversation_id,
        family_id=family_id,
        elder_id=elder_id,
        role=incoming_role,
        content=state["user_message"] if not is_legacy_checkin else "Check-in started",
        metadata_={"kind": "check_in"} if is_legacy_checkin else None,
    )
    saheli_msg = Message(
        conversation_id=conversation_id,
        family_id=family_id,
        elder_id=elder_id,
        role=MessageRole.saheli,
        content=state["reply"],
    )
    session.add(elder_msg)
    session.add(saheli_msg)
    await session.flush()

    if not is_legacy_checkin and incoming_role == MessageRole.elder:
        await process_elder_message_memories(
            session,
            family_id=family_id,
            elder_id=elder_id,
            message=state["user_message"],
            source_message_id=elder_msg.id,
        )

    conv = await session.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = datetime.now(timezone.utc)

    await session.commit()
    return {}


def build_saheli_graph(session: AsyncSession):
    graph = StateGraph(SaheliState)

    async def retrieve_node(state: SaheliState):
        return await node_retrieve(state, session)

    async def persist_node(state: SaheliState):
        return await node_persist(state, session)

    async def generate_node(state: SaheliState):
        return await node_generate(state, session)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("persist", persist_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


async def run_saheli_chat(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    companion_profile: dict[str, Any] | None = None,
    care_record_context: str | None = None,
    order_context: str | None = None,
) -> str:
    app = build_saheli_graph(session)
    result = await app.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "family_id": str(family_id),
            "elder_id": str(elder_id),
            "conversation_id": str(conversation_id),
            "user_message": message,
            "rag_context": "",
            "family_memories": "",
            "recent_chat": "",
            "companion_profile": companion_profile or {},
            "care_record_context": care_record_context or "",
            "order_context": order_context or "",
            "reply": "",
        }
    )
    return result["reply"]


class CaregiverSaheliState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    family_id: str
    elder_id: str
    conversation_id: str
    elder_display_name: str
    user_message: str
    rag_context: str
    family_memories: str
    elder_thread: str
    recent_chat: str
    care_record_context: str
    platform_elder_thread: str
    platform_labs: str
    session_context: str
    reply: str


async def node_caregiver_retrieve(state: CaregiverSaheliState, session: AsyncSession) -> dict:
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    conversation_id = uuid.UUID(state["conversation_id"])
    query = state["user_message"]

    chunks = await retrieve_context(session, family_id=family_id, elder_id=elder_id, query=query)
    rag_lines = [f"[{c.kind}:{c.source}] {c.content}" for c in chunks]
    rag_context = "\n".join(rag_lines) if rag_lines else "(No matching memory yet.)"

    memories = await retrieve_family_memories(
        session,
        family_id=family_id,
        elder_id=elder_id,
        query=query,
        limit=12,
        shareable_only=False,
    )
    family_memories = format_family_memories(memories)

    primary = await get_primary_conversation(session, family_id, elder_id)
    elder_thread = await get_elder_thread_context(
        session,
        family_id=family_id,
        elder_id=elder_id,
        conversation_id=primary.id,
    )
    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=16,
        conversation_id=conversation_id,
    )
    recent_chat = "\n".join(f"{role}: {content}" for role, content, _ts in recent)

    return {
        "rag_context": rag_context,
        "family_memories": family_memories,
        "elder_thread": elder_thread,
        "recent_chat": recent_chat,
    }


async def node_caregiver_generate(state: CaregiverSaheliState) -> dict:
    elder_name = state["elder_display_name"]
    platform_block = ""
    if state.get("care_record_context"):
        platform_block += f"\n- Kavach care timeline:\n{state['care_record_context'][:4000]}"
    if state.get("platform_elder_thread"):
        platform_block += f"\n- Elder messages (platform):\n{state['platform_elder_thread'][:2000]}"
    if state.get("platform_labs"):
        platform_block += f"\n- Saved lab documents:\n{state['platform_labs'][:3000]}"
    if state.get("session_context"):
        platform_block += f"\n- This dashboard chat session:\n{state['session_context'][:1500]}"

    system = f"""{CAREGIVER_SAHELI_SYSTEM}

Care recipient: {elder_name}

Reference data (use only when relevant to the caregiver's question):
- Family memories: {state["family_memories"]}
- Documents & labs (RAG): {state["rag_context"]}
- Recent elder Saheli thread: {state["elder_thread"]}
- Caregiver chat history: {state["recent_chat"]}
{platform_block}
"""
    reply = await chat_invoke(system, state["user_message"])
    return {"reply": reply, "messages": [AIMessage(content=reply)]}


async def node_caregiver_persist(state: CaregiverSaheliState, session: AsyncSession) -> dict:
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    conversation_id = uuid.UUID(state["conversation_id"])

    session.add(
        Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.family,
            content=state["user_message"],
        )
    )
    session.add(
        Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.saheli,
            content=state["reply"],
        )
    )

    conv = await session.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = datetime.now(timezone.utc)

    await session.commit()
    return {}


def build_caregiver_saheli_graph(session: AsyncSession):
    graph = StateGraph(CaregiverSaheliState)

    async def retrieve_node(state: CaregiverSaheliState):
        return await node_caregiver_retrieve(state, session)

    async def persist_node(state: CaregiverSaheliState):
        return await node_caregiver_persist(state, session)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", node_caregiver_generate)
    graph.add_node("persist", persist_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


async def run_saheli_caregiver_chat(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    care_record_context: str | None = None,
    elder_thread_context: str | None = None,
    labs_context: str | None = None,
    session_context: str | None = None,
    order_context: str | None = None,
    use_agent: bool = True,
    actor_user_id: str | None = None,
) -> dict:
    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=16,
        conversation_id=conversation_id,
    )
    history = db_messages_to_langchain(recent)

    if use_agent and actor_user_id:
        agent_result = await run_caregiver_agent(
            session,
            family_id=family_id,
            elder_id=elder_id,
            message=message,
            care_record_context=care_record_context,
            elder_thread_context=elder_thread_context,
            labs_context=labs_context,
            session_context=session_context,
            order_context=order_context,
            history_messages=history,
            actor_user_id=actor_user_id,
        )
        reply = agent_result["reply"]
        session.add(
            Message(
                conversation_id=conversation_id,
                family_id=family_id,
                elder_id=elder_id,
                role=MessageRole.family,
                content=message,
            )
        )
        session.add(
            Message(
                conversation_id=conversation_id,
                family_id=family_id,
                elder_id=elder_id,
                role=MessageRole.saheli,
                content=reply,
            )
        )
        conv = await session.get(Conversation, conversation_id)
        if conv:
            conv.updated_at = datetime.now(timezone.utc)
        await session.commit()
        return agent_result

    elder = await session.get(Elder, elder_id)
    elder_display_name = elder.display_name if elder else "Care recipient"

    app = build_caregiver_saheli_graph(session)
    result = await app.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "family_id": str(family_id),
            "elder_id": str(elder_id),
            "conversation_id": str(conversation_id),
            "elder_display_name": elder_display_name,
            "user_message": message,
            "rag_context": "",
            "family_memories": "",
            "elder_thread": "",
            "recent_chat": "",
            "care_record_context": care_record_context or "",
            "platform_elder_thread": elder_thread_context or "",
            "platform_labs": labs_context or "",
            "session_context": session_context or "",
            "reply": "",
        }
    )
    return {"reply": result["reply"], "order": None, "connect": None, "tool_trace": []}


def _format_schedule_lines(schedule_items: list[dict]) -> str:
    if not schedule_items:
        return "(No care items scheduled today.)"
    lines: list[str] = []
    for item in schedule_items:
        title = str(item.get("title") or "Care item").strip()
        time = str(item.get("time") or "").strip()
        dosage = str(item.get("dosage") or "").strip()
        kind = str(item.get("type") or "").strip()
        bit = f"- {title}"
        if time:
            bit += f" · {time}"
        if dosage:
            bit += f" · {dosage}"
        if kind:
            bit += f" ({kind})"
        lines.append(bit)
    return "\n".join(lines)


def _pick_outreach_topic(outreach_topics: list[str] | None = None) -> tuple[str, str]:
    pool = outreach_topics or list(OUTREACH_TOPIC_HINTS.keys())
    bucket = random.choice(pool) if pool else "day_life"
    hints = OUTREACH_TOPIC_HINTS.get(bucket, OUTREACH_TOPIC_HINTS["day_life"])
    return bucket, random.choice(hints)


async def run_saheli_check_in(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
    schedule_items: list[dict] | None = None,
    care_record_context: str | None = None,
    companion_profile: dict[str, Any] | None = None,
) -> str:
    items = schedule_items or []
    schedule_block = _format_schedule_lines(items)
    elder = await session.get(Elder, elder_id)
    display_name = elder.display_name if elder else "Care recipient"

    chunks = await retrieve_context(
        session, family_id=family_id, elder_id=elder_id, query="check-in medicines today"
    )
    rag_lines = [f"[{c.kind}:{c.source}] {c.content}" for c in chunks]
    rag_context = "\n".join(rag_lines) if rag_lines else "(No matching memory yet.)"

    memories = await retrieve_family_memories(
        session, family_id=family_id, elder_id=elder_id, query="today health mood", limit=8
    )
    family_memories = format_family_memories(memories)

    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=12,
        conversation_id=conversation_id,
    )
    recent_chat = "\n".join(f"{role}: {content}" for role, content, _ts in recent)

    system = build_outreach_system_prompt(
        elder_display_name=display_name,
        rag_context=rag_context,
        family_memories=family_memories,
        recent_chat=recent_chat,
        topic_hint="Today's care list — ask warmly in Hinglish",
        companion_profile=companion_profile,
        schedule_block=schedule_block,
        care_record_context=care_record_context,
        outreach_kind="care",
    )

    reply = await chat_invoke(
        system,
        "Start today's care check-in. Ask about the care list warmly. Do not speak as the elder.",
    )

    system_note = "Care check-in started"
    if items:
        titles = ", ".join(str(i.get("title") or "item") for i in items[:8])
        system_note = f"Care check-in started · today's list: {titles}"

    session.add(
        Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.system,
            content=system_note,
            metadata_={"kind": "check_in", "outreach_kind": "care"},
        )
    )
    session.add(
        Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.saheli,
            content=reply,
        )
    )
    conv = await session.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return reply


async def run_saheli_outreach(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
    outreach_kind: str = "casual",
    topic_bucket: str | None = None,
    topic_hint: str | None = None,
    care_record_context: str | None = None,
    companion_profile: dict[str, Any] | None = None,
    schedule_items: list[dict] | None = None,
) -> dict[str, Any]:
    """Proactive outreach — Saheli initiates like a child calling to chat."""
    elder = await session.get(Elder, elder_id)
    display_name = elder.display_name if elder else "Care recipient"
    profile = companion_profile or {}

    if topic_bucket and topic_hint:
        bucket, hint = topic_bucket, topic_hint
    else:
        topics = profile.get("outreach_topics") or profile.get("outreachTopics")
        bucket, hint = _pick_outreach_topic(topics if isinstance(topics, list) else None)

    query = f"{bucket} {hint} day life family"
    chunks = await retrieve_context(session, family_id=family_id, elder_id=elder_id, query=query)
    rag_lines = [f"[{c.kind}:{c.source}] {c.content}" for c in chunks]
    rag_context = "\n".join(rag_lines) if rag_lines else "(No matching memory yet.)"

    memories = await retrieve_family_memories(
        session, family_id=family_id, elder_id=elder_id, query=query, limit=10
    )
    family_memories = format_family_memories(memories)

    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=12,
        conversation_id=conversation_id,
    )
    recent_chat = "\n".join(f"{role}: {content}" for role, content, _ts in recent)

    schedule_block = _format_schedule_lines(schedule_items or []) if outreach_kind == "mixed" else None

    system = build_outreach_system_prompt(
        elder_display_name=display_name,
        rag_context=rag_context,
        family_memories=family_memories,
        recent_chat=recent_chat,
        topic_hint=hint,
        companion_profile=profile,
        schedule_block=schedule_block,
        care_record_context=care_record_context,
        outreach_kind=outreach_kind,
    )

    reply = await chat_invoke(
        system,
        f"Reach out to {display_name} now. Start a warm conversation about: {hint}",
    )

    session.add(
        Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.system,
            content=f"Saheli reached out · {bucket}: {hint}",
            metadata_={"kind": "outreach", "outreach_kind": outreach_kind, "topic_bucket": bucket},
        )
    )
    session.add(
        Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.saheli,
            content=reply,
        )
    )
    conv = await session.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = datetime.now(timezone.utc)
    await session.commit()

    return {
        "reply": reply,
        "topic_bucket": bucket,
        "topic_hint": hint,
        "outreach_kind": outreach_kind,
    }


async def run_saheli_family_share(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    share_summary: str,
    memory_ids: list[uuid.UUID] | None = None,
) -> str:
    """Post a family update to the caregiver thread — initiates family conversation."""
    elder = await session.get(Elder, elder_id)
    display_name = elder.display_name if elder else "Care recipient"

    memories = await retrieve_family_memories(
        session, family_id=family_id, elder_id=elder_id, limit=8, shareable_only=True
    )
    if memory_ids:
        memories = [m for m in memories if m.id in memory_ids] or memories

    family_memories = format_family_memories(memories)
    system = build_family_share_system_prompt(
        elder_display_name=display_name,
        share_summary=share_summary,
        family_memories=family_memories,
    )
    reply = await chat_invoke(system, "Write the family update message.")

    caregiver_conv = await get_caregiver_conversation(session, family_id, elder_id)
    session.add(
        Message(
            conversation_id=caregiver_conv.id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.system,
            content=f"Family update · from {display_name}'s conversation",
            metadata_={"kind": "family_share"},
        )
    )
    session.add(
        Message(
            conversation_id=caregiver_conv.id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.saheli,
            content=reply,
            metadata_={"kind": "family_share"},
        )
    )

    now = datetime.now(timezone.utc)
    for mem in memories:
        if mem.share_with_family and not mem.shared_at:
            mem.shared_at = now

    caregiver_conv.updated_at = now
    await session.commit()
    return reply
