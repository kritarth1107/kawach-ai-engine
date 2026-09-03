import uuid
from typing import Annotated, TypedDict

from datetime import datetime, timezone

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.provider import chat_invoke
from app.models.entities import Conversation, Elder, Message, MessageRole
from app.rag.ingest import ingest_chat_snippet
from app.rag.embeddings import embeddings_available
from app.rag.retrieve import get_elder_thread_context, get_recent_messages, retrieve_context
from app.services.family import get_primary_conversation

SAHELI_SYSTEM = """You are Saheli (सहेली) — a warm companion for Indian elders.

The elder may write Hindi, English, or Hinglish. Examples:
- "Maine Shelcal le liya" = they took Shelcal (their tablet).
- "theek hoon" = they feel okay.

Rules (non-negotiable):
- Report what the elder said faithfully. Never diagnose, interpret health, or invent facts.
- Acknowledge medicines they name only as something they reported taking — not as a clinical event you verified.
- You are a companion, not a clinician or monitor.
"""


class SaheliState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    family_id: str
    elder_id: str
    conversation_id: str
    user_message: str
    rag_context: str
    recent_chat: str
    reply: str
    save_snippet: bool


async def node_retrieve(state: SaheliState, session: AsyncSession) -> dict:
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    query = state["user_message"]

    chunks = await retrieve_context(session, family_id=family_id, elder_id=elder_id, query=query)
    rag_lines = [f"[{c.kind}:{c.source}] {c.content}" for c in chunks]
    rag_context = "\n".join(rag_lines) if rag_lines else "(No matching memory yet.)"

    primary = await get_primary_conversation(session, family_id, elder_id)
    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=16,
        conversation_id=primary.id,
    )
    recent_chat = "\n".join(f"{role}: {content}" for role, content, _ts in recent)

    return {"rag_context": rag_context, "recent_chat": recent_chat}


async def node_generate(state: SaheliState) -> dict:
    system = f"""{SAHELI_SYSTEM}

Retrieved family memory (RAG — reported only):
{state["rag_context"]}

Recent cross-chat history for this elder:
{state["recent_chat"]}
"""
    reply = await chat_invoke(system, state["user_message"])
    save = len(state["user_message"].strip()) >= 12 and len(state["user_message"]) < 200
    return {"reply": reply, "save_snippet": save, "messages": [AIMessage(content=reply)]}


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

    if state.get("save_snippet") and embeddings_available() and not is_legacy_checkin:
        await ingest_chat_snippet(
            session,
            family_id=family_id,
            elder_id=elder_id,
            content=state["user_message"],
            source_message_id=elder_msg.id,
            commit=False,
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

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", node_generate)
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
            "recent_chat": "",
            "reply": "",
            "save_snippet": False,
        }
    )
    return result["reply"]


CAREGIVER_SAHELI_SYSTEM = """You are Saheli (सहेली) — a family companion. The person messaging you is a caregiver, not the elder.

You DO have family memory in this request: retrieved documents and what the elder told Saheli. That is not a hospital EMR. It is text the family pasted or the elder said.

Rules (non-negotiable):
- Never diagnose or say if a lab is high/low/normal.
- If retrieved memory includes a printed lab value, quote the title, date, and numbers. Do not refuse. Do not say you cannot access records.
- If the elder reported taking a medicine or how they feel, repeat that as reported — not as a verified medical event.
- If memory is empty, say you have not heard from them yet. Do not invent.
- Use Hindi, English, or Hinglish naturally.
"""


class CaregiverSaheliState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    family_id: str
    elder_id: str
    conversation_id: str
    elder_display_name: str
    user_message: str
    rag_context: str
    elder_thread: str
    recent_chat: str
    reply: str


async def node_caregiver_retrieve(state: CaregiverSaheliState, session: AsyncSession) -> dict:
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    conversation_id = uuid.UUID(state["conversation_id"])
    query = state["user_message"]

    chunks = await retrieve_context(session, family_id=family_id, elder_id=elder_id, query=query)
    rag_lines = [f"[{c.kind}:{c.source}] {c.content}" for c in chunks]
    rag_context = "\n".join(rag_lines) if rag_lines else "(No matching memory yet.)"

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

    return {"rag_context": rag_context, "elder_thread": elder_thread, "recent_chat": recent_chat}


async def node_caregiver_generate(state: CaregiverSaheliState) -> dict:
    elder_name = state["elder_display_name"]
    system = f"""{CAREGIVER_SAHELI_SYSTEM}

Care recipient: {elder_name}

Retrieved family memory (RAG — reported only):
{state["rag_context"]}

What {elder_name} has reported to Saheli recently:
{state["elder_thread"]}

Your recent conversation with this family member:
{state["recent_chat"]}
"""
    reply = await chat_invoke(system, state["user_message"])
    return {"reply": reply, "messages": [AIMessage(content=reply)]}


async def node_caregiver_persist(state: CaregiverSaheliState, session: AsyncSession) -> dict:
    family_id = uuid.UUID(state["family_id"])
    elder_id = uuid.UUID(state["elder_id"])
    conversation_id = uuid.UUID(state["conversation_id"])

    family_msg = Message(
        conversation_id=conversation_id,
        family_id=family_id,
        elder_id=elder_id,
        role=MessageRole.family,
        content=state["user_message"],
    )
    saheli_msg = Message(
        conversation_id=conversation_id,
        family_id=family_id,
        elder_id=elder_id,
        role=MessageRole.saheli,
        content=state["reply"],
    )
    session.add(family_msg)
    session.add(saheli_msg)

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
) -> str:
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
            "elder_thread": "",
            "recent_chat": "",
            "reply": "",
        }
    )
    return result["reply"]


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


async def run_saheli_check_in(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
    schedule_items: list[dict] | None = None,
    care_record_context: str | None = None,
) -> str:
    """Prompt Papa about today's list. Never stored as an elder message."""
    items = schedule_items or []
    schedule_block = _format_schedule_lines(items)

    chunks = await retrieve_context(
        session, family_id=family_id, elder_id=elder_id, query="check-in medicines today"
    )
    rag_lines = [f"[{c.kind}:{c.source}] {c.content}" for c in chunks]
    rag_context = "\n".join(rag_lines) if rag_lines else "(No matching memory yet.)"

    recent = await get_recent_messages(
        session,
        family_id=family_id,
        elder_id=elder_id,
        limit=12,
        conversation_id=conversation_id,
    )
    recent_chat = "\n".join(f"{role}: {content}" for role, content, _ts in recent)

    system = f"""{SAHELI_SYSTEM}

You are starting a scheduled check-in. The elder has NOT spoken yet.
Do not invent that they already took medicines or feel a certain way.
Do not write as the elder.

Care Record timeline (reported only):
{care_record_context or "(No Care Record events yet.)"}

Today's care list — ask haan/nahi, warmly, in Hinglish. Medicines first if present:
{schedule_block}

Retrieved family memory (RAG — reported only):
{rag_context}

Recent thread (for tone only):
{recent_chat}
"""
    reply = await chat_invoke(
        system,
        "Start today's check-in. Ask about the care list. Do not speak as the elder.",
    )

    system_note = "Check-in started"
    if items:
        titles = ", ".join(str(i.get("title") or "item") for i in items[:8])
        system_note = f"Check-in started · today's list: {titles}"

    session.add(
        Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=MessageRole.system,
            content=system_note,
            metadata_={"kind": "check_in"},
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
