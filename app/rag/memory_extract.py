"""Extract structured family memories from elder messages."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.provider import chat_invoke
from app.models.entities import FamilyMemory, MemoryCategory
from app.rag.embeddings import embed_text, embeddings_available
from app.rag.ingest import ingest_chat_snippet

EXTRACT_SYSTEM = """You extract durable facts from an elder's message to Saheli (their child-like companion).

Return ONLY valid JSON array (no markdown). Each item:
{
  "content": "short fact in third person, e.g. Mama enjoyed morning walk in the park",
  "category": "casual|health|family|preference|mood|story",
  "topic": "2-4 word topic slug",
  "share_with_family": true/false,
  "importance": 1-5
}

Rules:
- Extract only what they explicitly said. Never infer diagnosis or clinical interpretation.
- Casual life facts (food, TV, visitors, mood) → share_with_family true if wholesome/non-sensitive.
- Pure health numbers → category health, share_with_family false unless they asked to tell family.
- Empty array [] if nothing worth remembering.
- Max 3 items per message."""

MEMORY_CATEGORIES = {
    "casual",
    "health",
    "family",
    "preference",
    "mood",
    "story",
}


def _parse_extract_json(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and "memories" in data:
        data = data["memories"]
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data[:3]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if len(content) < 8:
            continue
        category = str(item.get("category") or "casual").lower()
        if category not in MEMORY_CATEGORIES:
            category = "casual"
        out.append(
            {
                "content": content[:500],
                "category": category,
                "topic": str(item.get("topic") or "general")[:64],
                "share_with_family": bool(item.get("share_with_family")),
                "importance": min(5, max(1, int(item.get("importance") or 3))),
            }
        )
    return out


async def extract_memories_from_message(message: str) -> list[dict[str, Any]]:
    if len(message.strip()) < 8:
        return []
    try:
        raw = await chat_invoke(
            EXTRACT_SYSTEM,
            f"Elder message:\n{message.strip()[:1200]}",
        )
        return _parse_extract_json(raw)
    except Exception:
        return []


async def persist_family_memories(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    memories: list[dict[str, Any]],
    source_message_id: uuid.UUID | None = None,
    commit: bool = True,
) -> list[FamilyMemory]:
    saved: list[FamilyMemory] = []
    for mem in memories:
        content = mem["content"]
        vector = None
        if embeddings_available():
            try:
                vector = await embed_text(content)
            except Exception:
                vector = None

        row = FamilyMemory(
            family_id=family_id,
            elder_id=elder_id,
            category=MemoryCategory(mem["category"]),
            topic=mem.get("topic") or "general",
            content=content,
            embedding=vector,
            source_message_id=source_message_id,
            share_with_family=bool(mem.get("share_with_family")),
            importance=int(mem.get("importance") or 3),
        )
        session.add(row)
        saved.append(row)

        # Also index as snippet for unified RAG retrieval
        if embeddings_available():
            await ingest_chat_snippet(
                session,
                family_id=family_id,
                elder_id=elder_id,
                content=content,
                source=f"memory:{mem['category']}",
                source_message_id=source_message_id,
                commit=False,
            )

    if commit:
        await session.commit()
        for row in saved:
            await session.refresh(row)
    else:
        await session.flush()
    return saved


async def process_elder_message_memories(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    source_message_id: uuid.UUID | None = None,
) -> list[FamilyMemory]:
    extracted = await extract_memories_from_message(message)
    if not extracted:
        # Fallback: save substantive casual messages as snippets
        if 12 <= len(message.strip()) <= 400 and embeddings_available():
            await ingest_chat_snippet(
                session,
                family_id=family_id,
                elder_id=elder_id,
                content=message.strip()[:500],
                source="elder_message",
                source_message_id=source_message_id,
                commit=False,
            )
        return []
    return await persist_family_memories(
        session,
        family_id=family_id,
        elder_id=elder_id,
        memories=extracted,
        source_message_id=source_message_id,
        commit=False,
    )
