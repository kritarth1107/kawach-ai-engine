"""Extract structured family memories from elder messages."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.provider import chat_invoke
from app.models.entities import FamilyMemory, MemoryCategory
from app.rag.embeddings import embed_text, embeddings_available
from app.rag.memory_hash import content_hash

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


async def persist_inbox_memory(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    content: str,
    category: str = "casual",
    topic: str = "general",
    source_message_id: uuid.UUID | None = None,
    source_role: str = "elder",
    share_with_family: bool = False,
    importance: int = 3,
    commit: bool = True,
) -> FamilyMemory:
    digest = content_hash(content)
    existing = (
        await session.execute(
            select(FamilyMemory).where(
                FamilyMemory.elder_id == elder_id,
                FamilyMemory.content_hash == digest,
                FamilyMemory.forgotten_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if existing:
        existing.last_referenced_at = datetime.now(timezone.utc)
        return existing

    vector = None
    if embeddings_available():
        try:
            vector = await embed_text(content)
        except Exception:
            vector = None

    row = FamilyMemory(
        family_id=family_id,
        elder_id=elder_id,
        category=MemoryCategory(category if category in MEMORY_CATEGORIES else "casual"),
        topic=topic[:64],
        content=content[:500],
        content_hash=digest,
        source_role=source_role[:32],
        embedding=vector,
        source_message_id=source_message_id,
        share_with_family=share_with_family,
        importance=importance,
        entity_id=None,
    )
    session.add(row)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def persist_family_memories(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    memories: list[dict[str, Any]],
    source_message_id: uuid.UUID | None = None,
    source_role: str = "elder",
    commit: bool = True,
) -> list[FamilyMemory]:
    saved: list[FamilyMemory] = []
    for mem in memories:
        row = await persist_inbox_memory(
            session,
            family_id=family_id,
            elder_id=elder_id,
            content=mem["content"],
            category=mem["category"],
            topic=mem.get("topic") or "general",
            source_message_id=source_message_id,
            source_role=source_role,
            share_with_family=bool(mem.get("share_with_family")),
            importance=int(mem.get("importance") or 3),
            commit=False,
        )
        saved.append(row)

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
    source_role: str = "elder",
) -> list[FamilyMemory]:
    extracted = await extract_memories_from_message(message)
    if not extracted:
        if 12 <= len(message.strip()) <= 400:
            row = await persist_inbox_memory(
                session,
                family_id=family_id,
                elder_id=elder_id,
                content=message.strip()[:500],
                category="casual",
                topic="chat",
                source_message_id=source_message_id,
                source_role=source_role,
                share_with_family=False,
                importance=2,
                commit=False,
            )
            return [row]
        return []
    return await persist_family_memories(
        session,
        family_id=family_id,
        elder_id=elder_id,
        memories=extracted,
        source_message_id=source_message_id,
        source_role=source_role,
        commit=False,
    )
