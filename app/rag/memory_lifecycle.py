"""Forget, correct, and erase memory facts (DPDP)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import FamilyMemory, MemoryEntity, MemoryProfile
from app.rag.embeddings import embed_text, embeddings_available
from app.rag.memory_extract import persist_inbox_memory
from app.rag.memory_hash import content_hash
from app.rag.profile_render import render_memory_profile


async def forget_memory_fact(
    session: AsyncSession,
    *,
    fact_id: uuid.UUID,
    forgotten_by: str | None = None,
) -> dict:
    fact = await session.get(FamilyMemory, fact_id)
    if not fact or fact.forgotten_at:
        return {"forgotten": False, "reason": "not_found"}

    now = datetime.now(timezone.utc)
    fact.forgotten_at = now
    fact.forgotten_by = forgotten_by

    if fact.entity_id:
        entity = await session.get(MemoryEntity, fact.entity_id)
        if entity and entity.sources:
            sources = [s for s in entity.sources if str(s) != str(fact.id)]
            entity.sources = sources
            entity.dirty = True

    await session.flush()
    await render_memory_profile(
        session,
        family_id=fact.family_id,
        elder_id=fact.elder_id,
    )
    await session.commit()
    return {"forgotten": True, "fact_id": str(fact.id)}


async def correct_memory_fact(
    session: AsyncSession,
    *,
    fact_id: uuid.UUID,
    replacement_content: str,
    actor_user_id: str | None = None,
    source_role: str = "family",
) -> dict:
    old = await session.get(FamilyMemory, fact_id)
    if not old or old.forgotten_at:
        return {"corrected": False, "reason": "not_found"}

    new_fact = await persist_inbox_memory(
        session,
        family_id=old.family_id,
        elder_id=old.elder_id,
        content=replacement_content,
        category=old.category.value if hasattr(old.category, "value") else str(old.category),
        topic=old.topic,
        source_role=source_role,
        share_with_family=old.share_with_family,
        importance=old.importance,
        commit=False,
    )
    old.superseded_by = new_fact.id

    if old.entity_id:
        entity = await session.get(MemoryEntity, old.entity_id)
        if entity:
            entity.dirty = True
            if entity.kind in ("medication", "condition", "symptom"):
                entity.body_md += f"\n- **Correction:** {replacement_content[:400]}"
                if embeddings_available():
                    try:
                        entity.body_embedding = await embed_text(entity.body_md[:8000])
                    except Exception:
                        pass

    await session.commit()
    return {"corrected": True, "old_id": str(old.id), "new_id": str(new_fact.id)}


async def erase_elder_memory(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> dict:
    await session.execute(
        update(FamilyMemory)
        .where(FamilyMemory.family_id == family_id, FamilyMemory.elder_id == elder_id)
        .values(forgotten_at=datetime.now(timezone.utc), forgotten_by="system_erase")
    )
    entities = (
        await session.execute(
            select(MemoryEntity).where(
                MemoryEntity.family_id == family_id,
                MemoryEntity.elder_id == elder_id,
            )
        )
    ).scalars().all()
    for entity in entities:
        await session.delete(entity)

    profile = (
        await session.execute(
            select(MemoryProfile).where(
                MemoryProfile.family_id == family_id,
                MemoryProfile.elder_id == elder_id,
            )
        )
    ).scalar_one_or_none()
    if profile:
        profile.body_md = ""
        profile.rendered_at = None

    await session.commit()
    return {"erased": True, "entities_removed": len(entities)}
