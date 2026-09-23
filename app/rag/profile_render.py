"""Render memory_profiles.body_md (Instinct PROFILE.md)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompt_fence import fence_memory_block, new_memory_nonce
from app.models.entities import (
    EntityStatus,
    FamilyMemory,
    MemoryEntity,
    MemoryProfile,
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


async def render_memory_profile(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    life_context: str = "",
    care_context: str = "",
    autonomy_rules: str = "",
) -> MemoryProfile:
    now = datetime.now(timezone.utc)
    entities = (
        await session.execute(
            select(MemoryEntity)
            .where(
                MemoryEntity.family_id == family_id,
                MemoryEntity.elder_id == elder_id,
                MemoryEntity.status.in_([EntityStatus.active.value, EntityStatus.needs_review.value]),
            )
            .order_by(MemoryEntity.updated_at.desc())
            .limit(40)
        )
    ).scalars().all()

    inbox = (
        await session.execute(
            select(FamilyMemory)
            .where(
                FamilyMemory.family_id == family_id,
                FamilyMemory.elder_id == elder_id,
                FamilyMemory.entity_id.is_(None),
                FamilyMemory.forgotten_at.is_(None),
                FamilyMemory.superseded_by.is_(None),
            )
            .order_by(FamilyMemory.created_at.desc())
            .limit(20)
        )
    ).scalars().all()

    sections: list[str] = [
        f"Rendered {now.strftime('%Y-%m-%d %H:%M')} UTC. Facts newer than this are in the inbox block below.",
        "",
        "## Life context",
        life_context.strip() or "(No life context supplied.)",
        "",
        "## What is going on now",
    ]

    if care_context.strip():
        sections.append(care_context.strip()[:2000])
    else:
        for entity in entities[:8]:
            first_line = (entity.body_md or "").split("\n")
            gist = next((ln for ln in first_line if ln.startswith("-")), entity.title)
            sections.append(f"- {entity.slug} ({entity.kind}): {gist[:120]}")

    sections.extend(
        [
            "",
            "## Autonomy calibration",
            autonomy_rules.strip()
            or "- Saheli may log check-ins and schedule completions alone.\n"
            "- Orders and health changes need caregiver awareness.",
            "",
            "## Entity index",
        ]
    )
    for entity in entities:
        gist = entity.title[:80]
        sections.append(f"- `{entity.slug}` | {entity.kind} | {gist}")
    sections.append("")
    sections.append("Use memory_grep before answering about people, medicines, or history.")

    if inbox:
        sections.extend(["", "## Inbox (not yet consolidated)"])
        for fact in inbox:
            sections.append(f"- [{fact.category.value}] {fact.content[:200]}")

    body = "\n".join(sections)[:12000]
    nonce = new_memory_nonce()
    fenced = fence_memory_block(body, nonce)

    existing = (
        await session.execute(
            select(MemoryProfile).where(
                MemoryProfile.family_id == family_id,
                MemoryProfile.elder_id == elder_id,
            )
        )
    ).scalar_one_or_none()

    if existing:
        existing.body_md = fenced
        existing.token_estimate = _estimate_tokens(body)
        existing.rendered_at = now
        profile = existing
    else:
        profile = MemoryProfile(
            family_id=family_id,
            elder_id=elder_id,
            body_md=fenced,
            token_estimate=_estimate_tokens(body),
            rendered_at=now,
        )
        session.add(profile)

    await session.flush()
    return profile


async def bootstrap_memory_onepager(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> str:
    """Generate an on-the-fly memory one-pager from inbox facts when no profile exists.

    This ensures every elder WA turn gets a non-empty memory context even if the
    nightly dream job hasn't run yet.
    """
    entities = (
        await session.execute(
            select(MemoryEntity)
            .where(
                MemoryEntity.family_id == family_id,
                MemoryEntity.elder_id == elder_id,
                MemoryEntity.status.in_([EntityStatus.active.value, EntityStatus.needs_review.value]),
            )
            .order_by(MemoryEntity.updated_at.desc())
            .limit(20)
        )
    ).scalars().all()

    inbox = (
        await session.execute(
            select(FamilyMemory)
            .where(
                FamilyMemory.family_id == family_id,
                FamilyMemory.elder_id == elder_id,
                FamilyMemory.entity_id.is_(None),
                FamilyMemory.forgotten_at.is_(None),
                FamilyMemory.superseded_by.is_(None),
            )
            .order_by(FamilyMemory.importance.desc(), FamilyMemory.created_at.desc())
            .limit(15)
        )
    ).scalars().all()

    if not entities and not inbox:
        return "(Nothing saved yet — this is our first conversation.)"

    sections: list[str] = ["## What Saheli knows"]

    if entities:
        sections.append("")
        sections.append("### Saved entities")
        for entity in entities[:10]:
            first_line = (entity.body_md or "").split("\n")
            gist = next((ln for ln in first_line if ln.startswith("-")), entity.title)
            sections.append(f"- `{entity.slug}` ({entity.kind}): {gist[:100]}")

    if inbox:
        sections.append("")
        sections.append("### Recent facts (not yet consolidated)")
        for fact in inbox:
            cat = fact.category.value if hasattr(fact.category, "value") else str(fact.category)
            sections.append(f"- [{cat}] {fact.content[:150]}")

    sections.append("")
    sections.append("Use memory_grep before answering about people, medicines, or history.")

    body = "\n".join(sections)
    nonce = new_memory_nonce()
    return fence_memory_block(body, nonce)


async def load_memory_profile_text(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> str:
    """Load the rendered memory profile, or bootstrap one on-the-fly if none exists."""
    row = (
        await session.execute(
            select(MemoryProfile).where(
                MemoryProfile.family_id == family_id,
                MemoryProfile.elder_id == elder_id,
            )
        )
    ).scalar_one_or_none()

    if row and row.body_md and row.body_md.strip():
        return row.body_md

    return await bootstrap_memory_onepager(
        session, family_id=family_id, elder_id=elder_id
    )
