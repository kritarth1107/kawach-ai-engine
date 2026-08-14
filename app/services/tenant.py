"""Tenant isolation — every read/write must stay inside one family."""

import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Conversation, Elder, Family, FamilyMember, User


class TenantError(HTTPException):
    def __init__(self, detail: str = "Resource not found in this family") -> None:
        super().__init__(status_code=404, detail=detail)


async def assert_family_exists(session: AsyncSession, family_id: uuid.UUID) -> Family:
    family = await session.get(Family, family_id)
    if not family:
        raise TenantError("Family not found")
    return family


async def assert_elder_in_family(
    session: AsyncSession,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> Elder:
    elder = await session.get(Elder, elder_id)
    if not elder or elder.family_id != family_id:
        raise TenantError("Elder not found in this family")
    return elder


async def assert_conversation_in_family(
    session: AsyncSession,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> Conversation:
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.family_id != family_id or conv.elder_id != elder_id:
        raise TenantError("Conversation not found in this family")
    return conv


async def assert_user_in_family(
    session: AsyncSession,
    family_id: uuid.UUID,
    external_user_id: str,
) -> FamilyMember:
    stmt = (
        select(FamilyMember)
        .join(User, User.id == FamilyMember.user_id)
        .where(User.external_id == external_user_id, FamilyMember.family_id == family_id)
    )
    member = (await session.execute(stmt)).scalar_one_or_none()
    if not member:
        raise TenantError("User is not a member of this family")
    return member


async def scope_chat_request(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
) -> Conversation:
    """Validate tenant boundary before any chat/RAG work."""
    await assert_family_exists(session, family_id)
    await assert_elder_in_family(session, family_id, elder_id)

    if conversation_id:
        return await assert_conversation_in_family(
            session, family_id, elder_id, conversation_id
        )

    from app.services.family import get_primary_conversation

    return await get_primary_conversation(session, family_id, elder_id)


async def scope_caregiver_chat_request(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
) -> Conversation:
    """Validate tenant boundary for family-member Saheli chat."""
    await assert_family_exists(session, family_id)
    await assert_elder_in_family(session, family_id, elder_id)

    if conversation_id:
        conv = await assert_conversation_in_family(
            session, family_id, elder_id, conversation_id
        )
        if conv.channel != "saheli-family":
            raise TenantError("Not a caregiver conversation")
        return conv

    from app.services.family import get_caregiver_conversation

    return await get_caregiver_conversation(session, family_id, elder_id)


async def scope_document_request(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID | None,
) -> None:
    """Validate tenant boundary before document ingest/search."""
    await assert_family_exists(session, family_id)
    if elder_id is not None:
        await assert_elder_in_family(session, family_id, elder_id)
