import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Conversation, Elder, Family, FamilyMember, MemberRole, User


async def create_family_with_owner(
    session: AsyncSession,
    *,
    family_name: str,
    owner_external_id: str,
    owner_email: str | None = None,
    owner_name: str | None = None,
) -> tuple[Family, User]:
    user = User(external_id=owner_external_id, email=owner_email, name=owner_name)
    session.add(user)
    await session.flush()

    family = Family(name=family_name)
    session.add(family)
    await session.flush()

    session.add(
        FamilyMember(family_id=family.id, user_id=user.id, role=MemberRole.owner)
    )
    await session.commit()
    await session.refresh(family)
    await session.refresh(user)
    return family, user


async def add_family_member(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    external_id: str,
    email: str | None = None,
    name: str | None = None,
    role: MemberRole = MemberRole.family,
) -> FamilyMember:
    result = await session.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if not user:
        user = User(external_id=external_id, email=email, name=name)
        session.add(user)
        await session.flush()

    member = FamilyMember(family_id=family_id, user_id=user.id, role=role)
    session.add(member)
    await session.commit()
    await session.refresh(member)
    return member


async def create_elder(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    display_name: str,
    slug: str,
    preferred_language: str = "hinglish",
) -> Elder:
    elder = Elder(
        family_id=family_id,
        display_name=display_name,
        slug=slug,
        preferred_language=preferred_language,
    )
    session.add(elder)
    await session.flush()

    conv = Conversation(
        family_id=family_id,
        elder_id=elder.id,
        channel="saheli",
        title=f"{display_name} · Saheli",
        is_primary=True,
    )
    session.add(conv)
    await session.commit()
    await session.refresh(elder)
    return elder


async def get_primary_conversation(
    session: AsyncSession,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> Conversation:
    stmt = (
        select(Conversation)
        .where(
            Conversation.family_id == family_id,
            Conversation.elder_id == elder_id,
            Conversation.channel == "saheli",
            Conversation.is_primary.is_(True),
        )
        .limit(1)
    )
    conv = (await session.execute(stmt)).scalar_one_or_none()
    if conv:
        return conv

    elder = await session.get(Elder, elder_id)
    conv = Conversation(
        family_id=family_id,
        elder_id=elder_id,
        channel="saheli",
        title=f"{elder.display_name if elder else 'Elder'} · Saheli",
        is_primary=True,
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def get_caregiver_conversation(
    session: AsyncSession,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> Conversation:
    stmt = (
        select(Conversation)
        .where(
            Conversation.family_id == family_id,
            Conversation.elder_id == elder_id,
            Conversation.channel == "saheli-family",
        )
        .limit(1)
    )
    conv = (await session.execute(stmt)).scalar_one_or_none()
    if conv:
        return conv

    elder = await session.get(Elder, elder_id)
    conv = Conversation(
        family_id=family_id,
        elder_id=elder_id,
        channel="saheli-family",
        title=f"Family · {elder.display_name if elder else 'Elder'}",
        is_primary=False,
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def list_families_for_user(session: AsyncSession, external_id: str) -> list[Family]:
    stmt = (
        select(Family)
        .join(FamilyMember, FamilyMember.family_id == Family.id)
        .join(User, User.id == FamilyMember.user_id)
        .where(User.external_id == external_id)
    )
    return list((await session.execute(stmt)).scalars().all())
