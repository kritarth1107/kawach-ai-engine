#!/usr/bin/env python3
"""Seed Papa's AI family, lab, and Saheli threads. Uses a long PG timeout."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.models.entities import (
    Conversation,
    DocumentKind,
    Elder,
    Family,
    FamilyMember,
    MemberRole,
    MemoryDocument,
    Message,
    MessageRole,
    User,
)

SEED = "rahul-demo"
OWNER_EXTERNAL_ID = "86b50c4f-a9b7-4a5e-aaf9-33c982e48e21"
ELDER_SLUG = "r-81c463479a8e42fdbbbb178e4a8d6a6a"


def ago(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


PAPA_THREAD: list[tuple[MessageRole, str, float]] = [
    (MessageRole.system, "aaj ka check-in shuru", 3.2),
    (MessageRole.saheli, "Papa, subah kaisi lagi?", 3.1),
    (MessageRole.elder, "theek hun", 3.0),
    (MessageRole.saheli, "dawai le li kya? Telmisartan nashta ke baad.", 2.9),
    (MessageRole.elder, "haan, nashta ke baad le li", 2.8),
    (MessageRole.saheli, "accha. shaam ko BP likhna hai, yaad dilati rahungi.", 2.7),
]

FAMILY_THREAD: list[tuple[MessageRole, str, float]] = [
    (MessageRole.family, "Papa ne aaj dawai li?", 1.5),
    (
        MessageRole.saheli,
        "Papa ne kaha nashta ke baad Telmisartan le li. BP abhi report nahi kiya.",
        1.4,
    ),
]


async def get_or_create_user(session: AsyncSession, external_id: str, name: str) -> User:
    user = (await session.execute(select(User).where(User.external_id == external_id))).scalar_one_or_none()
    if user:
        return user
    user = User(external_id=external_id, name=name)
    session.add(user)
    await session.flush()
    return user


async def get_or_create_family(session: AsyncSession, owner: User) -> Family:
    existing = (
        await session.execute(
            select(Family)
            .join(FamilyMember, FamilyMember.family_id == Family.id)
            .where(FamilyMember.user_id == owner.id, Family.name == "Rahul's Family")
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    family = Family(name="Rahul's Family")
    session.add(family)
    await session.flush()
    session.add(FamilyMember(family_id=family.id, user_id=owner.id, role=MemberRole.owner))
    await session.flush()
    return family


async def get_or_create_elder(session: AsyncSession, family: Family) -> Elder:
    elder = (
        await session.execute(
            select(Elder).where(Elder.family_id == family.id, Elder.slug == ELDER_SLUG)
        )
    ).scalar_one_or_none()
    if elder:
        return elder
    elder = Elder(
        family_id=family.id,
        display_name="Papa",
        slug=ELDER_SLUG,
        preferred_language="hinglish",
    )
    session.add(elder)
    await session.flush()
    return elder


async def get_or_create_conversation(
    session: AsyncSession,
    family: Family,
    elder: Elder,
    channel: str,
    title: str,
    is_primary: bool,
) -> Conversation:
    conv = (
        await session.execute(
            select(Conversation).where(
                Conversation.family_id == family.id,
                Conversation.elder_id == elder.id,
                Conversation.channel == channel,
            )
        )
    ).scalar_one_or_none()
    if conv:
        return conv
    conv = Conversation(
        family_id=family.id,
        elder_id=elder.id,
        channel=channel,
        title=title,
        is_primary=is_primary,
    )
    session.add(conv)
    await session.flush()
    return conv


async def seed_messages(
    session: AsyncSession,
    conv: Conversation,
    rows: list[tuple[MessageRole, str, float]],
) -> int:
    count = (
        await session.execute(
            select(Message.id).where(Message.conversation_id == conv.id).limit(1)
        )
    ).first()
    if count:
        return 0
    for role, content, hours in rows:
        session.add(
            Message(
                conversation_id=conv.id,
                family_id=conv.family_id,
                elder_id=conv.elder_id,
                role=role,
                content=content,
                metadata_={"seed": SEED},
                created_at=ago(hours),
            )
        )
    return len(rows)


async def seed_lab(session: AsyncSession, family: Family, elder: Elder) -> bool:
    existing = (
        await session.execute(
            select(MemoryDocument).where(
                MemoryDocument.family_id == family.id,
                MemoryDocument.elder_id == elder.id,
                MemoryDocument.title == "TSH report",
            )
        )
    ).scalar_one_or_none()
    if existing:
        return False
    session.add(
        MemoryDocument(
            family_id=family.id,
            elder_id=elder.id,
            kind=DocumentKind.lab,
            title="TSH report",
            source="upload",
            raw_text=(
                "Lab report — thyroid\n"
                "Patient: Papa\n"
                "Date: 8 Aug 2026\n\n"
                "TSH 4.2 mIU/L\n"
                "Free T4 1.1 ng/dL\n\n"
                "Printed values only. No interpretation added."
            ),
            record_date="8 Aug 2026",
        )
    )
    return True


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"timeout": 120},
    )
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as session:
        await session.execute(text("SELECT 1"))
        owner = await get_or_create_user(session, OWNER_EXTERNAL_ID, "Rahul")
        family = await get_or_create_family(session, owner)
        elder = await get_or_create_elder(session, family)
        papa_conv = await get_or_create_conversation(
            session, family, elder, "saheli", "Papa · Saheli", True
        )
        family_conv = await get_or_create_conversation(
            session, family, elder, "saheli-family", "Family · Papa", False
        )
        papa_added = await seed_messages(session, papa_conv, PAPA_THREAD)
        family_added = await seed_messages(session, family_conv, FAMILY_THREAD)
        lab_added = await seed_lab(session, family, elder)
        await session.commit()
        print(
            {
                "ai_family_id": str(family.id),
                "ai_elder_id": str(elder.id),
                "papa_conversation_id": str(papa_conv.id),
                "caregiver_conversation_id": str(family_conv.id),
                "papa_messages_added": papa_added,
                "caregiver_messages_added": family_added,
                "lab_added": lab_added,
            }
        )

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
