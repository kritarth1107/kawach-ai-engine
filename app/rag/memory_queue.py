"""Background memory extraction — off the elder reply hot path.

For high-value chats (people, preferences, health), extraction runs eagerly
so the next turn can see new facts without waiting for the nightly dream job.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Literal

from app.db.session import SessionLocal
from app.rag.memory_extract import process_elder_message_memories

logger = logging.getLogger(__name__)

HIGH_VALUE_PATTERNS = [
    re.compile(r"\b(maa|papa|beti|beta|bhaiya|didi|chacha|mami|nana|nani|dada|dadi)\b", re.I),
    re.compile(r"\b(wife|husband|son|daughter|brother|sister|uncle|aunt|grandma|grandpa)\b", re.I),
    re.compile(r"\b(doctor|dr\.|hospital|clinic|medicine|tablet|dawai|goli)\b", re.I),
    re.compile(r"\b(bp|blood pressure|sugar|diabetes|heart|pain|dard|bukhar|fever)\b", re.I),
    re.compile(r"\b(favourite|favorite|pasand|like|love|hate|prefer)\b", re.I),
    re.compile(r"\b(birthday|anniversary|wedding|shadi|job|retire)\b", re.I),
    re.compile(r"\b(morning walk|yoga|exercise|temple|mandir|church|mosque)\b", re.I),
    re.compile(r"\b(osimertinib|tarceva|shelcal|thyronorm|metformin|aspirin)\b", re.I),
    re.compile(r"\b(passed away|guzar gaye|died|death|memory of)\b", re.I),
    re.compile(r"\b(allergic|allergy|cannot eat|nahi kha sakte)\b", re.I),
]


def is_high_value_message(message: str) -> bool:
    """Detect if a message contains high-value topics worth eager extraction.

    High-value topics include: people mentions, health/medicine, preferences,
    life events, and emotional content that should be remembered.
    """
    if len(message.strip()) < 12:
        return False

    text = message.lower()
    for pattern in HIGH_VALUE_PATTERNS:
        if pattern.search(text):
            return True

    return False


async def run_memory_extract_task(
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    source_message_id: uuid.UUID | None = None,
    source_role: str = "elder",
) -> list:
    """Run memory extraction and return persisted memories."""
    try:
        async with SessionLocal() as session:
            memories = await process_elder_message_memories(
                session,
                family_id=family_id,
                elder_id=elder_id,
                message=message,
                source_message_id=source_message_id,
                source_role=source_role,
            )
            await session.commit()
            return memories
    except Exception:
        logger.exception(
            "Memory extract failed family=%s elder=%s",
            family_id,
            elder_id,
        )
        return []


def schedule_memory_extract(
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    source_message_id: uuid.UUID | None = None,
    source_role: str = "elder",
    mode: Literal["background", "eager"] = "background",
) -> asyncio.Task | None:
    """Schedule memory extraction from an elder message.

    Args:
        mode: "background" (default) for fire-and-forget, "eager" returns the task
              so caller can await it for high-value chats.

    Returns:
        The asyncio.Task when mode="eager", None otherwise.
    """
    task = asyncio.create_task(
        run_memory_extract_task(
            family_id=family_id,
            elder_id=elder_id,
            message=message,
            source_message_id=source_message_id,
            source_role=source_role,
        ),
    )
    if mode == "eager":
        return task
    return None


async def eager_memory_extract(
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    source_message_id: uuid.UUID | None = None,
    source_role: str = "elder",
) -> list:
    """Run memory extraction eagerly and return persisted memories.

    Use this for high-value chats where we want the next turn to see new facts.
    This blocks until extraction completes, so use sparingly.
    """
    return await run_memory_extract_task(
        family_id=family_id,
        elder_id=elder_id,
        message=message,
        source_message_id=source_message_id,
        source_role=source_role,
    )
