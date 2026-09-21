"""Background memory extraction — off the elder reply hot path."""

from __future__ import annotations

import asyncio
import logging
import uuid

from app.db.session import SessionLocal
from app.rag.memory_extract import process_elder_message_memories

logger = logging.getLogger(__name__)


async def run_memory_extract_task(
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    source_message_id: uuid.UUID | None = None,
    source_role: str = "elder",
) -> None:
    try:
        async with SessionLocal() as session:
            await process_elder_message_memories(
                session,
                family_id=family_id,
                elder_id=elder_id,
                message=message,
                source_message_id=source_message_id,
                source_role=source_role,
            )
            await session.commit()
    except Exception:
        logger.exception(
            "Memory extract failed family=%s elder=%s",
            family_id,
            elder_id,
        )


def schedule_memory_extract(
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    source_message_id: uuid.UUID | None = None,
    source_role: str = "elder",
) -> None:
    asyncio.create_task(
        run_memory_extract_task(
            family_id=family_id,
            elder_id=elder_id,
            message=message,
            source_message_id=source_message_id,
            source_role=source_role,
        ),
    )
