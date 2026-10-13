"""One memory is the truth (trust plan A1): the move from the old stores, the person view the backend reads, and the
nightly count of writes to the retired Instinct tables."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.core.security import verify_api_secret
from app.db.session import get_db

router = APIRouter(prefix="/v2", tags=["one-memory"], dependencies=[Depends(verify_api_secret)])


@router.post("/jobs/memory-unify")
async def memory_unify_job(apply: bool = False, family: str = "", undo: bool = False, store: str = "") -> dict:
    """Move what is still useful from the old stores into the care record (app.care.memory_unify), every family, one
    transaction each. apply=false (the default) is the dry run: the same checks, nothing written. undo=true takes the
    imports back through memory versions (with apply=true; a dry undo counts what it would take back). family= limits it to
    ids starting with that; store= limits an undo to one old store. Counts only, never text."""
    from app.brain.host import LiveHost
    from app.care import memory_unify
    from app.db.session import SessionLocal

    return await memory_unify.run(SessionLocal, LiveHost(), apply=apply, family=family, undo=undo, store_name=store or None)


@router.get("/person/{family_id}/{subject_id}")
async def person(family_id: str, subject_id: str, session: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    """What the backend needs to write to a person itself (the reminder fallback line, the holding line, voice notes):
    the name to use and the names never to use (app.care.naming.for_person), and the saved language and voice setting.
    The backend caches it 60 s; app.care.person_sync drops that copy when either changes."""
    from app.care import language, naming, store

    who = await naming.for_person(session, family_id, subject_id)
    lang = await store.active_fact(session, family_id, subject_id, "language:preferred")
    speech = language.normalise(lang.value if lang else {})
    return {
        "name": who["name"],
        "avoid": who["avoid"],
        "language": speech.get("language"),
        "dialect": speech.get("dialect"),
        "script": speech.get("script"),
        "voice": speech.get("voice"),
        "known": bool(lang),
    }


@router.get("/jobs/legacy-writes")
async def legacy_writes(session: Annotated[AsyncSession, Depends(get_db)], hours: int = 24) -> dict:
    """Rows written in the last `hours` to the retired Instinct tables. After the move this must stay 0 (the backend's
    nightly check emails the team otherwise)."""
    from app.models.entities import FamilyMemory, MemoryEntity, MemoryProfile

    since = clock.now() - timedelta(hours=max(1, min(hours, 24 * 14)))
    counts = {
        "family_memories": (await session.execute(select(func.count()).select_from(FamilyMemory).where(
            or_(FamilyMemory.created_at >= since, FamilyMemory.forgotten_at >= since, FamilyMemory.shared_at >= since)))).scalar_one(),
        "memory_entities": (await session.execute(select(func.count()).select_from(MemoryEntity).where(
            or_(MemoryEntity.created_at >= since, MemoryEntity.updated_at >= since)))).scalar_one(),
        "memory_profiles": (await session.execute(select(func.count()).select_from(MemoryProfile).where(
            or_(MemoryProfile.created_at >= since, MemoryProfile.rendered_at >= since)))).scalar_one(),
    }
    return {"since": since.isoformat(), "hours": hours, "counts": counts, "total": sum(counts.values())}
