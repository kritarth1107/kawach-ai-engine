from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.llm.provider import llm_provider_label
from app.models.entities import MemoryProfile
from app.rag.embeddings import embedding_provider_label, embeddings_available

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "kawach-ai-engine",
        "llm": llm_provider_label(),
        "embeddings": embedding_provider_label(),
    }


@router.get("/health/memory")
async def health_memory(db: AsyncSession = Depends(get_db)):
    last_dream = (
        await db.execute(select(func.max(MemoryProfile.rendered_at)))
    ).scalar_one_or_none()
    return {
        "status": "ok",
        "embeddings_available": embeddings_available(),
        "last_profile_render_at": last_dream.isoformat() if last_dream else None,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }
