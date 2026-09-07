import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_secret
from app.db.session import get_db
from app.rag.retrieve import format_family_memories, retrieve_family_memories
from app.services.tenant import scope_elder

router = APIRouter(prefix="/memory", tags=["memory"], dependencies=[Depends(verify_api_secret)])


class MemoryOut(BaseModel):
    id: str
    category: str
    topic: str
    content: str
    share_with_family: bool
    importance: int
    created_at: str | None = None


class MemoryListResponse(BaseModel):
    memories: list[MemoryOut]


@router.get("/list", response_model=MemoryListResponse)
async def list_memories(
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
    shareable_only: bool = Query(default=False),
):
    await scope_elder(db, family_id=family_id, elder_id=elder_id)
    rows = await retrieve_family_memories(
        db,
        family_id=family_id,
        elder_id=elder_id,
        limit=limit,
        shareable_only=shareable_only,
    )
    return MemoryListResponse(
        memories=[
            MemoryOut(
                id=str(m.id),
                category=m.category.value if hasattr(m.category, "value") else str(m.category),
                topic=m.topic,
                content=m.content,
                share_with_family=bool(m.share_with_family),
                importance=int(m.importance or 3),
                created_at=m.created_at.isoformat() if m.created_at else None,
            )
            for m in rows
        ]
    )


class MemorySearchRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=8, ge=1, le=30)


class MemorySearchResponse(BaseModel):
    context: str
    count: int


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(body: MemorySearchRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    await scope_elder(db, family_id=body.family_id, elder_id=body.elder_id)
    rows = await retrieve_family_memories(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        query=body.query.strip(),
        limit=body.limit,
    )
    ctx = format_family_memories(rows)
    return MemorySearchResponse(context=ctx, count=len(rows))
