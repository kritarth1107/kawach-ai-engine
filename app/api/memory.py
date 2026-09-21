import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_secret
from app.db.session import get_db
from app.rag.memory_extract import persist_inbox_memory
from app.rag.memory_lifecycle import correct_memory_fact, erase_elder_memory, forget_memory_fact
from app.rag.profile_render import load_memory_profile_text
from app.rag.retrieve import (
    format_family_memories,
    get_fact_history,
    list_stale_health_entities,
    load_entity_body,
    memory_grep,
    retrieve_family_memories,
)
from app.services.tenant import scope_elder

router = APIRouter(prefix="/memory", tags=["memory"], dependencies=[Depends(verify_api_secret)])


class MemoryOut(BaseModel):
    id: str
    category: str
    topic: str
    content: str
    share_with_family: bool
    importance: int
    source_role: str | None = None
    created_at: str | None = None
    entity_id: str | None = None
    superseded_by: str | None = None


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
                source_role=getattr(m, "source_role", None),
                created_at=m.created_at.isoformat() if m.created_at else None,
                entity_id=str(m.entity_id) if m.entity_id else None,
                superseded_by=str(m.superseded_by) if m.superseded_by else None,
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


class InboxRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    content: str = Field(min_length=4, max_length=500)
    category: str = "casual"
    topic: str = "general"
    source_role: str = "saheli"


class ExtractRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    message: str = Field(min_length=4, max_length=2000)
    source_role: str = "elder"


@router.post("/extract")
async def enqueue_memory_extract(body: ExtractRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    """Queue background memory extraction (for WhatsApp bypass paths that skip the agent)."""
    await scope_elder(db, family_id=body.family_id, elder_id=body.elder_id)
    from app.rag.memory_queue import schedule_memory_extract

    schedule_memory_extract(
        family_id=body.family_id,
        elder_id=body.elder_id,
        message=body.message.strip(),
        source_role=body.source_role,
    )
    return {"queued": True}


@router.post("/inbox")
async def inbox_memory(body: InboxRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    await scope_elder(db, family_id=body.family_id, elder_id=body.elder_id)
    row = await persist_inbox_memory(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        content=body.content.strip(),
        category=body.category,
        topic=body.topic,
        source_role=body.source_role,
    )
    return {"saved": True, "fact_id": str(row.id)}


class ForgetRequest(BaseModel):
    forgotten_by: str | None = None


@router.post("/{fact_id}/forget")
async def forget_fact(
    fact_id: uuid.UUID,
    body: ForgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    return await forget_memory_fact(db, fact_id=fact_id, forgotten_by=body.forgotten_by)


class CorrectRequest(BaseModel):
    replacement_content: str = Field(min_length=4, max_length=500)
    actor_user_id: str | None = None
    source_role: str = "family"


@router.post("/{fact_id}/correct")
async def correct_fact(
    fact_id: uuid.UUID,
    body: CorrectRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    return await correct_memory_fact(
        db,
        fact_id=fact_id,
        replacement_content=body.replacement_content,
        actor_user_id=body.actor_user_id,
        source_role=body.source_role,
    )


@router.delete("/families/{family_id}/elders/{elder_id}")
async def erase_elder(
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await scope_elder(db, family_id=family_id, elder_id=elder_id)
    return await erase_elder_memory(db, family_id=family_id, elder_id=elder_id)


class GrepRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


@router.post("/grep")
async def grep_memory(body: GrepRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    await scope_elder(db, family_id=body.family_id, elder_id=body.elder_id)
    hits = await memory_grep(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        query=body.query.strip(),
        limit=body.limit,
    )
    return {
        "hits": [
            {
                "slug": h.slug,
                "kind": h.kind,
                "title": h.title,
                "snippet": h.snippet,
                "score": h.score,
                "match_type": h.match_type,
            }
            for h in hits
        ]
    }


@router.get("/entity/{slug}")
async def get_entity(
    slug: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID = Query(...),
):
    await scope_elder(db, family_id=family_id, elder_id=elder_id)
    entity = await load_entity_body(db, elder_id=elder_id, slug=slug)
    if not entity:
        return {"entity": None}
    return {
        "entity": {
            "slug": entity.slug,
            "kind": entity.kind,
            "title": entity.title,
            "status": entity.status,
            "body_md": entity.body_md,
            "review_by": entity.review_by.isoformat() if entity.review_by else None,
            "version": entity.version,
        }
    }


@router.get("/entity/{slug}/history")
async def entity_history(
    slug: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID = Query(...),
):
    await scope_elder(db, family_id=family_id, elder_id=elder_id)
    entity = await load_entity_body(db, elder_id=elder_id, slug=slug)
    if not entity or not entity.sources:
        return {"facts": []}
    facts = []
    for source_id in entity.sources[:20]:
        try:
            fid = uuid.UUID(str(source_id))
        except ValueError:
            continue
        chain = await get_fact_history(db, fact_id=fid)
        for f in chain:
            facts.append(
                {
                    "id": str(f.id),
                    "content": f.content,
                    "source_role": getattr(f, "source_role", "elder"),
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                    "superseded_by": str(f.superseded_by) if f.superseded_by else None,
                }
            )
    return {"facts": facts}


@router.get("/profile")
async def get_profile(
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID = Query(...),
):
    await scope_elder(db, family_id=family_id, elder_id=elder_id)
    body = await load_memory_profile_text(db, family_id=family_id, elder_id=elder_id)
    return {"profile_md": body}


@router.get("/stale-health")
async def stale_health(
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID = Query(...),
):
    await scope_elder(db, family_id=family_id, elder_id=elder_id)
    rows = await list_stale_health_entities(db, family_id=family_id, elder_id=elder_id)
    return {
        "entities": [
            {
                "slug": e.slug,
                "kind": e.kind,
                "title": e.title,
                "review_by": e.review_by.isoformat() if e.review_by else None,
                "status": e.status,
            }
            for e in rows
        ]
    }
