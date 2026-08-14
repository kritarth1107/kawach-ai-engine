import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_secret
from app.db.session import get_db
from app.models.entities import DocumentKind, MemoryDocument
from app.rag.ingest import ingest_document
from app.rag.retrieve import retrieve_context
from app.services.tenant import scope_document_request

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(verify_api_secret)])


class IngestDocumentRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID | None = None
    title: str
    raw_text: str
    kind: DocumentKind = DocumentKind.lab
    source: str = "upload"
    record_date: str | None = None
    summary: str | None = None


class SearchRequest(BaseModel):
    family_id: uuid.UUID
    elder_id: uuid.UUID | None = None
    query: str
    top_k: int = Field(default=8, ge=1, le=20)


@router.post("/ingest")
async def ingest(body: IngestDocumentRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    await scope_document_request(db, family_id=body.family_id, elder_id=body.elder_id)
    doc = await ingest_document(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        title=body.title,
        raw_text=body.raw_text,
        kind=body.kind,
        source=body.source,
        record_date=body.record_date,
        summary=body.summary,
    )
    return {
        "document_id": str(doc.id),
        "title": doc.title,
        "kind": doc.kind.value,
    }


@router.get("/list")
async def list_documents(
    db: Annotated[AsyncSession, Depends(get_db)],
    family_id: uuid.UUID = Query(...),
    elder_id: uuid.UUID | None = Query(default=None),
):
    await scope_document_request(db, family_id=family_id, elder_id=elder_id)
    stmt = (
        select(MemoryDocument)
        .where(MemoryDocument.family_id == family_id)
        .order_by(MemoryDocument.created_at.desc())
        .limit(50)
    )
    if elder_id is not None:
        stmt = stmt.where(
            (MemoryDocument.elder_id == elder_id) | (MemoryDocument.elder_id.is_(None))
        )
    docs = (await db.execute(stmt)).scalars().all()
    return {
        "documents": [
            {
                "document_id": str(d.id),
                "title": d.title,
                "kind": d.kind.value,
                "record_date": d.record_date,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]
    }


@router.post("/search")
async def search(body: SearchRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    await scope_document_request(db, family_id=body.family_id, elder_id=body.elder_id)
    chunks = await retrieve_context(
        db,
        family_id=body.family_id,
        elder_id=body.elder_id,
        query=body.query,
        top_k=body.top_k,
    )
    return {
        "results": [
            {
                "content": c.content,
                "source": c.source,
                "kind": c.kind,
                "score": round(c.score, 4),
            }
            for c in chunks
        ]
    }
