import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.entities import DocumentChunk, MemorySnippet, Message, MessageRole
from app.rag.embeddings import embed_text, embeddings_available


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float
    kind: str


async def retrieve_context(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID | None,
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    k = top_k or settings.rag_top_k

    if embeddings_available():
        vector_hits = await _vector_retrieve(
            session, family_id=family_id, elder_id=elder_id, query=query, k=k
        )
        if vector_hits:
            return vector_hits

    return await _keyword_retrieve(
        session, family_id=family_id, elder_id=elder_id, query=query, k=k
    )


def _query_tokens(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]{2,}", query.lower())]


async def _keyword_retrieve(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID | None,
    query: str,
    k: int,
) -> list[RetrievedChunk]:
    tokens = _query_tokens(query)
    if not tokens:
        return []

    stmt = select(DocumentChunk).where(DocumentChunk.family_id == family_id)
    if elder_id is not None:
        stmt = stmt.where(
            (DocumentChunk.elder_id == elder_id) | (DocumentChunk.elder_id.is_(None))
        )
    chunks = (await session.execute(stmt.limit(400))).scalars().all()

    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        hay = (chunk.content or "").lower()
        hits = sum(1 for t in tokens if t in hay)
        if not hits:
            continue
        meta = chunk.metadata_ or {}
        scored.append(
            RetrievedChunk(
                content=chunk.content,
                source=meta.get("title", "document"),
                score=hits / len(tokens),
                kind=meta.get("kind", "lab"),
            )
        )

    if elder_id:
        snip_stmt = select(MemorySnippet).where(
            MemorySnippet.family_id == family_id,
            MemorySnippet.elder_id == elder_id,
        )
        snippets = (await session.execute(snip_stmt.limit(200))).scalars().all()
        for snip in snippets:
            hay = (snip.content or "").lower()
            hits = sum(1 for t in tokens if t in hay)
            if not hits:
                continue
            scored.append(
                RetrievedChunk(
                    content=snip.content,
                    source=snip.source or "chat",
                    score=hits / len(tokens),
                    kind="snippet",
                )
            )

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:k]


async def _vector_retrieve(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID | None,
    query: str,
    k: int,
) -> list[RetrievedChunk]:
    query_vector = await embed_text(query)

    doc_sql = text("""
        SELECT content, metadata, 1 - (embedding <=> CAST(:qv AS vector)) AS score
        FROM document_chunks
        WHERE family_id = CAST(:family_id AS uuid)
          AND (
            (:elder_id IS NULL AND elder_id IS NULL)
            OR (
              :elder_id IS NOT NULL
              AND (elder_id IS NULL OR elder_id = CAST(:elder_id AS uuid))
            )
          )
          AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:qv AS vector)
        LIMIT :limit
    """)
    doc_rows = (
        await session.execute(
            doc_sql,
            {
                "qv": query_vector,
                "family_id": str(family_id),
                "elder_id": str(elder_id) if elder_id else None,
                "limit": k,
            },
        )
    ).mappings().all()

    snippet_rows = []
    if elder_id:
        snippet_sql = text("""
            SELECT content, source, 1 - (embedding <=> CAST(:qv AS vector)) AS score
            FROM memory_snippets
            WHERE family_id = CAST(:family_id AS uuid)
              AND elder_id = CAST(:elder_id AS uuid)
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:qv AS vector)
            LIMIT :limit
        """)
        snippet_rows = (
            await session.execute(
                snippet_sql,
                {
                    "qv": query_vector,
                    "family_id": str(family_id),
                    "elder_id": str(elder_id),
                    "limit": k,
                },
            )
        ).mappings().all()

    results: list[RetrievedChunk] = []
    for row in doc_rows:
        meta = row.get("metadata") or {}
        results.append(
            RetrievedChunk(
                content=row["content"],
                source=meta.get("title", "document"),
                score=float(row["score"] or 0),
                kind=meta.get("kind", "lab"),
            )
        )
    for row in snippet_rows:
        results.append(
            RetrievedChunk(
                content=row["content"],
                source=row.get("source", "chat"),
                score=float(row["score"] or 0),
                kind="snippet",
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


async def get_recent_messages(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    limit: int = 20,
    conversation_id: uuid.UUID | None = None,
) -> list[tuple[str, str, datetime | None]]:
    stmt = (
        select(Message.role, Message.content, Message.created_at)
        .where(Message.family_id == family_id, Message.elder_id == elder_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    rows = (await session.execute(stmt)).all()
    return [(r[0].value, r[1], r[2]) for r in reversed(rows)]


async def get_elder_thread_context(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
    limit: int = 16,
) -> str:
    stmt = (
        select(Message.role, Message.content)
        .where(
            Message.family_id == family_id,
            Message.elder_id == elder_id,
            Message.conversation_id == conversation_id,
            Message.role.in_([MessageRole.elder, MessageRole.saheli, MessageRole.system]),
        )
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    lines: list[str] = []
    for role, content in reversed(rows):
        label = "check-in" if role.value == "system" else role.value
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(No elder messages with Saheli yet.)"

