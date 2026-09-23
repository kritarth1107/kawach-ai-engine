import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.entities import (
    DocumentChunk,
    FamilyMemory,
    MemoryEntity,
    MemorySnippet,
    Message,
    MessageRole,
)
from app.rag.embeddings import embed_text, embeddings_available

logger = logging.getLogger(__name__)


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
    tokens = [t for t in re.findall(r"[a-zA-Z0-9\u0900-\u097F\u0C80-\u0CFF]{2,}", query.lower())]
    if tokens:
        return tokens
    return [t for t in re.findall(r"\w{2,}", query.lower(), flags=re.UNICODE)]


def _active_memory_filters(stmt):
    return stmt.where(
        FamilyMemory.forgotten_at.is_(None),
        FamilyMemory.superseded_by.is_(None),
    )


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
    qv = "[" + ",".join(str(x) for x in query_vector) + "]"

    if elder_id is None:
        doc_sql = text("""
            SELECT content, metadata, 1 - (embedding <=> CAST(:qv AS vector)) AS score
            FROM document_chunks
            WHERE family_id = CAST(:family_id AS uuid)
              AND elder_id IS NULL
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:qv AS vector)
            LIMIT :limit
        """)
        doc_params = {"qv": qv, "family_id": str(family_id), "limit": k}
    else:
        doc_sql = text("""
            SELECT content, metadata, 1 - (embedding <=> CAST(:qv AS vector)) AS score
            FROM document_chunks
            WHERE family_id = CAST(:family_id AS uuid)
              AND (elder_id IS NULL OR elder_id = CAST(:elder_id AS uuid))
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:qv AS vector)
            LIMIT :limit
        """)
        doc_params = {
            "qv": qv,
            "family_id": str(family_id),
            "elder_id": str(elder_id),
            "limit": k,
        }

    doc_rows = (await session.execute(doc_sql, doc_params)).mappings().all()

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
                    "qv": qv,
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


async def retrieve_family_memories(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    query: str | None = None,
    limit: int = 12,
    shareable_only: bool = False,
) -> list[FamilyMemory]:
    stmt = _active_memory_filters(
        select(FamilyMemory).where(
            FamilyMemory.family_id == family_id,
            FamilyMemory.elder_id == elder_id,
        )
    )
    if shareable_only:
        stmt = stmt.where(FamilyMemory.share_with_family.is_(True))

    if query and embeddings_available():
        query_vector = await embed_text(query)
        qv = "[" + ",".join(str(x) for x in query_vector) + "]"
        sql = text("""
            SELECT id FROM family_memories
            WHERE family_id = CAST(:family_id AS uuid)
              AND elder_id = CAST(:elder_id AS uuid)
              AND forgotten_at IS NULL
              AND superseded_by IS NULL
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:qv AS vector)
            LIMIT :limit
        """)
        params: dict = {
            "qv": qv,
            "family_id": str(family_id),
            "elder_id": str(elder_id),
            "limit": limit,
        }
        if shareable_only:
            sql = text("""
                SELECT id FROM family_memories
                WHERE family_id = CAST(:family_id AS uuid)
                  AND elder_id = CAST(:elder_id AS uuid)
                  AND forgotten_at IS NULL
                  AND superseded_by IS NULL
                  AND share_with_family = true
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:qv AS vector)
                LIMIT :limit
            """)
        rows = (await session.execute(sql, params)).all()
        if rows:
            ids = [uuid.UUID(str(r[0])) for r in rows]
            found = (
                await session.execute(select(FamilyMemory).where(FamilyMemory.id.in_(ids)))
            ).scalars().all()
            by_id = {m.id: m for m in found}
            ordered = [by_id[i] for i in ids if i in by_id]
            await _touch_memories(session, ordered)
            return ordered

    stmt = stmt.order_by(FamilyMemory.importance.desc(), FamilyMemory.created_at.desc()).limit(limit)
    rows = list((await session.execute(stmt)).scalars().all())
    await _touch_memories(session, rows)
    return rows


async def _touch_memories(session: AsyncSession, memories: list[FamilyMemory]) -> None:
    if not memories:
        return
    now = datetime.utcnow()
    for mem in memories:
        mem.last_referenced_at = now
    await session.commit()


def format_family_memories(memories: list[FamilyMemory]) -> str:
    if not memories:
        return "(Nothing saved yet.)"
    lines: list[str] = []
    for m in memories:
        cat = m.category.value if hasattr(m.category, "value") else str(m.category)
        lines.append(f"[{cat}:{m.topic}] {m.content}")
    return "\n".join(lines)


async def load_instinct_context(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    query: str | None = None,
    entity_limit: int = 3,
    empty_label: str = "(Nothing saved yet.)",
) -> str:
    from app.rag.profile_render import load_memory_profile_text

    try:
        profile = await load_memory_profile_text(session, family_id=family_id, elder_id=elder_id)
        entity_blocks: list[str] = []
        q = (query or "").strip()
        if q:
            grep_hits = await memory_grep(
                session,
                family_id=family_id,
                elder_id=elder_id,
                query=q,
                limit=entity_limit,
            )
            for hit in grep_hits:
                entity = await load_entity_body(session, elder_id=elder_id, slug=hit.slug)
                if entity and entity.body_md:
                    entity_blocks.append(entity.body_md[:2000])
        text = profile
        if entity_blocks:
            text = (profile + "\n\n" + "\n\n".join(entity_blocks)).strip()
        return text or empty_label
    except Exception:
        logger.exception(
            "load_instinct_context failed family=%s elder=%s",
            family_id,
            elder_id,
        )
        return empty_label


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


def db_messages_to_langchain(rows: list[tuple[str, str, datetime | None]]) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    out: list = []
    for role, content, _ts in rows:
        if role in ("elder", "family"):
            out.append(HumanMessage(content=content))
        elif role == "saheli":
            out.append(AIMessage(content=content))
    return out


async def sync_conversation_history(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    conversation_id: uuid.UUID,
    thread: str,
    messages: list[dict],
) -> int:
    from sqlalchemy import select

    synced = 0
    for row in messages:
        external_id = row.get("external_id")
        if not external_id:
            continue
        existing = (
            await session.execute(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.metadata_.contains({"external_id": external_id}),
                )
            )
        ).scalar_one_or_none()
        if existing:
            continue
        role_raw = str(row.get("role") or "family")
        if thread == "elder":
            role = MessageRole.elder if role_raw in ("elder", "family") else MessageRole.saheli
        else:
            role = MessageRole.family if role_raw == "family" else MessageRole.saheli
            if role_raw == "elder":
                role = MessageRole.family
        content = str(row.get("content") or "")[:8000]
        if not content:
            continue
        msg = Message(
            conversation_id=conversation_id,
            family_id=family_id,
            elder_id=elder_id,
            role=role,
            content=content,
            metadata_={"external_id": external_id, "synced": True},
        )
        session.add(msg)
        synced += 1
    if synced:
        await session.commit()
    return synced


@dataclass
class MemoryGrepHit:
    slug: str
    kind: str
    title: str
    snippet: str
    score: float
    match_type: str


async def memory_grep(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    query: str,
    limit: int = 5,
) -> list[MemoryGrepHit]:
    tokens = _query_tokens(query)
    q_lower = query.lower().strip()
    entities = (
        await session.execute(
            select(MemoryEntity).where(
                MemoryEntity.family_id == family_id,
                MemoryEntity.elder_id == elder_id,
            )
        )
    ).scalars().all()

    scored: list[MemoryGrepHit] = []
    for entity in entities:
        aliases = [a.lower() for a in (entity.aliases or [])]
        body = (entity.body_md or "").lower()
        score = 0.0
        match_type = "alias"
        if q_lower and q_lower in body:
            score += 2.0
        for token in tokens:
            if any(token in alias for alias in aliases):
                score += 1.5
            if token in body:
                score += 1.0
            if token in entity.slug:
                score += 0.8
        if score <= 0:
            continue
        snippet = (entity.body_md or "")[:400]
        scored.append(
            MemoryGrepHit(
                slug=entity.slug,
                kind=entity.kind,
                title=entity.title,
                snippet=snippet,
                score=score,
                match_type=match_type,
            )
        )

    scored.sort(key=lambda row: row.score, reverse=True)
    if scored:
        return scored[:limit]

    if embeddings_available() and query.strip():
        try:
            query_vector = await embed_text(query)
            qv = "[" + ",".join(str(x) for x in query_vector) + "]"
            sql = text("""
                SELECT slug, kind, title, body_md,
                       1 - (body_embedding <=> CAST(:qv AS vector)) AS score
                FROM memory_entities
                WHERE family_id = CAST(:family_id AS uuid)
                  AND elder_id = CAST(:elder_id AS uuid)
                  AND body_embedding IS NOT NULL
                ORDER BY body_embedding <=> CAST(:qv AS vector)
                LIMIT :limit
            """)
            rows = (
                await session.execute(
                    sql,
                    {
                        "qv": qv,
                        "family_id": str(family_id),
                        "elder_id": str(elder_id),
                        "limit": limit,
                    },
                )
            ).mappings().all()
            return [
                MemoryGrepHit(
                    slug=row["slug"],
                    kind=row["kind"],
                    title=row["title"],
                    snippet=(row["body_md"] or "")[:400],
                    score=float(row["score"] or 0),
                    match_type="vector",
                )
                for row in rows
            ]
        except Exception:
            logger.warning(
                "memory_grep vector fallback failed family=%s elder=%s",
                family_id,
                elder_id,
                exc_info=True,
            )
    return []


async def load_entity_body(
    session: AsyncSession,
    *,
    elder_id: uuid.UUID,
    slug: str,
) -> MemoryEntity | None:
    return (
        await session.execute(
            select(MemoryEntity).where(
                MemoryEntity.elder_id == elder_id,
                MemoryEntity.slug == slug,
            )
        )
    ).scalar_one_or_none()


async def get_fact_history(
    session: AsyncSession,
    *,
    fact_id: uuid.UUID,
) -> list[FamilyMemory]:
    chain: list[FamilyMemory] = []
    current = await session.get(FamilyMemory, fact_id)
    while current:
        chain.append(current)
        if not current.superseded_by:
            break
        next_fact = await session.get(FamilyMemory, current.superseded_by)
        if not next_fact or next_fact.id == current.id:
            break
        current = next_fact
    return chain


async def list_stale_health_entities(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> list[MemoryEntity]:
    from datetime import date

    return list(
        (
            await session.execute(
                select(MemoryEntity).where(
                    MemoryEntity.family_id == family_id,
                    MemoryEntity.elder_id == elder_id,
                    MemoryEntity.review_by.isnot(None),
                    MemoryEntity.review_by < date.today(),
                    MemoryEntity.kind.in_(["medication", "condition", "symptom"]),
                )
            )
        ).scalars().all()
    )


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

