import uuid

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import DocumentChunk, DocumentKind, MemoryDocument, MemorySnippet
from app.rag.chunker import split_text
from app.rag.embeddings import embed_text, embed_texts, embeddings_available


async def ingest_document(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID | None,
    title: str,
    raw_text: str,
    kind: DocumentKind = DocumentKind.lab,
    source: str = "upload",
    record_date: str | None = None,
    summary: str | None = None,
    highlights: dict | list | None = None,
) -> MemoryDocument:
    doc = MemoryDocument(
        family_id=family_id,
        elder_id=elder_id,
        kind=kind,
        title=title,
        source=source,
        raw_text=raw_text,
        summary=summary,
        record_date=record_date,
        highlights=highlights,
    )
    session.add(doc)
    await session.flush()

    chunks = split_text(raw_text)
    if chunks and embeddings_available():
        vectors = await embed_texts(chunks)
    else:
        vectors = [None] * len(chunks)

    for idx, (content, vector) in enumerate(zip(chunks, vectors, strict=False)):
        session.add(
            DocumentChunk(
                document_id=doc.id,
                family_id=family_id,
                elder_id=elder_id,
                chunk_index=idx,
                content=content,
                embedding=vector,
                metadata_={"title": title, "kind": kind.value},
            )
        )

    await session.commit()
    await session.refresh(doc)
    return doc


async def ingest_chat_snippet(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    content: str,
    source: str = "elder_message",
    source_message_id: uuid.UUID | None = None,
    commit: bool = True,
) -> MemorySnippet:
    vector = await embed_text(content)
    snippet = MemorySnippet(
        family_id=family_id,
        elder_id=elder_id,
        content=content[:500],
        source=source,
        source_message_id=source_message_id,
        embedding=vector,
    )
    session.add(snippet)
    if commit:
        await session.commit()
        await session.refresh(snippet)
    else:
        await session.flush()
    return snippet


async def reindex_document(session: AsyncSession, document_id: uuid.UUID) -> None:
    doc = await session.get(MemoryDocument, document_id)
    if not doc:
        return
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    chunks = split_text(doc.raw_text)
    if chunks and embeddings_available():
        vectors = await embed_texts(chunks)
    else:
        vectors = [None] * len(chunks)
    for idx, (content, vector) in enumerate(zip(chunks, vectors, strict=False)):
        session.add(
            DocumentChunk(
                document_id=doc.id,
                family_id=doc.family_id,
                elder_id=doc.elder_id,
                chunk_index=idx,
                content=content,
                embedding=vector,
                metadata_={"title": doc.title, "kind": doc.kind.value},
            )
        )
    await session.commit()
