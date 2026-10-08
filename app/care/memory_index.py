"""Meaning-based memory: every note line, event and fact gets an embedding, so recall finds "the knee pain"
when someone says "wahi dard phir se". Results are ranked by meaning, words, importance and recency together.

- importance: health and safety memories (red flags, outcomes, vitals, symptoms, medicine changes) outrank chit-chat
  and never fade from recall; ordinary memories fade with age (half-life ~45 days)
- the index is filled at night (the dream) and on demand; if no embedder is available (Vertex down, tests), recall
  falls back to keywords + importance + recency, so nothing breaks
- embedder: Vertex text-multilingual (app.rag.embeddings) by default; tests and the simulator can set a fake one
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Index, String, Text, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.care import store
from app.care.models import CareEvent, CareFact, MemoryNote
from app.core import clock
from app.core.config import get_settings
from app.db.session import Base

logger = logging.getLogger(__name__)
DIM = get_settings().embed_dim
HALF_LIFE_DAYS = 45.0
IMPORTANCE = {
    "outcome": 1.0, "alert_whatsapp": 1.0, "vital": 0.8, "symptom": 0.8, "dose_missed": 0.6, "dose_refused": 0.6,
    "fact_superseded": 0.7, "fact_stopped": 0.7, "pattern": 0.6, "mood": 0.5, "meal": 0.3, "dose_taken": 0.2, "sleep": 0.4,
    "social": 0.4, "routine": 0.3, "other": 0.3, "note": 0.5, "diary": 0.4, "fact": 0.8, "summary": 0.6,
    "report": 0.8,  # a health record the family chose to have remembered: never fades
}
NEVER_FADE = 0.8  # memories at least this important do not fade with age

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]
_embedder: Embedder | None = None


def set_embedder(fn: Embedder | None) -> None:
    """Tests and the simulator inject a fake; production uses Vertex embeddings."""
    global _embedder
    _embedder = fn


EMBED_TIMEOUT = 3.0  # seconds; a person's reply never waits longer than this for meaning-search
BREAKER_SECONDS = 300.0
_broken_until = 0.0


async def embed(texts: list[str]) -> list[list[float]] | None:
    """Embeddings, or None (keyword recall only) when off, failing, or slow. A failure pauses tries for 5 minutes."""
    global _broken_until
    import asyncio
    import os
    import time

    if _embedder is not None:
        return await _embedder(texts)
    if os.getenv("MEMORY_EMBEDDINGS", "on") != "on" or time.monotonic() < _broken_until:
        return None
    try:
        from app.rag.embeddings import embed_texts, embeddings_available

        if not embeddings_available():
            return None
        return await asyncio.wait_for(embed_texts(texts), timeout=EMBED_TIMEOUT if len(texts) == 1 else 60)
    except Exception:  # noqa: BLE001 — recall works without meaning-search
        _broken_until = time.monotonic() + BREAKER_SECONDS
        logger.warning("embeddings unavailable for %ss; keyword recall only", int(BREAKER_SECONDS))
        return None


class MemoryVector(Base):
    __tablename__ = "memory_vectors"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # hash of source + text
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16))  # note | event | fact
    ref: Mapped[str] = mapped_column(String(80))
    text: Mapped[str] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    importance: Mapped[float] = mapped_column(Float, default=0.3)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(DIM), nullable=True)

    __table_args__ = (Index("ix_memory_vectors_subject", "family_id", "subject_id", "at"),)


def _key(ref: str, text: str) -> str:
    return hashlib.sha256(f"{ref}|{text}".encode()).hexdigest()[:40]


def importance_of(kind: str, payload: dict | None = None) -> float:
    p = payload or {}
    if p.get("red_flag") or p.get("severity") == "concern":
        return 1.0
    if p.get("severity") == "watch":
        return max(0.6, IMPORTANCE.get(kind, 0.3))
    return IMPORTANCE.get(kind, 0.3)


def decay(at: datetime, importance: float, now: datetime | None = None) -> float:
    if importance >= NEVER_FADE:
        return 1.0
    days = max(0.0, ((now or clock.now()) - at).total_seconds() / 86400)
    return 0.5 ** (days / HALF_LIFE_DAYS)


@dataclass
class Item:
    source: str
    ref: str
    text: str
    at: datetime
    importance: float


async def _items(session: AsyncSession, family_id: str, subject_id: str, since: datetime | None) -> list[Item]:
    out: list[Item] = []
    for n in await store.notes(session, family_id, [subject_id]):
        if since and n.updated_at < since:
            continue
        for line in [ln.strip("- ").strip() for ln in (n.body_md or "").splitlines() if len(ln.strip()) > 12]:
            out.append(Item("note", f"note:{n.slug}", f"{n.title}: {line}"[:600], n.updated_at, IMPORTANCE["diary" if n.slug == "diary" else "note"]))
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id)
    if since:
        q = q.where(CareEvent.at >= since)
    for e in (await session.execute(q.order_by(CareEvent.at.desc()).limit(5000))).scalars():
        if e.kind in ("dream", "memory_extract", "feedback", "import_done") or (e.payload or {}).get("forgotten"):
            continue
        out.append(Item("event", f"event:{e.id}", f"[{e.kind}] {e.summary}"[:600], e.at, importance_of(e.kind, e.payload)))
    for f in await store.facts(session, family_id, subject_id, statuses=("active", "pending", "superseded", "stopped")):
        if since and f.recorded_at < since:
            continue
        out.append(Item("fact", f"fact:{f.id}", f"({f.status}) {f.text}"[:600], f.recorded_at, IMPORTANCE["fact"]))
    return out


async def index_subject(session: AsyncSession, family_id: str, subject_id: str, *, since: datetime | None = None, batch: int = 64) -> int:
    """Add new memories of one person to the index (embedding them when an embedder is available)."""
    items = await _items(session, family_id, subject_id, since)
    if not items:
        return 0
    keys = {_key(i.ref, i.text): i for i in items}
    have = set((await session.execute(select(MemoryVector.key).where(MemoryVector.key.in_(list(keys))))).scalars())
    new = [(k, i) for k, i in keys.items() if k not in have]
    added = 0
    for start in range(0, len(new), batch):
        chunk = new[start:start + batch]
        vecs = await embed([i.text for _, i in chunk])
        for j, (k, i) in enumerate(chunk):
            stmt = insert(MemoryVector).values(key=k, family_id=family_id, subject_id=subject_id, source=i.source, ref=i.ref, text=i.text,
                                               at=i.at, importance=i.importance, embedding=vecs[j] if vecs else None)
            await session.execute(stmt.on_conflict_do_nothing())
            added += 1
    return added


async def unindex(session: AsyncSession, family_id: str, ref: str) -> None:
    await session.execute(delete(MemoryVector).where(MemoryVector.family_id == family_id, MemoryVector.ref == ref))


async def search(session: AsyncSession, family_id: str, subject_ids: list[str], query: str, *, limit: int = 8) -> list[store.Hit]:
    """Hybrid recall: meaning (if indexed) + keywords, weighted by importance and recency (reciprocal rank fusion)."""
    keyword = await store.recall(session, family_id, subject_ids, query, limit=limit * 3)
    scores: dict[str, float] = {}
    hits: dict[str, store.Hit] = {}
    weight: dict[str, float] = {}
    for rank, h in enumerate(keyword):
        scores[h.ref] = scores.get(h.ref, 0) + 1.0 / (60 + rank)
        hits[h.ref] = h
    vec = await embed([query]) if query.strip() else None
    if vec:
        rows = (await session.execute(
            select(MemoryVector).where(MemoryVector.family_id == family_id, MemoryVector.subject_id.in_(subject_ids),
                                       MemoryVector.embedding.is_not(None))
            .order_by(MemoryVector.embedding.cosine_distance(vec[0])).limit(limit * 3)
        )).scalars()
        for rank, r in enumerate(rows):
            scores[r.ref] = scores.get(r.ref, 0) + 1.0 / (60 + rank)
            weight[r.ref] = r.importance * decay(r.at, r.importance)
            hits.setdefault(r.ref, store.Hit(r.source, r.at, r.text, r.ref))
    now = clock.now()
    for ref, h in hits.items():
        w = weight.get(ref)
        if w is None:
            kind = h.text[1:h.text.index("]")] if h.text.startswith("[") and "]" in h.text else h.source
            imp = importance_of(kind) if h.source == "event" else IMPORTANCE.get(h.source, 0.4)
            w = imp * decay(h.when, imp, now)
        scores[ref] *= 0.6 + 0.4 * w
    ranked = sorted(hits.values(), key=lambda h: scores[h.ref], reverse=True)
    return ranked[:limit]


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


async def nightly(session: AsyncSession, family_id: str, subject_id: str) -> int:
    """Index the last two days (and everything the first time)."""
    any_row = (await session.execute(select(MemoryVector.key).where(MemoryVector.family_id == family_id,
                                                                     MemoryVector.subject_id == subject_id).limit(1))).first()
    since = None if not any_row else clock.now() - timedelta(days=2)
    return await index_subject(session, family_id, subject_id, since=since)
