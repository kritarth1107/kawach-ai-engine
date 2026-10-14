"""Meaning-based memory: every note line, event and fact gets an embedding, so recall finds "the knee pain"
when someone says "wahi dard phir se". Results are ranked by meaning, words, importance and recency together.

- importance: health and safety memories (red flags, outcomes, vitals, symptoms, medicine changes) outrank chit-chat
  and never fade from recall; ordinary memories fade with age (half-life ~45 days)
- the index is filled at night (the dream) and on demand; if no embedder is available (Vertex down, tests), recall
  falls back to keywords + importance + recency, so nothing breaks
- embedder: Vertex text-multilingual (app.rag.embeddings) by default, Bedrock on the AWS copy (EMBED_PROVIDER); tests and
  the simulator can set a fake one
- the index always says what is true now (D-12, 2026-10-11: an old dose and a corrected note line came back next to
  the new ones, both "(active)"): a fact's row is relabelled the moment its status changes (an earlier value says when
  it ended, a turned-down proposal is dropped), a note line is dropped the moment a write takes it out of its note
  (dated diary lines stay: they are history), and search checks every hit against the record and the notes as they are
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Index, String, Text, delete, select, text
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
EARLIER_IMPORTANCE = 0.3  # an earlier value of a fact ranks below current ones and fades like everyday memories
HISTORY_NOTES = {"diary"}  # dated history: lines that drop out of these notes stay findable

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]
_embedder: Embedder | None = None


def set_embedder(fn: Embedder | None) -> None:
    """Tests and the simulator inject a fake; production uses Vertex embeddings."""
    global _embedder
    _embedder = fn


EMBED_TIMEOUT = 3.0  # seconds; a person's reply never waits longer than this for meaning-search
BREAKER_SECONDS = 300.0
_broken_until = 0.0


async def embed(texts: list[str], *, query: bool = False) -> list[list[float]] | None:
    """Embeddings, or None (keyword recall only) when off, failing, or slow. A failure pauses tries for 5 minutes.
    query=True: the texts are search queries (Cohere on Bedrock embeds those differently from memories)."""
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
        return await asyncio.wait_for(embed_texts(texts, query=query), timeout=EMBED_TIMEOUT if len(texts) == 1 else 60)
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


def _key(family_id: str, subject_id: str, ref: str, text: str) -> str:
    # per family and person: the same diary line ("Quiet day, nothing logged.") in two families is two rows
    return hashlib.sha256(f"{family_id}|{subject_id}|{ref}|{text}".encode()).hexdigest()[:40]


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


def fact_item(f: CareFact) -> Item | None:
    """A care-record row as it reads in the index: a current row with its status; an earlier one (replaced or stopped)
    marked with the day it ended, so meaning-search can still answer "what was the dose before?"; a proposal that was
    turned down or taken back not at all (it was never true)."""
    if f.status in ("active", "pending"):
        return Item("fact", f"fact:{f.id}", f"({f.status}) {f.text}"[:600], f.recorded_at, IMPORTANCE["fact"])
    if f.status in ("superseded", "stopped"):
        until = clock.ist_day(f.valid_to) if f.valid_to else "?"
        return Item("fact", f"fact:{f.id}", f"(earlier, until {until}) {f.text}"[:600], f.valid_to or f.recorded_at, EARLIER_IMPORTANCE)
    return None


def _line_texts(title: str, body: str) -> list[str]:
    """How each line of a note reads in the index (a line of a dozen letters or less says too little to find)."""
    return [f"{title}: {line}"[:600] for line in [ln.strip("- ").strip() for ln in (body or "").splitlines() if len(ln.strip()) > 12]]


def note_items(n: MemoryNote) -> list[Item]:
    imp = IMPORTANCE["diary" if n.slug == "diary" else "note"]
    return [Item("note", f"note:{n.slug}", t, n.updated_at, imp) for t in _line_texts(n.title, n.body_md)]


async def _items(session: AsyncSession, family_id: str, subject_id: str, since: datetime | None) -> list[Item]:
    out: list[Item] = []
    for n in await store.notes(session, family_id, [subject_id]):
        if since and n.updated_at < since:
            continue
        out += note_items(n)
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id)
    if since:
        q = q.where(CareEvent.at >= since)
    for e in (await session.execute(q.order_by(CareEvent.at.desc()).limit(5000))).scalars():
        if e.kind in ("dream", "memory_extract", "feedback", "import_done") or (e.payload or {}).get("forgotten"):
            continue
        out.append(Item("event", f"event:{e.id}", f"[{e.kind}] {e.summary}"[:600], e.at, importance_of(e.kind, e.payload)))
    for f in await store.facts(session, family_id, subject_id, statuses=("active", "pending", "superseded", "stopped")):
        if since and f.recorded_at < since and not (f.valid_to and f.valid_to >= since):
            continue
        if (item := fact_item(f)) is not None:
            out.append(item)
    return out


async def index_subject(session: AsyncSession, family_id: str, subject_id: str, *, since: datetime | None = None, batch: int = 64) -> int:
    """Add new memories of one person to the index (embedding them when an embedder is available)."""
    items = await _items(session, family_id, subject_id, since)
    if not items:
        return 0
    have = {(r.ref, r.text) for r in (await session.execute(select(MemoryVector.ref, MemoryVector.text).where(
        MemoryVector.family_id == family_id, MemoryVector.subject_id == subject_id))).all()}
    new: list[Item] = []
    for i in items:
        if (i.ref, i.text) not in have:
            have.add((i.ref, i.text))
            new.append(i)
    added = 0
    for start in range(0, len(new), batch):
        chunk = new[start:start + batch]
        vecs = await embed([i.text for i in chunk])
        for j, i in enumerate(chunk):
            stmt = insert(MemoryVector).values(key=_key(family_id, subject_id, i.ref, i.text), family_id=family_id, subject_id=subject_id,
                                               source=i.source, ref=i.ref, text=i.text, at=i.at, importance=i.importance,
                                               embedding=vecs[j] if vecs else None)
            await session.execute(stmt.on_conflict_do_nothing())
            added += 1
    return added


async def unindex(session: AsyncSession, family_id: str, ref: str, *, subject_id: str | None = None) -> None:
    """Drop every index row of one memory (a note's rows only for that person when subject_id is given: two people
    can each have a note called 'diary')."""
    q = delete(MemoryVector).where(MemoryVector.family_id == family_id, MemoryVector.ref == ref)
    if subject_id is not None:
        q = q.where(MemoryVector.subject_id == subject_id)
    await session.execute(q)


async def unindex_note_lines(session: AsyncSession, family_id: str, subject_id: str, slug: str, title: str, lines: list[str]) -> int:
    """Drop the index rows of these lines of a note, diary lines too (what someone asked Saheli to forget)."""
    gone = set(_line_texts(title, "\n".join(lines)))
    if not gone:
        return 0
    res = await session.execute(delete(MemoryVector).where(MemoryVector.family_id == family_id, MemoryVector.subject_id == subject_id,
                                                           MemoryVector.ref == f"note:{slug}", MemoryVector.text.in_(gone)))
    return res.rowcount or 0


async def sync_facts(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    """One index row per care-record fact of this person, saying what the fact is now. Called on every status change
    (a supersede, stop, approval, rejection, undo or restore all go through app.care.store) and at night. When only the
    label changes, the old embedding is reused: the meaning is the same."""
    rows = list((await session.execute(select(MemoryVector).where(
        MemoryVector.family_id == family_id, MemoryVector.subject_id == subject_id, MemoryVector.source == "fact"))).scalars())
    out = {"relabelled": 0, "removed": 0}
    if not rows:
        return out
    by_ref: dict[str, list[MemoryVector]] = {}
    for r in rows:
        by_ref.setdefault(r.ref, []).append(r)
    ids = []
    for ref in by_ref:
        try:
            ids.append(uuid.UUID(ref.split(":", 1)[1]))
        except (ValueError, IndexError):
            pass
    facts = {f"fact:{f.id}": f for f in (await session.execute(select(CareFact).where(CareFact.id.in_(ids)))).scalars()} if ids else {}
    for ref, have in by_ref.items():
        f = facts.get(ref)
        want = fact_item(f) if f is not None and f.family_id == family_id and f.subject_id == subject_id else None
        keep = next((r for r in have if want is not None and r.text == want.text), None)
        if want is not None and keep is None:
            reuse = next((r.embedding for r in have if r.embedding is not None), None)
            await session.execute(insert(MemoryVector).values(
                key=_key(family_id, subject_id, ref, want.text), family_id=family_id, subject_id=subject_id, source="fact", ref=ref,
                text=want.text, at=want.at, importance=want.importance, embedding=reuse).on_conflict_do_nothing())
            out["relabelled"] += 1
        stale = [r.key for r in have if r is not keep]
        if stale:
            await session.execute(delete(MemoryVector).where(MemoryVector.key.in_(stale)))
            out["removed"] += len(stale) - (1 if want is not None and keep is None else 0)
    return out


async def sync_notes(session: AsyncSession, family_id: str, subject_id: str, *, slug: str | None = None) -> int:
    """Drop the index rows of note lines that are no longer in their note (a rewrite, a rollup, an edit, an undo), or
    whose note is gone. Dated diary lines stay findable: they are history. Returns how many rows went."""
    q = select(MemoryVector.key, MemoryVector.ref, MemoryVector.text).where(
        MemoryVector.family_id == family_id, MemoryVector.subject_id == subject_id, MemoryVector.source == "note")
    if slug:
        q = q.where(MemoryVector.ref == f"note:{slug}")
    rows = (await session.execute(q)).all()
    if not rows:
        return 0
    current: dict[str, set[str]] = {}
    for n in await store.notes(session, family_id, [subject_id]):
        if not slug or n.slug == slug:
            current[f"note:{n.slug}"] = {i.text for i in note_items(n)}
    stale = [r.key for r in rows if r.ref.split(":", 1)[-1] not in HISTORY_NOTES and r.text not in current.get(r.ref, set())]
    if stale:
        await session.execute(delete(MemoryVector).where(MemoryVector.key.in_(stale)))
    return len(stale)


async def fill_embeddings(session: AsyncSession, family_id: str, subject_id: str, *, batch: int = 64) -> int:
    """Embed rows that have no embedding yet (a relabel whose old row had none, or a night without an embedder)."""
    rows = list((await session.execute(select(MemoryVector).where(
        MemoryVector.family_id == family_id, MemoryVector.subject_id == subject_id, MemoryVector.embedding.is_(None)).limit(500))).scalars())
    done = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        vecs = await embed([r.text for r in chunk])
        if not vecs:
            break
        for r, v in zip(chunk, vecs):
            r.embedding = v
            done += 1
    return done


async def reconcile(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    """Bring one person's index rows in line with their care record and notes (no embedding calls)."""
    facts = await sync_facts(session, family_id, subject_id)
    return {"relabelled": facts["relabelled"], "removed": facts["removed"] + await sync_notes(session, family_id, subject_id)}


CLEANUP_LOCK = 4_712_012  # Postgres advisory lock: one instance at a time runs the start-up cleanup


async def cleanup_all(session: AsyncSession) -> dict:
    """Reconcile every person who has index rows (idempotent: a second run changes nothing)."""
    out = {"people": 0, "relabelled": 0, "removed": 0}
    owners = (await session.execute(select(MemoryVector.family_id, MemoryVector.subject_id).distinct())).all()
    for family_id, subject_id in owners:
        got = await reconcile(session, family_id, subject_id)
        out["people"] += 1
        out["relabelled"] += got["relabelled"]
        out["removed"] += got["removed"]
    return out


async def cleanup_on_start(sessions) -> dict | None:
    """Run at every engine start (each deploy): fixes rows written before D-12 (stale "(active)" labels, lines no longer
    in their note) for every family. Idempotent; the first run that changes something leaves a learn_reports row."""
    async with sessions() as session:
        if not (await session.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": CLEANUP_LOCK})).scalar():
            return None  # another instance is doing it
        out = await cleanup_all(session)
        if out["relabelled"] or out["removed"]:
            from app.learn.models import LearnReport

            session.add(LearnReport(kind="index_cleanup", at=clock.now(), data=out))
        await session.commit()
    logger.info("memory index cleanup %s", out)
    return out


async def _current_note_lines(session: AsyncSession, family_id: str, subject_ids: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for n in await store.notes(session, family_id, subject_ids):
        out.setdefault(f"note:{n.slug}", set()).update(_line_texts(n.title, n.body_md))
    return out


async def _facts_by_ref(session: AsyncSession, family_id: str, refs: list[str]) -> dict[str, CareFact]:
    ids = []
    for ref in refs:
        try:
            ids.append(uuid.UUID(ref.split(":", 1)[1]))
        except (ValueError, IndexError):
            pass
    if not ids:
        return {}
    rows = (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.id.in_(ids)))).scalars()
    return {f"fact:{f.id}": f for f in rows}


async def search(session: AsyncSession, family_id: str, subject_ids: list[str], query: str, *, limit: int = 8) -> list[store.Hit]:
    """Hybrid recall: meaning (if indexed) + keywords, weighted by importance and recency (reciprocal rank fusion).

    Every hit says what is true now, whatever the index holds: a fact found by meaning is written from the care record
    as it is (an earlier value says when it ended and ranks as an earlier value; a turned-down one is dropped), and a
    note line found by meaning must still be in its note (dated diary lines excepted)."""
    keyword = await store.recall(session, family_id, subject_ids, query, limit=limit * 3)
    scores: dict[str, float] = {}
    hits: dict[str, store.Hit] = {}
    weight: dict[str, float] = {}
    for rank, h in enumerate(keyword):
        scores[h.ref] = scores.get(h.ref, 0) + 1.0 / (60 + rank)
        hits[h.ref] = h
    vec = await embed([query], query=True) if query.strip() else None
    if vec:
        rows = list((await session.execute(
            select(MemoryVector).where(MemoryVector.family_id == family_id, MemoryVector.subject_id.in_(subject_ids),
                                       MemoryVector.embedding.is_not(None))
            .order_by(MemoryVector.embedding.cosine_distance(vec[0])).limit(limit * 3)
        )).scalars())
        lines = await _current_note_lines(session, family_id, subject_ids) if any(r.source == "note" for r in rows) else {}
        facts = await _facts_by_ref(session, family_id, [r.ref for r in rows if r.source == "fact"])
        rank = 0
        for r in rows:
            hit, imp, at = store.Hit(r.source, r.at, r.text, r.ref), r.importance, r.at
            if r.source == "note" and r.ref.split(":", 1)[-1] not in HISTORY_NOTES and r.text not in lines.get(r.ref, set()):
                continue  # this line is no longer in its note
            if r.source == "fact":
                f = facts.get(r.ref)
                item = fact_item(f) if f is not None else None
                if item is None:
                    continue  # the fact is gone, or was a proposal that was turned down
                hit, imp, at = store.Hit("fact", f.recorded_at, store.fact_label(f), r.ref), item.importance, item.at
            scores[r.ref] = scores.get(r.ref, 0) + 1.0 / (60 + rank)
            weight[r.ref] = imp * decay(at, imp)
            hits.setdefault(r.ref, hit)
            rank += 1
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
    """Bring every row in line with the record and the notes, index the last two days (everything the first time), and
    embed rows that have none yet."""
    any_row = (await session.execute(select(MemoryVector.key).where(MemoryVector.family_id == family_id,
                                                                     MemoryVector.subject_id == subject_id).limit(1))).first()
    since = None if not any_row else clock.now() - timedelta(days=2)
    await reconcile(session, family_id, subject_id)
    added = await index_subject(session, family_id, subject_id, since=since)
    await fill_embeddings(session, family_id, subject_id)
    return added
