"""Care Memory reads and writes. Every function takes the session; callers own the transaction."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.domains import medicine_slug, merged_value, needs_confirmation
from app.care.models import CareEvent, CareFact, MemoryNote, OpenLoop, ThreadSummary, Turn
from app.care.redact import scrub_secrets
from app.core import clock


@dataclass
class FactWrite:
    result: str  # created | unchanged | superseded | pending | stopped
    fact: CareFact
    previous: CareFact | None = None


def _same(a: dict, b: dict) -> bool:
    return {k: v for k, v in (a or {}).items() if v not in (None, "", [])} == {
        k: v for k, v in (b or {}).items() if v not in (None, "", [])
    }


async def active_fact(session: AsyncSession, family_id: str, subject_id: str, key: str) -> CareFact | None:
    row = await _active_by_key(session, family_id, subject_id, key)
    if row or not key.startswith("medicine:"):
        return row
    # Older records may carry a dose in the key (medicine:thyronorm_50): match on the medicine itself.
    want = key.split(":", 1)[1]
    for f in (
        await session.execute(
            select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.domain == "medicine", CareFact.status == "active").with_for_update()
        )
    ).scalars():
        if medicine_slug(f.key.split(":", 1)[1]) == want:
            return f
    return None


async def _active_by_key(session: AsyncSession, family_id: str, subject_id: str, key: str) -> CareFact | None:
    return (
        await session.execute(
            select(CareFact)
            .where(CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.key == key)
            .where(CareFact.status == "active")
            .with_for_update()
        )
    ).scalar_one_or_none()


async def write_fact(
    session: AsyncSession,
    *,
    family_id: str,
    subject_id: str,
    domain: str,
    key: str,
    value: dict,
    text: str,
    source_kind: str,
    source_ref: str | None = None,
    stated_by: str | None = None,
    confidence: float = 1.0,
    note: str | None = None,
) -> FactWrite:
    """Save a fact. A change supersedes the old row; a weaker source changing a health fact waits as pending."""
    now = clock.now()
    old = await active_fact(session, family_id, subject_id, key)
    if old:
        key = old.key
        value = merged_value(old.value, value)
    if old and _same(old.value, value):
        if source_kind in ("caregiver_said", "dashboard", "prescription") and stated_by and not old.confirmed_by:
            old.confirmed_by = stated_by
        return FactWrite("unchanged", old)

    row = CareFact(
        id=uuid.uuid4(),
        family_id=family_id,
        subject_id=subject_id,
        domain=domain,
        key=key,
        value=value,
        text=text,
        source_kind=source_kind,
        source_ref=source_ref,
        stated_by=stated_by,
        confirmed_by=stated_by if source_kind in ("caregiver_said", "dashboard", "prescription") else None,
        confidence=confidence,
        valid_from=now,
        recorded_at=now,
        supersedes=old.id if old else None,
        note=note,
    )
    if old and needs_confirmation(domain, source_kind, old.source_kind, changes_existing=True):
        row.status = "pending"
        session.add(row)
        await session.flush()
        await open_loop(
            session,
            family_id=family_id,
            subject_id=subject_id,
            kind="confirm_fact",
            title=f"Confirm change: {old.text} → {text}",
            detail={"fact_id": str(row.id), "key": key, "old": old.value, "new": value},
            dedupe_key=f"confirm:{subject_id}:{key}",
        )
        return FactWrite("pending", row, old)

    if old:
        old.status = "superseded"
        old.valid_to = now
        await session.flush()
    row.status = "active"
    session.add(row)
    await session.flush()
    return FactWrite("superseded" if old else "created", row, old)


async def stop_fact(
    session: AsyncSession,
    *,
    family_id: str,
    subject_id: str,
    key: str,
    reason: str,
    source_kind: str,
    stated_by: str | None = None,
    source_ref: str | None = None,
) -> FactWrite | None:
    """End a fact (a stopped medicine, a lifted diet rule). A weak source only proposes the stop."""
    old = await active_fact(session, family_id, subject_id, key)
    if not old:
        return None
    key = old.key
    now = clock.now()
    if needs_confirmation(old.domain, source_kind, old.source_kind, changes_existing=True):
        row = CareFact(
            id=uuid.uuid4(),
            family_id=family_id,
            subject_id=subject_id,
            domain=old.domain,
            key=key,
            value={**old.value, "stopped": True},
            text=f"Stopped: {old.text}",
            status="pending",
            source_kind=source_kind,
            source_ref=source_ref,
            stated_by=stated_by,
            confidence=0.7,
            valid_from=now,
            recorded_at=now,
            supersedes=old.id,
            note=reason,
        )
        session.add(row)
        await session.flush()
        await open_loop(
            session,
            family_id=family_id,
            subject_id=subject_id,
            kind="confirm_fact",
            title=f"Confirm stop: {old.text} ({reason})",
            detail={"fact_id": str(row.id), "key": key, "old": old.value, "stop": True},
            dedupe_key=f"confirm:{subject_id}:{key}",
        )
        return FactWrite("pending", row, old)
    old.status = "stopped"
    old.valid_to = now
    old.note = reason
    await session.flush()
    return FactWrite("stopped", old)


async def resolve_pending(session: AsyncSession, *, fact_id: uuid.UUID, approve: bool, by: str) -> CareFact | None:
    row = await session.get(CareFact, fact_id, with_for_update=True)
    if not row or row.status != "pending":
        return None
    now = clock.now()
    if approve:
        old = await active_fact(session, row.family_id, row.subject_id, row.key)
        if old and not row.value.get("stopped"):
            # Approving a change keeps what it did not mention (times, dose) from the current record.
            row.value = merged_value(old.value, row.value)
            row.key = old.key
        if old:
            old.status = "superseded"
            old.valid_to = now
            await session.flush()
        row.status = "stopped" if row.value.get("stopped") else "active"
        if row.status == "stopped":
            row.valid_to = now
        row.confirmed_by = by
    else:
        row.status = "retracted"
        row.valid_to = now
        row.confirmed_by = by
    await session.execute(
        update(OpenLoop)
        .where(OpenLoop.family_id == row.family_id, OpenLoop.kind == "confirm_fact", OpenLoop.status == "open")
        .where(OpenLoop.detail["fact_id"].astext == str(row.id))
        .values(status="done", updated_at=now, closed_note="approved" if approve else "rejected")
    )
    await session.flush()
    return row


async def facts(
    session: AsyncSession,
    family_id: str,
    subject_id: str,
    *,
    domains: list[str] | None = None,
    statuses: tuple[str, ...] = ("active", "pending"),
) -> list[CareFact]:
    q = select(CareFact).where(
        CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.status.in_(statuses)
    )
    if domains:
        q = q.where(CareFact.domain.in_(domains))
    return list((await session.execute(q.order_by(CareFact.domain, CareFact.key, CareFact.recorded_at))).scalars())


async def fact_history(session: AsyncSession, family_id: str, subject_id: str, key: str) -> list[CareFact]:
    q = (
        select(CareFact)
        .where(CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.key == key)
        .order_by(CareFact.recorded_at)
    )
    return list((await session.execute(q)).scalars())


# ── events ─────────────────────────────────────────────────────────────────────


async def record_event(
    session: AsyncSession,
    *,
    family_id: str,
    subject_id: str,
    kind: str,
    summary: str = "",
    payload: dict | None = None,
    actor_id: str | None = None,
    ref: str | None = None,
    at: datetime | None = None,
) -> int | None:
    """Append to the ledger. A repeated ref (webhook retry, scheduler re-run) is ignored."""
    when = at or clock.now()
    stmt = (
        insert(CareEvent)
        .values(
            family_id=family_id,
            subject_id=subject_id,
            actor_id=actor_id,
            kind=kind,
            at=when,
            day=clock.ist_day(when),
            summary=summary,
            payload=payload or {},
            ref=ref,
            tsv=func.to_tsvector("simple", summary),
        )
        .on_conflict_do_nothing()
        .returning(CareEvent.id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def events(
    session: AsyncSession,
    family_id: str,
    subject_id: str,
    *,
    day: str | None = None,
    since: datetime | None = None,
    kinds: list[str] | None = None,
    limit: int = 200,
) -> list[CareEvent]:
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id)
    if day:
        q = q.where(CareEvent.day == day)
    if since:
        q = q.where(CareEvent.at >= since)
    if kinds:
        q = q.where(CareEvent.kind.in_(kinds))
    return list((await session.execute(q.order_by(CareEvent.at).limit(limit))).scalars())


# ── notes ──────────────────────────────────────────────────────────────────────


async def upsert_note(
    session: AsyncSession, *, family_id: str, subject_id: str, slug: str, title: str, body_md: str
) -> MemoryNote:
    now = clock.now()
    stmt = (
        insert(MemoryNote)
        .values(
            id=uuid.uuid4(),
            family_id=family_id,
            subject_id=subject_id,
            slug=slug,
            title=title,
            body_md=body_md,
            version=1,
            updated_at=now,
            tsv=func.to_tsvector("simple", f"{title}\n{body_md}"),
        )
        .on_conflict_do_update(
            index_elements=["family_id", "subject_id", "slug"],
            set_={
                "title": title,
                "body_md": body_md,
                "version": MemoryNote.version + 1,
                "updated_at": now,
                "tsv": func.to_tsvector("simple", f"{title}\n{body_md}"),
            },
        )
        .returning(MemoryNote)
    )
    # populate_existing: a note loaded earlier in this session must show what was just written.
    return (await session.execute(stmt, execution_options={"populate_existing": True})).scalar_one()


async def notes(session: AsyncSession, family_id: str, subject_ids: list[str]) -> list[MemoryNote]:
    q = (
        select(MemoryNote)
        .where(MemoryNote.family_id == family_id, MemoryNote.subject_id.in_(subject_ids))
        .order_by(MemoryNote.subject_id, MemoryNote.slug)
        .execution_options(populate_existing=True)  # notes are written with upserts; never return a stale copy
    )
    return list((await session.execute(q)).scalars())


# ── open loops ─────────────────────────────────────────────────────────────────


async def open_loop(
    session: AsyncSession,
    *,
    family_id: str,
    subject_id: str,
    kind: str,
    title: str,
    detail: dict | None = None,
    owner_id: str | None = None,
    wake_at: datetime | None = None,
    alert_rule: str | None = None,
    dedupe_key: str | None = None,
) -> OpenLoop:
    if dedupe_key:
        existing = (
            await session.execute(
                select(OpenLoop).where(
                    OpenLoop.family_id == family_id, OpenLoop.dedupe_key == dedupe_key, OpenLoop.status == "open"
                )
            )
        ).scalar_one_or_none()
        if existing:
            existing.title = title
            existing.detail = detail or existing.detail
            existing.wake_at = wake_at or existing.wake_at
            existing.updated_at = clock.now()
            return existing
    now = clock.now()
    loop = OpenLoop(
        id=uuid.uuid4(),
        family_id=family_id,
        subject_id=subject_id,
        kind=kind,
        title=title,
        detail=detail or {},
        owner_id=owner_id,
        wake_at=wake_at,
        alert_rule=alert_rule,
        dedupe_key=dedupe_key,
        created_at=now,
        updated_at=now,
    )
    session.add(loop)
    await session.flush()
    return loop


async def close_loop(session: AsyncSession, loop_id: uuid.UUID, *, status: str = "done", note: str = "") -> OpenLoop | None:
    loop = await session.get(OpenLoop, loop_id)
    if not loop or loop.status != "open":
        return None
    loop.status = status
    loop.closed_note = note
    loop.updated_at = clock.now()
    await session.flush()
    return loop


async def live_loops(session: AsyncSession, family_id: str, subject_ids: list[str]) -> list[OpenLoop]:
    q = (
        select(OpenLoop)
        .where(OpenLoop.family_id == family_id, OpenLoop.subject_id.in_(subject_ids), OpenLoop.status == "open")
        .order_by(OpenLoop.created_at)
    )
    return list((await session.execute(q)).scalars())


async def due_loops(session: AsyncSession, *, before: datetime, limit: int = 100) -> list[OpenLoop]:
    q = (
        select(OpenLoop)
        .where(OpenLoop.status == "open", OpenLoop.wake_at.is_not(None), OpenLoop.wake_at <= before)
        .order_by(OpenLoop.wake_at)
        .limit(limit)
    )
    return list((await session.execute(q)).scalars())


# ── working context ────────────────────────────────────────────────────────────


async def add_turn(
    session: AsyncSession,
    *,
    family_id: str,
    thread_id: str,
    role: str,
    text: str,
    speaker_id: str | None = None,
    meta: dict | None = None,
    message_ref: str | None = None,
) -> int | None:
    stmt = (
        insert(Turn)
        .values(
            family_id=family_id,
            thread_id=thread_id,
            speaker_id=speaker_id,
            role=role,
            text=scrub_secrets(text) if role == "user" else text,
            at=clock.now(),
            meta=meta or {},
            message_ref=message_ref,
        )
        .on_conflict_do_nothing()
        .returning(Turn.id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def recent_turns(session: AsyncSession, family_id: str, thread_id: str, *, after_id: int = 0, limit: int = 60) -> list[Turn]:
    q = (
        select(Turn)
        .where(Turn.family_id == family_id, Turn.thread_id == thread_id, Turn.id > after_id)
        .order_by(Turn.id.desc())
        .limit(limit)
    )
    return list(reversed(list((await session.execute(q)).scalars())))


async def recent_turns_many(session: AsyncSession, family_id: str, thread_ids: list[str], *, limit: int = 40) -> dict[str, list[Turn]]:
    """The last `limit` turns of several threads in one query (oldest first per thread)."""
    from sqlalchemy.orm import aliased

    ids = sorted({t for t in thread_ids if t})
    if not ids:
        return {}
    rn = func.row_number().over(partition_by=Turn.thread_id, order_by=Turn.id.desc()).label("rn")
    sub = select(Turn, rn).where(Turn.family_id == family_id, Turn.thread_id.in_(ids)).subquery()
    t = aliased(Turn, sub)
    rows = list((await session.execute(select(t).where(sub.c.rn <= limit).order_by(t.thread_id, t.id))).scalars())
    out: dict[str, list[Turn]] = {i: [] for i in ids}
    for r in rows:
        out[r.thread_id].append(r)
    return out


async def summary(session: AsyncSession, family_id: str, thread_id: str) -> ThreadSummary | None:
    return await session.get(ThreadSummary, (family_id, thread_id))


async def set_summary(session: AsyncSession, family_id: str, thread_id: str, text: str, covers_until_turn: int) -> None:
    stmt = (
        insert(ThreadSummary)
        .values(family_id=family_id, thread_id=thread_id, summary=text, covers_until_turn=covers_until_turn, updated_at=clock.now())
        .on_conflict_do_update(
            index_elements=["family_id", "thread_id"],
            set_={"summary": text, "covers_until_turn": covers_until_turn, "updated_at": clock.now()},
        )
    )
    await session.execute(stmt)


# ── recall ─────────────────────────────────────────────────────────────────────


@dataclass
class Hit:
    source: str  # fact | note | event
    when: datetime
    text: str
    ref: str


async def recall(
    session: AsyncSession,
    family_id: str,
    subject_ids: list[str],
    query: str,
    *,
    limit: int = 12,
    include_past_facts: bool = True,
) -> list[Hit]:
    """Keyword search over notes, the ledger, and the care record including past values."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in query.lower()).split() if len(w) > 1]
    if not words:
        return []
    tsq = func.to_tsquery("simple", " | ".join(f"{w}:*" for w in words[:12]))
    hits: list[Hit] = []

    note_rows = await session.execute(
        select(MemoryNote, func.ts_rank(MemoryNote.tsv, tsq).label("r"))
        .where(MemoryNote.family_id == family_id, MemoryNote.subject_id.in_(subject_ids), MemoryNote.tsv.op("@@")(tsq))
        .order_by(func.ts_rank(MemoryNote.tsv, tsq).desc())
        .limit(limit)
    )
    for n, _ in note_rows:
        hits.append(Hit("note", n.updated_at, f"{n.title}: {n.body_md[:600]}", f"note:{n.slug}"))

    ev_rows = await session.execute(
        select(CareEvent)
        .where(CareEvent.family_id == family_id, CareEvent.subject_id.in_(subject_ids), CareEvent.tsv.op("@@")(tsq),
               CareEvent.payload["forgotten"].astext.is_(None))
        .order_by(func.ts_rank(CareEvent.tsv, tsq).desc(), CareEvent.at.desc())
        .limit(limit)
    )
    for e in ev_rows.scalars():
        hits.append(Hit("event", e.at, f"[{e.kind}] {e.summary}", f"event:{e.id}"))

    statuses = ("active", "pending", "superseded", "stopped") if include_past_facts else ("active", "pending")
    like = or_(*[func.lower(CareFact.text).contains(w) for w in words[:12]], *[CareFact.key.contains(w) for w in words[:12]])
    fact_rows = await session.execute(
        select(CareFact)
        .where(CareFact.family_id == family_id, CareFact.subject_id.in_(subject_ids), CareFact.status.in_(statuses), like)
        .order_by(CareFact.recorded_at.desc())
        .limit(limit)
    )
    for f in fact_rows.scalars():
        span = f"{clock.ist_day(f.valid_from)}→{clock.ist_day(f.valid_to) if f.valid_to else 'now'}"
        hits.append(Hit("fact", f.recorded_at, f"({f.status}, {span}, {f.source_kind}) {f.text}", f"fact:{f.id}"))

    hits.sort(key=lambda h: h.when, reverse=True)
    return hits[:limit]


async def stale_unanswered(session: AsyncSession, family_id: str, subject_id: str, *, older_than: timedelta) -> list[OpenLoop]:
    cutoff = clock.now() - older_than
    q = select(OpenLoop).where(
        and_(
            OpenLoop.family_id == family_id,
            OpenLoop.subject_id == subject_id,
            OpenLoop.kind == "question",
            OpenLoop.status == "open",
            OpenLoop.created_at <= cutoff,
        )
    )
    return list((await session.execute(q)).scalars())


# ── roster ─────────────────────────────────────────────────────────────────────


async def save_roster(session: AsyncSession, family_id: str, elder: dict, members: list[dict]) -> None:
    from app.care.models import FamilyRoster

    stmt = (
        insert(FamilyRoster)
        .values(family_id=family_id, elder=elder, members=members, updated_at=clock.now())
        .on_conflict_do_update(index_elements=["family_id"], set_={"elder": elder, "members": members, "updated_at": clock.now()})
    )
    await session.execute(stmt)


async def roster(session: AsyncSession, family_id: str):
    from app.care.models import FamilyRoster

    return await session.get(FamilyRoster, family_id)
