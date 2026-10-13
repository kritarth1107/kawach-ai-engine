"""Saheli's memory as it stood at one moment, rebuilt from memory_versions (trust plan C1.3).

The full-memory judge grades a reply against what Saheli knew when she wrote it, not what she knows tonight: the care
record (active and pending facts), the memory notes, the links between record items and the "never say" list.

- Facts: every change to a fact is a version (write, pending, stop, approve, reject, archive). Replaying a key's
  versions up to the moment gives its active row and any pending rows then. A key with no version before the moment
  is read from its care_facts rows (facts saved before history started): a row counts as active from valid_from until
  valid_to.
- Notes: the newest version at or before the moment holds the whole note as it was; a note with no history then is
  taken as it is now if it was last changed before the moment.
- Links: care_links rows carry their own valid_from/valid_to, so a link is shown if it was in effect then.
- Rejected: what the family had said was wrong by then.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import CareFact, CareLink, MemoryNote
from app.care.versions import MemoryVersion

ACTIVE_STATES = ("active",)
GONE_STATES = ("stopped", "retracted", "archived", "superseded")


@dataclass
class Memory:
    at: datetime
    facts: dict[str, list[CareFact]] = field(default_factory=dict)  # subject -> active and pending rows as they were
    notes: list[MemoryNote] = field(default_factory=list)
    links: dict[str, list[CareLink]] = field(default_factory=dict)
    rejected: dict[str, list[CareFact]] = field(default_factory=dict)

    def record(self, subject_id: str, name: str) -> str:
        from app.care import digest

        return digest.care_record(name, self.facts.get(subject_id, []), {}, self.links.get(subject_id, []), rejected=self.rejected.get(subject_id, []))

    def notes_text(self, max_chars: int = 8000) -> str:
        from app.care import digest

        return digest.notes_block(self.notes, max_chars=max_chars)

    def active(self, subject_id: str) -> list[CareFact]:
        return [f for f in self.facts.get(subject_id, []) if f.status == "active"]

    def pending(self, subject_id: str) -> list[CareFact]:
        return [f for f in self.facts.get(subject_id, []) if f.status == "pending"]


def _ref(v: MemoryVersion) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(v.ref)) if v.ref else None
    except ValueError:
        return None


def replay_key(versions: list[MemoryVersion]) -> tuple[MemoryVersion | None, dict[str, MemoryVersion]]:
    """(the active version, {ref: pending version}) after these versions of one fact key, oldest first."""
    active: MemoryVersion | None = None
    pending: dict[str, MemoryVersion] = {}
    for v in versions:
        st = (v.status or "").lower()
        ref = str(v.ref or "")
        if st == "active":
            active = v  # a write (or an approved change) supersedes the row that was active
            pending.pop(ref, None)
        elif st == "pending":
            pending[ref] = v
        elif st in GONE_STATES:
            if active is not None and str(active.ref or "") == ref:
                active = None
            if v.op == "approve" and st == "stopped":
                active = None  # an approved stop ends the fact that was in effect
            pending.pop(ref, None)
    return active, pending


def _fact_from(v: MemoryVersion, status: str, row: CareFact | None, at: datetime) -> CareFact:
    pinned = row.pinned_at if row is not None and row.pinned_at is not None and row.pinned_at <= at else None
    pinned_by = row.pinned_by if row is not None and pinned is not None else None
    return CareFact(
        id=_ref(v) or uuid.uuid4(), family_id=v.family_id, subject_id=v.subject_id, domain=v.title or (v.target.split(":", 1)[0]),
        key=v.target, value=dict(v.value or {}), text=v.body or "", status=status,
        source_kind=row.source_kind if row is not None else "inferred", stated_by=row.stated_by if row is not None else None,
        confirmed_by=row.confirmed_by if row is not None else None, confidence=row.confidence if row is not None else 1.0,
        valid_from=v.at, recorded_at=v.at, pinned_at=pinned, pinned_by=pinned_by, note=row.note if row is not None else None,
    )


def _copy(row: CareFact, status: str) -> CareFact:
    return CareFact(
        id=row.id, family_id=row.family_id, subject_id=row.subject_id, domain=row.domain, key=row.key, value=dict(row.value or {}), text=row.text,
        status=status, source_kind=row.source_kind, stated_by=row.stated_by, confirmed_by=row.confirmed_by, confidence=row.confidence,
        valid_from=row.valid_from, recorded_at=row.recorded_at, pinned_at=row.pinned_at, pinned_by=row.pinned_by, note=row.note,
    )


def _state_without_history(row: CareFact, at: datetime) -> str | None:
    """A row saved before history started: was it active or pending at `at`?"""
    if row.recorded_at > at:
        return None
    ended = row.valid_to is not None and row.valid_to <= at
    if row.status in ("active", "superseded", "stopped") and not ended and (row.valid_from or row.recorded_at) <= at:
        return "active"
    if row.status == "pending" or (row.status in ("retracted", "archived") and not ended):
        return "pending"
    return None


async def facts_as_of(session: AsyncSession, family_id: str, subject_ids: list[str], at: datetime) -> dict[str, list[CareFact]]:
    subject_ids = [s for s in dict.fromkeys(subject_ids) if s]
    vs = list((await session.execute(
        select(MemoryVersion).where(MemoryVersion.family_id == family_id, MemoryVersion.subject_id.in_(subject_ids),
                                    MemoryVersion.kind == "fact", MemoryVersion.at <= at).order_by(MemoryVersion.id)
    )).scalars())
    rows = list((await session.execute(
        select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id.in_(subject_ids), CareFact.recorded_at <= at)
    )).scalars())
    by_id = {r.id: r for r in rows}
    missing = {r for r in (_ref(v) for v in vs) if r is not None and r not in by_id}
    if missing:
        for r in (await session.execute(select(CareFact).where(CareFact.id.in_(missing)))).scalars():
            by_id[r.id] = r
    by_key: dict[tuple[str, str], list[MemoryVersion]] = defaultdict(list)
    for v in vs:
        by_key[(v.subject_id, v.target)].append(v)
    out: dict[str, list[CareFact]] = {s: [] for s in subject_ids}
    for (sid, _key), kvs in by_key.items():
        active, pending = replay_key(kvs)
        if active is not None:
            out[sid].append(_fact_from(active, "active", by_id.get(_ref(active)), at))  # type: ignore[arg-type]
        for p in pending.values():
            out[sid].append(_fact_from(p, "pending", by_id.get(_ref(p)), at))  # type: ignore[arg-type]
    for r in rows:
        if (r.subject_id, r.key) in by_key:
            continue
        st = _state_without_history(r, at)
        if st:
            out.setdefault(r.subject_id, []).append(_copy(r, st))
    for sid in out:
        out[sid].sort(key=lambda f: (f.domain, f.key, f.recorded_at))
    return out


async def notes_as_of(session: AsyncSession, family_id: str, subject_ids: list[str], at: datetime) -> list[MemoryNote]:
    subject_ids = [s for s in dict.fromkeys(subject_ids) if s]
    vs = list((await session.execute(
        select(MemoryVersion).where(MemoryVersion.family_id == family_id, MemoryVersion.subject_id.in_(subject_ids),
                                    MemoryVersion.kind == "note", MemoryVersion.at <= at).order_by(MemoryVersion.id)
    )).scalars())
    later = {(sid, slug) for sid, slug in (await session.execute(
        select(MemoryVersion.subject_id, MemoryVersion.target).where(MemoryVersion.family_id == family_id, MemoryVersion.subject_id.in_(subject_ids),
                                                                     MemoryVersion.kind == "note", MemoryVersion.at > at)
    )).all()}
    newest: dict[tuple[str, str], MemoryVersion] = {}
    for v in vs:
        newest[(v.subject_id, v.target)] = v
    out = [MemoryNote(id=uuid.uuid4(), family_id=family_id, subject_id=sid, slug=slug, title=v.title or slug, body_md=v.body or "", updated_at=v.at)
           for (sid, slug), v in newest.items() if (v.body or "").strip()]
    now_rows = list((await session.execute(
        select(MemoryNote).where(MemoryNote.family_id == family_id, MemoryNote.subject_id.in_(subject_ids))
    )).scalars())
    for n in now_rows:
        key = (n.subject_id, n.slug)
        if key in newest:
            continue
        if key in later and n.updated_at > at:
            continue  # it changed after the moment and has no earlier version: it did not exist yet (or is unknown then)
        if n.updated_at <= at and (n.body_md or "").strip():
            out.append(n)
    return out


async def links_as_of(session: AsyncSession, family_id: str, subject_ids: list[str], at: datetime) -> dict[str, list[CareLink]]:
    rows = list((await session.execute(
        select(CareLink).where(CareLink.family_id == family_id, CareLink.subject_id.in_([s for s in subject_ids if s]), CareLink.recorded_at <= at)
    )).scalars())
    out: dict[str, list[CareLink]] = defaultdict(list)
    for r in rows:
        if r.valid_to is not None and r.valid_to <= at:
            continue
        status = "pending" if r.status in ("pending", "rejected") else "active"
        out[r.subject_id].append(CareLink(id=r.id, family_id=r.family_id, subject_id=r.subject_id, from_ref=r.from_ref, relation=r.relation,
                                          to_ref=r.to_ref, label=r.label, detail=dict(r.detail or {}), status=status, source_kind=r.source_kind,
                                          valid_from=r.valid_from, valid_to=None, recorded_at=r.recorded_at, note=r.note))
    return dict(out)


async def rejected_as_of(session: AsyncSession, family_id: str, subject_id: str, at: datetime) -> list[CareFact]:
    from app.care import rejected

    rows = await rejected.rejected(session, family_id, subject_id, now=at)
    return [r for r in rows if (r.valid_to or r.recorded_at) <= at]


async def memory_at(session: AsyncSession, family_id: str, subject_ids: list[str], at: datetime) -> Memory:
    """The care record, notes ("family" notes included), links and rejected list of these people at `at`."""
    people = [s for s in dict.fromkeys(subject_ids) if s and s != "family"]
    mem = Memory(at=at)
    mem.facts = await facts_as_of(session, family_id, people, at)
    mem.notes = await notes_as_of(session, family_id, [*people, "family"], at)
    mem.links = await links_as_of(session, family_id, people, at)
    for sid in people:
        mem.rejected[sid] = await rejected_as_of(session, family_id, sid, at)
    return mem
