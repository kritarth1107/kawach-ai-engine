"""Corrections, found by code from memory history (trust plan C1 check 3 and A2 truth 5).

A correction is a change a PERSON made to the care record (never an inference, import, the nightly job or Saheli on
her own):
- change: a fact a person said or set replaced an earlier value of the same key (chat or dashboard; it is pinned, A3);
- naming: a name to use or a name never to use, set by the person or their family (even the first time);
- stop: a person ended a fact (a stopped medicine, a lifted diet rule, a fact stopped as wrong);
- reject: a caregiver turned down a proposed fact (it goes on the "never say" list).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.domains import PERSON_SOURCES
from app.care.models import CareFact
from app.care.versions import MemoryVersion

NOT_A_PERSON = {"saheli", "nightly", "system", "import", "snapshot"}


@dataclass
class Correction:
    family_id: str
    subject_id: str
    key: str
    domain: str
    kind: str  # change | naming | stop | reject
    at: datetime
    by: str  # who made it (a person's id)
    source: str  # whatsapp | dashboard | ...
    old: str  # what must not be used any more (the old text, or the rejected/stopped item)
    new: str  # the value now ("" for a stop or a rejection)
    old_value: dict = field(default_factory=dict)
    new_value: dict = field(default_factory=dict)
    version_id: int = 0
    fact_id: str | None = None
    avoid_added: list[str] = field(default_factory=list)  # naming: the names newly put on the never list
    pinned: bool = False  # filled by the memory score: the fact it made is pinned now
    still_active: bool = True  # filled by the memory score: the corrected value is still the one in effect

    def as_dict(self) -> dict:
        return {"key": self.key, "domain": self.domain, "kind": self.kind, "at": self.at.isoformat(), "old": self.old[:300],
                "new": self.new[:300], "avoid_added": self.avoid_added, "by": self.by, "version": self.version_id}


def _by_person(v: MemoryVersion) -> bool:
    return (v.source or "") not in NOT_A_PERSON and (v.actor_id or "") not in ("", "saheli")


def _avoid(value: dict | None) -> list[str]:
    return [w.strip() for w in ((value or {}).get("avoid") or []) if isinstance(w, str) and w.strip()]


async def find(session: AsyncSession, family_id: str, since: datetime, until: datetime, *, subject_id: str | None = None) -> list[Correction]:
    """Every correction in [since, until), oldest first."""
    q = select(MemoryVersion).where(MemoryVersion.family_id == family_id, MemoryVersion.kind == "fact", MemoryVersion.at >= since,
                                    MemoryVersion.at < until, MemoryVersion.op.in_(("write", "stop", "reject")))
    if subject_id:
        q = q.where(MemoryVersion.subject_id == subject_id)
    vs = list((await session.execute(q.order_by(MemoryVersion.id))).scalars())
    if not vs:
        return []
    ids = []
    for v in vs:
        try:
            ids.append(uuid.UUID(str(v.ref)))
        except (TypeError, ValueError):
            continue
    rows = {str(r.id): r for r in (await session.execute(select(CareFact).where(CareFact.id.in_(ids)))).scalars()} if ids else {}
    out: list[Correction] = []
    for v in vs:
        row = rows.get(str(v.ref or ""))
        domain = v.title or v.target.split(":", 1)[0]
        if v.op == "reject":
            out.append(Correction(family_id, v.subject_id, v.target, domain, "reject", v.at, v.actor_id or "", v.source or "", v.body or "", "",
                                  dict(v.value or {}), {}, v.id, v.ref))
            continue
        if not _by_person(v):
            continue
        if v.op == "stop":
            if (v.status or "") != "stopped":
                continue
            out.append(Correction(family_id, v.subject_id, v.target, domain, "stop", v.at, v.actor_id or "", v.source or "", v.body or "", "",
                                  dict(v.value or {}), {}, v.id, v.ref))
            continue
        # a write: only a person's own statement, and only when it changes something already there (or sets naming)
        if row is None or row.source_kind not in PERSON_SOURCES or (v.status or "") != "active":
            continue
        prev = (await session.execute(
            select(MemoryVersion).where(MemoryVersion.family_id == family_id, MemoryVersion.subject_id == v.subject_id, MemoryVersion.kind == "fact",
                                        MemoryVersion.target == v.target, MemoryVersion.id < v.id, MemoryVersion.status == "active")
            .order_by(MemoryVersion.id.desc()).limit(1)
        )).scalar_one_or_none()
        if domain == "naming":
            added = [w for w in _avoid(v.value) if w.lower() not in {x.lower() for x in _avoid(prev.value if prev else None)}]
            name_changed = bool(prev) and (prev.value or {}).get("name") != (v.value or {}).get("name")
            if added or name_changed:
                old_name = str((prev.value or {}).get("name") or "") if prev and name_changed else ""
                out.append(Correction(family_id, v.subject_id, v.target, domain, "naming", v.at, v.actor_id or "", v.source or "",
                                      ", ".join([*added, *([old_name] if old_name else [])]), v.body or "", dict(prev.value or {}) if prev else {},
                                      dict(v.value or {}), v.id, v.ref, added))
            continue
        if prev is None or dict(prev.value or {}) == dict(v.value or {}):
            continue
        out.append(Correction(family_id, v.subject_id, v.target, domain, "change", v.at, v.actor_id or "", v.source or "", prev.body or "", v.body or "",
                              dict(prev.value or {}), dict(v.value or {}), v.id, v.ref))
    return out


async def latest(session: AsyncSession, family_id: str, subject_id: str, since: datetime, until: datetime) -> Correction | None:
    found = await find(session, family_id, since, until, subject_id=subject_id)
    return found[-1] if found else None
