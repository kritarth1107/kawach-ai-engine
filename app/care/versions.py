"""Version history for everything Saheli remembers, with undo (memory you can roll back, kept in Postgres).

Every change to a memory note (stories, diary, weekly and monthly summaries, life so far, profile card), a care-record
fact, a family skill, or a person's learned reply style is one row in memory_versions: what it became, who changed it,
from where, and why. Writers don't pass that along by hand: tools, the dashboard and the nightly job set it once with
`attribution(...)` and every write inside picks it up.

- undo(change) reverses one change. Notes go back line by line, so undoing Monday's wrong line keeps what came after.
  Undoing an undo puts the change back.
- restore(version) puts a note, a skill or a fact back the way it was at that version.
- Facts are undone through the care record's own rules (app.brain.tools carries out the plan made here): a weaker
  source can only propose a health change, reminders follow the record, and an undo never silently ends or restarts a
  medicine, allergy or condition; it waits for a caregiver's OK unless a caregiver confirmed it on the dashboard.
- Style is relearned every night, so it has history but no undo (a family skill overrides it).
- Versions older than a year are pruned (the newest of each item is always kept); the nightly per-family snapshot in
  the backup bucket keeps the long tail.
"""

from __future__ import annotations

import contextlib
import contextvars
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, delete, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core import clock
from app.db.session import Base

KINDS = ("note", "fact", "skill", "style")
KEEP_FOR = timedelta(days=365)
HEALTH_DOMAINS = ("medicine", "allergy", "condition", "vital_target")
LIST_LINES = 5


class MemoryVersion(Base):
    __tablename__ = "memory_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(8))  # note | fact | skill | style
    target: Mapped[str] = mapped_column(String(200))  # note slug | fact key | skill id | "style"
    version: Mapped[int] = mapped_column(Integer, default=1)
    # write | baseline | forget | restore | undo | snapshot (notes, skills) — write | pending | stop | approve | reject (facts)
    op: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(255), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ref: Mapped[str | None] = mapped_column(String(64), nullable=True)  # care_facts row id
    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(24), default="system")  # whatsapp | dashboard | nightly | saheli | snapshot | system
    reason: Mapped[str] = mapped_column(Text, default="")
    undoes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_memory_versions_target", "family_id", "subject_id", "kind", "target", "id"),
        Index("ix_memory_versions_family_at", "family_id", "at"),
    )


# ── who is changing memory right now ─────────────────────────────────────────


@dataclass
class Attribution:
    actor_id: str | None = None
    source: str = "system"
    reason: str = ""
    undoes: int | None = None


_current: contextvars.ContextVar[Attribution] = contextvars.ContextVar("memory_attribution", default=Attribution())


@contextlib.contextmanager
def attribution(*, actor_id: str | None = None, source: str | None = None, reason: str | None = None, undoes: int | None = None):
    """Every memory write inside carries this actor, source and reason (inner blocks override outer ones)."""
    outer = _current.get()
    token = _current.set(Attribution(
        actor_id=actor_id if actor_id is not None else outer.actor_id,
        source=source or outer.source,
        reason=reason if reason is not None else outer.reason,
        undoes=undoes if undoes is not None else outer.undoes,
    ))
    try:
        yield
    finally:
        _current.reset(token)


def current() -> Attribution:
    return _current.get()


# ── writing versions ─────────────────────────────────────────────────────────


async def _next_version(session: AsyncSession, family_id: str, subject_id: str, kind: str, target: str) -> int:
    n = (await session.execute(select(func.max(MemoryVersion.version)).where(
        MemoryVersion.family_id == family_id, MemoryVersion.subject_id == subject_id,
        MemoryVersion.kind == kind, MemoryVersion.target == target))).scalar()
    return (n or 0) + 1


async def _has_history(session: AsyncSession, family_id: str, subject_id: str, kind: str, target: str) -> bool:
    return (await session.execute(select(MemoryVersion.id).where(
        MemoryVersion.family_id == family_id, MemoryVersion.subject_id == subject_id,
        MemoryVersion.kind == kind, MemoryVersion.target == target).limit(1))).first() is not None


async def add(session: AsyncSession, *, family_id: str, subject_id: str, kind: str, target: str, op: str, title: str = "",
              body: str = "", value: dict | None = None, status: str | None = None, ref: str | None = None,
              actor_id: str | None = None, source: str | None = None, reason: str | None = None, undoes: int | None = None,
              at: datetime | None = None) -> MemoryVersion:
    a = current()
    row = MemoryVersion(
        family_id=family_id, subject_id=subject_id, kind=kind, target=target[:200],
        version=await _next_version(session, family_id, subject_id, kind, target[:200]), op=op, title=(title or "")[:255],
        body=body or "", value=value, status=status, ref=ref, actor_id=actor_id if actor_id is not None else a.actor_id,
        source=(source or a.source or "system")[:24], reason=(reason if reason is not None else a.reason or "")[:500],
        undoes=undoes if undoes is not None else a.undoes, at=at or clock.now(),
    )
    session.add(row)
    await session.flush()
    return row


async def before_note(session: AsyncSession, note) -> None:
    """A note written before history existed: keep how it was, so the first change can be undone."""
    if note and not await _has_history(session, note.family_id, note.subject_id, "note", note.slug):
        await add(session, family_id=note.family_id, subject_id=note.subject_id, kind="note", target=note.slug, op="baseline",
                  title=note.title, body=note.body_md, actor_id="", source="system", reason="as it was before history started",
                  undoes=None, at=note.updated_at)


async def record_note(session: AsyncSession, note, *, op: str = "write", meta: dict | None = None, **kw) -> MemoryVersion:
    return await add(session, family_id=note.family_id, subject_id=note.subject_id, kind="note", target=note.slug, op=op,
                     title=note.title, body=note.body_md, value=meta or None, **kw)


async def record_fact(session: AsyncSession, row, op: str) -> MemoryVersion:
    """One care-record change: the row it made (or changed) and what it says now."""
    a = current()
    return await add(session, family_id=row.family_id, subject_id=row.subject_id, kind="fact", target=row.key, op=op,
                     title=row.domain, body=row.text or "", value=dict(row.value or {}), status=row.status, ref=str(row.id),
                     actor_id=a.actor_id or row.confirmed_by or row.stated_by, reason=a.reason or (row.note or ""))


def skill_state(s) -> dict:
    return {"title": s.title, "body": s.body, "status": s.status, "source": s.source}


async def record_skill(session: AsyncSession, s, op: str, *, prior: dict | None = None, **kw) -> MemoryVersion | None:
    """A family skill changed. `prior` is how it was just before, kept as the baseline the first time."""
    if s.scope != "family" or not s.family_id:
        return None
    subject = s.subject_id or ""
    if prior and not await _has_history(session, s.family_id, subject, "skill", str(s.id)):
        await add(session, family_id=s.family_id, subject_id=subject, kind="skill", target=str(s.id), op="baseline",
                  title=prior["title"], body=prior["body"], status=prior["status"], value={"source": prior.get("source")},
                  actor_id="", source="system", reason="as it was before history started", undoes=None)
    return await add(session, family_id=s.family_id, subject_id=subject, kind="skill", target=str(s.id), op=op, title=s.title,
                     body=s.body, status=s.status, value={"source": s.source}, **kw)


async def record_style(session: AsyncSession, family_id: str, subject_id: str, old: dict | None, new: dict | None) -> MemoryVersion | None:
    """How a person likes replies, relearned nightly: a version only when the summary actually changes."""
    before, after = (old or {}).get("summary") or "", (new or {}).get("summary") or ""
    if before == after:
        return None
    if before and not await _has_history(session, family_id, subject_id, "style", "style"):
        await add(session, family_id=family_id, subject_id=subject_id, kind="style", target="style", op="baseline", title="Reply style",
                  body=before, value=dict(old or {}), actor_id="", source="system", reason="as it was before history started", undoes=None)
    return await add(session, family_id=family_id, subject_id=subject_id, kind="style", target="style", op="write", title="Reply style",
                     body=after, value=dict(new or {}), source="nightly", reason="relearned from their recent messages")


# ── reading history ──────────────────────────────────────────────────────────


async def get(session: AsyncSession, family_id: str, version_id: int) -> MemoryVersion | None:
    row = await session.get(MemoryVersion, version_id)
    return row if row and row.family_id == family_id else None


async def previous(session: AsyncSession, v: MemoryVersion, *, skip_ops: tuple[str, ...] = ()) -> MemoryVersion | None:
    q = select(MemoryVersion).where(
        MemoryVersion.family_id == v.family_id, MemoryVersion.subject_id == v.subject_id, MemoryVersion.kind == v.kind,
        MemoryVersion.target == v.target, MemoryVersion.id < v.id)
    if skip_ops:
        q = q.where(MemoryVersion.op.not_in(skip_ops))
    return (await session.execute(q.order_by(MemoryVersion.id.desc()).limit(1))).scalar_one_or_none()


async def latest(session: AsyncSession, family_id: str, subject_id: str, kind: str, target: str) -> MemoryVersion | None:
    return (await session.execute(select(MemoryVersion).where(
        MemoryVersion.family_id == family_id, MemoryVersion.subject_id == subject_id, MemoryVersion.kind == kind,
        MemoryVersion.target == target).order_by(MemoryVersion.id.desc()).limit(1))).scalar_one_or_none()


async def changes(session: AsyncSession, family_id: str, subject_ids: list[str], *, kinds: tuple[str, ...] | None = None,
                  target: str | None = None, words: str = "", limit: int = 20, include_baseline: bool = False) -> list[MemoryVersion]:
    q = select(MemoryVersion).where(MemoryVersion.family_id == family_id, MemoryVersion.subject_id.in_(subject_ids))
    if kinds:
        q = q.where(MemoryVersion.kind.in_(kinds))
    if target:
        q = q.where(MemoryVersion.target == target)
    if not include_baseline:
        q = q.where(MemoryVersion.op != "baseline")
    ws = [w for w in "".join(c if c.isalnum() else " " for c in (words or "").lower()).split() if len(w) > 2][:6]
    for w in ws:
        q = q.where(func.lower(MemoryVersion.title + " " + MemoryVersion.target + " " + MemoryVersion.body).contains(w))
    return list((await session.execute(q.order_by(MemoryVersion.id.desc()).limit(max(1, min(limit, 200))))).scalars())


def lines(text: str) -> list[str]:
    return [ln for ln in (text or "").splitlines() if ln.strip()]


def diff(before: str, after: str) -> tuple[list[str], list[str]]:
    """(added, removed) lines, in order, counting repeats."""
    b, a = Counter(lines(before)), Counter(lines(after))
    plus, minus = a - b, b - a
    added, removed = [], []
    for ln in lines(after):
        if plus[ln] > 0:
            added.append(ln)
            plus[ln] -= 1
    for ln in lines(before):
        if minus[ln] > 0:
            removed.append(ln)
            minus[ln] -= 1
    return added, removed


def invert(current_text: str, before: str, added: list[str], removed: list[str]) -> str:
    """Take back one change on today's text: drop the lines it added and put back the lines it removed, each after the
    line it followed before when that line is still there (so later changes stay)."""
    out = lines(current_text)
    for ln in added:
        if ln in out:
            out.remove(ln)
    prev_of: dict[str, str | None] = {}
    last = None
    for ln in lines(before):
        prev_of.setdefault(ln, last)
        last = ln
    for ln in removed:
        if ln in out:
            continue
        anchor = prev_of.get(ln)
        if anchor is None:
            out.insert(0, ln)
        elif anchor in out:
            out.insert(out.index(anchor) + 1, ln)
        else:
            out.append(ln)
    return "\n".join(out)


SOURCE_LABEL = {"whatsapp": "on WhatsApp", "dashboard": "on the dashboard", "nightly": "Saheli's nightly review", "saheli": "Saheli",
                "snapshot": "restored from a backup", "system": "Saheli", "prescription": "a prescription", "import": "imported"}


async def view(session: AsyncSession, v: MemoryVersion, *, names: dict[str, str] | None = None) -> dict:
    """One change for people: what it was, who made it, the lines it added or removed, and whether it can be undone."""
    names = names or {}
    who = names.get(v.actor_id or "") or ("Saheli" if not v.actor_id or v.actor_id in ("saheli", "dream") else "someone in the family")
    out = {
        "id": v.id, "kind": v.kind, "subjectId": v.subject_id, "target": v.target, "version": v.version, "op": v.op,
        "title": v.title, "status": v.status, "actorId": v.actor_id, "by": who, "source": v.source,
        "where": SOURCE_LABEL.get(v.source, v.source), "reason": v.reason, "undoes": v.undoes, "at": v.at.isoformat(),
        "deleted": bool((v.value or {}).get("deleted")),
    }
    if v.kind == "note":
        prev = await previous(session, v)
        added, removed = diff(prev.body if prev else "", v.body)
        out.update(label=v.title or v.target, added=added[:LIST_LINES], removed=removed[:LIST_LINES],
                   more=max(0, len(added) + len(removed) - 2 * LIST_LINES), body=v.body)
        out["summary"] = (
            "removed the note" if out["deleted"] else
            f"added: {added[0][:120]}" if added and not removed else
            f"removed: {removed[0][:120]}" if removed and not added else
            f"rewrote {len(removed)} line(s) as {len(added)}" if added or removed else "no change in text"
        )
    elif v.kind == "fact":
        label = (v.value or {}).get("name") or (v.value or {}).get("allergen") or v.target.split(":", 1)[-1].replace("_", " ")
        word = {"write": "saved", "pending": "proposed (waiting for a caregiver)", "stop": "stopped", "approve": "approved",
                "reject": "rejected"}.get(v.op, v.op)
        out.update(label=f"{v.title}: {label}", body=v.body, value=v.value, summary=f"{word}: {v.body[:140]}")
    elif v.kind == "skill":
        out.update(label=f"How they like things: {v.title[:80]}", body=v.body, summary=f"{v.op} ({v.status}): {v.body[:140]}")
    else:
        out.update(label="Reply style (learned nightly)", body=v.body, summary=f"now: {v.body[:140]}")
    can = v.op not in ("baseline", "reject") and v.kind != "style"
    if can and v.kind in ("fact", "skill"):  # these undo only from the newest change
        last = await latest(session, v.family_id, v.subject_id, v.kind, v.target)
        can = not last or last.id == v.id
    out["canUndo"] = can
    out["canRestore"] = v.kind in ("note", "skill") or (v.kind == "fact" and v.status in ("active", "stopped"))
    return out


# ── undo and restore: planning ───────────────────────────────────────────────


class Refused(Exception):
    pass


@dataclass
class FactPlan:
    """What the care record must do to undo or restore a fact change (carried out by app.brain.tools)."""

    action: str  # rewrite | restart | stop | retract | nothing
    domain: str
    key: str
    subject_id: str
    value: dict = field(default_factory=dict)
    text: str = ""
    fact_id: uuid.UUID | None = None
    needs_ok: bool = False  # a caregiver must confirm before it takes effect
    version_id: int = 0


async def _fact_state_before(session: AsyncSession, v: MemoryVersion) -> tuple[str, dict, str]:
    """('active' | 'absent', value, text) the record had just before change v."""
    from app.care.models import CareFact

    prev = await previous(session, v, skip_ops=("pending", "reject"))
    if prev:
        if prev.status == "active":
            return "active", dict(prev.value or {}), prev.body
        if prev.status == "stopped":
            return "absent", {}, ""
    row = await session.get(CareFact, uuid.UUID(v.ref)) if v.ref else None
    if not row:
        return "absent", {}, ""
    if v.op == "stop" and not (row.value or {}).get("stopped"):
        return "active", dict(row.value or {}), row.text  # a direct stop ends the row in place
    old = await session.get(CareFact, row.supersedes) if row.supersedes else None
    if old:
        return "active", {k: x for k, x in (old.value or {}).items() if k != "stopped"}, old.text
    return "absent", {}, ""


async def _fact_now(session: AsyncSession, family_id: str, subject_id: str, key: str):
    from app.care.models import CareFact

    return (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject_id,
                                                         CareFact.key == key, CareFact.status == "active"))).scalar_one_or_none()


def _same(a: dict, b: dict) -> bool:
    clean = lambda d: {k: x for k, x in (d or {}).items() if x not in (None, "", []) and k != "stopped"}  # noqa: E731
    return clean(a) == clean(b)


async def _plan_to(session: AsyncSession, v: MemoryVersion, state: str, value: dict, text: str, *, caregiver: bool, confirmed: bool) -> FactPlan:
    now = await _fact_now(session, v.family_id, v.subject_id, v.target)
    domain = v.title or v.target.split(":", 1)[0]
    plan = FactPlan(action="nothing", domain=domain, key=v.target, subject_id=v.subject_id, version_id=v.id)
    health = domain in HEALTH_DOMAINS
    if state == "active":
        value = {k: x for k, x in value.items() if k != "stopped"}
        if now and _same(now.value, value):
            return plan
        plan.value, plan.text = value, text or (now.text if now else "")
        plan.action = "rewrite" if now else "restart"
    elif now:
        plan.action, plan.fact_id = "stop", now.id
    else:
        return plan
    if health:
        # ending or restarting a health fact always needs an explicit caregiver OK; a caregiver may change values directly
        plan.needs_ok = not caregiver or (plan.action in ("stop", "restart") and not confirmed)
    return plan


async def plan_fact_undo(session: AsyncSession, v: MemoryVersion, *, caregiver: bool, actor_id: str, confirmed: bool = False) -> FactPlan:
    from app.care.models import CareFact

    if v.op == "reject":
        raise Refused("A rejected change did nothing to undo; say the change again if it is right.")
    last = await latest(session, v.family_id, v.subject_id, "fact", v.target)
    if last and last.id != v.id:
        raise Refused(f"There is a newer change to this ({last.body[:80]}); undo that one first, or say what is right now.")
    domain = v.title or v.target.split(":", 1)[0]
    if v.op == "pending":
        row = await session.get(CareFact, uuid.UUID(v.ref)) if v.ref else None
        if not row or row.status != "pending":
            raise Refused("That proposed change is no longer waiting; nothing to undo.")
        if not caregiver and row.stated_by != actor_id:
            raise Refused("Only a caregiver, or the person who proposed it, can take back this proposed change.")
        return FactPlan(action="retract", domain=domain, key=v.target, subject_id=v.subject_id, fact_id=row.id, version_id=v.id)
    state, value, text = await _fact_state_before(session, v)
    return await _plan_to(session, v, state, value, text, caregiver=caregiver, confirmed=confirmed)


async def plan_fact_restore(session: AsyncSession, v: MemoryVersion, *, caregiver: bool, confirmed: bool = False) -> FactPlan:
    if v.status not in ("active", "stopped"):
        raise Refused("That version was only a proposal; there is nothing to put back.")
    state = "active" if v.status == "active" else "absent"
    return await _plan_to(session, v, state, dict(v.value or {}), v.body, caregiver=caregiver, confirmed=confirmed)


# ── undo and restore: notes and skills (done here) ───────────────────────────


async def _note_row(session: AsyncSession, family_id: str, subject_id: str, slug: str):
    from app.care.models import MemoryNote

    return (await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id, MemoryNote.subject_id == subject_id,
                                                           MemoryNote.slug == slug).execution_options(populate_existing=True))).scalar_one_or_none()


async def _set_note(session: AsyncSession, v: MemoryVersion, title: str, body: str, *, op: str, reason: str, undoes: int | None) -> dict:
    from app.care import memory_index, store

    note = await _note_row(session, v.family_id, v.subject_id, v.target)
    if not lines(body):
        if not note:
            return {"result": "nothing", "why": "the note is already gone"}
        await before_note(session, note)
        gone_title = note.title
        await session.delete(note)
        await session.flush()
        await memory_index.unindex(session, v.family_id, f"note:{v.target}")
        await add(session, family_id=v.family_id, subject_id=v.subject_id, kind="note", target=v.target, op=op, title=gone_title,
                  body="", value={"deleted": True}, reason=reason, undoes=undoes)
        return {"result": "done", "note": v.target, "removed": True}
    if note and note.body_md == body:
        return {"result": "nothing", "why": "the note already reads like that"}
    await store.upsert_note(session, family_id=v.family_id, subject_id=v.subject_id, slug=v.target, title=title or v.target,
                            body_md=body, op=op, reason=reason, undoes=undoes)
    await memory_index.unindex(session, v.family_id, f"note:{v.target}")  # re-indexed from the new text tonight
    return {"result": "done", "note": v.target}


async def undo_note(session: AsyncSession, v: MemoryVersion, *, reason: str = "") -> dict:
    if v.op == "baseline":
        raise Refused("That is how the note was before history started; pick a later change to undo.")
    prev = await previous(session, v)
    before = prev.body if prev else ""
    added, removed = diff(before, v.body)  # a removed note has an empty body, so undoing it brings every line back
    note = await _note_row(session, v.family_id, v.subject_id, v.target)
    new = invert(note.body_md if note else "", before, added, removed)
    title = (note.title if note else "") or (prev.title if prev else "") or v.title
    return await _set_note(session, v, title, new, op="undo", reason=reason or "undo", undoes=v.id)


async def restore_note(session: AsyncSession, v: MemoryVersion, *, reason: str = "") -> dict:
    return await _set_note(session, v, v.title, "" if (v.value or {}).get("deleted") else v.body, op="restore",
                           reason=reason or f"restored version {v.version}", undoes=None)


async def _skill_to(session: AsyncSession, v: MemoryVersion, state: dict, *, op: str, reason: str, undoes: int | None, by: str) -> dict:
    from app.care import skillbook

    s = await session.get(skillbook.Skill, int(v.target))
    if not s or s.family_id != v.family_id or s.scope != "family":
        raise Refused("That skill is gone.")
    body = state.get("body") if state.get("body") is not None else s.body
    status = state.get("status") or s.status
    if status in ("active", "proposed"):
        bad = await skillbook.family_problems(session, v.family_id, s.subject_id or "", body)
        if bad:
            raise Refused("That older wording is not allowed any more: " + "; ".join(bad))
    if s.body == body and s.status == status and s.title == (state.get("title") or s.title):
        return {"result": "nothing", "why": "it already reads like that"}
    prior = skill_state(s)
    s.body, s.status, s.title = body, status, state.get("title") or s.title
    s.updated_at, s.updated_by, s.version = clock.now(), by, s.version + 1
    await session.flush()
    await record_skill(session, s, op, prior=prior, reason=reason, undoes=undoes)
    if s.status == "active":
        await skillbook._cap_active(session, v.family_id, s.subject_id or "")
    return {"result": "done", "skill": s.id, "status": s.status}


async def undo_skill(session: AsyncSession, v: MemoryVersion, *, by: str, reason: str = "") -> dict:
    if v.op == "baseline":
        raise Refused("That is how it was before history started; pick a later change to undo.")
    last = await latest(session, v.family_id, v.subject_id, "skill", v.target)
    if last and last.id != v.id:
        raise Refused("It changed again after that; undo the newest change first.")
    prev = await previous(session, v)
    if prev:
        state = {"title": prev.title, "body": prev.body, "status": prev.status}
    else:  # undoing its creation: a suggestion Saheli made stays blocked, anything else is set aside
        state = {"status": "blocked" if (v.value or {}).get("source") == "dream" else "archived"}
    return await _skill_to(session, v, state, op="undo", reason=reason or "undo", undoes=v.id, by=by)


async def restore_skill(session: AsyncSession, v: MemoryVersion, *, by: str, reason: str = "") -> dict:
    return await _skill_to(session, v, {"title": v.title, "body": v.body, "status": v.status}, op="restore",
                           reason=reason or f"restored version {v.version}", undoes=None, by=by)


# ── upkeep ───────────────────────────────────────────────────────────────────


async def prune(session: AsyncSession, *, keep_for: timedelta = KEEP_FOR) -> int:
    """Drop versions older than a year, except the newest of each item."""
    newest = select(func.max(MemoryVersion.id)).group_by(MemoryVersion.family_id, MemoryVersion.subject_id, MemoryVersion.kind,
                                                         MemoryVersion.target)
    res = await session.execute(delete(MemoryVersion).where(MemoryVersion.at < clock.now() - keep_for, MemoryVersion.id.not_in(newest)))
    return res.rowcount or 0
