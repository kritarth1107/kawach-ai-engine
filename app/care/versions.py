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


def invert(current_text: str, before: str, after: str) -> str:
    """Take back one change (before → after) on today's text: drop exactly the lines it added (the same repeat of a
    repeated line) and put back the lines it removed, each after the line it followed before when that line is still
    there. Lines changed since, and blank lines, stay."""
    raw = (current_text or "").splitlines()
    cb, ca = Counter(lines(before)), Counter(lines(after))
    for ln, extra in (ca - cb).items():
        for k in range(ca[ln], ca[ln] - extra, -1):  # the added copies are the last ones in `after`
            at = [i for i, x in enumerate(raw) if x == ln]
            if at:
                del raw[at[k - 1] if len(at) >= k else at[-1]]
    have = Counter(x for x in raw if x.strip())
    seen: Counter = Counter()
    prev_line: str | None = None
    for ln in lines(before):
        seen[ln] += 1
        # the copies this change removed are the last ones in `before`; bring one back unless it is already there again
        if seen[ln] > ca[ln] and have[ln] < cb[ln]:
            if prev_line is None:
                raw.insert(0, ln)
            elif prev_line in raw:
                raw.insert(len(raw) - raw[::-1].index(prev_line), ln)
            else:
                raw.append(ln)
            have[ln] += 1
        prev_line = ln
    return "\n".join(raw)


async def forgotten_lines(session: AsyncSession, family_id: str, subject_ids: list[str]) -> set[str]:
    """Note lines someone asked Saheli to forget (and not restored): never shown again in history."""
    from app.care.models import CareEvent

    rows = (await session.execute(select(CareEvent.payload).where(CareEvent.family_id == family_id, CareEvent.kind == "memory_forgotten",
                                                                  CareEvent.subject_id.in_(subject_ids)))).scalars()
    out: set[str] = set()
    for p in rows:
        p = p or {}
        done = set(p.get("restored_subjects") or ([] if not p.get("restored") else [i.get("subject") for i in p.get("note_lines") or []]))
        out |= {i["line"] for i in p.get("note_lines") or [] if i.get("subject") not in done and i.get("line")}
    return out


def _hide(text: str, hidden: set[str]) -> str:
    return "\n".join("(forgotten)" if ln in hidden else ln for ln in (text or "").splitlines()) if hidden else text


SOURCE_LABEL = {"whatsapp": "on WhatsApp", "dashboard": "on the dashboard", "nightly": "Saheli's nightly review", "saheli": "Saheli",
                "snapshot": "restored from a backup", "system": "Saheli", "prescription": "a prescription", "import": "imported"}


async def view(session: AsyncSession, v: MemoryVersion, *, names: dict[str, str] | None = None, hidden: set[str] | None = None) -> dict:
    """One change for people: what it was, who made it, the lines it added or removed, and whether it can be undone.
    Lines someone asked to forget (`hidden`) are shown as '(forgotten)', never as text."""
    names = names or {}
    hidden = hidden or set()
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
        if v.op == "forget":
            hidden = hidden | set(removed)  # what was forgotten is never repeated back
        added = ["(forgotten)" if ln in hidden else ln for ln in added]
        removed = ["(forgotten)" if ln in hidden else ln for ln in removed]
        out.update(label=v.title or v.target, added=added[:LIST_LINES], removed=removed[:LIST_LINES],
                   more=max(0, len(added) + len(removed) - 2 * LIST_LINES), body=_hide(v.body, hidden))
        out["summary"] = (
            "removed the note" if out["deleted"] else
            f"forgot {len(removed)} line{'s' if len(removed) != 1 else ''}" if v.op == "forget" else
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
    can = v.op != "baseline" and v.kind != "style"
    if can and v.kind in ("fact", "skill"):  # these undo only from the newest change
        last = await (last_effective(session, v) if v.kind == "fact" else latest(session, v.family_id, v.subject_id, v.kind, v.target))
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

    action: str  # rewrite | restart | stop | retract | repropose | nothing
    domain: str
    key: str
    subject_id: str
    value: dict = field(default_factory=dict)
    text: str = ""
    fact_id: uuid.UUID | None = None
    needs_ok: bool = False  # a caregiver must confirm before it takes effect
    version_id: int = 0
    stop: bool = False  # repropose: the proposal was to end the fact


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
    """The active fact for this key, matching an older dose-in-key medicine record the same way the care record does."""
    from app.care import store

    return await store.active_fact(session, family_id, subject_id, key)


async def last_effective(session: AsyncSession, v: MemoryVersion) -> MemoryVersion | None:
    """The newest change that still shapes this fact: a proposal that was turned down (and the turning down) don't count,
    so one rejected proposal never blocks undoing the change before it."""
    rows = (await session.execute(select(MemoryVersion).where(
        MemoryVersion.family_id == v.family_id, MemoryVersion.subject_id == v.subject_id, MemoryVersion.kind == "fact",
        MemoryVersion.target == v.target).order_by(MemoryVersion.id.desc()).limit(50))).scalars()
    rejected: set[str | None] = set()
    for r in rows:
        if r.op == "reject":
            if r.id == v.id:
                return r
            rejected.add(r.ref)
            continue
        if r.op == "pending" and r.ref in rejected and r.id != v.id:
            continue
        return r
    return None


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

    last = await last_effective(session, v)
    if last and last.id != v.id:
        raise Refused(f"There is a newer change to this ({last.body[:80]}); undo that one first, or say what is right now.")
    domain = v.title or v.target.split(":", 1)[0]
    if v.op == "reject":
        # undoing a "no" (or a take-back) proposes the change again; it still waits for a caregiver
        row = await session.get(CareFact, uuid.UUID(v.ref)) if v.ref else None
        if not row:
            raise Refused("That proposal is gone; say the change again if it is right.")
        return FactPlan(action="repropose", domain=domain, key=v.target, subject_id=v.subject_id, value=dict(row.value or {}), text=row.text,
                        needs_ok=True, version_id=v.id, stop=bool((row.value or {}).get("stopped")))
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
    note = await _note_row(session, v.family_id, v.subject_id, v.target)
    new = invert(note.body_md if note else "", before, v.body)  # a removed note has an empty body, so every line comes back
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


# ── what an undo or restore would do (the dashboard says it before the caregiver confirms) ──


def _fact_name(v: MemoryVersion, value: dict | None = None) -> str:
    val = value or v.value or {}
    return str(val.get("name") or val.get("allergen") or v.target.split(":", 1)[-1].replace("_", " "))


async def preview(session: AsyncSession, v: MemoryVersion, mode: str, *, caregiver: bool = True, confirmed: bool = True) -> dict:
    """{effect, text, button} for undo/restore of v, without changing anything. effect: change | stop | restart | retract |
    remove | nothing | refused."""
    try:
        if v.kind == "style":
            raise Refused("Reply style is relearned every night; add a 'How they like things' skill instead.")
        if v.kind == "fact":
            plan = await (plan_fact_undo(session, v, caregiver=caregiver, actor_id="", confirmed=confirmed) if mode == "undo"
                          else plan_fact_restore(session, v, caregiver=caregiver, confirmed=confirmed))
            name = _fact_name(v, plan.value)
            med = plan.domain == "medicine"
            if plan.action == "nothing":
                return {"effect": "nothing", "text": "Nothing would change: the care record already says that.", "button": ""}
            if plan.action == "retract":
                return {"effect": "retract", "text": f"This takes back the proposed change to {name}. The record stays as it is now.",
                        "button": "Yes, take it back"}
            if plan.action == "repropose":
                return {"effect": "pending", "text": f"This proposes the change to {name} again; it waits for a caregiver to confirm.",
                        "button": "Yes, propose it again"}
            from app.care.domains import SOURCE_RANK

            now = await _fact_now(session, v.family_id, v.subject_id, v.target)
            if now and plan.action in ("rewrite", "stop") and SOURCE_RANK.get(now.source_kind, 0) > SOURCE_RANK["caregiver_said"]:
                return {"effect": "pending", "text": f"{name} came from a {now.source_kind}, so this waits for a caregiver to confirm it "
                                                     "before anything changes (reminders stay as they are).", "button": "Yes, ask for confirmation"}
            if plan.action == "stop":
                word = {"medicine": f"This stops {name} now and switches off its reminders.",
                        "allergy": f"This removes the allergy to {name} from the care record now.",
                        "condition": f"This removes {name} from the conditions now."}.get(plan.domain, f"This removes '{v.body[:80]}' from the care record.")
                return {"effect": "stop", "text": word, "button": "Yes, stop it" if med else "Yes, remove it"}
            if plan.action == "restart":
                return {"effect": "restart", "text": f"This puts {name} back on the care record now" + (" and turns its reminders back on." if med else "."),
                        "button": "Yes, restart it" if med else "Yes, put it back"}
            return {"effect": "change", "text": f"The care record goes back to: {plan.text or name}." + (" Its reminders change to match." if med else ""),
                    "button": "Yes, change it"}
        if v.kind == "note":
            note = await _note_row(session, v.family_id, v.subject_id, v.target)
            current_text = note.body_md if note else ""
            if mode == "undo" and v.op == "forget" and (v.value or {}).get("forget_event"):
                return {"effect": "change", "text": "This brings back what was asked to be forgotten, in every note and record it was hidden from.",
                        "button": "Yes, bring it back"}
            if mode == "undo":
                if v.op == "baseline":
                    raise Refused("That is how the note was before history started; pick a later change to undo.")
                prev = await previous(session, v)
                new = invert(current_text, prev.body if prev else "", v.body)
            else:
                new = "" if (v.value or {}).get("deleted") else v.body
            gone, back = diff(current_text, new)[1], diff(current_text, new)[0]
            if not gone and not back:
                return {"effect": "nothing", "text": "Nothing would change: the note already reads like that.", "button": ""}
            if not lines(new):
                return {"effect": "remove", "text": f"This removes the note '{v.title}'. It stays in the history.", "button": "Yes, remove it"}
            bits = ([f"takes out {len(gone)} line{'s' if len(gone) != 1 else ''}"] if gone else []) + (
                [f"brings back {len(back)} line{'s' if len(back) != 1 else ''}"] if back else [])
            return {"effect": "change", "text": f"This {' and '.join(bits)} in '{v.title}'.", "button": "Yes, undo" if mode == "undo" else "Yes, put back"}
        from app.care import skillbook

        s = await session.get(skillbook.Skill, int(v.target))
        if mode == "undo":
            prev = await previous(session, v)
            last = await latest(session, v.family_id, v.subject_id, "skill", v.target)
            if last and last.id != v.id:
                raise Refused("It changed again after that; undo the newest change first.")
            if not prev:
                return {"effect": "remove", "text": "Saheli stops using this.", "button": "Yes, stop using it"}
            body, status = prev.body, prev.status
        else:
            body, status = v.body, v.status
        if s and s.body == body and s.status == status:
            return {"effect": "nothing", "text": "Nothing would change.", "button": ""}
        if status in ("active", "proposed"):
            bad = await skillbook.family_problems(session, v.family_id, v.subject_id, body)
            if bad:
                raise Refused("That older wording is not allowed any more: " + "; ".join(bad))
        used = {"active": "and Saheli uses it", "proposed": "as a suggestion"}.get(status or "", "but Saheli does not use it")
        return {"effect": "change", "text": f"It goes back to “{body[:120]}” {used}.", "button": "Yes, undo" if mode == "undo" else "Yes, put back"}
    except Refused as exc:
        return {"effect": "refused", "text": str(exc), "button": ""}


# ── upkeep ───────────────────────────────────────────────────────────────────


async def prune(session: AsyncSession, *, keep_for: timedelta = KEEP_FOR) -> int:
    """Drop versions older than a year, except the newest old one of each item: it is what the next change is compared
    against, so an undo never mistakes a year-old note for a new one."""
    cutoff = clock.now() - keep_for
    keep = (select(func.max(MemoryVersion.id)).where(MemoryVersion.at < cutoff)
            .group_by(MemoryVersion.family_id, MemoryVersion.subject_id, MemoryVersion.kind, MemoryVersion.target))
    res = await session.execute(delete(MemoryVersion).where(MemoryVersion.at < cutoff, MemoryVersion.id.not_in(keep)))
    return res.rowcount or 0
