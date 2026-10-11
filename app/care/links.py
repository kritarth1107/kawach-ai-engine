"""Links between things in a person's care record (D-13): what a medicine is for, who prescribed it, which health
record it came from, which doctor an appointment is with, how two people are related.

A small table in the engine's own Postgres (care_links), not a graph database. Each link joins two refs:
- a fact key: medicine:metformin, condition:diabetes, doctor:dr_sharma, appointment:…, contact:ankit;
- a household member: person:<id>;
- a health record: record:<document id>.
Because links name keys, not fact rows, a dose change (a new row for the same key) keeps every link.

They follow the care record's own rules (app.care.store):
- a link is never deleted: it ends (valid_to, the reason in note) and every change is a memory version, so it can be
  undone or restored like a fact;
- a health link (from a medicine, allergy, condition or target) from a weaker source than the medicine itself, or than
  the link it would replace, waits as pending for a caregiver; one from the nightly re-read always does;
- prescribed_by, from_record and with_doctor have one value: a new one replaces the old.

They are read by the care record digest (Saheli's CARE RECORD block), the doctor brief, the dashboard's medicine card
("for diabetes, Dr Sharma"), the nightly record check (a medicine for a condition not on the record, a prescribing
doctor not in the contacts) and the stop of a condition (Saheli asks about the medicines linked to it).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.domains import DOMAINS, HEALTH_DOMAINS, SOURCE_RANK, fact_key
from app.care.models import CareLink, OpenLoop
from app.core import clock, trace

logger = logging.getLogger(__name__)

PEOPLE = {"person", "contact", "family", "doctor", "home"}
# relation: (domains it may start from, domains it may point to)
RELATIONS: dict[str, tuple[set[str], set[str]]] = {
    "for": ({"medicine"}, {"condition"}),
    "prescribed_by": ({"medicine"}, {"doctor"}),
    "from_record": (set(HEALTH_DOMAINS), {"record"}),
    "with_doctor": ({"appointment"}, {"doctor"}),
    "related_to": (PEOPLE, PEOPLE),
}
SINGLE = {"prescribed_by", "from_record", "with_doctor"}  # one value: a new one replaces the old
BRAIN_RELATIONS = ("for", "prescribed_by", "with_doctor", "related_to")  # from_record is set by the health-record review
STRONG = ("caregiver_said", "dashboard", "prescription", "lab")  # a link from these counts as confirmed by whoever said it


@dataclass
class LinkWrite:
    result: str  # created | unchanged | replaced | pending
    link: CareLink
    previous: CareLink | None = None


class BadLink(ValueError):
    """A relation between things it cannot join (a medicine 'prescribed by' a condition)."""


# ── refs and how a link reads ────────────────────────────────────────────────


def ref(domain: str, name: str) -> str:
    """The ref of one end: a fact key, a household member (person:<id>) or a health record (record:<id>)."""
    if domain == "person":
        return f"person:{str(name).strip()[:120]}"
    if domain == "record":
        return f"record:{str(name).strip()[:120]}"
    return fact_key(domain, name)


def domain_of(r: str) -> str:
    return (r or "").split(":", 1)[0]


def name_of(r: str) -> str:
    return (r or "").split(":", 1)[-1].replace("_", " ").strip()


def target(link_or_from, relation: str | None = None, to_ref: str | None = None) -> str:
    """The memory-history target of a link: "from|relation|to"."""
    if relation is None:
        link_or_from, relation, to_ref = link_or_from.from_ref, link_or_from.relation, link_or_from.to_ref
    return f"{link_or_from}|{relation}|{to_ref}"[:200]


def is_health(link: CareLink) -> bool:
    return domain_of(link.from_ref) in HEALTH_DOMAINS


def label_of(link: CareLink) -> str:
    return (link.label or "").strip() or name_of(link.to_ref)


def phrase(link: CareLink) -> str:
    """How the link reads after the thing it starts from: 'for diabetes', 'prescribed by Dr Sharma'."""
    lab = label_of(link)
    if link.relation == "for":
        return f"for {lab}"
    if link.relation == "prescribed_by":
        return f"prescribed by {lab}"
    if link.relation == "from_record":
        return f"from the {lab}"
    if link.relation == "with_doctor":
        return f"with {lab}"
    rel = (link.detail or {}).get("as")
    return f"{lab} ({rel})" if rel else f"related to {lab}"


def short(link: CareLink) -> str:
    """The medicine card's words: 'for diabetes', 'Dr Sharma', 'from the 3 Sep prescription'."""
    return label_of(link) if link.relation in ("prescribed_by", "with_doctor") else phrase(link)


def readable(link: CareLink, from_name: str | None = None) -> str:
    start = (from_name or name_of(link.from_ref)).strip()
    return f"{start[:1].upper()}{start[1:]} {phrase(link)}"


def by_from(links: list[CareLink]) -> dict[str, list[CareLink]]:
    out: dict[str, list[CareLink]] = {}
    order = {"for": 0, "prescribed_by": 1, "with_doctor": 2, "from_record": 3, "related_to": 4}
    for ln in sorted(links, key=lambda x: (order.get(x.relation, 9), x.recorded_at)):
        out.setdefault(ln.from_ref, []).append(ln)
    return out


def check(relation: str, from_ref: str, to_ref: str) -> None:
    if relation not in RELATIONS:
        raise BadLink(f"unknown link {relation}")
    starts, ends = RELATIONS[relation]
    if domain_of(from_ref) not in starts or domain_of(to_ref) not in ends:
        raise BadLink(f"'{relation}' joins {'/'.join(sorted(starts))} to {'/'.join(sorted(ends))}, not {domain_of(from_ref)} to {domain_of(to_ref)}")
    if from_ref == to_ref:
        raise BadLink("a link needs two different things")
    for r in (from_ref, to_ref):
        d = domain_of(r)
        if d not in DOMAINS and d not in ("person", "record") or not name_of(r):
            raise BadLink(f"not something on the record: {r}")


# ── reading ──────────────────────────────────────────────────────────────────


async def links_for(session: AsyncSession, family_id: str, subject_id: str, *, statuses: tuple[str, ...] = ("active", "pending"),
                    from_ref: str | None = None) -> list[CareLink]:
    q = select(CareLink).where(CareLink.family_id == family_id, CareLink.subject_id == subject_id, CareLink.status.in_(statuses))
    if from_ref:
        q = q.where(CareLink.from_ref == from_ref)
    return list((await session.execute(q.order_by(CareLink.recorded_at))).scalars())


async def _active(session: AsyncSession, family_id: str, subject_id: str, from_ref: str, relation: str, to_ref: str | None = None) -> list[CareLink]:
    q = select(CareLink).where(CareLink.family_id == family_id, CareLink.subject_id == subject_id, CareLink.from_ref == from_ref,
                               CareLink.relation == relation, CareLink.status == "active")
    if to_ref is not None:
        q = q.where(CareLink.to_ref == to_ref)
    return list((await session.execute(q.with_for_update())).scalars())


async def _waiting(session: AsyncSession, family_id: str, subject_id: str, from_ref: str, relation: str, to_ref: str | None) -> list[CareLink]:
    """Proposals still waiting to add this link (to_ref None: to any value, for a one-value relation)."""
    q = select(CareLink).where(CareLink.family_id == family_id, CareLink.subject_id == subject_id, CareLink.from_ref == from_ref,
                               CareLink.relation == relation, CareLink.status == "pending")
    if to_ref is not None:
        q = q.where(CareLink.to_ref == to_ref)
    return [r for r in (await session.execute(q.with_for_update())).scalars() if not (r.detail or {}).get("end")]


async def _fact_source(session: AsyncSession, family_id: str, subject_id: str, from_ref: str) -> str | None:
    if domain_of(from_ref) not in DOMAINS:
        return None
    from app.care import store

    f = await store.active_fact(session, family_id, subject_id, from_ref)
    return f.source_kind if f else None


def needs_ok(from_ref: str, source_kind: str, *, fact_source: str | None, old_source: str | None, force: bool = False) -> bool:
    """A health link waits for a caregiver when it comes from the nightly re-read, or from a weaker source than the
    medicine it is about or the link it would replace (the same rule as a fact: app.care.domains.needs_confirmation)."""
    if force:
        return True
    if domain_of(from_ref) not in HEALTH_DOMAINS:
        return False
    if source_kind == "inferred":
        return True
    rank = SOURCE_RANK.get(source_kind, 0)
    return any(s is not None and rank < SOURCE_RANK.get(s, 0) for s in (fact_source, old_source))


# ── writing ──────────────────────────────────────────────────────────────────


async def _version(session: AsyncSession, link: CareLink, op: str, *, reason: str | None = None, extra: dict | None = None):
    from app.care import versions

    a = versions.current()
    return await versions.add(
        session, family_id=link.family_id, subject_id=link.subject_id, kind="link", target=target(link), op=op, title=link.relation,
        body=readable(link), value={"from": link.from_ref, "relation": link.relation, "to": link.to_ref, "label": link.label,
                                    "detail": dict(link.detail or {}), "source_kind": link.source_kind, **(extra or {})},
        status=link.status, ref=str(link.id), actor_id=a.actor_id or link.confirmed_by or link.stated_by,
        reason=reason if reason is not None else (a.reason or link.note or ""),
    )


async def _ask_to_confirm(session: AsyncSession, link: CareLink) -> None:
    from app.care import store

    await store.open_loop(
        session, family_id=link.family_id, subject_id=link.subject_id, kind="confirm_link",
        title=(f"Confirm: {readable(link)}" if not (link.detail or {}).get("end") else f"Confirm removing: {readable(link)}"),
        detail={"link_id": str(link.id), "from": link.from_ref, "relation": link.relation, "to": link.to_ref, "label": link.label,
                "end": bool((link.detail or {}).get("end"))},
        dedupe_key=f"confirmlink:{link.subject_id}:{target(link)}"[:160],
    )


async def _close_question(session: AsyncSession, link: CareLink, note: str) -> None:
    await session.execute(
        update(OpenLoop).where(OpenLoop.family_id == link.family_id, OpenLoop.kind == "confirm_link", OpenLoop.status == "open")
        .where(OpenLoop.detail["link_id"].astext == str(link.id))
        .values(status="done", updated_at=clock.now(), closed_note=note)
    )


async def _settle_waiting(session: AsyncSession, rows: list[CareLink], why: str) -> list[str]:
    """Older proposals for the same link (any value of a one-value relation) end when a newer proposal, a stronger
    source or an approval settles it: the caregiver is asked once, and a stale proposal can never be approved later."""
    now = clock.now()
    for r in rows:
        r.status, r.valid_to, r.note = "ended", now, why
        await _close_question(session, r, "superseded")
    if rows:
        await session.flush()
    return [str(r.id) for r in rows]


def _row(*, family_id, subject_id, from_ref, relation, to_ref, label, detail, status, source_kind, stated_by, confirmed_by,
         source_ref, note=None) -> CareLink:
    now = clock.now()
    return CareLink(
        id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, from_ref=from_ref[:200], relation=relation, to_ref=to_ref[:200],
        label=(label or "")[:200], detail=detail or {}, status=status, source_kind=source_kind, stated_by=stated_by,
        confirmed_by=confirmed_by, source_ref=(source_ref or "")[:160] or None, valid_from=now, recorded_at=now, note=note,
        trace_id=trace.current(),
    )


async def write_link(
    session: AsyncSession, *, family_id: str, subject_id: str, from_ref: str, relation: str, to_ref: str, label: str = "",
    source_kind: str, stated_by: str | None = None, source_ref: str | None = None, detail: dict | None = None,
    confirmed_by: str | None = None, force_confirm: bool = False, with_fact: uuid.UUID | str | None = None,
) -> LinkWrite:
    """Save a link. The same link again changes nothing; a new label, or a new value of a one-value relation, replaces
    the old one; a weaker source's health link waits as pending (with_fact: it waits with that pending fact and is
    decided with it). The same proposal again is the one already waiting; a different one, or a link that is saved
    straight away, settles the older proposals for it."""
    check(relation, from_ref, to_ref)
    detail = {k: v for k, v in (detail or {}).items() if v not in (None, "", [])}
    same = await _active(session, family_id, subject_id, from_ref, relation, to_ref)
    if same:
        cur = same[0]
        if (label or cur.label) == cur.label and {**(cur.detail or {}), **detail} == (cur.detail or {}):
            if source_kind in STRONG and not cur.confirmed_by and (confirmed_by or stated_by):
                cur.confirmed_by = confirmed_by or stated_by
            return LinkWrite("unchanged", cur)
    olds = same or (await _active(session, family_id, subject_id, from_ref, relation) if relation in SINGLE else [])
    pending = needs_ok(from_ref, source_kind, fact_source=await _fact_source(session, family_id, subject_id, from_ref),
                       old_source=olds[0].source_kind if olds else None, force=force_confirm or bool(with_fact))
    row = _row(family_id=family_id, subject_id=subject_id, from_ref=from_ref, relation=relation, to_ref=to_ref,
               label=label or (olds[0].label if same else ""), detail={**((same[0].detail or {}) if same else {}), **detail,
                                                                       **({"with_fact": str(with_fact)} if with_fact else {})},
               status="pending" if pending else "active", source_kind=source_kind, stated_by=stated_by,
               confirmed_by=None if pending else (confirmed_by or (stated_by if source_kind in STRONG else None)), source_ref=source_ref)
    waiting = await _waiting(session, family_id, subject_id, from_ref, relation, None if relation in SINGLE else to_ref)
    if pending:
        for w in waiting:
            if (w.to_ref, w.label or "", dict(w.detail or {})) == (row.to_ref, row.label or "", dict(row.detail or {})):
                return LinkWrite("pending", w, olds[0] if olds else None)  # already asked: one question, never two
        settled = await _settle_waiting(session, waiting, "replaced by a newer proposal")
        session.add(row)
        await session.flush()
        await _ask_to_confirm(session, row)
        await _version(session, row, "pending", extra={"supersedes": settled} if settled else None)
        return LinkWrite("pending", row, olds[0] if olds else None)
    settled = await _settle_waiting(session, waiting, f"settled: {readable(row)}")
    await _take_place_of(session, olds, row)
    session.add(row)
    await session.flush()
    extra = {**({"replaced": [str(o.id) for o in olds]} if olds else {}), **({"supersedes": settled} if settled else {})}
    await _version(session, row, "write", extra=extra or None)
    return LinkWrite("replaced" if olds else "created", row, olds[0] if olds else None)


async def _take_place_of(session: AsyncSession, olds: list[CareLink], new: CareLink) -> None:
    now = clock.now()
    for o in olds:
        o.status, o.valid_to, o.note = "ended", now, f"replaced by {label_of(new)}" if o.to_ref != new.to_ref else "replaced"
        await session.flush()
        if o.to_ref != new.to_ref:  # another target: its own history says it ended
            await _version(session, o, "end", reason=f"replaced by {readable(new)}")


async def end_link(session: AsyncSession, link: CareLink, *, reason: str, source_kind: str, stated_by: str | None = None,
                   force_confirm: bool = False) -> LinkWrite:
    """End a link (taken back, forgotten, undone). A weaker source only proposes ending a health link."""
    if link.status != "active":
        return LinkWrite("unchanged", link)
    if force_confirm or (is_health(link) and needs_ok(link.from_ref, source_kind, fact_source=None, old_source=link.source_kind)):
        row = _row(family_id=link.family_id, subject_id=link.subject_id, from_ref=link.from_ref, relation=link.relation, to_ref=link.to_ref,
                   label=link.label, detail={**(link.detail or {}), "end": True}, status="pending", source_kind=source_kind,
                   stated_by=stated_by, confirmed_by=None, source_ref=None, note=reason)
        session.add(row)
        await session.flush()
        await _ask_to_confirm(session, row)
        await _version(session, row, "pending", reason=reason)
        return LinkWrite("pending", row, link)
    link.status, link.valid_to, link.note = "ended", clock.now(), reason
    await session.flush()
    await _version(session, link, "end", reason=reason)
    return LinkWrite("ended", link)


async def resolve_pending_link(session: AsyncSession, *, link_id: uuid.UUID | str, approve: bool, by: str) -> CareLink | None:
    row = await session.get(CareLink, uuid.UUID(str(link_id)), with_for_update=True)
    if not row or row.status != "pending":
        return None
    now = clock.now()
    ending = bool((row.detail or {}).get("end"))
    extra: dict = {}
    if approve and ending:
        for cur in await _active(session, row.family_id, row.subject_id, row.from_ref, row.relation, row.to_ref):
            cur.status, cur.valid_to, cur.note = "ended", now, row.note or "ended"
            await session.flush()
            await _version(session, cur, "end", reason=f"approved by a caregiver: {row.note or 'ended'}")
        row.status, row.valid_to, row.confirmed_by = "ended", now, by
    elif approve:
        olds = await _active(session, row.family_id, row.subject_id, row.from_ref, row.relation, row.to_ref)
        if not olds and row.relation in SINGLE:
            olds = await _active(session, row.family_id, row.subject_id, row.from_ref, row.relation)
        others = [w for w in await _waiting(session, row.family_id, row.subject_id, row.from_ref, row.relation,
                                            None if row.relation in SINGLE else row.to_ref) if w.id != row.id]
        if settled := await _settle_waiting(session, others, f"settled: {readable(row)} was approved"):
            extra["supersedes"] = settled
        await _take_place_of(session, olds, row)
        if olds:
            extra["replaced"] = [str(o.id) for o in olds]
        row.status, row.confirmed_by = "active", by
        row.detail = {k: v for k, v in (row.detail or {}).items() if k != "with_fact"}
    else:
        row.status, row.valid_to, row.confirmed_by = "rejected", now, by
    await _close_question(session, row, "approved" if approve else "rejected")
    await session.flush()
    await _version(session, row, "approve" if approve else "reject", extra=extra or None)
    return row


async def resolve_with_fact(session: AsyncSession, fact, *, approve: bool, by: str) -> int:
    """A pending fact was decided: the links proposed with it go the same way."""
    rows = (await session.execute(select(CareLink).where(
        CareLink.family_id == fact.family_id, CareLink.subject_id == fact.subject_id, CareLink.status == "pending",
        CareLink.detail["with_fact"].astext == str(fact.id)))).scalars().all()
    for r in rows:
        await resolve_pending_link(session, link_id=r.id, approve=approve, by=by)
    return len(rows)


async def save_many(session: AsyncSession, *, family_id: str, subject_id: str, from_ref: str, items: list[dict], source_kind: str,
                    stated_by: str | None, source_ref: str | None, force_confirm: bool = False, with_fact=None,
                    allowed: tuple[str, ...] = BRAIN_RELATIONS, confirmed_by: str | None = None) -> dict:
    """Links given with a fact ({relation, to_domain, to_name, label?, as?}, from the brain, the dashboard or the
    re-read): each saved under the rules above; ones that cannot join are reported, never guessed."""
    out: dict = {"links": [], "skipped": []}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        relation, to_domain, to_name = str(it.get("relation") or ""), str(it.get("to_domain") or ""), str(it.get("to_name") or "").strip()
        if relation not in allowed or not to_name:
            out["skipped"].append(f"{relation or '?'} → {to_name or '?'}: not a link Saheli can save")
            continue
        try:
            to_ref = ref(to_domain, to_name)
            w = await write_link(session, family_id=family_id, subject_id=subject_id, from_ref=from_ref, relation=relation, to_ref=to_ref,
                                 label=str(it.get("label") or to_name)[:200], source_kind=source_kind, stated_by=stated_by,
                                 source_ref=source_ref, detail={"as": str(it.get("as") or "")[:40], "date": it.get("date")},
                                 force_confirm=force_confirm, with_fact=with_fact, confirmed_by=confirmed_by)
        except (BadLink, ValueError) as exc:
            out["skipped"].append(f"{relation} → {to_name}: {exc}")
            continue
        out["links"].append({"link": phrase(w.link), "result": w.result})
    return out


# ── after a condition stops: ask about the medicines that were for it ────────


async def medicines_for(session: AsyncSession, family_id: str, subject_id: str, condition_key: str) -> list[tuple[str, CareLink]]:
    """[(medicine name, link)] for active medicines linked 'for' this condition."""
    from app.care import store

    rows = (await session.execute(select(CareLink).where(
        CareLink.family_id == family_id, CareLink.subject_id == subject_id, CareLink.relation == "for", CareLink.to_ref == condition_key,
        CareLink.status == "active"))).scalars().all()
    out = []
    for ln in rows:
        med = await store.active_fact(session, family_id, subject_id, ln.from_ref)
        if med:
            out.append((str((med.value or {}).get("name") or name_of(med.key)), ln))
    return out


async def ask_about_linked_medicines(session: AsyncSession, family_id: str, subject_id: str, condition, meds: list[tuple[str, CareLink]],
                                     owner_id: str | None = None) -> OpenLoop | None:
    """One question to a caregiver: the condition is over; should the medicines that were for it stop too? Nothing is
    stopped here (a medicine stops only through the record's own rules)."""
    if not meds:
        return None
    from app.care import store

    cname = str((condition.value or {}).get("name") or name_of(condition.key))
    names = ", ".join(n for n, _ in meds)
    return await store.open_loop(
        session, family_id=family_id, subject_id=subject_id, kind="memory_check",
        title=f"{cname} was marked as over. {names} {'is' if len(meds) == 1 else 'are'} noted as for it: should "
              f"{'it' if len(meds) == 1 else 'they'} stop too, or carry on?",
        detail={"problem": f"condition {condition.key} stopped; linked medicines: {', '.join(ln.from_ref for _, ln in meds)}",
                "kind": "condition_stopped", "max_wakes": 1, "condition": condition.key, "medicines": [ln.from_ref for _, ln in meds]},
        owner_id=owner_id, wake_at=clock.ist().replace(hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1),
        alert_rule="dashboard", dedupe_key=f"memcheck:{subject_id}:condstop:{condition.key}"[:160],
    )


# ── undo and restore (memory history; called by app.brain.tools.undo_change) ─


class Refused(Exception):
    pass


async def _latest(session: AsyncSession, v):
    from app.care import versions

    return await versions.latest(session, v.family_id, v.subject_id, "link", v.target)


async def _link_of(session: AsyncSession, v) -> CareLink | None:
    try:
        return await session.get(CareLink, uuid.UUID(v.ref)) if v.ref else None
    except ValueError:
        return None


def _parts(v) -> dict:
    val = v.value or {}
    return {"from_ref": val.get("from"), "relation": val.get("relation") or v.title, "to_ref": val.get("to"), "label": val.get("label") or "",
            "detail": {k: x for k, x in (val.get("detail") or {}).items() if k not in ("with_fact", "end")}}


def _link_parts(link: CareLink) -> dict:
    return {"from_ref": link.from_ref, "relation": link.relation, "to_ref": link.to_ref, "label": link.label or "",
            "detail": {k: x for k, x in (link.detail or {}).items() if k not in ("with_fact", "end")}}


async def _replaced_by(session: AsyncSession, v) -> CareLink | None:
    """The earlier link this change replaced (a new doctor for 'prescribed by', a new label), if it is still ended."""
    for x in (v.value or {}).get("replaced") or []:
        try:
            old = await session.get(CareLink, uuid.UUID(str(x)))
        except ValueError:
            continue
        if old is not None and old.status == "ended":
            return old
    return None


async def plan(session: AsyncSession, v, mode: str) -> tuple[str, dict]:
    """What undo/restore of link version v would do: (effect, parts) with effect activate | end | revert (put back the
    link it replaced) | retract | repropose | nothing."""
    p = _parts(v)
    if not p["from_ref"] or not p["to_ref"]:
        raise Refused("That change no longer says what it linked.")
    now = await _active(session, v.family_id, v.subject_id, p["from_ref"], p["relation"], p["to_ref"])
    if mode == "restore":
        if v.status == "active":
            return ("nothing" if now and (now[0].label or "") == (p["label"] or now[0].label) else "activate"), p
        if v.status == "ended":
            return ("end" if now else "nothing"), p
        raise Refused("That version was only a proposal; there is nothing to put back.")
    last = await _latest(session, v)
    if last and last.id != v.id:
        raise Refused("This link changed again after that; undo the newest change first.")
    if v.op == "baseline":
        raise Refused("That is how it was before history started.")
    if v.op == "pending":
        row = await _link_of(session, v)
        if not row or row.status != "pending":
            raise Refused("That proposed link is no longer waiting; nothing to undo.")
        return "retract", p
    if v.op == "reject":
        return "repropose", p
    if v.op == "end" or (v.op == "approve" and v.status == "ended"):
        return ("nothing" if now else "activate"), p
    if now and await _replaced_by(session, v) is not None:
        return "revert", p
    return ("end" if now else "nothing"), p  # write / approve / undo / restore that left it active


async def undo_or_restore(session: AsyncSession, v, mode: str, *, caregiver: bool, by: str, source_kind: str, reason: str = "") -> dict:
    """Carry out an undo (mode 'undo') or restore of a link change through the link rules: a caregiver changes links
    directly; anyone else's change to a health link waits for a caregiver's OK."""
    from app.care import versions

    effect, p = await plan(session, v, mode)
    health = domain_of(p["from_ref"]) in HEALTH_DOMAINS
    wait = health and not caregiver
    why = reason or ("undo" if mode == "undo" else f"restored version {v.version}")
    with versions.attribution(reason=why, undoes=v.id if mode == "undo" else None):
        if effect == "nothing":
            return {"result": "nothing", "why": "the links already say that"}
        if effect == "retract":
            row = await _link_of(session, v)
            if not caregiver and row and row.stated_by != by:
                raise Refused("Only a caregiver, or the person who proposed it, can take back this proposed link.")
            await resolve_pending_link(session, link_id=row.id, approve=False, by=by)
            return {"result": "retracted"}
        if effect == "revert":  # the earlier link comes back and takes this one's place (a caregiver's OK for others' health links)
            cur = (await _active(session, v.family_id, v.subject_id, p["from_ref"], p["relation"], p["to_ref"]))[0]
            old = await _replaced_by(session, v)
            w = await write_link(session, family_id=v.family_id, subject_id=v.subject_id, source_kind=source_kind, stated_by=by,
                                 source_ref=f"undo:{v.id}", force_confirm=wait, confirmed_by=by if caregiver else None,
                                 **_link_parts(old))
            return {"result": w.result, "link": readable(w.link), "instead_of": readable(cur)}
        if effect == "end":
            cur = (await _active(session, v.family_id, v.subject_id, p["from_ref"], p["relation"], p["to_ref"]))[0]
            w = await end_link(session, cur, reason=why, source_kind=source_kind, stated_by=by, force_confirm=wait)
            return {"result": w.result, "link": readable(cur)}
        if effect == "repropose":
            w = await write_link(session, family_id=v.family_id, subject_id=v.subject_id, source_kind=source_kind, stated_by=by,
                                 source_ref=f"undo:{v.id}", force_confirm=True, **p)
            return {"result": w.result, "link": readable(w.link)}
        w = await write_link(session, family_id=v.family_id, subject_id=v.subject_id, source_kind=source_kind, stated_by=by,
                             source_ref=f"undo:{v.id}", force_confirm=wait, confirmed_by=by if caregiver else None, **p)
        return {"result": w.result, "link": readable(w.link)}


async def preview(session: AsyncSession, v, mode: str, *, caregiver: bool = True) -> dict:
    try:
        effect, p = await plan(session, v, mode)
    except Refused as exc:
        return {"effect": "refused", "text": str(exc), "button": ""}
    words = readable(CareLink(from_ref=p["from_ref"], relation=p["relation"], to_ref=p["to_ref"], label=p["label"], detail=p["detail"]))
    wait = domain_of(p["from_ref"]) in HEALTH_DOMAINS and not caregiver
    if effect == "nothing":
        return {"effect": "nothing", "text": "Nothing would change: the links already say that.", "button": ""}
    if effect == "retract":
        return {"effect": "retract", "text": f"This takes back the proposed link: {words}.", "button": "Yes, take it back"}
    if effect == "revert":
        back = readable(await _replaced_by(session, v))
        if wait:
            return {"effect": "pending", "text": f"This proposes putting back {back} (instead of {words}). It waits for a caregiver "
                                                 "to confirm.", "button": "Yes, propose it"}
        return {"effect": "change", "text": f"This puts back the earlier link: {back} (instead of {words}).", "button": "Yes, put it back"}
    if effect == "repropose" or wait:
        return {"effect": "pending", "text": f"This proposes again: {words}. It waits for a caregiver to confirm.", "button": "Yes, propose it"}
    if effect == "end":
        return {"effect": "stop", "text": f"This removes the link: {words}.", "button": "Yes, remove it"}
    return {"effect": "change", "text": f"This puts the link back: {words}.", "button": "Yes, put it back"}


def view_bits(v) -> dict:
    word = {"write": "linked", "pending": "proposed (waiting for a caregiver)", "end": "unlinked", "approve": "approved", "reject": "rejected",
            "snapshot": "restored from a backup"}.get(v.op, v.op)
    return {"label": f"Link: {v.body[:120]}", "body": v.body, "value": v.value, "summary": f"{word}: {v.body[:140]}"}


# ── forget (app.care.memory_upkeep.forget) ───────────────────────────────────


async def forget_matching(session: AsyncSession, family_id: str, subject_ids: list[str], words: list[str], *, by: str,
                          source_kind: str) -> tuple[list[str], list[str]]:
    """End the links whose words hold every word asked to forget: (ended ids, health links kept waiting for a caregiver)."""
    rows = (await session.execute(select(CareLink).where(CareLink.family_id == family_id, CareLink.subject_id.in_(subject_ids),
                                                         CareLink.status == "active"))).scalars().all()
    ended, waiting = [], []
    for ln in rows:
        text = f"{readable(ln)} {name_of(ln.to_ref)}".lower()
        if words and all(w in text for w in words):
            w = await end_link(session, ln, reason="forgotten on request", source_kind=source_kind, stated_by=by)
            if w.result == "ended":
                ended.append(str(ln.id))
            elif w.result == "pending":
                waiting.append(str(w.link.id))
    return ended, waiting


async def bring_back(session: AsyncSession, family_id: str, link_ids: list[str], *, subjects: list[str], by: str) -> list[str]:
    """Undo a forget: the links it ended come back (as new rows, so the history keeps both). Returns the ids handled."""
    done: list[str] = []
    for lid in link_ids or []:
        try:
            old = await session.get(CareLink, uuid.UUID(lid))
        except ValueError:
            continue
        if not old or old.family_id != family_id or old.subject_id not in subjects or old.status != "ended":
            continue
        await write_link(session, family_id=family_id, subject_id=old.subject_id, from_ref=old.from_ref, relation=old.relation,
                         to_ref=old.to_ref, label=old.label, detail=old.detail, source_kind=old.source_kind, stated_by=old.stated_by,
                         source_ref=old.source_ref, confirmed_by=old.confirmed_by or by)
        done.append(lid)
    return done


# ── the record check (app.care.memory_upkeep.health) ─────────────────────────


def _person_slug(name: str) -> str:
    s = name_of(name).lower().replace(".", " ")
    parts = [x for x in s.split() if x not in ("dr", "doctor")]
    return "_".join(parts)


async def record_problems(session: AsyncSession, family_id: str, subject_id: str, active_facts: list) -> list[dict]:
    """Links that point at nothing on the record: a medicine for a condition the record does not have, a prescribing
    doctor (or an appointment's doctor) who is not in the doctors or contacts. Data checks only; each is a question."""
    links = await links_for(session, family_id, subject_id, statuses=("active",))
    keys = {f.key for f in active_facts}
    meds = {f.key: str((f.value or {}).get("name") or name_of(f.key)) for f in active_facts if f.domain in ("medicine", "appointment")}
    people = {_person_slug(str((f.value or {}).get("name") or f.key)) for f in active_facts if f.domain in ("doctor", "contact")}
    out = []
    for ln in links:
        if ln.from_ref not in meds:
            continue  # only links of what is current
        what = meds[ln.from_ref]
        if ln.relation == "for" and ln.to_ref not in keys:
            out.append({"kind": "link_condition_missing", "key": f"linkcond:{ln.from_ref}:{ln.to_ref}"[:120],
                        "problem": f"{what} is noted as for {label_of(ln)}, but {label_of(ln)} is not in the conditions",
                        "ask": f"{what} is noted as for {label_of(ln)}, but {label_of(ln)} isn't in the conditions. Should I add it?"})
        if ln.relation in ("prescribed_by", "with_doctor") and ln.to_ref not in keys and _person_slug(ln.to_ref) not in people:
            verb = "prescribed" if ln.relation == "prescribed_by" else "is the doctor for"
            out.append({"kind": "link_doctor_missing", "key": f"linkdoc:{ln.from_ref}:{ln.to_ref}"[:120],
                        "problem": f"{label_of(ln)} {verb} {what}, but is not in the doctors or contacts",
                        "ask": f"{label_of(ln)} {verb} {what}, but isn't in the doctors. Should I add them, with their number if you have it?"})
    return out
