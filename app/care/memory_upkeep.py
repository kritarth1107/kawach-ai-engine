"""Keeping each person's memory small, true and easy to use, every night.

- rollups: the daily diary becomes a weekly summary (Sunday night), weekly summaries become a monthly one (1st of
  the month), and the months become "life so far". Each lives as a memory note, so recall and context find it; old
  diary lines can then go without losing anything.
- health: contradictions and loose ends in the care record (the same medicine twice, a medicine with no time, a
  medicine against an allergy, pending changes nobody confirmed for a week, the same person saved twice). Each
  becomes one gentle question to a caregiver, asked once.
- profile card: who this person is, in a few lines (name to use, language, conditions, medicines, allergies, what
  matters to them, how they like to be spoken to), rebuilt nightly from the record, notes and baselines, so every
  reply starts from a clean summary.
- forget / restore: what someone asks Saheli to forget is hidden from memory (with a record to undo it).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import medicine_slug
from app.care.models import CareEvent, CareFact, MemoryNote
from app.core import clock

logger = logging.getLogger(__name__)

ROLLUP_PROMPT = (
    "You keep a care companion's memory of one person. Summarise the diary lines below into {n} plain sentences: health "
    "and medicines, how they have been, what happened, what is still open. Keep dates that matter. Use ONLY what is "
    "written; nothing new. Plain text, English."
)
LIFE_PROMPT = (
    "You keep a care companion's memory of one person. Rewrite their 'life so far' summary (max 12 sentences) by merging "
    "the old one with the new monthly summary: health story, medicines changes, people around them, routines, what "
    "they enjoy, what worries them. Keep only what matters for care and companionship; use ONLY what is written."
)


async def _model(prompt: str, text: str, tokens: int = 800) -> str | None:
    from app.llm import router

    try:
        out = await router.complete("extract", system_stable=prompt, messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                                    max_tokens=tokens, effort="low", essential=False)
        return " ".join((out.text or "").split())[:2000] or None
    except router.AllModelsFailed:
        return None


async def _note(session: AsyncSession, family_id: str, subject_id: str, slug: str) -> MemoryNote | None:
    # populate_existing: notes are written with upserts, so a copy already in the session may be stale.
    return (await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id, MemoryNote.subject_id == subject_id,
                                                           MemoryNote.slug == slug).execution_options(populate_existing=True))).scalar_one_or_none()


def _lines_between(body: str, start: str, end: str) -> list[str]:
    out = []
    for ln in (body or "").splitlines():
        m = re.match(r"- (\d{4}-\d{2}-\d{2}):", ln)
        if m and start <= m.group(1) <= end:
            out.append(ln)
    return out


async def rollups(session: AsyncSession, family_id: str, subject_id: str, *, with_models: bool = True) -> dict:
    """Weekly on Sunday night, monthly on the 1st (for the month just ended), life-so-far after a monthly."""
    out = {"weekly": False, "monthly": False, "life": False}
    if not with_models:
        return out
    today = clock.ist()
    diary = await _note(session, family_id, subject_id, "diary")
    if today.weekday() == 6 and diary:
        start = (today - timedelta(days=6)).strftime("%Y-%m-%d")
        week = _lines_between(diary.body_md, start, today.strftime("%Y-%m-%d"))
        if len(week) >= 2:
            text = await _model(ROLLUP_PROMPT.format(n="3 to 5"), "\n".join(week))
            if text:
                label = today.strftime("%G-W%V")
                note = await _note(session, family_id, subject_id, "weekly")
                lines = [ln for ln in (note.body_md.splitlines() if note else []) if not ln.startswith(f"- {label}:")][-11:]
                await store.upsert_note(session, family_id=family_id, subject_id=subject_id, slug="weekly", title="Weekly summaries",
                                        body_md="\n".join(lines + [f"- {label}: {text}"]))
                out["weekly"] = True
    if today.day == 1:
        last = (today.replace(day=1) - timedelta(days=1))
        month = last.strftime("%Y-%m")
        weekly = await _note(session, family_id, subject_id, "weekly")
        source = [ln for ln in (diary.body_md.splitlines() if diary else []) if ln.startswith(f"- {month}")]
        if weekly:
            source += weekly.body_md.splitlines()[-5:]
        if len(source) >= 3:
            text = await _model(ROLLUP_PROMPT.format(n="5 to 8"), "\n".join(source[-60:]), tokens=1200)
            if text:
                note = await _note(session, family_id, subject_id, "monthly")
                lines = [ln for ln in (note.body_md.splitlines() if note else []) if not ln.startswith(f"- {month}:")][-23:]
                await store.upsert_note(session, family_id=family_id, subject_id=subject_id, slug="monthly", title="Monthly summaries",
                                        body_md="\n".join(lines + [f"- {month}: {text}"]))
                out["monthly"] = True
                life = await _note(session, family_id, subject_id, "life-so-far")
                merged = await _model(LIFE_PROMPT, f"OLD:\n{life.body_md if life else '(none)'}\n\nNEW MONTH ({month}):\n{text}", tokens=1500)
                if merged:
                    await store.upsert_note(session, family_id=family_id, subject_id=subject_id, slug="life-so-far", title="Life so far", body_md=merged)
                    out["life"] = True
    return out


# ── health of the record ───────────────────────────────────────────────────────


def _times(f: CareFact) -> list[str]:
    return [t for t in (f.value or {}).get("times") or [] if isinstance(t, str)]


async def health(session: AsyncSession, family_id: str, subject_id: str) -> list[dict]:
    """Problems in this person's care record, each {kind, key, problem, ask}."""
    from app.brain import policy

    rows = await store.facts(session, family_id, subject_id, statuses=("active", "pending"))
    active = [f for f in rows if f.status == "active"]
    issues: list[dict] = []
    meds = [f for f in active if f.domain == "medicine"]
    by_first: dict[str, list[CareFact]] = defaultdict(list)
    for f in meds:
        by_first[medicine_slug((f.value or {}).get("name") or f.key.split(":", 1)[1]).split("_")[0]].append(f)
    for first, fs in by_first.items():
        if len(fs) > 1:
            issues.append({"kind": "duplicate_medicine", "key": f"dup:{first}", "problem": f"{len(fs)} entries for {first}: " + "; ".join(x.text for x in fs),
                           "ask": f"I have {first} saved more than once. Which one is right now?"})
    for f in meds:
        if not _times(f):
            issues.append({"kind": "no_time", "key": f"notime:{f.key}", "problem": f"{f.text}: no time saved, so no reminder",
                           "ask": f"What time is {(f.value or {}).get('name') or f.key.split(':', 1)[1]} taken? I'll set the reminder."})
    allergies = [(f.value or {}).get("allergen") or f.key.split(":", 1)[1] for f in active if f.domain == "allergy"]
    for f in meds:
        for c in policy.order_conflicts(f.text, allergies, []):
            issues.append({"kind": "allergy_conflict", "key": f"allergy:{f.key}", "problem": f"{f.text} vs {c}",
                           "ask": f"{(f.value or {}).get('name')} is in the medicines, but there's an {c}. Has the doctor checked this?"})
    week_ago = clock.now() - timedelta(days=7)
    for f in rows:
        if f.status == "pending" and f.recorded_at < week_ago:
            issues.append({"kind": "stale_pending", "key": f"pending:{f.key}", "problem": f"waiting for a caregiver since {clock.ist_day(f.recorded_at)}: {f.text}",
                           "ask": f"A change has been waiting a week for your OK: \"{f.text}\". Should I keep it or drop it?"})
    people: dict[str, list[CareFact]] = defaultdict(list)
    for f in active:
        if f.domain in ("doctor", "contact", "home") and (f.value or {}).get("name"):
            people[re.sub(r"^(dr\.?|doctor)\s+", "", str(f.value["name"]).lower()).split()[0]].append(f)
    for name, fs in people.items():
        if len(fs) > 1 and len({x.key for x in fs}) > 1:
            issues.append({"kind": "duplicate_person", "key": f"person:{name}", "problem": "; ".join(x.text for x in fs),
                           "ask": f"I have {name.title()} saved {len(fs)} times. Are they the same person?"})
    return issues


async def ask_about_health(session: AsyncSession, family_id: str, subject_id: str, owner_id: str | None) -> int:
    """Each new problem becomes one follow-up to a caregiver (deduplicated, asked once)."""
    opened = 0
    for i in await health(session, family_id, subject_id):
        loop = await store.open_loop(
            session, family_id=family_id, subject_id=subject_id, kind="memory_check", title=i["ask"],
            detail={"problem": i["problem"], "kind": i["kind"], "max_wakes": 1}, owner_id=owner_id,
            wake_at=clock.ist().replace(hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1),
            alert_rule="dashboard", dedupe_key=f"memcheck:{subject_id}:{i['key']}",
        )
        opened += 1 if loop else 0
    return opened


# ── profile card ───────────────────────────────────────────────────────────────


async def profile_card(session: AsyncSession, family_id: str, person: dict) -> str:
    """A few lines that tell Saheli who this person is (built from the record, not invented)."""
    from app.care import baselines

    pid = person["id"]
    facts = await store.facts(session, family_id, pid, statuses=("active",))
    by: dict[str, list[CareFact]] = defaultdict(list)
    for f in facts:
        by[f.domain].append(f)
    lines = []
    call = next(((f.value or {}).get("name") for f in by.get("naming", []) if f.key.endswith("address_as")), None)
    lines.append(f"Name: {person.get('name')}" + (f"; call them '{call}'" if call else ""))
    if by.get("language"):
        lines.append("Language: " + "; ".join(f.text for f in by["language"][:2]))
    if by.get("condition"):
        lines.append("Conditions: " + ", ".join(((f.value or {}).get("name") or f.key.split(":", 1)[1]).replace("_", " ") for f in by["condition"][:8]))
    if by.get("medicine"):
        lines.append(f"Medicines: {len(by['medicine'])} (" + ", ".join(((f.value or {}).get("name") or f.key.split(":", 1)[1]) for f in by["medicine"][:10]) + ")")
    if by.get("allergy"):
        lines.append("Allergies: " + ", ".join(((f.value or {}).get("allergen") or f.key.split(":", 1)[1]) for f in by["allergy"]))
    if by.get("diet"):
        lines.append("Diet: " + "; ".join(f.text for f in by["diet"][:4]))
    if by.get("routine"):
        lines.append("Routine: " + "; ".join(f.text for f in by["routine"][:4]))
    if by.get("preference"):
        lines.append("Likes / dislikes: " + "; ".join(f.text for f in by["preference"][:5]))
    if by.get("family"):
        lines.append("Family rules: " + "; ".join(f.text for f in by["family"][:4] if f.key != "family:learning_consent"))
    b = await baselines.get(session, family_id, pid)
    style = (b or {}).get("style") or {}
    if style:
        lines.append("How they like replies: " + style.get("summary", ""))
    life = await _note(session, family_id, pid, "life-so-far")
    if life and life.body_md:
        lines.append("Life so far: " + life.body_md[:500])
    return "\n".join(ln for ln in lines if ln.split(":", 1)[1].strip())


async def save_profile_card(session: AsyncSession, family_id: str, person: dict) -> str:
    card = await profile_card(session, family_id, person)
    if card:
        await store.upsert_note(session, family_id=family_id, subject_id=person["id"], slug="profile-card", title="Profile card", body_md=card)
    return card


# ── forget and restore ─────────────────────────────────────────────────────────


async def forget(session: AsyncSession, family_id: str, subject_ids: list[str], what: str, *, by: str) -> dict:
    """Hide memories matching `what` (notes lines, events). Care-record facts are stopped instead (with history)."""
    from app.care import memory_index

    words = [w for w in re.findall(r"\w{3,}", (what or "").lower())]
    if not words:
        return {"forgotten": 0}
    removed_lines, rewrites = [], []
    notes = list((await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id, MemoryNote.subject_id.in_(subject_ids))
                                        .execution_options(populate_existing=True))).scalars())
    for note in notes:
        keep, drop = [], []
        for ln in (note.body_md or "").splitlines():
            (drop if all(w in ln.lower() for w in words) else keep).append(ln)
        if drop:
            rewrites.append((note, "\n".join(keep)))
            removed_lines += [{"subject": note.subject_id, "slug": note.slug, "title": note.title, "line": d} for d in drop]
    events = (await session.execute(select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id.in_(subject_ids)))).scalars()
    hide = [e for e in events if all(w in (e.summary or "").lower() for w in words) and not (e.payload or {}).get("forgotten")]
    # the record first, so each note's version can point at it (undoing any of them restores the whole forget)
    eid = await store.record_event(session, family_id=family_id, subject_id=subject_ids[0], kind="memory_forgotten", summary=f"Forgot: {what[:200]}",
                                   payload={"what": what, "by": by, "note_lines": removed_lines, "events": [e.id for e in hide], "forgotten": True},
                                   actor_id=by)
    for note, body in rewrites:
        await store.upsert_note(session, family_id=family_id, subject_id=note.subject_id, slug=note.slug, title=note.title, body_md=body,
                                op="forget", actor_id=by, reason=f"forget: {what[:120]}", meta={"forget_event": eid})
        await memory_index.unindex(session, family_id, f"note:{note.slug}")
    for e in hide:
        e.payload = {**(e.payload or {}), "forgotten": True, "forgotten_by": by}
        await memory_index.unindex(session, family_id, f"event:{e.id}")
    return {"forgotten": len(removed_lines) + len(hide)}


async def restore(session: AsyncSession, family_id: str, forget_event_id: int, *, by: str, subjects: list[str]) -> dict:
    """Bring back what one forget hid. Only a forget made on one of `subjects` (the person whose page or chat this is,
    and the family notes) can be restored, and only lines and events about those subjects come back: one caregiver can
    never bring back what another hid from their own self-care record."""
    row = await session.get(CareEvent, forget_event_id)
    if not row or row.family_id != family_id or row.kind != "memory_forgotten" or (row.payload or {}).get("restored"):
        return {"restored": 0}
    if row.subject_id not in subjects:
        return {"restored": 0}
    p = row.payload or {}
    n = 0
    for item in p.get("note_lines") or []:
        if item.get("subject") not in subjects:
            continue
        note = await _note(session, family_id, item["subject"], item["slug"])
        body = ((note.body_md + "\n") if note and note.body_md else "") + item["line"]
        await store.upsert_note(session, family_id=family_id, subject_id=item["subject"], slug=item["slug"], title=item.get("title") or item["slug"],
                                body_md=body, op="restore", actor_id=by, reason=f"brought back what was forgotten: {(p.get('what') or '')[:100]}")
        n += 1
    for eid in p.get("events") or []:
        e = await session.get(CareEvent, eid)
        if e and e.family_id == family_id and e.subject_id in subjects:
            e.payload = {k: v for k, v in (e.payload or {}).items() if k not in ("forgotten", "forgotten_by")}
            n += 1
    row.payload = {**p, "restored": True, "restored_by": by}
    return {"restored": n}
