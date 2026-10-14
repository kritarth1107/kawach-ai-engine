"""Keeping each person's memory small, true and easy to use, every night.

- rollups: the daily diary becomes a weekly summary (Sunday night), weekly summaries become a monthly one (1st of
  the month), and the months become "life so far". Each lives as a memory note, so recall and context find it; old
  diary lines can then go without losing anything.
- health: contradictions and loose ends in the care record (the same medicine twice, a medicine with no time, a
  medicine against an allergy, pending changes nobody confirmed for a week, the same person saved twice, a medicine
  linked to a condition the record does not have or to a doctor who is not in the contacts). Each becomes one gentle
  question to a caregiver, asked once.
- profile card: who this person is, in a few lines (name to use, language, conditions, medicines, allergies, what
  matters to them, how they like to be spoken to), rebuilt nightly from the record, notes and baselines, so every
  reply starts from a clean summary.
- forget / restore: what someone asks Saheli to forget is hidden from memory, matching links end (with a record to
  undo it all).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from datetime import timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import medicine_slug
from app.care.models import CareEvent, CareFact, MemoryNote, OpenLoop
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
    # a pending fact gets one final ask at day 14 and is archived at day 17 (app.care.pending), not asked here
    people: dict[str, list[CareFact]] = defaultdict(list)
    for f in active:
        if f.domain in ("doctor", "contact", "home") and (f.value or {}).get("name"):
            people[re.sub(r"^(dr\.?|doctor)\s+", "", str(f.value["name"]).lower()).split()[0]].append(f)
    for name, fs in people.items():
        if len(fs) > 1 and len({x.key for x in fs}) > 1:
            issues.append({"kind": "duplicate_person", "key": f"person:{name}", "problem": "; ".join(x.text for x in fs),
                           "ask": f"I have {name.title()} saved {len(fs)} times. Are they the same person?"})
    from app.care import links

    # links that point at nothing on the record (D-13): a medicine for a condition not on it, a doctor not in the contacts
    issues += await links.record_problems(session, family_id, subject_id, active)
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


# ── contradictions between notes and the record (D-12) ──────────────────────────

NO_CHECK_NOTES = {"diary", "weekly", "monthly", "profile-card"}  # dated history, and the card built from the record itself
MAX_CONTRADICTION_QUESTIONS = 3  # per person per night: a caregiver is never flooded
CONTRADICTION_PROMPT = (
    "You check a care companion's memory of one person for contradictions. CARE RECORD lines are confirmed and current; "
    "NOTES are free-text memory. List only real contradictions: (1) a note line that cannot be true at the same time as a "
    "CARE RECORD line (a different medicine, dose, time, allergy, condition, doctor, address or family rule); (2) two note "
    "lines that disagree with each other. NOT contradictions: history (a line telling what was true before: 'earlier', "
    "'used to', 'stopped on', a past date), a note adding detail the record does not have, the same thing in other words. "
    "Never decide which side is right. Return ONLY JSON: {\"contradictions\": [{\"record_key\": \"<the key in [ ] from the "
    "CARE RECORD, or empty>\", \"note\": \"<the note slug in [ ]>\", \"line\": \"<that note line, copied exactly>\", "
    "\"other_note\": \"<slug, only for two notes>\", \"other_line\": \"<that line, copied exactly>\", \"question\": \"<one "
    "short, kind question for a caregiver that would settle it, in English>\"}]}, with an empty list when there are none."
)


def _norm(line: str) -> str:
    return " ".join((line or "").strip().lstrip("-*• ").split()).lower()


def _find_line(body: str, quoted: str) -> str | None:
    """The note's own line the model quoted (the same, or one containing the other), or None when it is not there: a
    contradiction is only asked about when both sides can be pointed at."""
    want = _norm(quoted)
    if len(want) < 6:
        return None
    for ln in (body or "").splitlines():
        have = _norm(ln)
        if have and (have == want or (len(want) >= 12 and (want in have or have in want))):
            return ln.strip()
    return None


def _stopped_medicine_lines(facts: list[CareFact], notes: list[MemoryNote]) -> list[str]:
    """Note lines that name a medicine the record has stopped (no current entry for it under any dose): likely
    contradictions, given to the model to confirm (a line telling the history of the stop is not one). Names come from
    the record, not from a word list."""
    current = {medicine_slug(str((f.value or {}).get("name") or f.key.split(":", 1)[1])).split("_")[0]
               for f in facts if f.domain == "medicine" and f.status == "active"}
    ended: dict[str, str] = {}
    for f in facts:
        if f.domain == "medicine" and f.status in ("stopped", "superseded"):
            name = str((f.value or {}).get("name") or f.key.split(":", 1)[1]).replace("_", " ").strip()
            if len(name) >= 3 and medicine_slug(name).split("_")[0] not in current:
                ended[name.lower()] = f.status
    out = []
    for n in notes:
        for ln in (n.body_md or "").splitlines():
            for name in ended:
                if re.search(rf"\b{re.escape(name)}\b", ln.lower()):
                    out.append(f"[{n.slug}] {ln.strip()}  (the record has {name} stopped)")
    return out[:20]


async def _changed_since(session: AsyncSession, family_id: str, subject_id: str, since) -> bool:
    notes = await session.scalar(select(func.count()).select_from(MemoryNote).where(
        MemoryNote.family_id == family_id, MemoryNote.subject_id == subject_id, MemoryNote.updated_at >= since))
    if notes:
        return True
    return bool(await session.scalar(select(func.count()).select_from(CareFact).where(
        CareFact.family_id == family_id, CareFact.subject_id == subject_id, or_(CareFact.recorded_at >= since, CareFact.valid_to >= since))))


async def contradictions(session: AsyncSession, family_id: str, subject_id: str) -> list[dict]:
    """Contradictions between this person's notes and their care record, and between their notes: one cheap model call
    (essential=False, so it stays under the spend caps). Nothing is changed here, ever, in any domain: each one is a
    question for a caregiver."""
    from app.care.extract import parse_json
    from app.llm import router

    facts = await store.facts(session, family_id, subject_id, statuses=("active", "superseded", "stopped"))
    current = [f for f in facts if f.status == "active"]
    notes = [n for n in await store.notes(session, family_id, [subject_id]) if n.slug not in NO_CHECK_NOTES and (n.body_md or "").strip()]
    hints = _stopped_medicine_lines(facts, notes)
    if not notes or (not current and len(notes) < 2 and not hints):
        return []
    record = "\n".join(f"[{f.key}] {f.text}" for f in current) or "(nothing current)"
    body = "\n\n".join(f"[{n.slug}] {n.title}\n{n.body_md.strip()}" for n in notes)
    text = f"CARE RECORD (current):\n{record}\n\nNOTES:\n{body}"[:12000]
    if hints:
        text += ("\n\nCHECK THESE NOTE LINES (they name a medicine the record has stopped; say whether each is a contradiction or "
                 "only history):\n" + "\n".join(hints))
    try:
        out = await router.complete("extract", system_stable=CONTRADICTION_PROMPT,
                                    messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                                    max_tokens=1500, effort="low", essential=False)
    except router.AllModelsFailed:
        return []
    keys = {f.key for f in facts}  # a stopped medicine is a side too: the record says it ended
    by_slug = {n.slug: n for n in notes}
    found = []
    for c in (parse_json(out.text or "").get("contradictions") or [])[:10]:
        if not isinstance(c, dict):
            continue
        note = by_slug.get(str(c.get("note") or "").strip())
        line = _find_line(note.body_md, str(c.get("line") or "")) if note else None
        if not line:
            continue
        rkey = str(c.get("record_key") or "").strip()
        rkey = rkey if rkey in keys else ""
        other = by_slug.get(str(c.get("other_note") or "").strip())
        oline = _find_line(other.body_md, str(c.get("other_line") or "")) if other else None
        if oline and other.slug == note.slug and _norm(oline) == _norm(line):
            oline = None
        if not rkey and not oline:
            continue  # a contradiction needs a second side that really exists
        question = " ".join(str(c.get("question") or "").split())[:300] or (
            f"My notes say \"{line[:120]}\", which does not match what else I have. Which is right now?")
        found.append({"record_key": rkey, "note": note.slug, "line": line, "other_note": other.slug if oline else "",
                      "other_line": oline or "", "question": question})
    return found


async def ask_about_contradictions(session: AsyncSession, family_id: str, subject_id: str, owner_id: str | None) -> int:
    """For a person whose notes or care record changed since yesterday: each contradiction becomes one question to a
    caregiver through the memory_check path, asked once (never again for the same two lines, even after an answer).
    Nothing in the record or the notes is changed here; health facts change only through the record's own rules."""
    if not await _changed_since(session, family_id, subject_id, clock.now() - timedelta(hours=26)):
        return 0
    opened = 0
    for c in await contradictions(session, family_id, subject_id):
        if opened >= MAX_CONTRADICTION_QUESTIONS:
            break
        sig = hashlib.sha256("|".join([subject_id, c["record_key"], c["note"], _norm(c["line"]), c["other_note"], _norm(c["other_line"])])
                             .encode()).hexdigest()[:16]
        key = f"memcheck:{subject_id}:contra:{sig}"
        if await session.scalar(select(func.count()).select_from(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.dedupe_key == key)):
            continue  # asked before
        problem = (f"note '{c['note']}': {c['line']}" + (f" | record [{c['record_key']}]" if c["record_key"] else "")
                   + (f" | note '{c['other_note']}': {c['other_line']}" if c["other_line"] else ""))
        await store.open_loop(
            session, family_id=family_id, subject_id=subject_id, kind="memory_check", title=c["question"],
            detail={"problem": problem[:600], "kind": "contradiction", "max_wakes": 1, "record_key": c["record_key"], "note": c["note"],
                    "line": c["line"][:300], "other_note": c["other_note"], "other_line": c["other_line"][:300]},
            owner_id=owner_id, wake_at=clock.ist().replace(hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1),
            alert_rule="dashboard", dedupe_key=key,
        )
        opened += 1
    return opened


# ── profile card ───────────────────────────────────────────────────────────────


async def profile_card(session: AsyncSession, family_id: str, person: dict) -> str:
    """A few lines that tell Saheli who this person is (built from the record, not invented)."""
    from app.care import baselines

    pid = person["id"]
    from app.care.rejected import REJECTED_NOTE

    facts = await store.facts(session, family_id, pid, statuses=("active",))
    pending = await store.facts(session, family_id, pid, statuses=("pending",))
    withdrawn = {f.key for f in pending if (f.value or {}).get("stopped") and str(f.note or "").startswith(REJECTED_NOTE)}
    facts = [f for f in facts if f.key not in withdrawn]
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


async def forget(session: AsyncSession, family_id: str, subject_ids: list[str], what: str, *, by: str, source_kind: str = "caregiver_said", never_true: bool = False) -> dict:
    """Hide memories matching `what` (notes lines, events) and end matching links (a health link from someone weaker than
    who set it only waits for a caregiver). Matching care-record facts are stopped too (with history). never_true: they
    said it was a joke or never happened, so it is not read back and not proposed again."""
    from app.care import links, memory_index
    from app.care.rejected import REJECTED_NOTE

    words = [w for w in re.findall(r"\w{3,}", (what or "").lower())]
    if not words:
        return {"forgotten": 0, "note": "Nothing to remove. Call forget with the words of the saved event or fact."}
    removed_lines, rewrites = [], []
    notes = list((await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id, MemoryNote.subject_id.in_(subject_ids))
                                        .execution_options(populate_existing=True))).scalars())
    for note in notes:
        keep, drop = [], []
        for ln in (note.body_md or "").splitlines():
            (drop if all(w in ln.lower() for w in words) else keep).append(ln)
        if drop:
            rewrites.append((note, "\n".join(keep), drop))
            removed_lines += [{"subject": note.subject_id, "slug": note.slug, "title": note.title, "line": d} for d in drop]
    events = (await session.execute(select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id.in_(subject_ids)))).scalars()
    hide = [e for e in events if all(w in (e.summary or "").lower() for w in words) and not (e.payload or {}).get("forgotten")]
    ended_links, waiting_links = await links.forget_matching(session, family_id, subject_ids, words, by=by, source_kind=source_kind)
    facts = list((await session.execute(select(CareFact).where(
        CareFact.family_id == family_id, CareFact.subject_id.in_(subject_ids), CareFact.status == "active"))).scalars())
    matched_facts = [f for f in facts if all(w in f"{f.text} {f.key}".lower() for w in words)]
    why = f"asked to remove: {what[:180]}"
    if never_true:
        why = REJECTED_NOTE + why
    stopped_keys: list[str] = []
    waiting_keys: list[str] = []
    for f in matched_facts:
        w = await store.stop_fact(session, family_id=family_id, subject_id=f.subject_id, key=f.key, reason=why,
                                  source_kind=source_kind, stated_by=by)
        if w is None or w.result == "kept_pinned":
            continue
        (stopped_keys if w.result == "stopped" else waiting_keys).append(f.key)
    # the record first, so each note's version can point at it (undoing any of them restores the whole forget)
    eid = await store.record_event(session, family_id=family_id, subject_id=subject_ids[0], kind="memory_forgotten", summary=f"Forgot: {what[:200]}",
                                   payload={"what": what, "by": by, "note_lines": removed_lines, "events": [e.id for e in hide], "forgotten": True,
                                            "links": ended_links, "facts": [*stopped_keys, *waiting_keys]},
                                   actor_id=by)
    for note, body, drop in rewrites:
        title = note.title
        await store.upsert_note(session, family_id=family_id, subject_id=note.subject_id, slug=note.slug, title=title, body_md=body,
                                op="forget", actor_id=by, reason=f"forget: {what[:120]}", meta={"forget_event": eid})
        # exactly the forgotten lines (diary lines too, which a rewrite would otherwise keep findable as history)
        await memory_index.unindex_note_lines(session, family_id, note.subject_id, note.slug, title, drop)
    for e in hide:
        e.payload = {**(e.payload or {}), "forgotten": True, "forgotten_by": by}
        await memory_index.unindex(session, family_id, f"event:{e.id}")
    out: dict[str, object] = {"forgotten": len(removed_lines) + len(hide) + len(ended_links) + len(stopped_keys) + len(waiting_keys)}
    if stopped_keys:
        out["facts"] = stopped_keys
    if waiting_keys:
        out["facts_waiting"] = waiting_keys
    said: list[str] = []
    if out["forgotten"] == 0:
        said.append("Nothing matched. Call forget again with the words of the saved event or fact. A logged event is removed with forget, not with stop.")
    else:
        said.append("Removed from memory. Do not mention it again.")
    if waiting_keys:
        said.append("A health fact waits for a caregiver before the record ends it; do not state it as true meanwhile.")
    if waiting_links:
        out["links_waiting"] = len(waiting_links)
        said.append("A link between a medicine and its condition, doctor or record goes only once a caregiver confirms.")
    out["note"] = " ".join(said)
    return out


async def restore(session: AsyncSession, family_id: str, forget_event_id: int, *, by: str, subjects: list[str]) -> dict:
    """Bring back what one forget hid. Only a forget made on one of `subjects` (the person whose page or chat this is,
    and the family notes) can be restored, and only lines and events about those subjects come back: one caregiver can
    never bring back what another hid from their own self-care record."""
    row = await session.get(CareEvent, forget_event_id)
    if not row or row.family_id != family_id or row.kind != "memory_forgotten" or (row.payload or {}).get("restored"):
        return {"restored": 0}
    if row.subject_id not in subjects:
        return {"restored": 0}
    from app.care import versions

    p = row.payload or {}
    done = set(p.get("restored_subjects") or [])
    mine = [s for s in subjects if s not in done]
    n = 0
    by_note: dict[tuple[str, str], list[dict]] = {}
    for item in p.get("note_lines") or []:
        if item.get("subject") in mine:
            by_note.setdefault((item["subject"], item["slug"]), []).append(item)
    for (subj, slug), items in by_note.items():
        note = await _note(session, family_id, subj, slug)
        # put the lines back where they were, using the note as it was just before the forget
        fv = (await session.execute(select(versions.MemoryVersion).where(
            versions.MemoryVersion.family_id == family_id, versions.MemoryVersion.subject_id == subj, versions.MemoryVersion.kind == "note",
            versions.MemoryVersion.target == slug, versions.MemoryVersion.op == "forget",
            versions.MemoryVersion.value["forget_event"].astext == str(forget_event_id)))).scalars().first()
        before = (await versions.previous(session, fv)).body if fv and await versions.previous(session, fv) else None
        current = note.body_md if note else ""
        if before is not None:
            body = versions.invert(current, before, fv.body)
        else:
            body = "\n".join([*versions.lines(current), *[i["line"] for i in items]])
        await store.upsert_note(session, family_id=family_id, subject_id=subj, slug=slug, title=items[0].get("title") or slug,
                                body_md=body, op="restore", actor_id=by, reason=f"brought back what was forgotten: {(p.get('what') or '')[:100]}")
        n += len(items)
    for eid in p.get("events") or []:
        e = await session.get(CareEvent, eid)
        if e and e.family_id == family_id and e.subject_id in mine and (e.payload or {}).get("forgotten"):
            e.payload = {k: v for k, v in (e.payload or {}).items() if k not in ("forgotten", "forgotten_by")}
            n += 1
    todo = [x for x in p.get("links") or [] if x not in set(p.get("links_restored") or [])]
    if todo:
        from app.care import links

        back = await links.bring_back(session, family_id, todo, subjects=mine, by=by)
        n += len(back)
        p = {**p, "links_restored": [*(p.get("links_restored") or []), *back]}
    # the forget is fully undone only when every person and note it touched is back
    done |= set(mine)
    touched = {i.get("subject") for i in p.get("note_lines") or []} | {row.subject_id}
    row.payload = {**p, "restored_subjects": sorted(done), "restored": touched <= done, "restored_by": by}
    return {"restored": n}
