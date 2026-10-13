"""One memory is the truth (trust plan A1): what is still useful in the old stores moves into the care record, once.

The old stores:
- the backend's Mongo profile (ElderProfile: learned and confirmed facts, rejected and blocked ones, tuning, name to use,
  allergies, diet rules, medicines), ElderUsuals, SaheliCompanion dates, VoicePreference and SaheliPermissions, read
  through the backend tool `export_legacy_memory` (read only);
- the engine's own Instinct inbox: family_memories, memory_entities and memory_profiles (read only here).

Mapping (old → care record):

| Old | New | Source kind |
|---|---|---|
| ElderProfile fact, caregiver_confirmed / caregiver_edited | care fact in its domain (pinned by the store) | caregiver_said |
| ElderProfile fact, learned | care fact; a health domain waits as pending for a caregiver | inferred |
| ElderProfile fact, rejected / blocked | the rejected list: a stopped fact noted "rejected: …" | the caregiver's |
| ElderProfile fact, faded | a line in the person's "earlier" note (history only) | — |
| tuning.addressAs / avoidMaa / nameToUse | naming:address_as {name, avoid[]} (pinned) | caregiver_said if a caregiver set it, else elder_said |
| allergies, diet rules, medicines | allergy / diet / medicine facts; a medicine is never added beside the backend's own schedules, and otherwise waits for a caregiver | as recorded |
| ElderUsuals items and brand preferences | preference:usual_<item> facts | inferred |
| SaheliCompanion birthday / important dates | occasion facts | caregiver_said |
| Instinct inbox rows, entities, profile | the person's notes "about" and "life"; a health statement becomes a pending fact | inferred |
| VoicePreference | language:preferred (language, dialect, script, voice) when the record has none | the person's |
| SaheliPermissions | boundary:policy is already the truth: compared, counted when they differ, never copied over it | — |
| ElderWellbeingDay, DailySnapshot | not moved (the engine's mood, baselines and nightly brief rebuild them) | — |

Rules:
- The care record is newer than the old stores and wins: an item already there (the same key, or the same thing in other
  words by the duplicate check `dedupe.check`, model judgement) is not written again; one with more detail keeps that
  detail as a line in the "earlier" note, never over the newer fact. Pinned facts and the rejected list are honoured by
  the store itself. The model reads the old words (one call per person to sort them); nothing reads them with word lists.
- Every row carries source_ref "import:<store>:<id>", and a ledger event (kind memory_import, the same ref) per item, so
  a rerun changes nothing and every import is traceable.
- `undo_family` takes an import back through memory versions (a note line by line, a fact back to what it was).
- Dry run (apply=False) writes nothing and returns per-family counts only: no personal text leaves this module.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import dedupe, store, versions
from app.care.domains import DOMAINS, HEALTH_DOMAINS, fact_key, medicine_slug
from app.care.models import CareEvent, CareFact, OpenLoop
from app.core import clock

logger = logging.getLogger(__name__)

EVENT = "memory_import"
UNDONE = "memory_import_undone"
NOTE_TITLES = {"about": "About", "life": "Life", "earlier": "Earlier (history only)"}
REJECTED_NOTE = "rejected: "  # app.care.rejected.REJECTED_NOTE
CAREGIVER_STATUSES = ("caregiver_confirmed", "caregiver_edited")
ACTOR, SOURCE, REASON = "kavach", "import", "moved from the older memory (trust plan A1)"


@dataclass
class Item:
    """One thing an old store holds, and what the care record makes of it."""

    store: str  # elder_profile | elder_usuals | saheli_companion | voice_preference | saheli_permissions | instinct_inbox | instinct_entity | instinct_profile
    id: str
    subject_id: str
    kind: str  # fact | rejected | earlier | note | naming | language | boundary | medicine | free (the model sorts it)
    text: str = ""
    domain: str = ""
    name: str = ""
    value: dict = field(default_factory=dict)
    source_kind: str = "inferred"
    stated_by: str | None = None
    pending: bool = False
    note: str = ""
    when: str = ""
    old_status: str = ""
    family: str = ""  # store family for the report: profile_fact, profile_tuning, profile_lists, usuals, dates, voice, permissions, instinct
    outcome: str = ""
    covered: bool = False

    @property
    def ref(self) -> str:
        return f"import:{self.store}:{self.id}"[:160]


# ── reading the old stores ───────────────────────────────────────────────────


def _day(v: object) -> str:
    if isinstance(v, datetime):
        return clock.ist_day(v)
    s = str(v or "")
    return s[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", s) else ""


def _str(v: object, n: int = 500) -> str:
    return " ".join(str(v or "").split())[:n]


def profile_items(person: dict) -> list[Item]:
    """ElderProfile → items (the model sorts the free-text facts later)."""
    sid = str(person.get("id") or "")
    p = person.get("profile") or {}
    pid = str(p.get("id") or sid)
    caregivers = set(person.get("caregiver_ids") or [])
    out: list[Item] = []
    for f in p.get("facts") or []:
        fid, text = str(f.get("id") or ""), _str(f.get("text"))
        if not fid or not text:
            continue
        status = str(f.get("status") or "learned")
        when = _day(f.get("firstSeen") or f.get("lastConfirmed"))
        by = str(f.get("editedBy") or "") or None
        item = Item(store="elder_profile", id=fid, subject_id=sid, kind="free", text=text, when=when, old_status=status, family="profile_fact")
        if f.get("blocked") or status == "rejected":
            item.kind, item.source_kind, item.stated_by = "rejected", "caregiver_said", by
        elif status == "faded":
            item.kind, item.note = "earlier", "earlier"
        elif status in CAREGIVER_STATUSES or f.get("pinned"):
            item.source_kind, item.stated_by = "caregiver_said", by
        out.append(item)
    tuning = p.get("tuning") or {}
    name = _str(p.get("nameToUse") or ("" if tuning.get("avoidMaa") else tuning.get("addressAs")), 60)
    avoid = ["Maa"] if tuning.get("avoidMaa") else []  # the old profile's one structured avoid flag, as data
    if name or avoid:
        by = str(tuning.get("setBy") or "") or None
        caregiver = bool(by and by in caregivers)
        out.append(Item(store="elder_profile", id=f"{pid}#naming", subject_id=sid, kind="naming", domain="naming", name="address_as",
                        value={**({"name": name} if name else {}), **({"avoid": avoid} if avoid else {})},
                        text=(f"Call them {name}" if name else "") + ("; never call them " + ", ".join(avoid) if avoid else ""),
                        source_kind="caregiver_said" if caregiver else "elder_said", stated_by=by if caregiver else sid,
                        family="profile_tuning"))
    for i, a in enumerate(p.get("allergies") or []):
        a = _str(a, 120)
        if a:
            out.append(Item(store="elder_profile", id=f"{pid}#allergy{i}", subject_id=sid, kind="fact", domain="allergy", name=a,
                            value={"allergen": a}, text=f"Allergic to {a}", source_kind="import", family="profile_lists"))
    for i, r in enumerate(p.get("dietRules") or []):
        r = _str(r, 200)
        if r:  # a diet rule is free text: the model gives it its key
            out.append(Item(store="elder_profile", id=f"{pid}#diet{i}", subject_id=sid, kind="free", text=r, source_kind="import",
                            family="profile_lists"))
    for i, m in enumerate(p.get("medicines") or []):
        n = _str((m or {}).get("name"), 120)
        if not n or (m or {}).get("active") is False:
            continue
        val = {"name": n, **({"dose": _str(m["dose"], 60)} if m.get("dose") else {}), **({"times": [_str(m["time"], 10)]} if m.get("time") else {})}
        out.append(Item(store="elder_profile", id=f"{pid}#medicine{i}", subject_id=sid, kind="medicine", domain="medicine", name=n,
                        value=val, text=" ".join(x for x in (n, val.get("dose") or "", ("at " + m["time"]) if m.get("time") else "") if x),
                        source_kind="import", family="profile_lists"))
    return out


def usual_items(person: dict) -> list[Item]:
    sid = str(person.get("id") or "")
    u = person.get("usuals") or {}
    uid = str(u.get("id") or sid)
    out: list[Item] = []
    for i, it in enumerate(u.get("items") or []):
        n = _str(it.get("name") or it.get("key"), 120)
        if not n:
            continue
        val = {"item": n, **{k: it[k] for k in ("partner", "category", "count") if it.get(k) not in (None, "")}, "usual": True}
        out.append(Item(store="elder_usuals", id=f"{uid}#item{i}", subject_id=sid, kind="fact", domain="preference", name=f"usual_{n}",
                        value=val, text=f"Usually orders {n}" + (f" from {it['partner']}" if it.get("partner") else ""),
                        when=_day(it.get("lastAt")), family="usuals"))
    for i, b in enumerate(u.get("brandPrefs") or []):
        brand, item = _str(b.get("brand"), 80), _str(b.get("item"), 80)
        if not brand:
            continue
        n = item or brand
        out.append(Item(store="elder_usuals", id=f"{uid}#brand{i}", subject_id=sid, kind="fact", domain="preference", name=f"usual_{n}",
                        value={"item": n, "brand": brand, "usual": True}, text=_str(b.get("text"), 200) or f"Prefers {brand}" + (f" for {item}" if item else ""),
                        when=_day(b.get("at")), family="usuals"))
    return out


def date_items(person: dict) -> list[Item]:
    sid = str(person.get("id") or "")
    c = person.get("companion") or {}
    cid = str(c.get("id") or sid)
    out: list[Item] = []
    if c.get("birthday"):
        d = _day(c["birthday"]) or _str(c["birthday"], 20)
        out.append(Item(store="saheli_companion", id=f"{cid}#birthday", subject_id=sid, kind="fact", domain="occasion", name="birthday",
                        value={"what": "birthday", "date": d}, text=f"Birthday on {d}", source_kind="caregiver_said", family="dates"))
    for i, x in enumerate(c.get("importantDates") or []):
        label, d = _str((x or {}).get("label"), 80), _day((x or {}).get("date")) or _str((x or {}).get("date"), 20)
        if label and d:
            out.append(Item(store="saheli_companion", id=f"{cid}#date{i}", subject_id=sid, kind="fact", domain="occasion", name=label,
                            value={"what": label, "date": d}, text=f"{label} on {d}", source_kind="caregiver_said", family="dates"))
    return out


def voice_items(person: dict) -> list[Item]:
    sid = str(person.get("id") or "")
    v = person.get("voice") or {}
    if not v or not (v.get("language") or v.get("mode")):
        return []
    from app.care import language

    speech = language.normalise({k: v.get(k) for k in ("language", "dialect", "script")})
    if v.get("mode") in ("auto", "always", "never"):
        speech["voice"] = v["mode"]
    by = str(v.get("updatedBy") or "") or None
    caregiver = bool(by and by != sid and by in set(person.get("caregiver_ids") or []))
    return [Item(store="voice_preference", id=str(v.get("id") or sid), subject_id=sid, kind="language", domain="language", name="preferred",
                 value=speech, text=language.sentence(speech) if speech.get("language") else "Voice replies: " + str(speech.get("voice")),
                 source_kind="caregiver_said" if caregiver else "elder_said", stated_by=by or sid, family="voice")]


async def instinct_items(session: AsyncSession, family_id: str, people: list[dict]) -> list[Item]:
    """The Instinct inbox for each person the backend maps to an Instinct elder (AiTenant): live inbox rows, entity bodies
    and the rendered profile, one item per line."""
    from app.models.entities import FamilyMemory, MemoryEntity, MemoryProfile

    out: list[Item] = []
    for person in people:
        link = person.get("instinct") or {}
        try:
            fam, eld = uuid.UUID(str(link.get("aiFamilyId"))), uuid.UUID(str(link.get("aiElderId")))
        except (TypeError, ValueError):
            continue
        sid = str(person.get("id") or "")
        rows = (await session.execute(select(FamilyMemory).where(
            FamilyMemory.family_id == fam, FamilyMemory.elder_id == eld, FamilyMemory.forgotten_at.is_(None),
            FamilyMemory.superseded_by.is_(None)).order_by(FamilyMemory.created_at))).scalars().all()
        for r in rows:
            if r.entity_id is None and _str(r.content):  # consolidated rows live on in their entity's body
                out.append(Item(store="instinct_inbox", id=str(r.id), subject_id=sid, kind="free", text=_str(r.content),
                                when=_day(r.created_at), family="instinct"))
        for e in (await session.execute(select(MemoryEntity).where(MemoryEntity.family_id == fam, MemoryEntity.elder_id == eld)
                                        .order_by(MemoryEntity.created_at))).scalars().all():
            for n, line in enumerate(body_lines(e.body_md)):
                out.append(Item(store="instinct_entity", id=f"{e.id}#{n}", subject_id=sid, kind="free", text=line,
                                when=_day(e.updated_at), family="instinct"))
        prof = (await session.execute(select(MemoryProfile).where(MemoryProfile.family_id == fam, MemoryProfile.elder_id == eld))).scalar_one_or_none()
        if prof:
            for n, line in enumerate(body_lines(prof.body_md)):
                out.append(Item(store="instinct_profile", id=f"{prof.id}#{n}", subject_id=sid, kind="free", text=line,
                                when=_day(prof.rendered_at or prof.created_at), family="instinct"))
    return out


def body_lines(md: str) -> list[str]:
    """The content lines of an Instinct markdown body: no front matter, headings, fences or empty bullets."""
    text = md or ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        text = text[end + 4:] if end > 0 else ""
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith(("#", "```", "<", "---")):
            continue
        s = s.lstrip("-*• ").strip()
        if len(s) >= 4:
            lines.append(_str(s))
    return lines[:120]


# ── the model sorts the free text ────────────────────────────────────────────

SORT_PROMPT = """You move what is still useful from an older memory of one person into their care record.

OLD lists items from the older memory, each with an id. SAVED lists what the care record and its notes already hold.
For EVERY OLD item decide one target:
- "fact": it drives care and fits one of these domains: {domains}. Give domain, name (a short name of the thing itself:
  the medicine, the allergen, the rule, the person, the item, the place), details (structured fields), sentence (one
  plain sentence for the record, in English, keeping every detail of the item and nothing more).
- "note": a story, a person in their life, what they enjoy, how they like things, their background. Give note "about"
  (who they are, their people, likes and ways) or "life" (stories and events), name (a short topic) and sentence.
- "skip": no longer useful (a passing mood or a single day, Saheli's own chatter, a reminder log line, something only
  about the old app) or the same as another OLD item (give same_as with that item's id; keep the fuller one).
An item marked REJECTED is something the family said is WRONG: give it as a "fact" (or "note") with its domain, name and
sentence so it goes on the never-use list; never skip it unless it says nothing at all.
covered: true when SAVED already says the same thing (other words or another language count; a SAVED line marked
[rejected] counts as said).
Never invent anything that is not in the item. Keep names as written.
Return only JSON: {{"items": [{{"id": "...", "target": "fact|note|skip", "domain": "...", "name": "...", "details": {{}},
"sentence": "...", "note": "about|life", "covered": false, "same_as": null}}]}}"""


def _json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def _saved_lines(session: AsyncSession, family_id: str, subject_id: str) -> list[str]:
    from app.care import rejected

    lines = [f"- [{f.key}] {f.text}" for f in await store.facts(session, family_id, subject_id, statuses=("active", "pending"))]
    lines += [f"- [rejected] {f.text}" for f in await rejected.rejected(session, family_id, subject_id)]
    for n in await store.notes(session, family_id, [subject_id]):
        lines += [f"- [note {n.slug}] {ln.strip()[:300]}" for ln in (n.body_md or "").splitlines() if ln.strip()][:60]
    return lines[:300]


async def sort_free(session: AsyncSession, family_id: str, subject_id: str, items: list[Item]) -> int:
    """One model call per person (in chunks of 60): each free-text item gets a target. Returns the calls made. A failure
    leaves the items unsorted (outcome "unsorted"): nothing is guessed."""
    todo = [i for i in items if i.kind in ("free", "rejected")]
    if not todo:
        return 0
    from app.llm import router

    saved = "\n".join(await _saved_lines(session, family_id, subject_id)) or "- nothing yet"
    prompt = SORT_PROMPT.format(domains=", ".join(f"{k} ({v.split(':')[0]})" for k, v in DOMAINS.items()))
    calls = 0
    for start in range(0, len(todo), 60):
        chunk = todo[start:start + 60]
        old = "\n".join(f"- id={i.ref}{' REJECTED' if i.kind == 'rejected' else ''}: {i.text}" for i in chunk)
        calls += 1
        try:
            reply = await router.complete("extract", system_stable=prompt, effort="low", max_tokens=6000, timeout_s=90,
                                          messages=[{"role": "user", "content": [{"type": "text", "text": f"SAVED\n{saved}\n\nOLD\n{old}"}]}])
        except Exception:  # noqa: BLE001
            logger.warning("memory unify: sorting failed family=%s", family_id)
            continue
        by_ref = {i.ref: i for i in chunk}
        for d in _json(reply.text or "").get("items") or []:
            it = by_ref.get(str((d or {}).get("id") or ""))
            if it is None:
                continue
            _apply_sort(it, d)
    for i in todo:
        if i.kind == "free":
            i.outcome = i.outcome or "unsorted"
    return calls


def _apply_sort(it: Item, d: dict) -> None:
    target = str(d.get("target") or "")
    sentence = _str(d.get("sentence"), 500) or it.text
    name = _str(d.get("name"), 120)
    domain = str(d.get("domain") or "")
    raw = d.get("details")
    details: dict = dict(raw) if isinstance(raw, dict) else {}
    it.covered = d.get("covered") is True
    if it.kind == "rejected":
        it.domain = domain if domain in DOMAINS else "preference"
        it.name, it.value, it.text = name or sentence[:60], details, sentence
        return
    if target == "skip":
        it.kind, it.outcome = "skip", ("same_as_other_old_item" if d.get("same_as") else "not_useful")
    elif target == "fact" and domain in DOMAINS and name:
        it.kind, it.domain, it.name, it.value, it.text = "fact", domain, name, details, sentence
        if domain == "medicine":
            it.kind = "medicine"
        # a learned (or Instinct) health fact waits for a caregiver; what a caregiver confirmed does not
        it.pending = domain in HEALTH_DOMAINS and it.source_kind == "inferred"
    elif target in ("note", "fact"):
        it.kind, it.text = "note", sentence
        it.note = "life" if d.get("note") == "life" else "about"
    else:
        it.outcome = "unsorted"


# ── deciding and writing ─────────────────────────────────────────────────────


async def _imported(session: AsyncSession, family_id: str, ref: str) -> bool:
    return (await session.execute(select(CareEvent.id).where(
        CareEvent.family_id == family_id, CareEvent.kind == EVENT, CareEvent.ref == ref).limit(1))).first() is not None


async def _schedule_names(person: dict) -> set[str]:
    return {medicine_slug(str(s.get("title") or "")) for s in person.get("schedules") or [] if s.get("title")}


async def decide(session: AsyncSession, family_id: str, it: Item, *, schedules: set[str]) -> str:
    """What the care record makes of one item: written (fact, pending, rejected, note line, naming, language) or why not."""
    if it.outcome:
        return it.outcome
    if await _imported(session, family_id, it.ref):
        return "already_imported"
    if it.kind == "skip":
        return it.outcome or "not_useful"
    if it.kind in ("earlier", "note"):
        if it.kind == "note" and it.covered:
            return "in_engine"
        return "note_line"
    if it.kind == "naming":
        now = await store.active_fact(session, family_id, it.subject_id, "naming:address_as")
        if not now:
            return "write"
        have = {str(w).strip().lower() for w in (now.value or {}).get("avoid") or []}
        missing = [w for w in it.value.get("avoid") or [] if w.strip().lower() not in have]
        if not missing:
            return "in_engine"  # the record's name is newer than the old profile's; only a missing avoid word is added
        it.value = {"avoid": missing}
        return "write"
    if it.kind == "language":
        now = await store.active_fact(session, family_id, it.subject_id, "language:preferred")
        if not now:
            return "write"
        if it.value.get("voice") and not (now.value or {}).get("voice"):
            it.value = {**(now.value or {}), "voice": it.value["voice"]}  # the record's language stays; the voice setting joins it
            return "write"
        return "in_engine"
    if not it.domain or it.domain not in DOMAINS:
        return "unsorted"
    key = fact_key(it.domain, it.name)
    if it.kind == "medicine":
        if await store.active_fact(session, family_id, it.subject_id, key) or await _pending(session, family_id, it.subject_id, key):
            return "in_engine"
        if medicine_slug(it.name) in schedules:
            return "in_schedules"  # the backend's own schedule already carries it (claimed into the record at import)
        it.pending = True  # a caregiver confirms a medicine nobody scheduled; it never starts reminders on its own
        return "write"
    if await store.active_fact(session, family_id, it.subject_id, key) or await _pending(session, family_id, it.subject_id, key):
        return "in_engine"
    if it.kind == "rejected":
        same = await dedupe.check(session, family_id=family_id, subject_id=it.subject_id, domain=it.domain, key=key, text=it.text, value=it.value)
        if same is not None:
            return "already_rejected" if same.rejected else "in_engine"  # a person said it again since: the newer fact wins
        return "write"
    same = await dedupe.check(session, family_id=family_id, subject_id=it.subject_id, domain=it.domain, key=key, text=it.text, value=it.value)
    if same is not None:
        if same.rejected:
            return "already_rejected"
        if same.covered:
            return "in_engine"
        it.kind, it.note = "earlier", "earlier"  # more detail than the newer fact: kept as history, never written over it
        return "detail_to_note"
    if it.covered and it.domain not in dedupe.CHECK_DOMAINS:
        return "in_engine"  # domains the duplicate check does not cover: the sorting call's judgement
    return "write"


async def _pending(session: AsyncSession, family_id: str, subject_id: str, key: str) -> bool:
    return (await session.execute(select(CareFact.id).where(
        CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.key == key, CareFact.status == "pending").limit(1))).first() is not None


async def _write(session: AsyncSession, family_id: str, it: Item) -> dict:
    """Carry out one decided item; returns the ledger payload (what it became, the row and version to undo)."""
    if it.kind == "rejected":
        now = clock.now()
        row = CareFact(id=uuid.uuid4(), family_id=family_id, subject_id=it.subject_id, domain=it.domain, key=fact_key(it.domain, it.name),
                       value=it.value or {}, text=it.text, status="stopped", source_kind=it.source_kind, source_ref=it.ref,
                       stated_by=it.stated_by, confirmed_by=it.stated_by, confidence=1.0, valid_from=now, recorded_at=now, valid_to=now,
                       note=f"{REJECTED_NOTE}the family said this is wrong (earlier profile)")
        session.add(row)
        await session.flush()
        v = await versions.record_fact(session, row, "stop")
        return {"action": "rejected", "fact_id": str(row.id), "version": v.id}
    key = fact_key(it.domain, it.name)
    w = await store.write_fact(session, family_id=family_id, subject_id=it.subject_id, domain=it.domain, key=key, value=it.value,
                               text=it.text, source_kind=it.source_kind, source_ref=it.ref, stated_by=it.stated_by,
                               confidence=1.0 if it.source_kind in ("caregiver_said", "elder_said") else 0.8,
                               force_confirm=it.pending)
    v = await versions.latest(session, family_id, it.subject_id, "fact", w.fact.key)
    return {"action": w.result, "fact_id": str(w.fact.id), "version": v.id if v else None}


async def _write_notes(session: AsyncSession, family_id: str, subject_id: str, items: list[Item]) -> dict[str, int]:
    """Append each note item as a dated line to the person's note (one write per note); returns slug → version id."""
    out: dict[str, int] = {}
    by_slug: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_slug[it.note or "about"].append(it)
    existing = {n.slug: n for n in await store.notes(session, family_id, [subject_id])}
    for s, rows in by_slug.items():
        lines = [f"- {it.when + ': ' if it.when else ''}{it.text}" for it in rows]
        old = existing.get(s)
        body = "\n".join([*(old.body_md.rstrip().splitlines() if old and old.body_md.strip() else []), *lines])
        note = await store.upsert_note(session, family_id=family_id, subject_id=subject_id, slug=s, title=(old.title if old else NOTE_TITLES[s]),
                                       body_md=body, meta={"import_refs": [it.ref for it in rows][:200]})
        v = await versions.latest(session, family_id, subject_id, "note", note.slug)
        out[s] = v.id if v else 0
    return out


async def _ledger(session: AsyncSession, family_id: str, it: Item, payload: dict) -> None:
    await store.record_event(session, family_id=family_id, subject_id=it.subject_id, kind=EVENT, ref=it.ref, actor_id="kavach",
                             summary=f"Moved from the older memory ({it.store}): {payload.get('action')}",
                             payload={"store": it.store, "id": it.id, **payload})


# ── one family ───────────────────────────────────────────────────────────────


async def gather(session: AsyncSession, data: dict) -> list[Item]:
    people = [p for p in data.get("people") or [] if p.get("id")]
    items: list[Item] = []
    for p in people:
        items += profile_items(p) + usual_items(p) + date_items(p) + voice_items(p)
    items += await instinct_items(session, str(data.get("family_id") or ""), people)
    return items


async def permissions_check(session: AsyncSession, family_id: str, perms: dict | None) -> str:
    """SaheliPermissions against boundary:policy (already the truth): equal | differs | none. Nothing is copied."""
    if not perms:
        return "none"
    from app.care import boundaries

    policy = boundaries._clean(await boundaries.get(session, family_id))
    limit = perms.get("spendSoftLimitInr")
    allowed_all = all(perms.get(k, True) for k in ("groceries", "food", "medicines", "rides"))
    same = allowed_all and (limit in (None, 0) or int(limit) == policy["elder_order_limit"])
    return "equal" if same else "differs"


async def run_family(session: AsyncSession, data: dict, *, apply: bool) -> dict:
    """Sort, check and (apply) write one family's old memory. The caller owns the transaction: commit after apply.
    Returns counts only."""
    family_id = str(data.get("family_id") or "")
    people = {str(p.get("id")): p for p in data.get("people") or [] if p.get("id")}
    items = await gather(session, data)
    calls = 0
    by_person: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_person[it.subject_id].append(it)
    with versions.attribution(actor_id=ACTOR, source=SOURCE, reason=REASON):
        for sid, its in by_person.items():
            fresh = [i for i in its if not await _imported(session, family_id, i.ref)]
            for i in its:
                if i not in fresh:
                    i.outcome = "already_imported"
            calls += await sort_free(session, family_id, sid, fresh)
            schedules = await _schedule_names(people.get(sid) or {})
            notes: list[Item] = []
            for it in its:
                it.outcome = await decide(session, family_id, it, schedules=schedules)
                if it.outcome in ("note_line", "detail_to_note"):
                    notes.append(it)
                elif it.outcome == "write" and apply:
                    payload = await _write(session, family_id, it)
                    it.outcome = "written_" + ("pending" if payload["action"] == "pending" else
                                               "rejected" if payload["action"] == "rejected" else
                                               "kept_pinned" if payload["action"] == "kept_pinned" else "fact")
                    await _ledger(session, family_id, it, payload)
            if notes and apply:
                vids = await _write_notes(session, family_id, sid, notes)
                for it in notes:
                    await _ledger(session, family_id, it, {"action": "note_line", "slug": it.note, "version": vids.get(it.note)})
                    it.outcome = "written_note" if it.outcome == "note_line" else "written_detail_note"
    report = summarise(items)
    report["model_calls"] = calls
    report["permissions"] = await permissions_check(session, family_id, data.get("permissions"))
    report["people"] = len(by_person)
    return report


def summarise(items: list[Item]) -> dict:
    """Counts by store family and outcome; never text."""
    by: dict[str, Counter] = defaultdict(Counter)
    kinds: Counter = Counter()
    for it in items:
        by[it.family][it.outcome or "unsorted"] += 1
        if it.family == "profile_fact":
            kinds[f"profile_fact:{it.old_status}"] += 1
    return {"items": len(items), "by_store": {k: dict(v) for k, v in sorted(by.items())}, "profile_facts_by_status": dict(kinds)}


# ── undo ─────────────────────────────────────────────────────────────────────


async def undo_family(session: AsyncSession, family_id: str, *, store_name: str | None = None) -> dict:
    """Take back every import into this family (or one store's), newest first, through memory versions. A fact changed
    since the import is left as it is (a person or a later write owns it now). The ledger entries are marked undone, so
    the move can run again."""
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == EVENT)
    if store_name:
        q = q.where(CareEvent.ref.startswith(f"import:{store_name}:"))
    rows = list((await session.execute(q.order_by(CareEvent.id.desc()))).scalars())
    counts: Counter = Counter()
    notes_done: set[int] = set()
    with versions.attribution(actor_id="kavach", source="import", reason="undo: moved from the older memory"):
        for ev in rows:
            p = ev.payload or {}
            if p.get("action") == "note_line":
                vid = p.get("version")
                if vid and vid not in notes_done:
                    notes_done.add(vid)
                    v = await versions.get(session, family_id, int(vid))
                    if v:
                        await versions.undo_note(session, v, reason="undo: moved from the older memory")
                counts["note_lines"] += 1
            elif p.get("fact_id"):
                counts[await _undo_fact(session, uuid.UUID(p["fact_id"]), p.get("version"))] += 1
            ev.kind, ev.ref = UNDONE, f"{ev.ref}#undone:{ev.id}"[:160]
    await session.flush()
    return {"undone": len(rows), **dict(counts)}


async def _undo_fact(session: AsyncSession, fact_id: uuid.UUID, version_id: int | None) -> str:
    row = await session.get(CareFact, fact_id, with_for_update=True)
    if row is None:
        return "gone"
    now = clock.now()
    with versions.attribution(undoes=version_id):
        if row.status == "active":
            old = await session.get(CareFact, row.supersedes) if row.supersedes else None
            row.status, row.valid_to, row.note = "stopped", now, "undone: moved from the older memory"
            await session.flush()
            await versions.record_fact(session, row, "stop")
            store._person_changed(session, row)
            if old is not None and old.status == "superseded":
                old.status, old.valid_to = "active", None
                await session.flush()
                await versions.record_fact(session, old, "write")
                return "fact_restored"
            return "fact_removed"
        if row.status == "pending":
            row.status, row.valid_to, row.note = "stopped", now, "undone: moved from the older memory"
            await session.execute(update(OpenLoop).where(OpenLoop.family_id == row.family_id, OpenLoop.kind == "confirm_fact",
                                                         OpenLoop.status == "open", OpenLoop.detail["fact_id"].astext == str(row.id))
                                  .values(status="cancelled", updated_at=now, closed_note="undone import"))
            await session.flush()
            await versions.record_fact(session, row, "stop")
            return "pending_removed"
        if row.status == "stopped" and (row.note or "").startswith(REJECTED_NOTE) and (row.source_ref or "").startswith("import:"):
            row.note = "undone: was on the never-use list from the older memory"
            await session.flush()
            await versions.record_fact(session, row, "stop")
            return "rejected_removed"
    return "changed_since"


# ── the job ──────────────────────────────────────────────────────────────────


async def families(host) -> list[dict]:
    """Every family the backend holds old memory for: [{family_id, actor_id}]."""
    out = await host.call("export_legacy_memory", {"list": True}, family_id="-", subject_id="-", actor_id="-")
    return [f for f in out.get("families") or [] if f.get("family_id") and f.get("actor_id")]


async def export(host, fam: dict) -> dict:
    return await host.call("export_legacy_memory", {}, family_id=fam["family_id"], subject_id=fam["actor_id"], actor_id=fam["actor_id"])


def label(family_id: str) -> str:
    return family_id[:8]


async def run(sessions, host, *, apply: bool, family: str = "", undo: bool = False, store_name: str | None = None) -> dict:
    """Every family (or those whose id starts with `family`), one transaction each. Counts only."""
    out: dict = {}
    totals: Counter = Counter()
    for fam in await families(host):
        fid = fam["family_id"]
        if family and not fid.startswith(family):
            continue
        try:
            async with sessions() as session:
                if undo:
                    rep = await undo_family(session, fid, store_name=store_name)
                    if apply:
                        await session.commit()
                    else:
                        await session.rollback()
                else:
                    rep = await run_family(session, await export(host, fam), apply=apply)
                    if apply:
                        await session.commit()
                    else:
                        await session.rollback()
            out[label(fid)] = rep
            for fam_counts in (rep.get("by_store") or {}).values():
                for k, n in fam_counts.items():
                    totals[k] += n
        except Exception as exc:  # noqa: BLE001 — one family's failure never stops the others
            logger.exception("memory unify failed family=%s", label(fid))
            out[label(fid)] = {"error": type(exc).__name__}
    return {"applied": apply, "undo": undo, "families": out, "totals": dict(totals)}
