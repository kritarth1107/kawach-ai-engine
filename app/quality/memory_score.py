"""A memory score every night, per family (trust plan A2).

For each person with a care record, the 5 truths (code, from the record):
  1. the name to use and the names never to use (naming:address_as)
  2. active medicines: name, dose, times
  3. allergies (the list, or "none" recorded)
  4. diet rules
  5. the latest correction in the last 60 days (app.quality.corrections), with its new value
A truth missing from the record is "unknown": counted apart and turned into one gentle question (the memory_check open
loop a caregiver answers), never a recall failure. No correction in 60 days is simply "none".

Three parts, each 0-100:
- stored (code): each truth saved once and active, no duplicate (the nightly merge's groups), no conflicting pending fact
  older than 14 days, the latest correction pinned (or on the "never say" list), and present in the context the brain
  really gets for a turn (loop.family_block + loop.turn_context), which catches retrieval and ranking bugs;
- recalled (model): the production brain model gets exactly that context and answers 5 structured questions; code
  compares the answers field by field. Nothing is sent, nothing is written, no tools run (founder D1: this runs for
  every family, his own included);
- used (real conversations): every real Saheli message of the day to or about the person: no avoided name, medicine
  times and doses as on record, no rejected fact (code on structured values), plus the full-memory judge's verdict for
  paraphrases (wrong name/medicine/time, correction ignored). A day with no messages is "no data", not 100.
The person's score is the minimum of the parts with data; the family's is the minimum across its people. Every failure
becomes a replay case. Runs in the 02:00 dream after the duplicate merge, before the brief.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain import guards
from app.care import naming as care_naming
from app.care import rejected as care_rejected
from app.care import store
from app.care.domains import is_none_key
from app.care.models import CareFact, FamilyRoster, OpenLoop
from app.core import clock
from app.learn.models import ReplyLog
from app.quality import replay
from app.quality.corrections import Correction
from app.quality.corrections import latest as latest_correction
from app.quality.models import MemoryScore, ReplyVerdict

logger = logging.getLogger(__name__)

TRUTHS = ("name", "medicines", "allergies", "diet", "correction")
RECORD_DOMAINS = ("medicine", "allergy", "condition", "diet")
CORRECTION_DAYS = 60
STALE_PENDING = timedelta(days=14)
USED_ISSUES = ("wrong_name", "wrong_medicine", "wrong_time", "correction_ignored")
ASK_ORDER = ("allergies", "medicines", "diet", "name")
ASK = {
    "allergies": "Does {name} have any allergies (food or medicine)? If none, I'll note 'no allergies'.",
    "medicines": "Which medicines does {name} take regularly, and at what times? If none, I'll note that.",
    "diet": "Does {name} follow any diet rules (for example less salt, no sugar, vegetarian)? If not, that's fine too.",
    "name": "What would {name} like me to call them, and is there any name I should never use?",
}

ContextFn = Callable[[AsyncSession, FamilyRoster, str, str], Awaitable[tuple[list[str], str, list[dict]]]]


@dataclass
class Truths:
    subject_id: str
    name: str
    naming: dict | None = None  # {"name", "avoid", "key", "text"}
    medicines: list[dict] | None = None  # None = unknown, [] = "none" recorded
    allergies: list[dict] | None = None
    diet: list[dict] | None = None
    correction: Correction | None = None  # None = no correction in 60 days
    pending: list[CareFact] = field(default_factory=list)
    rejected: list[CareFact] = field(default_factory=list)

    @property
    def unknowns(self) -> list[str]:
        out = []
        if self.naming is None:
            out.append("name")
        for t in ("medicines", "allergies", "diet"):
            if getattr(self, t) is None:
                out.append(t)
        return out

    def as_dict(self) -> dict:
        return {"name": self.naming, "medicines": self.medicines, "allergies": self.allergies, "diet": self.diet,
                "correction": self.correction.as_dict() if self.correction else None}


def _med(f: CareFact) -> dict:
    v = f.value or {}
    times = [t for t in (v.get("times") or []) if isinstance(t, str)]
    return {"key": f.key, "name": str(v.get("name") or f.key.split(":", 1)[-1]), "dose": str(v.get("dose") or ""), "times": times, "text": f.text}


async def truths(session: AsyncSession, family_id: str, subject_id: str, name: str, now: datetime) -> Truths:
    rows = await store.facts(session, family_id, subject_id, statuses=("active", "pending"))
    active = [f for f in rows if f.status == "active"]
    t = Truths(subject_id=subject_id, name=name, pending=[f for f in rows if f.status == "pending"])
    nm = next((f for f in active if f.key == "naming:address_as"), None)
    if nm is not None:
        v = nm.value or {}
        avoid = [w for w in (v.get("avoid") or []) if isinstance(w, str) and w.strip()]
        raw = v.get("name")
        call = raw.strip() if isinstance(raw, str) and raw.strip() else None
        if call or avoid:
            t.naming = {"name": call, "avoid": avoid, "key": nm.key, "text": nm.text}
    for attr, domain in (("medicines", "medicine"), ("allergies", "allergy"), ("diet", "diet")):
        fs = [f for f in active if f.domain == domain]
        if not fs:
            continue
        real = [f for f in fs if not is_none_key(f.key)]
        if domain == "medicine":
            setattr(t, attr, [_med(f) for f in real])
        elif domain == "allergy":
            setattr(t, attr, [{"key": f.key, "allergen": str((f.value or {}).get("allergen") or f.key.split(":", 1)[-1]), "text": f.text} for f in real])
        else:
            setattr(t, attr, [{"key": f.key, "text": f.text, "value": f.value or {}} for f in real])
    t.correction = await latest_correction(session, family_id, subject_id, now - timedelta(days=CORRECTION_DAYS), now + timedelta(seconds=1))
    t.rejected = await care_rejected.rejected(session, family_id, subject_id, now=now)
    return t


# ── the context a real turn gets ─────────────────────────────────────────────

RECALL_QUESTION = (
    "[Memory check from the Kavach team. This is not a person and nothing you write is sent to anyone. Do not call tools.]\n"
    "Answer only from what you know right now about {name} (id {sid}). Return ONLY JSON, in English, copying names, "
    "doses and times exactly as you know them:\n"
    '{{"name_to_use": "the name you call them, or null if you do not know", "never_call": ["names you must never use"], '
    '"medicines": [{{"name": "", "dose": "", "times": ["HH:MM"]}}], "allergies": ["each allergy", or "none" if they have none, '
    'or [] if you do not know], "diet": ["each diet rule"], "latest_change": {{"what": "{changed}", "new": "what is true now '
    'about it (stopped, or not true, if so)"}} or null if you do not know}}'
)


def recall_question(t: "Truths") -> str:
    """The 5 questions for this person. Truth 5 names the item the family changed (code knows it from memory history): the
    brain's context carries no dates, so "the most recent change" was a guess about recency (13 Oct: 3 of 4 people failed it
    while their records held the new values); what must hold is that she knows the new value of that item."""
    c = t.correction
    changed = (f"{c.key.split(':', 1)[0]}: {c.key.split(':', 1)[-1].replace('_', ' ')}" if c else "the most recent thing the family changed")
    return RECALL_QUESTION.format(name=t.name or "this person", sid=t.subject_id, changed=changed.replace('"', "'"))


def req_for(roster: FamilyRoster, subject_id: str, text: str):
    from app.brain.loop import TurnRequest

    elder = dict(roster.elder or {})
    members = [dict(m) for m in roster.members or [] if isinstance(m, dict)]
    if subject_id == elder.get("id"):
        speaker = {"id": elder["id"], "name": elder.get("name"), "role": "elder"}
    else:
        m = next((x for x in members if x.get("id") == subject_id), {"id": subject_id})
        speaker = {"id": subject_id, "name": m.get("name"), "role": m.get("role") or "caregiver"}
    return TurnRequest(family_id=roster.family_id, elder=elder, speaker=speaker, members=members, text=text, message_ref=None)


async def real_context(session: AsyncSession, roster: FamilyRoster, subject_id: str, text: str) -> tuple[list[str], str, list[dict]]:
    """(system_stable, system_dynamic, history) exactly as loop._run_turn builds them for this person speaking now."""
    from app.brain import loop
    from app.brain.persona import PERSONA, REPLY_FORMAT
    from app.learn import playbook as learned
    from app.learn import situations

    req = req_for(roster, subject_id, text)
    data = loop.TurnData(session, req)
    fam_block, _known, _avoid = await loop.family_block(session, req, data)
    dynamic, _ = await loop.turn_context(session, req, data)
    msgs = await loop.history(session, req, data)
    profiles = await loop.writing_profiles(session, req.family_id, [req.elder, *[m for m in req.members if m.get("id") != req.elder.get("id")]], data)
    dynamic += "\n\n" + loop.writing_block(req, profiles)
    role = "elder" if req.speaker["id"] == req.elder.get("id") else "caregiver"
    pb, _arm = await learned.for_family(session, req.family_id)
    pb_block = learned.block(pb, situations.tag(text=text, role=role, tools=[]), situations.lang_of(profiles.get(req.speaker["id"])))
    if pb_block:
        dynamic += "\n\n" + pb_block
    return [PERSONA + "\n\n" + REPLY_FORMAT, fam_block], dynamic, msgs


# ── stored ───────────────────────────────────────────────────────────────────


def _line(f_key: str, text: str) -> str:
    return f"[{f_key}] {text}"


def stored_checks(t: Truths, context: str, groups: list[dict]) -> list[dict]:
    """[{check, ok, why}] for the known truths."""
    out: list[dict] = []
    dup_keys = {k for g in groups for k in [g.get("keep"), *(g.get("drop") or [])] if k}
    stale = {f.key for f in t.pending if f.recorded_at < clock.now() - STALE_PENDING}

    def add(check: str, ok: bool, why: str = "") -> None:
        out.append({"check": check, "ok": bool(ok), "why": "" if ok else why})

    if t.naming:
        n = t.naming
        add("name.no_stale_pending", n["key"] not in stale, "a change to the name waits unconfirmed for more than 14 days")
        if n.get("name"):
            add("name.in_context", f"call them {n['name']}" in context, f"the name to use ({n['name']}) is not in the brain's context")
        for w in n.get("avoid") or []:
            add("name.avoid_in_context", w in context and "NEVER call them" in context, f"the never-use name '{w}' is not in the brain's context")
    for attr in ("medicines", "allergies", "diet"):
        items = getattr(t, attr)
        if not items:
            continue
        keys = [i["key"] for i in items]
        dups = [k for k in keys if k in dup_keys]
        if attr == "medicines":
            firsts: dict[str, list[str]] = {}
            for i in items:
                first = (guards.words(i["name"]) or [i["name"].lower()])[0]
                firsts.setdefault(first, []).append(i["key"])
            dups += [k for ks in firsts.values() if len(ks) > 1 for k in ks]
        add(f"{attr}.once", not dups, f"saved more than once: {', '.join(sorted(set(dups)))}")
        bad = [k for k in keys if k in stale]
        add(f"{attr}.no_stale_pending", not bad, f"a change waits unconfirmed for more than 14 days: {', '.join(bad)}")
        missing = [i["key"] for i in items if _line(i["key"], i["text"]) not in context]
        add(f"{attr}.in_context", not missing, f"not in the brain's context: {', '.join(missing)}")
    c = t.correction
    if c is not None:
        if c.kind in ("change", "naming"):
            add("correction.pinned", bool(c.pinned), f"the correction of {c.key} is not pinned")
            if c.kind == "change":
                add("correction.in_context", _line(c.key, c.new) in context or not c.still_active, f"the corrected {c.key} is not in the brain's context")
        elif c.kind == "reject":
            on_list = any(r.key == c.key for r in t.rejected)
            add("correction.on_never_list", on_list, f"the rejected {c.key} is not on the never-say list")
            if on_list:
                add("correction.in_context", c.old in context, f"the rejected {c.key} is not in the brain's never-say block")
        elif c.kind == "stop":
            add("correction.in_context", _line(c.key, c.old) not in context, f"the stopped {c.key} is still shown as in effect")
    return out


# ── recalled ─────────────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s or "").lower()).split())


def _has(hay: str, needle: str) -> bool:
    a, b = _norm(hay), _norm(needle)
    return bool(b) and (b in a or a in b) if a else False


def _times(xs) -> set[int]:
    return guards.times_in(" ".join(str(x) for x in (xs or [])))


def compare(t: Truths, ans: dict) -> list[dict]:
    """Field by field: [{field, ok, expected, got}] for the known truths."""
    out: list[dict] = []

    def add(fld: str, ok: bool | None, expected, got) -> None:  # None: free text, settled by the equivalence check
        out.append({"field": fld, "ok": None if ok is None else bool(ok), "expected": expected, "got": got})

    if t.naming and t.naming.get("name"):
        add("name_to_use", _has(str(ans.get("name_to_use") or ""), t.naming["name"]), t.naming["name"], ans.get("name_to_use"))
    if t.naming and t.naming.get("avoid"):
        never = [str(x) for x in ans.get("never_call") or []]
        add("never_call", all(any(_has(n, w) for n in never) for w in t.naming["avoid"]), t.naming["avoid"], never)
    if t.medicines is not None:
        got = [m for m in ans.get("medicines") or [] if isinstance(m, dict)]
        ok = len(got) == len(t.medicines)
        for m in t.medicines:
            first = (guards.words(m["name"]) or [m["name"]])[0]
            match = next((g for g in got if _has(str(g.get("name") or ""), first)), None)
            if match is None:
                ok = False
                continue
            if m["times"] and _times(m["times"]) != _times(match.get("times")):
                ok = False
            want = guards.numbers_in(m["dose"])
            if want and not want <= guards.numbers_in(str(match.get("dose") or "")):
                ok = False
        add("medicines", ok, [{k: m[k] for k in ("name", "dose", "times")} for m in t.medicines], got)
    if t.allergies is not None:
        got = [str(a) for a in ans.get("allergies") or [] if str(a).strip()]
        if not t.allergies:
            ok = bool(got) and all(_norm(a) in ("none", "no allergies", "no known allergies", "nil") for a in got)
        else:
            ok = all(any(_has(g, a["allergen"]) for g in got) for a in t.allergies) and len(got) == len(t.allergies)
        add("allergies", ok, [a["allergen"] for a in t.allergies] or ["none"], got)
    if t.diet:
        got = [str(d) for d in ans.get("diet") or []]
        add("diet", None if got else False, [d["text"] for d in t.diet], got)
    if t.correction is not None:
        lc = ans.get("latest_change") if isinstance(ans.get("latest_change"), dict) else None
        add("latest_change", None if lc else False, t.correction.as_dict(), lc)
    return out


EQUIV_PROMPT = """You compare a care companion's answer with the care record, field by field. For each item, decide whether
GOT says the same thing as EXPECTED (other words or more detail are fine; anything missing, wrong or extra that matters
is not). For latest_change, GOT must name the same change and the value true now. Return ONLY JSON:
{"items": [{"field": "...", "same": true, "why": "short"}]}"""


async def settle(items: list[dict]) -> None:
    """Fields code could not decide (free text: diet rules, the latest change): one model equivalence check."""
    from app.llm import router

    todo = [i for i in items if i["ok"] is None]
    if not todo:
        return
    text = json.dumps([{"field": i["field"], "EXPECTED": i["expected"], "GOT": i["got"]} for i in todo], ensure_ascii=False, default=str)
    try:
        rep = await router.complete("judge", system_stable=EQUIV_PROMPT, messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                                    max_tokens=6000, effort="low", timeout_s=120, essential=False)
        m = re.search(r"\{.*\}", rep.text or "", re.S)
        got = {x.get("field"): x for x in (json.loads(m.group(0)).get("items") or [])} if m else {}
    except Exception as exc:  # noqa: BLE001 — undecided fields count as not recalled
        logger.warning("recall equivalence failed: %s", str(exc)[:160])
        got = {}
    for i in todo:
        i["ok"] = bool((got.get(i["field"]) or {}).get("same"))


async def recall(session: AsyncSession, roster: FamilyRoster, t: Truths, context_fn: ContextFn) -> tuple[list[dict], dict, str]:
    """(field results, the model's answer, the question). The brain model, exactly the turn's context, no tools."""
    from app.brain.loop import BRAIN_EFFORT
    from app.llm import router

    question = recall_question(t)
    stable, dynamic, msgs = await context_fn(session, roster, t.subject_id, question)
    msgs = [*msgs, {"role": "user", "content": [{"type": "text", "text": question}]}]
    rep = await router.complete("brain", system_stable=stable, system_dynamic=dynamic, messages=msgs, tools=None, max_tokens=6000,
                                effort=BRAIN_EFFORT, timeout_s=90, essential=False)
    m = re.search(r"\{.*\}", rep.text or "", re.S)
    try:
        ans = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        ans = {}
    items = compare(t, ans if isinstance(ans, dict) else {})
    await settle(items)
    return items, ans if isinstance(ans, dict) else {"raw": (rep.text or "")[:2000]}, question


# ── used ─────────────────────────────────────────────────────────────────────

DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|units?|iu|g)\b", re.I)


def dose_problems(text: str, meds: list[dict]) -> list[str]:
    """A dose written right after a medicine of the record that is not its dose (her own text only)."""
    out = []
    low = guards.ascii_digits(text or "").lower()
    for m in meds or []:
        want = guards.numbers_in(m.get("dose") or "")
        first = (guards.words(m["name"]) or [""])[0]
        if not want or len(first) < 4:
            continue
        for hit in re.finditer(rf"\b{re.escape(first)}", low):
            d = DOSE.search(low, hit.end(), hit.end() + 25)
            if d and (d.group(1).lstrip("0") or "0") not in want:
                out.append(f"gives {m['name']} as {d.group(0)}, but the record says {m['dose']}")
    return out


def used_problems(text: str, said: str, *, to_subject: bool, t: Truths) -> list[str]:
    out: list[str] = []
    if to_subject and t.naming and t.naming.get("avoid"):
        out += [f"used the never-use name '{w}'" for w in care_naming.uses_avoided(text, t.naming["avoid"])]
    meds = t.medicines or []
    if meds:
        record = guards.med_times([(m["name"], m["text"] + " " + " ".join(m["times"])) for m in meds])
        out += [p for p in guards.ungrounded(text, known="", meds=record) if p.startswith("gives ")]
        out += dose_problems(text, meds)
    for term in care_rejected.terms(t.rejected):
        if care_naming.uses_avoided(text, [term]) and not care_naming.uses_avoided(said or "", [term]):
            out.append(f"used '{term}', which the family said is wrong")
    return sorted(set(out))


async def used(session: AsyncSession, family_id: str, t: Truths, roster: FamilyRoster, start: datetime, end: datetime) -> tuple[float | None, int, list[dict]]:
    """(0-100 or None for no data, messages checked, failures)."""
    elder_id = (roster.elder or {}).get("id")
    q = select(ReplyLog).where(ReplyLog.family_id == family_id, ReplyLog.at >= start, ReplyLog.at < end)
    if t.subject_id != elder_id:
        q = q.where(ReplyLog.thread_id == t.subject_id)  # a self-care person's own messages; the elder is talked about everywhere
    rows = list((await session.execute(q.order_by(ReplyLog.at))).scalars())
    if not rows:
        return None, 0, []
    verdicts = {v.reply_id: v for v in (await session.execute(select(ReplyVerdict).where(ReplyVerdict.reply_id.in_([r.id for r in rows])))).scalars()}
    fails: list[dict] = []
    bad = 0
    for r in rows:
        to_subject = r.thread_id == t.subject_id
        probs = used_problems(r.text, r.user_text, to_subject=to_subject, t=t)
        v = verdicts.get(r.id)
        if v is not None and (to_subject or t.subject_id == elder_id):
            for name in USED_ISSUES:
                if name in (v.issues or []):
                    words = "; ".join(((v.verdict or {}).get("issues") or {}).get(name, {}).get("words") or [])
                    probs.append(f"judge: {name}" + (f" ({words[:120]})" if words else ""))
        if probs:
            bad += 1
            fails.append({"part": "used", "reply": r.id, "to": r.thread_id, "problems": probs[:5], "words": r.text[:300]})
    return round(100.0 * (len(rows) - bad) / len(rows), 1), len(rows), fails


# ── one person, one family ───────────────────────────────────────────────────


def _pct(items: list[dict], key: str = "ok") -> float | None:
    if not items:
        return None
    return round(100.0 * sum(1 for i in items if i[key]) / len(items), 1)


def people_scored(roster: FamilyRoster, active_subjects: set[str]) -> list[dict]:
    """The care recipient, and anyone else with their own care record (self care)."""
    elder = dict(roster.elder or {})
    out = [elder] if elder.get("id") else []
    for m in roster.members or []:
        if isinstance(m, dict) and m.get("id") and m.get("id") != elder.get("id") and m["id"] in active_subjects:
            out.append(dict(m))
    return out


async def _subjects_with_record(session: AsyncSession, family_id: str) -> set[str]:
    return {sid for (sid,) in (await session.execute(
        select(CareFact.subject_id).where(CareFact.family_id == family_id, CareFact.status == "active", CareFact.domain.in_(RECORD_DOMAINS)).distinct()
    )).all()}


async def ask_unknown(session: AsyncSession, family_id: str, roster: FamilyRoster, t: Truths) -> str | None:
    """One gentle question for the first unknown truth never asked before (memory_check loop to whoever can answer)."""
    if not t.unknowns:
        return None
    live = (await session.execute(select(func.count()).select_from(OpenLoop).where(
        OpenLoop.family_id == family_id, OpenLoop.subject_id == t.subject_id, OpenLoop.kind == "memory_check", OpenLoop.status == "open"))).scalar_one()
    if live:
        return None  # one question at a time per person: never a pile of questions
    owner = next((m.get("id") for m in roster.members or [] if isinstance(m, dict) and "primary" in str(m.get("role", "")).lower()), None)
    owner = owner if owner and owner != t.subject_id else t.subject_id
    for truth in ASK_ORDER:
        if truth not in t.unknowns:
            continue
        key = f"memscore:unknown:{t.subject_id}:{truth}"
        if (await session.execute(select(func.count()).select_from(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.dedupe_key == key))).scalar_one():
            continue  # asked once already, ever
        first = (t.name or "them").split()[0] if owner != t.subject_id else "you"
        title = ASK[truth].format(name=first).replace("Does you have", "Do you have").replace("does you take", "do you take") \
            .replace("Does you follow", "Do you follow").replace("would you like me to call them", "would you like me to call you")
        await store.open_loop(session, family_id=family_id, subject_id=t.subject_id, kind="memory_check", title=title,
                              detail={"problem": f"memory score: {truth} not on the record", "kind": "unknown_truth", "truth": truth, "max_wakes": 1},
                              owner_id=owner, wake_at=clock.ist().replace(hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1),
                              alert_rule="dashboard", dedupe_key=key)
        return truth
    return None


async def score_person(session: AsyncSession, roster: FamilyRoster, person: dict, day: str, *, groups: list[dict], context_fn: ContextFn | None = None,
                       with_models: bool = True, ask: bool = True) -> dict:
    fid = roster.family_id
    sid = person["id"]
    now = clock.now()
    start = datetime.fromisoformat(day).replace(tzinfo=clock.IST)
    end = start + timedelta(days=1)
    t = await truths(session, fid, sid, person.get("name") or "", now)
    if t.correction is not None:
        row = await store.active_fact(session, fid, sid, t.correction.key)
        t.correction.pinned = bool(row is not None and row.pinned_at is not None)
        t.correction.still_active = bool(row is not None and row.text == t.correction.new)
    context_fn = context_fn or real_context
    stable, dynamic, _msgs = await context_fn(session, roster, sid, recall_question(t))
    context = "\n".join(stable[1:]) + "\n" + dynamic
    checks = stored_checks(t, context, groups)
    failures: list[dict] = [{"part": "stored", **c} for c in checks if not c["ok"]]
    recalled = None
    answer: dict = {}
    question = ""
    items: list[dict] = []
    if with_models and (t.naming or t.medicines is not None or t.allergies is not None or t.diet or t.correction):
        try:
            items, answer, question = await recall(session, roster, t, context_fn)
            recalled = _pct(items)
            failures += [{"part": "recalled", **i} for i in items if not i["ok"]]
        except Exception as exc:  # noqa: BLE001 — no answer is a failed recall, not a skipped one
            logger.warning("recall check failed family=%s person=%s: %s", fid, sid, str(exc)[:160])
            recalled = 0.0
            failures.append({"part": "recalled", "field": "all", "ok": False, "expected": "an answer", "got": f"error: {str(exc)[:120]}"})
    used_pct, used_n, used_fails = await used(session, fid, t, roster, start, end)
    failures += used_fails
    stored_pct = _pct(checks)
    parts = [p for p in (stored_pct, recalled, used_pct) if p is not None]
    asked = await ask_unknown(session, fid, roster, t) if ask else None
    out = {"subject": sid, "name": t.name, "stored": stored_pct, "recalled": recalled, "used": used_pct, "used_n": used_n,
           "score": min(parts) if parts else None, "unknowns": t.unknowns, "asked": asked, "failures": failures,
           "truths": t.as_dict(), "answer": answer, "question": question, "context": context}
    return out


async def save(session: AsyncSession, family_id: str, day: str, people: list[dict]) -> dict:
    """Store each person's row and the family row (the minimum across its people); returns the family row."""
    d = date.fromisoformat(day)

    def row(subject: str, stored, recalled, used_, score, unknowns, failures, detail):
        values = dict(family_id=family_id, subject_id=subject, day=d, stored=stored, recalled=recalled, used=used_, score=score, unknowns=unknowns,
                      failures=failures, detail=detail, computed_at=clock.now())
        stmt = insert(MemoryScore).values(**values).on_conflict_do_update(
            index_elements=[MemoryScore.family_id, MemoryScore.subject_id, MemoryScore.day],
            set_={k: v for k, v in values.items() if k not in ("family_id", "subject_id", "day")})
        return stmt

    for p in people:
        await session.execute(row(p["subject"], p["stored"], p["recalled"], p["used"], p["score"], p["unknowns"],
                                  [{k: v for k, v in f.items() if k != "words"} | ({"words": f["words"][:200]} if f.get("words") else {})
                                   for f in p["failures"]][:50],
                                  {"name": (p.get("name") or "").split(" ")[0], "used_n": p["used_n"], "asked": p["asked"], "truths": p["truths"]}))

    def low(key: str):
        vals = [p[key] for p in people if p[key] is not None]
        return min(vals) if vals else None

    fam = {"stored": low("stored"), "recalled": low("recalled"), "used": low("used"), "score": low("score"),
           "unknowns": sorted({f"{(p.get('name') or '').split(' ')[0] or p['subject'][:6]}: {u}" for p in people for u in p["unknowns"]}),
           "failures": [{"person": (p.get("name") or "").split(" ")[0] or p["subject"][:6], **{k: v for k, v in f.items() if k != "words"}}
                        for p in people for f in p["failures"]][:80]}
    await session.execute(row("family", fam["stored"], fam["recalled"], fam["used"], fam["score"], fam["unknowns"], fam["failures"],
                              {"people": len(people)}))
    return fam


async def replay_cases(session: AsyncSession, roster: FamilyRoster, day: str, person: dict) -> int:
    """Each failure of this person's score as a replay case (anonymised snapshot, bad reply, what must hold)."""
    from app.quality import snapshot

    fid = roster.family_id
    sid = person["subject"]
    n = 0
    now_snap: dict | None = None
    for f in person["failures"]:
        if f["part"] == "used":
            r = await session.get(ReplyLog, f["reply"])
            if r is None:
                continue
            snap = await snapshot.take(session, roster, r.thread_id, r.at, message=r.user_text, speaker_role=r.speaker_role, trace=r.trace, kind=r.kind)
            words = [w for p in f["problems"] for w in re.findall(r"'([^']+)'", p)]
            rules: list[dict] = [{"type": "not_contains", "words": list(dict.fromkeys(words))}] if words else []
            if any(p.startswith("gives ") for p in f["problems"]):
                rules += [{"type": "medicine_time", "medicine": m["name"], "times": m["times"], "dose": m["dose"]}
                          for m in (person["truths"].get("medicines") or [])]
            ok = await replay.add_snapshot(session, source="memory_score", source_ref=f"memory:used:{r.id}", kind="used_" + _kind(f["problems"]),
                                           day=day, snap=snap, message=r.user_text, bad_reply=r.text, assertion=rules or [{"type": "grounded"}],
                                           rubric="; ".join(f["problems"])[:1000], family_id=fid, subject_id=sid, thread_id=r.thread_id)
        else:
            if now_snap is None:
                now_snap = await snapshot.take(session, roster, sid, clock.now(), message=person["question"], speaker_role="system", kind="memory_check")
            if f["part"] == "recalled":
                ok = await replay.add_snapshot(session, source="memory_score", source_ref=f"memory:recalled:{fid}:{sid}:{day}:{f['field']}",
                                               kind=f"recall_{f['field']}", day=day, snap=now_snap, message=person["question"],
                                               bad_reply=json.dumps(person["answer"], ensure_ascii=False, default=str)[:4000],
                                               assertion=[{"type": "recall", "field": f["field"], "expect": f["expected"]}],
                                               rubric=f"The answer for {f['field']} must match the care record: "
                                                      f"{json.dumps(f['expected'], ensure_ascii=False, default=str)[:300]}",
                                               family_id=fid, subject_id=sid)
            else:
                ok = await replay.add_snapshot(session, source="memory_score", source_ref=f"memory:stored:{fid}:{sid}:{day}:{f['check']}",
                                               kind=f"stored_{f['check']}"[:40], day=day, snap=now_snap, message="", bad_reply="",
                                               assertion=[{"type": "context_contains", "check": f["check"], "why": f["why"]}], rubric=f["why"],
                                               family_id=fid, subject_id=sid)
        n += 1 if ok else 0
    return n


def _kind(problems: list[str]) -> str:
    p = " ".join(problems)
    if "never-use name" in p or "wrong_name" in p:
        return "name"
    if "family said is wrong" in p or "correction_ignored" in p:
        return "rejected"
    return "medicine"


async def score_family(session: AsyncSession, roster: FamilyRoster, day: str, *, groups: dict[str, list[dict]] | None = None,
                       context_fn: ContextFn | None = None, with_models: bool = True, ask: bool = True, cases: bool = True) -> dict:
    """Score every person with a care record, store the rows, open unknown questions, file replay cases."""
    subjects = await _subjects_with_record(session, roster.family_id)
    people = []
    for p in people_scored(roster, subjects):
        people.append(await score_person(session, roster, p, day, groups=(groups or {}).get(p["id"], []), context_fn=context_fn,
                                         with_models=with_models, ask=ask))
    if not people:
        return {"people": 0}
    fam = await save(session, roster.family_id, day, people)
    filed = 0
    if cases:
        for p in people:
            filed += await replay_cases(session, roster, day, p)
    return {"people": len(people), "score": fam["score"], "stored": fam["stored"], "recalled": fam["recalled"], "used": fam["used"],
            "unknowns": len(fam["unknowns"]), "failures": len(fam["failures"]), "cases": filed, "per_person": people}
