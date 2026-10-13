"""One fact, one entry (plan D3).

Kavach NUdge found it on 2026-10-13: one person's single allergy (cucumber and roasted onion) was saved four times under
four keys (the brain saved the pair and each half, the re-read saved it again in other words); a home address twice;
wake, sleep and the gym both as routine facts and as the activities made from them. Every turn and every nudge carried
all of them.

Keys come from the words used, so the same thing said another way lands on another key. Two guards, both the model's
judgement over the record, never word lists:
- at write time (`check`): a fact about to start a NEW key in a domain that already has facts is compared with them.
  Already there → nothing is written. The same thing with more or changed detail → it goes to the existing key, with
  one sentence that keeps every detail of both. Something else → a new key as before.
- nightly (`merge_family`): per person and domain, the model groups duplicates already saved; each group becomes one
  fact (the fullest sentence, every detail, the strongest source) and the others stop with a version that undo brings
  back. A routine fact that an activity already says (same thing, same time) is marked `as_activity` and no longer
  listed twice; its times stay for the code that reads them (check-ins, the nightly brief). Medicines are only ever
  reported, never merged by code: a duplicate medicine is a reminder question for a person.

FACT_DEDUPE=off turns the write-time check off (tests that script every model reply). FACT_MERGE=dryrun|on|off: the
nightly merge reports (dryrun, the default until the founder approves) or applies.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import SOURCE_RANK, is_none_key, merged_value
from app.care.models import CareFact

logger = logging.getLogger(__name__)

# Domains where the same thing said twice is a duplicate. Not here: medicine (keys are matched by the medicine itself,
# and a merge would change reminders), activity (activities.adopt has its own check), routine (one fact per time of
# day), language/naming (one key each), appointment and occasion (two of them are usually two events).
CHECK_DOMAINS = ("allergy", "condition", "diet", "dish", "no_order", "preference", "family", "contact", "doctor",
                 "hospital", "home", "profile", "vital_target")
MERGE_DOMAINS = CHECK_DOMAINS + ("medicine",)
REPORT_ONLY = {"medicine"}
KEEP_KEYS = {"routine:daily_brief"}  # a setting, not a routine


def check_enabled() -> bool:
    return os.getenv("FACT_DEDUPE", "on").strip().lower() != "off"


def merge_mode() -> str:
    m = os.getenv("FACT_MERGE", "dryrun").strip().lower()
    return m if m in ("off", "dryrun", "on") else "dryrun"


def _line(f: CareFact) -> str:
    return f"- [{f.key}] {f.text} | {json.dumps(f.value or {}, ensure_ascii=False, sort_keys=True)[:300]}"


def _json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# ── at write time ────────────────────────────────────────────────────────────

SAME_PROMPT = """You keep a person's care record free of duplicates.

SAVED lists facts already in the record (one domain, one person). NEW is a fact about to be saved.
Decide whether NEW is about the same thing as one SAVED fact: the same allergy, condition, place, person, doctor, rule
or preference, even in other words, another language, or with more, less or changed detail. Two different things are
NOT the same (milk allergy vs peanut allergy; a home in Raipur vs a home in Patna; a son vs a daughter), and one saved
fact naming two things does not make each of them a duplicate of the other.

A SAVED line marked [rejected:…] is something the family said is WRONG: if NEW says that same thing, answer with that
marked key ("rejected:…").

Return only JSON: {"same_as": "<the saved key>" or null, "covered": true when the saved fact already says everything NEW
says, "sentence": "when same_as is set and not covered: one sentence for the record keeping every detail of both, newer
detail in place of what it corrects"}"""


@dataclass
class Same:
    fact: CareFact
    covered: bool
    text: str
    rejected: bool = False  # the family said this is wrong (app.care.rejected): an inference never writes it again


async def check(session: AsyncSession, *, family_id: str, subject_id: str, domain: str, key: str, text: str,
                value: dict | None) -> Same | None:
    """The saved fact that NEW repeats, or None. Never blocks a write: any failure means "new"."""
    if is_none_key(key):
        return None
    from app.care import rejected as care_rejected

    wrong = [f for f in await care_rejected.rejected(session, family_id, subject_id) if f.domain == domain]
    exact = next((f for f in wrong if f.key == key), None)
    if exact is not None and not await store.active_fact(session, family_id, subject_id, key):
        return Same(exact, True, exact.text, rejected=True)  # the very key the family rejected: any domain, no model
    if not check_enabled() or domain not in CHECK_DOMAINS:
        return None
    if await store.active_fact(session, family_id, subject_id, key):
        return None  # same key: write_fact merges it
    saved = [f for f in await store.facts(session, family_id, subject_id, domains=[domain], statuses=("active",))
             if not is_none_key(f.key)]
    if not saved and not wrong:
        return None
    new = f"[{key}] {text} | {json.dumps(value or {}, ensure_ascii=False, sort_keys=True)[:300]}"
    try:
        from app.llm import router

        reply = await router.complete("judge_fast", system_stable=SAME_PROMPT, effort="low", max_tokens=400, timeout_s=10,
                                      messages=[{"role": "user", "content": [{"type": "text", "text":
                                                 "SAVED\n" + ("\n".join(_line(f) for f in saved[:40]) or "- none")
                                                 + "".join(f"\n- [rejected:{f.key}] SAID TO BE WRONG: {f.text}" for f in wrong[:20])
                                                 + f"\n\nNEW\n{new}"}]}])
    except Exception:  # noqa: BLE001 — the fact is saved as new; the nightly merge catches a duplicate
        logger.warning("duplicate check failed family=%s domain=%s", family_id, domain)
        return None
    data = _json(reply.text or "")
    same_as = str(data.get("same_as") or "")
    if same_as.startswith("rejected:"):
        hit_wrong = next((f for f in wrong if f.key == same_as.removeprefix("rejected:")), None)
        return Same(hit_wrong, True, hit_wrong.text, rejected=True) if hit_wrong else None
    hit = next((f for f in saved if f.key == same_as), None)
    if hit is None:
        return None
    covered = data.get("covered") is True
    sentence = str(data.get("sentence") or "").strip()
    if not covered and not sentence:
        sentence = f"{hit.text.rstrip('.')}; {text}"  # keep both rather than lose a detail
    return Same(hit, covered, sentence[:500] if not covered else hit.text)


# ── nightly ──────────────────────────────────────────────────────────────────

GROUP_PROMPT = """You keep a person's care record free of duplicates. Below are the facts saved in one domain for one
person. Group the ones that are about the same thing (the same allergy, place, person, doctor, rule, preference or
medicine, even in other words, another language or with more or less detail). A group needs at least 2 facts. Different
things never go together (milk allergy vs peanut allergy; a home in Raipur vs a home in Patna). Facts that name the same
person or place but say different things about it are different facts too (who someone is vs whom to call first; an
address vs opening hours): group only facts where each one says nothing the others do not also mean.

For each group: keep = the key to keep (the clearest one), drop = the other keys, sentence = one sentence keeping every
detail from all of them (newer detail in place of what it corrects), details = the combined details object.
Return only JSON: {"groups": [{"keep": "...", "drop": ["..."], "sentence": "...", "details": {}}]}; an empty list when
nothing repeats."""

ROUTINE_PROMPT = """Below are a person's routine facts and their activities. List each routine fact that one activity
already says in full: the same thing at the same time (wake-up, a meal, sleep, the gym). A routine fact with a different
time, or with any part no single activity covers (a whole-day schedule with several items), is not listed.
Return only JSON: {"covered": [{"routine": "routine:...", "activity": "activity:..."}]}"""


@dataclass
class Group:
    domain: str
    keep: str
    drop: list[str]
    text: str
    details: dict = field(default_factory=dict)


async def _ask(prompt: str, text: str) -> dict:
    from app.llm import router

    reply = await router.complete("extract", system_stable=prompt, effort="low", max_tokens=2000, timeout_s=40,
                                  messages=[{"role": "user", "content": [{"type": "text", "text": text}]}])
    return _json(reply.text or "")


async def find(session: AsyncSession, family_id: str, subject_id: str) -> tuple[list[Group], list[tuple[str, str]]]:
    """The duplicate groups and the (routine, activity) pairs for one person; nothing is changed."""
    rows = [f for f in await store.facts(session, family_id, subject_id, statuses=("active",)) if not is_none_key(f.key)]
    by: dict[str, list[CareFact]] = {}
    for f in rows:
        by.setdefault(f.domain, []).append(f)
    groups: list[Group] = []
    for domain in MERGE_DOMAINS:
        facts = by.get(domain) or []
        if len(facts) < 2:
            continue
        keys = {f.key for f in facts}
        data = await _ask(GROUP_PROMPT, f"DOMAIN {domain}\n" + "\n".join(_line(f) for f in facts[:60]))
        used: set[str] = set()
        for g in data.get("groups") or []:
            if not isinstance(g, dict):
                continue
            keep = g.get("keep")
            drop = [k for k in (g.get("drop") or []) if isinstance(k, str) and k in keys and k != keep and k not in used]
            if keep not in keys or keep in used or not drop or not str(g.get("sentence") or "").strip():
                continue
            used |= {keep, *drop}
            groups.append(Group(domain, keep, drop, str(g["sentence"]).strip()[:500],
                                g["details"] if isinstance(g.get("details"), dict) else {}))
    pairs: list[tuple[str, str]] = []
    routine = [f for f in by.get("routine") or [] if f.key not in KEEP_KEYS and not (f.value or {}).get("as_activity")]
    acts = by.get("activity") or []
    if routine and acts:
        data = await _ask(ROUTINE_PROMPT, "ROUTINE\n" + "\n".join(_line(f) for f in routine) + "\n\nACTIVITIES\n"
                          + "\n".join(_line(f) for f in acts))
        rkeys, akeys = {f.key for f in routine}, {f.key for f in acts}
        for p in data.get("covered") or []:
            if isinstance(p, dict) and p.get("routine") in rkeys and p.get("activity") in akeys:
                pairs.append((p["routine"], p["activity"]))
    return groups, pairs


def _strongest(facts: list[CareFact]) -> CareFact:
    return max(facts, key=lambda f: (SOURCE_RANK.get(f.source_kind, 0), f.recorded_at))


async def apply(session: AsyncSession, family_id: str, subject_id: str, groups: list[Group], pairs: list[tuple[str, str]]) -> dict:
    """Merge each group into one fact and mark covered routine facts; every change is a version undo can reverse."""
    merged = stopped = marked = 0
    for g in groups:
        if g.domain in REPORT_ONLY:
            continue
        facts = [f for f in [await store.active_fact(session, family_id, subject_id, k) for k in [g.keep, *g.drop]] if f]
        keep = next((f for f in facts if f.key == g.keep), None)
        if keep is None or len(facts) < 2:
            continue
        pinned = [f for f in facts if f.pinned_at]
        if pinned and not keep.pinned_at:
            keep = max(pinned, key=lambda f: f.pinned_at or f.recorded_at)  # what a person said is the one kept (trust plan A3)
        top = _strongest(facts)
        value: dict = {}
        for f in sorted(facts, key=lambda x: x.recorded_at):
            value = merged_value(value, f.value)
        value = merged_value(value, g.details)
        await store.write_fact(session, family_id=family_id, subject_id=subject_id, domain=g.domain, key=keep.key, value=value,
                               text=g.text, source_kind=top.source_kind, stated_by=top.stated_by or top.confirmed_by,
                               note=f"merged: {', '.join(f.key for f in facts if f.key != keep.key)}"[:500], replace=True)
        merged += 1
        for f in facts:
            if f.key != keep.key:
                await store.stop_fact(session, family_id=family_id, subject_id=subject_id, key=f.key, source_kind=top.source_kind,
                                      reason=f"the same as {keep.key} (merged)", stated_by=top.stated_by)
                stopped += 1
    for rkey, akey in pairs:
        f = await store.active_fact(session, family_id, subject_id, rkey)
        if f is None or rkey in KEEP_KEYS:
            continue
        await store.write_fact(session, family_id=family_id, subject_id=subject_id, domain=f.domain, key=f.key,
                               value={**(f.value or {}), "as_activity": akey}, text=f.text, source_kind=f.source_kind,
                               stated_by=f.stated_by, note=f"shown as {akey}", replace=True)
        marked += 1
    return {"merged": merged, "stopped": stopped, "routine_marked": marked}


def describe(groups: list[Group], pairs: list[tuple[str, str]]) -> list[str]:
    out = [f"{g.domain}: keep {g.keep}, merge {', '.join(g.drop)} → \"{g.text}\"" + (" (report only)" if g.domain in REPORT_ONLY else "")
           for g in groups]
    out += [f"routine: {r} is already {a}" for r, a in pairs]
    return out


async def merge_family(session: AsyncSession, family_id: str, *, apply_changes: bool) -> dict:
    """Every person in the family: what repeats, and (apply_changes) the merge."""
    from sqlalchemy import select

    subjects = sorted({r[0] for r in await session.execute(
        select(CareFact.subject_id).where(CareFact.family_id == family_id, CareFact.status == "active").distinct())})
    out: dict = {}
    for sid in subjects:
        groups, pairs = await find(session, family_id, sid)
        if not groups and not pairs:
            continue
        entry: dict = {"found": describe(groups, pairs)}
        if apply_changes:
            from app.care import versions

            with versions.attribution(actor_id="saheli", source="nightly", reason="the same fact saved more than once"):
                entry["applied"] = await apply(session, family_id, sid, groups, pairs)
        out[sid] = entry
    return out
