"""Every failure becomes a replay case (trust plan A2.4, C1, for the pre-deploy runner of C3).

A case is self-contained: the context snapshot needed to run the turn again (memory as it was, the last turns, the
message, the household, settings, the time), the bad reply, and what must hold (structured assertions code can check,
plus one rubric line for the judge). The same failure never makes two cases (source_ref is unique).

Real families' cases live only in the production table replay_cases, anonymised (app.quality.snapshot: one consistent
mapping over the snapshot, the bad reply and the assertions), and are exported nightly to a private bucket in the test
project for the pre-deploy runner (app.quality.cases). Invented families' cases (the test world, founder bugs) may live in
the repo (eval/fixtures/replay_founder.jsonl).
"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.quality.models import ReplayCase

ASSERTION_TYPES = ("not_contains", "language", "medicine_time", "no_tool", "must_tool", "max_sentences", "grounded", "no_medical_advice",
                   "no_false_claim", "recall", "context_contains", "context_lacks")


async def add(session: AsyncSession, *, source: str, source_ref: str, kind: str, day: str, context: dict, message: str = "", bad_reply: str = "",
              assertion: list[dict] | None = None, rubric: str = "", family_id: str | None = None, subject_id: str | None = None,
              thread_id: str | None = None) -> bool:
    """Insert once; True when this is a new case."""
    rules = [a for a in (assertion or []) if isinstance(a, dict) and a.get("type") in ASSERTION_TYPES]
    stmt = insert(ReplayCase).values(
        source=source[:24], source_ref=source_ref[:200], kind=kind[:40], family_id=family_id, subject_id=subject_id, thread_id=thread_id,
        day=day, context=context, message=(message or "")[:4000], bad_reply=(bad_reply or "")[:4000], assertion=rules, rubric=(rubric or "")[:1000],
        status="open", anonymised=False, created_at=clock.now(),
    ).on_conflict_do_nothing(index_elements=[ReplayCase.source_ref]).returning(ReplayCase.id)
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def add_snapshot(session: AsyncSession, *, source: str, source_ref: str, kind: str, day: str, snap: dict, message: str, bad_reply: str,
                       assertion: list[dict], rubric: str = "", family_id: str | None = None, subject_id: str | None = None,
                       thread_id: str | None = None) -> bool:
    """A real family's case: stored anonymised (app.quality.snapshot), one consistent mapping over every field."""
    from app.quality.snapshot import anonymised_case

    a = anonymised_case(snap, message=message, bad_reply=bad_reply, assertion=assertion, rubric=rubric)
    rules = [r for r in (a["assertion"] or []) if isinstance(r, dict) and r.get("type") in ASSERTION_TYPES]
    stmt = insert(ReplayCase).values(
        source=source[:24], source_ref=source_ref[:200], kind=kind[:40], family_id=family_id, subject_id=subject_id, thread_id=thread_id,
        day=day, context=a["context"], message=(a["message"] or "")[:4000], bad_reply=(a["bad_reply"] or "")[:4000], assertion=rules,
        rubric=(a["rubric"] or "")[:1000], status="open", anonymised=True, created_at=clock.now(),
    ).on_conflict_do_nothing(index_elements=[ReplayCase.source_ref]).returning(ReplayCase.id)
    return (await session.execute(stmt)).scalar_one_or_none() is not None


def from_verdict(verdict: dict, *, avoid: list[str], script: str | None, language: str | None) -> tuple[list[dict], str]:
    """Assertions and the rubric line for a judge failure."""
    issues = verdict.get("issues") or {}
    rules: list[dict] = []
    found = [n for n, v in issues.items() if (v or {}).get("found")]
    words = [w for n in found for w in (issues[n].get("words") or [])]
    if avoid and ("wrong_name" in found or "correction_ignored" in found):
        rules.append({"type": "not_contains", "words": list(dict.fromkeys(avoid))})
    if "made_up_fact" in found:
        rules.append({"type": "grounded"})
        made = issues["made_up_fact"].get("words") or []
        if made:
            rules.append({"type": "not_contains", "words": made[:3]})
    if "own_medical_advice" in found:
        rules.append({"type": "no_medical_advice"})
    if "wrong_language" in found and (script or language):
        rules.append({"type": "language", "script": script, "language": language})
    if {"wrong_medicine", "wrong_time", "wrong_person"} & set(found):
        rules.append({"type": "grounded"})
    if "broken_promise" in found:
        rules.append({"type": "no_false_claim"})
    if "tone" in found and (issues["tone"].get("kind") == "too_long" or "long" in (issues["tone"].get("note") or "").lower()):
        rules.append({"type": "max_sentences", "n": 4})
    if "missed_alert" in found:
        rules.append({"type": "must_tool", "tools": ["alert_caregiver"]})
    if "unneeded_alert" in found:
        rules.append({"type": "no_tool", "tools": ["alert_caregiver"]})
    notes = "; ".join(f"{n}: {issues[n].get('note') or ''}".strip() for n in found)
    rubric = (f"Must not repeat: {verdict.get('summary') or ''} ({notes})" if found else "")[:1000]
    if words and not rules:
        rules.append({"type": "not_contains", "words": words[:3]})
    seen, out = set(), []
    for r in rules:
        k = repr(sorted(r.items()))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out, rubric
