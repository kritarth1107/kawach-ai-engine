"""Deterministic renderings of memory for the brain's context.

The care record digest changes only when a fact changes, so it sits in a cached prompt block.
Today's ledger and open loops change every turn and go into the turn context instead.
"""

from __future__ import annotations

from collections import defaultdict

from app.care.models import CareEvent, CareFact, MemoryNote, OpenLoop
from app.core import clock

SOURCE_TAG = {
    "prescription": "Rx",
    "lab": "lab",
    "caregiver_said": "caregiver",
    "dashboard": "dashboard",
    "import": "on file",
    "elder_said": "elder said",
    "inferred": "inferred",
}

ORDER = [
    "allergy",
    "medicine",
    "no_order",
    "condition",
    "diet",
    "vital_target",
    "dish",
    "naming",
    "language",
    "family",
    "routine",
    "occasion",
    "doctor",
    "hospital",
    "home",
    "contact",
    "preference",
    "profile",
    "appointment",
]

TITLES = {
    "allergy": "Allergies",
    "medicine": "Medicines",
    "no_order": "Never order",
    "condition": "Conditions",
    "diet": "Diet rules",
    "vital_target": "Targets",
    "dish": "Dishes they actually cook / eat",
    "naming": "Naming",
    "language": "Language",
    "family": "Family rules",
    "routine": "Routine",
    "occasion": "Occasions",
    "doctor": "Doctors",
    "hospital": "Hospital",
    "home": "Home",
    "contact": "Contacts",
    "preference": "Preferences",
    "profile": "Emergency profile",
    "appointment": "Appointments",
}


def care_record(name: str, rows: list[CareFact]) -> str:
    by_domain: dict[str, list[CareFact]] = defaultdict(list)
    for f in rows:
        by_domain[f.domain].append(f)
    lines = [f"CARE RECORD — {name}"]
    if not rows:
        lines.append("(Nothing saved yet. Do not assume any medicine, allergy, diet or dish.)")
    for domain in ORDER + sorted(set(by_domain) - set(ORDER)):
        items = by_domain.get(domain)
        if not items:
            continue
        lines.append(f"{TITLES.get(domain, domain)}:")
        for f in sorted(items, key=lambda x: (x.key, x.status != "active")):
            tag = SOURCE_TAG.get(f.source_kind, f.source_kind)
            if f.status == "pending":
                lines.append(f"  - [{f.key}] PENDING, not in effect until a caregiver confirms: {f.text} ({tag})")
            else:
                confirmed = "" if f.confirmed_by or f.source_kind != "elder_said" else ", unconfirmed"
                lines.append(f"  - [{f.key}] {f.text} ({tag}{confirmed})")
    missing = [TITLES[d] for d in ("allergy", "medicine", "diet") if d not in by_domain]
    if missing and rows:
        lines.append(f"Not on file: {', '.join(missing)}. Never guess these.")
    return "\n".join(lines)


def household(elder: dict, members: list[dict], speaker_id: str) -> str:
    lines = [f"HOUSEHOLD\n  Care recipient: {elder.get('name')} (id {elder.get('id')})"]
    for m in members:
        if m.get("id") == elder.get("id"):
            continue
        you = "  ← speaking now" if m.get("id") == speaker_id else ""
        lines.append(f"  {m.get('role', 'member')}: {m.get('name')} (id {m.get('id')}){you}")
    if speaker_id == elder.get("id"):
        lines.append(f"  {elder.get('name')} is speaking now.")
    lines.append("  You can message only the people listed here. Helpers, neighbours, doctors and anyone else are not reachable by you.")
    return "\n".join(lines)


RECALL_ONLY = {"diary", "weekly", "monthly"}  # long running logs: found by recall, not pasted into every turn
FIRST = {"profile-card": 0, "life-so-far": 1}


def notes_block(rows: list[MemoryNote], *, max_chars: int = 6000) -> str:
    rows = sorted((n for n in rows if n.slug not in RECALL_ONLY), key=lambda n: (FIRST.get(n.slug, 2), n.subject_id))
    if not rows:
        return ""
    out, used = ["PEOPLE AND LIFE (memory notes; the profile card is the summary of who they are)"], 0
    for n in rows:
        body = n.body_md.strip()
        if used + len(body) > max_chars:
            body = body[: max(0, max_chars - used)] + " …(use recall for more)"
        out.append(f"## {n.title} [{n.subject_id}/{n.slug}]\n{body}")
        used += len(body)
        if used >= max_chars:
            break
    return "\n".join(out)


def ledger(day_events: list[CareEvent]) -> str:
    if not day_events:
        return "TODAY SO FAR: nothing logged yet (no reminders sent, no doses marked)."
    lines = ["TODAY SO FAR (IST, from the ledger; this is what actually happened):"]
    for e in day_events:
        lines.append(f"  {clock.ist(e.at).strftime('%H:%M')} {e.kind}: {e.summary}")
    return "\n".join(lines)


def loops(rows: list[OpenLoop]) -> str:
    if not rows:
        return "OPEN LOOPS: none."
    lines = ["OPEN LOOPS (unfinished; close them when resolved):"]
    for r in rows:
        wake = f", check again {clock.ist(r.wake_at).strftime('%d %b %H:%M')}" if r.wake_at else ""
        lines.append(f"  - [{r.id}] {r.kind}: {r.title}{wake}")
    return "\n".join(lines)
