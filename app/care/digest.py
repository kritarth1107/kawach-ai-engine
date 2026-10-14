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
    "nutrition_target",
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
    "nutrition_target": "Nutrition targets and food rules (who set them; strict or a guide)",
    "setting": "Settings",
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
    "activity": "Activities and habits",
    "learned": "What Saheli learned (from what happened; the family can undo it)",
}


def link_text(links: list | None) -> str:
    """What an item's links add to its line: ' — for diabetes; prescribed by Dr Sharma; from the 3 Sep prescription', and
    a link still waiting for a caregiver said so."""
    from app.care import links as care_links

    if not links:
        return ""
    on = [care_links.phrase(ln) for ln in links if ln.status == "active"]
    waiting = [care_links.phrase(ln) for ln in links if ln.status == "pending" and not (ln.detail or {}).get("end")]
    going = [care_links.phrase(ln) for ln in links if ln.status == "pending" and (ln.detail or {}).get("end")]
    bits = ["; ".join(on)] if on else []
    if waiting:
        bits.append("PENDING link, not in effect until a caregiver confirms: " + "; ".join(waiting))
    if going:
        bits.append("removal waiting for a caregiver: " + "; ".join(going))
    return (" — " + " | ".join(bits)) if bits else ""


def care_record(name: str, rows: list[CareFact], stale: dict[str, str] | None = None, links: list | None = None,
                rejected: list | None = None) -> str:
    """stale: {key: "over 4 months ago"} from app.care.freshness, marked so old facts are re-checked before acting.
    links: the person's care links (app.care.links), each shown on the line of the item it starts from.
    rejected: what the family said is wrong (app.care.rejected), listed as "never say or use".
    A PINNED line was said or corrected by a person: only a person changes it (trust plan A3)."""
    from app.care import links as care_links
    from app.care import rejected as care_rejected

    stale = stale or {}
    # a pending stop marked never-true: the active line must not be read as fact while a caregiver confirms
    withdrawn = {f.key for f in rows if f.status == "pending" and (f.value or {}).get("stopped") and str(f.note or "").startswith(care_rejected.REJECTED_NOTE)}
    taken_back = [f for f in rows if f.status == "active" and f.key in withdrawn]
    rows = [f for f in rows if f.key not in withdrawn]
    linked = care_links.by_from(list(links or []))
    by_domain: dict[str, list[CareFact]] = defaultdict(list)
    activities = {f.key for f in rows if f.domain == "activity" and f.status == "active"}
    for f in rows:
        if f.domain == "routine" and (f.value or {}).get("as_activity") in activities:
            continue  # the same time is listed once, as the activity (app.care.dedupe)
        by_domain[f.domain].append(f)
    lines = [f"CARE RECORD — {name}"]
    naming = next((f for f in rows if f.key == "naming:address_as" and f.status == "active"), None)
    avoid = [w for w in ((naming.value or {}).get("avoid") or []) if isinstance(w, str)] if naming else []
    if naming and ((naming.value or {}).get("name") or avoid):
        call = (naming.value or {}).get("name")
        lines.append("ALWAYS: " + (f"call them {call}. " if call else "") + (f"NEVER call them {', '.join(avoid)} (they asked; this holds in every message)." if avoid else ""))
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
                old = f"; last confirmed {stale[f.key]}, check it is still right before acting on it" if f.key in stale else ""
                pin = ", PINNED" if getattr(f, "pinned_at", None) else ""
                lines.append(f"  - [{f.key}] {f.text} ({tag}{confirmed}{old}{pin}){link_text(linked.get(f.key))}")
    shown = {f.key for f in rows if f.status == "active"}
    people = [(ref, ls) for ref, ls in linked.items() if ref not in shown and care_links.domain_of(ref) in care_links.PEOPLE]
    if people:
        lines.append("People and how they are related:")
        for ref, ls in people:
            lines.append(f"  - {care_links.name_of(ref) if not ref.startswith('person:') else '[' + ref + ']'}{link_text(ls)}")
    missing = [TITLES[d] for d in ("allergy", "medicine", "diet") if d not in by_domain]
    if missing and rows:
        lines.append(f"Not on file: {', '.join(missing)}. Never guess these.")
    never = care_rejected.block(list(rejected or []))
    if never:
        lines.append(never)
    if taken_back:
        lines.append("Never say, assume or use (they said this was a joke or never true):\n" + "\n".join(f"  - {f.text}" for f in taken_back[:15] if f.text))
    return "\n".join(lines)


def relation(m: dict, names: dict[str, str]) -> str:
    """", mother of Prem" from the relation the family gave when adding someone (it is to the member who added them)."""
    rel = str(m.get("relation") or "").strip().lower()
    if not rel or rel in {"other", "self"}:
        return ""
    to = names.get(str(m.get("relation_to") or ""))
    if to and str(m.get("relation_to")) != str(m.get("id")):
        return f", {rel} of {to}"
    return f", relation given: {rel} (to the family member who added them)"


def household(elder: dict, members: list[dict], speaker_id: str, naming: dict[str, dict] | None = None) -> str:
    """naming: {person id: {"name", "avoid"}} (app.care.naming.for_people): how to address each person, so a message she
    starts to anyone (a check-in, an order update) uses the name they asked for, not the one on the roster."""
    from app.care import naming as care_naming

    roster = {str(m.get("id")): m for m in members}
    who = {**roster.get(str(elder.get("id")), {}), **{k: v for k, v in elder.items() if v}}
    names = {str(m.get("id")): str(m.get("name") or "") for m in [*members, elder] if m.get("name")}

    def call(pid) -> str:
        how = care_naming.line((naming or {}).get(str(pid)))
        return f" — {how}" if how else ""

    lines = [f"HOUSEHOLD\n  Care recipient: {elder.get('name')} (id {elder.get('id')}){relation(who, names)}{call(elder.get('id'))}"]
    for m in members:
        if m.get("id") == elder.get("id"):
            continue
        you = "  ← speaking now" if m.get("id") == speaker_id else ""
        lines.append(f"  {m.get('role', 'member')}: {m.get('name')} (id {m.get('id')}){relation(m, names)}{call(m.get('id'))}{you}")
    if speaker_id == elder.get("id"):
        lines.append(f"  {elder.get('name')} is speaking now.")
        if "caregiver" in str(roster.get(str(speaker_id), {}).get("role") or "").lower():
            # self care (as TurnCtx.self_care: the person cared for holds a caregiver role). Hard-turn A/B 13 Oct: a self-care
            # user asking about a new medicine was told "I will check with your family"; there is no one to check with.
            lines.append(f"  SELF CARE: {elder.get('name')} manages their own care. Their own changes are saved at once and they decide; never "
                         "say you will check with the family or a caregiver about their own care. A medical question goes to their own doctor.")
    lines.append("  You can message only the people listed here. Helpers, neighbours, doctors and anyone else are not reachable by you.")
    if len(members) > 1:
        lines.append("  Relations are only the ones written here (and what the care record says). Never guess one: the person cared for may be the"
                     " speaker's parent, spouse, grandparent, in-law or sibling, so where no relation is known use their name, not a word"
                     " like Maa, Papa or your son.")
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
    day_events = [e for e in day_events if e.kind != "dose_corrected" and not (e.payload or {}).get("forgotten")]  # a correction, or something they asked removed, did not happen
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
