"""Seed a family's care memory from what the backend already holds, once, before their first v2 turn."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain.host import ToolHost
from app.care import store
from app.care.doses import from_backend_days
from app.care.domains import fact_key, slug
from app.care.models import CareEvent

logger = logging.getLogger(__name__)

DIET_WORDS = {
    "low_salt": ("salt", "namak", "sodium"),
    "low_sugar": ("sugar", "cheeni", "sweet", "diabet"),
    "low_oil": ("oil", "fried", "tel"),
    "no_onion": ("onion", "pyaz", "pyaaz"),
    "no_garlic": ("garlic", "lehsun"),
    "no_spice": ("spice", "spicy", "mirch", "teekha"),
    "vegetarian": ("vegetarian", "veg only", "no meat", "shakahari"),
}


def to_hhmm(raw: str | None) -> str | None:
    """'1:00 PM', '8 am', '13:00', '08:30' → 'HH:MM' (24h). None when it is not a clock time."""
    m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*$", str(raw or ""), re.I)
    if not m or (m.group(2) is None and not m.group(3)):
        return None
    h, mins, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower().replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return f"{h:02d}:{mins:02d}" if h < 24 and mins < 60 else None


def diet_name(rule: str) -> str:
    low = rule.lower()
    for name, words in DIET_WORDS.items():
        if any(w in low for w in words):
            return name
    return slug(rule)


async def already_imported(session: AsyncSession, family_id: str, elder_id: str) -> bool:
    row = await session.execute(
        select(CareEvent.id).where(
            CareEvent.family_id == family_id, CareEvent.subject_id == elder_id, CareEvent.kind == "import_done"
        )
    )
    return row.first() is not None


async def import_family(session: AsyncSession, host: ToolHost, *, family_id: str, backend_family_id: str, elder_id: str) -> dict:
    """family_id is the memory namespace (may be shadow:…); backend_family_id is the real family."""
    data = await host.call("export_care_record", {}, family_id=backend_family_id, subject_id=elder_id, actor_id=elder_id)
    counts: dict[str, int] = defaultdict(int)

    async def put(domain: str, name: str, value: dict, text: str) -> None:
        w = await store.write_fact(
            session, family_id=family_id, subject_id=elder_id, domain=domain, key=fact_key(domain, name),
            value=value, text=text, source_kind="import", confidence=0.95,
        )
        counts[w.result] += 1

    for a in data.get("allergies") or []:
        await put("allergy", a, {"allergen": a}, f"Allergic to {a}")
    for r in data.get("dietRules") or []:
        await put("diet", diet_name(r), {"rule": diet_name(r), "detail": r}, r)
    if data.get("nameToUse"):
        await put("naming", "address_as", {"name": data["nameToUse"], **({"avoid": ["maa"]} if data.get("avoidMaa") else {})}, f"Call her {data['nameToUse']}")
    elif data.get("avoidMaa"):
        await put("naming", "address_as", {"avoid": ["maa"]}, "Never call her 'maa'")
    if data.get("language"):
        await put("language", "preferred", {"language": data["language"]}, f"Prefers {data['language']}")

    # Medicines: one fact per medicine, gathering its dose times; the rows are claimed so later syncs update them.
    meds: dict[str, dict] = {}
    claims: dict[str, list[str]] = defaultdict(list)
    for s in data.get("schedules") or []:
        if s.get("type") != "MEDICINE":
            if s.get("time"):
                t = to_hhmm(s["time"]) or s["time"]
                await put("routine", s["title"], {"what": s["title"], "time": t, "type": s.get("type")}, f"{s['title']} at {t}")
            continue
        key = s.get("sourceKey") or fact_key("medicine", s["title"])
        m = meds.setdefault(key, {"name": s["title"], "dose": s.get("dosage"), "times": [], "instructions": s.get("instructions"), "days": from_backend_days(s.get("daysOfWeek"))})
        if to_hhmm(s.get("time")):
            m["times"].append(to_hhmm(s["time"]))
        if not s.get("sourceKey"):
            claims[key].append(s["scheduleId"])
    for p in data.get("profileMedicines") or []:
        key = fact_key("medicine", p["name"])
        m = meds.setdefault(key, {"name": p["name"], "dose": p.get("dose"), "times": [], "instructions": None, "days": None})
        if to_hhmm(p.get("time")) and to_hhmm(p["time"]) not in m["times"]:
            m["times"].append(to_hhmm(p["time"]))
    for key, m in meds.items():
        m["times"] = sorted(set(m["times"]))
        when = ", ".join(m["times"]) or "no time set"
        w = await store.write_fact(
            session, family_id=family_id, subject_id=elder_id, domain="medicine", key=key,
            value={k: v for k, v in m.items() if v}, text=f"{m['name']}{(' ' + m['dose']) if m.get('dose') else ''} at {when}",
            source_kind="import", confidence=0.95,
        )
        counts[w.result] += 1
        if claims.get(key) and not family_id.startswith("shadow:"):
            await host.call("claim_schedule_rows", {"key": key, "scheduleIds": claims[key]}, family_id=backend_family_id, subject_id=elder_id, actor_id=elder_id)

    learned = data.get("learned") or []
    if learned:
        lines = [f"- {'(confirmed) ' if x.get('confirmed') else ''}{x['text']}" for x in learned[:80]]
        await store.upsert_note(session, family_id=family_id, subject_id=elder_id, slug="profile", title="Learned profile", body_md="\n".join(lines))
        counts["notes"] += len(lines)

    await store.record_event(
        session, family_id=family_id, subject_id=elder_id, kind="import_done", summary=f"Imported care record: {dict(counts)}",
        ref=f"import:{elder_id}",
    )
    logger.info("care import family=%s elder=%s %s", family_id, elder_id, dict(counts))
    return dict(counts)
