"""Seed a family's care memory from what the backend already holds, once, before their first v2 turn: the care schedules
(medicines and timed routines). What the old profile held (name to use, allergies, diet rules, learned facts, language)
moved into the care record once, with the memory move (app.care.memory_unify, trust plan A1); the backend no longer sends
it here."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.brain.host import ToolHost
from app.care import store
from app.care.doses import from_backend_days
from app.care.domains import fact_key
from app.care.models import CareEvent

logger = logging.getLogger(__name__)

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

    await store.record_event(
        session, family_id=family_id, subject_id=elder_id, kind="import_done", summary=f"Imported care record: {dict(counts)}",
        ref=f"import:{elder_id}",
    )
    logger.info("care import family=%s elder=%s %s", family_id, elder_id, dict(counts))
    return dict(counts)
