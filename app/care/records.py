"""Health records Saheli remembers, only after a person chose "let Saheli remember it".

Each saved record becomes one `report` event (searchable with recall, never fades) and the newest lines are kept in a
short "Health reports" note, so Saheli knows the latest results without searching. Forgetting a record (deleted, or
someone else's) removes both.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import memory_index, store
from app.care.models import CareEvent
from app.care.redact import scrub_secrets

NOTE_SLUG = "health-reports"
NOTE_LINES = 12


def _label(iso: str | None) -> str:
    try:
        d = date.fromisoformat(str(iso))
        return f"{d.day} {d.strftime('%b %Y')}"
    except (TypeError, ValueError):
        return "undated"


async def _report_events(session: AsyncSession, family_id: str, subject_id: str, document_id: str | None = None) -> list[CareEvent]:
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "report")
    rows = list((await session.execute(q)).scalars())
    rows = [e for e in rows if not (e.payload or {}).get("forgotten")]
    return [e for e in rows if (e.payload or {}).get("document_id") == document_id] if document_id else rows


async def _rebuild_note(session: AsyncSession, family_id: str, subject_id: str, actor_id: str | None) -> None:
    events = sorted(await _report_events(session, family_id, subject_id), key=lambda e: str((e.payload or {}).get("date") or ""), reverse=True)
    lines: list[str] = []
    for e in events:
        p = e.payload or {}
        for point in p.get("points") or []:
            lines.append(f"- {_label(p.get('date'))} · {p.get('title') or 'report'}: {point}")
    body = "\n".join(lines[:NOTE_LINES])
    await store.upsert_note(session, family_id=family_id, subject_id=subject_id, slug=NOTE_SLUG, title="Health reports (latest)",
                            body_md=body or "- No saved reports yet.", actor_id=actor_id, source="dashboard", reason="health record saved or removed")


async def _mark_forgotten(session: AsyncSession, family_id: str, events: list[CareEvent]) -> int:
    for e in events:
        e.payload = {**(e.payload or {}), "forgotten": True}
        await memory_index.unindex(session, family_id, f"event:{e.id}")
    return len(events)


async def remember(session: AsyncSession, family_id: str, subject_id: str, *, document_id: str, title: str, when: str | None,
                   points: list[str], actor_id: str | None) -> dict:
    clean = [scrub_secrets(p.strip())[:260] for p in points if p and p.strip()][:8]
    if not clean:
        return {"remembered": 0}
    # Saving the same record again replaces what was remembered from it.
    await _mark_forgotten(session, family_id, await _report_events(session, family_id, subject_id, document_id))
    title = scrub_secrets(title.strip())[:200] or "Health report"
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="report",
        summary=f"{title} ({_label(when)}): " + "; ".join(clean),
        payload={"document_id": document_id, "title": title, "date": when, "points": clean}, actor_id=actor_id,
    )
    await _rebuild_note(session, family_id, subject_id, actor_id)
    return {"remembered": len(clean)}


async def forget(session: AsyncSession, family_id: str, subject_id: str, *, document_id: str, memory_document_id: str | None,
                 actor_id: str | None) -> dict:
    n = await _mark_forgotten(session, family_id, await _report_events(session, family_id, subject_id, document_id))
    if n:
        await _rebuild_note(session, family_id, subject_id, actor_id)
    removed_doc = False
    if memory_document_id:
        # An older copy kept for the first chat version (memory_documents; its chunks go with it).
        try:
            from app.models.entities import MemoryDocument

            res = await session.execute(delete(MemoryDocument).where(MemoryDocument.id == uuid.UUID(memory_document_id)))
            removed_doc = bool(res.rowcount)
        except ValueError:
            pass
    return {"forgotten": n, "memory_document_removed": removed_doc}


async def end_finished_courses(session: AsyncSession, host_for, today: str) -> int:
    """Medicines added from a prescription with a number of days stop after the last day (their reminders too)."""
    from app.brain import tools
    from app.care.models import CareFact

    rows = list((await session.execute(select(CareFact).where(CareFact.domain == "medicine", CareFact.status == "active"))).scalars())
    ended = 0
    for f in rows:
        ends = str((f.value or {}).get("ends_on") or "")
        if not ends or ends >= today:
            continue
        # The end date was part of what the family saved, so the stop carries the same authority (no new confirm).
        w = await store.stop_fact(session, family_id=f.family_id, subject_id=f.subject_id, key=f.key,
                                  reason=f"course finished ({_label(ends)})", source_kind=f.source_kind, stated_by="saheli")
        if not w or w.result != "stopped":
            continue
        await store.record_event(session, family_id=f.family_id, subject_id=f.subject_id, kind="fact_stopped",
                                 summary=f"{(f.value or {}).get('name') or f.key}: course finished ({_label(ends)})", payload={"key": f.key})
        ctx = tools.TurnCtx(
            session=session, host=host_for(f.family_id), family_id=f.family_id, elder={"id": f.subject_id, "name": ""},
            speaker={"id": "saheli", "name": "Saheli", "role": "system"}, members=[{"id": f.subject_id}], channel="scheduler",
        )
        try:
            await tools._sync_medicine(ctx, f.subject_id, f.key, f.value or {}, active=False)
        except Exception:  # noqa: BLE001 — the record is stopped; the reminder sync is retried by the dashboard's next edit
            pass
        ended += 1
    return ended
