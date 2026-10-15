"""Same-person messages that arrive during a turn become one follow-up. A different person does not wait.

The family lock stays on scheduled turns. A person's own turn takes a speaker lock only, so two people in one
family can both be in a model call. Messages that land while the lock is held are claimed together by the next
one of them, and only that one calls the model. The claim/resume path in the turn itself is unchanged: a ref
that already has a reply comes back as a duplicate, and a saved line with no reply is continued.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.brain.host import ToolHost
from app.care.models import TurnInbox
from app.core import clock, trace

logger = logging.getLogger(__name__)

_TOGETHER = "They sent these one after another. Answer them together.\n"


def _speaker_key(family_id: str, speaker_id: str) -> str:
    return f"speaker:{family_id}:{speaker_id}"


def _engine(session: AsyncSession) -> AsyncEngine:
    """A commit returns a session's connection to the pool, which would strand a session advisory lock."""
    bind = session.bind
    if isinstance(bind, AsyncEngine):
        return bind
    engine = getattr(bind, "engine", None)
    if not isinstance(engine, AsyncEngine):
        raise RuntimeError("speaker lock needs a database engine")
    return engine


async def _lock_speaker(conn, key: str, wait_s: float) -> None:
    from app.brain.loop import FamilyBusy

    # This connection is only the lock, and it closes when the turn finishes. Setting the
    # timeout needs no read of the old value and no restore.
    await conn.execute(text("SELECT set_config('lock_timeout', :t, false)"), {"t": f"{int(wait_s * 1000)}ms"})
    try:
        await conn.execute(text("SELECT pg_advisory_lock(hashtext(:k))"), {"k": key})
    except DBAPIError as exc:
        await conn.rollback()
        orig = getattr(exc, "orig", None)
        timed_out = getattr(orig, "sqlstate", None) == "55P03" or getattr(orig, "pgcode", None) == "55P03"
        if timed_out or "lock timeout" in str(exc).lower():
            raise FamilyBusy(f"speaker busy for {wait_s:.0f}s") from exc
        raise
    await conn.commit()


async def _unlock_speaker(conn, key: str) -> None:
    try:
        await conn.execute(text("SELECT pg_advisory_unlock(hashtext(:k))"), {"k": key})
    except Exception:  # noqa: BLE001 — a failed turn leaves the connection aborted; the lock still has to drop
        await conn.rollback()
        await conn.execute(text("SELECT pg_advisory_unlock(hashtext(:k))"), {"k": key})
    await conn.commit()


def _payload(req) -> dict:
    return {
        "images": req.images or [],
        "media_note": req.media_note or "",
        "location": req.location,
        "modality": req.modality,
        "voice_confidence": req.voice_confidence,
        "voice_language": req.voice_language,
    }


def _answered(row: TurnInbox, message_ref: str):
    from app.brain.loop import TurnResult

    if row.lead and row.message_ref == message_ref:
        return TurnResult(reply=row.reply or "", actions=[], alerts=[], duplicate=True)
    return TurnResult(reply="", actions=[], alerts=[], duplicate=True)


def _joined(rows: list[TurnInbox], req):
    # One message stays as the person sent it, including an empty text whose media note
    # the turn places at the end. The together-line is only for a real burst.
    if len(rows) == 1:
        return req
    texts = [r.text for r in rows if (r.text or "").strip()]
    text_out = _TOGETHER + "\n".join(texts)
    images: list = []
    notes: list[str] = []
    location = None
    modality = req.modality
    voice_confidence = req.voice_confidence
    voice_language = req.voice_language
    for row in rows:
        extra = row.payload or {}
        images.extend(extra.get("images") or [])
        if extra.get("media_note"):
            notes.append(extra["media_note"])
        if extra.get("location") and location is None:
            location = extra["location"]
        if extra.get("modality") == "voice":
            modality = "voice"
            voice_confidence = extra.get("voice_confidence")
            voice_language = extra.get("voice_language")
    lead = rows[0]
    return replace(
        req,
        text=text_out,
        message_ref=lead.message_ref,
        images=images,
        media_note=" ".join(notes),
        location=location if location is not None else req.location,
        modality=modality,
        voice_confidence=voice_confidence,
        voice_language=voice_language,
    )


async def _claim(session: AsyncSession, family_id: str, speaker_id: str) -> list[TurnInbox]:
    session.expire_all()
    rows = (
        await session.execute(
            select(TurnInbox)
            .where(TurnInbox.family_id == family_id, TurnInbox.speaker_id == speaker_id, TurnInbox.status.in_(("waiting", "running")))
            .order_by(TurnInbox.id)
            .with_for_update()
        )
    ).scalars().all()
    for row in rows:
        row.status = "running"
    await session.commit()
    return list(rows)


async def _while_locked(session: AsyncSession, host: ToolHost, req, waited: float):
    from app.brain import loop

    # The row was loaded before this wait. Another turn may have answered it since.
    session.expire_all()
    row = (
        await session.execute(
            select(TurnInbox).where(TurnInbox.family_id == req.family_id, TurnInbox.message_ref == req.message_ref)
        )
    ).scalar_one()
    if row.status == "answered":
        trace.note_step("lock", (time.monotonic() - waited) * 1000)
        return _answered(row, req.message_ref)
    batch = await _claim(session, req.family_id, req.speaker["id"])
    trace.note_step("lock", (time.monotonic() - waited) * 1000)
    if not any(item.message_ref == req.message_ref for item in batch):
        row = (
            await session.execute(
                select(TurnInbox).where(TurnInbox.family_id == req.family_id, TurnInbox.message_ref == req.message_ref)
            )
        ).scalar_one()
        if row.status == "answered":
            return _answered(row, req.message_ref)
        batch = [row]
        row.status = "running"
        await session.commit()
    lead = batch[0]
    result = await loop._run_turn(session, host, _joined(batch, req), take_lock=False)
    keep = bool((result.reply or "").strip()) or result.duplicate
    ids = [item.id for item in batch]
    if keep:
        # A column refresh after the turn would drop an in-memory status change and the next
        # message would answer the same batch again.
        await session.execute(update(TurnInbox).where(TurnInbox.id.in_(ids)).values(status="answered"))
        await session.execute(
            text("UPDATE turn_inbox SET payload = COALESCE(payload, '{}'::jsonb) - 'images' WHERE id = ANY(:ids)"),
            {"ids": ids},
        )
        await session.execute(
            update(TurnInbox).where(TurnInbox.id == lead.id).values(lead=True, reply=result.reply or "")
        )
        await session.commit()
    else:
        await session.execute(
            update(TurnInbox).where(TurnInbox.id.in_(ids), TurnInbox.status == "running").values(status="waiting")
        )
        await session.commit()
    if req.message_ref == lead.message_ref:
        return result
    return loop.TurnResult(reply="", actions=[], alerts=[], duplicate=True)


async def run(session: AsyncSession, host: ToolHost, req):
    """One person's messages. A system turn keeps the family lock. A button waits for this person, then runs alone."""
    from app.brain import loop
    from app.care import outcomes

    speaker_id = (req.speaker or {}).get("id") or ""
    if req.speaker.get("role") == "system" or not speaker_id:
        return await loop._run_turn(session, host, req)
    if outcomes.parse_button(req.text or "") or not req.message_ref:
        key = _speaker_key(req.family_id, speaker_id)
        waited = time.monotonic()
        async with _engine(session).connect() as conn:
            await _lock_speaker(conn, key, loop.LOCK_WAIT_S["message"])
            trace.note_step("lock", (time.monotonic() - waited) * 1000)
            try:
                return await loop._run_turn(session, host, req, take_lock=False)
            finally:
                await _unlock_speaker(conn, key)

    await session.execute(
        pg_insert(TurnInbox)
        .values(
            family_id=req.family_id,
            speaker_id=speaker_id,
            message_ref=req.message_ref,
            text=req.text or "",
            status="waiting",
            lead=False,
            reply="",
            payload=_payload(req),
            created_at=clock.now(),
        )
        .on_conflict_do_nothing(index_elements=["family_id", "message_ref"])
    )
    await session.commit()
    row = (
        await session.execute(select(TurnInbox).where(TurnInbox.family_id == req.family_id, TurnInbox.message_ref == req.message_ref))
    ).scalar_one()
    if row.status == "answered":
        return _answered(row, req.message_ref)

    key = _speaker_key(req.family_id, speaker_id)
    waited = time.monotonic()
    async with _engine(session).connect() as conn:
        await _lock_speaker(conn, key, loop.LOCK_WAIT_S["message"])
        try:
            return await _while_locked(session, host, req, waited)
        finally:
            await _unlock_speaker(conn, key)
