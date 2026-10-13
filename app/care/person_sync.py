"""Tell the backend when a person's name-to-use or language/voice setting changes (trust plan A1, A3).

The backend reads both from the engine (GET /v2/person/…, cached 60 s) for the text it writes itself: the reminder
fallback line, the holding line when the brain cannot answer, voice notes. A change here drops its cached copy at once
through the host tool `person_memory_changed`, sent after the write commits (never from inside the transaction, never
for shadow or simulated families). A failed push is only logged: the 60 s cache catches up.
"""

from __future__ import annotations

import asyncio
import logging
import os

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

DOMAINS = ("naming", "language")
_KEY = "person_changed"
_tasks: set[asyncio.Task] = set()


def enabled() -> bool:
    return os.getenv("PERSON_SYNC", "on").strip().lower() != "off"


def mark(session: AsyncSession, family_id: str, subject_id: str, domain: str) -> None:
    """A naming or language fact of this person changed in this session; the backend hears after the commit."""
    if domain not in DOMAINS or not enabled() or family_id.startswith(("shadow:", "sim-")) or subject_id == "family":
        return
    sync = session.sync_session
    sync.info.setdefault(_KEY, set()).add((family_id, subject_id))
    if not event.contains(sync, "after_commit", _after_commit):
        event.listen(sync, "after_commit", _after_commit)
        event.listen(sync, "after_rollback", _after_rollback)


def _after_rollback(sync) -> None:
    sync.info.pop(_KEY, None)


def _after_commit(sync) -> None:
    changed = sync.info.pop(_KEY, set())
    if not changed:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for family_id, subject_id in changed:
        t = loop.create_task(notify(family_id, subject_id))
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)


async def notify(family_id: str, subject_id: str, host=None) -> bool:
    if host is None:
        from app.brain.host import LiveHost

        host = LiveHost()
    try:
        await host.call("person_memory_changed", {}, family_id=family_id, subject_id=subject_id, actor_id=subject_id)
        return True
    except Exception:  # noqa: BLE001 — the backend's 60 s cache catches up
        logger.warning("person change push failed family=%s", family_id[:8])
        return False
