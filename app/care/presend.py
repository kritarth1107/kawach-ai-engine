"""One check every scheduled message passes right before it goes (founder 20 Oct: "nudges should check before sending").

A reminder, a dose message, an activity check-in or a follow-up was planned earlier. Since then the person may have
said or logged something that cancels it, says it is already done, postpones it, or makes it wrong ("skipping gym",
"doctor moved the appointment", "travelling till Sunday", "already took it"). Each sender used to check only its own
record, so a change said in the chat reached none of the others (live 20 Oct: "Skipping gym" at 19:15, then a gym
reminder at 19:30 and "are you back from the gym?" at 21:15).

What they wrote and what was logged about them today is read; when there is nothing, the message goes with no model
call. Otherwise a fast model answers send or hold. Words are judged by the model, never a word list. If the check
cannot run, the message goes (a missed reminder is worse than one extra).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import CareEvent, Turn
from app.core import clock

logger = logging.getLogger(__name__)

# Bookkeeping events say nothing the person decided.
_IGNORE = ("memory_extract", "learned", "pattern", "dream", "reminder_sent", "reminder_failed", "loop_expired", "web_search",
           "alert_whatsapp", "alert_dashboard", "nudge", "brief", "nudge_held")

CHECK = (
    "A message is about to be sent to a person on WhatsApp by their care assistant. PLANNED is what it is about. TODAY "
    "is what that person wrote and what was recorded about them today, oldest first. Decide whether the planned message "
    "should still go. Hold it only when something in TODAY cancels it for today, says it is already done, postpones it "
    "past now, or makes it wrong (another time, stopped by a doctor, away, unwell for it). Unrelated things never hold "
    'it. Answer only JSON: {"send": true|false, "why": "<a few words>"}.')


def _start_of_day(now: datetime) -> datetime:
    local = clock.ist(now)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


async def today_lines(session: AsyncSession, family_id: str, person: str, now: datetime | None = None) -> list[str]:
    """What they wrote and what was recorded about them since midnight (IST), as short timed lines."""
    now = now or clock.now()
    since = _start_of_day(now)
    said = (await session.execute(select(Turn.at, Turn.text).where(
        Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user", Turn.at >= since, Turn.at <= now)
        .order_by(Turn.at).limit(40))).all()
    logged = (await session.execute(select(CareEvent.at, CareEvent.kind, CareEvent.summary).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == person, CareEvent.at >= since, CareEvent.at <= now,
        CareEvent.kind.notin_(_IGNORE)).order_by(CareEvent.at).limit(40))).all()
    rows = [(at, f"they wrote: {(t or '').strip()[:200]}") for at, t in said if (t or "").strip()]
    rows += [(at, f"recorded {k}: {(s or '').strip()[:160]}") for at, k, s in logged if (s or "").strip()]
    return [f"{clock.ist(at).strftime('%H:%M')} {line}" for at, line in sorted(rows, key=lambda r: r[0])]


async def still_right(session: AsyncSession, family_id: str, person: str, about: str, now: datetime | None = None) -> tuple[bool, str]:
    """(send, why). Nothing said or logged today: send, no model call."""
    if os.getenv("PRESEND_CHECK", "on").strip().lower() == "off" or not (about or "").strip():
        return True, ""
    lines = await today_lines(session, family_id, person, now)
    if not lines:
        return True, ""
    from app.llm import router

    text = f"PLANNED: {about.strip()[:300]}\nNOW: {clock.ist(now or clock.now()).strftime('%H:%M')}\nTODAY:\n" + "\n".join(lines[-30:])
    try:
        out = await router.complete("judge_fast", system_stable=CHECK, effort="low", max_tokens=80, timeout_s=8,
                                    messages=[{"role": "user", "content": [{"type": "text", "text": text}]}])
        m = re.search(r"\{.*\}", out.text or "", re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001 — the message goes; a reminder missed is worse than one extra
        logger.warning("presend check failed family=%s; sending", family_id[:8])
        return True, ""
    send = data.get("send") is not False
    why = str(data.get("why") or "")[:120]
    if not send:
        logger.info("presend held family=%s about=%s why=%s", family_id[:8], about[:60], why)
    return send, why

