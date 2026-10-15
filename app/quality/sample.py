"""Which replies the live check and the nightly judge spend on.

Ordinary turns are sampled: at most JUDGE_PER_FAMILY a family a day (default 3), and at most JUDGE_DAILY_CAP
ordinary judgments a day across every family (default 90). The cap does not grow with the number of families, so the
day's judge bill stays flat. A turn is never sampled away when a guard rewrote it, it raised or needed an alert, the
voice note was unsure, the person corrected it, or the founder reported that family that day.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock

ALWAYS = "always"
SAMPLE = "sample"
SKIP = "skip"


def per_family() -> int:
    """Ordinary turns judged for one family on one day. Negative: judge every ordinary turn."""
    return int(os.getenv("JUDGE_PER_FAMILY", "3"))


def daily_cap() -> int:
    """Ordinary judgments for the whole day. 0: no cap beyond the per-family one."""
    return int(os.getenv("JUDGE_DAILY_CAP", "90"))


def always_live(*, voice_unsure: bool, alerts: bool, health: bool, tools: list[str], situation: str) -> bool:
    """A live turn the check always runs on. The fields are the turn's own, never a reading of what was said."""
    if voice_unsure or alerts or health:
        return True
    if situation == "emergency":
        return True
    return "alert_caregiver" in (tools or [])


def always_logged(row, founder_threads: set[str]) -> bool:
    """A logged reply the nightly judge always grades."""
    if getattr(row, "judge_band", None) == ALWAYS:
        return True
    if row.situation == "emergency" or "alert_caregiver" in (row.tools or []):
        return True
    if row.corrected:
        return True
    return bool(row.thread_id and row.thread_id in founder_threads)


def admitted(family_id: str, day: str, n_families: int) -> bool:
    """Whether this family may take an ordinary slot today. Stable for a given day and family count."""
    cap, per = daily_cap(), per_family()
    if cap <= 0 or per < 0:
        return True
    slots = max(1, cap // max(1, per))
    if n_families <= slots:
        return True
    digest = hashlib.sha256(f"{day}:{family_id}".encode()).hexdigest()
    return int(digest[:8], 16) % n_families < slots


def pick(ordinary: list, *, mine: int, total: int) -> list:
    """The ordinary rows to grade, earliest first, inside this family's room and the day's cap."""
    per = per_family()
    if per < 0:
        return list(ordinary)
    room = per - mine
    cap = daily_cap()
    if cap > 0:
        room = min(room, cap - total)
    if room <= 0:
        return []
    return list(ordinary[:room])


def bounds(day: str | None = None) -> tuple[datetime, datetime, str]:
    start = datetime.fromisoformat(day).replace(tzinfo=clock.IST) if day else clock.now().astimezone(clock.IST).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1), start.date().isoformat()


async def _counts(session: AsyncSession, start: datetime, end: datetime, family_id: str | None) -> int:
    from app.learn.models import ReplyLog

    q = select(func.count()).select_from(ReplyLog).where(
        ReplyLog.at >= start, ReplyLog.at < end, ReplyLog.judge_band == SAMPLE)
    if family_id is not None:
        q = q.where(ReplyLog.family_id == family_id)
    return int((await session.execute(q)).scalar_one() or 0)


async def ordinary_room(session: AsyncSession, family_id: str) -> int:
    """How many more ordinary turns this family may judge today. 0: this turn is not in the sample."""
    per = per_family()
    if per < 0:
        return 1_000_000
    start, end, day = bounds()
    mine = await _counts(session, start, end, family_id)
    if mine >= per:
        return 0
    cap = daily_cap()
    if cap <= 0:
        return per - mine
    total = await _counts(session, start, end, None)
    if total >= cap:
        return 0
    from app.learn.models import ReplyLog

    n = int((await session.execute(
        select(func.count(func.distinct(ReplyLog.family_id))).where(ReplyLog.at >= start, ReplyLog.at < end)
    )).scalar_one() or 0)
    if mine == 0 and not admitted(family_id, day, max(n, 1)):
        return 0
    return min(per - mine, cap - total)
