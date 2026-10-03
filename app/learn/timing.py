"""When to message: learned per person (and across families) for messages Saheli starts on her own.

Only non-urgent proactive messages are timed: follow-ups after an alert, the weekly check-in, nudges about
patterns. Medicine reminders keep their set time and alerts go at once: they are never delayed or experimented on.

- best hour: Thompson sampling over two-hour windows between 08:00 and 21:00. Each window's chance of a reply comes
  from this person's history, with all families' history for the same kind of message as a weak prior. Mostly the
  best window wins; sometimes another is tried, which is how it keeps learning.
- quiet: after three unanswered messages in a row, Saheli stops starting non-urgent conversations with that person
  until they write again.
- daily cap: at most 4 such messages a day, 2 for someone who rarely replies.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.learn.models import ReplyLog

WINDOWS = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 21)]
PRIOR_WEIGHT = 0.2  # how much all families count against this person's own history
QUIET_AFTER = 3
CAP, CAP_LOW = 4, 2
LOW_REPLY_RATE = 0.3
TIMED = {"followup", "checkin"}
_global: dict[str, tuple[float, dict]] = {}
GLOBAL_TTL = 3600.0


def window_of(hour: int) -> int | None:
    for i, (a, b) in enumerate(WINDOWS):
        if a <= hour < b:
            return i
    return None


async def _global_counts(session: AsyncSession, situation: str) -> dict[int, list[int]]:
    hit = _global.get(situation)
    if hit and time.monotonic() - hit[0] < GLOBAL_TTL:
        return hit[1]
    rows = (await session.execute(
        select(ReplyLog.sent_hour, ReplyLog.replied).where(ReplyLog.kind == "proactive", ReplyLog.situation == situation,
                                                            ReplyLog.replied.is_not(None), ReplyLog.at >= clock.now() - timedelta(days=60))
    )).all()
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for hour, replied in rows:
        w = window_of(hour)
        if w is not None:
            counts[w][0] += 1
            counts[w][1] += 1 if replied else 0
    out = dict(counts)
    _global[situation] = (time.monotonic(), out)
    if len(_global) > 64:
        _global.pop(next(iter(_global)))
    return out


async def _person_rows(session: AsyncSession, family_id: str, person: str, days: int = 60) -> list:
    return list((await session.execute(
        select(ReplyLog).where(ReplyLog.family_id == family_id, ReplyLog.thread_id == person, ReplyLog.kind == "proactive",
                               ReplyLog.at >= clock.now() - timedelta(days=days)).order_by(ReplyLog.at)
    )).scalars())


async def best_hour(session: AsyncSession, family_id: str, person: str, situation: str, *, default: int, seed: str = "") -> int:
    """The hour (start of a window) to send this kind of message to this person."""
    rows = [r for r in await _person_rows(session, family_id, person) if r.replied is not None]
    glob = await _global_counts(session, situation)
    own: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        w = window_of(r.sent_hour)
        if w is not None:
            own[w][0] += 1
            own[w][1] += 1 if r.replied else 0
    if not own and not glob:
        return default
    rng = random.Random(f"{family_id}:{person}:{situation}:{clock.ist_day()}:{seed}")
    dflt = window_of(default)
    best, best_draw = default, -1.0
    for i, (a, _) in enumerate(WINDOWS):
        g_sent, g_rep = glob.get(i, [0, 0])
        o_sent, o_rep = own.get(i, [0, 0])
        alpha = 1 + o_rep + PRIOR_WEIGHT * g_rep + (1 if i == dflt else 0)  # a small nudge toward the usual time
        beta = 1 + (o_sent - o_rep) + PRIOR_WEIGHT * (g_sent - g_rep)
        draw = rng.betavariate(alpha, beta)
        if draw > best_draw:
            best, best_draw = a, draw
    return best


async def next_send_at(session: AsyncSession, family_id: str, person: str, situation: str, *, default_hour: int, earliest):
    """The next time at or after `earliest` that falls in this person's best window."""
    hour = await best_hour(session, family_id, person, situation, default=default_hour)
    local = clock.ist(earliest)
    at = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if at < local:
        at += timedelta(days=1)
    return at


async def quiet_reason(session: AsyncSession, family_id: str, person: str, situation: str) -> str | None:
    """Why Saheli should not start a non-urgent conversation with this person now, or None."""
    if situation not in TIMED:
        return None
    rows = await _person_rows(session, family_id, person, days=28)
    from app.care.models import Turn

    last_user = (await session.execute(
        select(Turn.at).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user").order_by(Turn.id.desc()).limit(1)
    )).scalar_one_or_none()
    since_reply = [r for r in rows if last_user is None or r.at > last_user]
    if len(since_reply) >= QUIET_AFTER:
        return f"they have not answered your last {len(since_reply)} messages; wait until they write (reply none)"
    today = clock.ist_day()
    sent_today = sum(1 for r in rows if clock.ist_day(r.at) == today)
    answered = [r for r in rows if r.replied is not None]
    rate = sum(1 for r in answered if r.replied) / len(answered) if len(answered) >= 10 else 1.0
    cap = CAP_LOW if rate < LOW_REPLY_RATE else CAP
    if sent_today >= cap:
        return f"you already started {sent_today} conversations with them today (their limit is {cap}); wait for tomorrow (reply none)"
    return None


async def summary(session: AsyncSession) -> list[dict]:
    """Reply rate per window and situation across families (the admin view)."""
    out = []
    for situation in sorted(TIMED):
        g = await _global_counts(session, situation)
        out.append({"situation": situation, "windows": [
            {"from": f"{a:02d}:00", "to": f"{b:02d}:00", "sent": g.get(i, [0, 0])[0],
             "replyRate": round(g.get(i, [0, 0])[1] / g[i][0], 3) if g.get(i, [0, 0])[0] else None}
            for i, (a, b) in enumerate(WINDOWS)]})
    return out


def reset_cache() -> None:
    _global.clear()
