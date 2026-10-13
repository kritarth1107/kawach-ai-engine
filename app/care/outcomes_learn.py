"""Saheli learns from what happens, not just from chats, and every learning is a fact the family sees and can undo
(trust plan A5, §10.1; founder 2026-10-13: "If the person always skips a nudge at 2 pm, she learns to try a different
time. Save it as a fact the caregiver can see and undo." and "she needs to be self-learning rather than always changing
code and updating instructions").

Nightly, from 28 days of outcomes per person:
- message timing: when Saheli starts a conversation (check-ins, follow-ups, regular nudges; ReplyLog "proactive"), which
  two-hour windows get answers. A window answered at most 20% of the time while another gets at least 60% (6+ tries
  each, smoothed with all families as the prior) becomes `learned:timing_messages` {avoid_hours, prefer_hours, evidence}.
  Check-ins and nudges then stay out of the avoided hours.
- habits: an activity skipped on the same weekday at least 5 of the last 7 times becomes
  `learned:habit_<activity>_<weekday>`, so Saheli asks about that day's plans instead of nudging.
- Medicine times are never changed by a learning (the caregiver sets doses); dose delays stay a suggestion to the
  caregiver (app/care/baselines.py).

A learning is a care fact in domain "learned", source "learned": it shows on the dashboard ("What Saheli learned"), in
memory history, and in the next nightly brief ("I noticed … so I'll …"). Undo stops it; a key stopped or rejected in the
last 60 days is not learned again. A caregiver changing it makes a pinned fact a learning never overwrites.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import CareEvent, CareFact
from app.core import clock

logger = logging.getLogger(__name__)

LEARN_DAYS = 28
MIN_TRIES = 6
LOW, HIGH = 0.2, 0.6
PRIOR_WEIGHT = 2.0  # pseudo-tries of the all-family rate added to each window
COOLDOWN = timedelta(days=60)
WINDOWS = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22)]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMING_KEY = "learned:timing_messages"  # a slug, so "stop learned timing_messages" finds it


def _window(hour: int) -> int | None:
    return next((i for i, (a, b) in enumerate(WINDOWS) if a <= hour < b), None)


def _label(i: int) -> str:
    a, b = WINDOWS[i]
    return f"{a % 12 or 12}–{b % 12 or 12} {'pm' if b >= 12 else 'am'}"


async def _cooling(session: AsyncSession, family_id: str, subject_id: str, key: str, now) -> bool:
    """Undone, stopped or rejected in the last 60 days: not learned again yet."""
    row = (await session.execute(select(CareFact.id).where(
        CareFact.family_id == family_id, CareFact.subject_id == subject_id, CareFact.key == key,
        CareFact.status.in_(("stopped", "retracted")), CareFact.recorded_at >= now - COOLDOWN).limit(1))).first()
    return row is not None


async def save_learned(session: AsyncSession, *, family_id: str, subject_id: str, key: str, value: dict, text: str,
                       now=None) -> str:
    """Write one learning as a visible, undoable fact. Returns the write result, or why it was not written."""
    now = now or clock.now()
    if not key.startswith("learned:"):
        raise ValueError("a learning key starts with learned:")
    if await _cooling(session, family_id, subject_id, key, now):
        return "cooling"
    w = await store.write_fact(session, family_id=family_id, subject_id=subject_id, domain="learned", key=key, value=value,
                               text=text, source_kind="learned", note="learned from what happened")
    if w.result in ("created", "superseded"):
        await store.record_event(session, family_id=family_id, subject_id=subject_id, kind="learned",
                                 summary=text[:300], payload={"key": key})
    return w.result


async def timing(session: AsyncSession, family_id: str, person: str, *, now=None) -> dict | None:
    """The message-timing learning for one person, or None when nothing is clear yet."""
    from app.learn.models import ReplyLog

    now = now or clock.now()
    since = now - timedelta(days=LEARN_DAYS)
    rows = (await session.execute(select(ReplyLog.thread_id, ReplyLog.sent_hour, ReplyLog.replied).where(
        ReplyLog.kind == "proactive", ReplyLog.replied.is_not(None), ReplyLog.at >= since))).all()
    allc = [0, 0]
    own: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for thread, hour, replied in rows:
        allc[0] += 1
        allc[1] += 1 if replied else 0
        w = _window(hour)
        if thread == person and w is not None:
            own[w][0] += 1
            own[w][1] += 1 if replied else 0
    prior = (allc[1] / allc[0]) if allc[0] else 0.5
    rate = {w: (yes + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT) for w, (n, yes) in own.items() if n >= MIN_TRIES}
    bad = sorted((w for w, r in rate.items() if r <= LOW), key=lambda w: rate[w])
    good = sorted((w for w, r in rate.items() if r >= HIGH), key=lambda w: -rate[w])
    if not bad or not good:
        return None
    avoid = [h for w in bad for h in range(*WINDOWS[w])]
    prefer = [h for h in range(*WINDOWS[good[0]])]
    b, g = bad[0], good[0]
    text = (f"Rarely answers messages around {_label(b)} ({own[b][1]} of {own[b][0]}); Saheli now writes around {_label(g)} "
            f"instead ({own[g][1]} of {own[g][0]} answered).")
    return {"key": TIMING_KEY, "text": text, "value": {
        "avoid_hours": avoid, "prefer_hours": prefer, "learned_on": clock.ist_day(now),
        "evidence": {_label(w): {"sent": n, "answered": yes} for w, (n, yes) in sorted(own.items())}}}


async def habits(session: AsyncSession, family_id: str, person: str, *, now=None) -> list[dict]:
    """Activities skipped on the same weekday at least 5 of the last 7 times."""
    now = now or clock.now()
    evs = list((await session.execute(select(CareEvent).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == person, CareEvent.kind.in_(("activity_done", "activity_skipped")),
        CareEvent.at >= now - timedelta(days=60)).order_by(CareEvent.at))).scalars())
    by: dict[tuple[str, int], list[str]] = defaultdict(list)
    names: dict[str, str] = {}
    for e in evs:
        key = (e.payload or {}).get("key")
        if not key:
            continue
        names[key] = (e.payload or {}).get("label") or str(key).split(":", 1)[-1].replace("_", " ")
        by[(key, clock.ist(e.at).weekday())].append(e.kind)
    out = []
    for (key, wd), kinds in by.items():
        last = kinds[-7:]
        skipped = sum(1 for k in last if k == "activity_skipped")
        if len(last) >= 5 and skipped >= 5:
            slug = str(key).split(":", 1)[-1]
            out.append({"key": f"learned:habit_{slug}_{WEEKDAYS[wd].lower()}",
                        "text": f"Usually skips {names[key]} on {WEEKDAYS[wd]}s ({skipped} of the last {len(last)}); Saheli asks about "
                                f"{WEEKDAYS[wd]} plans instead of reminding.",
                        "value": {"activity": key, "weekday": wd, "skipped": skipped, "of": len(last), "learned_on": clock.ist_day(now)}})
    return out


async def run_family(session: AsyncSession, family_id: str, people: list[str], *, now=None) -> dict:
    now = now or clock.now()
    out = {"learned": 0, "cooling": 0}
    for p in people:
        found = []
        t = await timing(session, family_id, p, now=now)
        if t:
            found.append(t)
        found += await habits(session, family_id, p, now=now)
        for f in found:
            res = await save_learned(session, family_id=family_id, subject_id=p, key=f["key"], value=f["value"], text=f["text"], now=now)
            out["learned"] += res in ("created", "superseded")
            out["cooling"] += res == "cooling"
    return out


async def avoid_now(session: AsyncSession, family_id: str, person: str, *, now=None) -> str | None:
    """Why not to start a conversation with them at this hour, from what was learned (for check-ins and nudges)."""
    f = await store.active_fact(session, family_id, person, TIMING_KEY)
    if not f:
        return None
    hour = clock.ist(now or clock.now()).hour
    if hour in set((f.value or {}).get("avoid_hours") or []):
        prefer = (f.value or {}).get("prefer_hours") or []
        return ("they rarely answer at this hour (learned); wait" + (f" until around {prefer[0] % 12 or 12} {'pm' if prefer[0] >= 12 else 'am'}" if prefer else ""))
    return None


async def new_since(session: AsyncSession, family_id: str, person: str, *, hours: int = 24, now=None) -> list[str]:
    """Learnings made in the last day, for the nightly brief ("I noticed … so I'll …")."""
    now = now or clock.now()
    rows = (await session.execute(select(CareEvent.summary).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == person, CareEvent.kind == "learned",
        CareEvent.at >= now - timedelta(hours=hours)))).all()
    return [r[0] for r in rows]
