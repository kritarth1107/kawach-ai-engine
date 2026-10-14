"""Repeated problems, grouped for the doctor (founder 2026-10-13).

"Let's say I log symptoms of a fall across three different timelines. The common data point is that I fell: the first
time because I didn't see the dog, the second time I was probably dehydrated, the third time I came out of the shower.
When I go to a doctor, which of these would the care brief pull?" Before this, the report kept only the last six symptom
lines of its window (7 days by default), one by one, so two of the three falls never reached the doctor and nothing said
they were one pattern.

Now every symptom, outcome (a fall, a hospital visit) and red flag of the last 90 days is read by a model that groups the
ones that are the same kind of problem, keeps each one's date and circumstances in the person's own terms, and writes
one neutral question for the doctor. The grouping is the model's judgement over structured events, never word lists.
Cached per subject until a new event arrives.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import CareEvent
from app.core import clock

logger = logging.getLogger(__name__)

LOOK_BACK_DAYS = 90
KINDS = ("symptom", "outcome", "red_flag")

PROMPT = """You prepare the "repeated problems" part of a care brief that a family takes to the doctor.

You get every health problem logged for one person in the last months: symptoms, outcomes (falls, hospital or emergency
visits) and red flags, each with its date and the words it was logged with.

Group the ones that are the same kind of problem (several falls, recurring dizziness, repeated stomach trouble, headaches
coming back). A group needs at least 2 events. For each group keep EVERY event: its date and what happened, with the
circumstances exactly as logged (where, why, what they were doing), never merged into one line. Then one short, neutral
question the family could ask the doctor about the pattern (for falls, e.g. whether balance, blood pressure on standing
or hydration should be checked). No diagnosis, no advice, nothing that is not in the events.

Return only JSON: {"patterns": [{"theme": "Falls", "count": 3, "events": [{"date": "YYYY-MM-DD", "what": "Fell after not
seeing the dog"}], "ask_doctor": "..."}]} — an empty list when nothing repeats."""


async def events(session: AsyncSession, family_id: str, subject_id: str, *, days: int = LOOK_BACK_DAYS, now=None) -> list[CareEvent]:
    now = now or clock.now()
    return list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind.in_(KINDS),
                                CareEvent.payload["forgotten"].astext.is_(None),
                                CareEvent.at >= now - timedelta(days=days)).order_by(CareEvent.at)
    )).scalars())


def _parse(raw: str) -> list[dict]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return []
    out = []
    for p in data.get("patterns") or []:
        if not isinstance(p, dict):
            continue
        evs = [{"date": str(e.get("date") or "")[:10], "what": str(e.get("what") or "")[:240]}
               for e in (p.get("events") or []) if isinstance(e, dict) and e.get("what")]
        if len(evs) < 2:
            continue
        out.append({"theme": str(p.get("theme") or "Repeated problem")[:60], "count": len(evs), "events": evs,
                    "askDoctor": str(p.get("ask_doctor") or "")[:300]})
    return out


async def patterns(session: AsyncSession, family_id: str, subject_id: str, *, days: int = LOOK_BACK_DAYS, now=None) -> list[dict]:
    """[{theme, count, events: [{date, what}], askDoctor}] for problems that came back in the last `days` days."""
    evs = await events(session, family_id, subject_id, days=days, now=now)
    if len(evs) < 2:
        return []
    ref = f"sympat:{subject_id}:{days}:{len(evs)}:{evs[-1].id}"
    cached = (await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == "symptom_patterns", CareEvent.ref == ref)
    )).scalars().first()
    if cached is not None:
        return list((cached.payload or {}).get("patterns") or [])
    lines = [f"{clock.ist_day(e.at)} [{e.kind}] {e.summary[:300]}" for e in evs]
    try:
        from app.llm import router

        reply = await router.complete("extract", system_stable=PROMPT, effort="low", max_tokens=2000, timeout_s=40,
                                      messages=[{"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}]}])
        found = _parse(reply.text or "")
    except Exception:  # noqa: BLE001 — the brief still lists every symptom without the grouping
        logger.warning("symptom patterns failed family=%s", family_id)
        return []
    await store.record_event(session, family_id=family_id, subject_id=subject_id, kind="symptom_patterns",
                             summary=f"{len(found)} repeated problem(s) in {days} days", payload={"patterns": found}, ref=ref)
    return found


def text(found: list[dict]) -> str:
    """The grouped problems as lines for a brief or a report."""
    out = []
    for p in found:
        dated = "; ".join(f"{e['date']}: {e['what']}" for e in p["events"])
        out.append(f"{p['theme']} ({p['count']} times): {dated}" + (f". Ask: {p['askDoctor']}" if p.get("askDoctor") else ""))
    return "\n".join(out)
