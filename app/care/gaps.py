"""What the care record still lacks for the person speaking, so Saheli can ask one thing in the conversation.

The list is data: today's food log, saved facts, today's activities, and what was already asked (a check-in event or a
sent nudge). She picks one. A sent nudge today already covers a meal (field meal:<slot>), water (water:count) or sleep
(sleep:when). A meal or any water already in today's food log, or a sleep or routine event saved today, is not asked
again. gap_asked writes the same check-in record the scheduled asks read. Nothing here reads the person's words.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock

logger = logging.getLogger(__name__)

MAX_ITEMS = 4


def asked_keys(topics: list[str], fields: list[str]) -> set[str]:
    """Topic and field names already asked. water:glasses counts as water; activity:gym:done counts as the gym habit."""
    out: set[str] = set()
    for raw in [*topics, *fields]:
        if not raw:
            continue
        out.add(raw)
        out.add(raw.split(":")[0])
        parts = raw.split(":")
        if len(parts) >= 3 and parts[0] == "activity" and parts[1]:
            out.add("habit:" + parts[1])
    return out


def ledger_topics(key: str) -> list[str]:
    """The check-in topics a scheduled ask already reads as its field name, so the two never ask the same thing."""
    if key == "meal":
        return [key, "meal:day", "meal:breakfast", "meal:lunch", "meal:dinner", "meal:snack"]
    if key == "water":
        return [key, "water:count"]
    if key == "sleep":
        return [key, "sleep:when"]
    if key.startswith("habit:") and key.split(":", 1)[1]:
        name = key.split(":", 1)[1]
        return [key, f"activity:{name}:done", f"activity:{name}:plan"]
    return [key]


def nudge_covers(kind: str, field: str | None, sent_day: str, today: str) -> str | None:
    """The gap a sent nudge already covered today. meal:<slot>, water:count, or sleep:when. Any other row covers nothing."""
    if sent_day != today or not field:
        return None
    if kind == "meal" and field.startswith("meal:"):
        return "meal"
    if kind == "water" and field == "water:count":
        return "water"
    if kind == "sleep" and field == "sleep:when":
        return "sleep"
    return None


def food_gaps(had_meal: bool, had_water: bool, hour: int) -> list[tuple[str, str]]:
    if had_meal and not had_water:
        return [("water", "no water logged today, and a meal is")]
    if not had_meal and hour >= 14:
        return [("meal", "no meal logged today")]
    return []


def pick(items: list[tuple[str, str]], asked: set[str]) -> list[tuple[str, str]]:
    out = []
    for key, line in items:
        if key in asked or key.split(":")[0] in asked:
            continue
        out.append((key, line))
        if len(out) == MAX_ITEMS:
            break
    return out


def _water(payload: dict | None, nutrients: dict | None) -> bool:
    p = payload or {}
    n = nutrients or {}
    if p.get("meal") == "water" or float(p.get("water_ml") or 0) > 0 or float(n.get("water_ml") or 0) > 0:
        return True
    tags = p.get("tags") or []
    return "water" in tags


def render(name: str, items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    lines = "\n".join(f"- {key}: {line}" for key, line in items)
    return (f"NOT KNOWN YET for {name} (at most one question, at a natural moment; call gap_asked with the key):\n" + lines)


async def block(session: AsyncSession, family_id: str, person: str, name: str) -> str:
    """The short list for this turn. Empty during their night, and empty when nothing is missing or everything was asked today."""
    from app.brain.wake import quiet_for

    try:
        quiet, _ = await quiet_for(session, family_id, person)
    except Exception:  # noqa: BLE001
        logger.exception("gap night check failed family=%s", family_id)
        quiet = False
    if quiet:
        return ""
    try:
        async with session.begin_nested():
            items = await _items(session, family_id, person)
    except Exception:  # noqa: BLE001 — a gap list never costs the reply
        logger.exception("gap list failed family=%s", family_id)
        return ""
    return render(name or "them", items)


async def _items(session: AsyncSession, family_id: str, person: str) -> list[tuple[str, str]]:
    from app.care import activities, checkins, store
    from app.care.models import NudgeLedger
    from app.nutrition.models import NutritionEntry

    day = clock.ist_day()
    hour = clock.ist().hour
    events = await store.events(session, family_id, person, day=day, kinds=["meal"])
    entries = list((await session.execute(
        select(NutritionEntry).where(NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person,
                                      NutritionEntry.day == day, NutritionEntry.status == "active")
    )).scalars())
    had_water = any(_water(e.payload, None) for e in events) or any(_water(None, r.nutrients) for r in entries) or any(
        r.meal == "water" for r in entries)
    had_meal = any(not _water(e.payload, None) for e in events) or any(
        r.kind == "food" and r.meal != "water" and not _water(None, r.nutrients) for r in entries)
    rest = await store.events(session, family_id, person, day=day, kinds=["sleep", "routine"])
    facts = [f for f in await store.facts(session, family_id, person) if f.status == "active"]
    sleep = bool(rest) or any(f.key in ("activity:sleep", "routine:sleep") or (f.value or {}).get("type") == "sleep" for f in facts)
    preference = any(f.domain == "preference" for f in facts)
    habits = [("habit:" + r["key"].split(":")[-1], f"today's {r['label']} has no result yet")
              for r in await activities.status(session, family_id, person)
              if r.get("today") == "not_heard" and r.get("kind") == "habit"]
    since = clock.now() - timedelta(days=1)
    today_start = clock.ist().replace(hour=0, minute=0, second=0, microsecond=0)
    topics = [t for t, at, sent in await checkins._asked_rows(session, family_id, person) if sent and at >= since]
    sent_rows = list((await session.execute(
        select(NudgeLedger).where(NudgeLedger.family_id == family_id, NudgeLedger.person_id == person,
                                   NudgeLedger.status == "sent", NudgeLedger.sent_at >= today_start)
    )).scalars())
    fields = [covered for r in sent_rows if (covered := nudge_covers(r.kind, r.field, clock.ist_day(r.sent_at), day))]
    fields += [r.field for r in sent_rows if (r.field or "").split(":")[0] in ("preference", "habit", "activity")]
    candidates = [*food_gaps(had_meal, had_water, hour)]
    if not sleep:
        candidates.append(("sleep", "when they sleep is not on record"))
    candidates.extend(habits[:2])
    if not preference:
        candidates.append(("preference", "nothing saved about what they enjoy"))
    return pick(candidates, asked_keys(topics, fields))
