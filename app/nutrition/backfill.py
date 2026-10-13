"""Backfill (plan §N11): recent meals Saheli noted as plain `log_event kind=meal` events become food-log entries.

- Quantities as they were said, otherwise `unquantified` (no numbers made up); `source=backfill`, `source_ref=event:<id>`.
- A model reads each note into structured items (what was eaten, the number when the note says one, or that the meal was
  skipped); code never reads the words. Foods then go through the normal lookup chain.
- It asks nobody anything. A dry run (the default) lists what it would write.
- Safe to run again: an event already turned into entries is skipped. People with tracking off are skipped.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import CareEvent
from app.core import clock
from app.llm import router
from app.nutrition import entries, lookup, settings
from app.nutrition import portions as P
from app.nutrition.models import NutritionEntry

logger = logging.getLogger(__name__)

PROMPT = """Each line is a short note a care companion wrote about a meal someone had. For each note, list the foods eaten:
name in plain English with its preparation, and quantity and unit only when the note states a number (one of: piece katori
bowl plate cup glass tsp tbsp slice handful scoop pack serving ml g). If the note says the meal was skipped or nothing
was eaten, skipped = true and no items. Never add a food or a number the note does not say.

Return only JSON: {"notes": [{"id": 0, "skipped": false, "items": [{"name": "boiled egg", "quantity": 4, "unit": "piece"}]}]}"""


async def _read(rows: list[CareEvent]) -> dict[int, dict]:
    if not rows:
        return {}
    body = json.dumps({"notes": [{"id": e.id, "note": (e.summary or "")[:300]} for e in rows]}, ensure_ascii=False)
    try:
        got = await lookup.ask("nutrition", PROMPT, body, essential=False, max_tokens=4000)
    except router.AllModelsFailed:
        return {}
    out = {}
    for n in (got.get("notes") if isinstance(got, dict) else None) or []:
        if isinstance(n, dict) and isinstance(n.get("id"), int):
            out[n["id"]] = n
    return out


async def family(session: AsyncSession, family_id: str, *, days: int = 7, apply: bool = False) -> list[dict]:
    since = clock.now() - timedelta(days=days)
    events = [e for e in (await session.execute(select(CareEvent).where(
        CareEvent.family_id == family_id, CareEvent.kind == "meal", CareEvent.at >= since).order_by(CareEvent.at))).scalars()
        if not (e.payload or {}).get("nutrition") and not (e.payload or {}).get("forgotten")]
    if not events:
        return []
    done = set((await session.execute(select(NutritionEntry.source_ref).where(
        NutritionEntry.family_id == family_id, NutritionEntry.source_ref.in_([f"event:{e.id}" for e in events])))).scalars())
    todo = []
    for e in events:
        if f"event:{e.id}" in done:
            continue
        if not settings.tracked(await settings.get(session, family_id, e.subject_id)):
            continue
        todo.append(e)
    read: dict[int, dict] = {}
    for i in range(0, len(todo), 25):
        read.update(await _read(todo[i:i + 25]))
    out = []
    for e in todo:
        r = read.get(e.id) or {}
        items = [{"name": str(i.get("name"))[:120], **({"quantity": i["quantity"]} if isinstance(i.get("quantity"), (int, float)) and i["quantity"] > 0 else {}),
                  **({"unit": i["unit"]} if i.get("unit") in P.UNITS else {})} for i in r.get("items") or [] if isinstance(i, dict) and i.get("name")]
        plan = {"event": e.id, "subject": e.subject_id, "at": clock.ist(e.at).strftime("%Y-%m-%d %H:%M"), "note": (e.summary or "")[:200],
                "skipped": bool(r.get("skipped")), "items": items}
        if not r:
            plan["why"] = "not read (model unavailable)"
        elif apply and (items or r.get("skipped")):
            by = e.actor_id or "saheli"
            if r.get("skipped") and not items:
                row = await entries.save_skipped(session, family_id=family_id, subject_id=e.subject_id, logged_by=by, meal=P.meal_for(e.at),
                                                 eaten_at=e.at, source="backfill", source_ref=f"event:{e.id}")
                plan["written"] = [str(row.id)]
            else:
                found = await lookup.resolve(session, family_id, items, essential=False, no_questions=True)
                rows = await entries.save_meal(
                    session, family_id=family_id, subject_id=e.subject_id, logged_by=by,
                    items=[entries.Item(name=it["name"], food=f.food if not f.medicine else None, quantity=it.get("quantity"), unit=it.get("unit"),
                                        confidence="said" if it.get("quantity") else "unquantified") for it, f in zip(items, found) if not f.medicine],
                    meal=P.meal_for(e.at), eaten_at=e.at, source="backfill", source_ref=f"event:{e.id}", reason="from an earlier meal note")
                plan["written"] = [entries.line(x) + (f" ({x.nutrients.get('energy_kcal', 0):.0f} kcal)" if x.nutrients else "") for x in rows]
        out.append(plan)
    return out


async def run(sessions, *, days: int = 7, apply: bool = False, family_prefix: str = "") -> dict:
    since = clock.now() - timedelta(days=days)
    async with sessions() as session:
        fids = sorted({f for f in (await session.execute(select(CareEvent.family_id).where(
            CareEvent.kind == "meal", CareEvent.at >= since).distinct())).scalars()
            if f.startswith(family_prefix) and not f.startswith(("shadow:", "sim-"))})
    out: dict = {}
    for fid in fids:
        try:
            async with sessions() as session:
                got = await family(session, fid, days=days, apply=apply)
                if apply:
                    await session.commit()
            if got:
                out[fid[:8]] = got
        except Exception as exc:  # noqa: BLE001 — one family must not stop the rest
            logger.exception("nutrition backfill failed family=%s", fid)
            out[fid[:8]] = [{"error": str(exc)[:160]}]
    return {"families": len(fids), "applied": apply, "days": days, "meals": out}
