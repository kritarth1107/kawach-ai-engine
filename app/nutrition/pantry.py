"""Food at home (plan §N6.2, founder Q5): delivered grocery orders go into the pantry with counts and the pack's nutrition;
nothing is logged as eaten. When they later say "had a protein bar", log_food uses the pantry item (exact nutrition) and
the count goes down. People can also say what they bought ("6 protein bars at home") with the pantry tool.

Order Engineer calls `add_from_order(session, task)` when a grocery order is delivered (INTERFACES.md §1.1).
"""

from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.llm import router
from app.nutrition import nutrients as N
from app.nutrition.lookup import ask, save_food
from app.nutrition.models import NutritionFood, NutritionPantry

logger = logging.getLogger(__name__)

PACK_PROMPT = """You read grocery listings from an Indian store. For each, say how many separately eaten pieces one pack holds
and what one piece is: "Protein bar, pack of 6" → 6 piece; "Eggs, 12 pcs" → 12 piece; "Amul milk 500 ml" → 1 pack;
"Atta 5 kg" → 1 pack. Also whether it is food (false for soap, detergent, medicine).

Return only JSON: {"items": [{"i": 0, "pieces": 6, "unit": "piece|pack", "food": true}]}"""


def pantry_json(p: NutritionPantry) -> dict:
    n = p.per_unit_nutrients or {}
    return {"id": str(p.id), "name": p.name, "brand": p.brand, "count": p.count, "unit": p.unit, "unitG": p.unit_g,
            "perUnit": {"kcal": N.kcal(n), "protein_g": N.grams_of(n, "protein_g")} if n else None, "fromTask": p.task_id,
            "source": p.source, "subjectId": p.subject_id, "addedAt": p.added_at.isoformat(),
            "bestBefore": p.best_before.isoformat() if p.best_before else None, "status": p.status}


async def in_stock(session: AsyncSession, family_id: str, subject_id: str | None = None) -> list[NutritionPantry]:
    """What is at home for this person: their own and the household's (subject NULL)."""
    q = select(NutritionPantry).where(NutritionPantry.family_id == family_id, NutritionPantry.status == "in_stock", NutritionPantry.count > 0)
    if subject_id:
        q = q.where(or_(NutritionPantry.subject_id == subject_id, NutritionPantry.subject_id.is_(None)))
    return list((await session.execute(q.order_by(NutritionPantry.added_at.desc()).limit(40))).scalars())


def per_unit(food: NutritionFood | None, unit_g: float | None) -> dict:
    if food is None or not unit_g:
        return {}
    return N.scale(food.per_100g or {}, unit_g)


async def add(session: AsyncSession, *, family_id: str, subject_id: str | None, name: str, count: float, unit: str = "piece",
              brand: str | None = None, source: str = "said", task_id: str | None = None, food: NutritionFood | None = None,
              unit_g: float | None = None) -> NutritionPantry:
    """Add to what is at home. The same item (same name and owner, no order) already in stock gets the count added."""
    now = clock.now()
    if task_id is None:
        same = (await session.execute(select(NutritionPantry).where(
            NutritionPantry.family_id == family_id, NutritionPantry.status == "in_stock",
            func.lower(NutritionPantry.name) == name.strip().lower(),
            NutritionPantry.subject_id.is_(None) if subject_id is None else NutritionPantry.subject_id == subject_id).limit(1))).scalar_one_or_none()
        if same is not None:
            same.count = round((same.count or 0) + count, 2)
            same.updated_at = now
            await session.flush()
            return same
    if food is not None and not unit_g:
        unit_g = (food.units or {}).get("piece") or food.serving_g
    row = NutritionPantry(id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, food_id=food.id if food else None, name=name.strip()[:200],
                          brand=(brand or None) and brand[:120], count=round(count, 2), unit=unit if unit in ("piece", "pack") else "piece",
                          unit_g=unit_g, per_unit_nutrients=per_unit(food, unit_g), task_id=task_id, source=source, added_at=now,
                          updated_at=now, status="in_stock")
    session.add(row)
    await session.flush()
    return row


async def use(session: AsyncSession, row: NutritionPantry, count: float = 1.0) -> NutritionPantry:
    row.count = round(max(0.0, (row.count or 0) - count), 2)
    if row.count <= 0:
        row.status = "finished"
    row.updated_at = clock.now()
    await session.flush()
    return row


async def remove(session: AsyncSession, row: NutritionPantry) -> NutritionPantry:
    row.status, row.updated_at = "removed", clock.now()
    await session.flush()
    return row


async def find(session: AsyncSession, family_id: str, pantry_id: str) -> NutritionPantry | None:
    try:
        row = await session.get(NutritionPantry, uuid.UUID(str(pantry_id)))
    except ValueError:
        return None
    return row if row and row.family_id == family_id else None


# ── from a delivered grocery order ───────────────────────────────────────────


def _order_items(task) -> list[dict]:
    from app.specialists import channels

    r, d = task.result or {}, task.details or {}
    placed = [i for i in r.get("items") or [] if isinstance(i, dict) and i.get("name")] if (d.get("cart_fp") or r.get("placed")) else []
    if placed:
        return [{"name": str(i["name"]), "qty": int(i.get("qty") or 1), "pack": i.get("pack"), "brand": i.get("brand"),
                 "nutrition": i.get("nutrition"), "barcode": i.get("barcode"), "units_per_pack": i.get("units_per_pack")} for i in placed]
    asked = d.get("items") or []
    return [{"name": str(c.get("name")), "qty": channels._qty_for(asked, c), "pack": c.get("pack"), "brand": c.get("brand"),
             "nutrition": c.get("nutrition"), "barcode": c.get("barcode"), "units_per_pack": c.get("units_per_pack")}
            for c in d.get("chosen") or [] if isinstance(c, dict) and c.get("name")]


async def _pieces(items: list[dict]) -> list[dict]:
    """How many pieces one pack holds, read from the listing by a model (store data, never people's words)."""
    need = [i for i, it in enumerate(items) if not it.get("units_per_pack")]
    out = [{"pieces": N.number(it.get("units_per_pack")) or None, "unit": "piece" if it.get("units_per_pack") else None, "food": True}
           for it in items]
    if not need:
        return out
    body = json.dumps({"items": [{"i": i, "name": items[i]["name"], "pack": items[i].get("pack")} for i in need]}, ensure_ascii=False)
    try:
        got = await ask("nutrition", PACK_PROMPT, body, essential=False, max_tokens=800)
    except router.AllModelsFailed:
        return out
    for r in (got.get("items") if isinstance(got, dict) else None) or []:
        if isinstance(r, dict) and isinstance(r.get("i"), int) and 0 <= r["i"] < len(items):
            out[r["i"]] = {"pieces": N.number(r.get("pieces")), "unit": r.get("unit") if r.get("unit") in ("piece", "pack") else "pack",
                           "food": r.get("food") is not False}
    return out


async def add_from_order(session: AsyncSession, task, *, items: list[dict] | None = None) -> dict:
    """A delivered grocery order → the pantry (safe to call again: one row per order item)."""
    from app.tasks.skills import SKILLS

    if task.kind != "order" or SKILLS.get(task.service, {}).get("kind") != "grocery":
        return {"added": [], "kept": [], "skipped": "not a grocery order"}
    rows = items if items is not None else _order_items(task)
    if not rows:
        return {"added": [], "kept": [], "skipped": "no items on the order"}
    packs = await _pieces(rows)
    added, kept = [], []
    for it, pk in zip(rows, packs):
        if not pk.get("food", True):
            continue
        name = str(it.get("name") or "").strip()[:200]
        have = (await session.execute(select(NutritionPantry).where(
            NutritionPantry.family_id == task.family_id, NutritionPantry.task_id == str(task.id),
            func.lower(NutritionPantry.name) == name.lower()))).scalar_one_or_none()
        if have is not None:
            kept.append({"id": str(have.id), "name": have.name, "count": have.count, "unit": have.unit})
            continue
        food = None
        nut = it.get("nutrition") or {}
        if isinstance(nut, dict) and N.clean(nut.get("per_100g")).get("energy_kcal") is not None:
            food = await save_food(session, {"name": name, "brand": it.get("brand"), "kind": "packaged", "per_100g": nut.get("per_100g"),
                                             "per_serving": nut.get("per_serving"), "serving_g": nut.get("serving_g"),
                                             "units": {"serving": nut.get("serving_g")} if nut.get("serving_g") else {},
                                             "default_unit": "serving" if nut.get("serving_g") else None, "source": "listing",
                                             "barcode": it.get("barcode")}, family_id=task.family_id, alias=name)
        pieces = pk.get("pieces") or 1
        unit = "piece" if pk.get("unit") == "piece" and pieces else "pack"
        count = float(it.get("qty") or 1) * (pieces if unit == "piece" else 1)
        row = await add(session, family_id=task.family_id, subject_id=task.subject_id, name=name, brand=it.get("brand"), count=count,
                        unit=unit, source="order", task_id=str(task.id), food=food)
        added.append({"id": str(row.id), "name": row.name, "count": row.count, "unit": row.unit})
    return {"added": added, "kept": kept, "skipped": None}
