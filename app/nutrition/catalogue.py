"""Common Indian dishes and single foods, loaded before any estimate.

Dishes are the Indian Nutrient Databank (Jaacks et al., 2024): per 100 g as eaten, and the grams of one
household unit. Single foods are the Indian Food Composition Tables 2017 (Longvah et al., NIN): fruits,
vegetables, nuts and dairy, per 100 g. A name already here is that food; the model is not asked.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import logging

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.models import NutritionFood

logger = logging.getLogger(__name__)

_DISHES = Path(__file__).with_name("dishes.json")
_FOODS = Path(__file__).with_name("foods.json")
_DISH_URL = "https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-"
_FOOD_URL = "https://www.nin.res.in/"


def dishes() -> list[dict]:
    return json.loads(_DISHES.read_text())["dishes"]


def foods() -> list[dict]:
    return json.loads(_FOODS.read_text())["foods"]


def _row(name: str, per: dict, aliases: list, unit: str, grams: float, source: str, url: str, now) -> NutritionFood | None:
    if unit not in P.UNITS or "energy_kcal" not in per or not P.ok_unit_g(unit, grams):
        return None
    clean = sorted({str(a).strip().lower()[:120] for a in aliases if str(a).strip()})[:40]
    return NutritionFood(
        id=uuid.uuid4(), family_id=None, name=name[:200], aliases=clean, kind="generic",
        per_100g=per, serving_g=grams, units={unit: grams}, default_unit=unit,
        typical_qty=2 if unit == "piece" and grams <= 50 else (100 if unit == "g" else 1),
        countable=unit == "piece", tags=[], allergens=[], source=source, source_urls=[url], confidence=0.9,
        checked={"ok": True, "rough": False, "problems": []}, uses=0, created_at=now, updated_at=now,
    )


async def _load(session: AsyncSession, source: str, rows: list[NutritionFood]) -> None:
    """Insert the source's rows once. A shared food already in the catalogue under the same name (a web or model look-up
    saved before the databank came: live 17 Oct, "Chilli paneer") keeps its row; the rest go in. A failure here never
    fails the food save that called it (live 17-20 Oct: every log_food failed on that one name)."""
    n = await session.scalar(select(func.count()).select_from(NutritionFood).where(NutritionFood.source == source))
    if n or not rows:
        return
    have = {(a or "", b or "", c or "") for a, b, c in (await session.execute(select(
        func.lower(NutritionFood.name), func.lower(NutritionFood.brand), func.lower(NutritionFood.restaurant)).where(
        NutritionFood.family_id.is_(None)))).all()}
    seen: set[tuple[str, str, str]] = set()
    fresh = []
    for r in rows:
        key = (r.name.lower(), (r.brand or "").lower(), (r.restaurant or "").lower())
        if key in have or key in seen:
            continue
        seen.add(key)
        fresh.append(r)
    if not fresh:
        return
    try:
        async with session.begin_nested():
            session.add_all(fresh)
            await session.flush()
    except IntegrityError:  # another turn loaded it at the same moment
        logger.info("catalogue %s already loaded by another turn", source)


async def ensure(session: AsyncSession) -> None:
    """Insert the shared dishes and single foods once. A later call in the same database does nothing."""
    now = clock.now()
    dish_rows = []
    for dish in dishes():
        row = _row(str(dish["name"]), N.clean(dish.get("per_100g")), dish.get("aliases") or [],
                   dish["unit"], float(dish["grams"]), "indb", _DISH_URL, now)
        if row is not None:
            dish_rows.append(row)
    food_rows = []
    for food in foods():
        row = _row(str(food["name"]), N.clean(food.get("per_100g")), food.get("aliases") or [],
                   "g", 1.0, "ifct", _FOOD_URL, now)
        if row is not None:
            food_rows.append(row)
    await _load(session, "indb", dish_rows)
    await _load(session, "ifct", food_rows)
