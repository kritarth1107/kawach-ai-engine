"""Common Indian dishes, loaded before any estimate.

Values are the Indian Nutrient Databank (Jaacks et al., 2024): per 100 g as eaten, and the grams of one
household unit of that dish. The authors posted the table as publicly and freely available. A name already
here is that dish; the model is not asked.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.models import NutritionFood

_PATH = Path(__file__).with_name("dishes.json")
_URL = "https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-"


def dishes() -> list[dict]:
    return json.loads(_PATH.read_text())["dishes"]


async def ensure(session: AsyncSession) -> None:
    """Insert the shared dishes once. A later call in the same database does nothing."""
    n = await session.scalar(select(func.count()).select_from(NutritionFood).where(NutritionFood.source == "indb"))
    if n:
        return
    now = clock.now()
    rows = []
    for dish in dishes():
        unit = dish["unit"] if dish["unit"] in P.UNITS else None
        grams = float(dish["grams"])
        per = N.clean(dish.get("per_100g"))
        if unit is None or "energy_kcal" not in per or not P.ok_unit_g(unit, grams):
            continue
        aliases = sorted({str(a).strip().lower()[:120] for a in dish.get("aliases") or [] if str(a).strip()})[:40]
        rows.append(NutritionFood(
            id=uuid.uuid4(), family_id=None, name=str(dish["name"])[:200], aliases=aliases, kind="generic",
            per_100g=per, serving_g=grams, units={unit: grams}, default_unit=unit,
            typical_qty=2 if unit == "piece" and grams <= 50 else 1, countable=unit == "piece",
            tags=[], allergens=[], source="indb", source_urls=[_URL], confidence=0.9,
            checked={"ok": True, "rough": False, "problems": []}, uses=0, created_at=now, updated_at=now,
        ))
    session.add_all(rows)
    await session.flush()
