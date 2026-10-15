"""Units → grams (plan §N3.3). The brain passes a unit from a fixed list (structured data); each food carries its own grams
per unit, filled by the lookup and checked against the bounds here. Personal sizes ("my katori is big") win. Nothing
here reads people's words.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core import clock

UNITS = ("piece", "katori", "katori_small", "katori_large", "bowl", "plate", "cup", "glass", "tsp", "tbsp", "slice", "handful", "scoop",
         "pack", "serving", "ml", "g")
# Household containers: a person's own size (setting:nutrition sizes) replaces the food's or the usual one.
CONTAINERS = {"katori": "katori_ml", "katori_small": "katori_ml", "katori_large": "katori_ml", "bowl": "bowl_ml", "cup": "cup_ml",
              "glass": "glass_ml", "plate": "plate_g"}
SIZE_RATIO = {"katori_small": 0.66, "katori_large": 1.66}  # against their own katori
# The usual grams of a unit when the food does not say (Indian household measures).
DEFAULT_G = {"katori": 150.0, "katori_small": 100.0, "katori_large": 250.0, "bowl": 250.0, "plate": 300.0, "cup": 150.0, "glass": 250.0,
             "tsp": 5.0, "tbsp": 15.0, "slice": 30.0, "handful": 30.0, "scoop": 30.0, "ml": 1.0, "g": 1.0}
# What one unit of any food can weigh; a looked-up value outside is not used (an egg's piece is 35–70 g, a roti 25–80 g:
# both inside the piece bounds; the food's own lookup is checked again by the model's plausibility).
BOUNDS = {"piece": (1.0, 1200.0), "katori": (80.0, 300.0), "katori_small": (50.0, 170.0), "katori_large": (150.0, 450.0),
          "bowl": (120.0, 600.0), "plate": (120.0, 900.0), "cup": (60.0, 350.0), "glass": (120.0, 500.0), "tsp": (1.5, 10.0),
          "tbsp": (5.0, 25.0), "slice": (5.0, 120.0), "handful": (8.0, 80.0), "scoop": (10.0, 70.0), "pack": (2.0, 5000.0),
          "serving": (2.0, 2000.0), "ml": (0.9, 1.2), "g": (1.0, 1.0)}
PORTION = {"all": 1.0, "most": 0.75, "half": 0.5, "little": 0.25}
MEALS = ("breakfast", "lunch", "snack", "dinner", "after_dinner", "late_night", "other")
MAIN_MEALS = ("breakfast", "lunch", "dinner")
# The morning starts at 06:00. From 19:00 until midnight is dinner, or after dinner once dinner is saved.
# From midnight until 06:00 is still last night: late night once dinner is saved, otherwise that previous day's dinner.
DAY_START = 6 * 60
DINNER_FROM = 19 * 60


def ok_unit_g(unit: str, grams: float | None) -> bool:
    lo, hi = BOUNDS.get(unit, (0.5, 5000.0))
    return grams is not None and lo <= grams <= hi


def clean_units(units: dict | None) -> dict:
    """The food's grams per unit, keeping only known units inside their bounds."""
    out: dict = {}
    for u, g in (units or {}).items():
        try:
            gf = float(g)
        except (TypeError, ValueError):
            continue
        if u in UNITS and ok_unit_g(u, gf):
            out[u] = round(gf, 1)
    return out


@dataclass
class Portion:
    grams: float | None
    estimate: bool  # the grams are a usual portion, not what they said
    how: str  # "food unit" | "personal size" | "usual unit" | "serving" | "usual portion" | "none"


def grams_for(food, quantity: float | None, unit: str | None, sizes: dict | None = None) -> Portion:
    """Grams eaten: quantity × grams per unit. No quantity: the food's usual portion, marked an estimate."""
    sizes = sizes or {}
    units = dict(getattr(food, "units", None) or {})
    serving = getattr(food, "serving_g", None)
    estimate = quantity is None
    qty = float(quantity) if quantity is not None else float(getattr(food, "typical_qty", None) or 1.0)
    u = unit if unit in UNITS else None
    if u is None:
        u = getattr(food, "default_unit", None) if getattr(food, "default_unit", None) in UNITS else None
    if u in ("g",):
        return Portion(round(qty, 1), estimate, "said")
    if u == "ml":
        return Portion(round(qty, 1), estimate, "said")
    if u in CONTAINERS and sizes.get(CONTAINERS[u]):
        try:
            base = float(sizes[CONTAINERS[u]])
        except (TypeError, ValueError):
            base = 0.0
        if base > 0:
            return Portion(round(qty * base * SIZE_RATIO.get(u, 1.0), 1), estimate, "personal size")
    if u and u in units:
        return Portion(round(qty * float(units[u]), 1), estimate, "food unit")
    if u in ("pack", "serving", "piece") and serving:
        return Portion(round(qty * float(serving), 1), estimate, "serving")
    if u in DEFAULT_G:
        return Portion(round(qty * DEFAULT_G[u], 1), estimate, "usual unit")
    if serving:
        return Portion(round(qty * float(serving), 1), True, "usual portion")
    if units:
        first = next(iter(units.values()))
        return Portion(round(qty * float(first), 1), True, "usual portion")
    return Portion(None, True, "none")


@dataclass(frozen=True)
class Slot:
    """Where a log belongs: the meal name, and the diary day (which can be the previous date)."""

    meal: str
    day: str


def place(at: datetime, *, dinner_saved: bool = False, only_water: bool = False) -> Slot:
    """The slot for a clock time. A named meal does not decide this; dinner_saved is whether that food-day already has a dinner.

    Water never becomes a dinner. Before 06:00 it is late night. After 19:00, with no dinner yet, it is not called dinner.
    """
    local = clock.ist(at)
    minute = local.hour * 60 + local.minute
    day = local.date()
    if minute < DAY_START:
        if dinner_saved or only_water:
            return Slot("late_night", day.isoformat())
        return Slot("dinner", (day - timedelta(days=1)).isoformat())
    if minute >= DINNER_FROM:
        if only_water and not dinner_saved:
            return Slot("other", day.isoformat())
        if dinner_saved:
            return Slot("after_dinner", day.isoformat())
        return Slot("dinner", day.isoformat())
    if minute < 11 * 60:
        return Slot("breakfast", day.isoformat())
    if minute < 15 * 60 + 30:
        return Slot("lunch", day.isoformat())
    return Slot("snack", day.isoformat())


def resolve(at: datetime, *, dinner_saved: bool = False, only_water: bool = False, named: str | None = None) -> Slot:
    """Night and late evening follow the clock. In the daytime a meal the brain named is kept."""
    placed = place(at, dinner_saved=dinner_saved, only_water=only_water)
    local = clock.ist(at)
    minute = local.hour * 60 + local.minute
    if minute < DAY_START or minute >= DINNER_FROM:
        return placed
    if named in ("breakfast", "lunch", "snack", "dinner", "other"):
        return Slot(named, placed.day)
    return placed


def meal_for(at: datetime) -> str:
    """The meal slot a time falls in (IST) when nothing is known about dinner yet: a time, not words."""
    return place(at).meal


def qty_label(quantity: float | None, unit: str | None) -> str:
    if quantity is None:
        return ""
    q = int(quantity) if float(quantity).is_integer() else round(float(quantity), 2)
    if not unit or unit == "piece":
        return f"{q}"
    word = {"katori_small": "small katori", "katori_large": "large katori", "tsp": "tsp", "tbsp": "tbsp", "g": "g", "ml": "ml"}.get(unit, unit)
    return f"{q} {word}"
