"""Photos (plan §N4.3, §N4.4), through the router role `vision` (Gemini Flash in Mumbai in production).

- A meal photo → the items on the plate, a portion for each, and how sure it is. Always confirmed with the person before
  anything is saved (founder Q4).
- A label photo (the back of a pack) → the nutrition table per 100 g and per serving, the serving size, ingredients,
  allergens, and the barcode digits when they show. It is saved as the family's food, so next time needs no question.
"""

from __future__ import annotations

import logging

from app.llm import router
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.lookup import ask

logger = logging.getLogger(__name__)

MEAL_PROMPT = """You look at a photo someone sent to their family's care companion, to log what they ate.

Say only what is visible. List each food or drink on the plate or in the picture, in plain English with its preparation
("boiled egg", "chapati", "dal tadka", "tea with milk"), with how much is there as household units: quantity and unit
(one of: piece katori bowl plate cup glass tsp tbsp slice handful scoop pack serving ml g) and your estimate of its grams.
Count pieces you can see. confidence 0-1 for each item. If it is not food (a medicine strip, a report, a person), say so.
If a pack's nutrition label or a barcode is visible, say so.
name is the dish you think it is. also is one close dish it could be instead, or null. Chapati and paratha, poori and
bhatura, samosa and pakora, gulab jamun and lyangcha are the kind of pair that means.

HINTS (what the person or the companion said about it, may be empty): {hints}

Return only JSON: {"food": true, "items": [{"name": "...", "also": null, "quantity": 2, "unit": "piece", "grams": 100, "confidence": 0.8}],
"description": "one short line of what is on the plate", "label_visible": false, "barcode": null, "unsure": false}"""

# Used only when the full reading comes back empty (a cut-off answer, a refusal, or no JSON).
FALLBACK_PROMPT = """Name the main food in this photo.

Return only JSON: {"food": true, "items": [{"name": "the dish", "also": "one close alternative, or null", "quantity": 1, "unit": "piece", "grams": 80, "confidence": 0.5}]}
If it is not food, return {"food": false, "items": []}."""

LABEL_PROMPT = """You read the nutrition label on the photo of a food pack, exactly as printed.

Copy the numbers exactly from the nutrition table: per 100 g (or 100 ml) and per serving when both are printed, with the
serving size in grams. Convert energy in kJ to kcal only when kcal is not printed. Sodium in mg (if only salt is printed,
sodium = salt × 400 mg per g). Also the product name and brand, the ingredients line, allergens it declares, veg/non-veg
mark, the barcode digits if visible, and how many pieces or servings the pack holds when printed.
Leave out anything not printed; never guess a number.

Return only JSON: {"label": true, "name": "...", "brand": "...",
"per_100g": {"energy_kcal": 0, "protein_g": 0, "carbs_g": 0, "sugar_g": 0, "added_sugar_g": 0, "fibre_g": 0, "fat_g": 0,
"sat_fat_g": 0, "cholesterol_mg": 0, "sodium_mg": 0, "calcium_mg": 0, "iron_mg": 0},
"per_serving": {"energy_kcal": 0, "protein_g": 0}, "serving_g": 0, "servings_per_pack": 0, "pieces_per_pack": 0,
"ingredients": "...", "allergens": ["milk"], "veg": "veg|egg|nonveg|vegan", "barcode": null}
If the photo is not a nutrition label, return {"label": false}."""


def _images(images: list[dict] | None) -> list[dict]:
    return [i for i in images or [] if str(i.get("mime") or "").startswith("image/") and i.get("data")][:3]


async def read_meal(images: list[dict] | None, hints: str = "", *, essential: bool = True) -> dict | None:
    """{"food", "items": [{name, quantity, unit, grams, confidence}], "description", "label_visible", "barcode", "unsure"};
    None when there is no image or no model answered."""
    imgs = _images(images)
    if not imgs:
        return None
    prompt = MEAL_PROMPT.replace("{hints}", hints[:400] or "(none)")
    d = await _ask_meal(prompt, imgs, essential=essential, max_tokens=4000)
    if not _named(d):
        # The long answer was empty: a timeout, a refusal, or JSON cut off before the items. One shorter ask.
        d = await _ask_meal(FALLBACK_PROMPT, imgs, essential=essential, max_tokens=1500)
    if not isinstance(d, dict):
        return None
    items = []
    for it in d.get("items") or []:
        if not isinstance(it, dict) or not str(it.get("name") or "").strip():
            continue
        unit = it.get("unit") if it.get("unit") in P.UNITS else None
        also = str(it.get("also") or "").strip()[:80]
        items.append({"name": str(it["name"]).strip()[:120], **({"also": also} if also and also.lower() != "null" else {}),
                      "quantity": N.number(it.get("quantity")), "unit": unit,
                      "grams": N.number(it.get("grams")), "confidence": max(0.0, min(1.0, N.number(it.get("confidence")) or 0.5))})
    return {"food": bool(d.get("food", True)) and bool(items), "items": items[:12], "description": str(d.get("description") or "")[:300],
            "label_visible": bool(d.get("label_visible")), "barcode": _digits(d.get("barcode")), "unsure": bool(d.get("unsure"))}


async def _ask_meal(prompt: str, imgs: list[dict], *, essential: bool, max_tokens: int) -> dict:
    try:
        d = await ask("vision", prompt, "The photo is attached.", images=imgs, essential=essential, max_tokens=max_tokens)
    except router.AllModelsFailed:
        logger.warning("meal photo: no vision model answered")
        return {}
    return d if isinstance(d, dict) else {}


def _named(d) -> bool:
    if not isinstance(d, dict):
        return False
    return any(isinstance(it, dict) and str(it.get("name") or "").strip() for it in d.get("items") or [])


async def read_label(images: list[dict] | None, *, essential: bool = True) -> dict | None:
    """The pack's nutrition panel as a food dict (source "label"), or None when it is not a readable label."""
    imgs = _images(images)
    if not imgs:
        return None
    try:
        d = await ask("vision", LABEL_PROMPT, "The photo is attached.", images=imgs, essential=essential, max_tokens=2000)
    except router.AllModelsFailed:
        logger.warning("label photo: no vision model answered")
        return None
    if not isinstance(d, dict) or not d.get("label"):
        return None
    per = N.clean(d.get("per_100g"))
    serving = N.number(d.get("serving_g"))
    per_serving = N.clean(d.get("per_serving")) or None
    if not per and per_serving and serving:  # only a per-serving column is printed
        per = N.times(per_serving, 100.0 / serving)
    if "energy_kcal" not in per:
        return None
    pieces = N.number(d.get("pieces_per_pack")) or N.number(d.get("servings_per_pack"))
    units = {"serving": serving, "pack": serving * pieces if serving and pieces else None, "piece": serving if serving else None}
    return {"name": str(d.get("name") or "").strip()[:200] or "Packaged food", "brand": (str(d.get("brand") or "").strip()[:120] or None),
            "kind": "packaged", "per_100g": per, "per_serving": per_serving, "serving_g": serving,
            "units": P.clean_units({k: v for k, v in units.items() if v}), "default_unit": "serving" if serving else "g",
            "typical_qty": 1 if serving else 100, "countable": bool(serving),
            "ingredients": str(d.get("ingredients") or "")[:2000] or None,
            "allergens": [str(a).lower()[:30] for a in d.get("allergens") or [] if a][:12],
            "veg": d.get("veg") if d.get("veg") in ("veg", "egg", "nonveg", "vegan", "jain") else None,
            "barcode": _digits(d.get("barcode")), "source": "label", "pieces_per_pack": pieces}


def _digits(v) -> str | None:
    d = "".join(ch for ch in str(v or "") if ch.isdigit())
    return d if 8 <= len(d) <= 14 else None


def describe(reading: dict) -> str:
    """'2 boiled eggs, 1 cup tea with milk' from a meal reading (for the confirm question)."""
    parts = []
    for it in reading.get("items") or []:
        q = P.qty_label(it.get("quantity"), it.get("unit"))
        bit = f"{q} {it['name']}".strip()
        if it.get("also"):
            bit += f", or it might be {it['also']}"
        parts.append(bit)
    return ", ".join(parts) or (reading.get("description") or "")
