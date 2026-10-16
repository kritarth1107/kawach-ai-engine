"""The nutrition lookup chain (plan §N3): accurate, cited, cached. It stops at the first good answer.

1. The family's own foods (home recipes, packs they sent a label for) and their pantry.
2. The generic cache, matched by an AI-normalised name: one cheap model call reads every item of a meal against the
   candidates the database found and returns candidate ids (code only compares ids). A name the AI matched before is
   kept as an alias, so the same words find the same food next time without a call.
3. Packaged food with a barcode or a brand: Open Food Facts (only the barcode or the product name is sent).
4. Gemini in Mumbai grounded on Google Search (app.brain.websearch.grounded): per-100 g nutrients, units in grams,
   allergens and source URLs, as JSON. IFCT 2017 (NIN-ICMR) for Indian foods, USDA FoodData Central, brand sites and
   restaurant menus are named in the prompt.
5. A model estimate without sources, marked as an estimate (home dishes nobody has published).

Code then checks the data (app.nutrition.nutrients.check_per_100g, app.nutrition.portions bounds). A failed check is
kept rough and refined after the person has been told; the hot path does not search a second time. Generic foods hold
no personal data and are shared by every family.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field

import httpx
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.llm import router
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.models import NutritionFood, NutritionPantry

logger = logging.getLogger(__name__)

# One grounded search, shared by every place it tries. Longer than this, the meal is saved from an estimate and the
# search result replaces it afterwards (the person is not left waiting minutes).
WEB_S = 12.0

KINDS = ("generic", "packaged", "restaurant", "recipe")
TAGS = ("egg", "tea", "coffee", "water", "fruit", "vegetable", "sweet", "sugar", "dairy", "fried", "alcohol", "sugary_drink", "grain",
        "pulse", "meat", "fish", "snack", "nuts")
VEG = ("veg", "egg", "nonveg", "vegan", "jain")
SOURCES = ("ifct", "usda", "brand", "restaurant", "web")
CANDIDATES = 6


class Slow(Exception):
    """The look-up needs more time than the turn has: the job goes on in the background."""


def search_location() -> str:
    return os.getenv("NUTRITION_SEARCH_LOCATION", "asia-south1")


# ── model calls ──────────────────────────────────────────────────────────────


def parse_json(raw: str):
    """The first JSON object or array in a model's answer ({} when none)."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = text.find(opener), text.rfind(closer)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    return {}


async def ask(role: str, system: str, user: str, *, images: list[dict] | None = None, essential: bool = True, max_tokens: int = 3000,
              timeout_s: float = 30.0):
    content: list[dict] = [{"type": "text", "text": user}]
    content += [{"type": "image", "mime": i["mime"], "data": i["data"]} for i in images or [] if i.get("data")]
    reply = await router.complete(role, system_stable=system, messages=[{"role": "user", "content": content}], max_tokens=max_tokens,
                                  effort="low", timeout_s=timeout_s, essential=essential)
    return parse_json(reply.text)


NORMALISE_PROMPT = """You are the food normaliser of a nutrition tracker for Indian families.

ITEMS are foods someone ate (any language, already put in English where the speaker could). For each item, CANDIDATES
lists foods already in the catalogue with their ids (the family's own foods, their pantry, the shared catalogue).

For each item decide:
- match: the id of the candidate that is the same food as eaten (same dish and preparation; for packs the same product
  and brand; a pantry item when it is what they ate). null when none is the same food. A boiled egg is not an omelette;
  plain dosa is not masala dosa; a restaurant's biryani matches a candidate from that restaurant first.
- canonical: the food's plain English catalogue name with its preparation, e.g. "Egg, whole, boiled", "Chapati (whole wheat)",
  "Dal, toor, cooked", "Masala dosa", "Tea with milk and sugar". No person's name in it.
- kind: generic (a common food or dish), packaged (a branded pack), restaurant (a dish from a named restaurant), recipe (a
  dish made a particular way at home that is named after a person or the family, e.g. "Mummy's dal").
- medicine: true for a medicine, tablet or supplement capsule (never food). A protein powder or a health drink is food.
- needs: "label" when it is a packaged product with no brand or product name given and no candidate matches; else null.
  Never "which". When several kinds exist, put the common one in canonical (plain dosa, mixed namkeen, a regular laddoo)
  and leave needs null. The common one is saved, and they can say it was a different one.
- options: leave empty.
- countable: true when it is eaten in whole pieces people count (eggs, rotis, idlis, biscuits, bananas, bars).
- default_unit: one of: piece katori bowl plate cup glass tsp tbsp slice handful scoop pack serving ml g.
- typical_qty: a usual portion in default_unit for one adult (e.g. 2 for rotis, 1 for a katori of dal).
- tags: any that fit from: egg tea coffee water fruit vegetable sweet sugar dairy fried alcohol sugary_drink grain pulse meat fish snack nuts.

Return only JSON: {"items": [{"i": 0, "match": "id or null", "canonical": "...", "kind": "...", "medicine": false,
"needs": null, "options": [], "countable": false, "default_unit": "...", "typical_qty": 1, "tags": []}]}"""


PARTS_PROMPT = """You decide whether these items are one dish or several foods.

ITEMS are already in English. Two or more items that are flavours, fillings, or varieties of the same dish are that one dish, even when each name sounds like a whole food. Keep one of them: set part_of on every other item to that item's index, and leave part_of null on the one you keep. One amount stated for the mix is one portion, including when that same amount was copied onto more than one item. A filling named beside its dish is that dish.

Two different foods eaten together stay separate, each with part_of null. Never set part_of to an item's own index.

Return only JSON: {"items": [{"i": 0, "part_of": null}]}"""


def merge_parts(items: list[dict], rows: list) -> list[dict]:
    """Drop items the model said only describe another item. One stated amount stays one portion, on the dish."""
    n = len(items)
    parent: dict[int, int] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        i, p = row.get("i"), row.get("part_of")
        if isinstance(i, int) and isinstance(p, int) and i != p and 0 <= i < n and 0 <= p < n:
            parent[i] = p

    def root(i: int) -> int:
        seen: set[int] = set()
        while i in parent:
            if i in seen:
                return i
            seen.add(i)
            i = parent[i]
        return i

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = root(i)
        if r in parent:  # a cycle: leave every item in it as its own food
            groups[i] = [i]
            continue
        groups.setdefault(r, []).append(i)
    if len(groups) == n:
        return items
    out: list[dict] = []
    for r, members in groups.items():
        dish = dict(items[r])
        parts = [items[i] for i in members if i != r]
        extra = [str(p.get("name") or "").strip() for p in parts if str(p.get("name") or "").strip()]
        if extra:
            dish["name"] = ", ".join([str(dish.get("name") or "").strip(), *extra]).strip(", ")
        if not dish.get("quantity") and not dish.get("_grams"):
            stated = next((p for p in parts if p.get("quantity") or p.get("_grams")), None)
            if stated is not None:
                if stated.get("quantity"):
                    dish["quantity"] = stated["quantity"]
                if stated.get("unit"):
                    dish["unit"] = stated["unit"]
                if stated.get("_grams"):
                    dish["_grams"] = stated["_grams"]
        out.append(dish)
    return out


async def fold_descriptions(items: list[dict], *, essential: bool = True, timeout_s: float = 4.0) -> list[dict]:
    """A flavour or a filling named beside a dish becomes part of that dish. The items stay as given when the model fails."""
    if len(items) < 2:
        return items
    body = json.dumps({"ITEMS": [{"i": i, "name": it.get("name"), **({"quantity": it.get("quantity")} if it.get("quantity") else {}),
                                  **({"unit": it.get("unit")} if it.get("unit") else {})} for i, it in enumerate(items)]},
                      ensure_ascii=False)
    try:
        got = await ask("nutrition", PARTS_PROMPT, body, essential=essential, max_tokens=2000, timeout_s=timeout_s)
    except router.AllModelsFailed:
        return items
    rows = got.get("items") if isinstance(got, dict) else got
    return merge_parts(items, rows if isinstance(rows, list) else [])


FOOD_JSON = """{"name": "plain English name with preparation", "kind": "generic|packaged|restaurant|recipe",
 "per_100g": {"energy_kcal": 0, "protein_g": 0, "carbs_g": 0, "sugar_g": 0, "added_sugar_g": 0, "fibre_g": 0, "fat_g": 0,
              "sat_fat_g": 0, "cholesterol_mg": 0, "sodium_mg": 0, "potassium_mg": 0, "calcium_mg": 0, "iron_mg": 0,
              "vit_d_mcg": 0, "b12_mcg": 0, "caffeine_mg": 0, "water_g": 0},
 "per_serving": null, "serving_g": null,
 "units": {"piece": 0, "katori": 0, "cup": 0, "glass": 0, "plate": 0, "tsp": 0, "slice": 0},
 "default_unit": "piece", "typical_qty": 1, "countable": true,
 "allergens": ["milk", "peanut", "tree_nut", "egg", "gluten", "soy", "fish", "shellfish", "sesame", "mustard"],
 "veg": "veg|egg|nonveg|vegan", "tags": [], "source_kind": "ifct|usda|brand|restaurant|web"}"""

SEARCH_PROMPT = """Look up the nutrition of this food as it is eaten in India, using current web results.

FOOD: {food}

Prefer, in order: IFCT 2017 (Indian Food Composition Tables, NIN-ICMR) for Indian foods and dishes; USDA FoodData Central;
the brand's own website or pack for a branded product; the restaurant's published menu or nutrition for a restaurant dish.
Give values per 100 g of the food as eaten (cooked, prepared), and the usual weight in grams of one unit people use for it
in Indian homes (a piece, a katori of about 150 ml, a cup, a glass, a plate, a teaspoon); leave out units that do not fit.
Leave out any nutrient you cannot find (do not write 0 for unknown). Allergens: only those it contains; tags from: egg tea
coffee water fruit vegetable sweet sugar dairy fried alcohol sugary_drink grain pulse meat fish snack nuts.

Return only JSON, nothing else:
""" + FOOD_JSON

ESTIMATE_PROMPT = """You estimate the nutrition of a food eaten in India when no published value was found.

FOODS is every food in the meal that has no published value. Build each from its usual recipe and ingredients (and the
ingredients given, if any). Give values per 100 g as eaten, and the usual grams of the units people use for it in Indian
homes. Leave out nutrients you cannot estimate. A medicine or a supplement is not food: for that item return only
{"i": 0, "medicine": true}. If one of that food's candidates is the same food, return only {"i": 0, "match": "that id"}.
A pack you cannot identify returns {"i": 0, "needs": "label"}. When several versions are common, return the usual one
with "assumed": true. A home recipe with no usual amounts returns {"i": 0, "kind": "recipe"}.

Return only JSON: {"items": [{"i": 0}]} and each item has this shape plus its i:
""" + FOOD_JSON


async def normalise(items: list[dict], candidates: list[list[dict]], *, essential: bool = True, timeout_s: float = 8.0) -> list[dict]:
    """One call for all the items of a meal. Returns one decision per item (an empty dict when the model failed)."""
    body = json.dumps({"ITEMS": [{"i": i, **{k: v for k, v in it.items() if v}} for i, it in enumerate(items)],
                       "CANDIDATES": {str(i): c for i, c in enumerate(candidates)}}, ensure_ascii=False)
    try:
        got = await ask("nutrition", NORMALISE_PROMPT, body, essential=essential, max_tokens=2500, timeout_s=timeout_s)
    except router.AllModelsFailed:
        logger.warning("nutrition normaliser unavailable")
        return [{} for _ in items]
    rows = got.get("items") if isinstance(got, dict) else got
    out: list[dict] = [{} for _ in items]
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and isinstance(r.get("i"), int) and 0 <= r["i"] < len(items):
            out[r["i"]] = r
    return out


# ── Open Food Facts (only a barcode or a product name leaves India) ──────────

OFF_FIELDS = "product_name,brands,nutriments,serving_size,serving_quantity,ingredients_text,allergens_tags,quantity,code"
OFF_MAP = {"energy-kcal_100g": "energy_kcal", "proteins_100g": "protein_g", "carbohydrates_100g": "carbs_g", "sugars_100g": "sugar_g",
           "added-sugars_100g": "added_sugar_g", "fiber_100g": "fibre_g", "fat_100g": "fat_g", "saturated-fat_100g": "sat_fat_g",
           "cholesterol_100g": "cholesterol_mg", "sodium_100g": "sodium_mg", "potassium_100g": "potassium_mg", "calcium_100g": "calcium_mg",
           "iron_100g": "iron_mg", "vitamin-d_100g": "vit_d_mcg", "vitamin-b12_100g": "b12_mcg", "caffeine_100g": "caffeine_mg"}
# OFF keeps grams for everything; these keys are milligrams or micrograms here
OFF_SCALE = {"cholesterol_mg": 1000, "sodium_mg": 1000, "potassium_mg": 1000, "calcium_mg": 1000, "iron_mg": 1000, "caffeine_mg": 1000,
             "vit_d_mcg": 1_000_000, "b12_mcg": 1_000_000}


def off_base() -> str:
    return os.getenv("OPEN_FOOD_FACTS_URL", "https://world.openfoodfacts.org").rstrip("/")


def _off_headers() -> dict:
    return {"User-Agent": "KavachCare/1.0 (nutrition lookup; https://kavach.care)"}


def from_off(p: dict) -> dict | None:
    """An Open Food Facts product → a food dict (None without an energy value)."""
    nm = p.get("nutriments") or {}
    per: dict = {}
    for k, ours in OFF_MAP.items():
        v = N.number(nm.get(k))
        if v is not None:
            per[ours] = v * OFF_SCALE.get(ours, 1)
    if "energy_kcal" not in per and N.number(nm.get("energy_100g")) is not None:
        per["energy_kcal"] = round((N.number(nm.get("energy_100g")) or 0) / 4.184, 1)  # kJ
    if "energy_kcal" not in per:
        return None
    serving = N.number(p.get("serving_quantity"))
    allergens = [str(a).split(":", 1)[-1].replace("-", "_") for a in p.get("allergens_tags") or []][:12]
    return {"name": str(p.get("product_name") or "")[:200], "brand": (str(p.get("brands") or "").split(",")[0].strip() or None),
            "kind": "packaged", "per_100g": per, "serving_g": serving, "units": {"pack": serving} if serving else {},
            "default_unit": "serving" if serving else "g", "typical_qty": 1 if serving else 100, "countable": bool(serving),
            "ingredients": str(p.get("ingredients_text") or "")[:2000] or None, "allergens": allergens,
            "barcode": str(p.get("code") or "") or None, "source": "off",
            "source_urls": [f"{off_base()}/product/{p.get('code')}"] if p.get("code") else []}


async def off_barcode(barcode: str) -> dict | None:
    code = "".join(ch for ch in str(barcode) if ch.isdigit())
    if not 8 <= len(code) <= 14:
        return None
    try:
        async with httpx.AsyncClient(timeout=4.0, headers=_off_headers()) as c:
            r = await c.get(f"{off_base()}/api/v2/product/{code}.json", params={"fields": OFF_FIELDS})
        data = r.json() if r.status_code == 200 else {}
    except Exception:  # noqa: BLE001 — the chain goes on
        logger.info("open food facts barcode look-up failed", exc_info=True)
        return None
    return from_off(data.get("product") or {}) if data.get("status") == 1 else None


async def off_search(name: str, brand: str | None) -> dict | None:
    """A branded product by name: the first result of that brand that has nutrition values."""
    q = " ".join(x for x in (brand, name) if x).strip()[:120]
    if not q:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_off_headers()) as c:
            r = await c.get(f"{off_base()}/cgi/search.pl", params={"search_terms": q, "search_simple": 1, "json": 1, "page_size": 8,
                                                                    "fields": OFF_FIELDS})
        products = (r.json() or {}).get("products") or [] if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        logger.info("open food facts search failed", exc_info=True)
        return None
    want = (brand or "").strip().lower()
    for p in products:
        brands = str(p.get("brands") or "").lower()
        if want and want not in brands:
            continue  # the store's brand field, against the brand the brain gave (data against data)
        got = from_off(p)
        if got:
            return got
    return None


# ── grounded web look-up and estimate ────────────────────────────────────────


def _food_from_json(d, *, source: str, urls: list[str] | None = None) -> dict | None:
    if not isinstance(d, dict):
        return None
    per = N.clean(d.get("per_100g"))
    if "energy_kcal" not in per:
        return None
    unit = d.get("default_unit") if d.get("default_unit") in P.UNITS else None
    src = d.get("source_kind") if d.get("source_kind") in SOURCES else None
    return {
        "name": str(d.get("name") or "")[:200], "kind": d.get("kind") if d.get("kind") in KINDS else "generic",
        "per_100g": per, "per_serving": N.clean(d.get("per_serving")) or None, "serving_g": N.number(d.get("serving_g")),
        "units": P.clean_units(d.get("units")), "default_unit": unit, "typical_qty": N.number(d.get("typical_qty")),
        "countable": bool(d.get("countable")), "allergens": [str(a).lower()[:30] for a in d.get("allergens") or [] if a][:12],
        "veg": d.get("veg") if d.get("veg") in VEG else None, "tags": [t for t in d.get("tags") or [] if t in TAGS],
        "source": {"ifct": "ifct", "usda": "usda"}.get(src or "", "web") if source == "web" else source,
        "source_urls": list(urls or [])[:4],
    }


def _describe(item: dict, canonical: str | None) -> str:
    bits = [canonical or item.get("name") or ""]
    if item.get("brand"):
        bits.append(f"brand {item['brand']}")
    if item.get("restaurant"):
        bits.append(f"from the restaurant {item['restaurant']}")
    if item.get("notes"):
        bits.append(f"({str(item['notes'])[:200]})")
    return ", ".join(b for b in bits if b)


async def web(item: dict, canonical: str | None, *, timeout_s: float, problem: str = "") -> dict | None:
    """Gemini grounded on Google Search (Mumbai first; only the food's name goes out)."""
    from app.brain import websearch

    prompt = SEARCH_PROMPT.replace("{food}", _describe(item, canonical))
    if problem:
        prompt += f"\n\nA first look-up gave values that do not add up ({problem}); check the source again."
    # one budget for every place: a hung Mumbai search must not be followed by another full wait
    started = time.monotonic()
    for loc in dict.fromkeys((search_location(), "global")):
        left = timeout_s - (time.monotonic() - started)
        if left < 1.5:
            raise Slow()
        try:
            got = await asyncio.wait_for(websearch.grounded(prompt, location=loc, role="nutrition_search", max_tokens=1500, max_sources=4,
                                                            model=os.getenv("NUTRITION_SEARCH_MODEL") or None), timeout=left)
        except asyncio.TimeoutError:
            raise Slow() from None
        except Exception as exc:  # noqa: BLE001 — a region without grounding falls to the next; then the estimate
            logger.info("grounded nutrition look-up failed at %s: %s", loc, str(exc)[:160])
            continue
        food = _food_from_json(parse_json(got.text), source="web", urls=[s["url"] for s in got.sources])
        if food:
            return food
    return None


def _decorate(item: dict, decision: dict, food: dict) -> dict:
    """The normaliser's kind and units fill only what the look-up left blank."""
    canonical = str(decision.get("canonical") or "").strip() or None
    kind = decision.get("kind") if decision.get("kind") in KINDS else ("packaged" if item.get("brand") else "generic")
    food.setdefault("kind", kind)
    if kind in ("restaurant", "packaged", "recipe"):
        food["kind"] = kind
    if item.get("restaurant") and food["kind"] == "restaurant":
        food["restaurant"] = str(item["restaurant"])[:160]
    if item.get("brand") and not food.get("brand"):
        food["brand"] = str(item["brand"])[:120]
    if not food.get("name"):
        food["name"] = canonical or str(item.get("name") or "")
    for k in ("countable", "default_unit", "typical_qty"):
        if decision.get(k) not in (None, "") and (k != "default_unit" or decision[k] in P.UNITS) and not food.get(k):
            food[k] = decision[k]
    food["tags"] = sorted({*(food.get("tags") or []), *[t for t in decision.get("tags") or [] if t in TAGS]})
    return food


async def estimate_all(pairs: list[tuple[dict, dict]], *, essential: bool = True, timeout_s: float = 6.0,
                       cards: list[list[dict]] | None = None, may_ask: bool = True) -> list[dict | None]:
    """One call for every food in the meal that still has no published value. None where the model failed.

    A row may be a food, or only a match, a medicine, a label question, or a recipe with no amounts.
    """
    if not pairs:
        return []
    cards = cards or [[] for _ in pairs]
    body = json.dumps({"ask": may_ask, "FOODS": [{"i": i, "food": _describe(item, str(decision.get("canonical") or "") or None),
                                              **({"candidates": cards[i]} if cards[i] else {})}
                                              for i, ((item, decision), _c) in enumerate(zip(pairs, cards))]}, ensure_ascii=False)
    try:
        got = await ask("nutrition", ESTIMATE_PROMPT, body, essential=essential, max_tokens=2000, timeout_s=timeout_s)
    except router.AllModelsFailed:
        return [None] * len(pairs)
    rows = got.get("items") if isinstance(got, dict) else None
    out: list[dict | None] = [None] * len(pairs)
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("i"), int) and 0 <= row["i"] < len(pairs):
                if row.get("medicine") is True:
                    out[row["i"]] = {"medicine": True}
                    continue
                if isinstance(row.get("match"), str) and row.get("match"):
                    out[row["i"]] = {"match": row["match"]}
                    continue
                if row.get("needs") == "label":
                    out[row["i"]] = {"needs": "label", "options": [str(o)[:60] for o in row.get("options") or []][:4]}
                    continue
                food = _food_from_json(row, source="model")
                if food is not None:
                    if row.get("assumed") is True:
                        food["_assumed"] = True
                    out[row["i"]] = _decorate(pairs[row["i"]][0], pairs[row["i"]][1], food)
                elif row.get("kind") in KINDS:
                    out[row["i"]] = {"kind": row.get("kind")}
    elif len(pairs) == 1 and isinstance(got, dict):
        food = _food_from_json(got, source="model")
        if food is not None:
            out[0] = _decorate(pairs[0][0], pairs[0][1], food)
    return out


async def reply_lookup(session: AsyncSession, family_id: str, items: list[dict], *, pantry_rows: list[NutritionPantry] | None = None,
                       essential: bool = True, no_questions: bool = False) -> list[Resolved]:
    """What the person's reply may wait for: a catalogue hit, or one estimate for the whole meal.

    The grounded search is not started here. A saved estimate is marked for that search after they have been told.
    """
    found = await resolve(session, family_id, items, pantry_rows=pantry_rows, cache_only=True)
    missing = [i for i, r in enumerate(found) if r.food is None and not r.medicine]
    if not missing:
        return found
    pantry_rows = list(pantry_rows or [])
    by_id = {str(p.id): p for p in pantry_rows}
    by_food = {p.food_id: p for p in pantry_rows if p.food_id}
    card_lists = []
    for i in missing:
        cards, _exact = await candidates(session, family_id, items[i], pantry_rows)
        card_lists.append(cards)
    # One call. Live nutrition calls are a few seconds; this is the only one the reply waits for.
    foods = await estimate_all([(items[i], {}) for i in missing], essential=essential, timeout_s=3.0, cards=card_lists,
                               may_ask=not no_questions)
    for i, food in zip(missing, foods):
        if not isinstance(food, dict):
            found[i].needs = "photo"
            continue
        if food.get("medicine") is True:
            found[i].medicine = True
            continue
        if food.get("needs") == "label":
            found[i].needs = "label"
            found[i].options = list(food.get("options") or [])
            continue
        match = str(food.get("match") or "")
        if match.startswith("pantry:") and match[7:] in by_id:
            pantry = by_id[match[7:]]
            found[i].pantry = pantry
            if pantry.food_id:
                found[i].food = await session.get(NutritionFood, pantry.food_id)
            if found[i].food is not None:
                continue
        fid = _uuid(match) if match else None
        if fid is not None:
            row = await session.get(NutritionFood, fid)
            if row is not None and row.family_id in (None, family_id):
                found[i].food = row
                found[i].pantry = by_food.get(row.id)
                continue
        if food.get("per_100g"):
            if food.pop("_assumed", None):
                found[i].assumed_kind = True
            await _keep_fetched(session, family_id, items[i], food, found[i])
            continue
        if food.get("kind") in KINDS:
            found[i].decision = {"kind": food["kind"]}
            found[i].needs = "photo"
            continue
        found[i].needs = "photo"
    for r in found:
        if r.food is not None:
            r.food.uses = (r.food.uses or 0) + 1
    return found


# ── the catalogue ────────────────────────────────────────────────────────────


def card(f: NutritionFood) -> dict:
    """How a catalogue food is shown to the normaliser (no personal data in generic ones by construction)."""
    per = f.per_100g or {}
    return {"id": str(f.id), "name": f.name, **({"brand": f.brand} if f.brand else {}), **({"restaurant": f.restaurant} if f.restaurant else {}),
            "kind": f.kind, "kcal_100g": per.get("energy_kcal"), **({"family": True} if f.family_id else {})}


async def candidates(session: AsyncSession, family_id: str, item: dict, pantry_rows: list[NutritionPantry]) -> tuple[list[dict], NutritionFood | None]:
    """Candidates for one item, and an exact hit that needs no model call: the same name, alias or barcode (data the AI
    matched before, compared as data)."""
    name = str(item.get("name") or "").strip().lower()[:200]
    brand = str(item.get("brand") or "").strip().lower()
    scope = or_(NutritionFood.family_id.is_(None), NutritionFood.family_id == family_id)
    exact: NutritionFood | None = None
    if item.get("barcode"):
        exact = (await session.execute(select(NutritionFood).where(scope, NutritionFood.barcode == str(item["barcode"]))
                                       .order_by(NutritionFood.family_id.is_(None)).limit(1))).scalar_one_or_none()
    if exact is None and name:
        rows = list((await session.execute(select(NutritionFood).where(
            scope, or_(func.lower(NutritionFood.name) == name, NutritionFood.aliases.contains([name]))).limit(8))).scalars())
        # the family's own first; a brand they named must be the food's brand; a restaurant's dish only for that restaurant
        rows.sort(key=lambda f: (f.family_id is None, -(f.uses or 0)))
        for f in rows:
            if brand and (f.brand or "").lower() != brand:
                continue
            if (f.restaurant or "").lower() != str(item.get("restaurant") or "").strip().lower():
                continue
            exact = f
            break
    found: list[NutritionFood] = []
    if name:
        sim = func.similarity(func.lower(NutritionFood.name), name)
        found = list((await session.execute(select(NutritionFood).where(scope).order_by(sim.desc()).limit(CANDIDATES))).scalars())
    cards = [card(f) for f in found]
    for p in pantry_rows:
        cards.append({"id": f"pantry:{p.id}", "name": p.name, **({"brand": p.brand} if p.brand else {}), "kind": "pantry",
                      "left": p.count})
    return cards, exact


async def save_food(session: AsyncSession, food: dict, *, family_id: str | None, alias: str | None = None) -> NutritionFood:
    """Add a looked-up food to the catalogue (a generic one is shared; a recipe or label stays with the family). The same
    generic food written at the same moment by another turn is read back instead."""
    now = clock.now()
    per = N.clean(food.get("per_100g"))
    chk = N.check_per_100g(per)
    rough = bool(food.get("rough")) or not chk.ok
    conf = {"label": 0.95, "off": 0.85, "ifct": 0.9, "usda": 0.9, "web": 0.8, "caregiver": 0.9, "seed": 0.9}.get(food.get("source") or "", 0.4)
    if rough:
        conf = min(conf, 0.45)
    name = (food.get("name") or alias or "food").strip()[:200]
    row = NutritionFood(
        id=uuid.uuid4(), family_id=family_id, name=name, aliases=sorted({a for a in [alias and alias.lower()[:120], name.lower()] if a}),
        brand=food.get("brand"), restaurant=food.get("restaurant"), kind=food.get("kind") if food.get("kind") in KINDS else "generic",
        per_100g=per, per_serving=N.clean(food.get("per_serving")) or None, serving_g=N.number(food.get("serving_g")),
        units=P.clean_units(food.get("units")), default_unit=food.get("default_unit") if food.get("default_unit") in P.UNITS else None,
        typical_qty=N.number(food.get("typical_qty")), countable=bool(food.get("countable")),
        tags=[t for t in food.get("tags") or [] if t in TAGS], ingredients=food.get("ingredients"),
        allergens=list(food.get("allergens") or [])[:12], veg=food.get("veg") if food.get("veg") in VEG else None,
        source=str(food.get("source") or "model")[:12], source_urls=list(food.get("source_urls") or [])[:4], confidence=conf,
        checked={**chk.to_dict(), "rough": rough}, barcode=food.get("barcode"), uses=0, created_at=now, updated_at=now,
    )
    if family_id is None:
        try:
            async with session.begin_nested():
                session.add(row)
                await session.flush()
        except IntegrityError:
            same = (await session.execute(select(NutritionFood).where(
                NutritionFood.family_id.is_(None), func.lower(NutritionFood.name) == name.lower(),
                func.coalesce(func.lower(NutritionFood.brand), "") == (row.brand or "").lower(),
                func.coalesce(func.lower(NutritionFood.restaurant), "") == (row.restaurant or "").lower()))).scalar_one_or_none()
            if same is None:
                raise
            await add_alias(session, same, alias)
            return same
        return row
    session.add(row)
    await session.flush()
    return row


async def add_alias(session: AsyncSession, food: NutritionFood, alias: str | None) -> None:
    a = (alias or "").strip().lower()[:120]
    if a and a not in (food.aliases or []) and len(food.aliases or []) < 40:
        food.aliases = [*(food.aliases or []), a]
        food.updated_at = clock.now()
        await session.flush()


# ── the chain ────────────────────────────────────────────────────────────────


@dataclass
class Resolved:
    item: dict
    food: NutritionFood | None = None
    pantry: NutritionPantry | None = None
    needs: str | None = None  # label | photo
    options: list[str] = field(default_factory=list)
    medicine: bool = False
    assumed_kind: bool = False  # several kinds exist; the common one is saved and they can correct it
    decision: dict = field(default_factory=dict)  # the normaliser's answer

    @property
    def estimate(self) -> bool:
        return bool(self.food and (self.food.source == "model" or (self.food.checked or {}).get("rough")))


async def fetch(item: dict, decision: dict, *, deadline: float | None, estimates_only: bool = False) -> dict | None:
    """Barcode, brand, or one web look-up for an item the catalogue does not have. No database, so a meal's new foods
    run together. An estimate is a separate call for the whole meal."""
    canonical = str(decision.get("canonical") or "").strip() or None
    kind = decision.get("kind") if decision.get("kind") in KINDS else ("packaged" if item.get("brand") else "generic")
    food: dict | None = None
    if item.get("barcode"):
        food = await off_barcode(str(item["barcode"]))
    if food is None and kind == "packaged" and item.get("brand"):
        food = await off_search(str(item.get("name") or canonical or ""), str(item["brand"]))
    if food is None and kind != "recipe" and not estimates_only:
        remaining = (deadline - time.monotonic()) if deadline else WEB_S
        if remaining < 2.0:
            raise Slow()
        try:
            food = await web(item, canonical, timeout_s=min(remaining, WEB_S))
        except Slow:
            food = None
            # the turn has no time left to estimate: the background run saves an estimate and refines it
            if deadline is not None and (deadline - time.monotonic()) < 4.0:
                raise
        if food:
            chk = N.check_per_100g(food["per_100g"])
            if not chk.ok:
                food["rough"] = True  # refined after they are told; not a second search before the save
    if food is None:
        return None
    return _decorate(item, decision, food)


async def _keep_fetched(session: AsyncSession, family_id: str, item: dict, food: dict, resolved: Resolved) -> None:
    kept = await keep(session, family_id, item, food)
    resolved.food = kept
    p = resolved.pantry
    if p is not None and p.food_id is None:
        from app.nutrition import pantry as pantry_mod

        p.food_id = kept.id
        p.unit_g = p.unit_g or (kept.units or {}).get("piece") or kept.serving_g
        p.per_unit_nutrients = pantry_mod.per_unit(kept, p.unit_g)


async def keep(session: AsyncSession, family_id: str, item: dict, food: dict) -> NutritionFood:
    """Save a fetched food: generic ones are shared; a recipe, or an unpublished estimate, stays with the family until a
    source agrees (plan §N11)."""
    family = family_id if food.get("kind") == "recipe" or food.get("source") == "model" else None
    return await save_food(session, food, family_id=family, alias=str(item.get("name") or ""))


async def resolve(session: AsyncSession, family_id: str, items: list[dict], *, pantry_rows: list[NutritionPantry] | None = None,
                  deadline: float | None = None, essential: bool = True, no_questions: bool = False,
                  estimates_only: bool = False, cache_only: bool = False) -> list[Resolved]:
    """Find every item's food. Raises Slow when a web look-up does not fit before `deadline` (time.monotonic()).
    no_questions: the meal has had its two questions; a vague or unlabelled item is looked up (or estimated) as it is."""
    pantry_rows = list(pantry_rows or [])
    by_id = {str(p.id): p for p in pantry_rows}
    by_food = {p.food_id: p for p in pantry_rows if p.food_id}
    out = [Resolved(item=it) for it in items]
    cards: list[list[dict]] = []
    todo: list[int] = []
    for i, it in enumerate(items):
        fid = _uuid(it.get("food_id")) if it.get("food_id") else None
        if fid is not None:
            f = await session.get(NutritionFood, fid)
            if f and f.family_id in (None, family_id):
                out[i].food = f
                cards.append([])
                continue
        c, exact = await candidates(session, family_id, it, pantry_rows)
        cards.append(c)
        if exact is not None:
            # A name already in the catalogue is that food. Pantry stock is counted down when it is the same food;
            # it does not send a known food back to the model.
            out[i].food = exact
            out[i].pantry = by_food.get(exact.id)
        else:
            todo.append(i)
    if cache_only or not todo:
        if not (cache_only and todo):
            for r in out:
                if r.food is not None:
                    r.food.uses = (r.food.uses or 0) + 1
        return out
    slow: list[int] = []
    if todo:
        left = (deadline - time.monotonic()) if deadline else 4.0
        if deadline is not None and left < 2.0:
            raise Slow()
        decisions = await normalise([items[i] for i in todo], [cards[i] for i in todo], essential=essential,
                                    timeout_s=min(4.0, left))
        for i, d in zip(todo, decisions):
            r = out[i]
            r.decision = d or {}
            if d.get("medicine"):
                r.medicine = True
                continue
            m = str(d.get("match") or "")
            if m.startswith("pantry:") and m[7:] in by_id:
                r.pantry = by_id[m[7:]]
                r.food = await session.get(NutritionFood, r.pantry.food_id) if r.pantry.food_id else None
                if r.food is None:
                    slow.append(i)
                continue
            if m and m in {c["id"] for c in cards[i]}:
                r.food = await session.get(NutritionFood, uuid.UUID(m))
                if r.food is not None:
                    await add_alias(session, r.food, str(items[i].get("name") or ""))
                    r.pantry = by_food.get(r.food.id)  # the same pack is at home: it is counted down
                    continue
            if d.get("needs") == "which":
                # A vague kind is the common version of what they named, not a question.
                r.assumed_kind = True
            elif d.get("needs") == "label" and not no_questions:
                r.needs = "label"
                r.options = [str(o)[:60] for o in d.get("options") or []][:4]
                continue
            slow.append(i)
    if slow:
        def asked(i: int) -> dict:
            p = out[i].pantry
            return items[i] if p is None else {**items[i], "name": p.name, "brand": p.brand}

        got = await asyncio.gather(*(fetch(asked(i), out[i].decision, deadline=deadline, estimates_only=estimates_only)
                                     for i in slow), return_exceptions=True)
        if any(isinstance(g, Slow) for g in got):
            raise Slow()
        missing: list[int] = []
        for i, g in zip(slow, got):
            if isinstance(g, BaseException):
                logger.warning("nutrition look-up failed: %s", str(g)[:200])
                g = None
            if g is None:
                missing.append(i)
                continue
            await _keep_fetched(session, family_id, asked(i), g, out[i])
        if missing:
            left = (deadline - time.monotonic()) if deadline else 4.0
            if deadline is not None and left < 1.5:
                raise Slow()
            foods = await estimate_all([(asked(i), out[i].decision) for i in missing], essential=essential,
                                       timeout_s=min(4.0, max(1.5, left)))
            for i, food in zip(missing, foods):
                if food is None:
                    out[i].needs = "photo"
                    continue
                await _keep_fetched(session, family_id, asked(i), food, out[i])
    for r in out:
        if r.food is not None:
            r.food.uses = (r.food.uses or 0) + 1
    return out


def _uuid(v) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(v))
    except ValueError:
        return None
