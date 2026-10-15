"""Fake models and web for the nutrition tests (the real ones are measured by eval/nutrition_eval.py). Invented foods data
from public reference values; no real people anywhere."""

from __future__ import annotations

import json

import pytest

from app.brain import websearch
from app.llm import router, spend
from app.llm.router import LLMReply
from app.nutrition import agent, backfill, lookup, pantry, report, vision

EGG = {"name": "Egg, whole, boiled", "kind": "generic", "per_100g": {"energy_kcal": 155, "protein_g": 12.6, "carbs_g": 1.1, "fat_g": 10.6,
       "sodium_mg": 124, "cholesterol_mg": 373, "water_g": 75}, "units": {"piece": 50}, "default_unit": "piece", "typical_qty": 2,
       "countable": True, "allergens": ["egg"], "veg": "egg", "tags": ["egg"], "source_kind": "usda"}
ROTI = {"name": "Chapati (whole wheat)", "kind": "generic", "per_100g": {"energy_kcal": 264, "protein_g": 8.7, "carbs_g": 46.4, "fat_g": 4.5,
        "fibre_g": 9.0, "sodium_mg": 300}, "units": {"piece": 40}, "default_unit": "piece", "typical_qty": 2, "countable": True,
        "allergens": ["gluten"], "veg": "vegan", "tags": ["grain"], "source_kind": "ifct"}
DAL = {"name": "Dal, toor, cooked", "kind": "generic", "per_100g": {"energy_kcal": 116, "protein_g": 6.8, "carbs_g": 16.0, "fat_g": 2.8,
       "fibre_g": 3.5, "sodium_mg": 250}, "units": {"katori": 150}, "default_unit": "katori", "typical_qty": 1, "countable": False,
       "allergens": [], "veg": "veg", "tags": ["pulse"], "source_kind": "ifct"}
TEA = {"name": "Tea with milk and sugar", "kind": "generic", "per_100g": {"energy_kcal": 40, "protein_g": 1.2, "carbs_g": 6.5, "fat_g": 1.1,
       "sugar_g": 6.0, "added_sugar_g": 4.0}, "units": {"cup": 150}, "default_unit": "cup", "typical_qty": 1, "countable": False,
       "allergens": ["milk"], "veg": "veg", "tags": ["tea", "dairy"], "source_kind": "web"}
WATER = {"name": "Water", "kind": "generic", "per_100g": {"energy_kcal": 0, "water_g": 100}, "units": {"glass": 250}, "default_unit": "glass",
         "typical_qty": 1, "countable": False, "tags": ["water"], "source_kind": "usda"}
BIRYANI = {"name": "Chicken dum biryani", "kind": "restaurant", "per_100g": {"energy_kcal": 180, "protein_g": 9.0, "carbs_g": 22.0, "fat_g": 6.5},
           "units": {"serving": 450, "plate": 450}, "default_unit": "serving", "typical_qty": 1, "countable": False, "tags": ["meat", "grain"],
           "source_kind": "restaurant"}
CHUTNEY = {"name": "Peanut chutney", "kind": "generic", "per_100g": {"energy_kcal": 300, "protein_g": 12.0, "carbs_g": 12.0, "fat_g": 24.0},
           "units": {"tbsp": 15, "katori": 100}, "default_unit": "tbsp", "typical_qty": 2, "countable": False, "allergens": ["peanut"],
           "tags": [], "source_kind": "web"}
REDBULL = {"name": "Red Bull", "kind": "packaged", "per_100g": {"energy_kcal": 45, "protein_g": 0, "carbs_g": 11, "sugar_g": 11},
           "serving_g": 250, "units": {"serving": 250, "pack": 250}, "default_unit": "serving", "typical_qty": 1, "countable": True,
           "tags": ["sugary_drink"], "source_kind": "brand"}
WEB = {"boiled egg": EGG, "egg": EGG, "eggs": EGG, "chapati": ROTI, "roti": ROTI, "dal": DAL, "tea": TEA, "water": WATER,
       "chicken dum biryani": BIRYANI, "chicken biryani": BIRYANI, "peanut chutney": CHUTNEY, "red bull": REDBULL}


class FakeNutritionModels:
    """Answers each nutrition prompt by what it is (normaliser, estimate, vision, checks), from scripts the test can change."""

    def __init__(self):
        self.calls: list[str] = []
        self.decisions: dict[str, dict] = {}  # item name → normaliser decision (match is resolved by candidate name)
        self.meal: dict | None = None
        self.label: dict | None = None
        self.allergy: dict[str, list[str]] = {}  # food name (lower) → allergens
        self.rule_hits: dict[str, list[str]] = {}  # food name (lower) → rule keys
        self.estimates: dict[str, dict] = {}
        self.item_ids: list[str] | None = None
        self.notes: dict[int, dict] = {}

    async def complete(self, route, *, system_stable, messages, **kw):
        system = " ".join(system_stable) if isinstance(system_stable, list) else system_stable
        text = messages[-1]["content"][0]["text"]
        if system.startswith(lookup.NORMALISE_PROMPT[:40]):
            self.calls.append("normalise")
            body = json.loads(text)
            out = []
            for it in body["ITEMS"]:
                d = dict(self.decisions.get(it["name"].lower()) or {})
                cands = body["CANDIDATES"].get(str(it["i"])) or []
                want = d.pop("match_name", None)
                match = None
                if want:
                    match = next((c["id"] for c in cands if c["name"].lower() == want.lower()), None)
                out.append({"i": it["i"], "match": match, "canonical": d.get("canonical", it["name"]), "kind": d.get("kind", "generic"),
                            "medicine": d.get("medicine", False), "needs": d.get("needs"), "options": d.get("options", []),
                            "countable": d.get("countable", False), "default_unit": d.get("default_unit", "serving"), "tags": d.get("tags", [])})
            return LLMReply(text=json.dumps({"items": out}), tool_calls=[], model="fake")
        if system.startswith(lookup.ESTIMATE_PROMPT[:40]):
            self.calls.append("estimate")
            name = text.split("FOOD: ", 1)[-1].split(",")[0].strip().lower()
            return LLMReply(text=json.dumps(self.estimates.get(name) or {}), tool_calls=[], model="fake")
        if system.startswith(vision.MEAL_PROMPT[:40]):
            self.calls.append("meal_photo")
            return LLMReply(text=json.dumps(self.meal or {"food": False, "items": []}), tool_calls=[], model="fake")
        if system.startswith(vision.LABEL_PROMPT[:40]):
            self.calls.append("label_photo")
            return LLMReply(text=json.dumps(self.label or {"label": False}), tool_calls=[], model="fake")
        if system.startswith(agent.CHECK_PROMPT[:40]):
            self.calls.append("check")
            body = json.loads(text)
            items = [{"i": f["i"], "allergy": self.allergy.get(f["name"].lower(), []), "rules": self.rule_hits.get(f["name"].lower(), [])}
                     for f in body["FOODS"]]
            return LLMReply(text=json.dumps({"items": items}), tool_calls=[], model="fake")
        if system.startswith(pantry.PACK_PROMPT[:40]):
            self.calls.append("pack")
            body = json.loads(text)
            return LLMReply(text=json.dumps({"items": [{"i": x["i"], "pieces": 6 if "6" in (x.get("pack") or x["name"]) else 1,
                                                        "unit": "piece" if "6" in (x.get("pack") or x["name"]) else "pack", "food": True}
                                                       for x in body["items"]]}), tool_calls=[], model="fake")
        if system.startswith(report.ITEM_PROMPT[:40]):
            self.calls.append("item")
            body = json.loads(text)
            ids = self.item_ids if self.item_ids is not None else [f["id"] for f in body["FOODS"] if body["ITEM"].lower() in f["name"].lower()]
            return LLMReply(text=json.dumps({"ids": ids}), tool_calls=[], model="fake")
        if system.startswith(backfill.PROMPT[:40]):
            self.calls.append("backfill")
            body = json.loads(text)
            return LLMReply(text=json.dumps({"notes": [self.notes.get(n["id"], {"id": n["id"], "items": []}) | {"id": n["id"]}
                                                       for n in body["notes"]]}), tool_calls=[], model="fake")
        self.calls.append("other")
        return LLMReply(text="Okay.", tool_calls=[], model="fake")


class FakeWeb:
    def __init__(self):
        self.asked: list[str] = []
        self.foods = dict(WEB)
        self.slow = False

    async def grounded(self, prompt, **kw):
        food = prompt.split("FOOD: ", 1)[-1].split("\n", 1)[0].split(",")[0].strip().lower()
        self.asked.append(food)
        if self.slow:
            import asyncio

            await asyncio.sleep(30)
        got = self.foods.get(food)
        if not got:
            raise RuntimeError("no answer from the search")
        return websearch.Grounded(text=json.dumps(got), sources=[{"title": "ref", "url": f"https://example.org/{food.replace(' ', '-')}"}],
                                  model="fake", inr=0.0)


@pytest.fixture
def nfake(monkeypatch):
    f = FakeNutritionModels()
    router.register_provider("fake", f)
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["fake:m"] for r in ("nutrition", "vision", "brain", "judge_fast", "extract", "worker",
                                                                          "classify")}))
    router.reset_breakers()
    spend.reset()
    web = FakeWeb()
    monkeypatch.setattr(websearch, "grounded", web.grounded)
    offs: dict = {"barcode": {}, "search": {}}

    async def off_barcode(code):
        return offs["barcode"].get(code)

    async def off_search(name, brand):
        return offs["search"].get((name.lower(), (brand or "").lower()))

    monkeypatch.setattr(lookup, "off_barcode", off_barcode)
    monkeypatch.setattr(lookup, "off_search", off_search)
    f.web, f.off = web, offs  # type: ignore[attr-defined]
    yield f
    from app.nutrition import jobs

    if jobs._refine is not None and not jobs._refine.done():
        jobs._refine.cancel()
    router._providers.pop("fake", None)
    router.reset_breakers()
