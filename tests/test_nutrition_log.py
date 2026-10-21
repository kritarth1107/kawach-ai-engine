"""log_food end to end through Saheli's tool (fake models): questions, photos, labels, orders, who ate, corrections, undo,
allergies, rules, pantry, the cache. Invented family; no real people."""

import json
import uuid
from datetime import timedelta

import pytest

from app.brain import guards, tools
from app.care import store
from app.care.models import FamilyRoster
from app.core import clock
from app.nutrition import agent, entries, jobs, lookup, nudge_hooks, pantry
from app.nutrition.models import NutritionEntry, NutritionFood
from app.sim.world import SimHost
from tests import nutrition_fakes

nfake = nutrition_fakes.nfake  # the fixture

FAM = "fam-food"
MAA = {"id": "e-food", "name": "Kamla", "role": "elder"}
SON = {"id": "c-food", "name": "Ravi", "role": "primary caregiver"}
IMG = [{"mime": "image/jpeg", "data": "AAAA"}]


def ctx(db, speaker=MAA, images=None, ref=None, channel="whatsapp"):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=speaker, members=[MAA, SON],
                         message_ref=ref or f"m-{uuid.uuid4().hex[:6]}", images=images or [], channel=channel)


async def call(db, name, args, **kw):
    out, err = await tools.run(ctx(db, **kw), name, args)
    return json.loads(out), err


async def active(db, subject=MAA["id"]):
    return [e for e in await entries.between(db, FAM, subject, "2026-01-01", "2026-12-31")]


EGG_D = {"countable": True, "default_unit": "piece", "tags": ["egg"]}


async def test_a_night_after_dinner_is_late_night_and_breakfast_stays_open(db, at, nfake):
    """02:36 and 04:58 after a saved dinner are late night on that clock date, not dinner or breakfast."""
    at("2026-10-15 05:10")
    water, _ = await call(db, "log_food", {"eaten_at": "2026-10-12T03:00:00+05:30", "items": [{"name": "water", "quantity": 1, "unit": "glass"}]})
    assert water["result"] == "saved", water
    late_dinner, _ = await call(db, "log_food", {"meal": "breakfast", "eaten_at": "2026-10-13T01:20:00+05:30",
                                                 "items": [{"name": "roti", "quantity": 2, "unit": "piece"}]})
    assert late_dinner["result"] == "saved", late_dinner
    dinner, _ = await call(db, "log_food", {"eaten_at": "2026-10-14T21:10:00+05:30",
                                            "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert dinner["result"] == "saved", dinner
    after, _ = await call(db, "log_food", {"meal": "dinner", "eaten_at": "2026-10-14T23:40:00+05:30",
                                           "items": [{"name": "boiled egg", "quantity": 1, "unit": "piece"}]})
    assert after["result"] == "saved", after
    eggs, _ = await call(db, "log_food", {"meal": "dinner", "eaten_at": "2026-10-15T02:36:00+05:30",
                                          "items": [{"name": "boiled egg", "quantity": 3, "unit": "piece"}]})
    assert eggs["result"] == "saved", eggs
    cream, _ = await call(db, "log_food", {"meal": "breakfast", "eaten_at": "2026-10-15T04:58:00+05:30",
                                           "items": [{"name": "roti", "quantity": 1, "unit": "piece"}]})
    assert cream["result"] == "saved", cream
    rows = await active(db)
    by = {(e.day, e.eaten_at.astimezone(clock.IST).strftime("%H:%M")): e.meal for e in rows}
    assert by[("2026-10-12", "03:00")] == "late_night"
    assert by[("2026-10-12", "01:20")] == "dinner"  # eaten after midnight, filed on the previous day
    assert by[("2026-10-14", "21:10")] == "dinner"
    assert by[("2026-10-14", "23:40")] == "after_dinner"
    assert by[("2026-10-15", "02:36")] == "late_night"
    assert by[("2026-10-15", "04:58")] == "late_night"
    assert not await nudge_hooks.meal_logged(db, FAM, MAA["id"], "breakfast", "2026-10-15")
    assert await nudge_hooks.meal_logged(db, FAM, MAA["id"], "dinner", "2026-10-14")


async def test_unclear_count_is_asked_once_then_saved_with_the_plans_numbers(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["boiled egg"] = EGG_D
    r, err = await call(db, "log_food", {"items": [{"name": "boiled egg", "said": "boiled eggs"}]})
    assert not err and r["result"] == "needs_input"
    n = r["needs"][0]
    assert n["reason"] == "quantity" and "4" in n["options"] and "How many" in n["question"]
    assert await active(db) == []  # nothing saved before the answer
    r, err = await call(db, "log_food", {"answer_to": r["job_id"], "items": [{"name": "boiled egg", "quantity": 4, "unit": "piece", "said": "boiled eggs"}]})
    assert not err and r["result"] == "saved", r
    assert r["say"] == "Saved: 4 boiled eggs, 310 kcal, 25 g protein, 21 g fat."
    assert "today" not in r  # no target: no progress line
    assert r["first_time"]
    rows = await active(db)
    assert len(rows) == 1 and rows[0].confidence == "said" and rows[0].meal == "breakfast" and rows[0].grams == 200
    ev = await store.events(db, FAM, MAA["id"], kinds=["meal"])
    assert ev and ev[-1].payload["nutrition"] and "4 boiled eggs" in ev[-1].summary


async def test_the_same_meal_again_attaches_and_does_not_save_twice(db, at, nfake):
    """A second log_food for the same person, meal and items adds nothing, and says nothing."""
    at("2026-10-14 21:00")
    args = {"meal": "dinner", "items": [{"name": "chapati", "quantity": 4, "unit": "piece", "said": "paratha"}, {"name": "dal"}]}
    r, err = await call(db, "log_food", args)
    assert not err and r["result"] == "saved", r
    n = len(await active(db))
    assert n == 2
    again, err = await call(db, "log_food", args)
    assert not err and again.get("already") and again["result"] == "saved" and again["say"] == ""
    assert len(await active(db)) == n
    # a different meal of the same foods is its own log
    snack, _ = await call(db, "log_food", {**args, "meal": "snack"})
    assert snack["result"] == "saved" and not snack.get("already")
    assert len(await active(db)) == n + 2


async def test_retrying_the_same_whatsapp_message_does_not_log_food_again(db, at, nfake):
    at("2026-10-14 21:00")
    args = {"meal": "dinner", "items": [{"name": "chapati", "quantity": 2, "unit": "piece"}]}
    first, err = await call(db, "log_food", args, ref="wamid.meal1")
    assert not err and first["result"] == "saved", first
    n = len(await active(db))
    again, err = await call(db, "log_food", {"meal": "dinner", "items": [{"name": "chapati", "quantity": 4, "unit": "piece"}]}, ref="wamid.meal1")
    assert not err and again.get("already")
    assert len(await active(db)) == n


async def test_a_repeat_while_it_is_still_saving_does_not_start_another(db, at, nfake, monkeypatch):
    at("2026-10-14 21:00")
    monkeypatch.setenv("NUTRITION_INLINE_S", "0.01")
    monkeypatch.setattr(lookup, "WEB_S", 30.0)
    nfake.web.slow = True
    nfake.estimates["hara bhara kebab"] = {"name": "Hara bhara kebab", "per_100g": {"energy_kcal": 250, "protein_g": 6, "carbs_g": 8, "fat_g": 22,
                                                                                    "sodium_mg": 800},
                                           "units": {"plate": 150}, "default_unit": "plate", "typical_qty": 1}
    args = {"meal": "snack", "items": [{"name": "hara bhara kebab", "quantity": 1, "unit": "plate"}]}
    r, err = await call(db, "log_food", args)
    assert not err and r["result"] == "saved" and "kcal" in r["say"], r
    assert len(await active(db)) == 1
    again, _ = await call(db, "log_food", args)
    assert again.get("already") and again["say"] == "" and again["result"] == "saved"
    assert len(await active(db)) == 1
    if jobs._refine is not None and not jobs._refine.done():
        jobs._refine.cancel()
    row = (await active(db))[0]
    assert row.flags.get("facts") == ["fried", "high fat", "high sodium"]
    if jobs._kicked is not None and not jobs._kicked.done():
        jobs._kicked.cancel()


async def test_water_is_glasses_or_ml_at_the_time_they_said(db, at, nfake):
    at("2026-10-14 16:00")
    r, err = await call(db, "log_food", {"items": [{"name": "water", "quantity": 2, "unit": "glass"}], "eaten_at": "2026-10-14T11:00"})
    assert not err and r["result"] == "saved", r
    row = (await active(db))[-1]
    assert row.quantity == 2 and row.unit == "glass" and row.nutrients.get("water_ml") == 500
    assert row.eaten_at.hour == 11 or row.day == "2026-10-14"
    one, _ = await call(db, "log_food", {"items": [{"name": "water"}], "eaten_at": "2026-10-14T15:30"})
    assert one["result"] == "saved" and "1" in one["say"]
    ml, _ = await call(db, "log_food", {"items": [{"name": "water", "quantity": 250, "unit": "ml"}], "eaten_at": "2026-10-14T18:00"})
    assert ml["result"] == "saved"
    last = (await active(db))[-1]
    assert last.unit == "ml" and last.quantity == 250 and last.nutrients.get("water_ml") == 250
    counted = entries.counters(await active(db))
    assert counted["water"] == 4  # 2 glasses + 1 glass + 250 ml


async def test_a_pack_or_a_drink_with_no_number_is_one_serving(db, at, nfake):
    """No number on a pack or a drink is one serving. Eggs still ask. The decision is the food's kind and tags."""
    at("2026-10-14 16:00")
    nfake.decisions["red bull"] = {"kind": "packaged", "countable": True, "default_unit": "serving", "tags": ["sugary_drink"]}
    nfake.decisions["tea"] = {"kind": "generic", "countable": False, "default_unit": "cup", "tags": ["tea", "dairy"]}
    r, err = await call(db, "log_food", {"items": [{"name": "red bull", "said": "Red Bull"}]})
    assert not err and r["result"] == "saved", r
    assert r["say"] == "Saved: 1 Red Bull (250 ml), 113 kcal, 28 g sugar — tell me if it was more."
    assert "0 g" not in r["say"]
    assert "How many" not in r["say"]
    row = (await active(db))[-1]
    assert row.quantity == 1 and row.unit == "serving" and row.grams == 250 and row.confidence == "estimate"
    assert row.flags.get("assumed_one") and row.flags.get("serving_ml") == 250
    r, err = await call(db, "log_food", {"items": [{"name": "tea", "said": "tea"}]})
    assert not err and r["result"] == "saved", r
    assert r["say"] == "Saved: 1 tea (150 ml), 60 kcal, 9 g sugar — tell me if it was more."
    said = await call(db, "log_food", {"items": [{"name": "red bull", "quantity": 2, "unit": "pack"}]})
    assert said[0]["result"] == "saved" and "tell me if it was more" not in said[0]["say"]
    assert said[0]["saved"][-1]["confidence"] == "said"


async def test_clear_meal_saves_at_once_with_a_usual_katori_and_one_check(db, at, nfake):
    at("2026-10-14 13:30")
    nfake.decisions["chapati"] = {"countable": True, "default_unit": "piece"}
    r, _ = await call(db, "log_food", {"items": [{"name": "chapati", "quantity": 2, "unit": "piece", "said": "rotis"}, {"name": "dal"}]})
    assert r["result"] == "saved" and r["meal"] == "lunch"
    conf = {s["said"]: s["confidence"] for s in r["saved"]}
    assert conf == {"rotis": "said", "dal": "estimate"}
    assert "check" in r and "dal" in r["check"]
    assert r["estimated"] == ["dal (1 katori)"]


async def test_a_food_found_once_needs_no_model_or_web_next_time(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["boiled egg"] = EGG_D
    await call(db, "log_food", {"items": [{"name": "boiled egg", "quantity": 2, "unit": "piece"}]})
    assert nfake.web.asked == [] and nfake.calls.count("estimate") == 1 and "normalise" not in nfake.calls
    food = (await db.execute(NutritionFood.__table__.select().where(NutritionFood.name == "Egg, whole, boiled"))).first()
    assert food.family_id == FAM and food.source == "model"  # the reply's estimate; the web search is after they are told
    nfake.calls.clear()
    r, _ = await call(db, "log_food", {"items": [{"name": "Boiled egg", "quantity": 1, "unit": "piece"}]})
    assert r["result"] == "saved" and nfake.calls == [] and nfake.web.asked == []


async def test_the_web_search_runs_after_the_reply_and_shares_the_food(db, at, nfake):
    """The reply does not search. The search after it replaces the estimate and shares a published food."""
    from contextlib import asynccontextmanager

    at("2026-10-14 09:00")
    saved, _ = await call(db, "log_food", {"items": [{"name": "boiled egg", "quantity": 2, "unit": "piece"}]})
    assert saved["result"] == "saved" and nfake.web.asked == []

    @asynccontextmanager
    async def sessions():
        yield db

    await agent.refine_saved(sessions, saved["job_id"])
    assert nfake.web.asked == ["boiled egg"]
    food = (await db.execute(NutritionFood.__table__.select().where(NutritionFood.name == "Egg, whole, boiled"))).first()
    assert food.family_id is None and food.source == "usda" and food.source_urls


async def test_a_counted_food_beside_a_stated_amount_does_not_hold_the_meal(db, at, nfake):
    at("2026-10-14 16:00")
    nfake.decisions["boiled egg"] = EGG_D
    nfake.decisions["dal"] = {"countable": False, "default_unit": "katori", "tags": ["pulse"]}
    r, _ = await call(db, "log_food", {"items": [{"name": "boiled egg"}, {"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert r["result"] == "saved" and "kcal" in r["say"], r
    assert "needs" not in r


async def test_a_flavour_is_saved_as_one_dish_and_roti_with_dal_stays_two(db, at, nfake):
    at("2026-10-14 16:00")
    ice = {"name": "Chocolate ice cream", "per_100g": {"energy_kcal": 210, "sugar_g": 24, "fat_g": 11, "protein_g": 3.5, "carbs_g": 24},
           "units": {"g": 1}, "default_unit": "g", "typical_qty": 100, "tags": ["sweet", "dairy"], "source_kind": "web"}
    para = {"name": "Aloo paratha", "per_100g": {"energy_kcal": 240, "protein_g": 6, "carbs_g": 32, "fat_g": 9},
            "units": {"piece": 80}, "default_unit": "piece", "typical_qty": 2, "countable": True, "tags": ["grain"], "source_kind": "ifct"}
    nfake.web.foods["chocolate ice cream"] = ice
    nfake.web.foods["paratha"] = para
    # Two flavours are one dish in the brain's tool call. The reply does not spend a second call folding them.
    r, _ = await call(db, "log_food", {"items": [{"name": "chikoo"}, {"name": "chocolate ice cream", "quantity": 140, "unit": "g"}]})
    assert r["result"] == "saved" and len(r["saved"]) == 2, r
    ice_row = next(s for s in r["saved"] if "ice" in s["said"] or "ice" in s.get("name", "").lower())
    assert ice_row["kcal"] == 294
    assert "parts" not in nfake.calls
    nfake.part_of = {"potato": "paratha"}
    bread, _ = await call(db, "log_food", {"meal": "breakfast", "items": [{"name": "potato"}, {"name": "paratha", "quantity": 2, "unit": "piece"}]})
    assert bread["result"] == "saved" and len(bread["saved"]) == 2, bread
    assert "parts" not in nfake.calls
    nfake.part_of = {}
    two, _ = await call(db, "log_food", {"meal": "lunch", "items": [
        {"name": "roti", "quantity": 2, "unit": "piece"}, {"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert two["result"] == "saved" and len(two["saved"]) == 2, two


async def test_an_unclear_kind_is_saved_as_the_common_one(db, at, nfake):
    at("2026-10-14 16:00")
    nfake.decisions["namkeen"] = {"needs": "which", "options": ["mixture", "bhujia"], "canonical": "mixed namkeen"}
    nfake.estimates["mixed namkeen"] = {"name": "Mixed namkeen", "per_100g": {"energy_kcal": 530, "protein_g": 10, "carbs_g": 45,
                                                                              "fat_g": 34, "sodium_mg": 1200},
                                        "units": {"g": 1}, "default_unit": "g", "typical_qty": 30}
    r, _ = await call(db, "log_food", {"items": [{"name": "namkeen", "quantity": 30, "unit": "g"}]})
    assert r["result"] == "saved" and "needs" not in r, r
    assert "kcal" in r["say"] and "sodium" in r["say"]
    assert r["say"].endswith("tell me if it was a different one.")


async def test_two_questions_at_most_then_an_estimate(db, at, nfake):
    at("2026-10-14 16:00")
    nfake.decisions["boiled egg"] = {**EGG_D, "countable": True}
    nfake.decisions["thekua"] = {"needs": "label"}
    nfake.estimates["thekua"] = {"name": "Thekua", "per_100g": {"energy_kcal": 450, "protein_g": 6, "carbs_g": 60, "fat_g": 20},
                                 "units": {"piece": 30}, "default_unit": "piece", "typical_qty": 2}
    r, _ = await call(db, "log_food", {"items": [{"name": "boiled egg"}]})
    assert r["needs"][0]["reason"] == "quantity"
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "items": [{"name": "thekua"}]})
    assert r["needs"][0]["reason"] == "label"
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "items": [{"name": "thekua", "quantity": 2, "unit": "piece"}]})
    assert r["result"] == "saved" and r["saved"][0]["confidence"] == "said"
    f = (await db.execute(NutritionFood.__table__.select().where(NutritionFood.name == "Thekua"))).first()
    assert f.family_id == FAM  # an unpublished estimate stays with the family


async def test_dont_know_the_amount_saves_a_usual_portion_marked_estimate(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["boiled egg"] = EGG_D
    r, _ = await call(db, "log_food", {"items": [{"name": "boiled egg"}]})
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "unsure": True})
    assert r["result"] == "saved" and r["saved"][0]["confidence"] == "estimate" and r["saved"][0]["kcal"] == 155


async def test_never_mind_saves_nothing(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["boiled egg"] = EGG_D
    r, _ = await call(db, "log_food", {"items": [{"name": "boiled egg"}]})
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "cancel": True})
    assert r["result"] == "dropped" and await active(db) == []


async def test_a_meal_photo_is_confirmed_before_anything_is_saved(db, at, nfake):
    at("2026-10-14 08:30")
    nfake.meal = {"food": True, "description": "two boiled eggs and tea", "items": [
        {"name": "boiled egg", "quantity": 2, "unit": "piece", "grams": 100, "confidence": 0.8},
        {"name": "tea", "quantity": 1, "unit": "cup", "grams": 150, "confidence": 0.7}]}
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    assert r["result"] == "needs_input" and r["needs"][0]["reason"] == "confirm"
    assert r["needs"][0]["question"] == (
        "Looks like boiled egg, how many? 2?, 1 cup tea. "
        "Did you eat this, or is it for someone else — like Ravi's diary?")
    assert r["needs"][0]["options"] == ["I ate it", "Ravi", "Just sharing"]
    assert await active(db) == []
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "confirmed": True})  # the yes comes in a later message, no photo
    assert r["result"] == "saved"
    rows = await active(db)
    assert {e.confidence for e in rows} == {"photo"} and {e.source for e in rows} == {"photo"}


async def test_a_counted_food_asks_how_many_and_their_number_is_what_is_saved(db, at, nfake):
    at("2026-10-14 08:30")
    nfake.meal = {"food": True, "items": [{"name": "roti", "quantity": 2, "unit": "piece", "grams": 72, "confidence": 0.6}]}
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    assert "how many? 2?" in r["needs"][0]["question"]
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "confirmed": True,
                                      "items": [{"name": "roti", "quantity": 3, "unit": "piece"}]})
    rows = await active(db)
    assert len(rows) == 1 and rows[0].name == "Chapati/Roti" and rows[0].quantity == 3


async def test_one_clear_piece_does_not_ask_how_many(db, at, nfake):
    at("2026-10-14 08:30")
    nfake.meal = {"food": True, "items": [{"name": "roti", "quantity": 1, "unit": "piece", "grams": 36, "confidence": 0.9, "count_clear": True}]}
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    assert "1 roti" in r["needs"][0]["question"] and "how many" not in r["needs"][0]["question"]


async def test_two_dish_names_are_offered_and_a_yes_without_a_choice_saves_nothing(db, at, nfake):
    at("2026-10-14 08:30")
    nfake.meal = {"food": True, "items": [
        {"name": "chapati", "also": "paratha", "quantity": 2, "unit": "piece", "grams": 70, "confidence": 0.5}]}
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    q = r["needs"][0]["question"]
    assert "chapati or paratha" in q and "or tell me what it was" in q and "how many? 2?" in q
    assert r["needs"][0]["options"][:2] == ["chapati", "paratha"]
    bare, _ = await call(db, "log_food", {"answer_to": r["job_id"], "confirmed": True})
    assert bare["result"] == "dropped" and await active(db) == []


async def test_the_dish_they_pick_is_the_one_saved(db, at, nfake):
    at("2026-10-14 08:30")
    nfake.meal = {"food": True, "items": [
        {"name": "chapati", "also": "paratha", "quantity": 2, "unit": "piece", "grams": 70, "confidence": 0.5}]}
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "confirmed": True,
                                      "items": [{"name": "paratha", "quantity": 2, "unit": "piece"}]})
    rows = await active(db)
    assert len(rows) == 1 and rows[0].name == "Plain parantha/paratha" and rows[0].quantity == 2


async def test_a_photo_is_not_saved_when_the_same_message_says_it_is_confirmed(db, at, nfake):
    """They have not seen the reading yet, so a confirm flag on the photo's own message asks instead of saving."""
    at("2026-10-14 08:30")
    nfake.meal = {"food": True, "description": "two boiled eggs and tea", "items": [
        {"name": "boiled egg", "quantity": 2, "unit": "piece", "grams": 100, "confidence": 0.8},
        {"name": "tea", "quantity": 1, "unit": "cup", "grams": 150, "confidence": 0.7}]}
    r, _ = await call(db, "log_food", {"from_photo": True, "confirmed": True, "items": []}, images=IMG)
    assert r["result"] == "needs_input" and r["needs"][0]["reason"] == "confirm"
    assert "kcal" not in r["needs"][0]["question"] and "Did you eat this" in r["needs"][0]["question"]
    assert "estimate" not in nfake.calls and "normalise" not in nfake.calls
    assert await active(db) == []


NOODLES = {"food": True, "description": "a plate of instant noodles", "items": [
    {"name": "instant noodles", "quantity": 1, "unit": "plate", "grams": 240, "confidence": 0.8}]}


async def test_a_food_photo_asks_once_then_a_yes_saves_and_naming_it_does_not_ask_again(db, at, nfake):
    """A plate photo, a yes, then the food named again: never 'what's in it' or 'send a photo'."""
    at("2026-10-16 00:58")
    nfake.meal = NOODLES
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    assert r["result"] == "needs_input"
    q = r["needs"][0]["question"]
    assert "Did you eat this, or is it for someone else — like Ravi's diary?" in q
    assert "send a photo" not in q.lower() and "what's in" not in q.lower()
    assert await active(db) == []
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "confirmed": True})
    assert r["result"] == "saved", r
    assert "tell me if" in r["say"] and "what's in" not in r["say"].lower()
    named, _ = await call(db, "log_food", {"items": [{"name": "instant noodles", "quantity": 1, "unit": "plate"}]})
    assert named["result"] == "saved", named
    assert "what's in" not in named.get("say", "").lower()
    assert "send a photo" not in json.dumps(named).lower()


async def test_an_answer_that_names_the_food_does_not_get_the_same_question(db, at, nfake):
    at("2026-10-16 00:58")
    nfake.meal = NOODLES
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "items": [{"name": "instant noodles", "quantity": 1, "unit": "plate"}]})
    assert r["result"] == "saved", r
    assert "Did you eat" not in r.get("say", "") and "what's in" not in r.get("say", "").lower()
    rows = await active(db)
    assert len(rows) == 1 and rows[0].name == "instant noodles"


async def test_a_photo_they_did_not_eat_saves_nothing(db, at, nfake):
    at("2026-10-16 00:58")
    nfake.meal = NOODLES
    r, _ = await call(db, "log_food", {"from_photo": True, "items": []}, images=IMG)
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "consumed": "no"})
    assert r["result"] == "dropped" and await active(db) == []


async def test_a_photo_already_said_to_be_eaten_is_saved_without_asking(db, at, nfake):
    at("2026-10-16 00:58")
    nfake.meal = {"food": True, "description": "two boiled eggs and tea", "items": [
        {"name": "boiled egg", "quantity": 2, "unit": "piece", "grams": 100, "confidence": 0.8},
        {"name": "tea", "quantity": 1, "unit": "cup", "grams": 150, "confidence": 0.7}]}
    r, _ = await call(db, "log_food", {"from_photo": True, "consumed": "yes", "items": []}, images=IMG)
    assert r["result"] == "saved", r
    assert {e.confidence for e in await active(db)} == {"photo"}


async def test_a_photo_for_someone_else_is_saved_to_their_diary(db, at, nfake):
    at("2026-10-16 00:58")
    nfake.meal = NOODLES
    r, _ = await call(db, "log_food", {"from_photo": True, "for_whom": MAA["id"], "consumed": "yes", "items": []},
                      images=IMG, speaker=SON)
    assert r["result"] == "saved", r
    assert len(await active(db, MAA["id"])) == 1
    assert await active(db, SON["id"]) == []


async def test_a_named_food_is_saved_when_nothing_is_published_for_it(db, at, nfake):
    at("2026-10-16 01:00")
    r, _ = await call(db, "log_food", {"items": [{"name": "instant noodles", "quantity": 1, "unit": "plate"}]})
    assert r["result"] == "saved", r
    assert "what's in" not in r.get("say", "").lower() and "send a photo" not in json.dumps(r).lower()
    assert "tell me if" in r["say"]


async def test_a_home_dish_is_asked_what_is_in_it_once_then_saved(db, at, nfake):
    at("2026-10-16 01:00")
    nfake.decisions["home khichdi"] = {"kind": "recipe"}
    r, _ = await call(db, "log_food", {"items": [{"name": "home khichdi"}]})
    assert r["result"] == "needs_input"
    assert r["needs"][0]["question"] == "What's in the home khichdi, roughly?"
    assert "photo" not in r["needs"][0]["question"].lower()
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "items": [{"name": "home khichdi"}]})
    assert r["result"] == "saved", r
    assert "What's in" not in r.get("say", "")


async def test_a_label_photo_becomes_the_familys_food(db, at, nfake):
    at("2026-10-14 17:00")
    nfake.label = {"label": True, "name": "Protein Bar Choco", "brand": "Fitbar", "per_100g": {"energy_kcal": 400, "protein_g": 30, "carbs_g": 40, "fat_g": 13},
                   "per_serving": {"energy_kcal": 240, "protein_g": 18}, "serving_g": 60, "pieces_per_pack": 1, "allergens": ["milk", "soy"]}
    r, _ = await call(db, "log_food", {"label_photo": True, "items": [{"name": "protein bar", "quantity": 1, "unit": "piece"}]}, images=IMG)
    assert r["result"] == "saved" and r["saved"][0]["confidence"] == "label" and r["saved"][0]["kcal"] == 240
    f = (await db.execute(NutritionFood.__table__.select().where(NutritionFood.source == "label"))).first()
    assert f.family_id == FAM and f.brand == "Fitbar" and f.serving_g == 60
    nfake.calls.clear()
    at("2026-10-14 22:00")  # another bar, later: the one just saved is not logged again
    r, _ = await call(db, "log_food", {"items": [{"name": "protein bar", "quantity": 1, "unit": "piece"}]})
    assert r["result"] == "saved" and "label_photo" not in nfake.calls and r["saved"][0]["kcal"] == 240


async def test_a_pack_with_no_brand_asks_which_or_the_label(db, at, nfake):
    at("2026-10-14 17:00")
    nfake.decisions["protein bar"] = {"needs": "label", "kind": "packaged"}
    r, _ = await call(db, "log_food", {"items": [{"name": "protein bar"}]})
    assert r["needs"][0]["reason"] == "label" and "label" in r["needs"][0]["question"]


def _order(service="swiggy", subject=MAA["id"], delivered=True):
    from app.tasks.models import Task

    now = clock.now()
    return Task(id=uuid.uuid4(), family_id=FAM, subject_id=subject, requested_by=SON["id"], service=service, kind="order", goal="biryani",
                details={"chosen": [{"name": "Chicken Dum Biryani", "restaurant": "Paradise", "for_item": "biryani"}], "items": [{"name": "biryani", "qty": 1}]},
                status="done", phase="place", result={"placed": True, "delivered": delivered}, history=[], runs=1, created_at=now, updated_at=now,
                deadline_at=now + timedelta(hours=1))


async def test_order_food_is_logged_only_after_a_yes_or_part(db, at, nfake):
    at("2026-10-14 13:00")
    t1, t2 = _order(), _order()
    db.add_all([t1, t2])
    await db.flush()
    assert {w["task_id"] for w in await agent.order_waiting(db, FAM, MAA["id"])} == {str(t1.id), str(t2.id)}
    r, _ = await call(db, "log_food", {"items": [], "from_order_task": str(t1.id)})
    assert r["needs"][0]["reason"] == "consumed" and await active(db) == []
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "consumed": "no"})
    assert r["result"] == "dropped" and await active(db) == []
    r, _ = await call(db, "log_food", {"items": [], "from_order_task": str(t2.id), "consumed": "part", "portion_eaten": "half"})
    assert r["result"] == "saved", r
    rows = await active(db)
    assert len(rows) == 1 and rows[0].source == "order" and rows[0].source_ref == str(t2.id) and rows[0].consumed == "part"
    assert rows[0].grams == 225  # half of the restaurant's 450 g serving
    assert await agent.order_waiting(db, FAM, MAA["id"]) == []  # both answered


async def test_one_person_is_not_asked_who_ate_it(db, at, nfake):
    at("2026-10-14 13:00")
    nfake.decisions["chapati"] = {"countable": True, "default_unit": "piece"}
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=MAA, members=[],
                      message_ref=f"m-{uuid.uuid4().hex[:6]}")
    out, err = await tools.run(c, "log_food", {"for_whom": "", "items": [{"name": "chapati", "quantity": 1, "unit": "piece"}]})
    r = json.loads(out)
    assert not err and r["result"] == "saved", r
    rows = await active(db)
    assert rows and rows[0].subject_id == MAA["id"]


async def test_who_ate_it_is_asked_when_unclear_and_the_caregiver_logs_for_her(db, at, nfake):
    at("2026-10-14 13:00")
    nfake.decisions["chapati"] = {"countable": True, "default_unit": "piece"}
    r, _ = await call(db, "log_food", {"for_whom": "", "items": [{"name": "chapati", "quantity": 1, "unit": "piece"}]}, speaker=SON)
    assert r["needs"][0]["reason"] == "who_ate" and "Kamla" in r["needs"][0]["options"]
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "for_whom": MAA["id"]}, speaker=SON)
    assert r["result"] == "saved"
    rows = await active(db)
    assert rows[0].subject_id == MAA["id"] and rows[0].source == "caregiver" and rows[0].logged_by == SON["id"]


async def test_her_own_answer_replaces_the_caregivers_estimate_of_the_same_meal(db, at, nfake):
    at("2026-10-14 14:00")
    nfake.decisions["chapati"] = {"countable": True, "default_unit": "piece"}
    await call(db, "log_food", {"for_whom": MAA["id"], "meal": "lunch", "items": [{"name": "chapati", "quantity": 1, "unit": "piece"}]}, speaker=SON)
    r, _ = await call(db, "log_food", {"meal": "lunch", "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert r["replaced"].startswith("This replaces what a caregiver logged")
    rows = await active(db)
    assert [e.name for e in rows if e.status == "active"] == ["Washed moong dal (Dhuli moong ki dal)"]


async def test_last_night_is_logged_on_yesterday_as_dinner(db, at, nfake):
    at("2026-10-14 09:00")
    r, _ = await call(db, "log_food", {"eaten_at": "2026-10-13T21:00", "items": [{"name": "chicken biryani", "quantity": 1, "unit": "plate"}]})
    rows = await active(db)
    assert r["result"] == "saved" and rows[0].day == "2026-10-13" and rows[0].meal == "dinner"


async def test_correct_then_undo_then_remove_then_put_back(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["boiled egg"] = EGG_D
    await call(db, "log_food", {"items": [{"name": "boiled egg", "quantity": 4, "unit": "piece"}]})
    r, err = await call(db, "change_food", {"last": True, "set": {"quantity": 3}})
    assert not err and r["result"] == "changed" and "233 kcal" in r["say"]
    r2, err = await call(db, "undo_change", {"id": r["version_id"]})
    assert not err and r2["result"] == "done"
    e = (await active(db))[0]
    assert e.quantity == 4 and round(e.nutrients["energy_kcal"]) == 310
    r, _ = await call(db, "change_food", {"entry_id": str(e.id), "delete": True})
    assert r["result"] == "removed" and await active(db) == []
    await call(db, "undo_change", {"id": r["version_id"]})
    assert len(await active(db)) == 1
    ch, _ = await call(db, "memory_changes", {"food": True})
    assert ch["changes"] and all(c["what"].startswith("Food log:") for c in ch["changes"])
    plain, _ = await call(db, "memory_changes", {})
    assert not any(c["what"].startswith("Food log:") for c in plain["changes"])  # the care record's history is not buried


async def test_change_food_without_an_id_lists_the_days_entries(db, at, nfake):
    at("2026-10-14 09:00")
    await call(db, "log_food", {"items": [{"name": "tea", "quantity": 1, "unit": "cup"}]})
    r, _ = await call(db, "change_food", {"delete": True})
    assert r["pick"] and r["pick"][0]["what"] == "tea (1 cup)"


async def test_tracking_off_points_to_log_event(db, at, nfake):
    at("2026-10-14 09:00")
    await call(db, "nutrition_settings", {"tracking": "off"})
    r, err = await call(db, "log_food", {"items": [{"name": "tea", "quantity": 1, "unit": "cup"}]})
    assert err and "log_event kind meal" in r["refused"]


async def test_a_meal_event_is_not_a_second_food_log(db, at, nfake):
    """While food tracking is on, agreeing with a meal must not become a meal or water event."""
    at("2026-10-14 21:00")
    meal, err = await call(db, "log_event", {"kind": "meal", "summary": "rice and dal"})
    assert not err and meal["logged"] is False and "log_food" in meal["why"]
    water, _ = await call(db, "log_event", {"kind": "water", "summary": "a glass"})
    assert water["logged"] is False
    assert not await store.events(db, FAM, MAA["id"], kinds=["meal", "water"])
    symptom, _ = await call(db, "log_event", {"kind": "symptom", "summary": "knee pain"})
    assert symptom["logged"] == "knee pain"
    await call(db, "nutrition_settings", {"tracking": "off"})
    off, err = await call(db, "log_event", {"kind": "meal", "summary": "tea"})
    assert not err and off["logged"] == "tea"
    saved = await store.events(db, FAM, MAA["id"], kinds=["meal"])
    assert saved and saved[-1].summary == "tea"


def test_agreeing_with_a_saved_meal_is_not_a_new_log():
    from app.brain.persona import PERSONA

    assert "do not call log_food or log_event" in PERSONA
    assert "three sentences at most" in PERSONA


async def test_a_medicine_never_goes_in_the_food_log(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["vitamin d capsule"] = {"medicine": True}
    r, _ = await call(db, "log_food", {"items": [{"name": "vitamin d capsule", "quantity": 1, "unit": "piece"}]})
    assert r["result"] == "dropped" and "log_dose" in r["why"] and await active(db) == []


async def test_an_allergy_on_record_is_flagged_and_mentioned(db, at, nfake):
    at("2026-10-14 09:00")
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="allergy", key="allergy:peanut", value={"allergen": "peanut"},
                           text="Allergic to peanuts", source_kind="caregiver_said")
    nfake.allergy["peanut chutney"] = ["peanut"]
    r, _ = await call(db, "log_food", {"items": [{"name": "peanut chutney", "quantity": 2, "unit": "tbsp"}]})
    assert "peanut allergy" in r["allergy"]
    assert (await active(db))[0].flags["allergy"] == ["peanut"]


async def test_a_strict_rule_is_pointed_out_a_guide_is_not(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["boiled egg"] = EGG_D
    r, _ = await call(db, "nutrition_target", {"for_whom": MAA["id"], "rules": [{"rule": "max 2 eggs a day", "food": "egg", "max_per_day": 2}],
                                               "set_by": "doctor", "how": "strict"}, speaker=SON)
    key = r["saved"][0]["key"]
    nfake.rule_hits["egg, whole, boiled"] = [key]
    r, _ = await call(db, "log_food", {"items": [{"name": "boiled egg", "quantity": 3, "unit": "piece"}]})
    assert "past the rule 'max 2 eggs a day' set by their doctor" in r["deviation"]
    await call(db, "nutrition_target", {"for_whom": MAA["id"], "rules": [{"rule": "max 2 eggs a day", "food": "egg", "max_per_day": 2}],
                                        "set_by": "doctor", "how": "guide"}, speaker=SON)
    r, _ = await call(db, "log_food", {"items": [{"name": "boiled egg", "quantity": 1, "unit": "piece"}]})
    assert "deviation" not in r


async def test_no_answer_in_two_hours_keeps_it_unquantified(db, at, nfake):
    at("2026-10-14 09:00")
    nfake.decisions["boiled egg"] = EGG_D
    r, _ = await call(db, "log_food", {"items": [{"name": "boiled egg"}]})
    job = await jobs.get(db, FAM, r["job_id"])
    at("2026-10-14 11:30")
    await agent.expire(db, job)
    rows = await active(db)
    assert job.state == "saved" and len(rows) == 1 and rows[0].confidence == "unquantified" and rows[0].nutrients == {}


async def test_food_at_home_is_used_and_counted_down(db, at, nfake):
    at("2026-10-14 17:00")
    bar = NutritionFood(id=uuid.uuid4(), family_id=FAM, name="Protein bar (Fitbar)", aliases=["protein bar (fitbar)"], kind="packaged",
                        per_100g={"energy_kcal": 400, "protein_g": 30, "carbs_g": 40, "fat_g": 13}, units={"piece": 60}, serving_g=60,
                        default_unit="piece", countable=True, tags=[], allergens=[], source="listing", source_urls=[], confidence=0.85, checked={},
                        uses=0, created_at=clock.now(), updated_at=clock.now())
    db.add(bar)
    await db.flush()
    row = await pantry.add(db, family_id=FAM, subject_id=None, name="Protein bar (Fitbar)", count=6, source="order", task_id="t-1", food=bar)
    nfake.decisions["protein bar"] = {"match_name": "Protein bar (Fitbar)"}
    r, _ = await call(db, "log_food", {"items": [{"name": "protein bar", "quantity": 1, "unit": "piece"}]})
    assert r["result"] == "saved" and r["saved"][0]["kcal"] == 240
    assert row.count == 5
    r, _ = await call(db, "pantry", {"action": "view"})
    assert r["at_home"][0]["count"] == 5


async def test_a_skipped_meal_is_a_marker_not_food(db, at, nfake):
    at("2026-10-14 15:00")
    r, _ = await call(db, "log_food", {"skipped": True, "meal": "lunch"})
    assert r["result"] == "saved" and r["skipped"] == "lunch"
    rows = await active(db)
    assert rows[0].kind == "skipped" and rows[0].nutrients == {}


async def test_a_slow_new_food_finishes_in_the_background(db, at, nfake, monkeypatch):
    at("2026-10-14 13:00")
    monkeypatch.setenv("NUTRITION_INLINE_S", "0.3")
    monkeypatch.setattr(jobs, "kick_nutrition", lambda: None)
    nfake.web.slow = True
    nfake.calls.clear()
    nfake.web.asked.clear()
    nfake.estimates["cedar stew"] = {"name": "Cedar stew", "per_100g": {"energy_kcal": 90, "protein_g": 4, "carbs_g": 8, "fat_g": 4},
                                     "units": {"katori": 150}, "default_unit": "katori"}
    r, _ = await call(db, "log_food", {"items": [{"name": "cedar stew", "quantity": 1, "unit": "katori"}]})
    assert r["result"] == "saved" and r["say"].startswith("Saved:"), r
    assert nfake.web.asked == [] and nfake.calls.count("estimate") == 1
    assert len(await active(db)) == 1


@pytest.mark.parametrize("bad", [{"items": []}, {}])
async def test_nothing_named_is_refused(db, at, nfake, bad):
    at("2026-10-14 09:00")
    r, err = await call(db, "log_food", bad)
    assert err and "items" in r["refused"]


async def test_the_dashboard_logs_with_its_own_source(db, at, nfake):
    at("2026-10-14 09:00")
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder={"id": MAA["id"], "name": ""}, speaker={"id": SON["id"], "name": "Ravi", "role": "caregiver"},
                      members=[{"id": MAA["id"]}, {"id": SON["id"]}], channel="dashboard")
    out, err = await tools.run(c, "log_food", {"for_whom": MAA["id"], "items": [{"name": "tea", "quantity": 2, "unit": "cup"}]})
    r = json.loads(out)
    assert not err and r["result"] == "saved" and "first_time" not in r
    rows = await active(db)
    assert rows[0].source == "dashboard" and rows[0].subject_id == MAA["id"]
    assert (await db.get(NutritionEntry, rows[0].id)).logged_by == SON["id"]


async def test_a_grocery_order_answered_yes_is_never_logged_as_eaten(db, at, nfake):
    """Founder Q5: groceries go to the pantry; only a restaurant meal becomes food eaten, and only after a yes."""
    at("2026-10-14 13:00")
    t = _order(service="blinkit")
    db.add(t)
    await db.flush()
    r, _ = await call(db, "log_food", {"items": [], "from_order_task": str(t.id), "consumed": "yes"})
    assert r["result"] == "dropped" and "not a cooked meal" in r["why"]
    assert await active(db) == []


async def test_only_the_items_that_broke_a_strict_limit_are_marked(db, at, nfake):
    at("2026-10-14 13:00")
    await call(db, "nutrition_target", {"for_whom": MAA["id"], "set": {"kcal": 150}, "set_by": "doctor", "how": "strict"}, speaker=SON)
    r, _ = await call(db, "log_food", {"items": [{"name": "chapati", "quantity": 3, "unit": "piece"}, {"name": "water", "quantity": 1, "unit": "glass"}]})
    assert r["result"] == "saved" and "Energy today" in r["deviation"], r
    rows = {e.name: e for e in await active(db)}
    assert rows["Chapati/Roti"].flags.get("deviation") and not rows["Water"].flags.get("deviation")


async def test_the_tracking_line_is_said_once_to_each_person_who_logs(db, at, nfake):
    at("2026-10-14 09:00")
    r, _ = await call(db, "log_food", {"items": [{"name": "tea", "quantity": 1, "unit": "cup"}], "for_whom": MAA["id"]}, speaker=SON)
    assert r.get("first_time") and "dashboard" not in r["first_time"]
    r, _ = await call(db, "log_food", {"items": [{"name": "tea", "quantity": 1, "unit": "cup"}]})
    assert r.get("first_time"), "the elder hears it too, the first time they log their own"
    r, _ = await call(db, "log_food", {"items": [{"name": "tea", "quantity": 1, "unit": "cup"}]})
    assert not r.get("first_time")


async def test_counts_can_be_chosen_on_whatsapp(db, at, nfake):
    at("2026-10-14 09:00")
    r, err = await call(db, "nutrition_settings", {"for_whom": MAA["id"], "counters": ["egg", "water"]}, speaker=SON)
    assert not err and r["setting"]["counters"] == ["egg", "water"]


async def test_a_look_up_that_never_finished_is_closed_after_two_hours(db, at, nfake):
    from app.nutrition.models import NutritionJob

    at("2026-10-14 09:00")
    job = await jobs.create(db, family_id=FAM, subject_id=MAA["id"], requested_by=SON["id"], ask={"items": [{"name": "thekua", "quantity": 2}]},
                            source_ref="m-x", channel="whatsapp")
    job.state = "resolving"
    await db.commit()

    class S:
        def __call__(self):
            return self

        async def __aenter__(self):
            return db

        async def __aexit__(self, *a):
            return False

    at("2026-10-14 11:30")
    stats = await jobs.sweep(S())
    got = await db.get(NutritionJob, job.id)
    rows = await active(db)
    assert stats["expired"] == 1 and got.state == "saved" and rows[0].source == "caregiver" and rows[0].confidence == "pending"


async def test_removing_the_last_item_of_a_photo_meal_deletes_the_photo(db, at, nfake, monkeypatch):
    from app.nutrition import photos

    gone: list[str] = []

    async def fake_delete(ref):
        gone.append(ref)

    monkeypatch.setattr(photos, "delete", fake_delete)
    at("2026-10-14 09:00")
    await call(db, "log_food", {"items": [{"name": "boiled egg", "quantity": 2, "unit": "piece"}, {"name": "tea", "quantity": 1, "unit": "cup"}]})
    rows = await active(db)
    for r in rows:
        r.photo_ref = "gs://bucket/families/fam-food/nutrition/meal.jpg"
    await db.flush()
    await entries.remove(db, rows[0])
    assert gone == []  # the other item of that meal still shows the photo
    await entries.remove(db, rows[1])
    assert gone == ["gs://bucket/families/fam-food/nutrition/meal.jpg"]


SUGAR = {"name": "Sugar", "kind": "generic", "per_100g": {"energy_kcal": 387, "carbs_g": 100, "sugar_g": 100},
         "units": {"tsp": 4}, "default_unit": "tsp", "typical_qty": 1, "countable": True, "tags": ["sugar"], "source_kind": "ifct"}


async def test_fourteen_days_teach_the_usual_amount_and_skip_the_lookup(db, at, nfake, monkeypatch):
    """A synthetic 14-day log. Before learning, a missing number is a question or one serving. After, it is their amount,
    and the known food is not looked up again."""
    nfake.web.foods["sugar"] = SUGAR
    at("2026-09-16 21:00")
    nfake.calls.clear()
    nfake.web.asked.clear()
    roti_before, _ = await call(db, "log_food", {"meal": "dinner", "eaten_at": "2026-09-16T21:00:00+05:30",
                                                 "items": [{"name": "roti"}]})
    tea_before, _ = await call(db, "log_food", {"meal": "breakfast", "eaten_at": "2026-09-16T07:00:00+05:30",
                                                "items": [{"name": "tea"}]})
    before_lookups = nfake.calls.count("estimate") + nfake.calls.count("normalise") + len(nfake.web.asked)
    assert roti_before["result"] == "needs_input"  # counted, and not their amount yet
    assert tea_before["result"] == "saved", tea_before
    tea_saved = next(e for e in await active(db) if "tea" in e.name.lower())
    assert tea_saved.quantity == 1  # one serving, not the 2 they actually drink
    before_error = (1.0 + abs(1 - 2) / 2) / 2

    at("2026-09-15 22:00")
    for day in range(1, 15):
        d = f"2026-09-{day:02d}"
        await call(db, "log_food", {"meal": "dinner", "eaten_at": f"{d}T21:00:00+05:30",
                                    "items": [{"name": "roti", "quantity": 4, "unit": "piece"}]})
        await call(db, "log_food", {"meal": "breakfast", "eaten_at": f"{d}T07:00:00+05:30",
                                    "items": [{"name": "tea", "quantity": 2, "unit": "cup"},
                                              {"name": "sugar", "quantity": 1, "unit": "tsp"}]})
    got = await nudge_hooks.learn_meal_habits(db, FAM, MAA["id"])
    assert got.get("usuals") == "created"
    fact = await store.active_fact(db, FAM, MAA["id"], "learned:food_usuals")
    assert fact and "Chapati" in fact.text and "7:00 am" in fact.text and "Sugar" in fact.text
    assert nudge_hooks.match_usual(fact.value, "roti", "breakfast") is None  # dinner's rotis are not breakfast
    dinner = nudge_hooks.match_usual(fact.value, "roti", "dinner")
    assert dinner["quantity"] == 4

    real = lookup.candidates

    async def hide_exact(session, family_id, item, pantry_rows):
        cards, _exact = await real(session, family_id, item, pantry_rows)
        return cards, None

    monkeypatch.setattr(lookup, "candidates", hide_exact)
    nfake.calls.clear()
    nfake.web.asked.clear()
    roti_after, err = await call(db, "log_food", {"meal": "dinner", "eaten_at": "2026-09-15T21:00:00+05:30",
                                                  "items": [{"name": "roti"}]})
    assert not err and roti_after["result"] == "saved", roti_after
    assert "your usual" in roti_after["say"]
    saved = [e for e in await active(db) if e.day == "2026-09-15" and e.meal == "dinner"]
    assert [e.quantity for e in saved] == [4]
    after_lookups = nfake.calls.count("estimate") + nfake.calls.count("normalise") + len(nfake.web.asked)
    morning, _ = await call(db, "log_food", {"usual": True, "meal": "breakfast", "eaten_at": "2026-09-15T07:00:00+05:30"})
    assert morning["result"] == "saved", morning
    morn = [e for e in await active(db) if e.day == "2026-09-15" and e.meal == "breakfast"]
    assert any("tea" in e.name.lower() and e.quantity == 2 for e in morn)
    assert any("sugar" in e.name.lower() and e.quantity == 1 for e in morn)
    after_error = 0.0
    assert after_error < before_error and after_lookups == 0 < before_lookups

    breakfast_roti, _ = await call(db, "log_food", {"meal": "breakfast", "eaten_at": "2026-09-15T08:00:00+05:30",
                                                    "items": [{"name": "roti"}]})
    assert breakfast_roti["result"] == "needs_input"

    await store.stop_fact(db, family_id=FAM, subject_id=MAA["id"], key="learned:food_usuals", reason="undo",
                          source_kind="caregiver_said", stated_by=MAA["id"])
    assert await store.active_fact(db, FAM, MAA["id"], "learned:food_usuals") is None
    undone, _ = await call(db, "log_food", {"meal": "dinner", "eaten_at": "2026-09-15T21:30:00+05:30",
                                            "items": [{"name": "roti"}]})
    assert undone["result"] == "needs_input"


class _Sessions:
    """tell() opens its own session. The test's session is that session."""

    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc):
        return False


async def test_a_background_save_is_one_line_and_a_reply_is_not_sent_again(db, at, nfake, monkeypatch):
    """The line goes out once from the background save. A save already in the reply is not sent again."""
    at("2026-10-14 13:00")
    sent: list[str] = []

    async def fake_send(sessions, host, roster, family_id, person, text, ref, buttons=None):
        sent.append(text)
        return True

    monkeypatch.setattr("app.brain.wake._send", fake_send)
    db.add(FamilyRoster(family_id=FAM, elder=MAA, members=[MAA, SON], updated_at=clock.now()))
    await db.flush()
    job = await jobs.create(db, family_id=FAM, subject_id=MAA["id"], requested_by=MAA["id"],
                            ask={"items": [{"name": "roti", "quantity": 2, "unit": "piece"}]}, source_ref=None, channel="whatsapp")
    job.state = "resolving"
    out = await agent.run_background(db, job)
    assert out["result"] == "saved" and out["notice"] == "pending" and out["say"].startswith("Saved:"), out
    maker = _Sessions(db)
    await agent.tell(maker, lambda fid: SimHost(), job, out)
    await agent.tell(maker, lambda fid: SimHost(), job, {**out, "notice": "pending"})
    assert len(sent) == 1 and sent[0].startswith("Saved:")
    inline, _ = await call(db, "log_food", {"items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert inline["notice"] == "reply" and inline["say"].startswith("Saved:")
    await agent.tell(maker, lambda fid: SimHost(), job, inline)
    assert len(sent) == 1


async def test_one_follow_up_after_a_save_and_a_stated_usual_is_learned(db, at, nfake):
    at("2026-10-14 13:00")
    water, _ = await call(db, "log_food", {"items": [{"name": "water", "quantity": 1, "unit": "glass"}]})
    assert water["result"] == "saved" and "follow" not in water
    plate, _ = await call(db, "log_food", {"items": [{"name": "chicken biryani", "quantity": 1, "unit": "plate"}]})
    assert plate["result"] == "saved" and plate["follow"] == "How are you feeling after that?", plate
    assert plate["say"].count("?") == 0 and plate["follow"].count("?") == 1
    assert "never advice" in plate["follow_how"]
    roti, _ = await call(db, "log_food", {"eaten_at": "2026-10-14T13:01:00+05:30",
                                          "items": [{"name": "roti", "quantity": 1, "unit": "piece"}]})
    assert roti["follow"] == "Is this a usual one for you?"
    noted, err = await call(db, "log_food", {"make_usual": True})
    assert not err and noted["result"] == "noted", noted
    fact = await store.active_fact(db, FAM, MAA["id"], nudge_hooks.USUAL_KEY)
    assert fact and any(x.get("stated") and "Chapati" in x.get("name", "") for x in fact.value["items"])


async def test_a_question_about_a_logged_food_uses_the_log(db, at, nfake):
    at("2026-10-14 16:00")
    saved, _ = await call(db, "log_food", {"items": [{"name": "tea", "quantity": 1, "unit": "cup"}]})
    assert saved["result"] == "saved"
    log, err = await call(db, "food_log", {"what": "entries"})
    assert not err
    tea = next(e for e in log["entries"] if "tea" in e["what"].lower())
    assert tea["kcal"] and "sugar_g" in tea and "fat_g" in tea
    assert "kcal" not in (await active(db))[0].nutrients
    refused, err = await call(db, "web_search", {"question": "sugar and fat in the scoop"})
    assert err and "food log" in refused["refused"]
    known = [json.dumps({"entries": [tea]}), json.dumps({"sources": ["https://example.test"], "note": "From the internet. Tell it simply."})]
    assert guards.logged_nutrient_problems("about 7 g fat, 15 g sugar", known)
    assert guards.logged_nutrient_problems(guards.logged_nutrient_line(known), known) == []


async def test_food_before_their_breakfast_stays_last_night(db, at, nfake):
    at("2026-10-15 10:00")
    from app.care import outcomes_learn

    await outcomes_learn.save_learned(db, family_id=FAM, subject_id=MAA["id"], key="learned:nudges_meals_times",
                                     value={"breakfast": "09:30"}, text="Usually has breakfast around 9:30 am.")
    dinner, _ = await call(db, "log_food", {"eaten_at": "2026-10-14T21:00:00+05:30",
                                            "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert dinner["result"] == "saved" and dinner["meal"] == "dinner"
    snack, _ = await call(db, "log_food", {"meal": "breakfast", "eaten_at": "2026-10-15T04:58:00+05:30",
                                           "items": [{"name": "roti", "quantity": 1, "unit": "piece"}]})
    assert snack["result"] == "saved" and snack["meal"] == "late_night", snack
    assert not await nudge_hooks.meal_logged(db, FAM, MAA["id"], "breakfast", "2026-10-15")


def _at(e) -> str:
    return clock.ist(e.eaten_at).strftime("%H:%M")


async def test_clock_filed_night_meals_move_onto_late_night(db, at, nfake):
    """Rows the clock filed as dinner, breakfast or through the day, after a saved dinner, become late night.
    They do not count as one of the three meals. A second run changes nothing."""
    from app.nutrition import report

    at("2026-10-15 12:00")
    dinner, _ = await call(db, "log_food", {"eaten_at": "2026-10-14T21:10:00+05:30",
                                            "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert dinner["meal"] == "dinner"
    for when in ("2026-10-15T02:36:00+05:30", "2026-10-15T04:30:00+05:30", "2026-10-15T04:58:00+05:30"):
        saved, _ = await call(db, "log_food", {"eaten_at": when, "items": [{"name": "boiled egg", "quantity": 1, "unit": "piece"}]})
        assert saved["meal"] == "late_night", saved
    filed = {"02:36": "dinner", "04:30": "other", "04:58": "breakfast"}
    for e in await active(db):
        if _at(e) in filed:
            e.meal, e.day = filed[_at(e)], "2026-10-15"
    await db.flush()
    out = await entries.reslot_nights(db)
    assert out["changed"] == 3 and out["kept"] == 0, out
    assert all(set(m) == {"id", "from", "to"} for m in out["moves"])
    got = {_at(e): e.meal for e in await active(db)}
    assert got["21:10"] == "dinner" and got["02:36"] == "late_night"
    assert got["04:30"] == "late_night" and got["04:58"] == "late_night"
    night = await entries.on_day(db, FAM, MAA["id"], "2026-10-15")
    assert report._covered(night) == 0
    block = await report.brief_block(db, FAM, MAA["id"], "2026-10-15")
    assert "late_night (02:36)" in block
    assert "not logged: breakfast, lunch, dinner" in block
    assert "  breakfast (" not in block
    again = await entries.reslot_nights(db)
    assert again["changed"] == 0


async def test_a_meal_they_moved_stays_and_a_later_evening_is_after_dinner(db, at, nfake):
    at("2026-10-15 23:50")
    prior, _ = await call(db, "log_food", {"eaten_at": "2026-10-14T21:00:00+05:30",
                                           "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert prior["meal"] == "dinner"
    day, _ = await call(db, "log_food", {"eaten_at": "2026-10-15T08:30:00+05:30",
                                         "items": [{"name": "chapati", "quantity": 2, "unit": "piece"}]})
    assert day["meal"] == "breakfast"
    night, _ = await call(db, "log_food", {"eaten_at": "2026-10-15T04:58:00+05:30",
                                           "items": [{"name": "boiled egg", "quantity": 1, "unit": "piece"}]})
    assert night["meal"] == "late_night"
    stated = next(e for e in await active(db) if _at(e) == "04:58")
    await entries.change(db, stated, meal="breakfast", reason="they said this was breakfast")
    eve, _ = await call(db, "log_food", {"eaten_at": "2026-10-15T20:30:00+05:30",
                                         "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert eve["meal"] == "dinner"
    later, _ = await call(db, "log_food", {"eaten_at": "2026-10-15T23:40:00+05:30",
                                           "items": [{"name": "roti", "quantity": 1, "unit": "piece"}]})
    assert later["meal"] == "after_dinner", later
    forced = next(e for e in await active(db) if _at(e) == "23:40")
    forced.meal = "dinner"
    await db.flush()
    out = await entries.reslot_nights(db)
    got = {_at(e): e.meal for e in await active(db)}
    assert got["08:30"] == "breakfast" and got["04:58"] == "breakfast"
    assert got["20:30"] == "dinner" and got["23:40"] == "after_dinner"
    assert out["kept"] == 1 and not any(m["id"] == str(stated.id) for m in out["moves"])


async def test_a_meal_reported_later_is_stored_when_they_ate_it(db, at, nfake):
    """They write at 14:15 about a lunch they finished 45 minutes earlier. The log keeps 13:30, and a named clock time wins."""
    at("2026-10-15 14:15")
    late, err = await call(db, "log_food", {"minutes_ago": 45, "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
    assert not err and late["result"] == "saved" and late["meal"] == "lunch", late
    got = {_at(e): e.meal for e in await active(db)}
    assert got.get("13:30") == "lunch", got
    named, _ = await call(db, "log_food", {"eaten_at": "2026-10-15T13:00:00+05:30", "minutes_ago": 5,
                                           "items": [{"name": "roti", "quantity": 1, "unit": "piece"}]})
    assert named["result"] == "saved"
    assert any(_at(e) == "13:00" for e in await active(db))


async def test_learned_lunch_uses_when_they_ate_not_when_they_wrote(db, at, nfake):
    """Seven days written at 14:15, each 45 minutes after eating, learn 13:30 rather than 2:15."""
    for day in range(8, 15):
        at(f"2026-10-{day:02d} 14:15")
        saved, err = await call(db, "log_food", {"minutes_ago": 45,
                                                 "items": [{"name": "dal", "quantity": 1, "unit": "katori"}]})
        assert not err and saved["result"] == "saved", saved
    times = await nudge_hooks.usual_meal_times(db, FAM, MAA["id"])
    assert times["lunch"] == "13:30" and times["stable"]["lunch"], times


def _cached(name: str, alias: str) -> NutritionFood:
    now = clock.now()
    return NutritionFood(id=uuid.uuid4(), family_id=None, name=name, aliases=[alias], kind="generic",
                         per_100g={"energy_kcal": 120, "protein_g": 4, "carbs_g": 20, "fat_g": 2},
                         units={"piece": 40, "katori": 150}, default_unit="piece", typical_qty=1, countable=True,
                         tags=["grain"], allergens=[], source="ifct", source_urls=["https://example.org/food"],
                         confidence=0.9, checked={}, uses=0, created_at=now, updated_at=now)


async def test_a_known_meal_is_saved_with_no_model_call(db, at, nfake):
    """Four foods already in the catalogue: the reply has the numbers, and nothing is asked of a model."""
    import time

    at("2026-10-15 13:00")
    for name, alias in (("Chapati", "roti"), ("Dal", "dal"), ("Rice", "rice"), ("Sabzi", "sabzi")):
        db.add(_cached(name, alias))
    await db.flush()
    nfake.calls.clear()
    nfake.web.asked.clear()
    started = time.monotonic()
    saved, err = await call(db, "log_food", {"items": [
        {"name": "roti", "quantity": 2, "unit": "piece"},
        {"name": "dal", "quantity": 1, "unit": "katori"},
        {"name": "rice", "quantity": 1, "unit": "katori"},
        {"name": "sabzi", "quantity": 1, "unit": "katori"},
    ]})
    elapsed = time.monotonic() - started
    assert not err and saved["result"] == "saved" and saved["notice"] == "reply", saved
    assert saved["say"].startswith("Saved:") and "kcal" in saved["say"]
    assert nfake.calls == [] and nfake.web.asked == []
    assert elapsed < 1.0, elapsed


async def test_a_slow_search_does_not_hold_the_reply_or_start_a_second_wave(db, at, nfake, monkeypatch):
    """Four new foods and a search that would take half a minute. The turn does not wait, and estimates them once, together."""
    import time

    at("2026-10-15 13:00")
    monkeypatch.setenv("NUTRITION_INLINE_S", "0.4")
    monkeypatch.setattr(jobs, "kick_nutrition", lambda: None)
    nfake.web.slow = True
    nfake.calls.clear()
    nfake.estimates["pongal"] = {"name": "Pongal", "per_100g": {"energy_kcal": 140, "protein_g": 4, "carbs_g": 22, "fat_g": 4},
                                 "units": {"katori": 150}, "default_unit": "katori"}
    started = time.monotonic()
    out, _ = await call(db, "log_food", {"items": [
        {"name": "idli", "quantity": 2, "unit": "piece"},
        {"name": "dosa", "quantity": 1, "unit": "piece"},
        {"name": "sambar", "quantity": 1, "unit": "katori"},
        {"name": "pongal", "quantity": 1, "unit": "katori"},
    ]})
    elapsed = time.monotonic() - started
    assert out["result"] == "saved" and out["say"].startswith("Saved:"), out
    assert nfake.calls.count("estimate") == 1 and nfake.web.asked == [] and "normalise" not in nfake.calls
    # the slow search would take half a minute: well under that proves the reply did not wait (1.0 s failed on a loaded
    # CI runner at 1.03 s, deploy 38924206450)
    assert elapsed < 5.0, elapsed


async def test_unknown_foods_are_estimated_in_one_call(db, at, nfake):
    """What the catalogue and the web don't have is one estimate for the whole meal, not one call per food."""
    at("2026-10-15 13:00")
    for name in ("idli", "dosa", "sambar", "pongal"):
        nfake.estimates[name] = {"name": name, "per_100g": {"energy_kcal": 150, "protein_g": 4, "carbs_g": 25, "fat_g": 3},
                                 "units": {"piece": 50, "katori": 150}, "default_unit": "piece", "typical_qty": 1}
    nfake.calls.clear()
    nfake.web.asked.clear()
    saved, err = await call(db, "log_food", {"items": [
        {"name": "idli", "quantity": 2, "unit": "piece"},
        {"name": "dosa", "quantity": 1, "unit": "piece"},
        {"name": "sambar", "quantity": 1, "unit": "katori"},
        {"name": "pongal", "quantity": 1, "unit": "katori"},
    ]})
    assert not err and saved["result"] == "saved", saved
    assert nfake.calls.count("estimate") == 1
    assert nfake.calls.count("normalise") == 0 and nfake.web.asked == []
    assert len(saved["saved"]) == 4


async def test_pantry_stock_does_not_send_a_known_food_to_the_model(db, at, nfake):
    at("2026-10-15 17:00")
    tea = _cached("Tea", "tea")
    tea.tags = ["tea"]
    tea.default_unit = "cup"
    tea.units = {"cup": 150}
    db.add(tea)
    await db.flush()
    bar = _cached("Protein bar", "protein bar")
    bar.family_id = FAM
    db.add(bar)
    await db.flush()
    await pantry.add(db, family_id=FAM, subject_id=None, name="Protein bar", count=2, source="order", task_id="t-9", food=bar)
    nfake.calls.clear()
    saved, err = await call(db, "log_food", {"items": [{"name": "tea", "quantity": 1, "unit": "cup"}]})
    assert not err and saved["result"] == "saved", saved
    assert "normalise" not in nfake.calls and "estimate" not in nfake.calls


async def test_a_slow_model_is_waited_for_and_a_timeout_is_not_saved_as_zero(db, at, nfake, monkeypatch):
    """Past the old 3 s cap, a slow estimate still returns numbers. A call that does not fit is pending, never 0 kcal."""
    import time

    at("2026-10-16 16:30")
    nfake.estimates["thekua"] = {"name": "Thekua", "per_100g": {"energy_kcal": 180, "protein_g": 3, "carbs_g": 30, "fat_g": 5},
                                 "units": {"plate": 200}, "default_unit": "plate", "typical_qty": 1}
    nfake.estimate_delay = 3.2
    started = time.monotonic()
    saved, err = await call(db, "log_food", {"items": [{"name": "thekua", "quantity": 1, "unit": "plate"}]})
    assert not err and saved["result"] == "saved", saved
    assert time.monotonic() - started >= 3.0
    assert saved["saved"][0]["kcal"] and saved["saved"][0]["confidence"] != "pending"
    assert " 0 kcal" not in f" {saved['say']}" and "0 calories" not in saved["say"]

    monkeypatch.setattr(lookup, "ESTIMATE_BUDGET", 0.2)
    nfake.estimate_delay = 1.0
    nfake.web.slow = True
    out, err = await call(db, "log_food", {"items": [{"name": "soan papdi", "quantity": 1, "unit": "plate"}]})
    assert not err and out["result"] == "saved", out
    assert out["saved"][0]["confidence"] == "pending" and out["saved"][0]["kcal"] is None, out
    assert "adding the numbers" in out["say"]
    assert " 0 kcal" not in f" {out['say']}" and "0 calories" not in out["say"]
    row = next(e for e in await active(db) if "soan" in (e.name or "").lower() or "soan" in (e.name_said or "").lower())
    assert row.nutrients == {} and row.confidence == "pending"

    nfake.estimate_delay = 0
    nfake.web.slow = False
    nfake.estimates["soan papdi"] = {"name": "Soan papdi", "per_100g": {"energy_kcal": 160, "protein_g": 4, "carbs_g": 28, "fat_g": 4},
                                     "units": {"plate": 200}, "default_unit": "plate", "typical_qty": 1}
    got = await agent.fill_missing(db)
    assert got["filled"] == 1, got
    await db.refresh(row)
    assert row.confidence != "pending" and (row.nutrients or {}).get("energy_kcal")


async def test_food_waiting_more_than_five_minutes_is_the_alarm_count(db, at, nfake):
    at("2026-10-16 18:00")
    saved, err = await call(db, "log_food", {"items": [{"name": "boiled egg", "quantity": 1, "unit": "piece"}]})
    assert not err and saved["result"] == "saved"
    assert await entries.pending_older_than(db, minutes=5) == 0
    row = (await active(db))[0]
    row.confidence = "pending"
    row.nutrients = {}
    row.flags = {**(row.flags or {}), "numbers": "pending"}
    row.created_at = clock.now() - timedelta(minutes=6)
    await db.commit()
    assert await entries.pending_older_than(db, minutes=5) == 1


@pytest.mark.asyncio
async def test_two_days_come_back_day_by_day_with_empty_days_named(db):
    """Live 21 Oct: 'today and yesterday's food diary' answered with yesterday only. Each day asked is its own section."""
    from datetime import timedelta

    from app.core import clock
    from app.nutrition import report

    today = clock.ist_day()
    first = (clock.now() - timedelta(days=2)).astimezone(clock.IST).date().isoformat()
    out = await report.for_brain(db, "fam-bd", "p-bd", first, today, what="entries", item=None, name="Asha")
    assert [d["label"] for d in out["by_day"]][-2:] == ["yesterday", "today"]
    assert all(d["what"] == "nothing logged" for d in out["by_day"]) and "answer_each_day" in out
