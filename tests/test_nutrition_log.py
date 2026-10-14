"""log_food end to end through Saheli's tool (fake models): questions, photos, labels, orders, who ate, corrections, undo,
allergies, rules, pantry, the cache. Invented family; no real people."""

import json
import uuid
from datetime import timedelta

import pytest

from app.brain import tools
from app.care import store
from app.core import clock
from app.nutrition import agent, entries, jobs, lookup, pantry
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
    assert r["say"] == "Saved: 4 boiled eggs, 310 kcal, 25 g protein."
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
    assert not err and r["result"] == "looking_up", r
    assert await active(db) == []
    again, _ = await call(db, "log_food", args)
    assert again.get("already") and again["say"] == "" and again["result"] == "looking_up"
    assert await active(db) == []
    nfake.web.slow = False
    nfake.web.foods["hara bhara kebab"] = nfake.estimates["hara bhara kebab"] | {"source_kind": "web", "tags": ["snack"]}
    job = await jobs.get(db, FAM, r["job_id"])
    out = await agent.run_background(db, job)
    assert out["result"] == "saved", out
    assert "fried" in out["say"] and "high fat" in out["say"] and "high sodium" in out["say"]
    rows = await active(db)
    assert len(rows) == 1 and rows[0].flags.get("facts") == ["fried", "high fat", "high sodium"]
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
    assert r["say"] == "Saved: 1 Red Bull (250 ml), 113 kcal, 0 g protein — tell me if it was more."
    assert "How many" not in r["say"]
    row = (await active(db))[-1]
    assert row.quantity == 1 and row.unit == "serving" and row.grams == 250 and row.confidence == "estimate"
    assert row.flags.get("assumed_one") and row.flags.get("serving_ml") == 250
    r, err = await call(db, "log_food", {"items": [{"name": "tea", "said": "tea"}]})
    assert not err and r["result"] == "saved", r
    assert r["say"] == "Saved: 1 tea (150 ml), 60 kcal, 1.8 g protein — tell me if it was more."
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
    assert nfake.web.asked == ["boiled egg"]
    food = (await db.execute(NutritionFood.__table__.select())).first()
    assert food.family_id is None and food.source == "usda" and food.source_urls  # generic, shared, cited
    nfake.calls.clear()
    r, _ = await call(db, "log_food", {"items": [{"name": "Boiled egg", "quantity": 1, "unit": "piece"}]})
    assert r["result"] == "saved" and nfake.calls == [] and nfake.web.asked == ["boiled egg"]


async def test_two_questions_at_most_then_an_estimate(db, at, nfake):
    at("2026-10-14 16:00")
    nfake.decisions["sweet"] = {"needs": "which", "options": ["laddoo", "barfi"]}
    nfake.decisions["thekua"] = {"needs": "label"}
    nfake.estimates["thekua"] = {"name": "Thekua", "per_100g": {"energy_kcal": 450, "protein_g": 6, "carbs_g": 60, "fat_g": 20},
                                 "units": {"piece": 30}, "default_unit": "piece", "typical_qty": 2}
    r, _ = await call(db, "log_food", {"items": [{"name": "sweet"}]})
    assert r["needs"][0]["reason"] == "which"
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "items": [{"name": "thekua"}]})
    assert r["needs"][0]["reason"] == "label"
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "items": [{"name": "thekua", "quantity": 2, "unit": "piece"}]})
    assert r["result"] == "saved" and r["saved"][0]["confidence"] == "estimate"
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
    assert r["needs"][0]["question"] == "Looks like 2 boiled egg, 1 cup tea. Right?"
    assert await active(db) == []
    r, _ = await call(db, "log_food", {"answer_to": r["job_id"], "confirmed": True})  # the yes comes in a later message, no photo
    assert r["result"] == "saved"
    rows = await active(db)
    assert {e.confidence for e in rows} == {"photo"} and {e.source for e in rows} == {"photo"}


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
    assert [e.name for e in rows if e.status == "active"] == ["Dal, toor, cooked"]


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
    r, _ = await call(db, "log_food", {"items": [{"name": "chicken biryani", "quantity": 1, "unit": "plate"}]})
    assert r["result"] == "looking_up" and await active(db) == []
    nfake.web.slow = False
    job = await jobs.get(db, FAM, r["job_id"])
    out = await agent.run_background(db, job)
    assert out["result"] == "saved" and len(await active(db)) == 1


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
    r, _ = await call(db, "log_food", {"items": [{"name": "chapati", "quantity": 2, "unit": "piece"}, {"name": "water", "quantity": 1, "unit": "glass"}]})
    assert r["result"] == "saved" and "Energy today" in r["deviation"], r
    rows = {e.name: e for e in await active(db)}
    assert rows["Chapati (whole wheat)"].flags.get("deviation") and not rows["Water"].flags.get("deviation")


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
    assert stats["expired"] == 1 and got.state == "saved" and rows[0].source == "caregiver" and rows[0].confidence == "unquantified"


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
