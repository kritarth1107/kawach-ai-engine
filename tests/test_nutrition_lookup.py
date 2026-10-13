"""The lookup chain (plan §N3): the family's foods, the generic cache by AI-normalised name, Open Food Facts by barcode or
brand, the grounded web look-up with its data checks, then an estimate."""

import pytest

from app.nutrition import lookup
from app.nutrition.models import NutritionFood
from tests import nutrition_fakes

nfake = nutrition_fakes.nfake  # the fixture

FAM = "fam-lookup"


async def test_open_food_facts_by_barcode_comes_before_the_web(db, at, nfake):
    at("2026-10-14 10:00")
    nfake.off["barcode"]["8901234567890"] = lookup.from_off({
        "product_name": "Masala Oats", "brands": "OatCo", "code": "8901234567890", "serving_quantity": 40,
        "nutriments": {"energy-kcal_100g": 380, "proteins_100g": 11, "carbohydrates_100g": 62, "fat_100g": 9, "sodium_100g": 0.8}})
    nfake.decisions["masala oats"] = {"kind": "packaged"}
    got = await lookup.resolve(db, FAM, [{"name": "masala oats", "barcode": "8901234567890"}])
    f = got[0].food
    assert f is not None and f.source == "off" and f.per_100g["sodium_mg"] == 800 and f.serving_g == 40
    assert nfake.web.asked == []  # only the barcode left India


def test_open_food_facts_mapping_and_units():
    f = lookup.from_off({"product_name": "Bar", "brands": "X, Y", "nutriments": {"energy_100g": 1674, "proteins_100g": 20, "iron_100g": 0.004},
                         "allergens_tags": ["en:milk", "en:soybeans"]})
    assert f and f["per_100g"]["energy_kcal"] == 400.1 and f["per_100g"]["iron_mg"] == 4 and f["brand"] == "X" and f["allergens"] == ["milk", "soybeans"]
    assert lookup.from_off({"product_name": "no energy"}) is None


async def test_values_that_do_not_add_up_are_looked_up_again_then_kept_as_rough(db, at, nfake):
    at("2026-10-14 10:00")
    nfake.web.foods["mystery laddoo"] = {"name": "Mystery laddoo", "per_100g": {"energy_kcal": 900, "protein_g": 5, "carbs_g": 50, "fat_g": 10},
                                         "units": {"piece": 40}}
    got = await lookup.resolve(db, FAM, [{"name": "mystery laddoo"}])
    f = got[0].food
    assert nfake.web.asked == ["mystery laddoo", "mystery laddoo"]  # one more look-up
    assert f is not None and f.checked["rough"] and f.confidence < 0.5 and got[0].estimate


async def test_a_familys_recipe_stays_with_the_family(db, at, nfake):
    at("2026-10-14 10:00")
    nfake.decisions["mummy's dal"] = {"kind": "recipe", "canonical": "Dal (home recipe)"}
    nfake.estimates["dal (home recipe)"] = {"name": "Dal (home recipe)", "per_100g": {"energy_kcal": 120, "protein_g": 7, "carbs_g": 16, "fat_g": 3},
                                            "units": {"katori": 150}}
    got = await lookup.resolve(db, FAM, [{"name": "mummy's dal"}])
    assert got[0].food is not None and got[0].food.family_id == FAM and nfake.web.asked == []
    other = await lookup.resolve(db, "fam-other", [{"name": "mummy's dal"}])
    assert other[0].food is None or other[0].food.family_id != FAM  # never another family's


async def test_generic_foods_are_shared_and_hold_no_family(db, at, nfake):
    at("2026-10-14 10:00")
    a = await lookup.resolve(db, FAM, [{"name": "boiled egg"}])
    b = await lookup.resolve(db, "fam-other", [{"name": "boiled egg"}])
    assert a[0].food is not None and b[0].food is not None and a[0].food.id == b[0].food.id and a[0].food.family_id is None
    assert nfake.web.asked == ["boiled egg"]


async def test_nothing_found_anywhere_asks_what_is_in_it(db, at, nfake):
    at("2026-10-14 10:00")
    got = await lookup.resolve(db, FAM, [{"name": "unheard dish"}])
    assert got[0].food is None and got[0].needs == "photo"


async def test_a_web_look_up_past_the_turns_time_goes_to_the_background(db, at, nfake):
    import time

    at("2026-10-14 10:00")
    nfake.web.slow = True
    with pytest.raises(lookup.Slow):
        await lookup.resolve(db, FAM, [{"name": "boiled egg"}], deadline=time.monotonic() + 0.3)


async def test_the_same_generic_food_written_twice_is_one_row(db, at, nfake):
    at("2026-10-14 10:00")
    one = await lookup.save_food(db, {"name": "Poha", "per_100g": {"energy_kcal": 180, "protein_g": 3, "carbs_g": 30, "fat_g": 5}, "source": "ifct"},
                                 family_id=None, alias="poha")
    two = await lookup.save_food(db, {"name": "poha", "per_100g": {"energy_kcal": 185, "protein_g": 3, "carbs_g": 31, "fat_g": 5}, "source": "web"},
                                 family_id=None, alias="chivda poha")
    assert one.id == two.id and "chivda poha" in two.aliases
    rows = (await db.execute(NutritionFood.__table__.select().where(NutritionFood.family_id.is_(None)))).all()
    assert len(rows) == 1


def test_model_json_is_read_from_fences():
    assert lookup.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert lookup.parse_json("no json") == {}
