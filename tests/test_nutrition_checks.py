"""Nutrition maths and data checks (plan §N3.2, §N3.3): Atwater, plausible ranges, unit bounds, portions, personal sizes."""

from datetime import datetime

from app.core import clock
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.entries import compute
from app.nutrition.models import NutritionFood


def food(**kw):
    base = dict(per_100g={"energy_kcal": 155, "protein_g": 12.6, "carbs_g": 1.1, "fat_g": 10.6}, units={"piece": 50}, default_unit="piece",
                typical_qty=2, serving_g=None, tags=["egg"])
    base.update(kw)
    return NutritionFood(**base)


def test_atwater_passes_for_real_values_and_catches_wrong_energy():
    assert N.check_per_100g({"energy_kcal": 155, "protein_g": 12.6, "carbs_g": 1.1, "fat_g": 10.6}).ok  # boiled egg
    bad = N.check_per_100g({"energy_kcal": 400, "protein_g": 12.6, "carbs_g": 1.1, "fat_g": 10.6})
    assert not bad.ok and bad.atwater and bad.atwater > 2 and "energy" in bad.problems[0]


def test_tiny_values_are_not_judged_on_ratio():
    assert N.check_per_100g({"energy_kcal": 1, "protein_g": 0.1, "carbs_g": 0.2, "fat_g": 0}).ok  # black tea


def test_implausible_ranges_are_problems():
    c = N.check_per_100g({"energy_kcal": 950, "protein_g": 60, "carbs_g": 60, "fat_g": 5, "sodium_mg": 50000})
    assert not c.ok
    text = " ".join(c.problems)
    assert "energy_kcal 950" in text and "sodium_mg" in text and "protein + carbs + fat" in text


def test_clean_reads_label_strings_and_drops_unknown_keys():
    assert N.clean({"energy_kcal": "250 kcal", "protein_g": "12.5g", "colour": 3, "fat_g": -1}) == {"energy_kcal": 250.0, "protein_g": 12.5}


def test_units_outside_their_bounds_are_dropped():
    assert P.clean_units({"piece": 50, "katori": 900, "tsp": 5, "spoonful": 10}) == {"piece": 50.0, "tsp": 5.0}


def test_grams_from_food_unit_personal_size_and_usual_portion():
    egg = food()
    assert P.grams_for(egg, 4, "piece").grams == 200 and not P.grams_for(egg, 4, "piece").estimate
    dal = food(units={"katori": 150}, default_unit="katori", typical_qty=1)
    assert P.grams_for(dal, 1, "katori").grams == 150
    assert P.grams_for(dal, 1, "katori", {"katori_ml": 200}).grams == 200  # "my katori is big"
    usual = P.grams_for(egg, None, None)
    assert usual.grams == 100 and usual.estimate  # 2 pieces, marked an estimate
    assert P.grams_for(food(units={}), 2, "glass").grams == 500  # the usual glass
    poha = food(units={"katori_small": 55}, default_unit="katori_small", typical_qty=1)
    assert P.grams_for(poha, 1, "katori").grams == 55  # they said katori; the dish's own katori is the small one


def test_compute_scales_nutrients_and_part_eaten():
    grams, vals, est = compute(food(), 4, "piece")
    assert grams == 200 and round(vals["energy_kcal"]) == 310 and round(vals["protein_g"]) == 25 and not est  # the plan's eggs
    grams, vals, _ = compute(food(), 2, "piece", portion="half")
    assert grams == 50 and round(vals["energy_kcal"]) == 78


def test_water_counts_as_water_drunk():
    water = food(per_100g={"energy_kcal": 0, "water_g": 100}, units={"glass": 250}, default_unit="glass", tags=["water"])
    _, vals, _ = compute(water, 2, "glass")
    assert vals["water_ml"] == 500


def test_meal_slot_from_time_only():
    def at(h, m=0):
        return datetime(2026, 10, 14, h, m, tzinfo=clock.IST)

    assert [P.meal_for(at(h)) for h in (8, 13, 17, 21, 1)] == ["breakfast", "lunch", "snack", "dinner", "dinner"]


def test_format():
    assert N.fmt(1240.4) == "1,240" and N.fmt(25.4) == "25" and N.fmt(2.5) == "2.5"
