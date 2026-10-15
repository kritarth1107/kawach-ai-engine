"""The save line names the nutrients that matter for that portion, from the numbers, not the food's name."""

from app.nutrition import nutrients as N


def test_ice_cream_names_sugar_and_fat_not_protein():
    # The founder's bowl: 140 g, the numbers he was given when he asked.
    line = N.highlight_phrase({"energy_kcal": 294, "sugar_g": 30.8, "fat_g": 15.5, "protein_g": 4.9, "carbs_g": 33.6})
    assert line == "294 kcal, 31 g sugar, 16 g fat"
    assert "protein" not in line


def test_eggs_name_protein():
    # 4 boiled eggs, 200 g, the catalogue values the food log already uses.
    line = N.highlight_phrase({"energy_kcal": 310, "protein_g": 25.2, "fat_g": 21.2, "carbs_g": 2.2, "sodium_mg": 248,
                               "cholesterol_mg": 746})
    assert line == "310 kcal, 25 g protein, 21 g fat"
    assert "cholesterol" not in line


def test_namkeen_names_sodium():
    line = N.highlight_phrase({"energy_kcal": 160, "protein_g": 4, "carbs_g": 14, "fat_g": 10, "sodium_mg": 700, "fibre_g": 1})
    assert line.startswith("160 kcal, ") and "mg sodium" in line
    assert line.index("sodium") < line.index("fat")


def test_dal_and_rice_name_carbs_and_protein():
    # A plate: rice and dal, totals only. No food names are read.
    line = N.highlight_phrase({"energy_kcal": 420, "protein_g": 14, "carbs_g": 70, "fat_g": 6, "fibre_g": 8, "sodium_mg": 400})
    assert "kcal" in line and "carbs" in line and "protein" in line
    assert line.index("carbs") < line.index("protein")


def test_a_missing_nutrient_is_not_shown_as_zero():
    line = N.highlight_phrase({"energy_kcal": 100, "protein_g": 12})
    assert line == "100 kcal, 12 g protein"
    assert "0 g" not in line and "0 mg" not in line
    assert N.highlight_phrase({"energy_kcal": 113, "protein_g": 0, "sugar_g": 27.5}) == "113 kcal, 28 g sugar"
    assert N.highlight_phrase({}) == ""
    assert N.highlight_phrase({"protein_g": 20}) == ""


def test_a_condition_or_a_target_breaks_a_near_tie():
    close = {"energy_kcal": 180, "sugar_g": 10, "fat_g": 14.1}
    assert N.highlight_phrase(close).startswith("180 kcal, 14 g fat")
    assert N.highlight_phrase(close, N.relevant_from(["condition:type_2_diabetes"], [])).startswith("180 kcal, 10 g sugar")
    salty = {"energy_kcal": 200, "protein_g": 10.2, "sodium_mg": 376}
    assert "protein" in N.highlight_phrase(salty).split(",")[1]
    assert N.highlight_phrase(salty, N.relevant_from(["condition:high_bp"], [])).split(",")[1].endswith("sodium")
    assert "potassium" in N.highlight_phrase(
        {"energy_kcal": 150, "protein_g": 8, "potassium_mg": 700, "carbs_g": 18},
        N.relevant_from(["condition:kidney_disease"], ["potassium_mg"]))


def test_a_flavour_folds_into_the_dish_and_one_amount_stays_one_portion():
    from app.nutrition.lookup import merge_parts

    items = [{"name": "chikoo"}, {"name": "chocolate ice cream", "quantity": 140, "unit": "g"}]
    folded = merge_parts(items, [{"i": 0, "part_of": 1}, {"i": 1, "part_of": None}])
    assert folded == [{"name": "chocolate ice cream, chikoo", "quantity": 140, "unit": "g"}]
    # the amount was copied onto both names: it is still one portion, not the sum
    both = [{"name": "vanilla", "quantity": 150, "unit": "g"}, {"name": "chocolate ice cream", "quantity": 150, "unit": "g"}]
    once = merge_parts(both, [{"i": 0, "part_of": 1}])
    assert once[0]["quantity"] == 150 and len(once) == 1
    # the amount sat on the flavour: it moves to the dish, still once
    moved = merge_parts(
        [{"name": "ice cream"}, {"name": "mango", "quantity": 200, "unit": "g"}],
        [{"i": 1, "part_of": 0}])
    assert moved == [{"name": "ice cream, mango", "quantity": 200, "unit": "g"}]
    separate = merge_parts(
        [{"name": "roti", "quantity": 2, "unit": "piece"}, {"name": "dal", "quantity": 1, "unit": "katori"}],
        [{"i": 0, "part_of": None}, {"i": 1, "part_of": None}])
    assert len(separate) == 2
    cycle = merge_parts([{"name": "a", "quantity": 1}, {"name": "b", "quantity": 1}], [{"i": 0, "part_of": 1}, {"i": 1, "part_of": 0}])
    assert len(cycle) == 2


def test_a_reply_that_drops_the_saved_numbers_is_sent_back():
    from app.brain import guards

    say = "Saved: dal (1 katori), rice (1 katori), 369 kcal, 66 g carbs, 14 g protein."
    known = ['{"result": "saved", "say": "' + say + '"}']
    assert guards.food_line_problems("Did you have a glass of water with your lunch?", known)
    assert guards.food_line_problems("Saved: dal and rice, 369 kcal, 66 g carbs. Did you have water?", known) == []
    assert guards.saved_food_line(known) == say
    confirm = '{"result": "needs_input", "say": "Looks like 2 boiled egg, about 215 kcal, 14 g protein. Right?"}'
    assert guards.food_line_problems("Looks like 2 boiled eggs. Is that correct?", [confirm])
    assert guards.food_line_problems("Looks like 2 boiled egg, about 215 kcal. Right?", [confirm]) == []


def test_a_bengali_save_is_not_sent_as_the_english_line():
    from app.brain import guards

    say = "Saved: 1 Red Bull (250 ml), 113 kcal, 28 g sugar — tell me if it was more."
    profile = {"saved": {"language": "bn"}, "now": "bengali"}
    line = guards.saved_line_for(profile, say)
    assert guards.script_of(line) == "bengali"
    assert "113" in line
    assert guards.saved_line_for({"saved": {"language": "en"}}, say) == say


def test_a_new_medicine_order_asks_what_happened_and_confirms_the_address():
    from app.brain import guards

    known = ['start_task: {"status": "not started yet", "ask_what_happened": true, "confirm_address": "Home"}']
    assert guards.medicine_order_problems("Should I deliver this to Home?", known)
    split = "Did something hurt?\n---\nShould I deliver this to Home?"
    assert guards.medicine_order_problems(split, known) == []
    assert guards.medicine_order_problems("Should I deliver this to Home?", ["start_task: {\"confirm_address\": \"Home\"}"]) == []


def test_high_fibre_is_named_from_the_grams():
    line = N.highlight_phrase({"energy_kcal": 180, "protein_g": 6, "carbs_g": 28, "fat_g": 3, "fibre_g": 12, "sodium_mg": 80})
    assert "fibre" in line and line.index("fibre") < line.index("carbs")
