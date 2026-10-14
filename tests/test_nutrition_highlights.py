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


def test_high_fibre_is_named_from_the_grams():
    line = N.highlight_phrase({"energy_kcal": 180, "protein_g": 6, "carbs_g": 28, "fat_g": 3, "fibre_g": 12, "sodium_mg": 80})
    assert "fibre" in line and line.index("fibre") < line.index("carbs")
