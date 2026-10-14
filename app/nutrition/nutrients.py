"""Nutrient keys, the data checks on nutrition values (plan §N3.2), and the arithmetic. Data checks only: nothing here reads
what a person said.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Per-100 g keys (plan §N2), in the order the dashboard and the brief show them.
KEYS = ("energy_kcal", "protein_g", "carbs_g", "sugar_g", "added_sugar_g", "fibre_g", "fat_g", "sat_fat_g", "cholesterol_mg",
        "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "vit_d_mcg", "b12_mcg", "caffeine_mg", "water_g")
MACROS = ("energy_kcal", "protein_g", "carbs_g", "fat_g", "fibre_g", "sugar_g", "sodium_mg")
LABELS = {
    "energy_kcal": ("Energy", "kcal"), "protein_g": ("Protein", "g"), "carbs_g": ("Carbs", "g"), "sugar_g": ("Sugar", "g"),
    "added_sugar_g": ("Added sugar", "g"), "fibre_g": ("Fibre", "g"), "fat_g": ("Fat", "g"), "sat_fat_g": ("Saturated fat", "g"),
    "cholesterol_mg": ("Cholesterol", "mg"), "sodium_mg": ("Sodium", "mg"), "potassium_mg": ("Potassium", "mg"),
    "calcium_mg": ("Calcium", "mg"), "iron_mg": ("Iron", "mg"), "vit_d_mcg": ("Vitamin D", "mcg"), "b12_mcg": ("Vitamin B12", "mcg"),
    "caffeine_mg": ("Caffeine", "mg"), "water_g": ("Water in food", "g"), "water_ml": ("Water", "ml"),
}
# In entries and totals beside the per-100 g keys: plain water drunk (ml), from foods tagged "water" (water in food and in
# tea is not counted as water drunk).
TOTAL_KEYS = KEYS + ("water_ml",)
# Target fields (nutrition_target tool) → the total they are about.
TARGETS = {"kcal": "energy_kcal", "protein_g": "protein_g", "carbs_g": "carbs_g", "fat_g": "fat_g", "fibre_g": "fibre_g",
           "sugar_g": "sugar_g", "sodium_mg": "sodium_mg", "water_ml": "water_ml"}
# What a target on each nutrient means when only a number is given: a floor (eat at least), a ceiling (at most), or a goal.
TARGET_SENSE = {"kcal": "target", "protein_g": "min", "carbs_g": "max", "fat_g": "max", "fibre_g": "min", "sugar_g": "max",
                "sodium_mg": "max", "water_ml": "min"}

ATWATER_TOLERANCE = 0.20
MAX_PER_100G = {"energy_kcal": 900.0, "protein_g": 100.0, "carbs_g": 100.0, "fat_g": 100.0, "sugar_g": 100.0, "added_sugar_g": 100.0,
                "fibre_g": 100.0, "sat_fat_g": 100.0, "sodium_mg": 40000.0, "potassium_mg": 20000.0, "calcium_mg": 10000.0,
                "iron_mg": 500.0, "cholesterol_mg": 5000.0, "vit_d_mcg": 2500.0, "b12_mcg": 5000.0, "caffeine_mg": 5000.0, "water_g": 100.0}
EXTREME_KCAL = (1200, 3500)  # a daily target outside this gets one line suggesting a check with the doctor (founder Q6)
# Facts read off the food's own per-100 g numbers (UK front-of-pack "high": fat 17.5 g, sodium 0.6 g).
# Fat at 20 g or more is also an oily or fried food. These are facts about the numbers, never advice.
FAT_HIGH_G = 17.5
FAT_FRIED_G = 20.0
SODIUM_HIGH_MG = 600.0


def fact_tags(per_100g: dict | None) -> list[str]:
    """fried / high fat / high sodium, only when this food's own per-100 g numbers pass the cut-offs."""
    v = per_100g or {}
    fat = float(v.get("fat_g") or 0)
    sodium = float(v.get("sodium_mg") or 0)
    tags: list[str] = []
    if fat >= FAT_FRIED_G:
        tags.append("fried")
    if fat >= FAT_HIGH_G:
        tags.append("high fat")
    if sodium >= SODIUM_HIGH_MG:
        tags.append("high sodium")
    return tags


def clean(values: dict | None) -> dict:
    """Known keys with a number ≥ 0 (strings like "12.5 g" from a label reader are read as numbers; anything else dropped)."""
    out: dict = {}
    for k, v in (values or {}).items():
        if k not in KEYS:
            continue
        n = number(v)
        if n is not None and n >= 0:
            out[k] = round(n, 3)
    return out


def number(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        digits = ""
        for ch in s:
            if ch.isdigit() or (ch == "." and "." not in digits):
                digits += ch
            elif digits:
                break
        try:
            return float(digits) if digits else None
        except ValueError:
            return None
    return None


@dataclass
class Check:
    ok: bool = True
    atwater: float | None = None  # stated kcal / computed kcal
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "atwater": self.atwater, "problems": self.problems}


def check_per_100g(v: dict) -> Check:
    """Plausible per-100 g values and Atwater (kcal ≈ 4·protein + 4·carbs + 9·fat, ±20%)."""
    c = Check()
    if "energy_kcal" not in v:
        c.problems.append("no energy value")
    for k, top in MAX_PER_100G.items():
        if k in v and not (0 <= v[k] <= top):
            c.problems.append(f"{k} {v[k]} is outside 0–{top:g} per 100 g")
    macro = sum(v.get(k, 0.0) for k in ("protein_g", "carbs_g", "fat_g"))
    if macro > 105:  # rounding on labels
        c.problems.append(f"protein + carbs + fat = {macro:g} g per 100 g")
    if all(k in v for k in ("energy_kcal", "protein_g", "carbs_g", "fat_g")):
        calc = 4 * v["protein_g"] + 4 * v["carbs_g"] + 9 * v["fat_g"]
        stated = v["energy_kcal"]
        if max(calc, stated) >= 25:  # water, black tea, diet drinks: too small to judge a ratio
            ratio = stated / calc if calc > 0 else 99.0
            c.atwater = round(ratio, 2)
            # fibre counted in carbs (US labels) or alcohol make the stated kcal a little off; ±20% covers it
            if abs(ratio - 1) > ATWATER_TOLERANCE:
                c.problems.append(f"energy {stated:g} kcal vs {calc:.0f} from protein, carbs and fat")
    c.ok = not c.problems
    return c


def scale(per_100g: dict, grams: float, factor: float = 1.0) -> dict:
    return {k: round(x * grams / 100.0 * factor, 2) for k, x in (per_100g or {}).items() if k in KEYS}


def times(values: dict, n: float) -> dict:
    return {k: round(x * n, 2) for k, x in (values or {}).items() if k in TOTAL_KEYS}


def add(total: dict, more: dict) -> dict:
    out = dict(total)
    for k, x in (more or {}).items():
        if k in TOTAL_KEYS and isinstance(x, (int, float)):
            out[k] = round(out.get(k, 0.0) + x, 2)
    return out


def half_up(x: float) -> int:
    """232.5 → 233 (people round halves up; Python's round() would say 232)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def kcal(v: dict | None) -> int:
    return half_up(float((v or {}).get("energy_kcal") or 0))


def grams_of(v: dict | None, key: str) -> float:
    return round(float((v or {}).get(key) or 0), 1)


def fmt(n: float) -> str:
    """1240 → "1,240"; 25.4 → "25"; 2.5 → "2.5" (small amounts keep one decimal)."""
    if n >= 10 or float(n).is_integer():
        return f"{half_up(float(n)):,}"
    return f"{n:.1f}"
