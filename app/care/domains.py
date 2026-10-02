"""Care record vocabulary: domains, key shapes, and which facts need a second pair of eyes.

Keys are stable slugs so the same thing said twice lands on the same row:
    medicine:metformin, allergy:milk, diet:low_salt, naming:address_as, family:call_first
"""

from __future__ import annotations

import re
import unicodedata

DOMAINS = {
    "medicine": "Medicines: name, dose, times of day, food timing, days",
    "no_order": "Items never to order again",
    "allergy": "Allergies: food, medicine or other, with reaction",
    "diet": "Diet rules: low_salt, low_sugar, fasting, no_onion, no_spice, vegetarian, …",
    "dish": "Dishes they actually cook or eat",
    "condition": "Health conditions",
    "vital_target": "Target ranges for BP, sugar, weight",
    "routine": "Wake, sleep, walk, exercise, prayer, calls, maid or cook time",
    "occasion": "Pension day, festivals, birthdays",
    "family": "Who to call first, who pays, who must not be told, caregiver away, neighbour",
    "naming": "The name to use, words to avoid",
    "language": "Preferred language",
    "doctor": "Doctors, family doctor",
    "hospital": "Hospital preference",
    "home": "Helpers, cook, maid, home setup",
    "contact": "Other people and numbers",
    "preference": "Likes and dislikes that matter for care",
    "profile": "Emergency profile: blood_group, height, weight, insurance, mobility, id_note (one fact each)",
    "appointment": "Doctor appointments: doctor, when (YYYY-MM-DDTHH:MM IST), place, purpose, questions to ask",
}

# A change from a weaker source to these waits for a caregiver to confirm.
HEALTH_DOMAINS = {"medicine", "allergy", "condition", "vital_target"}

# Stronger sources outrank weaker ones; elder_said is trusted for preferences, not for doses.
SOURCE_RANK = {
    "prescription": 5,
    "lab": 5,
    "caregiver_said": 4,
    "dashboard": 4,
    "import": 3,
    "elder_said": 2,
    "inferred": 1,
}

COMMON_ALLERGENS = ("nut", "milk", "gluten", "egg", "shellfish", "soy")

FOOD_TIMING = ("before_food", "after_food", "with_food", "empty_stomach", "any")

SLOTS = {"morning": "08:00", "afternoon": "13:00", "evening": "18:00", "night": "21:00"}


def slug(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"\b(\d+(\.\d+)?)\s*(mg|mcg|ml|g|iu)\b", "", t)
    t = re.sub(r"\b(tablet|tab|capsule|cap|syrup)s?\b", "", t)
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t[:120] or "item"


def fact_key(domain: str, name: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"unknown care domain: {domain}")
    return f"{domain}:{slug(name)}"


def needs_confirmation(domain: str, new_source: str, old_source: str | None, changes_existing: bool) -> bool:
    """A weaker source may not overwrite or remove a health fact on its own."""
    if domain not in HEALTH_DOMAINS or not changes_existing or old_source is None:
        return False
    return SOURCE_RANK.get(new_source, 0) < SOURCE_RANK.get(old_source, 0)
