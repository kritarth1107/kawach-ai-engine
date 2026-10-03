"""Rules the model cannot talk its way around. Pure functions; the loop and tools call them."""

from __future__ import annotations

import re
from dataclasses import dataclass

ALERT_REASONS = {
    "red_flag": "Medical red flag or emergency",
    "no_answer": "The elder did not answer a check",
    "safety": "High-confidence mood or safety concern",
    "approval": "An order or booking needs a caregiver's approval",
}

# Below this, a mood or safety concern goes to the dashboard instead of WhatsApp.
SAFETY_MIN_CONFIDENCE = 0.75


@dataclass
class AlertDecision:
    whatsapp: bool
    dashboard: bool
    why: str


def alert_decision(reason: str, confidence: float) -> AlertDecision:
    if reason not in ALERT_REASONS:
        return AlertDecision(False, True, f"'{reason}' is not a WhatsApp alert reason; dashboard only")
    if reason == "safety" and confidence < SAFETY_MIN_CONFIDENCE:
        return AlertDecision(False, True, "low confidence; dashboard and daily snapshot only")
    return AlertDecision(True, True, ALERT_REASONS[reason])


def issue_ref(reason: str, issue: str, day: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (issue or "general").lower()).strip("-")[:60] or "general"
    return f"alert:{reason}:{slug}:{day}"


# ── vital red flags (thresholds agreed with clinical guidance for home monitoring) ──


def vital_red_flag(kind: str, value: str) -> str | None:
    kind = (kind or "").lower()
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", value or "")]
    if not nums:
        return None
    if kind in ("bp", "blood_pressure") and len(nums) >= 2:
        sys_, dia = nums[0], nums[1]
        if sys_ >= 180 or dia >= 120:
            return f"very high BP {value}"
        if sys_ < 90 or dia < 60:
            return f"low BP {value}"
    if kind in ("sugar", "glucose", "blood_sugar"):
        if nums[0] >= 300:
            return f"very high sugar {value}"
        if nums[0] < 70:
            return f"low sugar {value}"
    if kind in ("temperature", "temp"):
        t = nums[0]
        if (t > 50 and t >= 103) or (t <= 50 and t >= 39.4):
            return f"high fever {value}"
    if kind in ("spo2", "oxygen") and nums[0] < 92:
        return f"low oxygen {value}"
    if kind in ("pulse", "heart_rate") and (nums[0] > 120 or nums[0] < 45):
        return f"abnormal pulse {value}"
    return None


# ── reply guard ────────────────────────────────────────────────────────────────

DISHES = re.compile(
    r"\b(paneer(?: butter masala)?|butter masala|biryani|dal makhani|khichdi|rajma|chole|paratha|idli|dosa|"
    r"samosa|halwa|kheer|pulao|palak|upma|poha|sambar|dhokla|kadhi|besan chilla|chilla|pakora|aloo gobi|"
    r"bhindi|baingan bharta|thepla|uttapam|vada|pongal|rasam)\b",
    re.I,
)
TECH = re.compile(r"\b(browser|model|server|api|mcp|captcha|bots?|automated|automation|automatic reminders?|llm|gemini|claude|tool call|(?:my|the|our) system|database|logged in the system)\b", re.I)

PROMISE = re.compile(
    r"\b(tell|ask|inform|message|text|remind|let\b.{0,20}\bknow|bol(?:ungi|ti|bo|e|o|na)?|bata(?:ungi|ti|ti hoon|o)?|kah(?:ungi|engi|ti)?|"
    r"keh(?:ti|ungi)?|janab[oa]|sang(?:en|te|ate|u|ungi)|bolchi|boli)\b",
    re.I,
)


def unreachable_promises(reply: str, unreachable: list[str]) -> list[str]:
    """A promise to pass something on to someone Saheli cannot message (a helper, neighbour, doctor)."""
    hits = []
    for sentence in re.split(r"[.!?\n\u0964]+", reply):
        for name in unreachable:
            if name and re.search(rf"\b{re.escape(name)}\b", sentence, re.I) and PROMISE.search(sentence):
                hits.append(name)
    return sorted(set(hits))


HINGLISH = {
    "hai", "hain", "nahi", "nahin", "kya", "kar", "karo", "kardo", "mujhe", "aap", "aapko", "main", "raha", "rahi",
    "theek", "thik", "haan", "ji", "acha", "achha", "kab", "kaise", "mein", "bhi", "aur", "beta", "dawai", "goli",
    "kyun", "kuch", "abhi", "batao", "bataiye", "lijiye", "dijiye", "mera", "meri", "tha", "thi", "gaya", "gayi",
}


def hinglish_words(text: str) -> int:
    return sum(1 for w in re.findall(r"[a-z]+", (text or "").lower()) if w in HINGLISH)


def reply_problems(reply: str, *, known_text: str, avoid_words: list[str], user_text: str, unreachable: list[str] | None = None) -> list[str]:
    """Things the elder must never read. known_text is everything in memory the reply may draw on."""
    problems = []
    for name in unreachable_promises(reply, unreachable or []):
        problems.append(f"promises to pass something to {name}, whom you cannot message; say what you can do instead (remind them, tell the family)")
    known = known_text.lower() + "\n" + user_text.lower()
    for m in DISHES.finditer(reply):
        if m.group(0).lower() not in known:
            problems.append(f"mentions the dish '{m.group(0)}', which is not in their memory; suggest only saved dishes or ask")
    if TECH.search(reply):
        problems.append("mentions technology words; speak like a person")
    for w in avoid_words:
        if w and re.search(rf"\b{re.escape(w)}\b", reply, re.I):
            problems.append(f"uses '{w}', which they asked not to be called")
    if hinglish_words(user_text) >= 2 and not re.search(r"[\u0900-\u097F]", reply) and len(reply.split()) >= 4 and hinglish_words(reply) == 0:
        problems.append("they wrote in Hinglish (Roman letters) and the reply is in English; reply in Hinglish like they did")
    for price in re.findall(r"₹\s?\d[\d,]*(?:\.\d+)?", reply):
        if price.replace(" ", "") not in known.replace(" ", ""):
            problems.append(f"states the price {price}, which no tool returned")
    return problems


ALLERGEN_WORDS = {
    "nut": ["nut", "peanut", "groundnut", "moongphali", "almond", "badam", "cashew", "kaju", "walnut", "akhrot", "pista", "pistachio"],
    "milk": ["milk", "doodh", "dairy", "paneer", "curd", "dahi", "ghee", "butter", "cheese", "cream", "malai", "lassi", "kheer", "khoya", "mawa", "milkshake", "chaas"],
    "gluten": ["gluten", "wheat", "atta", "maida", "bread", "roti", "chapati", "suji", "sooji", "rava", "barley", "biscuit"],
    "egg": ["egg", "anda", "omelette", "omelet", "mayonnaise", "mayo"],
    "shellfish": ["shellfish", "prawn", "shrimp", "jhinga", "crab", "lobster"],
    "soy": ["soy", "soya", "tofu"],
}


def allergen_words(allergen: str) -> list[str]:
    a = (allergen or "").lower().strip()
    for key, words in ALLERGEN_WORDS.items():
        if a == key or a in words or a.rstrip("s") == key:
            return words
    return [a] if a else []


def order_conflicts(goal: str, allergies: list[str], never_order: list[str]) -> list[str]:
    text = (goal or "").lower()
    hits = []
    for a in allergies:
        word = next((w for w in allergen_words(a) if re.search(rf"\b{re.escape(w)}", text)), None)
        if word:
            hits.append(f"allergy to {a} ({word})")
    hits += [f"never order: {n}" for n in never_order if n and n.lower() in text]
    return hits
