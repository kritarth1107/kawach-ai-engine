"""The test world's coverage grid: every dimension a family is built from, with quotas and constraints.

All of it is invented. Phones are fake (+91 90000 xxxxx, or +99 placeholders like self-sign-up accounts); no real
person, and never the founder's family. The engine, backend and dashboard repos are public.
Plan: QUALITY_AND_NUDGES_PLAN.md Part A.
"""

from __future__ import annotations

WORLD_VERSION = 1
SEED = 20261013
PROFILE_FAMILIES = 900
STORY_FAMILIES = 100
# The world's "today" (day 0 of probes and of the story families' next week). History runs before it.
TODAY = "2026-10-19"  # a Monday

# ── A1.1 Languages: 30, each in at least 25 families ─────────────────────────────────────────────────────────────
# id: (name, engine language, engine dialect or None, script, roman, regions)
LANGS: dict[str, dict] = {
    "hi": {"name": "Hindi", "lang": "hi", "dialect": None, "script": "devanagari", "roman": False, "region": "north"},
    "hinglish": {"name": "Hinglish", "lang": "hi", "dialect": None, "script": "latin", "roman": True, "region": "north"},
    "en": {"name": "English (Indian)", "lang": "en", "dialect": None, "script": "latin", "roman": False, "region": "metro"},
    "bn": {"name": "Bengali", "lang": "bn", "dialect": None, "script": "bengali", "roman": False, "region": "bengal"},
    "mr": {"name": "Marathi", "lang": "mr", "dialect": None, "script": "devanagari", "roman": False, "region": "maharashtra"},
    "ta": {"name": "Tamil", "lang": "ta", "dialect": None, "script": "tamil", "roman": False, "region": "tamil"},
    "tanglish": {"name": "Tanglish", "lang": "ta", "dialect": None, "script": "latin", "roman": True, "region": "tamil"},
    "te": {"name": "Telugu", "lang": "te", "dialect": None, "script": "telugu", "roman": False, "region": "telugu"},
    "gu": {"name": "Gujarati", "lang": "gu", "dialect": None, "script": "gujarati", "roman": False, "region": "gujarat"},
    "kn": {"name": "Kannada", "lang": "kn", "dialect": None, "script": "kannada", "roman": False, "region": "karnataka"},
    "ml": {"name": "Malayalam", "lang": "ml", "dialect": None, "script": "malayalam", "roman": False, "region": "kerala"},
    "pa": {"name": "Punjabi", "lang": "pa", "dialect": None, "script": "gurmukhi", "roman": False, "region": "punjab"},
    "or": {"name": "Odia", "lang": "or", "dialect": None, "script": "odia", "roman": False, "region": "odisha"},
    "as": {"name": "Assamese", "lang": "as", "dialect": None, "script": "bengali", "roman": False, "region": "assam"},
    "ur": {"name": "Urdu", "lang": "ur", "dialect": None, "script": "arabic", "roman": False, "region": "urdu"},
    "ne": {"name": "Nepali", "lang": "ne", "dialect": None, "script": "devanagari", "roman": False, "region": "nepali"},
    "kok": {"name": "Konkani", "lang": "kok", "dialect": None, "script": "devanagari", "roman": False, "region": "goa"},
    "mwr": {"name": "Marwari", "lang": "hi", "dialect": "mwr", "script": "devanagari", "roman": False, "region": "rajasthan"},
    "bho": {"name": "Bhojpuri", "lang": "hi", "dialect": "bho", "script": "devanagari", "roman": False, "region": "purvanchal"},
    "mai": {"name": "Maithili", "lang": "hi", "dialect": "mai", "script": "devanagari", "roman": False, "region": "mithila"},
    "hne": {"name": "Chhattisgarhi", "lang": "hi", "dialect": "hne", "script": "devanagari", "roman": False, "region": "chhattisgarh"},
    "bgc": {"name": "Haryanvi", "lang": "hi", "dialect": "bgc", "script": "devanagari", "roman": False, "region": "haryana"},
    "awa": {"name": "Awadhi", "lang": "hi", "dialect": "awa", "script": "devanagari", "roman": False, "region": "awadh"},
    "gbm": {"name": "Garhwali", "lang": "hi", "dialect": "gbm", "script": "devanagari", "roman": False, "region": "garhwal"},
    "kfy": {"name": "Kumaoni", "lang": "hi", "dialect": "kfy", "script": "devanagari", "roman": False, "region": "kumaon"},
    "doi": {"name": "Dogri", "lang": "hi", "dialect": "doi", "script": "devanagari", "roman": False, "region": "jammu"},
    "tcy": {"name": "Tulu", "lang": "kn", "dialect": "tcy", "script": "kannada", "roman": False, "region": "tulunadu"},
    "syl": {"name": "Sylheti", "lang": "bn", "dialect": "syl", "script": "bengali", "roman": False, "region": "barak"},
    "bns": {"name": "Bundeli", "lang": "hi", "dialect": "bns", "script": "devanagari", "roman": False, "region": "bundelkhand"},
    "kth": {"name": "Kathiawadi", "lang": "gu", "dialect": "kth", "script": "gujarati", "roman": False, "region": "saurashtra"},
}
MIN_PER_LANGUAGE = 25
# Profile families: 25 per language (750) plus 150 more where most users are.
PROFILE_LANG_EXTRA = {"hi": 30, "hinglish": 25, "en": 25, "bn": 10, "mr": 10, "ta": 10, "te": 10, "gu": 10, "kn": 10, "ml": 10}
# Story families: 3 per language (90) plus 10.
STORY_LANG_EXTRA = {"hi": 2, "hinglish": 2, "en": 2, "ta": 2, "bn": 2}

# Unicode scripts, for the script check (validate) and for names written in their own script.
SCRIPT_RANGES = {
    "devanagari": [(0x0900, 0x097F)], "bengali": [(0x0980, 0x09FF)], "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)], "odia": [(0x0B00, 0x0B7F)], "tamil": [(0x0B80, 0x0BFF)], "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)], "malayalam": [(0x0D00, 0x0D7F)], "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A)],
}

# Where a second family member's language comes from in a mixed family (15 %): the younger one writes this.
MIXED_SHARE = 0.15
MIXED_YOUNG = {"default": ["en", "hinglish"], "tamil": ["en", "tanglish"]}
VOICE_SHARE = 0.10

# ── Cities: 40+ across tier 1/2/3, by region ───────────────────────────────────────────────────────────────────────
# name: (tier, state, pincode, lat, lng, regions)
CITIES: dict[str, tuple] = {
    "Delhi": (1, "Delhi", "110017", 28.5355, 77.2100, ["north", "metro", "urdu", "punjab", "haryana"]),
    "Mumbai": (1, "Maharashtra", "400050", 19.0596, 72.8295, ["maharashtra", "metro", "gujarat", "goa"]),
    "Bengaluru": (1, "Karnataka", "560034", 12.9352, 77.6245, ["karnataka", "metro", "kerala", "tamil", "telugu"]),
    "Chennai": (1, "Tamil Nadu", "600020", 13.0067, 80.2570, ["tamil", "metro"]),
    "Kolkata": (1, "West Bengal", "700029", 22.5170, 88.3640, ["bengal", "metro", "purvanchal"]),
    "Hyderabad": (1, "Telangana", "500034", 17.4126, 78.4482, ["telugu", "urdu", "metro"]),
    "Pune": (1, "Maharashtra", "411038", 18.5089, 73.8259, ["maharashtra", "metro"]),
    "Ahmedabad": (1, "Gujarat", "380015", 23.0300, 72.5300, ["gujarat", "metro"]),
    "Jaipur": (2, "Rajasthan", "302015", 26.8800, 75.8000, ["rajasthan", "north"]),
    "Lucknow": (2, "Uttar Pradesh", "226010", 26.8500, 81.0000, ["awadh", "urdu", "north"]),
    "Patna": (2, "Bihar", "800001", 25.6100, 85.1400, ["purvanchal", "mithila", "north"]),
    "Indore": (2, "Madhya Pradesh", "452010", 22.7500, 75.9000, ["north", "bundelkhand"]),
    "Bhopal": (2, "Madhya Pradesh", "462016", 23.2100, 77.4300, ["north", "urdu", "bundelkhand"]),
    "Chandigarh": (2, "Chandigarh", "160022", 30.7300, 76.7700, ["punjab", "haryana", "north"]),
    "Ludhiana": (2, "Punjab", "141001", 30.9000, 75.8500, ["punjab"]),
    "Coimbatore": (2, "Tamil Nadu", "641018", 11.0100, 76.9600, ["tamil"]),
    "Madurai": (2, "Tamil Nadu", "625001", 9.9200, 78.1200, ["tamil"]),
    "Kochi": (2, "Kerala", "682020", 9.9600, 76.3000, ["kerala"]),
    "Thiruvananthapuram": (2, "Kerala", "695010", 8.5200, 76.9400, ["kerala"]),
    "Visakhapatnam": (2, "Andhra Pradesh", "530017", 17.7300, 83.3100, ["telugu"]),
    "Vijayawada": (2, "Andhra Pradesh", "520010", 16.5100, 80.6400, ["telugu"]),
    "Mysuru": (2, "Karnataka", "570009", 12.3000, 76.6400, ["karnataka"]),
    "Mangaluru": (2, "Karnataka", "575003", 12.8700, 74.8400, ["tulunadu", "karnataka", "goa"]),
    "Surat": (2, "Gujarat", "395007", 21.1700, 72.8300, ["gujarat"]),
    "Rajkot": (2, "Gujarat", "360005", 22.2900, 70.7800, ["saurashtra", "gujarat"]),
    "Bhubaneswar": (2, "Odisha", "751007", 20.2900, 85.8200, ["odisha"]),
    "Guwahati": (2, "Assam", "781006", 26.1600, 91.7700, ["assam"]),
    "Nagpur": (2, "Maharashtra", "440010", 21.1400, 79.0600, ["maharashtra", "chhattisgarh"]),
    "Varanasi": (2, "Uttar Pradesh", "221005", 25.2800, 83.0000, ["purvanchal", "awadh"]),
    "Dehradun": (2, "Uttarakhand", "248001", 30.3200, 78.0300, ["garhwal", "kumaon", "north"]),
    "Jammu": (2, "Jammu and Kashmir", "180004", 32.7100, 74.8600, ["jammu"]),
    "Panaji": (2, "Goa", "403001", 15.4900, 73.8300, ["goa"]),
    "Siliguri": (2, "West Bengal", "734001", 26.7200, 88.4200, ["nepali", "bengal"]),
    "Jodhpur": (3, "Rajasthan", "342003", 26.2700, 73.0100, ["rajasthan"]),
    "Bikaner": (3, "Rajasthan", "334001", 28.0200, 73.3100, ["rajasthan"]),
    "Darbhanga": (3, "Bihar", "846004", 26.1500, 85.9000, ["mithila"]),
    "Gorakhpur": (3, "Uttar Pradesh", "273001", 26.7600, 83.3700, ["purvanchal"]),
    "Bilaspur": (3, "Chhattisgarh", "495001", 22.0800, 82.1400, ["chhattisgarh"]),
    "Durg": (3, "Chhattisgarh", "491001", 21.1900, 81.2800, ["chhattisgarh"]),
    "Rohtak": (3, "Haryana", "124001", 28.9000, 76.5800, ["haryana"]),
    "Hisar": (3, "Haryana", "125001", 29.1500, 75.7200, ["haryana"]),
    "Ayodhya": (3, "Uttar Pradesh", "224001", 26.7900, 82.2000, ["awadh"]),
    "Pauri": (3, "Uttarakhand", "246001", 30.1500, 78.7800, ["garhwal"]),
    "Almora": (3, "Uttarakhand", "263601", 29.6000, 79.6600, ["kumaon"]),
    "Sagar": (3, "Madhya Pradesh", "470002", 23.8400, 78.7400, ["bundelkhand"]),
    "Jhansi": (3, "Uttar Pradesh", "284001", 25.4500, 78.5700, ["bundelkhand"]),
    "Silchar": (3, "Assam", "788001", 24.8300, 92.7800, ["barak"]),
    "Karimganj": (3, "Assam", "788710", 24.8700, 92.3600, ["barak"]),
    "Udupi": (3, "Karnataka", "576101", 13.3400, 74.7400, ["tulunadu"]),
    "Junagadh": (3, "Gujarat", "362001", 21.5200, 70.4600, ["saurashtra"]),
    "Bhavnagar": (3, "Gujarat", "364001", 21.7600, 72.1500, ["saurashtra"]),
    "Darjeeling": (3, "West Bengal", "734101", 27.0400, 88.2700, ["nepali"]),
    "Gangtok": (3, "Sikkim", "737101", 27.3300, 88.6100, ["nepali"]),
    "Margao": (3, "Goa", "403601", 15.2800, 73.9600, ["goa"]),
    "Cuttack": (3, "Odisha", "753001", 20.4600, 85.8800, ["odisha"]),
    "Jorhat": (3, "Assam", "785001", 26.7500, 94.2200, ["assam"]),
    "Aligarh": (3, "Uttar Pradesh", "202001", 27.8900, 78.0800, ["urdu", "north"]),
    "Udhampur": (3, "Jammu and Kashmir", "182101", 32.9200, 75.1400, ["jammu"]),
    "Kumbakonam": (3, "Tamil Nadu", "612001", 10.9600, 79.3800, ["tamil"]),
    "Thrissur": (3, "Kerala", "680001", 10.5200, 76.2100, ["kerala"]),
    "Kolhapur": (3, "Maharashtra", "416003", 16.7000, 74.2400, ["maharashtra"]),
    "Warangal": (3, "Telangana", "506002", 17.9700, 79.5900, ["telugu"]),
    "Hubballi": (3, "Karnataka", "580020", 15.3600, 75.1200, ["karnataka"]),
    "Amritsar": (3, "Punjab", "143001", 31.6300, 74.8700, ["punjab"]),
}
NRI_ZONES = {"US": "America/New_York", "UK": "Europe/London", "UAE": "Asia/Dubai", "Singapore": "Asia/Singapore", "Australia": "Australia/Sydney"}

# ── A1.2 Household types (share of 1,000) ──────────────────────────────────────────────────────────────────────────
HOUSEHOLDS: dict[str, float] = {
    "elder_alone_india": 0.14,
    "elder_alone_nri": 0.08,
    "two_elders": 0.12,
    "elder_with_helper": 0.05,
    "grandparent_by_grandchild": 0.04,
    "in_law": 0.04,
    "siblings_share": 0.06,
    "caregiver_also_self": 0.05,
    "self_young": 0.14,
    "self_midlife": 0.12,
    "self_senior": 0.06,
    "post_surgery": 0.04,
    "early_dementia": 0.03,
    "caregiver_only": 0.03,
}
SELF_CARE_HOUSEHOLDS = {"self_young", "self_midlife", "self_senior"}

# ── A1.3 Health ────────────────────────────────────────────────────────────────────────────────────────────────────
# condition: (label, medicines it brings: [(name, dose, slots, food)], weight for elders, weight for adults)
CONDITIONS: dict[str, dict] = {
    "t2_diabetes": {"label": "Type 2 diabetes", "meds": [("Metformin", "500 mg", ["morning", "night"], "after_food"), ("Glimepiride", "1 mg", ["morning"], "before_food"), ("Sitagliptin", "100 mg", ["morning"], "any")], "elder": 30, "adult": 12},
    "t2_diabetes_insulin": {"label": "Type 2 diabetes on insulin", "meds": [("Insulin Glargine", "14 units", ["night"], "any"), ("Insulin Aspart", "6 units", ["breakfast", "lunch", "dinner"], "before_food"), ("Metformin", "1000 mg", ["morning"], "after_food")], "elder": 8, "adult": 3},
    "hypertension": {"label": "High blood pressure", "meds": [("Amlodipine", "5 mg", ["morning"], "any"), ("Telmisartan", "40 mg", ["morning"], "any"), ("Losartan", "50 mg", ["night"], "any")], "elder": 40, "adult": 14},
    "hypothyroid": {"label": "Hypothyroidism", "meds": [("Levothyroxine", "50 mcg", ["early"], "empty_stomach")], "elder": 15, "adult": 10},
    "ckd": {"label": "Chronic kidney disease (stage 3)", "meds": [("Sodium bicarbonate", "500 mg", ["morning", "night"], "after_food"), ("Calcium acetate", "667 mg", ["lunch", "dinner"], "with_food")], "elder": 5, "adult": 1},
    "copd_asthma": {"label": "COPD / asthma", "meds": [("Budesonide-Formoterol inhaler", "2 puffs", ["morning", "night"], "any"), ("Montelukast", "10 mg", ["night"], "any")], "elder": 10, "adult": 6},
    "heart_disease": {"label": "Heart disease (after angioplasty)", "meds": [("Clopidogrel", "75 mg", ["afternoon"], "after_food"), ("Aspirin", "75 mg", ["afternoon"], "after_food"), ("Atorvastatin", "40 mg", ["night"], "any"), ("Metoprolol", "25 mg", ["morning", "night"], "any")], "elder": 12, "adult": 3},
    "arthritis": {"label": "Osteoarthritis (knees)", "meds": [("Calcium + Vitamin D3", "500 mg", ["afternoon"], "after_food"), ("Paracetamol", "650 mg", ["prn"], "after_food")], "elder": 25, "adult": 4},
    "parkinsons": {"label": "Parkinson's disease", "meds": [("Levodopa-Carbidopa", "110 mg", ["morning", "afternoon", "evening"], "before_food")], "elder": 4, "adult": 0},
    "early_dementia": {"label": "Early dementia", "meds": [("Donepezil", "5 mg", ["night"], "any")], "elder": 0, "adult": 0},
    "depression_anxiety": {"label": "Depression / anxiety", "meds": [("Escitalopram", "10 mg", ["morning"], "after_food")], "elder": 6, "adult": 10},
    "cancer_followup": {"label": "Breast cancer follow-up", "meds": [("Letrozole", "2.5 mg", ["morning"], "any")], "elder": 3, "adult": 1},
    "menopause": {"label": "Menopause symptoms", "meds": [], "elder": 0, "adult": 5},
    "pcos": {"label": "PCOS", "meds": [("Myo-inositol", "1 g", ["morning", "night"], "after_food")], "elder": 0, "adult": 6},
    "vitamin_d_low": {"label": "Low vitamin D", "meds": [("Cholecalciferol", "60000 IU", ["weekly"], "after_food")], "elder": 10, "adult": 8},
    "acidity": {"label": "Acidity", "meds": [("Pantoprazole", "40 mg", ["early"], "empty_stomach")], "elder": 12, "adult": 8},
}
# Medicine count bands (share): 0, 1–3, 4–8, 9+.
MED_BANDS = [(0, 0, 0.20), (1, 3, 0.40), (4, 8, 0.30), (9, 12, 0.10)]
EXTRA_MEDS = [  # to reach a band when conditions alone give too few
    ("Vitamin B12", "1500 mcg", ["morning"], "after_food"), ("Folic acid", "5 mg", ["morning"], "after_food"),
    ("Iron", "100 mg", ["afternoon"], "after_food"), ("Omega-3", "1 g", ["night"], "after_food"),
    ("Rosuvastatin", "10 mg", ["night"], "any"), ("Furosemide", "20 mg", ["morning"], "any"),
    ("Tamsulosin", "0.4 mg", ["night"], "after_food"), ("Domperidone", "10 mg", ["breakfast"], "before_food"),
    ("Multivitamin", "1 tablet", ["afternoon"], "after_food"), ("Magnesium", "250 mg", ["night"], "any"),
    ("Cetirizine", "10 mg", ["prn"], "any"), ("Ondansetron", "4 mg", ["prn"], "any"),
]
SLOT_TIMES = {
    "early": ["06:30", "07:00", "06:00"], "morning": ["08:00", "08:30", "09:00"], "breakfast": ["08:15", "08:45"],
    "afternoon": ["13:00", "14:00"], "lunch": ["13:15", "13:45"], "evening": ["18:00", "19:00"],
    "dinner": ["20:15", "20:45"], "night": ["21:00", "21:30", "22:00"],
}
ALLERGENS = ["milk", "peanuts", "sulfa drugs", "penicillin", "shellfish", "eggs", "gluten", "dust", "aspirin", "brinjal"]
ALLERGY_BANDS = [("none", 0.45), ("one", 0.35), ("several", 0.20)]
DIETS = ["low_salt", "low_sugar", "vegetarian", "no_onion_garlic", "jain", "eggetarian", "low_oil", "soft_food", "diabetic", "renal_diet"]
READINGS = ["none", "bp_daily", "sugar_fasting_pp", "weight_weekly", "steps_watch", "inbody_photo", "lab_photo"]
BLOOD_GROUPS = ["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]

# ── A1.4 Activities and routines ───────────────────────────────────────────────────────────────────────────────────
HABITS = {"gym": [0, 1, 2, 3, 4, 5], "walk": [0, 1, 2, 3, 4, 5, 6], "morning_walk": [0, 1, 2, 3, 4, 5, 6], "yoga": [0, 2, 4],
          "pooja": [0, 1, 2, 3, 4, 5, 6], "prayer": [0, 1, 2, 3, 4, 5, 6], "swimming": [1, 3, 5], "cycling": [5, 6]}
QUITS = ["no_smoking", "less_sugar", "no_alcohol"]
OTHER_ACTIVITIES = ["water_intake", "study", "work", "class"]
SLEEP_KINDS = [("early", 0.45, ("21:30", "05:30")), ("normal", 0.35, ("23:00", "07:00")), ("late", 0.15, ("03:00", "11:00")),
               ("shift", 0.05, ("09:00", "17:00"))]
NON_MED_REMINDERS = [
    ("call", "Call {who} at {time}", ["20:30", "19:00"], "daily"),
    ("bill", "Pay the electricity bill", ["10:00"], "once"),
    ("in_an_hour", "Take the clothes off the line", [], "once"),
    ("hourly_quit", "No cigarette, have water instead", [], "hourly"),
    ("water", "Drink a glass of water", ["11:00", "15:00", "17:00"], "daily"),
]

# ── A1.5 Orders and rides ──────────────────────────────────────────────────────────────────────────────────────────
ORDER_USE = {"none": 0.35, "groceries": 0.30, "medicines": 0.20, "food": 0.10, "rides": 0.15}
ORDER_SITUATIONS = ["named_store", "compare", "later_with_max", "spend_limit_approval", "cancel_after_place", "prescription_medicine",
                    "not_deliverable", "login_code_other_person"]
STORES = {"groceries": ["blinkit", "instamart", "zepto"], "medicines": ["apollo", "1mg", "pharmeasy"], "food": ["swiggy", "zomato"],
          "rides": ["uber", "ola", "rapido"]}
RIDE_COORDS_SHARE = 0.70  # ride families with an address that has coordinates

# ── A1.6 Behaviour ─────────────────────────────────────────────────────────────────────────────────────────────────
CHAT_STYLES = ["chatty", "terse", "never_replies", "replies_late", "emoji_heavy", "typos", "all_caps", "voice_only", "sends_photos"]
MOOD_ARCS = ["steady", "low_week", "grief", "loneliness"]
TRICKY = ["scam_call", "fall", "fever", "missed_dose", "double_dose", "refuses_medicine", "family_argument", "alarming_joke"]
SAFETY_SHARE = 0.03

# ── A1.7 Account and data shapes ───────────────────────────────────────────────────────────────────────────────────
SHAPES = ["plus99_caregiver", "same_first_name", "no_surname", "unicode_name", "emoji_name", "empty_record", "conflicting_facts",
          "stale_facts", "same_fact_4_ways", "long_notes", "events_300", "paused", "window_closed", "template_language_missing",
          "unknown_enum"]
SHAPE_SHARE = {"plus99_caregiver": 0.10, "same_first_name": 0.03, "no_surname": 0.05, "unicode_name": 0.05, "emoji_name": 0.02,
               "empty_record": 0.03, "conflicting_facts": 0.04, "stale_facts": 0.05, "same_fact_4_ways": 0.03, "long_notes": 0.03,
               "events_300": 0.01, "paused": 0.02, "window_closed": 0.15, "template_language_missing": 0.0, "unknown_enum": 0.02}

# ── A2.1 Probe topics (rotated so every topic appears in every language) ───────────────────────────────────────────
PROBE_TOPICS = ["dose_taken", "medicine_change", "reading", "symptom", "habit_report", "new_routine", "whats_due", "order", "ride",
                "maps_link", "photo", "remind_me", "stop_calling_me", "about_cared_person", "crisis", "joke_idiom", "small_talk"]
PROBES_PER_FAMILY = (10, 14)

# ── Names (invented pools by region; never the founder's family) ──────────────────────────────────────────────────
BLOCKED_NAMES = {"kritarth", "babita", "vish", "sudha", "medha", "gouri", "singhal"}
NAMES: dict[str, dict[str, list[str]]] = {
    "north": {"f_old": ["Kamla", "Savitri", "Shanti", "Urmila", "Pushpa", "Kaushalya", "Saroj", "Prem", "Asha", "Vimla", "Sushila", "Krishna"],
              "m_old": ["Ramesh", "Suresh", "Om Prakash", "Rajendra", "Mahesh", "Jagdish", "Harish", "Naresh", "Ved Prakash", "Satish"],
              "f_young": ["Neha", "Pooja", "Ritu", "Shweta", "Anjali", "Priyanka", "Ankita", "Swati", "Nidhi", "Shalini"],
              "m_young": ["Amit", "Rohit", "Vikas", "Ankit", "Saurabh", "Nitin", "Gaurav", "Manish", "Abhishek", "Varun"],
              "sur": ["Sharma", "Verma", "Gupta", "Agarwal", "Mishra", "Tiwari", "Srivastava", "Saxena", "Pandey", "Chauhan"]},
    "metro": {"f_old": ["Margaret", "Leela", "Indira", "Shobha", "Nalini", "Rosy", "Usha"], "m_old": ["Thomas", "Prakash", "Ashok", "Victor", "Arvind", "Kishore"],
              "f_young": ["Tanya", "Ria", "Aisha", "Meera", "Sana", "Kavya", "Natasha"], "m_young": ["Kabir", "Arjun", "Rehan", "Karan", "Neil", "Aditya", "Dev"],
              "sur": ["D'Souza", "Fernandes", "Kapoor", "Mehta", "Iyer", "Khanna", "Malhotra", "Bose", "Menon"]},
    "bengal": {"f_old": ["Bela", "Sabita", "Manju", "Gita", "Anima", "Kalpana", "Mira"], "m_old": ["Subrata", "Tapan", "Dilip", "Ashim", "Pradip", "Nirmal"],
               "f_young": ["Riya", "Sreya", "Debolina", "Payel", "Moumita"], "m_young": ["Arnab", "Sayan", "Souvik", "Rajarshi", "Anirban"],
               "sur": ["Banerjee", "Chatterjee", "Mukherjee", "Das", "Ghosh", "Sen", "Roy", "Dutta"]},
    "maharashtra": {"f_old": ["Sunanda", "Sulabha", "Mangala", "Vasudha", "Shaila"], "m_old": ["Vasant", "Dattatray", "Prabhakar", "Madhukar", "Shrikant"],
                    "f_young": ["Snehal", "Rutuja", "Pallavi", "Ketaki"], "m_young": ["Omkar", "Sagar", "Tushar", "Aniket", "Prasad"],
                    "sur": ["Patil", "Deshpande", "Kulkarni", "Joshi", "Jadhav", "Pawar", "Gokhale"]},
    "tamil": {"f_old": ["Lakshmi", "Kamakshi", "Saraswathi", "Valli", "Meenakshi", "Rukmini"], "m_old": ["Raghavan", "Sundaram", "Natarajan", "Murugan", "Venkatesan"],
              "f_young": ["Divya", "Keerthana", "Priya", "Nandhini", "Janani"], "m_young": ["Karthik", "Vignesh", "Arun", "Prakash", "Hari"],
              "sur": ["Iyer", "Subramanian", "Krishnan", "Pillai", "Raman", "Selvam"]},
    "telugu": {"f_old": ["Annapurna", "Satyavathi", "Padma", "Kanakam"], "m_old": ["Venkata Rao", "Subba Rao", "Narasimha", "Ramaiah"],
               "f_young": ["Sravani", "Bhavana", "Swathi", "Harika"], "m_young": ["Srinivas", "Praveen", "Kiran", "Teja", "Sai"],
               "sur": ["Reddy", "Naidu", "Rao", "Chowdary", "Varma"]},
    "gujarat": {"f_old": ["Hansa", "Jayaben", "Kokila", "Manjula", "Pushpa"], "m_old": ["Jayantilal", "Dhirubhai", "Mansukh", "Harshad", "Chandrakant"],
                "f_young": ["Krupa", "Dhwani", "Hetal", "Riddhi"], "m_young": ["Parth", "Jay", "Harsh", "Chirag", "Darshan"],
                "sur": ["Patel", "Shah", "Desai", "Mehta", "Trivedi", "Parikh"]},
    "karnataka": {"f_old": ["Sharada", "Girija", "Nagamma", "Kamalamma"], "m_old": ["Shivanna", "Nagaraj", "Basavaraj", "Ramakrishna"],
                  "f_young": ["Sahana", "Deepika", "Pavithra", "Ashwini"], "m_young": ["Chetan", "Pradeep", "Manjunath", "Suhas"],
                  "sur": ["Gowda", "Hegde", "Rao", "Shetty", "Bhat"]},
    "kerala": {"f_old": ["Ammini", "Thankamma", "Sarojini", "Mariamma"], "m_old": ["Gopalan", "Varghese", "Kunjappan", "Raghavan"],
               "f_young": ["Anju", "Reshma", "Sneha", "Ann"], "m_young": ["Arjun", "Jithin", "Akhil", "Sreejith"],
               "sur": ["Nair", "Menon", "Kurian", "Pillai", "Thomas", "Varghese"]},
    "punjab": {"f_old": ["Gurmeet", "Surjit", "Paramjit", "Harbans"], "m_old": ["Gurdev", "Balwinder", "Harjinder", "Kuldip"],
               "f_young": ["Simran", "Harleen", "Jaspreet", "Navneet"], "m_young": ["Gurpreet", "Manpreet", "Harsimran", "Jaskaran"],
               "sur": ["Singh", "Kaur", "Gill", "Sandhu", "Grewal", "Dhillon"]},
    "odisha": {"f_old": ["Kuni", "Basanti", "Sukanti", "Pramila"], "m_old": ["Bijay", "Prafulla", "Gopinath", "Bhagaban"],
               "f_young": ["Smruti", "Subhashree", "Lipsa"], "m_young": ["Soumya", "Debasish", "Biswajit"],
               "sur": ["Mohanty", "Patnaik", "Sahoo", "Mishra", "Panda"]},
    "assam": {"f_old": ["Pranati", "Junu", "Minoti"], "m_old": ["Hemanta", "Dhiren", "Pabitra"], "f_young": ["Bornali", "Jyotishmita", "Ankita"],
              "m_young": ["Bikram", "Partha", "Rituraj"], "sur": ["Bora", "Gogoi", "Saikia", "Baruah", "Kalita"]},
    "urdu": {"f_old": ["Zubaida", "Shamim", "Rehana", "Naseem"], "m_old": ["Iqbal", "Rashid", "Anwar", "Salim"],
             "f_young": ["Sana", "Alina", "Zoya", "Mehreen"], "m_young": ["Faisal", "Imran", "Adil", "Sameer"],
             "sur": ["Khan", "Siddiqui", "Ansari", "Qureshi", "Rizvi"]},
    "nepali": {"f_old": ["Maya", "Kalpana", "Durga"], "m_old": ["Bhim", "Hari Bahadur", "Krishna Prasad"], "f_young": ["Pema", "Sunita", "Sabina"],
               "m_young": ["Roshan", "Bikash", "Suraj"], "sur": ["Thapa", "Gurung", "Rai", "Chettri", "Tamang"]},
    "goa": {"f_old": ["Maria", "Filomena", "Shalini", "Prema"], "m_old": ["Anthony", "Dattaram", "Francis", "Vinayak"], "f_young": ["Sonia", "Pooja", "Joanna"],
            "m_young": ["Ryan", "Siddhesh", "Melroy"], "sur": ["Naik", "Kamat", "Fernandes", "Rodrigues", "Prabhu"]},
    "rajasthan": {"f_old": ["Bhanwari", "Kesar", "Gulab", "Santosh"], "m_old": ["Bhanwar Lal", "Mangilal", "Ratan Singh", "Kishan Lal"],
                  "f_young": ["Kavita", "Manisha", "Komal"], "m_young": ["Mahendra", "Dinesh", "Pankaj"], "sur": ["Rathore", "Choudhary", "Bishnoi", "Maheshwari", "Jain"]},
    "purvanchal": {"f_old": ["Phulmati", "Chanda", "Lalti", "Sona"], "m_old": ["Ram Pravesh", "Sukhdeo", "Rambabu", "Shivnath"],
                   "f_young": ["Khushboo", "Rinki", "Sonam"], "m_young": ["Bittu", "Raju", "Sonu", "Pintu"], "sur": ["Yadav", "Singh", "Kushwaha", "Prasad", "Tiwari"]},
    "mithila": {"f_old": ["Sarita", "Kanti", "Rukmini"], "m_old": ["Indrakant", "Shambhu", "Ramakant"], "f_young": ["Puja", "Archana", "Nisha"],
                "m_young": ["Aditya", "Ravi", "Shashank"], "sur": ["Jha", "Mishra", "Thakur", "Choudhary"]},
    "chhattisgarh": {"f_old": ["Sukhiya", "Dulari", "Janki"], "m_old": ["Lakhan", "Bisahu", "Ramlal"], "f_young": ["Teena", "Rashmi", "Hemlata"],
                     "m_young": ["Deepak", "Lokesh", "Tikesh"], "sur": ["Sahu", "Verma", "Dewangan", "Sonwani", "Patel"]},
    "haryana": {"f_old": ["Bimla", "Santosh", "Rajbala"], "m_old": ["Jai Singh", "Randhir", "Dharambir"], "f_young": ["Sonia", "Monika", "Preeti"],
                "m_young": ["Sandeep", "Pradeep", "Sumit"], "sur": ["Dahiya", "Hooda", "Malik", "Sangwan", "Yadav"]},
    "awadh": {"f_old": ["Ramdulari", "Kalawati", "Shakuntala"], "m_old": ["Ram Kishore", "Shiv Shankar", "Raghunath"], "f_young": ["Shivani", "Kajal", "Aarti"],
              "m_young": ["Anuj", "Shubham", "Prashant"], "sur": ["Shukla", "Dwivedi", "Tripathi", "Awasthi", "Maurya"]},
    "garhwal": {"f_old": ["Bachuli", "Kamla Devi", "Sateshwari"], "m_old": ["Bachan Singh", "Govind Ram", "Mohan Singh"], "f_young": ["Neelam", "Rekha", "Anita"],
                "m_young": ["Rakesh", "Mukesh", "Pawan"], "sur": ["Negi", "Rawat", "Bisht", "Nautiyal", "Semwal"]},
    "kumaon": {"f_old": ["Heera", "Basanti", "Durga"], "m_old": ["Keshav Dutt", "Chandra Shekhar", "Prem Ballabh"], "f_young": ["Deepa", "Hema", "Kanchan"],
               "m_young": ["Himanshu", "Lalit", "Naveen"], "sur": ["Pant", "Joshi", "Bisht", "Tewari", "Kandpal"]},
    "jammu": {"f_old": ["Shanta", "Raj Rani", "Kamlesh"], "m_old": ["Kuldeep Raj", "Om Raj", "Kewal Krishan"], "f_young": ["Shikha", "Ruchi", "Megha"],
              "m_young": ["Vikram", "Rohan", "Ashish"], "sur": ["Sharma", "Gupta", "Manhas", "Slathia", "Raina"]},
    "tulunadu": {"f_old": ["Seethamma", "Kamala", "Parvathi"], "m_old": ["Sanjeeva", "Ramanna", "Babu"], "f_young": ["Shreya", "Prathiksha", "Rakshitha"],
                 "m_young": ["Karthik", "Sujith", "Prajwal"], "sur": ["Shetty", "Poojary", "Bangera", "Kotian", "Suvarna"]},
    "barak": {"f_old": ["Rokeya", "Hasina", "Anjali"], "m_old": ["Abdul Karim", "Nurul", "Haripada"], "f_young": ["Farhana", "Mousumi", "Tania"],
              "m_young": ["Rubel", "Jahid", "Sourav"], "sur": ["Ahmed", "Choudhury", "Laskar", "Das", "Uddin"]},
    "bundelkhand": {"f_old": ["Ramkali", "Bhagwati", "Kusum"], "m_old": ["Kallu Ram", "Hariram", "Bhagirath"], "f_young": ["Rani", "Jyoti", "Sapna"],
                    "m_young": ["Ajay", "Brajesh", "Rinku"], "sur": ["Ahirwar", "Rajput", "Lodhi", "Kushwaha", "Patel"]},
    "saurashtra": {"f_old": ["Jiviben", "Maniben", "Raliyat"], "m_old": ["Bhikhabhai", "Valjibhai", "Ranchhodbhai"], "f_young": ["Bhumi", "Nirali", "Jinal"],
                   "m_young": ["Kishan", "Mayur", "Hardik"], "sur": ["Vaghela", "Jadeja", "Chavda", "Rathod", "Sojitra"]},
}
# Names written in their own script (the unicode_name shape), by script.
NATIVE_NAMES = {
    "devanagari": ["शान्ति देवी", "रामेश्वर", "सुमित्रा", "गोविन्द"], "bengali": ["বেলা রায়", "সুব্রত", "মীরা"], "tamil": ["லட்சுமி", "சுந்தரம்"],
    "telugu": ["అన్నపూర్ణ", "వెంకట రావు"], "gujarati": ["હંસાબેન", "મનસુખભાઈ"], "kannada": ["ಶಾರದಾ", "ನಾಗರಾಜ್"], "malayalam": ["അമ്മിണി", "ഗോപാലൻ"],
    "gurmukhi": ["ਸੁਰਜੀਤ ਕੌਰ", "ਗੁਰਦੇਵ ਸਿੰਘ"], "odia": ["ବାସନ୍ତୀ", "ପ୍ରଫୁଲ୍ଲ"], "arabic": ["زبیدہ", "رشید"], "latin": ["Zoë", "Renée"],
}
EMOJI_NAMES = ["Maa 🌸", "Papa ji 🙏", "Nani 💕", "Didi ⭐"]
