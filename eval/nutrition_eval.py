"""Nutrition plan §N14: the real-model eval of food tracking, with the plan's bars. Paid: sunny-ship only, with a hard cap.

Sets (each with its bar):
  phrases  84 food messages in 12 languages and scripts, said by an elder, a caregiver about the elder, or a self-care
           adult, through the real brain (run_turn): log_food gets the right items and amounts      (≥ 90 %)
  foods    40 standard foods through the look-up chain against published per-100 g values (USDA SR Legacy):
           kcal within ±15 % and protein within ±20 %                                                (≥ 85 % each)
  labels   15 nutrition labels (rendered here from known values: invented products) read by vision:
           every printed value exact                                                                 (≥ 95 %)
  photos   CC0 / public-domain meal photos (eval/data/nutrition_photos/manifest.json; no people):
           the dishes in the photo are the ones read                                                 (≥ 85 %)
  orders   20 delivered restaurant meals, then a message about them: food is logged only after a yes or a part,
           and never without one                                                                     (0 without consent)
It reports latency (p50 / p90 per logging turn) and the cost per log (₹, measured from the model spend).

The checks are the eval's own: amounts and dates are compared in code; whether a logged name is the expected food is
judged by a model (never by words in code). Invented families and products only.

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_nutrition_eval \\
    DEBUG=false GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 MEMORY_EMBEDDINGS=off \\
    NUTRITION_EVAL_MAX_INR=300 PYTHONPATH=. python eval/nutrition_eval.py [phrases foods labels photos orders] [--limit N]
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import base64
import io
import json
import statistics
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

ELDER = {"id": "nut-eval-elder", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "nut-eval-son", "name": "Ravi", "role": "primary caregiver"}
SELF = {"id": "nut-eval-self", "name": "Anil", "role": "primary caregiver"}
OUT = Path(os.getenv("NUTRITION_EVAL_OUT", "runs/nutrition_eval"))
BARS = {"phrases": 90.0, "foods_kcal": 85.0, "foods_protein": 85.0, "labels": 95.0, "photos": 85.0, "orders_without_consent": 0}

# ── phrases: (language, who speaks, message, expected [(food, qty, unit)], extra) ──────────────────────────────────────
# who: elder (about themselves) | caregiver (the son about the elder) | self (a self-care adult). extra: {"day": -1} = yesterday
P = [
    ("English", "self", "Had 2 rotis with dal and bhindi for lunch", [("roti / chapati", 2, None), ("dal", None, None), ("bhindi (okra) sabzi", None, None)], {}),
    ("English", "self", "Just finished 3 boiled eggs", [("boiled egg", 3, None)], {}),
    ("English", "self", "2 cups of tea since morning", [("tea", 2, "cup")], {}),
    ("English", "self", "Last night I had chicken biryani for dinner", [("chicken biryani", None, None)], {"day": -1}),
    ("English", "caregiver", "Mom had one bowl of khichdi for lunch", [("khichdi", 1, "bowl")], {}),
    ("English", "self", "ate some rice", [("rice", None, None)], {}),
    ("English", "self", "drank 3 glasses of water", [("water", 3, "glass")], {}),
    ("Hinglish", "elder", "Lunch mein 2 roti, dal aur aloo gobhi khayi", [("roti", 2, None), ("dal", None, None), ("aloo gobhi", None, None)], {}),
    ("Hinglish", "self", "subah 2 ande khaye", [("egg", 2, None)], {}),
    ("Hinglish", "self", "abhi tak 3 cup chai pi li", [("tea", 3, "cup")], {}),
    ("Hinglish", "self", "kal raat paneer butter masala aur 2 naan khaye the", [("paneer butter masala", None, None), ("naan", 2, None)], {"day": -1}),
    ("Hinglish", "caregiver", "Mummy ne nashte mein ek plate poha khaya", [("poha", 1, "plate")], {}),
    ("Hinglish", "elder", "thoda sa chawal khaya", [("rice", None, None)], {}),
    ("Hinglish", "self", "ek glass lassi pi", [("lassi", 1, "glass")], {}),
    ("Hindi", "elder", "दोपहर में दो रोटी और एक कटोरी दाल खाई", [("roti", 2, None), ("dal", 1, "katori")], {}),
    ("Hindi", "elder", "नाश्ते में दो इडली और सांभर खाया", [("idli", 2, None), ("sambar", None, None)], {}),
    ("Hindi", "elder", "आज चार कप चाय पी", [("tea", 4, "cup")], {}),
    ("Hindi", "caregiver", "माँ ने रात को एक कटोरी खिचड़ी खाई", [("khichdi", 1, "katori")], {}),
    ("Hindi", "elder", "कल शाम को दो समोसे खाए थे", [("samosa", 2, None)], {"day": -1}),
    ("Hindi", "elder", "थोड़ी सी सब्ज़ी खाई", [("vegetable sabzi", None, None)], {}),
    ("Hindi", "elder", "एक केला खाया", [("banana", 1, None)], {}),
    ("Bengali", "elder", "দুপুরে ভাত, ডাল আর মাছের ঝোল খেলাম", [("rice", None, None), ("dal", None, None), ("fish curry (macher jhol)", None, None)], {}),
    ("Bengali", "elder", "সকালে দুটো ডিম সেদ্ধ খেয়েছি", [("boiled egg", 2, None)], {}),
    ("Bengali", "elder", "দুই কাপ চা খেয়েছি", [("tea", 2, "cup")], {}),
    ("Bengali", "caregiver", "মা রাতে দুটো রুটি আর তরকারি খেয়েছেন", [("roti", 2, None), ("vegetable curry (tarkari)", None, None)], {}),
    ("Bengali", "elder", "কাল রাতে বিরিয়ানি খেয়েছিলাম", [("biryani", None, None)], {"day": -1}),
    ("Bengali", "elder", "একটু মুড়ি খেলাম", [("puffed rice (muri)", None, None)], {}),
    ("Bengali", "elder", "একটা রসগোল্লা খেলাম", [("rasgulla", 1, None)], {}),
    ("Tamil", "elder", "மதியம் சாதம், சாம்பார், ரசம் சாப்பிட்டேன்", [("rice", None, None), ("sambar", None, None), ("rasam", None, None)], {}),
    ("Tamil", "elder", "காலையில் மூன்று இட்லி சாப்பிட்டேன்", [("idli", 3, None)], {}),
    ("Tamil", "elder", "இரண்டு காபி குடித்தேன்", [("coffee", 2, "cup")], {}),
    ("Tamil", "caregiver", "அம்மா இரவு இரண்டு தோசை சாப்பிட்டாங்க", [("dosa", 2, None)], {}),
    ("Tamil", "elder", "நேற்று இரவு பொங்கல் சாப்பிட்டேன்", [("pongal", None, None)], {"day": -1}),
    ("Tamil", "elder", "கொஞ்சம் தயிர் சாதம் சாப்பிட்டேன்", [("curd rice", None, None)], {}),
    ("Tamil", "elder", "ஒரு வாழைப்பழம் சாப்பிட்டேன்", [("banana", 1, None)], {}),
    ("Telugu", "elder", "మధ్యాహ్నం అన్నం, పప్పు, పెరుగు తిన్నాను", [("rice", None, None), ("dal (pappu)", None, None), ("curd", None, None)], {}),
    ("Telugu", "elder", "ఉదయం రెండు దోసెలు తిన్నాను", [("dosa", 2, None)], {}),
    ("Telugu", "elder", "మూడు కప్పుల టీ తాగాను", [("tea", 3, "cup")], {}),
    ("Telugu", "caregiver", "అమ్మ రాత్రి రెండు చపాతీలు తిన్నారు", [("chapati", 2, None)], {}),
    ("Telugu", "elder", "నిన్న రాత్రి బిర్యానీ తిన్నాను", [("biryani", None, None)], {"day": -1}),
    ("Telugu", "elder", "కొంచెం ఉప్మా తిన్నాను", [("upma", None, None)], {}),
    ("Telugu", "elder", "ఒక గ్లాసు మజ్జిగ తాగాను", [("buttermilk", 1, "glass")], {}),
    ("Marathi", "elder", "दुपारी दोन चपाती, वरण आणि भात खाल्ला", [("chapati", 2, None), ("dal (varan)", None, None), ("rice", None, None)], {}),
    ("Marathi", "elder", "सकाळी एक प्लेट पोहे खाल्ले", [("poha", 1, "plate")], {}),
    ("Marathi", "elder", "दोन कप चहा घेतला", [("tea", 2, "cup")], {}),
    ("Marathi", "caregiver", "आईने रात्री एक वाटी खिचडी खाल्ली", [("khichdi", 1, "katori")], {}),
    ("Marathi", "elder", "काल रात्री पुरणपोळी खाल्ली", [("puran poli", None, None)], {"day": -1}),
    ("Marathi", "elder", "थोडी भाजी खाल्ली", [("vegetable sabzi (bhaji)", None, None)], {}),
    ("Marathi", "elder", "दोन अंडी खाल्ली", [("egg", 2, None)], {}),
    ("Kannada", "elder", "ಮಧ್ಯಾಹ್ನ ಅನ್ನ, ಸಾರು, ಪಲ್ಯ ಊಟ ಮಾಡಿದೆ", [("rice", None, None), ("saaru (rasam)", None, None), ("palya (vegetable stir-fry)", None, None)], {}),
    ("Kannada", "elder", "ಬೆಳಿಗ್ಗೆ ಎರಡು ಇಡ್ಲಿ ತಿಂದೆ", [("idli", 2, None)], {}),
    ("Kannada", "elder", "ಎರಡು ಕಪ್ ಕಾಫಿ ಕುಡಿದೆ", [("coffee", 2, "cup")], {}),
    ("Kannada", "caregiver", "ಅಮ್ಮ ರಾತ್ರಿ ಎರಡು ಚಪಾತಿ ತಿಂದರು", [("chapati", 2, None)], {}),
    ("Kannada", "elder", "ನಿನ್ನೆ ರಾತ್ರಿ ಬಿಸಿಬೇಳೆಬಾತ್ ತಿಂದೆ", [("bisi bele bath", None, None)], {"day": -1}),
    ("Kannada", "elder", "ಸ್ವಲ್ಪ ಉಪ್ಪಿಟ್ಟು ತಿಂದೆ", [("upma (uppittu)", None, None)], {}),
    ("Kannada", "elder", "ಒಂದು ಲೋಟ ಹಾಲು ಕುಡಿದೆ", [("milk", 1, "glass")], {}),
    ("Malayalam", "elder", "ഉച്ചയ്ക്ക് ചോറും സാമ്പാറും അവിയലും കഴിച്ചു", [("rice", None, None), ("sambar", None, None), ("avial", None, None)], {}),
    ("Malayalam", "elder", "രാവിലെ രണ്ട് കഷണം പുട്ടും കടലക്കറിയും കഴിച്ചു", [("puttu", 2, None), ("kadala curry", None, None)], {}),
    ("Malayalam", "elder", "രണ്ട് ഗ്ലാസ് ചായ കുടിച്ചു", [("tea", 2, "glass")], {}),
    ("Malayalam", "caregiver", "അമ്മ രാത്രി മൂന്ന് ദോശ കഴിച്ചു", [("dosa", 3, None)], {}),
    ("Malayalam", "elder", "ഇന്നലെ രാത്രി ബിരിയാണി കഴിച്ചു", [("biryani", None, None)], {"day": -1}),
    ("Malayalam", "elder", "കുറച്ച് കഞ്ഞി കുടിച്ചു", [("rice gruel (kanji)", None, None)], {}),
    ("Malayalam", "elder", "ഒരു പഴം കഴിച്ചു", [("banana", 1, None)], {}),
    ("Gujarati", "elder", "બપોરે બે રોટલી, દાળ અને ભાત ખાધા", [("roti", 2, None), ("dal", None, None), ("rice", None, None)], {}),
    ("Gujarati", "elder", "સવારે એક પ્લેટ ઢોકળા ખાધા", [("dhokla", 1, "plate")], {}),
    ("Gujarati", "elder", "બે કપ ચા પીધી", [("tea", 2, "cup")], {}),
    ("Gujarati", "caregiver", "બાએ રાત્રે એક વાટકી ખીચડી ખાધી", [("khichdi", 1, "katori")], {}),
    ("Gujarati", "elder", "ગઈકાલે રાત્રે થેપલા ખાધા હતા", [("thepla", None, None)], {"day": -1}),
    ("Gujarati", "elder", "થોડું શાક ખાધું", [("vegetable sabzi (shaak)", None, None)], {}),
    ("Gujarati", "elder", "એક ગ્લાસ છાશ પીધી", [("buttermilk", 1, "glass")], {}),
    ("Punjabi", "elder", "ਦੁਪਹਿਰੇ ਦੋ ਰੋਟੀਆਂ ਤੇ ਸਾਗ ਖਾਧਾ", [("roti", 2, None), ("saag", None, None)], {}),
    ("Punjabi", "elder", "ਸਵੇਰੇ ਇੱਕ ਆਲੂ ਪਰਾਂਠਾ ਖਾਧਾ", [("aloo paratha", 1, None)], {}),
    ("Punjabi", "elder", "ਦੋ ਕੱਪ ਚਾਹ ਪੀਤੀ", [("tea", 2, "cup")], {}),
    ("Punjabi", "caregiver", "ਮਾਂ ਨੇ ਰਾਤ ਨੂੰ ਇੱਕ ਕੌਲੀ ਦਾਲ ਖਾਧੀ", [("dal", 1, "katori")], {}),
    ("Punjabi", "elder", "ਕੱਲ੍ਹ ਰਾਤ ਰਾਜਮਾ ਚੌਲ ਖਾਧੇ ਸਨ", [("rajma chawal", None, None)], {"day": -1}),
    ("Punjabi", "elder", "ਥੋੜ੍ਹਾ ਦਹੀਂ ਖਾਧਾ", [("curd", None, None)], {}),
    ("Punjabi", "elder", "ਇੱਕ ਗਲਾਸ ਲੱਸੀ ਪੀਤੀ", [("lassi", 1, "glass")], {}),
    ("Odia", "elder", "ମଧ୍ୟାହ୍ନରେ ଭାତ, ଡାଲି ଆଉ ଶାଗ ଖାଇଲି", [("rice", None, None), ("dal", None, None), ("saag", None, None)], {}),
    ("Odia", "elder", "ସକାଳେ ଦୁଇଟା ଅଣ୍ଡା ଖାଇଲି", [("egg", 2, None)], {}),
    ("Odia", "elder", "ଦୁଇ କପ ଚା ପିଇଲି", [("tea", 2, "cup")], {}),
    ("Odia", "caregiver", "ମା ରାତିରେ ଦୁଇଟା ରୁଟି ଖାଇଲେ", [("roti", 2, None)], {}),
    ("Odia", "elder", "କାଲି ରାତିରେ ପଖାଳ ଖାଇଥିଲି", [("pakhala (fermented rice)", None, None)], {"day": -1}),
    ("Odia", "elder", "ଟିକେ ଚୁଡ଼ା ଖାଇଲି", [("flattened rice (chuda)", None, None)], {}),
    ("Odia", "elder", "ଗୋଟିଏ ରସଗୋଲା ଖାଇଲି", [("rasgulla", 1, None)], {}),
]
# households of units that mean the same container for this check
UNIT_CLASS = {"katori": "bowl", "katori_small": "bowl", "katori_large": "bowl", "bowl": "bowl", "cup": "cup", "glass": "cup", "plate": "plate",
              "piece": None, None: None}

# ── foods: published per-100 g values (USDA SR Legacy), (name as asked, kcal, protein g) ─────────────────────────────
FOODS = [
    ("boiled egg", 155, 12.6), ("steamed white rice (cooked)", 130, 2.7), ("banana", 89, 1.1), ("apple", 52, 0.3),
    ("whole milk, 3.25% fat", 61, 3.2), ("plain curd (whole milk yogurt)", 61, 3.5), ("roasted chicken breast, no skin", 165, 31.0),
    ("boiled lentils (masoor dal, cooked)", 116, 9.0), ("boiled chickpeas (chana, cooked)", 164, 8.9), ("boiled rajma (kidney beans, cooked)", 127, 8.7),
    ("boiled potato, no skin", 86, 1.7), ("raw peanuts", 567, 25.8), ("almonds", 579, 21.2), ("salted butter", 717, 0.9), ("ghee", 876, 0.3),
    ("white sugar", 387, 0.0), ("honey", 304, 0.3), ("raw spinach (palak)", 23, 2.9), ("raw tomato", 18, 0.9), ("raw onion", 40, 1.1),
    ("raw carrot", 41, 0.9), ("cucumber with peel", 15, 0.7), ("orange", 47, 0.9), ("mango", 60, 0.8), ("papaya", 43, 0.5),
    ("watermelon", 30, 0.6), ("boiled green peas", 84, 5.4), ("boiled sweet potato", 76, 1.4), ("boiled moong (mung beans, cooked)", 105, 7.0),
    ("cheddar cheese", 403, 24.9), ("smooth peanut butter", 588, 25.1), ("tender coconut water", 19, 0.7), ("whole wheat bread", 252, 12.5),
    ("raw cauliflower", 25, 1.9), ("guava", 68, 2.6), ("pomegranate", 83, 1.7), ("grapes", 69, 0.7), ("walnuts", 654, 15.2),
    ("cashew nuts", 553, 18.2), ("raisins", 299, 3.1),
]

# ── labels: invented products; per 100 g (and per serving) as printed ───────────────────────────────────────────────
LABEL_KEYS = ("energy_kcal", "protein_g", "carbs_g", "sugar_g", "fat_g", "sodium_mg")
LABELS = [
    ("Sunrise Oats Crunch", "Kalpa Foods", 30, (389, 13.5, 66.2, 8.4, 7.1, 12)),
    ("Masala Peanut Mix", "Desi Bites", 25, (562, 21.4, 32.5, 3.2, 38.6, 840)),
    ("Ragi Cookies", "Millet Home", 20, (472, 7.2, 68.4, 22.5, 18.9, 310)),
    ("Protein Bar Choco", "FitKart", 60, (398, 33.3, 38.2, 5.1, 12.4, 215)),
    ("Plain Makhana", "Lotus Seed Co", 25, (347, 9.7, 76.9, 0.0, 0.1, 5)),
    ("Instant Poha Mix", "Morning Plate", 80, (371, 7.8, 76.4, 4.6, 3.9, 980)),
    ("Toned Milk", "Green Valley Dairy", 200, (58, 3.1, 4.7, 4.7, 3.0, 42)),
    ("Whole Wheat Rusk", "Chai Time", 15, (412, 11.2, 72.0, 14.8, 8.6, 450)),
    ("Roasted Chana", "Snack Patra", 30, (369, 22.1, 58.4, 1.2, 5.3, 25)),
    ("Mango Fruit Drink", "Aamras", 250, (62, 0.0, 15.4, 14.9, 0.0, 18)),
    ("Bhujia Sev", "Namkeen Ghar", 30, (585, 12.6, 39.8, 1.4, 41.9, 1120)),
    ("Greek Style Yogurt", "Kalpa Dairy", 100, (98, 9.1, 4.2, 4.2, 5.0, 36)),
    ("Multigrain Atta", "Chakki Fresh", 100, (342, 12.1, 69.8, 1.5, 2.2, 6)),
    ("Dark Chocolate 70%", "Cocoa Coast", 20, (581, 7.9, 45.3, 28.6, 41.2, 11)),
    ("Soya Chunks", "Nutri Patra", 50, (345, 52.0, 33.0, 9.0, 0.5, 20)),
]


def _label_png(name: str, brand: str, serving: float, v: tuple, *, kj: bool) -> dict:
    from PIL import Image, ImageDraw, ImageFont

    per_s = [round(x * serving / 100.0, 1) for x in v]
    lines = [f"{brand}", f"{name}", "", "NUTRITION INFORMATION", f"Serving size {serving:g} g", "",
             f"{'':<22}{'Per 100 g':>12}{'Per serving':>14}"]
    rows = [("Energy (kcal)", v[0], per_s[0]), ("Protein (g)", v[1], per_s[1]), ("Carbohydrate (g)", v[2], per_s[2]),
            ("  of which Sugars (g)", v[3], per_s[3]), ("Total Fat (g)", v[4], per_s[4]), ("Sodium (mg)", v[5], per_s[5])]
    if kj:
        lines.append(f"{'Energy (kJ)':<22}{round(v[0] * 4.184):>12}{round(per_s[0] * 4.184):>14}")
    for label, a, b in rows:
        lines.append(f"{label:<22}{a:>12g}{b:>14g}")
    lines += ["", "Ingredients: as printed on the pack.", "Veg."]
    img = Image.new("RGB", (900, 120 + 46 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([20, 20, 880, 100 + 46 * len(lines)], outline="black", width=3)
    for i, line in enumerate(lines):
        d.text((40, 40 + i * 46), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"mime": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}


# ── orders: a delivered restaurant meal, then what they say (language, message, consent expected yes|part|no|unknown) ─
ORDERS = [
    ("Hinglish", "Biryani aa gayi thi, maine poori kha li", "yes"), ("Hinglish", "khana aa gaya hai, abhi khaya nahi", "no"),
    ("Hinglish", "aadhi biryani khayi, baaki fridge mein rakh di", "part"), ("Hinglish", "order aa gaya", "unknown"),
    ("English", "The thali came and I ate all of it", "yes"), ("English", "Food arrived but I'm not hungry yet", "no"),
    ("English", "I only managed a little of the thali", "part"), ("English", "the delivery guy just left", "unknown"),
    ("Hindi", "खाना आ गया, अभी नहीं खाया", "no"), ("Hindi", "जो खाना आया था वो पूरा खा लिया", "yes"),
    ("Tamil", "ஆர்டர் வந்தது, இன்னும் சாப்பிடவில்லை", "no"), ("Tamil", "வந்த சாப்பாட்டை முழுசா சாப்பிட்டேன்", "yes"),
    ("Bengali", "খাবার এসে গেছে, এখনও খাইনি", "no"), ("Bengali", "অর্ধেকটা খেয়েছি", "part"),
    ("Telugu", "ఆర్డర్ వచ్చింది", "unknown"), ("Telugu", "వచ్చిన భోజనం మొత్తం తిన్నాను", "yes"),
    ("Marathi", "जेवण आलं, अजून खाल्लं नाही", "no"), ("Kannada", "ಊಟ ಬಂತು, ಸ್ವಲ್ಪ ತಿಂದೆ", "part"),
    ("Hinglish", "beta ne kha liya, maine nahi khaya", "no"), ("Gujarati", "જમવાનું આવી ગયું, હજી ખાધું નથી", "no"),
]

JUDGE_PROMPT = """You check a food tracker's reading. EXPECTED lists the foods a person said they ate (in English). LOGGED
lists the item names the tracker wrote down. For each expected food, give the index of the logged item that is that food
(a regional name, spelling or a fuller name of the same dish counts; a different dish does not), or null. Also list the
indexes of logged items that are not any expected food.
Return only JSON: {"match": [index or null, ...], "extra": [index, ...]}"""


async def judge_names(expected: list[str], logged: list[str]) -> dict:
    from app.nutrition import lookup

    if not logged:
        return {"match": [None] * len(expected), "extra": []}
    d = await lookup.ask("classify", JUDGE_PROMPT, json.dumps({"EXPECTED": expected, "LOGGED": [{"i": i, "name": n} for i, n in enumerate(logged)]},
                                                              ensure_ascii=False), essential=False, max_tokens=500)
    if not isinstance(d, dict) or len(d.get("match") or []) != len(expected):
        return {"match": [None] * len(expected), "extra": [], "judge_failed": True}
    return d


def pctile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 2)


class Run:
    def __init__(self, cap: float):
        from app.llm import spend

        self.cap = cap
        self.spend = spend
        self.rows: dict[str, list[dict]] = {}

    def left(self) -> bool:
        return self.spend.process_spent() < self.cap

    def add(self, section: str, row: dict) -> None:
        self.rows.setdefault(section, []).append(row)


async def phrase_case(run: Run, sessions, case: tuple) -> dict:
    from app.brain.loop import TurnRequest, run_turn
    from app.core import clock
    from app.sim.world import SimHost

    lang, who, text, expected, extra = case
    fam = f"nut-eval-{uuid.uuid4().hex[:8]}"
    elder, speaker, members = (SELF, SELF, [SELF]) if who == "self" else (ELDER, SON if who == "caregiver" else ELDER, [ELDER, SON])
    before = run.spend.process_spent()
    t0 = time.monotonic()
    async with sessions() as s:
        res = await run_turn(s, SimHost(), TurnRequest(family_id=fam, elder=elder, speaker=speaker, members=members, text=text,
                                                       message_ref=f"m-{uuid.uuid4().hex[:8]}", channel="whatsapp"))
    secs = time.monotonic() - t0
    calls = [a for a in res.actions if a.get("tool") == "log_food" and a.get("ok", True)]
    row = {"language": lang, "who": who, "text": text, "reply": res.reply, "secs": round(secs, 2), "inr": round(run.spend.process_spent() - before, 3),
           "tools": [a["tool"] for a in res.actions]}
    problems: list[str] = []
    if not calls:
        problems.append("log_food was not called")
        row.update({"pass": False, "problems": problems})
        return row
    args = calls[0].get("args") or {}
    items = [i for i in args.get("items") or [] if isinstance(i, dict)]
    row["logged"] = items
    j = await judge_names([e[0] for e in expected], [str(i.get("name") or "") for i in items])
    for (food, qty, unit), m in zip(expected, j.get("match") or []):
        it = items[m] if isinstance(m, int) and 0 <= m < len(items) else None
        if it is None:
            problems.append(f"{food}: not logged")
            continue
        got_q = it.get("quantity")
        if qty is not None and (got_q is None or abs(float(got_q) - qty) > 1e-6):
            problems.append(f"{food}: amount {got_q} not {qty}")
        if qty is None and got_q is not None:
            problems.append(f"{food}: an amount ({got_q}) nobody said")
        if unit and UNIT_CLASS.get(it.get("unit"), it.get("unit")) != UNIT_CLASS.get(unit, unit):
            problems.append(f"{food}: unit {it.get('unit')} not {unit}")
    for x in j.get("extra") or []:
        if isinstance(x, int) and 0 <= x < len(items):
            problems.append(f"an item nobody said: {items[x].get('name')}")
    if who == "caregiver" and args.get("for_whom") not in (ELDER["id"],):
        problems.append(f"for_whom {args.get('for_whom')!r}, the elder was meant")
    if extra.get("day") == -1:
        want = (clock.ist() - timedelta(days=1)).date().isoformat()
        if not str(args.get("eaten_at") or "").startswith(want):
            problems.append(f"eaten_at {args.get('eaten_at')!r}, yesterday ({want}) was meant")
    if j.get("judge_failed"):
        problems.append("the name judge did not answer")
    row.update({"pass": not problems, "problems": problems})
    return row


async def food_case(run: Run, sessions, name: str, kcal: float, protein: float) -> dict:
    from app.nutrition import lookup

    before = run.spend.process_spent()
    t0 = time.monotonic()
    async with sessions() as s:
        out = await lookup.resolve(s, f"nut-eval-{uuid.uuid4().hex[:6]}", [{"name": name}], essential=True, no_questions=True)
        await s.commit()
    r = out[0]
    per = (r.food.per_100g or {}) if r.food is not None else {}
    got_k, got_p = per.get("energy_kcal"), per.get("protein_g")
    k_ok = got_k is not None and abs(float(got_k) - kcal) <= 0.15 * kcal
    p_ok = got_p is not None and (abs(float(got_p) - protein) <= 0.20 * protein if protein >= 1 else abs(float(got_p) - protein) <= 0.5)
    return {"food": name, "want": {"kcal": kcal, "protein_g": protein}, "got": {"kcal": got_k, "protein_g": got_p},
            "matched": r.food.name if r.food is not None else None, "source": r.food.source if r.food is not None else None,
            "kcal_ok": k_ok, "protein_ok": p_ok, "secs": round(time.monotonic() - t0, 2), "inr": round(run.spend.process_spent() - before, 3)}


async def label_case(run: Run, i: int, case: tuple) -> dict:
    from app.nutrition import vision

    name, brand, serving, v = case
    img = _label_png(name, brand, serving, v, kj=i % 4 == 0)
    before = run.spend.process_spent()
    t0 = time.monotonic()
    got = await vision.read_label([img])
    per = (got or {}).get("per_100g") or {}
    fields = {k: (want, per.get(k)) for k, want in zip(LABEL_KEYS, v)}
    exact = {k: g is not None and abs(float(g) - w) < 0.051 for k, (w, g) in fields.items()}
    return {"product": f"{brand} {name}", "read": bool(got), "fields": {k: {"want": w, "got": g} for k, (w, g) in fields.items()},
            "exact": sum(exact.values()), "of": len(exact), "serving_ok": bool(got) and abs(float(got.get("serving_g") or 0) - serving) < 0.051,
            "secs": round(time.monotonic() - t0, 2), "inr": round(run.spend.process_spent() - before, 3)}


async def photo_case(run: Run, item: dict, folder: Path) -> dict:
    from app.nutrition import vision

    data = base64.b64encode((folder / item["file"]).read_bytes()).decode()
    mime = "image/png" if item["file"].lower().endswith(".png") else "image/jpeg"
    before = run.spend.process_spent()
    t0 = time.monotonic()
    got = await vision.read_meal([{"mime": mime, "data": data}])
    names = [str(x.get("name") or "") for x in (got or {}).get("items") or []]
    j = await judge_names(item["expect"], names)
    hit = [m is not None for m in j.get("match") or []]
    return {"file": item["file"], "source": item.get("source"), "license": item.get("license"), "expect": item["expect"], "read": names,
            "confidence": (got or {}).get("confidence"), "items_ok": bool(hit) and all(hit), "extra": len(j.get("extra") or []),
            "secs": round(time.monotonic() - t0, 2), "inr": round(run.spend.process_spent() - before, 3)}


async def order_case(run: Run, sessions, lang: str, text: str, consent: str) -> dict:
    from sqlalchemy import select

    from app.brain.loop import TurnRequest, run_turn
    from app.core import clock
    from app.nutrition.models import NutritionEntry
    from app.sim.world import SimHost
    from app.tasks.models import Task

    fam = f"nut-eval-{uuid.uuid4().hex[:8]}"
    came = clock.now() - timedelta(minutes=40)
    async with sessions() as s:
        s.add(Task(id=uuid.uuid4(), family_id=fam, subject_id=ELDER["id"], requested_by=SON["id"], service="swiggy", kind="order",
                   goal="Veg thali and a chicken biryani for lunch", status="done", phase="place",
                   details={"chosen": [{"name": "Veg thali", "restaurant": "Annapurna Kitchen"}, {"name": "Chicken biryani", "restaurant": "Annapurna Kitchen"}]},
                   result={"placed": True, "delivered": True, "items": [{"name": "Veg thali", "qty": 1, "restaurant": "Annapurna Kitchen"},
                                                                         {"name": "Chicken biryani", "qty": 1, "restaurant": "Annapurna Kitchen"}]},
                   history=[], created_at=came - timedelta(minutes=45), updated_at=came, deadline_at=came + timedelta(hours=2)))
        await s.commit()
    t0 = time.monotonic()
    before = run.spend.process_spent()
    async with sessions() as s:
        res = await run_turn(s, SimHost(), TurnRequest(family_id=fam, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text=text,
                                                       message_ref=f"m-{uuid.uuid4().hex[:8]}", channel="whatsapp"))
    async with sessions() as s:
        saved = list((await s.execute(select(NutritionEntry).where(NutritionEntry.family_id == fam, NutritionEntry.kind == "food",
                                                                   NutritionEntry.status == "active"))).scalars())
    without_consent = consent in ("no", "unknown") and bool(saved)
    logged_ok = consent in ("no", "unknown") or bool(saved)
    return {"language": lang, "text": text, "consent": consent, "saved": [(e.name, e.consumed, e.portion_eaten) for e in saved], "reply": res.reply,
            "tools": [a["tool"] for a in res.actions], "without_consent": without_consent, "logged_ok": logged_ok,
            "secs": round(time.monotonic() - t0, 2), "inr": round(run.spend.process_spent() - before, 3)}


def summary(run: Run) -> dict:
    r = run.rows
    out: dict = {"spent_inr": round(run.spend.process_spent(), 2), "cap_inr": run.cap}
    ph = r.get("phrases", [])
    if ph:
        langs: dict[str, list[bool]] = {}
        for x in ph:
            langs.setdefault(x["language"], []).append(x["pass"])
        secs = [x["secs"] for x in ph]
        out["phrases"] = {"n": len(ph), "pass_pct": round(100 * sum(x["pass"] for x in ph) / len(ph), 1),
                          "by_language": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(langs.items())},
                          "latency_p50_s": pctile(secs, 0.5), "latency_p90_s": pctile(secs, 0.9),
                          "inr_per_log": round(statistics.mean([x["inr"] for x in ph]), 3)}
    fo = r.get("foods", [])
    if fo:
        out["foods"] = {"n": len(fo), "kcal_pct": round(100 * sum(x["kcal_ok"] for x in fo) / len(fo), 1),
                        "protein_pct": round(100 * sum(x["protein_ok"] for x in fo) / len(fo), 1),
                        "latency_p50_s": pctile([x["secs"] for x in fo], 0.5), "latency_p90_s": pctile([x["secs"] for x in fo], 0.9),
                        "inr_per_new_food": round(statistics.mean([x["inr"] for x in fo]), 3)}
    la = r.get("labels", [])
    if la:
        out["labels"] = {"n": len(la), "values_exact_pct": round(100 * sum(x["exact"] for x in la) / max(1, sum(x["of"] for x in la)), 1),
                         "read": sum(x["read"] for x in la), "serving_ok": sum(x["serving_ok"] for x in la)}
    pp = r.get("photos", [])
    if pp:
        out["photos"] = {"n": len(pp), "items_ok_pct": round(100 * sum(x["items_ok"] for x in pp) / len(pp), 1)}
    od = r.get("orders", [])
    if od:
        out["orders"] = {"n": len(od), "without_consent": sum(x["without_consent"] for x in od),
                         "consented_logged": f"{sum(1 for x in od if x['consent'] in ('yes', 'part') and x['logged_ok'])}/"
                                             f"{sum(1 for x in od if x['consent'] in ('yes', 'part'))}"}
    bars = []
    if "phrases" in out:
        bars.append(("phrases", out["phrases"]["pass_pct"] >= BARS["phrases"]))
    if "foods" in out:
        bars += [("foods kcal", out["foods"]["kcal_pct"] >= BARS["foods_kcal"]), ("foods protein", out["foods"]["protein_pct"] >= BARS["foods_protein"])]
    if "labels" in out:
        bars.append(("labels", out["labels"]["values_exact_pct"] >= BARS["labels"]))
    if "photos" in out:
        bars.append(("photos", out["photos"]["items_ok_pct"] >= BARS["photos"]))
    if "orders" in out:
        bars.append(("orders without consent", out["orders"]["without_consent"] == 0))
    out["bars"] = {k: v for k, v in bars}
    return out


async def main(sections: list[str], limit: int | None) -> int:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.llm import spend
    from eval.world.seed_engine import assert_local, prepare

    cap = float(os.getenv("NUTRITION_EVAL_MAX_INR", "0") or 0)
    if cap <= 0:
        print("Refusing to start: set NUTRITION_EVAL_MAX_INR (₹ this run may spend).")
        return 2
    if os.getenv("GCP_PROJECT_ID") == "kavach-care":
        print("Refusing: paid evals go to the sunny-ship test project, never kavach-care.")
        return 2
    url = os.environ["DATABASE_URL"]
    assert_local(url)
    engine = create_async_engine(url, pool_size=10)
    await prepare(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    spend.configure(sessions)
    run = Run(cap)
    stamp = time.strftime("%Y-%m-%d_%H%M")
    out_dir = OUT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        (out_dir / "rows.json").write_text(json.dumps(run.rows, ensure_ascii=False, indent=1, default=str))
        (out_dir / "summary.json").write_text(json.dumps(summary(run), ensure_ascii=False, indent=1))

    sem = asyncio.Semaphore(int(os.getenv("NUTRITION_EVAL_PARALLEL", "4")))

    async def guarded(section: str, coro_fn) -> None:
        async with sem:
            if not run.left():
                run.add(section, {"skipped": "budget"})
                return
            try:
                run.add(section, await coro_fn())
            except Exception as exc:  # noqa: BLE001 — one case must not stop the run
                run.add(section, {"error": f"{type(exc).__name__}: {exc}"[:300], "pass": False, "kcal_ok": False, "protein_ok": False,
                                  "exact": 0, "of": len(LABEL_KEYS), "read": False, "serving_ok": False, "items_ok": False,
                                  "without_consent": False, "logged_ok": False, "secs": 0, "inr": 0, "language": "?", "consent": "?"})

    if "foods" in sections:
        await asyncio.gather(*(guarded("foods", lambda c=c: food_case(run, sessions, *c)) for c in FOODS[:limit]))
        save()
        print("foods", summary(run).get("foods"), flush=True)
    if "labels" in sections:
        await asyncio.gather(*(guarded("labels", lambda i=i, c=c: label_case(run, i, c)) for i, c in enumerate(LABELS[:limit])))
        save()
        print("labels", summary(run).get("labels"), flush=True)
    if "photos" in sections:
        folder = Path(__file__).parent / "data" / "nutrition_photos"
        manifest = json.loads((folder / "manifest.json").read_text()) if (folder / "manifest.json").exists() else []
        await asyncio.gather(*(guarded("photos", lambda m=m: photo_case(run, m, folder)) for m in manifest[:limit]))
        save()
        print("photos", summary(run).get("photos"), flush=True)
    if "phrases" in sections:
        await asyncio.gather(*(guarded("phrases", lambda c=c: phrase_case(run, sessions, c)) for c in P[:limit]))
        save()
        print("phrases", summary(run).get("phrases"), flush=True)
    if "orders" in sections:
        await asyncio.gather(*(guarded("orders", lambda c=c: order_case(run, sessions, *c)) for c in ORDERS[:limit]))
        save()
        print("orders", summary(run).get("orders"), flush=True)
    save()
    s = summary(run)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    print(f"rows and summary in {out_dir}")
    await engine.dispose()
    return 0 if all(s["bars"].values()) else 1


def production_routes() -> str | None:
    import re

    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy-gcp.yml"
    m = re.search(r"MODEL_ROUTES: '(\{.*\})'", wf.read_text()) if wf.exists() else None
    return m.group(1) if m else None


if __name__ == "__main__":
    if not os.getenv("MODEL_ROUTES") and production_routes():
        os.environ["MODEL_ROUTES"] = production_routes() or ""
    args = sys.argv[1:]
    lim = None
    if "--limit" in args:
        i = args.index("--limit")
        lim = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    sys.exit(asyncio.run(main(args or ["foods", "labels", "photos", "phrases", "orders"], lim)))
