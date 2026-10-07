"""Languages and dialects Saheli speaks, and the script each is written in.

Saheli writes every Indian language in its own script (Hindi and the Hindi-belt dialects in Devanagari, Tamil in Tamil
script, …), also when the person types it in Roman letters, unless they asked for Roman letters. A dialect (Marwari,
Maithili, Chhattisgarhi…) is spoken the way the person speaks it, in its base language's script.
Mirrors kavach-backend src/services/language.service.ts; keep the two lists the same.
"""

from __future__ import annotations

LANGUAGES: dict[str, tuple[str, str, str]] = {  # code: (name, native name, script)
    "hi": ("Hindi", "हिन्दी", "devanagari"),
    "en": ("English", "English", "latin"),
    "bn": ("Bengali", "বাংলা", "bengali"),
    "mr": ("Marathi", "मराठी", "devanagari"),
    "ta": ("Tamil", "தமிழ்", "tamil"),
    "te": ("Telugu", "తెలుగు", "telugu"),
    "gu": ("Gujarati", "ગુજરાતી", "gujarati"),
    "kn": ("Kannada", "ಕನ್ನಡ", "kannada"),
    "ml": ("Malayalam", "മലയാളം", "malayalam"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ", "gurmukhi"),
    "or": ("Odia", "ଓଡ଼ିଆ", "odia"),
    "as": ("Assamese", "অসমীয়া", "bengali"),
    "ur": ("Urdu", "اردو", "arabic"),
    "ne": ("Nepali", "नेपाली", "devanagari"),
    "kok": ("Konkani", "कोंकणी", "devanagari"),
}

DIALECTS: dict[str, tuple[str, str, str, str]] = {  # code: (name, native, base language, greeting)
    "mwr": ("Marwari", "मारवाड़ी", "hi", "राम राम सा"),
    "mtr": ("Mewari", "मेवाड़ी", "hi", "राम राम सा"),
    "dhd": ("Dhundhari (Jaipuri)", "ढूंढाड़ी", "hi", "राम राम सा"),
    "swv": ("Shekhawati", "शेखावाटी", "hi", "राम राम सा"),
    "hoj": ("Hadoti", "हाड़ौती", "hi", "राम राम सा"),
    "wbr": ("Wagdi", "वागड़ी", "hi", ""),
    "bgc": ("Haryanvi", "हरियाणवी", "hi", "राम राम"),
    "bho": ("Bhojpuri", "भोजपुरी", "hi", "प्रणाम"),
    "mai": ("Maithili", "मैथिली", "hi", "प्रणाम"),
    "mag": ("Magahi", "मगही", "hi", "प्रणाम"),
    "anp": ("Angika", "अंगिका", "hi", "प्रणाम"),
    "bjj": ("Bajjika", "बज्जिका", "hi", "प्रणाम"),
    "awa": ("Awadhi", "अवधी", "hi", "राम राम"),
    "bns": ("Bundeli", "बुंदेली", "hi", "राम राम"),
    "bfy": ("Bagheli", "बघेली", "hi", "राम राम"),
    "hne": ("Chhattisgarhi", "छत्तीसगढ़ी", "hi", "जय जोहार"),
    "bra": ("Braj", "ब्रज", "hi", "राधे राधे"),
    "mup": ("Malvi", "मालवी", "hi", "राम राम"),
    "noe": ("Nimadi", "निमाड़ी", "hi", ""),
    "gbm": ("Garhwali", "गढ़वाली", "hi", "सेवा लगाणु छौं"),
    "kfy": ("Kumaoni", "कुमाऊँनी", "hi", "पैलाग"),
    "him": ("Pahari (Himachali)", "पहाड़ी", "hi", ""),
    "doi": ("Dogri", "डोगरी", "hi", ""),
    "sck": ("Sadri (Nagpuri)", "सादरी", "hi", "जोहार"),
    "vah": ("Varhadi", "वऱ्हाडी", "mr", "राम राम"),
    "mlv": ("Malvani", "मालवणी", "mr", ""),
    "ahr": ("Ahirani", "अहिराणी", "mr", "राम राम"),
    "tcy": ("Tulu", "ತುಳು", "kn", "ನಮಸ್ಕಾರ"),
    "kfa": ("Kodava", "ಕೊಡವ", "kn", ""),
    "syl": ("Sylheti", "সিলেটি", "bn", ""),
    "spv": ("Sambalpuri", "ସମ୍ବଲପୁରୀ", "or", ""),
    "kth": ("Kathiawadi", "કાઠિયાવાડી", "gu", ""),
}

_LANG_ALIAS = {"hinglish": "hi", "hindustani": "hi", "bangla": "bn", "oriya": "or", "panjabi": "pa", "gujrati": "gu"}
_DIALECT_ALIAS = {"rajasthani": "mwr", "jaipuri": "dhd", "nagpuri": "sck", "sadri": "sck", "himachali": "him", "pahari": "him",
                  "kumauni": "kfy", "chattisgarhi": "hne", "chhatisgarhi": "hne", "marwadi": "mwr", "haryanavi": "bgc",
                  "mythili": "mai", "maithli": "mai"}


def _lang_code(v: object) -> str | None:
    k = str(v or "").strip().lower()
    if not k:
        return None
    if k in LANGUAGES:
        return k
    if k in _LANG_ALIAS:
        return _LANG_ALIAS[k]
    for code, (name, native, _) in LANGUAGES.items():
        if k in (name.lower(), native.lower()):
            return code
    base = k.split("-")[0].split("_")[0]
    return base if base in LANGUAGES else None


def _dialect_code(v: object) -> str | None:
    k = str(v or "").strip().lower()
    if not k:
        return None
    if k in DIALECTS:
        return k
    if k in _DIALECT_ALIAS:
        return _DIALECT_ALIAS[k]
    for code, (name, native, _, _) in DIALECTS.items():
        if k in (name.lower(), native.lower(), name.lower().split(" (")[0]):
            return code
    return None


def normalise(value: dict | None) -> dict:
    """{language, dialect, script} with codes from whatever was said or stored ('Marwari', 'hinglish', {'language': 'hi'})."""
    v = value or {}
    out: dict = {}
    d = _dialect_code(v.get("dialect")) or _dialect_code(v.get("language"))
    if d:
        out["dialect"] = d
        out["language"] = DIALECTS[d][2]
    else:
        lang = _lang_code(v.get("language"))
        if lang:
            out["language"] = lang
    if v.get("script") in ("roman", "native"):
        out["script"] = v["script"]
    return out


def merge(current: dict | None, asked: dict | None) -> dict:
    """What was said ('Tamil', 'Marwari', 'Roman letters') on top of what is saved. A new language without a dialect
    drops the old dialect (Marwari is a way of speaking Hindi, not Tamil); 'none' / 'standard' clears it; the script
    choice stays unless they change it."""
    now = normalise(current)
    a = asked or {}
    said = normalise({"language": a.get("language")}) if a.get("language") else {}
    if a.get("language") and not said:
        return {}  # a language we don't know: never silently keep the old one
    clear = str(a.get("dialect") or "").strip().lower() in ("none", "no", "standard")
    out = {"language": said.get("language") or now.get("language"), "script": a.get("script") or now.get("script")}
    if a.get("dialect") and not clear:
        out["dialect"] = a["dialect"]
    elif said.get("dialect"):
        out["dialect"] = said["dialect"]
    elif not clear and (not said.get("language") or said["language"] == now.get("language")):
        out["dialect"] = now.get("dialect")
    return normalise(out)


def script_for(speech: dict | None) -> str | None:
    """The script replies to this person must use: 'latin' for English or when they asked for Roman letters."""
    s = speech or {}
    if s.get("script") == "roman":
        return "latin"
    lang = s.get("language")
    return LANGUAGES[lang][2] if lang in LANGUAGES else None


def label(speech: dict | None) -> str:
    """'Marwari (मारवाड़ी, Hindi family)' or 'Tamil (தமிழ்)'."""
    s = speech or {}
    if s.get("dialect") in DIALECTS:
        name, native, base, _ = DIALECTS[s["dialect"]]
        out = f"{name} ({native}, {LANGUAGES[base][0]} family)"
    elif s.get("language") in LANGUAGES:
        name, native, _ = LANGUAGES[s["language"]]
        out = name if s["language"] == "en" else f"{name} ({native})"
    else:
        return "not set"
    return out + (", in Roman letters (they asked for that)" if s.get("script") == "roman" else "")


def greeting(speech: dict | None) -> str:
    s = speech or {}
    return DIALECTS[s["dialect"]][3] if s.get("dialect") in DIALECTS else ""


def sentence(speech: dict) -> str:
    """One line for the care record."""
    script = script_for(speech)
    where = "in Roman letters" if script == "latin" and speech.get("language") != "en" else (f"in {script.capitalize()}" if script and script != "latin" else "")
    return f"Speaks {label(speech)}" + (f"; write to them {where}" if where else "")
