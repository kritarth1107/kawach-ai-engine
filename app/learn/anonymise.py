"""Strip anything that identifies a family before an example is shared across families.

Names of household members and anyone in the care record, phone numbers, emails, pincodes, street addresses,
order/ride ids and medicine names are replaced with placeholders. Kinship words (Mummy, Papa, beta, ji) stay: they
carry tone, not identity.

Names are matched in every script a family writes: a name saved as "Neha Joshi" also catches नेहा, জোশী and ஜோஷி,
by comparing how the words sound (a phonetic key over the shared Brahmic letter layout), and catches suffixed forms
(Gopaler, Venkat's, கோபாலுக்கு). Capitalised words that are not names (English and common romanised Indian words,
greetings, foods) are kept, so examples keep their meaning. Urdu in Arabic script is not covered by the phonetic keys.
"""

from __future__ import annotations

import gzip
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"

PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PINCODE = re.compile(r"(?<!\d)[1-9]\d{2}\s?\d{3}(?!\d)")
URL = re.compile(r"https?://\S+")
IDS = re.compile(r"\b[A-Z]{2,5}-?\d{4,}\b|\b\d{8,}\b")
# "Flat 4B", "house no. 12", "H.No 7/3"; "MG Road", "Lajpat Nagar" (a capitalised word before the street word).
# Not "household", "doorbell", "beta-blockers", "dahi lane" or "the colony".
ADDRESS = re.compile(
    r"\b(?i:flat|house|h\.?\s?no\.?|plot|door|block|tower|wing)(?:\s+(?i:no)\.?)?\s*[-#:]?\s*[A-Za-z]?\d[\w/-]*|"
    r"\b([A-Z][\w-]*)\s+(?i:road|rd|street|nagar|colony|sector|layout|marg|lane|enclave|vihar|apartments?|society|towers?)\b"
)
# A word just before one of these is a name, unless it is a common or kinship word ("Gopal bhaiya", "कमला जी", "Ravi garu").
# Not बेटा in Indic scripts: the word before it is usually a verb or a greeting ("खुश रहो बेटा", "शुभ रात्रि बेटा").
HONORIFIC = re.compile(
    r"\b([A-Z][a-z]{2,})(?=\s+(?:ji|jee|beta|bhaiya|bhai|didi|di|sahab|saab|babu|garu|aunty|uncle|madam|sir|bhabhi|mausi|chacha|kaka|kaki|maasi)\b)|"
    r"([ऀ-෿]{2,})(?=\s*(?:जी|भैया|दीदी|साहब|बाबू|आंटी|अंकल|গাৰু|বাবু|দিদি|ଜୀ|ବାବୁ|గారు|அவர்கள்))"
)
KIN_INDIC = {"आंटी", "मम्मी", "पापा", "बाबा", "अम्मा", "दादी", "नानी", "माँ", "मां", "अम्मी", "दादा", "नाना", "भैया", "दीदी", "बहन", "भाई", "बेटा", "बेटी",
             "डॉक्टर", "सर", "मैडम", "मम्मीजी", "पापाजी", "माताजी", "पिताजी", "बाबूजी", "चाचा", "चाची", "मौसी", "बुआ", "मामा", "मामी", "काका", "काकी"}
# Capitalised words that are not names (kept when they start a sentence or are just common).
COMMON = set("""
i im i'm ive i've ill i'll id i'd a an the and or but so if then when what why how who where which this that these those it its he she
they we you your yours my me our us his her their them is are was were be been am do does did have has had will would should could
can may might must please thank thanks thankyou ok okay yes no not hi hello hey dear good morning afternoon evening night great nice
sure sorry also just now today tomorrow yesterday tonight here there all any some every each one two three first last next soon
monday tuesday wednesday thursday friday saturday sunday mondays tuesdays wednesdays thursdays fridays saturdays sundays january
february march april may june july august september october november december sugar bp blood pressure medicine medicines tablet
tablets doctor doctors dr hospital clinic report reports test tests walk water tea chai lunch dinner breakfast food time reminder
reminders note noted saheli whatsapp kavach india indian hindi english jai shri shree krishna ram radhe ji beta haan ha han nahi nahin
acha accha achha theek thik chalo bas aap aapka aapki aapke main mai maine kya kaise kab kahan kyun abhi phir bhi aur toh to ki ka ke
hai hain tha thi ho hoon hu rab rakha sat akal salaam namaste namaskar pranam shubh ratri suprabhat subah shaam raat din dawai dawa
goli insulin metformin don't dont can't cant won't wont that's let let's lets also still yet only very much more most less few many
well right wrong true fine better best worse oh ah wow busy free home work office walk sleep rest pain headache fever cough cold mild
normal later call message noted please done mummy papa bapa baba ammi abbu aunty uncle sir madam professor sahab bhaiya didi happy
birthday diwali holi eid festival puja apni apna apne aaj kal bahut since take apply sunkar unhone unki unka unke agar sare saare
tumhi tumhala ami aai allah oct nov dec sep aug jan feb mar apr jun jul bolus after before with without because about from into upon
kindly anything something nothing everything everyone someone kuch koi sab sabhi mera meri mere tera teri hamara hamari humne hum tum
tumhe unko usko isko yeh ye woh wo yahan wahan jab tab kyunki lekin magar par aaram dhyan khana pani neend dard chinta khush physio
ecg ct mri icu opd otp upi cod sos ok pls
""".split())
COMMON_INDIC = {"ठीक", "है", "हैं", "था", "थी", "हो", "शुभ", "बहुत", "धन्यवाद", "नमस्ते", "हाँ", "हां", "जी", "अच्छा", "सुप्रभात", "शुक्रिया", "जय", "श्री", "राम", "कृष्णा",
                "राधे", "खुश", "प्यारे", "मेरे", "मेरी", "आपका", "आपकी", "नहीं", "सब", "चलो", "अरे", "आज", "कल", "रात", "सुबह", "शाम", "रात्रि", "प्रभात",
                "रहो", "रहिए", "हुई", "हुआ", "गई", "गया", "आशीर्वाद", "प्रणाम", "नमस्कार", "दवा", "दवाई", "गोली", "खाना", "पानी", "चाय", "डॉक्टर",
                "कीजिए", "लीजिए", "दीजिए", "बोलिए", "सुनिए", "रहे", "रही", "रहा", "किया", "दिया", "लिया", "करो", "करें", "बताइए", "समझ", "वाली", "वाले"}
NAME_LIKE = re.compile(r"(?<![\w\[])([A-Z][a-z]{2,})(?![\w\]])")
KEEP = {"mummy", "papa", "maa", "amma", "appa", "baba", "dadi", "dada", "nani", "nana", "beta", "beti", "ji", "aunty", "uncle", "didi",
        "bhaiya", "bhai", "saheli", "doctor", "sir", "madam", "bapa", "achan", "ammi", "abbu", "kaku", "kaki", "kaka", "mama", "mami",
        "mausi", "bua", "chacha", "chachi", "amma", "ammachi", "chechi", "anna", "akka", "aai", "ajji", "thatha", "paati", "tai", "dadu",
        "dida", "thakuma", "boudi", "nanna", "khala"}
# Text that is not a person talking: simulator internals, code, a player's reasoning. Never goes into a shared corpus.
JUNK = re.compile(r"orchestrator|\.raw_output|simulated_users|The user prompt says|\bdef \w+\(|^\s*import \w+|Traceback \(most|\bNOW:\s", re.M)


@lru_cache(maxsize=1)
def english() -> frozenset[str]:
    with gzip.open(DATA / "english.txt.gz", "rt") as f:
        return frozenset(f.read().split())


@lru_cache(maxsize=1)
def indic_roman() -> frozenset[str]:
    return frozenset((DATA / "indic_roman.txt").read_text().split())


@lru_cache(maxsize=1)
def brands() -> frozenset[str]:
    words = {w.lower() for line in (DATA / "medicines.txt").read_text().splitlines() if not line.startswith("#") for w in line.split()}
    return frozenset(w for w in words if len(w) >= 4 and w not in english())


def is_common(word: str) -> bool:
    w = word.lower().removesuffix("'s").removesuffix("’s")
    return w in COMMON or w in KEEP or w in english() or w in indic_roman()


def looks_like_junk(text: str) -> bool:
    return bool(JUNK.search(text or ""))


# ── phonetic keys: how a word sounds, the same for "Neha", नेहा, নেহা and நேஹா ─────────────────────────────

# Latin spellings to sound classes. Voiced and unvoiced stops share a class (Tamil writes both with one letter);
# m and n share one (anusvara is either); s/sh share one; vowels are kept except a/aa, which romanisation and
# inherent vowels make unreliable (only a word-initial vowel is kept).
_LATIN = [("chh", "C"), ("ksh", "KS"), ("shh", "S"), ("sh", "S"), ("ch", "C"), ("kh", "K"), ("gh", "K"), ("jh", "C"), ("th", "T"),
          ("dh", "T"), ("ph", "P"), ("bh", "P"), ("zh", "L"), ("aa", ""), ("ee", "I"), ("ii", "I"), ("ie", "I"), ("oo", "U"), ("uu", "U"),
          ("ai", "E"), ("ei", "E"), ("au", "O"), ("ow", "O"), ("ou", "O"),
          ("k", "K"), ("g", "K"), ("q", "K"), ("c", "K"), ("j", "C"), ("z", "C"), ("t", "T"), ("d", "T"), ("p", "P"), ("b", "P"),
          ("f", "P"), ("m", "N"), ("n", "N"), ("y", "Y"), ("r", "R"), ("l", "L"), ("v", "V"), ("w", "V"), ("s", "S"), ("h", "H"),
          ("x", "KS"), ("a", ""), ("i", "I"), ("e", "E"), ("o", "O"), ("u", "U")]
_VOWELS = set("IEOU")


def latin_key(word: str) -> str:
    w, out, i = word.lower(), [], 0
    if w[:1] in "aeiou":  # a word-initial vowel is kept ("Amit" ≠ "Mit")
        out.append({"a": "A", "e": "E", "i": "I", "o": "O", "u": "U"}[w[0]])
        i = 2 if w[:2] in ("aa", "ee", "ii", "oo", "uu", "ai", "au") else 1
    while i < len(w):
        for src, dst in _LATIN:
            if w.startswith(src, i):
                if src == "y" and i == len(w) - 1 and i > 0 and w[i - 1] not in "aeiou":
                    dst = "I"  # "Lucy", "Ammy"
                out.append(dst)
                i += len(src)
                break
        else:
            i += 1
    return _collapse("".join(out))


# Devanagari letters (and, by the shared ISCII layout, the same letters in Bengali, Gurmukhi, Gujarati, Odia, Tamil,
# Telugu, Kannada and Malayalam) to the same sound classes.
_DEVA: dict[int, str] = {}
for _cps, _cls in ((range(0x915, 0x919), "K"), ((0x919,), "N"), (range(0x91A, 0x91E), "C"), ((0x91E,), "N"), (range(0x91F, 0x923), "T"),
                   ((0x923,), "N"), (range(0x924, 0x928), "T"), ((0x928, 0x929, 0x92E), "N"), (range(0x92A, 0x92E), "P"), ((0x92F, 0x95F), "Y"),
                   ((0x930, 0x931), "R"), ((0x932, 0x933, 0x934), "L"), ((0x935,), "V"), ((0x936, 0x937, 0x938), "S"), ((0x939,), "H"),
                   ((0x958, 0x959, 0x95A), "K"), ((0x95B,), "C"), ((0x95C, 0x95D), "R"), ((0x95E,), "P"),
                   ((0x93F, 0x940, 0x907, 0x908), "I"), ((0x941, 0x942, 0x909, 0x90A), "U"), ((0x943, 0x944, 0x90B), "RI"),
                   ((0x945, 0x946, 0x947, 0x948, 0x90D, 0x90E, 0x90F, 0x910), "E"), ((0x949, 0x94A, 0x94B, 0x94C, 0x911, 0x912, 0x913, 0x914), "O"),
                   ((0x901, 0x902, 0x970), "N"), ((0x94E,), "T")):
    for _cp in _cps:
        _DEVA[_cp] = _cls
_CHILLU = {0xD7A: "N", 0xD7B: "N", 0xD7C: "R", 0xD7D: "L", 0xD7E: "L", 0xD7F: "K"}  # Malayalam final consonants
INDIC_WORD = re.compile(r"[ऀ-ൿ]+")


def indic_key(word: str) -> str:
    out = []
    for n, ch in enumerate(word):
        cp = ord(ch)
        if cp in _CHILLU:
            out.append(_CHILLU[cp])
            continue
        if 0x980 <= cp <= 0xD7F:
            cp = 0x900 + (cp & 0x7F)  # same letter, Devanagari position
        if n == 0 and cp in (0x905, 0x906):
            out.append("A")
            continue
        out.append(_DEVA.get(cp, ""))
    return _collapse("".join(out))


def _collapse(key: str) -> str:
    return re.sub(r"(.)\1+", r"\1", key)


def _name_words(names) -> set[str]:
    out = set()
    for n in names:
        for w in re.findall(r"[^\W\d_]{3,}", n or ""):
            if w.lower() not in KEEP:
                out.add(w)
    return out


def _med_words(medicines) -> set[str]:
    """Each word of each medicine name that identifies it ("Lantus", "Glargine"; not "Insulin" or "Tablet")."""
    out = set()
    for m in medicines:
        words = re.findall(r"[^\W\d_]{3,}", m or "")
        for w in words:
            if len(w) >= 4 and (w.lower() not in english() or len(words) == 1):
                out.add(w)
    return out


def _skeleton(key: str) -> str:
    return re.sub(r"[AIEOU]", "", key)


def anonymise(text: str, *, names: list[str] | tuple = (), medicines: list[str] | tuple = ()) -> str:
    t = text or ""
    t = URL.sub("[LINK]", t)
    t = EMAIL.sub("[EMAIL]", t)
    t = PHONE.sub("[PHONE]", t)
    t = ADDRESS.sub(lambda m: m.group(0) if m.group(1) and m.group(1).lower() in COMMON else "[ADDRESS]", t)  # not "The colony"
    t = PINCODE.sub("[PIN]", t)
    t = IDS.sub("[ID]", t)
    meds = {}
    for w in sorted(_med_words(medicines), key=len, reverse=True):
        meds.setdefault(w.lower(), f"[MED{chr(65 + len(meds) % 26)}]")
        t = re.sub(rf"\b{re.escape(w)}\w*", meds[w.lower()], t, flags=re.I)
    t = re.sub(r"\b[A-Za-z]{4,}\b", lambda m: "[MED]" if m.group(0).lower() in brands() else m.group(0), t)
    people: dict[str, str] = {}
    for w in sorted(_name_words(names), key=len, reverse=True):
        people.setdefault(w.lower(), f"[PERSON{len(people) + 1}]")
        # suffixed forms for longer names: Gopaler, Gopal-ke, Venkat's
        tail = r"(?:['’-]?\w{0,4})" if len(w) >= 4 else r"(?:['’]s|-\w+)?"
        t = re.sub(rf"\b{re.escape(w)}{tail}\b", people[w.lower()], t, flags=re.I)
    # The same names and medicines written in an Indian script.
    name_keys = {latin_key(w): people[w.lower()] for w in _name_words(names) if len(latin_key(w)) >= 2}
    med_keys = {_skeleton(latin_key(w)): meds[w.lower()] for w in _med_words(medicines) if len(_skeleton(latin_key(w))) >= 3}
    if name_keys or med_keys:
        def indic(m: re.Match) -> str:
            k = indic_key(m.group(0))
            if k in name_keys:
                return name_keys[k]
            for nk, tag in name_keys.items():
                if len(nk) >= 4 and k.startswith(nk):  # case endings joined to the name (கோபாலுக்கு)
                    return tag
            if len(_skeleton(k)) >= 3 and _skeleton(k) in med_keys:
                return med_keys[_skeleton(k)]
            return m.group(0)

        t = INDIC_WORD.sub(indic, t)

    # Names that were never saved anywhere: before an honorific, or capitalised and not a known word.
    def hon(m: re.Match) -> str:
        w = m.group(1) or m.group(2)
        if w in KIN_INDIC or w in COMMON_INDIC or is_common(w):
            return w
        return "[NAME]"

    t = HONORIFIC.sub(hon, t)
    t = NAME_LIKE.sub(lambda m: m.group(1) if is_common(m.group(1)) else "[NAME]", t)
    return t


def leaks(text: str, *, names: list[str] | tuple = (), medicines: list[str] | tuple = ()) -> list[str]:
    """What identifying bits are still in an anonymised text (the corpus refuses anything that leaks)."""
    out = []
    if PHONE.search(text):
        out.append("phone")
    if EMAIL.search(text):
        out.append("email")
    if PINCODE.search(text):
        out.append("pincode")
    for w in _name_words(names):
        if re.search(rf"\b{re.escape(w)}\b", text, re.I):
            out.append(f"name:{w}")
    keys = {latin_key(w): w for w in _name_words(names) if len(latin_key(w)) >= 2}
    meds = {_skeleton(latin_key(w)): w for w in _med_words(medicines) if len(_skeleton(latin_key(w))) >= 3}
    for word in INDIC_WORD.findall(text):
        k = indic_key(word)
        if k in keys:
            out.append(f"name:{keys[k]}")
        elif len(_skeleton(k)) >= 3 and _skeleton(k) in meds:
            out.append(f"medicine:{meds[_skeleton(k)]}")
    for w in _med_words(medicines):
        if re.search(rf"\b{re.escape(w)}", text, re.I):
            out.append(f"medicine:{w}")
    return out
