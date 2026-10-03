"""Strip anything that identifies a family before an example is shared across families.

Names of household members and anyone in the care record, phone numbers, emails, pincodes, street addresses,
order/ride ids and exact medicine names are replaced with placeholders. Kinship words (Mummy, Papa, beta, ji)
stay: they carry tone, not identity.
"""

from __future__ import annotations

import re

PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PINCODE = re.compile(r"(?<!\d)[1-9]\d{2}\s?\d{3}(?!\d)")
URL = re.compile(r"https?://\S+")
IDS = re.compile(r"\b[A-Z]{2,5}-?\d{4,}\b|\b\d{8,}\b")
ADDRESS = re.compile(
    r"\b(?:flat|house|h\.?\s?no\.?|plot|door|block|tower|wing)\s*[\w/-]+|"
    r"\b[\w-]+\s+(?:road|rd|street|nagar|colony|sector|layout|marg|lane|enclave|vihar|apartments?|society|towers?)\b",
    re.I,
)
# A word just before one of these is a name, unless it is a kinship word ("Gopal bhaiya", "कमला जी", "Ravi garu").
HONORIFIC = re.compile(
    r"\b([A-Z][a-z]{2,})(?=\s+(?:ji|jee|beta|bhaiya|bhai|didi|di|sahab|saab|babu|garu|aunty|uncle|madam|sir|bhabhi|mausi|chacha|kaka|kaki|maasi)\b)|"
    r"([\u0900-\u0DFF]{2,})(?=\s*(?:जी|बेटा|भैया|दीदी|साहब|बाबू|आंटी|अंकल|গাৰু|বাবু|দিদি|ଜୀ|ବାବୁ|గారు|அவர்கள்))"
)
KIN_INDIC = {"आंटी", "मम्मी", "पापा", "बाबा", "अम्मा", "दादी", "नानी", "माँ", "मां", "अम्मी", "दादा", "नाना", "भैया", "दीदी", "बहन", "भाई", "बेटा", "बेटी",
             "डॉक्टर", "सर", "मैडम"}
# Capitalised words that are not names (kept when they start a sentence or are just common).
COMMON = set("""
i im i'm ive i've ill i'll id i'd a an the and or but so if then when what why how who where which this that these those it its he she
they we you your yours my me our us his her their them is are was were be been am do does did have has had will would should could
can may might must please thank thanks thankyou ok okay yes no not hi hello hey dear good morning afternoon evening night great nice
sure sorry also just now today tomorrow yesterday tonight here there all any some every each one two three first last next soon
monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november
december sugar bp blood pressure medicine medicines tablet tablets doctor doctors dr hospital clinic report reports test tests walk
water tea chai lunch dinner breakfast food time reminder reminders note noted saheli whatsapp kavach india indian hindi english
jai shri shree krishna ram radhe ji beta haan ha han nahi nahin acha accha achha theek thik chalo bas aap aapka aapki aapke main mai
maine kya kaise kab kahan kyun abhi phir bhi aur toh to ki ka ke hai hain tha thi ho hoon hu rab rakha sat akal salaam namaste
namaskar pranam shubh ratri suprabhat subah shaam raat din dawai dawa goli insulin metformin don't dont can't cant won't wont that's
let let's lets also still yet only very much more most less few many well right wrong true fine better best worse oh ah wow
busy free home work office walk sleep rest pain headache fever cough cold mild normal later call message noted please done
mummy papa bapa baba ammi abbu aunty uncle sir madam professor sahab bhaiya didi happy birthday diwali holi eid festival puja
apni apna apne aaj kal bahut since take apply sunkar unhone unki unka unke agar sare saare tumhi tumhala ami aai allah oct nov dec
sep aug jan feb mar apr jun jul bolus after before with without because about from into upon please kindly also anything something
nothing everything everyone someone kuch koi sab sabhi mera meri mere tera teri hamara hamari humne hum tum tumhe unko usko isko
yeh ye woh wo yahan wahan jab tab kyunki lekin magar par phir aaram dhyan khana pani neend dard chinta khush
""".split())
COMMON_INDIC = {"ठीक", "शुभ", "बहुत", "धन्यवाद", "नमस्ते", "हाँ", "हां", "जी", "अच्छा", "सुप्रभात", "शुक्रिया", "जय", "श्री", "राम", "कृष्णा",
                "राधे", "खुश", "प्यारे", "मेरे", "मेरी", "आपका", "आपकी", "नहीं", "सब", "चलो", "अरे", "आज", "कल", "रात", "सुबह", "शाम"}
NAME_LIKE = re.compile(r"(?<![\w\[])([A-Z][a-z]{2,})(?![\w\]])")
KEEP = {"mummy", "papa", "maa", "amma", "appa", "baba", "dadi", "dada", "nani", "nana", "beta", "beti", "ji", "aunty", "uncle", "didi",
        "bhaiya", "bhai", "saheli", "doctor", "sir", "madam", "bapa", "achan", "amma", "ammi", "abbu"}


def anonymise(text: str, *, names: list[str] | tuple = (), medicines: list[str] | tuple = ()) -> str:
    t = text or ""
    t = URL.sub("[LINK]", t)
    t = EMAIL.sub("[EMAIL]", t)
    t = PHONE.sub("[PHONE]", t)
    t = ADDRESS.sub("[ADDRESS]", t)
    t = PINCODE.sub("[PIN]", t)
    t = IDS.sub("[ID]", t)
    for i, med in enumerate(sorted({m for m in medicines if m and len(m) >= 3}, key=len, reverse=True)):
        t = re.sub(rf"\b{re.escape(med)}\w*", f"[MED{chr(65 + i % 26)}]", t, flags=re.I)
    words = set()
    for n in names:
        for w in re.findall(r"[^\W\d_]{3,}", n or ""):
            if w.lower() not in KEEP:
                words.add(w)
    people = {}
    for w in sorted(words, key=len, reverse=True):
        people.setdefault(w.lower(), f"[PERSON{len(people) + 1}]")
        t = re.sub(rf"\b{re.escape(w)}\b", people[w.lower()], t, flags=re.I)
    # Names that were never saved anywhere: before an honorific, or capitalised and not a common word.
    def hon(m: re.Match) -> str:
        w = m.group(1) or m.group(2)
        if w.lower() in KEEP or w in KIN_INDIC or w in COMMON_INDIC or w.lower() in COMMON:
            return w
        return "[NAME]"

    t = HONORIFIC.sub(hon, t)
    t = NAME_LIKE.sub(lambda m: m.group(1) if (m.group(1).lower() in COMMON or m.group(1).lower() in KEEP) else "[NAME]", t)
    return t


def leaks(text: str, *, names: list[str] | tuple = ()) -> list[str]:
    """What identifying bits are still in an anonymised text (the corpus refuses anything that leaks)."""
    out = []
    if PHONE.search(text):
        out.append("phone")
    if EMAIL.search(text):
        out.append("email")
    if PINCODE.search(text):
        out.append("pincode")
    for n in names:
        for w in re.findall(r"[^\W\d_]{3,}", n or ""):
            if w.lower() not in KEEP and re.search(rf"\b{re.escape(w)}\b", text, re.I):
                out.append(f"name:{w}")
    return out
