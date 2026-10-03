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
    r"\b[\w-]+\s+(?:road|rd|street|st|nagar|colony|sector|layout|marg|lane|cross|main|enclave|vihar|apartments?|society|towers?)\b",
    re.I,
)
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
