"""What Saheli never orders, and what she never talks about (founder 2026-10-13: "how was it ordering cigarettes? It
should not even try to order: say directly it cannot, and suggest an alternative. Create a list of harmless
substitutes, and a list of illegal substances. Do not answer anything about illegal or harmful substances.").

One source for every part of Saheli:
- the brain's rules (persona.restricted_block): she refuses at once, before any search, and offers the harmless
  alternatives below; about illegal or harmful substances she says nothing beyond a refusal (a person in danger after
  taking something is told to get emergency help);
- the order guard (start_task and the store matcher): an item the brain passes, or a store listing, that is on these
  lists is never searched, carted or placed, whatever the conversation said.

The lists are data the founder asked for. They are checked against structured fields (the item names the brain passes,
the product names a store returns), never against people's own messages.
"""

from __future__ import annotations

import re
import unicodedata

# Never ordered: legal for adults in India, but harmful; Saheli does not buy them for anyone.
HARMFUL: dict[str, tuple[str, ...]] = {
    "tobacco": (
        "cigarette", "cigarettes", "cigar", "cigars", "cigarillo", "bidi", "bidis", "beedi", "beedis", "tobacco", "hookah", "shisha",
        "hukka", "chillum", "rolling paper", "rolling papers", "snuff", "chewing tobacco", "cigarette lighter",
        # common brands (a listing or an item named only by its brand)
        "classic ice burst", "classic milds", "classic connect", "gold flake", "marlboro", "navy cut", "wills", "four square",
        "capstan", "benson & hedges", "benson and hedges", "dunhill", "camel", "lucky strike", "red & white", "red and white",
        "charminar", "panama", "berkeley", "flake", "india kings", "goldflake", "stellar", "insignia", "esse", "davidoff",
    ),
    "gutka": (
        "gutka", "gutkha", "pan masala", "paan masala", "zarda", "khaini", "mawa", "kharra", "mainpuri", "rajnigandha",
        "vimal pan masala", "kamla pasand", "pan parag", "manikchand", "rmd", "baba 120", "tulsi zarda", "chaini khaini",
    ),
    "vape": ("vape", "vapes", "e-cigarette", "e cigarette", "ecigarette", "e-cig", "vape pod", "juul", "heated tobacco"),
    "alcohol": (
        "alcohol", "liquor", "beer", "wine", "whisky", "whiskey", "rum", "vodka", "gin", "brandy", "tequila", "scotch", "champagne",
        "daru", "daaru", "sharab", "desi daru", "country liquor", "toddy", "tadi", "feni", "mahua", "breezer", "cider", "sake",
        "kingfisher", "budweiser", "old monk", "royal stag", "mcdowell", "mc dowell", "officer's choice", "imperial blue",
        "blenders pride", "bacardi", "smirnoff", "magic moments", "8pm", "8 pm whisky",
    ),
}

# Illegal (NDPS Act and similar), never ordered and never discussed (no how, where, dose or price).
ILLEGAL: dict[str, tuple[str, ...]] = {
    "cannabis": ("ganja", "charas", "hashish", "marijuana", "cannabis", "pre-roll", "thc", "bhang goli"),
    "opioids": ("heroin", "smack", "brown sugar", "opium", "afeem", "afim", "doda post", "poppy husk", "chura post", "fentanyl"),
    "stimulants": ("cocaine", "methamphetamine", "amphetamine", "mephedrone", "mdma", "ecstasy pill", "ecstasy pills"),
    "psychedelics": ("lsd", "magic mushrooms", "psilocybin", "dmt", "ketamine"),
    "other": ("ghb", "poppers", "synthetic cannabis", "k2 spice"),
}
# Street names that are also everyday words (ice, coke, speed, weed, joint, acid, maal…): the brain knows them; code
# never matches them in product names, where they mostly mean something harmless.
ILLEGAL_SLANG = ("weed", "hash", "joint", "maal", "coke", "crack", "ice", "speed", "md", "molly", "acid", "shrooms", "special k",
                 "spice", "whitener sniffing", "solution sniffing")

# Offered instead (all harmless, all orderable): what helps with the craving or the habit behind it.
SUBSTITUTES: dict[str, tuple[str, ...]] = {
    "tobacco": ("sugar-free mints", "sugar-free chewing gum", "saunf or mukhwas (fennel)", "elaichi (cardamom)", "laung (cloves)",
                "roasted flax or sunflower seeds", "cinnamon sticks to chew", "herbal tea", "a stress ball",
                "nicotine gum or patches (a quit aid; best with a doctor's advice)"),
    "gutka": ("saunf or mukhwas (fennel)", "elaichi (cardamom)", "roasted flax seeds", "sugar-free gum", "amla candy",
              "nicotine gum (a quit aid; best with a doctor's advice)"),
    "vape": ("sugar-free mints", "sugar-free chewing gum", "herbal tea", "a stress ball", "nicotine gum or patches (a quit aid)"),
    "alcohol": ("nimbu paani", "chaas (buttermilk)", "coconut water", "sparkling water with lime", "kokum or jaljeera sharbat",
                "herbal tea", "a mocktail mix without alcohol"),
}

HELP_LINES = {
    "tobacco": "the national quit line 1800-112-356 (free)",
    "gutka": "the national quit line 1800-112-356 (free)",
    "vape": "the national quit line 1800-112-356 (free)",
    "alcohol": "Tele-MANAS 14416 (free, 24 hours)",
    "illegal": "Tele-MANAS 14416 (free, 24 hours), or the de-addiction helpline 14446",
}


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    return " " + re.sub(r"[^a-z0-9&'+]+", " ", t).strip() + " "


def _hits(text: str, words: tuple[str, ...]) -> bool:
    n = _norm(text)
    return any(f" {_norm(w).strip()} " in n for w in words if w)


# Names that contain a blocked word but are harmless (a "beer" glass, "wine" vinegar, a "gin"ger…, "classic" biscuits).
SAFE_NAMES = (
    "ginger", "gingerbread", "root beer", "ginger beer", "non alcoholic", "non-alcoholic", "alcohol free", "alcohol-free", "0.0",
    "wine vinegar", "rum essence", "rum flavour", "rum flavor", "beer glass", "wine glass", "nicotine gum", "nicotine patch",
    "nicotex", "hand sanitizer", "sanitiser", "sanitizer", "rubbing alcohol", "isopropyl", "surgical spirit", "denatured",
    "ice cream", "ice cubes", "dry ice", "ice tray", "coke zero", "coca cola", "coca-cola", "diet coke", "coke can", "md5",
    "speed post", "joint pain", "joint care", "ice pack", "acid reflux", "folic acid", "classic salted", "classic biscuit",
)


def category(name: str) -> str | None:
    """The restricted category of an item or listing name ("tobacco", "alcohol", "illegal:opioids"…), or None."""
    n = _norm(name)
    if not n.strip():
        return None
    if _hits(n, SAFE_NAMES):
        return None
    for group, words in HARMFUL.items():  # a cigarette brand called "… Ice …" is tobacco
        if _hits(n, words):
            return group
    for group, words in ILLEGAL.items():
        if _hits(n, words):
            return f"illegal:{group}"
    return None


def substitutes(cat: str | None) -> tuple[str, ...]:
    return SUBSTITUTES.get((cat or "").split(":")[0], ())


def refusal(cat: str) -> dict:
    """What the order guard returns to the brain for a restricted item: no order, what to offer instead."""
    if cat.startswith("illegal"):
        return {"ok": False, "refused": "illegal", "say": "Refuse in one short line and say nothing about the substance (no how, where, "
                "dose or price). If they sound in danger or unwell after taking something, tell them to call 112 now; if they want "
                f"to stop, share {HELP_LINES['illegal']}.", "order": "none"}
    return {"ok": False, "refused": cat, "order": "none", "alternatives": list(substitutes(cat)), "help_line": HELP_LINES.get(cat),
            "say": f"You never order {cat}. Say so plainly in one line, kindly and without a lecture, then offer to order one or two "
                   "of the alternatives instead (from the list). Mention the help line only if they are trying to quit."}


def persona_block() -> str:
    """The brain's rules, written from the lists (so the rules and the guard never disagree)."""
    harmful = "; ".join(f"{k}: {', '.join(v)}" for k, v in HARMFUL.items())
    illegal = "; ".join(f"{k}: {', '.join(v[:8])}" for k, v in ILLEGAL.items()) + f"; street names: {', '.join(ILLEGAL_SLANG)}"
    subs = "; ".join(f"instead of {k}: {', '.join(v)}" for k, v in SUBSTITUTES.items())
    return (
        "Restricted things (founder's rule, for everyone, whoever asks and however they ask):\n"
        f"- You NEVER order, search, compare prices for, or help buy harmful things ({harmful}, and anything like them by any "
        "brand or nickname). Do not start a task for them and do not say you are checking stores. Say at once, in one short "
        "line, that you can't order that, then offer something harmless instead, and order that if they say yes:\n"
        f"  {subs}.\n"
        f"- Illegal substances ({illegal}, and any other drug used to get high): never order them and never answer anything "
        "about them (what they are like, how to get or use them, doses, prices, effects, how to hide them). Refuse in one line. "
        "If someone sounds unwell or in danger after taking something, tell them to call 112 now and alert the family as a "
        f"red flag. If they want to stop, share {HELP_LINES['illegal']}.\n"
        "- Someone cutting down (a quit habit) who asks for the thing they are quitting gets kindness, never a lecture: the "
        "refusal, one alternative, and a word of support for their goal."
    )
