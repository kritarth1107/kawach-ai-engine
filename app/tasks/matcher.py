"""Which store listing is the thing the person asked for: an AI decision, not word rules (founder 2026-10-10: people
order in many languages and dialects; predefined words cannot judge "peri peri muruku" vs "butter muruku", "doodh" vs
"milk", "muruku" vs "Murukku").

The brain turns the person's words into a structured item (product words, quantity, the traits that must match, a price
limit, cheapest). This module shows a fast model that item and the store's listings and gets back, per item, the listing
to buy and whether it is exactly what was asked, another variant of it, or nothing. Code then only checks data (in
stock, the serving store, price limit) and never puts another variant in silently.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.llm import router

logger = logging.getLogger(__name__)

PROMPT = """You choose groceries and medicines for an elderly person in India from a store's search results.
For each ITEM you get the store's LISTINGS (numbered). Return JSON only:
{"picks": [{"item": <item number>, "exact": [<listing numbers that are exactly the item, best first>], "closest": <listing number or null>, "why": "<short reason>"}]}

- exact: listings that ARE the product asked for: every trait in must_match is true of it (flavour, variant, brand,
  size, strength) and it is the same kind of product. A flavour or variant in the item's own name is asked for too,
  even when must_match is empty ("Methi Khakhra" is never a masala khakhra; "Choco Berry bar" never a choco almond
  one). Spelling, script, language and word order do not matter:
  "muruku" = "Murukku", "doodh" = milk, "khakra" = "khakhra", "atta" = wheat flour, "anda" = eggs. A brand or flavour
  that was not asked for is fine. Order them best first: the listing for ONE unit of what was asked (qty is how many
  of the chosen listing go in the cart, so "4 bars" is 4 x the single bar: a multi-pack such as "x 4", "pack of 6" or a
  combo comes only when they asked for that pack itself, never to make up qty); within max_price if given; the cheapest
  first if cheapest is true; otherwise the most ordinary everyday choice among the first few, not a premium or gift
  pack. Never include a listing marked unavailable.
- closest: when exact is empty, the nearest other variant of the same kind of product (asked peri peri muruku, there is
  only butter muruku → closest is butter muruku), else null. It will NOT be bought without asking the person.
- Medicines (medicine: true): exact only when the medicine name and strength match.
- Food with restaurant given: exact only from that restaurant (spelling may differ: "Haldiram" = "Haldiram's"). A listing
  marked closed or unavailable is never exact or closest.
- FAMILY (when given): products this family bought before and what they said about them, what they like or must avoid.
  Among exact listings, put one they bought before and liked first; never rank first one they did not like, or one against
  their diet or "never order" notes.
"""


def _listing(i: int, x: dict) -> str:
    bits = [f"{i}. {x.get('name')}", str(x.get("pack") or ""), str(x.get("price") or "")]
    if x.get("restaurant"):
        bits.append(f"from {x['restaurant']}")
    if x.get("closed"):
        bits.append("[restaurant closed now" + (f", {x['opens']}" if x.get("opens") else "") + "]")
    elif x.get("available") is False:
        bits.append("[unavailable]")
    return " | ".join(b for b in bits if b)


async def choose(items: list[dict], listings: list[list[dict]], *, medicine: bool = False, family: list[str] | None = None) -> list[dict] | None:
    """items[i] (structured ask) with listings[i] (store results) → per item {"exact": [listing idx, best first],
    "closest": idx | None, "why"}. None when the model could not answer (the person then picks from the listings)."""
    if not items:
        return []
    blocks = []
    for n, (it, ls) in enumerate(zip(items, listings), 1):
        ask = {k: it.get(k) for k in ("name", "qty", "must_match", "max_price", "cheapest", "restaurant") if it.get(k) not in (None, "", [], False)}
        blocks.append(f"ITEM {n}: {json.dumps(ask, ensure_ascii=False)}" + (" medicine: true" if medicine else "")
                      + "\nLISTINGS:\n" + ("\n".join(_listing(i, x) for i, x in enumerate(ls, 1)) or "(none)"))
    try:
        fam = ("FAMILY:\n" + "\n".join(f"- {x}" for x in family[:40]) + "\n\n") if family else ""
        reply = await router.complete("classify", system_stable=PROMPT, effort="low", max_tokens=1200, timeout_s=25,
                                      messages=[{"role": "user", "content": [{"type": "text", "text": fam + "\n\n".join(blocks)}]}])
        from app.care.extract import parse_json

        data = parse_json(reply.text)
    except Exception as exc:  # noqa: BLE001 — no answer: the person picks from the listings
        logger.warning("matcher failed: %s", exc)
        return None
    out: list[dict] = [{"exact": [], "closest": None, "why": "no answer"} for _ in items]
    for p in data.get("picks") or []:
        try:
            n = int(p.get("item")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= n < len(items):
            continue
        ls = listings[n]
        ok = lambda i: isinstance(i, int) and 1 <= i <= len(ls) and ls[i - 1].get("available") is not False  # noqa: E731
        exact = [i - 1 for i in dict.fromkeys(p.get("exact") or []) if ok(i)]
        closest = p.get("closest") - 1 if ok(p.get("closest")) and not exact else None
        out[n] = {"exact": exact, "closest": closest, "why": str(p.get("why") or "")[:160]}
    return out


PLACE_PROMPT = """Two delivery addresses in India. SAVED is the family's chosen place; SHOWN is what the store's checkout shows.
Are they the same place (the same house or flat in the same building, area and city)? Different spellings, a missing
pincode, a label like "Home" or a name before the address do not matter; another house, flat, block, building or area
does. JSON only: {"same": true | false, "why": "<a few words>"}"""


async def same_place(saved_full: str, pincode: str, shown: str) -> bool | None:
    """Is the store's checkout address the family's saved place? None when the model could not answer (the cart check's own
    pincode rule still applies). Live run 2026-10-11: the agent picked the account's old "Home" in another area for the saved place."""
    if not saved_full or not shown:
        return None
    try:
        reply = await router.complete("classify", system_stable=PLACE_PROMPT, effort="low", max_tokens=300, timeout_s=20,
                                      messages=[{"role": "user", "content": [{"type": "text", "text":
                                                 f"SAVED: {saved_full} (pincode {pincode or 'unknown'})\nSHOWN: {shown}"}]}])
        from app.care.extract import parse_json

        data = parse_json(reply.text)
    except Exception as exc:  # noqa: BLE001 — the pincode check still stands
        logger.warning("same_place failed: %s", exc)
        return None
    return data.get("same") if isinstance(data.get("same"), bool) else None


WHICH_PLACE_PROMPT = """A person in India named a delivery place in their own words (any language or script). SAVED lists the family's
saved places, numbered: each one's name, address, and "(default)" for the person's usual place. Which saved place do the
words mean?
- Words for one's own home in any language ("ghar", "mera ghar", "my place", "home", "घर", "veedu", "bari", "ghari") mean
  the place named for home (Home, Ghar, …), else the default place.
- A relation with a home word ("beta ke ghar" = the son's home, "beti ka flat", "didi ke yahan") means the place whose name
  is for that person (Beta's flat, Beti's house, …).
- The place's own name in any spelling or script, its area, city or pincode, or its kind ("clinic", "office") also count,
  when only one saved place fits.
- 0 when the words mean a place that is not saved (another address, a landmark, a station, a city with no saved place)
  or could be more than one of them.
JSON only: {"place": <number of the saved place, or 0>, "why": "<a few words>"}"""


async def _which_place_once(words: str, named: list[dict]) -> str | None:
    listed = "\n".join(f"{n}. {p.get('name')}: {p.get('address') or ''}" + (" (default)" if p.get("isDefault") or p.get("default") else "")
                       for n, p in enumerate(named, 1))
    reply = await router.complete("classify", system_stable=WHICH_PLACE_PROMPT, effort="low", max_tokens=200, timeout_s=15,
                                  messages=[{"role": "user", "content": [{"type": "text", "text": f"WORDS: {words}\nSAVED:\n{listed}"}]}])
    from app.care.extract import parse_json

    got = parse_json(reply.text).get("place")
    names = [str(p.get("name")) for p in named]
    if isinstance(got, str) and got in names:  # the name instead of its number
        return got
    try:
        n = int(got)
    except (TypeError, ValueError):
        return None
    return names[n - 1] if 1 <= n <= len(names) else None


async def which_place(words: str, places: list[dict]) -> str | None:
    """The saved place a person's words mean, by its exact saved name; None when it is none of them or unclear. An AI
    decision (founder 2026-10-10: no word lists on people's words; this replaces the backend's place word list), asked
    twice at once: a place either answer names is used unless the two name different places (then the person is asked)."""
    named = [p for p in places if p.get("name")]
    if not words or not named:
        return None
    got = await asyncio.gather(_which_place_once(words, named), _which_place_once(words, named), return_exceptions=True)
    errors = [g for g in got if isinstance(g, Exception)]
    if len(errors) == len(got):
        logger.warning("which_place failed: %s", errors[0])
        raise errors[0]  # the backend's own lookup still runs on the words
    picks = {g for g in got if isinstance(g, str)}
    return picks.pop() if len(picks) == 1 else None
