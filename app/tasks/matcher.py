"""Which store listing is the thing the person asked for: an AI decision, not word rules (founder 2026-10-10: people
order in many languages and dialects; predefined words cannot judge "peri peri muruku" vs "butter muruku", "doodh" vs
"milk", "muruku" vs "Murukku").

The brain turns the person's words into a structured item (product words, quantity, the traits that must match, a price
limit, cheapest). This module shows a fast model that item and the store's listings and gets back, per item, the listing
to buy and whether it is exactly what was asked, another variant of it, or nothing. Code then only checks data (in
stock, the serving store, price limit) and never puts another variant in silently.
"""

from __future__ import annotations

import json
import logging

from app.llm import router

logger = logging.getLogger(__name__)

PROMPT = """You choose groceries and medicines for an elderly person in India from a store's search results.
For each ITEM you get the store's LISTINGS (numbered). Return JSON only:
{"picks": [{"item": <item number>, "exact": [<listing numbers that are exactly the item, best first>], "closest": <listing number or null>, "why": "<short reason>"}]}

- exact: listings that ARE the product asked for: every trait in must_match is true of it (flavour, variant, brand,
  size, strength) and it is the same kind of product. Spelling, script, language and word order do not matter:
  "muruku" = "Murukku", "doodh" = milk, "khakra" = "khakhra", "atta" = wheat flour, "anda" = eggs. A brand or flavour
  that was not asked for is fine. Order them best first: a single pack unless more were asked (no "x 2", "pack of 6",
  "combo" unless asked); within max_price if given; the cheapest first if cheapest is true; otherwise the most
  ordinary everyday choice among the first few, not a premium or gift pack. Never include a listing marked unavailable.
- closest: when exact is empty, the nearest other variant of the same kind of product (asked peri peri muruku, there is
  only butter muruku → closest is butter muruku), else null. It will NOT be bought without asking the person.
- Medicines (medicine: true): exact only when the medicine name and strength match.
- FAMILY (when given): products this family bought before and what they said about them, what they like or must avoid.
  Among exact listings, put one they bought before and liked first; never rank first one they did not like, or one against
  their diet or "never order" notes.
"""


def _listing(i: int, x: dict) -> str:
    bits = [f"{i}. {x.get('name')}", str(x.get("pack") or ""), str(x.get("price") or "")]
    if x.get("available") is False:
        bits.append("[unavailable]")
    return " | ".join(b for b in bits if b)


async def choose(items: list[dict], listings: list[list[dict]], *, medicine: bool = False, family: list[str] | None = None) -> list[dict] | None:
    """items[i] (structured ask) with listings[i] (store results) → per item {"exact": [listing idx, best first],
    "closest": idx | None, "why"}. None when the model could not answer (the person then picks from the listings)."""
    if not items:
        return []
    blocks = []
    for n, (it, ls) in enumerate(zip(items, listings), 1):
        ask = {k: it.get(k) for k in ("name", "qty", "must_match", "max_price", "cheapest") if it.get(k) not in (None, "", [], False)}
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
