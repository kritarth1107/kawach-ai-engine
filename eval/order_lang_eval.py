"""Order understanding across languages: the brain turns a person's words into structured items, and the AI matcher picks
the store listing. Run before deploying order changes (paid: a few dozen fast-model calls, about ₹2).

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_langeval \\
    GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 \\
    MODEL_ROUTES='{"brain": ["gemini:gemini-3.5-flash@global"], "classify": ["gemini:gemini-3.5-flash@global"], ...}' \\
    PYTHONPATH=. python eval/order_lang_eval.py

The checks below are the test's own (English keywords on the brain's English item names); production code never
matches words.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

from app.brain.loop import TurnRequest, run_turn
from app.db.session import Base, SessionLocal, engine
from app.sim.world import SimHost
from app.tasks import matcher

ELDER = {"id": "elder-eval", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "son-eval", "name": "Ravi", "role": "primary caregiver"}

# (language, message, expected: list of {any: [keywords, one must appear in name or must_match], qty?, must?, max_price?, cheapest?})
BRAIN_CASES = [
    ("Hinglish", "Blinkit se 2 Amul doodh aur ek bread mangwa do", [{"any": ["milk"], "qty": 2}, {"any": ["bread"]}]),
    ("Marwari", "म्हाने पेरी पेरी मुरुक्कू अर मेथी खाखरो मंगा दो, इंस्टामार्ट सूं", [{"any": ["muruk"], "must": "peri"}, {"any": ["khakh", "khakr"], "must": "methi"}]),
    ("Hindi", "सस्ता वाला आटा मंगा दो ब्लिंकिट से", [{"any": ["atta", "flour"], "cheapest": True}]),
    ("Tamil", "இன்ஸ்டாமார்ட்டில் இரண்டு லிட்டர் பால் வாங்கு, 100 ரூபாய்க்குள்", [{"any": ["milk"], "max_price": 100}]),
    ("Bengali", "ব্লিংকিট থেকে এক প্যাকেট মারি বিস্কুট আর ডিম পাঠিয়ে দাও", [{"any": ["marie", "biscuit"]}, {"any": ["egg"]}]),
    ("English", "order khakhra under 100 rupees from instamart", [{"any": ["khakh", "khakr"], "max_price": 100}]),
    ("Hinglish", "instamart se noice ka methi aur masala khakhra dono mangaao saath me peri peri muruku bhi",
     [{"any": ["methi"]}, {"any": ["masala"]}, {"any": ["muruk"], "must": "peri"}]),
]

L = lambda name, price, pack="", available=True: {"name": name, "price": price, "pack": pack, "available": available}  # noqa: E731
# (case, item, listings, medicine, expected exact first listing name or None, expected closest name or None)
MATCH_CASES = [
    ("variant not swapped", {"name": "Peri Peri Muruku", "must_match": ["peri peri"]}, [L("Modern Kitchens Butter Muruku", "₹35", "150 g")],
     False, None, "Modern Kitchens Butter Muruku"),
    ("spelling", {"name": "Peri Peri Muruku", "must_match": ["peri peri"]},
     [L("Modern Kitchens Butter Muruku", "₹35", "150 g"), L("Haldiram Peri Peri Murukku", "₹40", "150 g")], False, "Haldiram Peri Peri Murukku", None),
    ("hindi word", {"name": "doodh"}, [L("Amul Butter", "₹60", "100 g"), L("Amul Taaza Toned Milk", "₹30", "500 ml")], False, "Amul Taaza Toned Milk", None),
    ("cheapest under 100", {"name": "Khakhra", "max_price": 100, "cheapest": True},
     [L("Prolicious High Protein Khakhra", "₹412", "7 x 50 g"), L("Jabsons Roasted Wheat Khakhra (Methi)", "₹80", "180 g"), L("Charliee Methi Khakhra", "₹65", "150 g")],
     False, "Charliee Methi Khakhra", None),
    ("single over multipack", {"name": "Diet Coke"}, [L("Diet Coke", "₹220", "6 x 300 ml"), L("Diet Coke Can", "₹40", "300 ml")], False, "Diet Coke Can", None),
    ("medicine strength", {"name": "Dolo 650"}, [L("Dolo 500 Tablet", "₹30", "15 tablets"), L("Dolo 650 Tablet", "₹33", "15 tablets")], True, "Dolo 650 Tablet", None),
    ("brand asked", {"name": "butter", "must_match": ["Amul"]}, [L("Vachan Butter", "₹117", "200 g"), L("Amul Butter", "₹60", "100 g")], False, "Amul Butter", None),
    ("unavailable skipped", {"name": "bread"}, [L("Britannia Bread", "₹50", "400 g", available=False), L("Modern Bread", "₹45", "400 g")], False, "Modern Bread", None),
]


def _hit(item: dict, kw: list[str]) -> bool:
    text = (str(item.get("name") or "") + " " + " ".join(item.get("must_match") or [])).lower()
    return any(k in text for k in kw)


async def brain_case(lang: str, message: str, expected: list[dict]) -> list[str]:
    fam = f"langeval-{uuid.uuid4().hex[:8]}"
    host = SimHost()
    async with SessionLocal() as session:
        res = await run_turn(session, host, TurnRequest(family_id=fam, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text=message,
                                                        message_ref=f"m-{uuid.uuid4().hex[:8]}", channel="whatsapp"))
        await session.rollback()
    calls = [a for a in res.actions if a.get("tool") == "start_task"]
    if not calls:
        return [f"no start_task (reply: {str(res.reply)[:120]})"]
    items = [i for i in (calls[-1].get("args") or {}).get("items") or []]
    problems = []
    if len(items) != len(expected):
        problems.append(f"{len(items)} items, expected {len(expected)}: {json.dumps(items, ensure_ascii=False)}")
    for e in expected:
        it = next((i for i in items if _hit(i, e["any"])), None)
        if not it:
            problems.append(f"no item for {e['any']}: {json.dumps(items, ensure_ascii=False)}")
            continue
        if e.get("qty") and int(it.get("qty") or 1) != e["qty"]:
            problems.append(f"{it.get('name')}: qty {it.get('qty')} not {e['qty']}")
        if e.get("must") and not any(e["must"] in str(m).lower() for m in it.get("must_match") or []):
            problems.append(f"{it.get('name')}: must_match {it.get('must_match')} lacks {e['must']}")
        if e.get("max_price") and float(it.get("max_price") or 0) != e["max_price"]:
            problems.append(f"{it.get('name')}: max_price {it.get('max_price')} not {e['max_price']}")
        if e.get("cheapest") and not it.get("cheapest"):
            problems.append(f"{it.get('name')}: cheapest not set")
    return problems


async def match_case(item: dict, listings: list[dict], medicine: bool, want_exact: str | None, want_closest: str | None) -> list[str]:
    out = await matcher.choose([{"qty": 1, **item}], [listings], medicine=medicine)
    if out is None:
        return ["no answer"]
    r = out[0]
    first = listings[r["exact"][0]]["name"] if r["exact"] else None
    closest = listings[r["closest"]]["name"] if r.get("closest") is not None else None
    problems = []
    if first != want_exact:
        problems.append(f"exact first {first!r}, expected {want_exact!r} ({r.get('why')})")
    if want_closest and closest != want_closest:
        problems.append(f"closest {closest!r}, expected {want_closest!r}")
    return problems


async def main() -> int:
    async with engine.begin() as conn:
        from app.care import baselines, memory_index, models  # noqa: F401
        from app.learn import models as learn_models  # noqa: F401
        from app.llm import spend  # noqa: F401
        from app.models import entities  # noqa: F401
        from app.specialists import channels  # noqa: F401
        from app.tasks import models as task_models  # noqa: F401

        await conn.run_sync(Base.metadata.create_all)
        from app.db.migrate import run_v2_migrations

        await run_v2_migrations(conn)
    failed = 0
    only = set(sys.argv[1:])
    if not only or "brain" in only:
        for lang, msg, exp in BRAIN_CASES:
            probs = await brain_case(lang, msg, exp)
            failed += bool(probs)
            print(("PASS" if not probs else "FAIL"), f"brain/{lang}: {msg}", *(f"\n    - {p}" for p in probs))
    if not only or "match" in only:
        for name, item, ls, med, want, closest in MATCH_CASES:
            probs = await match_case(item, ls, med, want, closest)
            failed += bool(probs)
            print(("PASS" if not probs else "FAIL"), f"match/{name}", *(f"\n    - {p}" for p in probs))
    total = (len(BRAIN_CASES) if not only or "brain" in only else 0) + (len(MATCH_CASES) if not only or "match" in only else 0)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
