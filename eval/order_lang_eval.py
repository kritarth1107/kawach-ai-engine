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
    # every family, not one (2026-10-11): more languages and the Bangalore order of that day
    ("English", "Hey can u order 4 rite bite choco berry bars from blinkit", [{"any": ["choco berry"], "qty": 4}]),
    ("Kannada", "ಬ್ಲಿಂಕಿಟ್‌ನಿಂದ ಎರಡು ಪ್ಯಾಕೆಟ್ ನಂದಿನಿ ಹಾಲು ತರಿಸಿ", [{"any": ["nandini"], "qty": 2}]),
    ("Telugu", "జెప్టో నుండి ఒక కిలో ఉల్లిపాయలు పంపండి", [{"any": ["onion"]}]),
    ("Gujarati", "બ્લિંકિટથી અમૂલ દૂધ મંગાવો", [{"any": ["milk", "amul"]}]),
    ("Punjabi", "ਇੰਸਟਾਮਾਰਟ ਤੋਂ ਦੋ ਪੈਕਟ ਬ੍ਰੈੱਡ ਭੇਜ ਦਿਓ", [{"any": ["bread"], "qty": 2}]),
    ("Marathi", "झेप्टो वरून दोन पाकीट गोकुळ दूध पाठवा", [{"any": ["gokul"], "qty": 2}]),
]

# the stores they named (founder 2026-10-11: every store the person named is searched and reported)
STORE_CASES = [
    ("Hinglish", "Karachi Bakery ke fruit biscuit chahiye, Blinkit, Instamart aur Zepto teeno pe dekh lo", {"blinkit", "instamart", "zepto"}),
    ("Hindi", "ब्लिंकिट और ज़ेप्टो दोनों पर अमूल मक्खन देखो", {"blinkit", "zepto"}),
    ("Tamil", "ஆவின் பால் இன்ஸ்டாமார்ட் மற்றும் ப்ளிங்கிட் இரண்டிலும் பாருங்கள்", {"instamart", "blinkit"}),
    ("English", "get me amul butter from blinkit", {"blinkit"}),
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
    # a flavour in the item's own name counts even when the brain left must_match empty (lang eval 2026-10-11)
    ("flavour in the name only", {"name": "Methi Khakhra"}, [L("Charliee Masala Khakhra", "₹65", "150 g"), L("Charliee Methi Khakhra", "₹65", "150 g")],
     False, "Charliee Methi Khakhra", None),
    ("flavour in the name only, not there", {"name": "Methi Khakhra"}, [L("Charliee Masala Khakhra", "₹65", "150 g")],
     False, None, "Charliee Masala Khakhra"),
    # qty is how many of the listing go in the cart: "4 bars" is 4 x the single bar, never 4 x a 4-pack (Instamart, 2026-10-11)
    ("4 bars: the single bar, not the 4-pack", {"name": "RiteBite Choco Berry protein bar", "qty": 4, "must_match": ["Choco Berry", "RiteBite"]},
     [L("RiteBite Max Protein Daily Choco Berry 10g Protein Bar", "₹316", "50 g x 4"), L("RiteBite Max Protein Daily Choco Berry 10g Protein Bar", "₹79", "50 g")],
     False, "RiteBite Max Protein Daily Choco Berry 10g Protein Bar|50 g", None),
    # what Blinkit listed live that day: the everyday 50 g bar first
    ("the live Blinkit listings", {"name": "RiteBite Choco Berry protein bar", "qty": 4, "must_match": ["Choco Berry", "RiteBite"]},
     [L("RiteBite Max Protein Daily Choco Berry 10 g Protein Bar", "₹80", "50 g"), L("RiteBite Max Protein Choco Berry 30 g Protein Bar", "₹177", "100 g"),
      L("RiteBite Max Protein Daily Choco Almond Protein Bar + RiteBite Max Protein Daily Choco Berry Protein Bar Combo", "₹151", "50 g + 50 g")],
     False, "RiteBite Max Protein Daily Choco Berry 10 g Protein Bar", None),
]


# which saved place a person's words mean (an AI decision; the backend matches the exact name it gives)
SAVED = [{"name": "Home", "address": "21 5th Main, Jayanagar, Bangalore 560011", "isDefault": True},
         {"name": "Beta's flat", "address": "B-7, Silver Oak Residency, Vijay Nagar, Indore 452010"},
         {"name": "Clinic", "address": "Shop 3, Arera Medical Plaza, Bhopal 462016"}]
PLACE_CASES = [("ghar pe hi", "Home"), ("घर", "Home"), ("வீட்டுக்கு", "Home"), ("বাড়িতে", "Home"), ("my place", "Home"),
               ("beta ke ghar", "Beta's flat"), ("Indore wale flat pe", "Beta's flat"), ("clinic", "Clinic"),
               ("Bhopal medical plaza", "Clinic"), ("railway station", None), ("Mumbai mein meri behen ke yahan", None)]


# is the ride app's pickup or drop the place meant? (live 2026-10-12: a saved home became the city in the app)
RIDE_END_CASES = [
    ("12 Civil Lines, Raipur 492001", True, "Raipur, Chhattisgarh", False),
    ("12 Civil Lines, Raipur 492001", True, "Home, 12, Civil Lines, Raipur, Chhattisgarh 492001", True),
    ("Flat 3B, Lake View Apartments, Salt Lake Sector 1, Kolkata 700064", True, "Lake View Apartments, Sector 1, Salt Lake City, Kolkata", True),
    ("Flat 3B, Lake View Apartments, Salt Lake Sector 1, Kolkata 700064", True, "Salt Lake City, Kolkata, West Bengal", False),
    ("B-14, Lajpat Nagar 2, New Delhi 110024", True, "B-14, Lajpat Nagar 4, New Delhi", False),
    ("Chennai Central station", False, "Puratchi Thalaivar Dr. M.G. Ramachandran Central Railway Station, Chennai", True),
    ("Hitech City metro station", False, "Hyderabad, Telangana", False),
    ("Pune", False, "Pune, Maharashtra, India", True),
    ("एम्स अस्पताल", False, "All India Institute of Medical Sciences, Ansari Nagar, New Delhi", True),
    ("எழும்பூர் ரயில் நிலையம்", False, "Chennai Egmore Railway Station, Egmore, Chennai", True),
]


async def ride_end_case(asked: str, saved: bool, shown: str, want: bool) -> list[str]:
    got = await matcher.same_ride_end(asked, shown, saved=saved)
    if got is None:  # the model did not answer (a quota error on the shared eval project): once more
        await asyncio.sleep(20)
        got = await matcher.same_ride_end(asked, shown, saved=saved)
    return [] if got is want else [f"{asked!r} vs {shown!r} → {got!r}, expected {want!r}"]


async def place_case(words: str, want: str | None) -> list[str]:
    got = await matcher.which_place(words, SAVED)
    return [] if got == want else [f"{words!r} → {got!r}, expected {want!r}"]


def _hit(item: dict, kw: list[str]) -> bool:
    text = (str(item.get("name") or "") + " " + " ".join(item.get("must_match") or [])).lower()
    return any(k in text for k in kw)


class PlacesHost(SimHost):
    """A family with one saved place, as every real family that orders has (without it the brain rightly asks for an
    address first, and the case would test that instead)."""

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "delivery_place":
            return {"addressId": "addr-ghar", "nickname": "Ghar", "full": "12 Civil Lines, Raipur 492001", "pincode": "492001", "matched": True}
        if tool == "list_places":
            return {"places": [{"name": "Ghar", "address": "12 Civil Lines, Raipur 492001", "isDefault": True}]}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


async def brain_case(lang: str, message: str, expected: list[dict]) -> list[str]:
    fam = f"langeval-{uuid.uuid4().hex[:8]}"
    host = PlacesHost()
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
        # the asked flavour must reach the picker: in must_match or in the name (the picker treats both as asked)
        if e.get("must") and not any(e["must"] in str(m).lower() for m in [it.get("name"), *(it.get("must_match") or [])]):
            problems.append(f"{it.get('name')}: neither the name nor must_match {it.get('must_match')} has {e['must']}")
        # a limit for the whole ask may be split per unit ("2 litres under 100" → 2 x 1 litre, each up to 50)
        per_unit = e.get("max_price") and e["max_price"] / max(1, int(it.get("qty") or 1))
        if e.get("max_price") and float(it.get("max_price") or 0) not in (e["max_price"], per_unit):
            problems.append(f"{it.get('name')}: max_price {it.get('max_price')} not {e['max_price']}")
        if e.get("cheapest") and not it.get("cheapest"):
            problems.append(f"{it.get('name')}: cheapest not set")
    return problems


async def store_case(lang: str, message: str, want: set[str]) -> list[str]:
    fam = f"langeval-{uuid.uuid4().hex[:8]}"
    async with SessionLocal() as session:
        res = await run_turn(session, PlacesHost(), TurnRequest(family_id=fam, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text=message,
                                                                message_ref=f"m-{uuid.uuid4().hex[:8]}", channel="whatsapp"))
        await session.rollback()
    calls = [a for a in res.actions if a.get("tool") == "start_task"]
    if not calls:
        return [f"no start_task (reply: {str(res.reply)[:120]})"]
    args = calls[-1].get("args") or {}
    got = set(args.get("services") or []) | ({args["service"]} if args.get("service") else set())
    return [] if got == want else [f"stores {sorted(got)}, expected {sorted(want)}"]


async def match_case(item: dict, listings: list[dict], medicine: bool, want_exact: str | None, want_closest: str | None) -> list[str]:
    out = await matcher.choose([{"qty": 1, **item}], [listings], medicine=medicine)
    if out is None:
        return ["no answer"]
    r = out[0]
    first = listings[r["exact"][0]]["name"] if r["exact"] else None
    if want_exact and "|" in want_exact and r["exact"]:  # two listings with one name: told apart by the pack
        first = f"{first}|{listings[r['exact'][0]]['pack']}"
    closest = listings[r["closest"]]["name"] if r.get("closest") is not None else None
    problems = []
    if first != want_exact:
        problems.append(f"exact first {first!r}, expected {want_exact!r} ({r.get('why')})")
    if want_closest and closest != want_closest:
        problems.append(f"closest {closest!r}, expected {want_closest!r}")
    return problems


async def _case(fn, *args) -> list[str]:
    """One case; a model quota error (shared eval project) waits and tries again instead of ending the whole run."""
    from app.llm.router import AllModelsFailed

    for wait in (20, 60, 0):
        try:
            return await fn(*args)
        except AllModelsFailed as exc:
            if not wait:
                return [f"no model answered: {str(exc)[:120]}"]
            await asyncio.sleep(wait)
    return []


async def main() -> int:
    async with engine.begin() as conn:
        from app.care import baselines, memory_index, models, why  # noqa: F401  (why: the trace_turns table)
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
            probs = await _case(brain_case, lang, msg, exp)
            failed += bool(probs)
            print(("PASS" if not probs else "FAIL"), f"brain/{lang}: {msg}", *(f"\n    - {p}" for p in probs))
    if not only or "stores" in only:
        for lang, msg, want in STORE_CASES:
            probs = await _case(store_case, lang, msg, want)
            failed += bool(probs)
            print(("PASS" if not probs else "FAIL"), f"stores/{lang}: {msg}", *(f"\n    - {p}" for p in probs))
    if not only or "match" in only:
        for name, item, ls, med, want, closest in MATCH_CASES:
            probs = await _case(match_case, item, ls, med, want, closest)
            failed += bool(probs)
            print(("PASS" if not probs else "FAIL"), f"match/{name}", *(f"\n    - {p}" for p in probs))
    if not only or "place" in only:
        for words, want in PLACE_CASES:
            probs = await _case(place_case, words, want)
            failed += bool(probs)
            print(("PASS" if not probs else "FAIL"), f"place/{words}", *(f"\n    - {p}" for p in probs))
    if not only or "ride" in only:
        for asked, saved, shown, want in RIDE_END_CASES:
            probs = await _case(ride_end_case, asked, saved, shown, want)
            failed += bool(probs)
            print(("PASS" if not probs else "FAIL"), f"ride/{asked} vs {shown}", *(f"\n    - {p}" for p in probs))
    total = ((len(BRAIN_CASES) if not only or "brain" in only else 0) + (len(MATCH_CASES) if not only or "match" in only else 0)
             + (len(STORE_CASES) if not only or "stores" in only else 0)
             + (len(PLACE_CASES) if not only or "place" in only else 0) + (len(RIDE_END_CASES) if not only or "ride" in only else 0))
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
