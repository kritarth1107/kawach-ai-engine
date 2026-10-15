"""200-fact multilingual recall bench.

Each case is one current fact (medicines, doses, allergies, names, preferences, dated events, and corrections).
A correction also stores the replaced value. recall@1 / recall@3 ask whether the current value is the first hit or
in the top 3. wrong_version counts a replaced value that still comes back.

    GCP_PROJECT_ID=kavach-eval-1015 DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5433/kawach_recall_bench \\
        PYTHONPATH=. python -m eval.recall_200

The unit test covers the same scorer with a fake embedder. The Vertex run is the 90% bar.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

# (lang, domain, key, current text, query, needle, old text or "")
# The query does not contain the needle. The old text, when set, must not come back.
_ROWS: list[tuple] = []


def _add(lang, domain, key, text, query, needle, old=""):
    _ROWS.append((lang, domain, key, text, query, needle, old))


def _build() -> None:
    if len(_ROWS) == 200:
        return
    _ROWS.clear()
    meds = [
        ("metformin", "Metformin", "sugar tablet", "शुगर की गोली", "sugar ki goli", "সুগারের বড়ি", "சர்க்கரை மாத்திரை", "sugar tablet शुगर"),
        ("amlodipine", "Amlodipine", "blood pressure tablet", "बीपी की गोली", "BP ki goli", "বিপির বড়ি", "பிபி மாத்திரை", "BP tablet बीपी"),
        ("atorvastatin", "Atorvastatin", "cholesterol tablet", "कोलेस्ट्रॉल की गोली", "cholesterol ki goli", "কোলেস্টেরলের বড়ি", "கொழுப்பு மாத்திரை", "cholesterol गोली"),
        ("thyroxine", "Thyroxine", "thyroid tablet", "थायराइड की गोली", "thyroid ki goli", "থাইরয়েডের বড়ি", "தைராய்டு மாத்திரை", "thyroid गोली"),
        ("telmisartan", "Telmisartan", "evening pressure tablet", "शाम की बीपी गोली", "shaam ki BP goli", "সন্ধ্যার বিপির বড়ি", "மாலை பிபி மாத்திரை", "evening BP गोली"),
        ("glimepiride", "Glimepiride", "breakfast sugar tablet", "नाश्ते की शुगर गोली", "nashte ki sugar goli", "নাশতার সুগার বড়ি", "காலை சர்க்கரை மாத்திரை", "breakfast sugar गोली"),
        ("pantoprazole", "Pantoprazole", "acidity tablet", "एसिडिटी की गोली", "acidity ki goli", "অ্যাসিডিটির বড়ি", "அமில மாத்திரை", "acidity गोली"),
        ("vitamin_d", "Vitamin D", "vitamin D", "विटामिन डी", "vitamin D", "ভিটামিন ডি", "விட்டமின் டி", "vitamin D विटामिन"),
    ]
    doses = ["2.5 mg", "5 mg", "10 mg", "20 mg", "25 mg", "40 mg", "50 mg", "75 mg", "100 mg", "500 mg", "1000 mg"]
    times = ["06:30", "07:00", "08:00", "09:00", "13:00", "14:00", "20:00", "21:00"]
    qmed = {
        "en": "what dose is the {topic}",
        "hi": "{topic} की खुराक क्या है",
        "hinglish": "{topic} ki dose kya hai",
        "bn": "{topic} এর ডোজ কত",
        "ta": "{topic} எவ்வளவு",
        "mixed": "{topic} dose कितनी है",
    }
    langs = ("en", "hi", "hinglish", "bn", "ta", "mixed")
    topic_i = {"en": 2, "hi": 3, "hinglish": 4, "bn": 5, "ta": 6, "mixed": 7}
    plain_i = {"en": 1, "hi": 2, "hinglish": 3, "bn": 4, "ta": 5, "mixed": 6}
    n = 0
    for li, lang in enumerate(langs):
        for mi, med in enumerate(meds):
            dose = doses[(li + mi) % len(doses)]
            when = times[(li + mi) % len(times)]
            old = ""
            if mi < 2:
                prev = doses[(li + mi + 3) % len(doses)]
                old = f"{med[1]} {prev} at {when}"
            text = f"{med[1]} {dose} at {when}. {med[topic_i[lang]]}"
            _add(lang, "medicine", f"medicine:{med[0]}", text, qmed[lang].format(topic=med[topic_i[lang]]), f"{med[1]} {dose}", old)
            n += 1
    allergens = [
        ("sulfa", "sulfa drugs", "सल्फा दवा", "sulfa dawai", "সালফা ওষুধ", "சல்ஃபா மருந்து", "sulfa दवा"),
        ("penicillin", "penicillin", "पेनिसिलिन", "penicillin", "পেনিসিলিন", "பெனிசிலின்", "penicillin पेनिसिलिन"),
        ("peanut", "peanuts", "मूंगफली", "moongphali", "চিনাবাদাম", "வேர்க்கடலை", "peanut मूंगफली"),
        ("lactose", "milk", "दूध", "doodh", "দুধ", "பால்", "milk दूध"),
    ]
    qal = {
        "en": "what are they allergic to, about {topic}",
        "hi": "{topic} से एलर्जी है क्या",
        "hinglish": "{topic} se allergy hai kya",
        "bn": "{topic} অ্যালার্জি আছে কি",
        "ta": "{topic} ஒவ்வாமை உள்ளதா",
        "mixed": "{topic} allergy है क्या",
    }
    for li, lang in enumerate(langs):
        for ai, al in enumerate(allergens):
            name = al[plain_i[lang]]
            _add(lang, "allergy", f"allergy:{al[0]}", f"Allergic to {name}; reaction {al[0]}-rash",
                 qal[lang].format(topic=name), f"{al[0]}-rash")
    people = [
        ("sister", "Leela", "बहन लीला", "behan Leela", "বোন লীলা", "சகோதரி லீலா", "sister लीला"),
        ("son", "Arun", "बेटा अरुण", "beta Arun", "ছেলে অরুণ", "மகன் அருண்", "son अरुण"),
        ("doctor", "Dr Sen", "डॉक्टर सेन", "doctor Sen", "ডাক্তার সেন", "டாக்டர் சென்", "doctor सेन"),
        ("neighbour", "Rafi", "पड़ोसी रफी", "padosi Rafi", "প্রতিবেশী রাফি", "அண்டை வீட்டு ரஃபி", "neighbour रफी"),
    ]
    qname = {
        "en": "who is the {role}",
        "hi": "{role} कौन है",
        "hinglish": "{role} kaun hai",
        "bn": "{role} কে",
        "ta": "{role} யார்",
        "mixed": "{role} कौन है",
    }
    roles = {"en": "sister son doctor neighbour".split(), "hi": "बहन बेटा डॉक्टर पड़ोसी".split(),
             "hinglish": ["behan", "beta", "doctor", "padosi"], "bn": "বোন ছেলে ডাক্তার প্রতিবেশী".split(),
             "ta": "சகோதரி மகன் டாக்டர் அண்டை".split(), "mixed": ["sister", "beta", "doctor", "पड़ोसी"]}
    for lang in langs:
        for i, person in enumerate(people):
            shown = person[plain_i[lang]]
            _add(lang, "naming", f"naming:{person[0]}", f"The {roles[lang][i]} is {shown}",
                 qname[lang].format(role=roles[lang][i]), shown.split()[-1])
    prefs = [
        ("tea", "ginger tea without sugar", "बिना चीनी की अदरक चाय", "adrak wali chai bina cheeni", "চিনি ছাড়া আদা চা", "சர்க்கரை இல்லாத இஞ்சி தேநீர்", "adrak chai बिना चीनी"),
        ("walk", "a walk at 06:30", "साढ़े छह बजे टहलना", "sadhe 6 baje walk", "সাড়ে ছটায় হাঁটা", "ஆறுரை மணி நடை", "6:30 walk टहलना"),
        ("music", "morning bhajans", "सुबह के भजन", "subah ke bhajan", "সকালের ভজন", "காலை பஜன்", "morning bhajan भजन"),
        ("food", "moong dal khichdi", "मूंग दाल की खिचड़ी", "moong dal khichdi", "মুগ ডালের খিচুড়ি", "பாசிப்பருப்பு கிச்சடி", "moong khichdi खिचड़ी"),
        ("sleep", "lights out at 22:00", "रात दस बजे सोना", "raat 10 baje sona", "রাত দশটায় ঘুম", "இரவு பத்து மணி தூக்கம்", "10 baje sona सोना"),
        ("call", "a call from Arun on Sunday", "रविवार को अरुण का फोन", "Sunday ko Arun ka phone", "রবিবার অরুণের ফোন", "ஞாயிறு அருண் அழைப்பு", "Sunday Arun फोन"),
    ]
    qpref = {
        "en": "what do they like for {slot}",
        "hi": "{slot} में क्या पसंद है",
        "hinglish": "{slot} mein kya pasand hai",
        "bn": "{slot} এ কী পছন্দ",
        "ta": "{slot} என்ன பிடிக்கும்",
        "mixed": "{slot} mein क्या पसंद",
    }
    slots = {"en": "tea walk music lunch sleep sunday".split(), "hi": "चाय टहलना भजन दोपहर नींद रविवार".split(),
             "hinglish": "chai walk bhajan lunch neend Sunday".split(), "bn": "চা হাঁটা ভজন দুপুর ঘুম রবিবার".split(),
             "ta": "தேநீர் நடை பஜன் மதியம் தூக்கம் ஞாயிறு".split(), "mixed": "chai walk bhajan lunch sleep रविवार".split()}
    for lang in langs:
        for i, pref in enumerate(prefs):
            shown = pref[plain_i[lang]]
            _add(lang, "preference", f"preference:{pref[0]}", shown, qpref[lang].format(slot=slots[lang][i]), shown)
    events = [
        ("2026-09-02", "eye check", "आँख की जाँच", "aankh ki jaanch", "চোখ পরীক্ষা", "கண் பரிசோதனை", "eye check आँख"),
        ("2026-09-14", "blood test", "खून की जाँच", "khoon ki jaanch", "রক্ত পরীক্ষা", "இரத்த பரிசோதனை", "blood test खून"),
        ("2026-08-21", "fell in the bathroom", "बाथरूम में गिरे", "bathroom mein gire", "বাথরুমে পড়েছিলেন", "குளியலறையில் விழுந்தார்", "bathroom गिरे"),
        ("2026-07-11", "started the morning walk", "सुबह की सैर शुरू", "subah ki sair shuru", "সকালের হাঁটা শুরু", "காலை நடை தொடக்கம்", "morning walk सैर"),
        ("2026-06-03", "new spectacles", "नया चश्मा", "naya chashma", "নতুন চশমা", "புதிய கண்ணாடி", "new chashma चश्मा"),
        ("2026-05-19", "dental visit", "दांतों की जाँच", "daant ki jaanch", "দাঁত পরীক্ষা", "பல் பரிசோதனை", "dental दांत"),
        ("2026-04-08", "flu vaccine", "फ्लू का टीका", "flu ka teeka", "ফ্লুর টিকা", "காய்ச்சல் தடுப்பூசி", "flu teeka टीका"),
        ("2026-03-27", "knee physiotherapy", "घुटना की फिजियो", "ghutna ki physio", "হাঁটুর ফিজিও", "முட்டி பிசியோ", "knee physio घुटना"),
    ]
    qev = {
        "en": "when was the {topic}",
        "hi": "{topic} कब हुई",
        "hinglish": "{topic} kab hui",
        "bn": "{topic} কবে হয়েছিল",
        "ta": "{topic} எப்போது",
        "mixed": "{topic} kab था",
    }
    for lang in langs:
        for ev in events:
            shown = ev[plain_i[lang]]
            _add(lang, "event", f"event:{ev[0]}", f"{ev[0]}: {shown}", qev[lang].format(topic=shown), ev[0])
    # Two more medicines per language so the bench is 200 current facts (6*8 + 6*4 + 6*4 + 6*6 + 6*8 = 180; +20).
    extra = [("ecosprin", "Ecosprin", "75 mg"), ("losartan", "Losartan", "25 mg"), ("aspirin", "Aspirin", "75 mg"),
             ("metoprolol", "Metoprolol", "25 mg")]
    for li, lang in enumerate(langs):
        for j in range(3):
            drug = extra[(li + j) % len(extra)]
            text = f"{drug[1]} {drug[2]} at night"
            _add(lang, "medicine", f"medicine:{drug[0]}_{j}", text, f"{drug[1]} at night?", f"{drug[1]} {drug[2]}")
    # 6*3 = 18, total 198. Two dated events in English and Hindi.
    _add("en", "event", "event:2026-02-02", "2026-02-02: joined the park yoga group", "when did the park yoga start", "2026-02-02")
    _add("hi", "event", "event:2026-01-15", "2026-01-15: पार्क योग शुरू", "पार्क योग कब शुरू", "2026-01-15")


def bench_cases() -> list[dict]:
    _build()
    out = []
    for lang, domain, key, text, query, needle, old in _ROWS:
        out.append({"lang": lang, "domain": domain, "key": key, "text": text, "query": query, "needle": needle,
                    "old": old, "subject": f"bench-{lang}"})
    if len(out) != 200:
        raise RuntimeError(f"bench has {len(out)} cases, want 200")
    return out


def score(cases: list[dict], hits_for) -> dict:
    """hits_for(case) -> list of hit texts, best first."""
    r1 = r3 = wrong = 0
    misses = []
    r1_miss = []
    for c in cases:
        texts = hits_for(c)
        top = texts[0] if texts else ""
        top3 = texts[:3]
        if c["needle"] in top:
            r1 += 1
        else:
            r1_miss.append(c["domain"] + "/" + c["lang"])
        if any(c["needle"] in t for t in top3):
            r3 += 1
        else:
            misses.append(c["key"] + "/" + c["lang"])
        if c["old"] and any(c["old"] in t for t in top3):
            wrong += 1
    n = len(cases)
    return {"n": n, "recall_at_1": r1 / n, "recall_at_3": r3 / n, "wrong_version": wrong,
            "misses": misses, "r1_miss": r1_miss}


async def run_vertex(url: str) -> dict:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.care import memory_index, store
    from app.core import clock
    from eval.world.seed_engine import prepare

    os.environ.setdefault("MEMORY_EMBEDDINGS", "on")
    engine = create_async_engine(url)
    await prepare(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    cases = bench_cases()
    fam = "bench-recall-200"
    search_only = os.environ.get("RECALL_SEARCH_ONLY") == "1"
    async with sessions() as s:
        if not search_only:
            for c in cases:
                domain = "occasion" if c["domain"] == "event" else c["domain"]
                if c["old"]:
                    clock.set_now(datetime(2026, 10, 1, 9, 0, tzinfo=clock.IST))
                    await store.write_fact(s, family_id=fam, subject_id=c["subject"], domain=domain, key=c["key"],
                                           value={"text": c["old"]}, text=c["old"], source_kind="caregiver_said")
                clock.set_now(datetime(2026, 10, 10, 9, 0, tzinfo=clock.IST))
                await store.write_fact(s, family_id=fam, subject_id=c["subject"], domain=domain, key=c["key"],
                                       value={"text": c["text"]}, text=c["text"], source_kind="caregiver_said", replace=True)
                if c["domain"] == "event":
                    await store.record_event(s, family_id=fam, subject_id=c["subject"], kind="visit", summary=c["text"])
            for subject in {c["subject"] for c in cases}:
                await memory_index.index_subject(s, fam, subject)
            await s.commit()
        from sqlalchemy import func, select

        filled = (await s.execute(select(func.count()).select_from(memory_index.MemoryVector).where(
            memory_index.MemoryVector.family_id == fam, memory_index.MemoryVector.embedding.is_not(None)))).scalar_one()
        total = (await s.execute(select(func.count()).select_from(memory_index.MemoryVector).where(
            memory_index.MemoryVector.family_id == fam))).scalar_one()
        print(f"indexed {filled}/{total} with embeddings", flush=True)

        def hits_for(c):
            return _last[c["key"] + c["subject"]]

        _last: dict[str, list[str]] = {}
        for c in cases:
            found = await memory_index.search(s, fam, [c["subject"]], c["query"], limit=3)
            _last[c["key"] + c["subject"]] = [h.text for h in found]
        result = score(cases, hits_for)
        await s.commit()
    await engine.dispose()
    return result


def main() -> int:
    url = os.environ["DATABASE_URL"]
    if "kavach-care" in os.environ.get("GCP_PROJECT_ID", ""):
        raise SystemExit("refusing kavach-care")
    result = asyncio.run(run_vertex(url))
    print(f"n={result['n']} recall@1={result['recall_at_1']:.1%} recall@3={result['recall_at_3']:.1%} "
          f"wrong_version={result['wrong_version']} top3_misses={result['misses']} r1_miss={result['r1_miss']}")
    return 0 if result["recall_at_1"] >= 0.90 and result["wrong_version"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
