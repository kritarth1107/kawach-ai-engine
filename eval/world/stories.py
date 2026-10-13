"""The 100 story families: a past (45 days of WhatsApp history in which facts change), a scripted next week with
expectations, and a 10-question memory quiz at the end of it.

Code plans everything (deterministic, consistent with the family's truth at day 0); the model only writes the words,
in each person's language, script and style (eval/world/write.py rules). Texts are cached in
eval/world/data/text/stories/<family>.json by a hash of the outline.

    DEBUG=false GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 MEMORY_EMBEDDINGS=off \
    MODEL_ROUTES='{"sim": ["gemini:gemini-3.8-flash"]}' WORLD_MAX_INR=2000 PYTHONPATH=. python -m eval.world.stories [w0901 …]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time

from eval.world import grid as G
from eval.world.make import day, hm, load
from eval.world.validate import probe_script_ok
from eval.world.write import SCRIPT_NAMES, STYLE, ask, parse

STORY_DIR = __import__("eval.world.make", fromlist=["DATA"]).DATA / "text" / "stories"
OUTLINE_VERSION = 2
HISTORY_DAYS = 45

LIFE_DETAILS = [
    ("birthday", "{grand} (grandchild) has a birthday on {date}", "When is {grand}'s birthday?", "{date}"),
    ("keys", "the neighbour {neigh} keeps a spare key to the house", "Who has the spare key to the house?", "{neigh}, the neighbour"),
    ("music", "{who} loves old songs by {singer} and listens every evening", "Whose songs does {who} love?", "{singer}"),
    ("trip", "the family plans a trip to {place} next month", "Where is the family planning to go next month?", "{place}"),
    ("food", "{who}'s favourite dish is {dish}", "What is {who}'s favourite dish?", "{dish}"),
    ("pet", "the family has a cat called {pet}", "What is the cat's name?", "{pet}"),
    ("tv", "{who} never misses the serial '{serial}' at 9 pm", "Which TV serial does {who} watch at 9 pm?", "{serial}"),
    ("friend", "{who}'s best friend from school is {friend}, who calls on Sundays", "Who calls {who} on Sundays?", "{friend}"),
]
SINGERS = ["Kishore Kumar", "Mohammed Rafi", "M. S. Subbulakshmi", "Bhupen Hazarika", "Jagjit Singh", "K. J. Yesudas", "Manna Dey", "Ghantasala"]
PLACES = ["Haridwar", "Rameswaram", "Puri", "Shirdi", "Kedarnath", "Goa", "Munnar", "Darjeeling", "Udaipur", "Varanasi"]
DISHES = {"north": ["rajma chawal", "kadhi chawal", "aloo paratha"], "bengal": ["macher jhol", "shukto"], "maharashtra": ["puran poli", "pithla bhakri"],
          "tamil": ["pongal", "kara kuzhambu"], "kerala": ["appam with stew", "puttu kadala"], "gujarat": ["dhokla", "undhiyu"],
          "saurashtra": ["sev tameta", "bajra rotla"], "rajasthan": ["dal baati", "ker sangri"], "purvanchal": ["litti chokha"], "mithila": ["dahi chura"],
          "telugu": ["pesarattu", "gongura pachadi"], "karnataka": ["bisi bele bath", "ragi mudde"], "punjab": ["sarson da saag", "chole bhature"],
          "odisha": ["dalma", "pakhala"], "assam": ["masor tenga"], "urdu": ["nihari", "sheer khurma"], "goa": ["fish curry rice"],
          "tulunadu": ["neer dosa", "kori rotti"], "barak": ["shutki bhorta"], "chhattisgarh": ["chila", "fara"], "garhwal": ["kafuli"], "kumaon": ["bhatt ki churkani"],
          "jammu": ["rajma with rice", "kalari"], "haryana": ["bajra khichdi"], "awadh": ["tehri", "kachori sabzi"], "bundelkhand": ["bafauri"],
          "nepali": ["momo", "gundruk soup"], "metro": ["khichdi", "idli sambar"]}
PETS = ["Billu", "Mishti", "Kaalu", "Snowy", "Chintu"]
SERIALS = ["Ghar Ki Rasoi", "Saath Phere", "Nanhi Kali", "Pyaar Ka Rishta", "Aangan"]


def _med_old(m: dict) -> tuple[str, str]:
    """An earlier dose for a medicine whose dose changed: half the number, or the time an hour earlier."""
    num = re.search(r"\d+(\.\d+)?", m["dose"] or "")
    if num and not m["half"]:
        x = float(num.group(0)) / 2
        return m["dose"].replace(num.group(0), str(int(x)) if x.is_integer() else str(x), 1), "dose"
    return hm(m["times"][0], -60) if m["times"] else "08:00", "time"


def outline(fam: dict) -> dict:
    """Deterministic plan of the story: history lines, the changes over time, next week's events and the quiz."""
    rng = random.Random(f"{G.SEED}:story:{fam['id']}")
    people = {p["id"]: p for p in fam["people"]}
    talkers = [p for p in fam["people"] if p["usesWhatsapp"]]
    teller = next((p for p in fam["people"] if p["role"] == "primary caregiver"), talkers[0])
    subjects = sorted(fam["truth"])  # the same order built or read back from JSON
    sid = subjects[0] if subjects else teller["id"]
    s = people[sid]
    t = fam["truth"].get(sid, {"medicines": [], "conditions": [], "allergies": ["none"], "doctor": None, "callAs": None, "activities": []})
    arc = fam["story"]["arc"]
    meds = [m for m in t["medicines"] if not m["prn"]]
    timeline: list[dict] = []
    lines: list[dict] = []

    def line(d: int, tm: str, who: dict, brief: str, kind: str = "chat") -> None:
        lines.append({"id": f"h{len(lines) + 1:03d}", "day": day(d), "time": tm, "who": who["key"], "brief": brief, "kind": kind})

    # day -45: setup, as the first teller would give it (the arc changes it later)
    arc_med = rng.choice(meds) if meds else None
    # a new diagnosis: the last condition that brought one of their medicines (both are told only on that day)
    diag = None
    for c in reversed(t["conditions"]):
        names = {x[0] for x in G.CONDITIONS.get(c["key"], {}).get("meds", [])}
        mine = next((m for m in meds if m["name"] in names), None)
        if mine and len(t["conditions"]) > 1:
            diag = (c, mine)
            break
    # an arc the record cannot carry becomes one it can (never a silent switch to another family's story)
    if arc == "new_diagnosis" and not diag:
        arc = "dose_change" if arc_med else "new_caregiver"
    if arc == "dose_change" and not arc_med:
        arc = "new_caregiver"
    told_meds = []
    for m in meds:
        if arc == "new_diagnosis" and diag and m is diag[1]:
            continue  # this one starts with the diagnosis
        if arc == "hospital_stay" and m is meds[0] and len(meds) > 1:
            continue  # this one starts at discharge
        if arc == "dose_change" and m is arc_med:
            old, how = _med_old(m)
            told_meds.append(f"{m['name']} {old if how == 'dose' else m['dose']} at {old if how == 'time' else ', '.join(m['times'])}")
        else:
            told_meds.append(f"{m['name']} {m['dose']} at {', '.join(m['times'])}")
    conds = [c["label"] for c in t["conditions"] if not (arc == "new_diagnosis" and diag and c is diag[0])]
    who_s = "themselves" if teller["id"] == sid else s["first"]
    setup = f"{teller['first']} sets up Saheli for {who_s}: conditions {', '.join(conds) or 'none'}; medicines {'; '.join(told_meds) or 'none'}; " \
            f"allergies {', '.join(t['allergies'])}; doctor {t['doctor']['name'] if t['doctor'] else 'not yet'}" + (f"; call them '{t['callAs']}'" if t.get("callAs") else "")
    line(-HISTORY_DAYS, "10:00", teller, setup, "setup")
    timeline.append({"day": day(-HISTORY_DAYS), "what": "setup", "detail": setup})

    # the arc
    if arc == "dose_change" and arc_med:
        old, how = _med_old(arc_med)
        d = -rng.randint(15, 25)
        change = f"to {arc_med['dose']}" if how == "dose" else f"to {', '.join(arc_med['times'])}"
        line(d, "19:30", teller, f"{teller['first']} says the doctor changed {s['first']}'s {arc_med['name']} {change} from tomorrow", "arc")
        timeline.append({"day": day(d), "what": "dose_change", "medicine": arc_med["name"], "from": old, "to": arc_med["dose"] if how == "dose" else arc_med["times"]})
    elif arc == "new_diagnosis" and diag:
        c, m = diag[0]["label"], diag[1]
        d = -rng.randint(20, 32)
        line(d, "18:00", teller, f"{teller['first']} says the tests came back: {s['first']} has {c}; the doctor started {m['name']} {m['dose']} at {', '.join(m['times'])}", "arc")
        timeline.append({"day": day(d), "what": "new_diagnosis", "condition": c, "medicine": m["name"]})
    elif arc == "hospital_stay":
        d = -rng.randint(14, 22)
        why = rng.choice(["a fall in the bathroom", "pneumonia", "low sodium", "chest pain that turned out to be gastric"])
        m = meds[0] if len(meds) > 1 else None
        line(d, "07:40", teller, f"{teller['first']} says {s['first']} was admitted to {t.get('hospital', 'the hospital')} last night after {why}", "arc")
        line(d + 4, "16:00", teller, f"{teller['first']} says {s['first']} is discharged and home" + (f"; new medicine {m['name']} {m['dose']} at {', '.join(m['times'])}" if m else ""), "arc")
        timeline.append({"day": day(d), "what": "hospital_stay", "why": why, "discharged": day(d + 4), "medicine": m["name"] if m else None})
    elif arc == "new_caregiver":
        helper = next((p for p in fam["people"] if p["id"] != teller["id"] and p["id"] != sid), None)
        d = -rng.randint(22, 30)
        name = helper["first"] if helper else rng.choice(G.NAMES[G.LANGS[fam["language"]]["region"] if G.LANGS[fam["language"]]["region"] in G.NAMES else "north"]["f_young"])
        line(d, "21:00", teller, f"{teller['first']} says {name} will also look after {s['first']} from now on and may message Saheli", "arc")
        timeline.append({"day": day(d), "what": "new_caregiver", "name": name})
    elif arc == "move":
        old_city = rng.choice(sorted(c for c in G.CITIES if c != fam["city"]["name"]))
        d = -rng.randint(30, 38)
        line(d, "12:00", teller, f"{teller['first']} says {s['first']} has moved from {old_city} to {fam['city']['name']}; the new address is {fam['places'][0].get('line1', '')}, {fam['city']['name']}", "arc")
        timeline.append({"day": day(d), "what": "move", "from": old_city, "to": fam["city"]["name"]})
    else:  # death_in_family
        rel = rng.choice(["elder sister", "younger brother", "old friend", "cousin"])
        d = -rng.randint(25, 32)
        line(d, "09:00", talkers[0], f"{talkers[0]['first']} shares, sadly, that {s['first']}'s {rel} passed away yesterday", "arc")
        line(d + 3, "20:00", talkers[0], f"{talkers[0]['first']} says {s['first']} is very quiet since the {rel} died and not eating well", "arc")
        timeline.append({"day": day(d), "what": "death_in_family", "relation": rel})

    # life details for the quiz
    region = G.LANGS[fam["language"]]["region"]
    region = region if region in G.NAMES else "north"
    taken = {p["first"] for p in fam["people"]}

    def fresh(pool: list[str]) -> str:
        return rng.choice([n for n in pool if n not in taken] or pool)

    vals = {"grand": fresh(G.NAMES[region]["f_young"] + G.NAMES[region]["m_young"]), "date": f"{rng.randint(1, 28)} {rng.choice(['November', 'December', 'January'])}",
            "neigh": f"{fresh(G.NAMES[region]['f_old'])} aunty", "who": s["first"], "singer": rng.choice(SINGERS), "place": rng.choice(PLACES),
            "dish": rng.choice(DISHES.get(G.LANGS[fam["language"]]["region"], DISHES["metro"])), "pet": rng.choice(PETS), "serial": rng.choice(SERIALS), "friend": fresh(G.NAMES[region]["f_old"] + G.NAMES[region]["m_old"])}
    details = rng.sample(LIFE_DETAILS, 5)
    for k, (key, said, q, a) in enumerate(details):
        d = -HISTORY_DAYS + 3 + k * 7 + rng.randint(0, 3)
        who = rng.choice(talkers)
        line(d, hm(who["sleep"]["wake"], rng.randint(60, 600)), who, f"{who['first']} mentions in passing that {said.format(**vals)}", "detail")
        timeline.append({"day": day(d), "what": "detail", "key": key, "text": said.format(**vals)})

    # ordinary days: doses, readings, small talk, habits, moods
    n_lines = rng.randint(28, 95)
    while len(lines) < n_lines:
        who = rng.choice(talkers)
        d = -rng.randint(1, HISTORY_DAYS - 1)
        tm = hm(who["sleep"]["wake"], rng.randint(30, 13 * 60))
        own = who["id"] in fam["truth"]
        kind = rng.choice(["dose", "dose", "reading", "small_talk", "habit", "mood", "question"])
        tt = fam["truth"].get(who["id"] if own else sid, t)
        mm = [m for m in tt["medicines"] if not m["prn"]]
        if kind == "dose" and mm:
            m = rng.choice(mm)
            brief = f"{who['first']} says {'they' if own else s['first']} took {m['name']} after breakfast"
        elif kind == "reading" and any(r != "none" for r in tt["readings"]):
            brief = f"{who['first']} shares a BP reading {rng.randint(118, 160)}/{rng.randint(72, 98)}"
        elif kind == "habit" and any(a["kind"] == "habit" for a in tt["activities"]):
            a = rng.choice([a for a in tt["activities"] if a["kind"] == "habit"])
            brief = f"{who['first']} says the {a['label'].lower()} {'was done' if rng.random() < 0.7 else 'was skipped today'}"
        elif kind == "mood":
            brief = f"{who['first']} says {rng.choice(['they feel a bit low today', 'it was a happy day, family visited', 'they are tired', 'they slept well'])}"
        elif kind == "question":
            brief = f"{who['first']} asks {rng.choice(['what is due this evening', 'whether a medicine can be taken with tea', 'when the next doctor visit is'])}"
        else:
            brief = f"{who['first']} chats about {rng.choice(['the weather', 'a festival coming up', 'a neighbour', 'the grandchildren', 'cooking'])}"
        line(d, tm, who, brief)
    lines.sort(key=lambda x: (x["day"], x["time"]))
    for i, x in enumerate(lines):
        x["id"] = f"h{i + 1:03d}"

    # next week: scripted life events with expectations (month_sim event shape plus new keys)
    elder_talks = s["usesWhatsapp"]
    actor = s if elder_talks else teller
    week: list[dict] = []

    def ev(d: int, tm: str, who: dict, brief: str, expect: dict) -> None:
        week.append({"id": f"e{len(week) + 1:02d}", "day": day(d), "time": tm, "who": who["key"], "brief": brief, "expect": expect})

    wake = actor["sleep"]["wake"]
    me = actor["id"] == sid
    subj, subj_has = ("they", "they have") if me else (s["first"], f"{s['first']} has")
    if meds:
        m = meds[0]
        ev(0, hm(wake, 90), actor, f"{actor['first']} says {'they' if actor['id'] == sid else s['first']} took {m['name']} this morning", {"tools": ["log_dose"]})
    ev(1, hm(wake, 60), actor, f"{actor['first']} shares a BP reading {rng.randint(120, 150)}/{rng.randint(76, 94)}", {"tools": ["log_vital"], "reading": "bp"})
    tricky = (fam.get("tricky") or ["fever"])[0]
    trick = {
        "fall": (f"{actor['first']} says {subj} slipped in the bathroom, a small bruise on the knee, can walk", {"tools": ["log_event"], "alert": "red_flag" if s["role"] == "elder" and len(fam["people"]) > 1 else "none"}),
        "fever": (f"{actor['first']} says {subj_has} fever 100.8 since afternoon", {"tools": ["log_event", "log_vital"]}),
        "scam_call": (f"{actor['first']} says someone called saying they are from the bank and asked for the OTP to keep the account open", {"tools_none": ["start_task"], "alert": "none"}),
        "missed_dose": (f"{actor['first']} says {subj} forgot the morning tablets", {"tools": ["log_dose"]}),
        "double_dose": (f"{actor['first']} worries {subj} may have taken the morning tablet twice", {"tools": ["log_dose", "log_event"]}),
        "refuses_medicine": (f"{actor['first']} says {'they do not want' if me else s['first'] + ' refuses'} to take the night tablet today", {"tools": ["log_dose", "log_event"]}),
        "family_argument": (f"{actor['first']} vents about an argument at home and feels upset", {"tools_none": ["start_task"], "alert": "none"}),
        "alarming_joke": (f"{actor['first']} jokes that the heat has killed them today (an idiom)", {"tools_none": ["alert_caregiver"], "alert": "none"}),
    }[tricky]
    ev(2, "16:30", actor, trick[0], trick[1])
    ev(3, hm(wake, 120), actor, f"{actor['first']} says from tomorrow there is an evening walk every day at 6:30 pm", {"tools": ["activity"], "activity": "evening_walk"})
    if set(fam["orders"]["uses"]) - {"rides"}:
        ev(4, "11:00", teller, f"{teller['first']} asks Saheli to order a pack of {rng.choice(['oats', 'brown bread', 'fruit', 'curd'])} for {s['first']}", {"tools": ["start_task", "places"], "order": True})
    else:
        ev(4, "11:00", actor, f"{actor['first']} chats about a festival coming up", {"tools_none": ["start_task"]})
    ev(5, "08:00", actor, f"{actor['first']} does not reply to anything today (silence)", {"nudge": {"max_per_day": 3}, "silence": True})
    quiz = []
    if meds:
        m = arc_med or meds[0]
        quiz.append({"q": f"What dose of {m['name']} does {s['first']} take now?", "a": f"{m['dose']} at {', '.join(m['times'])}"})
    quiz.append({"q": f"Is {s['first']} allergic to anything?", "a": ", ".join(t["allergies"])})
    if t.get("doctor"):
        quiz.append({"q": f"Who is {s['first']}'s doctor?", "a": t["doctor"]["name"]})
    quiz.append({"q": f"How many regular medicines does {s['first']} take now?", "a": str(len(meds))})
    arc_t = timeline[1] if len(timeline) > 1 and timeline[1]["what"] not in ("detail",) else None
    if arc_t:
        q = {"dose_change": f"What changed in {s['first']}'s medicines this month?", "new_diagnosis": f"What did the tests show for {s['first']} this month?",
             "hospital_stay": f"Why was {s['first']} in hospital, and when did they come home?", "new_caregiver": "Who else started looking after them this month?",
             "move": f"Where did {s['first']} move from?", "death_in_family": "What sad news did the family share this month?"}[arc_t["what"]]
        quiz.append({"q": q, "a": json.dumps({k: v for k, v in arc_t.items() if k not in ("day", "what")}, ensure_ascii=False)})
    for key, _said, q, a in details:
        quiz.append({"q": q.format(**vals), "a": a.format(**vals)})
    quiz = quiz[:10]
    asker = teller
    ev(6, "20:30", asker, f"{asker['first']} asks Saheli {len(quiz)} questions to check what she remembers (one message per question)",
       {"memory_quiz": [{"id": f"z{i + 1:02d}", **x} for i, x in enumerate(quiz)]})
    return {"version": OUTLINE_VERSION, "arc": arc, "history": lines, "timeline": timeline, "nextWeek": week,
            "quiz": [{"id": f"z{i + 1:02d}", "by": asker["key"], **x} for i, x in enumerate(quiz)]}


STORY_SYSTEM = """You write WhatsApp messages for an invented Indian family talking to Saheli, a care companion. For each
item, write the person's message from the brief in THEIR language, dialect and script (rules below), and Saheli's short
reply (one or two sentences, warm and direct, in the same language and script as the person's message; never promise to
order or book anything; no medical advice beyond what a caring companion would say).

Rules: dialects must read like the dialect, not the standard language; Roman-script languages use Roman letters only;
English (Indian) is natural Indian English. Follow each person's chat style. Keep names, medicines, doses, numbers,
dates and addresses exactly. In items of kind "setup" or "arc", write every medicine name in Latin letters exactly as in
the brief and leave none out. Do not add facts. Never write phone numbers or real public people's names (a singer named
in the brief is fine).

Return JSON only: {"items": {"<id>": {"text": "<person's message>", "saheli": "<Saheli's reply or empty>"}}}"""


def people_for_writer(fam: dict) -> list[dict]:
    out = []
    for p in fam["people"]:
        lang = G.LANGS[p["language"]]
        out.append({"key": p["key"], "first_name": p["first"], "age": p["age"], "relation": p["relation"], "language": lang["name"],
                    "script": SCRIPT_NAMES["latin" if lang["roman"] else lang["script"]], "style": STYLE[p["style"]], "persona": p.get("persona")})
    return out


def outline_hash(fam: dict, ol: dict) -> str:
    """What the texts depend on: the outline and the people (persona lines excluded, so a rewritten persona keeps the story)."""
    people = [{k: v for k, v in p.items() if k != "persona"} for p in people_for_writer(fam)]
    return hashlib.sha256(json.dumps({"v": OUTLINE_VERSION, "p": people, "o": ol}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def item_problems(fam: dict, items: list[dict], got: dict, need_saheli: bool) -> list[str]:
    errs = []
    keys = {p["key"]: p for p in fam["people"]}
    for it in items:
        row = got.get(it["id"]) or {}
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            errs.append(f"{it['id']}: missing")
            continue
        lang = G.LANGS[keys[it.get("who") or it.get("by")]["language"]]
        want = "latin" if lang["roman"] or lang["lang"] == "en" else lang["script"]
        if not probe_script_ok(text, want, it.get("brief") or it.get("q", "")):
            errs.append(f"{it['id']}: must be in {SCRIPT_NAMES[want]} script")
        if it.get("kind") in ("setup", "arc"):
            for m in {m["name"] for t in fam["truth"].values() for m in t["medicines"]}:
                if m.lower() in (it.get("brief") or "").lower() and m.split()[0].lower() not in text.lower():
                    errs.append(f"{it['id']}: write the medicine {m} in Latin letters as in the brief")
        if need_saheli and not (row.get("saheli") or "").strip():
            errs.append(f"{it['id']}: Saheli's reply missing")
        if re.search(r"\+?\d[\d -]{9,}\d", text) and not re.search(r"9190000|\+99", text):
            errs.append(f"{it['id']}: phone-like number")
    return errs


async def write_items(fam: dict, items: list[dict], need_saheli: bool) -> dict:
    payload = {"people": people_for_writer(fam), "items": [{k: v for k, v in it.items() if k in ("id", "day", "time", "who", "by", "brief", "q", "kind")} for it in items]}
    user = json.dumps(payload, ensure_ascii=False, indent=1)
    got = (parse(await ask(STORY_SYSTEM, user, tokens=12000)).get("items") or {})
    errs = item_problems(fam, items, got, need_saheli)
    if errs:
        fix = f"{user}\n\nFix these problems and return the whole JSON again:\n- " + "\n- ".join(errs[:25])
        got2 = (parse(await ask(STORY_SYSTEM, fix, tokens=12000)).get("items") or {})
        for k, v in got2.items():
            got[k] = v
        errs = item_problems(fam, items, got, need_saheli)
    if errs:
        raise ValueError(f"{fam['id']}: {len(errs)} problems: {errs[:3]}")
    return got


async def write_story(fam: dict) -> dict:
    ol = outline(fam)
    texts: dict[str, dict] = {}
    hist = ol["history"]
    for i in range(0, len(hist), 30):
        texts.update(await write_items(fam, hist[i:i + 30], need_saheli=True))
    week = [e for e in ol["nextWeek"] if not e["expect"].get("silence") and not e["expect"].get("memory_quiz")]
    texts.update(await write_items(fam, week, need_saheli=False))
    quiz = [{"id": z["id"], "by": z["by"], "q": z["q"], "brief": f"asks: {z['q']}"} for z in ol["quiz"]]
    texts.update(await write_items(fam, quiz, need_saheli=False))
    return {"family": fam["id"], "hash": outline_hash(fam, ol), "texts": {k: {"text": v["text"], "saheli": v.get("saheli") or ""} for k, v in texts.items()}}


def attach_stories(families: list[dict]) -> int:
    """Fill each story family's history, next week and quiz (outline + cached texts where the cache matches)."""
    n = 0
    for fam in families:
        if fam["tier"] != "story":
            continue
        ol = outline(fam)
        p = STORY_DIR / f"{fam['id']}.json"
        texts = {}
        if p.exists():
            row = json.loads(p.read_text())
            if row["hash"] == outline_hash(fam, ol):
                texts = row["texts"]
                n += 1
        for x in (*ol["history"], *ol["nextWeek"], *ol["quiz"]):
            x["text"] = (texts.get(x["id"]) or {}).get("text")
            if x in ol["history"]:
                x["saheli"] = (texts.get(x["id"]) or {}).get("saheli")
        fam["story"].update({"history": ol["history"], "timeline": ol["timeline"], "nextWeek": ol["nextWeek"], "quiz": ol["quiz"]})
    return n


async def run(ids: list[str], parallel: int) -> int:
    from app.llm import spend

    cap = float(os.getenv("WORLD_MAX_INR", "0") or 0)
    if cap <= 0:
        print("Refusing to start: set WORLD_MAX_INR.")
        return 2
    from eval.world.write import attach

    fams = [f for f in load(ids or None) if f["tier"] == "story"]
    attach(fams)  # personas help the story writer
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for f in fams:
        p = STORY_DIR / f"{f['id']}.json"
        if not (p.exists() and json.loads(p.read_text())["hash"] == outline_hash(f, outline(f))):
            todo.append(f)
    print(f"{len(fams)} story families, {len(fams) - len(todo)} cached, {len(todo)} to write (cap ₹{cap:.0f})")
    sem = asyncio.Semaphore(parallel)
    ok = bad = 0
    t0 = time.monotonic()

    async def one(f: dict) -> None:
        nonlocal ok, bad
        async with sem:
            if spend.process_spent() >= cap:
                return
            try:
                row = await write_story(f)
                (STORY_DIR / f"{f['id']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  {exc}"[:300])
                bad += 1
            if (ok + bad) % 10 == 0:
                print(f"  {ok + bad}/{len(todo)} ({bad} failed), ₹{spend.process_spent():.0f}, {time.monotonic() - t0:.0f}s")

    await asyncio.gather(*(one(f) for f in todo))
    print(f"done: {ok} written, {bad} failed; spent ₹{spend.process_spent():.0f} in {time.monotonic() - t0:.0f}s")
    return 0 if not bad else 1


def main(argv: list[str]) -> int:
    par = 8
    if "--parallel" in argv:
        i = argv.index("--parallel")
        par = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if os.getenv("GCP_PROJECT_ID") == "kavach-care":
        print("Refusing: paid world runs go to the sunny-ship test project, never kavach-care.")
        return 2
    return asyncio.run(run(argv, par))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
