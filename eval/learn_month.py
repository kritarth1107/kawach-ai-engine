"""Learning and memory scorecard for the 30-day simulation (SIM_LEARN_SCORE=1 with eval/month_sim.py).

On top of the month's ordinary life, each family gets:
  - a true daily routine for the person cared for (or the self-care user): meal and water times with a little noise
    (working people shift at weekends), what they ate, and whether they tell Saheli; the simulated person knows
    what they ate today, so they can answer a meal question
  - a busy window when they never answer messages Saheli starts (prayer, a nap, physio, meetings), and an evening
    window when they nearly always do
  - Saheli's real regular nudges (meal, water, sleep, habit asks) every 15 minutes of the day, her hourly meal and
    rhythm learners, and the night's reply scoring before the dream (so the outcome learner sees real reply rates)
  - personal facts told in passing in the first week (favourite dish, a food they refuse, a relative and her city,
    an evening habit, an appointment), a correction of one of them, and a food log corrected the same morning
  - checkpoint quizzes on days 7, 14, 21 and 30 (a caregiver asks; a self-care user asks about herself)
Measured: recall by kind and checkpoint, corrections taken, routine and timing learnings (accuracy, day learned, nudges
still sent in the busy window), food log coverage, memory hygiene (duplicates, stale facts, old values repeated), and
the weekly judged pass rate.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select

from app.core import clock

ON = os.getenv("SIM_LEARN_SCORE") == "1"
CHECKPOINTS = tuple(int(x) for x in os.getenv("SIM_LEARN_CHECKPOINTS", "7,14,21,30").split(","))  # a short run can quiz earlier
NUDGE_EVERY = int(os.getenv("SIM_NUDGE_EVERY_MIN", "15"))
# events that must not run in a shared test project (no crisis, self-harm, overdose, drug, alcohol or tobacco probes)
UNSAFE = re.compile(r"no point|kill|suicid|self.?harm|\bdie\b|overdose|twice tonight|2 sleeping pills|double of|alcohol|drunk|whisky|"
                    r"cigar|smok|tobacco|gutka|bidi", re.I)


def safe_events(events: list[tuple]) -> list[tuple]:
    return [e for e in events if not UNSAFE.search(str(e[3]))]


# ── each family's hidden life ──────────────────────────────────────────────────────────────────────────────────────

# meals: slot -> "HH:MM" (weekdays); weekend: minutes added at weekends; busy/good: hours they ignore / answer
LIFE = {
    "sharma": {
        "who": "sh-kamla", "asker": "sh-ankit", "log_p": 0.55, "water": ["07:00", "16:00"],
        "meals": {"breakfast": "08:00", "lunch": "13:15", "dinner": "19:45"}, "weekend": 0, "busy": (10, 12), "good": (18, 20),
        "busy_why": "you are at the temple and doing puja every morning from 10 to 12 and never look at the phone then",
        "dishes": {"breakfast": ["besan chilla", "dalia", "poha"], "lunch": ["moong dal khichdi", "lauki sabzi with two rotis"],
                   "dinner": ["two rotis with lauki sabzi", "moong dal with rice"]},
        "fav": "kadhi chawal", "dislike": "karela", "rel": ("younger sister", "Usha", "Jaipur", "Pune"),
        "hobby": "listening to old Vividh Bharati songs on the radio at 7 pm", "appt": ("eye check-up", "Shroff Eye Hospital", 20, "11:00"),
        "tok": {"fav": "kadhi", "dislike": "karela", "hobby": "vividh", "appt": "shroff"},
        "fix": ("poha", "dalia"),
    },
    "banerjee": {
        "who": "bn-subrata", "asker": "bn-rina", "log_p": 0.35, "water": ["08:00", "17:00"],
        "meals": {"breakfast": "09:00", "lunch": "13:30", "dinner": "20:30"}, "weekend": 0, "busy": (14, 16), "good": (18, 20),
        "busy_why": "you sleep after lunch from 2 to 4 every afternoon and never see the phone then",
        "dishes": {"breakfast": ["toast and an egg", "muri with tea", "chire"], "lunch": ["rice, dal and fish curry", "rice with shukto"],
                   "dinner": ["two rotis with alur dom", "rice and moong dal"]},
        "fav": "rui macher jhol", "dislike": "lau ghonto", "rel": ("brother", "Tapan", "Shillong", "Siliguri"),
        "hobby": "listening to Rabindrasangeet at 7 pm", "appt": ("hearing test", "AMRI Dhakuria", 20, "10:30"),
        "tok": {"fav": "rui mach", "dislike": "ghonto", "hobby": "rabindra", "appt": "amri"},
        "fix": ("muri", "chire"),
    },
    "reddy": {
        "who": "re-lakshmi", "asker": "re-kiran", "log_p": 0.55, "water": ["07:30", "15:30"],
        "meals": {"breakfast": "08:15", "lunch": "12:45", "dinner": "19:30"}, "weekend": 0, "busy": (16, 18), "good": (20, 22),
        "busy_why": "you do physio and then rest from 4 to 6 every evening and never look at the phone then",
        "dishes": {"breakfast": ["idli with chutney", "upma", "pesarattu"], "lunch": ["rice, sambar and beans curry", "curd rice with pickle"],
                   "dinner": ["two chapatis with dal", "ragi mudde with sambar"]},
        "fav": "pesarattu with allam chutney", "dislike": "bitter gourd", "rel": ("cousin", "Saroja", "Vijayawada", "Visakhapatnam"),
        "hobby": "watching the Telugu news at 7 pm", "appt": ("dental check", "Care Dental, Banjara Hills", 20, "12:00"),
        "tok": {"fav": "pesarattu", "dislike": "bitter", "hobby": "telugu news", "appt": "dental"},
        "fix": ("upma", "pesarattu"),
    },
    "gupta": {
        "who": "gu-savitri", "asker": "gu-neha", "log_p": 0.6, "water": ["06:30", "16:30"],
        "meals": {"breakfast": "08:30", "lunch": "13:00", "dinner": "20:00"}, "weekend": 0, "busy": (12, 14), "good": (18, 20),
        "busy_why": "from 12 to 2 you cook, eat and read the Ramayan and never look at the phone then",
        "dishes": {"breakfast": ["poha", "daliya", "besan chilla"], "lunch": ["dal, roti and lauki", "rajma with rice"],
                   "dinner": ["moong dal khichdi", "two rotis with tori sabzi"]},
        "fav": "dal bafla", "dislike": "baingan", "rel": ("sister", "Kusum", "Ujjain", "Bhopal"),
        "hobby": "watching the Ramayan serial at 7 pm", "appt": ("eye check-up", "Choithram Netralaya", 20, "11:30"),
        "tok": {"fav": "bafla", "dislike": "baingan", "hobby": "ramayan", "appt": "choithram"},
        "fix": ("poha", "daliya"),
    },
    "menon": {
        "who": "me-priya", "asker": "me-priya", "log_p": 0.5, "water": ["09:00", "15:00"],
        "meals": {"breakfast": "07:30", "lunch": "13:30", "dinner": "21:00"}, "weekend": 90, "busy": (10, 12), "good": (20, 22),
        "busy_why": "weekday mornings from 10 to 12 you are in back-to-back work meetings and never answer the phone then",
        "dishes": {"breakfast": ["puttu and kadala curry", "oats with banana", "appam with stew"], "lunch": ["office canteen meals: rice, sambar, thoran", "chapati with egg curry"],
                   "dinner": ["kanji with payar", "dosa with chutney"]},
        "fav": "appam with vegetable stew", "dislike": "okra", "rel": ("brother", "Arun", "Dubai", "Muscat"),
        "hobby": "a 7 pm walk by the backwaters", "appt": ("thyroid blood test", "Aster Medcity lab", 20, "08:30"),
        "tok": {"fav": "appam", "dislike": "okra", "hobby": "backwater", "appt": "aster"},
        "fix": ("oats with banana", "puttu and kadala curry"),
    },
    "iyer": {
        "who": "iy-raghavan", "asker": "iy-meena", "log_p": 0.6, "water": ["06:30", "15:00"],
        "meals": {"breakfast": "07:45", "lunch": "12:30", "dinner": "19:30"}, "weekend": 0, "busy": (16, 18), "good": (20, 22),
        "busy_why": "from 4 to 6 every evening you walk to the temple and sit at the Ramakrishna Mission reading room, phone left at home",
        "dishes": {"breakfast": ["two idlis with sambar", "pongal", "rava upma"], "lunch": ["rice, rasam and beans poriyal", "curd rice with vadu mangai"],
                   "dinner": ["two dosas", "rice with mor kuzhambu"]},
        "fav": "vathal kuzhambu with rice", "dislike": "cauliflower", "rel": ("sister", "Kamakshi", "Madurai", "Coimbatore"),
        "hobby": "doing The Hindu crossword at 7 pm", "appt": ("cardiology review", "Apollo Greams Road", 20, "10:00"),
        "tok": {"fav": "vathal", "dislike": "cauliflower", "hobby": "crossword", "appt": "cardiology"},
        "fix": ("pongal", "rava upma"),
    },
}
CORRECT_DAY = 16  # the relative moved
FIX_DAY = 11  # the breakfast log corrected


def meal_ok(err: int | None) -> bool:
    """A learned meal time is right when it is at most 20 minutes early or 45 late: Saheli asks after the meal, and people
    tell her after eating, so a learned time a little after the true one is the design, not an error."""
    return err is not None and -20 <= err <= 45


def hm(t: str) -> int:
    h, m = (int(x) for x in t.split(":"))
    return h * 60 + m


def clock_str(mins: int) -> str:
    mins = max(0, min(23 * 60 + 59, int(mins)))
    return f"{mins // 60:02d}:{mins % 60:02d}"


@dataclass
class Learn:
    life: dict
    who: str
    asker: str
    today: dict = field(default_factory=dict)  # slot -> {"at": "HH:MM", "dish": str, "told": bool}
    days: dict = field(default_factory=dict)  # day -> today's meals (truth for the food log check)
    quiz: list[dict] = field(default_factory=list)
    reads: list[dict] = field(default_factory=list)
    learned_on: dict = field(default_factory=dict)  # what -> first day it was right
    nudges: list[dict] = field(default_factory=list)  # regular nudges to the person: day, hour, kind
    stats: Counter = field(default_factory=Counter)
    stale_mentions: list[str] = field(default_factory=list)


def setup(f) -> Learn | None:
    life = LIFE.get(f.spec["key"])
    if not life:
        return None
    f.host.world.nudge_mode = "on"
    f.host.world.meals_mode = "on"
    return Learn(life=life, who=life["who"], asker=life["asker"])


START_WEEKDAY = datetime(2026, 11, 4, 12, 0, tzinfo=clock.IST)  # a Wednesday (only the weekday matters)
START_WEEKEND = datetime(2026, 11, 7, 12, 0, tzinfo=clock.IST)  # a Saturday


def is_weekend(day: int) -> bool:
    from eval.month_sim import START

    return (START + timedelta(days=day - 1)).weekday() >= 5


def _name(f, pid: str) -> str:
    return f.people[pid]["name"].split()[0]


# ── the day ───────────────────────────────────────────────────────────────────────────────────────────────────────


def plan_day(f, day: int) -> None:
    """Today's true meals and water, which of them the person tells Saheli about, the probes, nudge ticks, learners."""
    from eval.month_sim import day_dt

    L: Learn = f.learn
    life = L.life
    shift = life["weekend"] if is_weekend(day) else 0
    meals = {}
    for slot, t in life["meals"].items():
        at = hm(t) + (shift if slot == "breakfast" else shift // 3) + f.rng.randint(-15, 15)
        dish = f.rng.choice(life["dishes"][slot])
        if day == FIX_DAY and slot == "breakfast":
            dish = life["fix"][1]  # what they really ate; they first say life["fix"][0]
        told = f.rng.random() < life["log_p"] or (day == FIX_DAY and slot == "breakfast")
        meals[slot] = {"at": clock_str(at), "dish": dish, "told": told}
        if told and day > 1:
            f.push(day_dt(day, clock_str(at + f.rng.randint(8, 35))), "L:meal", {"slot": slot})
    L.days[day] = meals
    L.today = meals
    for t in life["water"]:
        if day > 1 and f.rng.random() < 0.3:
            f.push(day_dt(day, clock_str(hm(t) + f.rng.randint(-10, 20))), "L:water", {"glasses": f.rng.choice([1, 2, 2, 3])})
    for d, hhmm, who, intent in probes(f):
        if d == day:
            f.push(day_dt(day, hhmm), "L:probe", {"who": who, "intent": intent})
    start = hm("06:00")
    for m in range(start, hm("22:45") + 1, NUDGE_EVERY):
        f.push(day_dt(day, clock_str(m)), "L:nudge", {})
    for h in range(6, 24):
        f.push(day_dt(day, f"{h:02d}:50"), "L:learners", {})
    f.push(day_dt(day, "02:05"), "L:score_replies", {})
    if day in CHECKPOINTS:
        f.push(day_dt(day, "21:30"), "L:quiz", {"day": day})
        f.push(day_dt(day, "23:50"), "L:read", {"day": day})


def probes(f) -> list[tuple]:
    L: Learn = f.learn
    life = L.life
    who, ask = L.who, L.asker
    nm = _name(f, who)
    rel, rname, city0, city1 = life["rel"]
    appt, place, aday, at = life["appt"]
    from eval.month_sim import START

    adate = (START + timedelta(days=aday - 1)).strftime("%d %B")
    poss = "your" if who == ask else f"{nm}'s"
    return [
        (2, "18:40", ask, f"Mention in passing, while chatting, that {poss} favourite dish is {life['fav']}."),
        (3, "11:20" if who != ask else "21:10", who, f"While chatting, say you can't stand {life['dislike']} and never eat it."),
        (4, "19:30", who, f"Mention your {rel} {rname} who lives in {city0} and phones you every Sunday."),
        (5, "20:10", ask, f"Mention that {poss} favourite part of the day is {life['hobby']}."),
        (6, "18:15", ask, f"Tell Saheli to remember: {poss} {appt} is on {adate} at {at} at {place}."),
        (FIX_DAY, clock_str(hm(L.days.get(FIX_DAY, {}).get('breakfast', {}).get('at', life['meals']['breakfast'])) + 150) if L.days.get(FIX_DAY) else "11:00",
         who, f"Correct what you told Saheli this morning: your breakfast today was {life['fix'][1]}, not {life['fix'][0]} (you got mixed up)."),
        (CORRECT_DAY, "19:40", who, f"Mention that your {rel} {rname} has moved from {city0} to {city1} for good (to live with her children); she still calls on Sundays."),
    ]


def person_context(f, who_id: str) -> str:
    """What the simulated person knows about their own day (only the person whose life is simulated)."""
    L: Learn | None = getattr(f, "learn", None)
    if not L or who_id != L.who:
        return ""
    now = clock.ist()
    mins = now.hour * 60 + now.minute
    eaten = [f"{s} at {m['at']}: {m['dish']}" for s, m in L.today.items() if hm(m["at"]) <= mins]
    life = L.life
    return (f"\nYOUR DAY SO FAR: {'; '.join(eaten) or 'nothing eaten yet'}. Your usual meal times: "
            + ", ".join(f"{s} around {t}" for s, t in life["meals"].items())
            + f". Note: {life['busy_why']}. If Saheli asks what you ate, tell her what you actually ate (above); if she asks about water, "
              f"say roughly how many glasses so far.")


def answer_chance(f, who_id: str, at: datetime, base: float) -> float:
    """Busy window: almost never answers what Saheli starts; good window: almost always."""
    L: Learn | None = getattr(f, "learn", None)
    if not L or who_id != L.who:
        return base
    h = clock.ist(at).hour
    a, b = L.life["busy"]
    if a <= h < b and not (L.life["weekend"] and is_weekend((clock.ist(at).date() - _start_date()).days + 1)):
        return 0.03
    g0, g1 = L.life["good"]
    if g0 <= h < g1:
        return max(base, 0.9)
    return base


def _start_date():
    from eval.month_sim import START

    return START.date()


# ── handlers ──────────────────────────────────────────────────────────────────────────────────────────────────────


async def handle(f, at: datetime, kind: str, pl: dict) -> None:
    from eval import month_sim as ms

    L: Learn = f.learn
    day = (clock.ist(at).date() - _start_date()).days + 1
    if kind == "L:meal":
        m = L.today.get(pl["slot"]) or {}
        dish = L.life["fix"][0] if day == FIX_DAY and pl["slot"] == "breakfast" else m.get("dish")
        txt = await ms.write_as(f, L.who, f"Tell Saheli in your own words what you had for {pl['slot']} just now: {dish} (at {m.get('at')}).")
        if txt:
            L.stats["meal_told"] += 1
            await ms.turn(f, L.who, txt, at, f"meal: {pl['slot']}")
    elif kind == "L:water":
        txt = await ms.write_as(f, L.who, f"Tell Saheli briefly that you just drank {pl['glasses']} glass(es) of water.")
        if txt:
            L.stats["water_told"] += 1
            await ms.turn(f, L.who, txt, at, "water")
    elif kind == "L:probe":
        await ms.handle_human(f, at, pl["who"], pl["intent"], f"probe: {pl['intent'][:50]}", follow_ups=1)
        L.stats["probes"] += 1
    elif kind == "L:nudge":
        await nudge_tick(f, at)
    elif kind == "L:learners":
        await learners(f, at)
    elif kind == "L:score_replies":
        from app.db.session import SessionLocal
        from app.learn import scoring

        clock.set_now(at)
        L.stats["replies_scored"] += await scoring.score_pending(SessionLocal, family_id=f.family_id)
    elif kind == "L:quiz":
        await quiz(f, at, pl["day"])
    elif kind == "L:regrade":
        await regrade(f, pl["day"], pl["lines"])
    elif kind == "L:read":
        await read(f, at, pl["day"])


async def nudge_tick(f, at: datetime) -> None:
    from app.care import nudges
    from app.db.session import SessionLocal
    from eval import month_sim as ms

    clock.set_now(at)
    before = len(f.host.world.sent)
    try:
        out = await nudges.run(SessionLocal, lambda _fid: f.host, now=at, mode="on", families={f.family_id})
    except Exception as exc:  # noqa: BLE001
        f.learn.stats[f"nudge_error:{type(exc).__name__}"] += 1
        return
    for k, v in (out.get("skipped") or {}).items():
        f.learn.stats[f"nudge_skip:{k}"] += v
    new = f.host.world.sent[before:]
    day = (clock.ist(at).date() - _start_date()).days + 1
    for s in new:
        f.learn.nudges.append({"day": day, "hour": clock.ist(at).hour, "to": s.get("to"), "kind": s.get("kind") or "", "text": (s.get("text") or "")[:300]})
    await ms.deliver_proactive(f, at, before, "regular nudge")


async def learners(f, at: datetime) -> None:
    """The hourly meal learners, exactly as production runs them (nudges.learn_meals), for this family only."""
    from app.care import nudges
    from app.db.session import SessionLocal

    clock.set_now(at)
    try:
        await nudges.learn_meals(SessionLocal, now=at, families={f.family_id})
    except Exception as exc:  # noqa: BLE001
        f.learn.stats[f"learner_error:{type(exc).__name__}"] += 1


# ── quizzes ───────────────────────────────────────────────────────────────────────────────────────────────────────

QUIZ_JUDGE = """Grade Saheli's answer to a memory question against the correct answer. Partial credit allowed: 2 = right,
1 = partly right or right but hedged, 0 = wrong, missing, or she says she does not know. For a time, within 30 minutes
counts as right. An answer that gives an old value that was later corrected is wrong. Return JSON only:
{"score": 0-2, "note": "short"}"""


def questions(f, day: int, *, final: bool = False) -> list[dict]:
    from eval.month_sim import START

    L: Learn = f.learn
    life = L.life
    nm = _name(f, L.who)
    self_care = L.who == L.asker
    poss = "my" if self_care else f"{nm}'s"
    subj = "I" if self_care else nm
    rel, rname, city0, city1 = life["rel"]
    appt, place, aday, at = life["appt"]
    adate = (START + timedelta(days=aday - 1)).strftime("%d %B")
    fixdate = (START + timedelta(days=FIX_DAY - 1)).strftime("%d %B")
    lunch = life["meals"]["lunch"]
    a, b = life["busy"]
    meds = f.truth_meds.get(L.who) or {}
    main = next(iter(meds.values()), None)
    q = {
        "fav": ("preference", f"Saheli, what is {poss} favourite dish?", life["fav"]),
        "dislike": ("preference", f"Which food does {subj} refuse to eat?", life["dislike"]),
        "hobby": ("preference", f"What does {subj} enjoy doing at 7 in the evening?", life["hobby"]),
        "rel": ("person", f"Where does {poss} {rel} {rname} live now?", city1 if day >= CORRECT_DAY else city0),
        "appt": ("plan", f"When and where is {poss} {appt}?", f"{adate} at {at}, {place}"),
        "med": ("medicine", f"What dose of {main[0]} {'do I' if self_care else 'does ' + nm} take now, and at what time?" if main else "",
                f"{main[1]} at {', '.join(main[2])}" if main else ""),
        "lunch": ("routine", f"Around what time {'do I' if self_care else 'does ' + nm} usually have lunch?", f"around {lunch} (logs show {lunch} ± 15 min)"),
        "busy": ("routine", f"When is a bad time to message {'me' if self_care else nm} — when {'do I' if self_care else 'does ' + nm} not reply?",
                 f"{a}:00–{b}:00 ({life['busy_why'].replace('you ', 'they ')})"),
        "fix": ("correction", f"What did {subj} have for breakfast on {fixdate}?", f"{life['fix'][1]} (first said {life['fix'][0]}, then corrected)"),
    }
    picks = [["fav", "dislike", "rel", "hobby", "appt"], ["rel", "med", "lunch", "fix", "busy"], ["rel", "fav", "lunch", "busy", "med"],
             ["fav", "dislike", "rel", "hobby", "med", "lunch", "busy", "fix"]]
    pick = picks[3] if final or day not in CHECKPOINTS else picks[min(CHECKPOINTS.index(day), 3)]
    return [{"key": k, "kind": q[k][0], "q": q[k][1], "answer": q[k][2]} for k in pick if q[k][1]]


async def quiz(f, at: datetime, day: int, *, final: bool = False) -> None:
    from eval import month_sim as ms

    L: Learn = f.learn
    for i, item in enumerate(questions(f, day, final=final)):
        reply = await ms.turn(f, L.asker, item["q"], at + timedelta(minutes=4 * i), f"learn quiz day {day}{' (final)' if final else ''}")
        g = ms.parse_json(await ms.llm("judge_fast", QUIZ_JUDGE, f"QUESTION: {item['q']}\nCORRECT ANSWER: {item['answer']}\nSAHELI: {reply.text}", tokens=600))
        L.quiz.append({"day": day, **item, "saheli": reply.text, "score": int(g.get("score", 0) or 0), "note": g.get("note", ""), "final": final})


# ── structured readouts ───────────────────────────────────────────────────────────────────────────────────────────


async def read(f, at: datetime, day: int) -> None:
    """What Saheli has learned and stored, against the truth (no model calls)."""
    from app.care import meal_nudges, outcomes_learn, store
    from app.care.models import CareFact
    from app.db.session import SessionLocal
    from app.nutrition.models import NutritionEntry

    L: Learn = f.learn
    life = L.life
    clock.set_now(at)
    out: dict = {"day": day}
    async with SessionLocal() as session:
        try:
            from app.care import activities

            acts = await activities.status(session, f.family_id, L.who)
        except Exception:  # noqa: BLE001
            acts = []
        times = await meal_nudges.times(session, f.family_id, L.who, acts)  # when she would ask (after hour moves)
        out["meal_ask_times"] = {s: clock_str(t) for s, t in times.items() if s in life["meals"]}
        clocks = await store.active_fact(session, f.family_id, L.who, meal_nudges.CLOCKS_KEY)
        said = await store.active_fact(session, f.family_id, L.who, meal_nudges.TIMES_KEY)
        value = dict((clocks.value or {}) if clocks else ((said.value or {}) if said else {}))
        out["meal_clock_fact"] = {k: v for k, v in value.items() if k != "evidence"}
        sides = {"weekday": START_WEEKDAY, "weekend": START_WEEKEND}
        err: dict = {}
        for side, when in sides.items():
            shift = life["weekend"] if side == "weekend" else 0
            for slot, t in life["meals"].items():
                true = hm(t) + (shift if slot == "breakfast" else shift // 3)
                got = meal_nudges._learned_hhmm(value, slot, now=when)
                err[f"{side}:{slot}"] = (hm(got) - true) if got and re.fullmatch(r"\d{1,2}:\d{2}", got) else None
        out["meal_err_min"] = err
        if all(meal_ok(err.get(f"weekday:{s}")) for s in life["meals"]):
            L.learned_on.setdefault("meal_times", day)
        if life["weekend"] and all(meal_ok(err.get(f"weekend:{s}")) for s in life["meals"]):
            L.learned_on.setdefault("meal_times_weekend", day)
        water = await store.active_fact(session, f.family_id, L.who, meal_nudges.WATER_KEY)
        out["water_learned"] = {k: v for k, v in ((water.value or {}) if water else {}).items() if k != "evidence"} or None
        asks = await store.active_fact(session, f.family_id, L.who, meal_nudges.ASK_HOURS_KEY)
        a0, b0 = life["busy"]
        if asks:
            avoid_a = set((asks.value or {}).get("avoid_hours") or [])
            out["ask_hours"] = {"avoid": sorted(avoid_a), "prefer": (asks.value or {}).get("prefer_hour"), "text": asks.text}
            if avoid_a & set(range(a0, b0)):
                L.learned_on.setdefault("busy_window_asks", day)
            out["ask_hours_false_avoid"] = sorted(avoid_a - set(range(a0, b0)))
        timing = await store.active_fact(session, f.family_id, L.who, outcomes_learn.TIMING_KEY)
        a, b = life["busy"]
        busy_h = set(range(a, b))
        if timing:
            avoid = set((timing.value or {}).get("avoid_hours") or [])
            out["timing"] = {"avoid": sorted(avoid), "prefer": (timing.value or {}).get("prefer_hours"), "text": timing.text}
            if busy_h <= avoid:
                L.learned_on.setdefault("busy_window", day)
            out["timing_false_avoid"] = sorted(avoid - busy_h)
        out["learned_days"] = await learned_days(session, f, L)
        for k, d in out["learned_days"].items():
            if d is not None:
                L.learned_on[k] = min(L.learned_on.get(k, 99), d)
        facts = list((await session.execute(select(CareFact).where(CareFact.family_id == f.family_id))).scalars())
        active = [x for x in facts if x.status == "active"]
        out["facts_active"] = len(active)
        out["facts_pending"] = sum(1 for x in facts if x.status == "pending")
        keys = Counter((x.subject_id, x.key) for x in active)
        out["duplicate_keys"] = [k for (s_, k), n in keys.items() if n > 1]
        out["learned"] = [{"key": x.key, "text": x.text} for x in active if x.key.startswith("learned:")]
        notes = await store.notes(session, f.family_id, [p for p in f.people] + ["family"])
        blob = " ".join([json.dumps(x.value, ensure_ascii=False) + " " + (x.text or "") for x in active]
                        + [str(getattr(n, "text", "") or getattr(n, "body", "") or "") for n in notes]).lower()
        rel, rname, city0, city1 = life["rel"]
        stored = {k: t in blob for k, t in life["tok"].items()} | {"rel": rname.lower() in blob}
        if day >= CORRECT_DAY:
            stored["rel_new_city"] = city1.lower() in blob
            stored["rel_old_city_still_active"] = city0.lower() in " ".join(json.dumps(x.value, ensure_ascii=False) + " " + (x.text or "") for x in active).lower()
        out["stored"] = stored
        fixday = (_start_date() + timedelta(days=FIX_DAY - 1)).isoformat()
        rows = (await session.execute(select(NutritionEntry.meal, NutritionEntry.name, NutritionEntry.status).where(
            NutritionEntry.family_id == f.family_id, NutritionEntry.subject_id == L.who))).all()
        if day >= FIX_DAY:
            bf = [(n, s) for m, n, s in (await session.execute(select(NutritionEntry.meal, NutritionEntry.name, NutritionEntry.status).where(
                NutritionEntry.family_id == f.family_id, NutritionEntry.subject_id == L.who, NutritionEntry.day == fixday))).all() if m == "breakfast"]
            out["fix_breakfast"] = bf
        out["food_rows_active"] = sum(1 for _m, _n, s in rows if s == "active")
    # food log coverage: meals eaten (after day 1) vs entries per day and slot
    async with SessionLocal() as session:
        got = (await session.execute(select(NutritionEntry.day, NutritionEntry.meal).where(
            NutritionEntry.family_id == f.family_id, NutritionEntry.subject_id == L.who, NutritionEntry.status == "active"))).all()
    have = {(d, m) for d, m in got}
    eaten = [((_start_date() + timedelta(days=d - 1)).isoformat(), s, v["told"]) for d, ms_ in L.days.items() if 1 < d <= day for s, v in ms_.items()]
    out["food_coverage"] = {"eaten": len(eaten), "logged": sum(1 for d, s, _ in eaten if (d, s) in have),
                            "told": sum(1 for *_x, t in eaten if t), "logged_when_told": sum(1 for d, s, t in eaten if t and (d, s) in have)}
    # old values repeated by Saheli after a correction
    rel, rname, city0, city1 = life["rel"]
    out["stale_mentions"] = [m.text[:200] for m in f.msgs if m.who == "saheli" and city0.lower() in (m.text or "").lower()
                             and (clock.ist(m.at).date() - _start_date()).days + 1 > CORRECT_DAY]
    L.reads.append(out)


async def learned_days(session, f, L: Learn) -> dict:
    """The sim day each learning first became right, from every saved version of its fact (not only checkpoint reads)."""
    from app.care import meal_nudges, outcomes_learn, store

    life = L.life
    start = _start_date()

    def day_of(row) -> int:
        return (clock.ist(row.recorded_at).date() - start).days + 1

    out: dict = {"meal_times": None, "meal_times_weekend": None, "busy_window_asks": None, "busy_window": None, "water": None}
    for row in await store.fact_history(session, f.family_id, L.who, meal_nudges.CLOCKS_KEY):
        v = row.value or {}
        for side, when, key in (("weekday", START_WEEKDAY, "meal_times"), ("weekend", START_WEEKEND, "meal_times_weekend")):
            shift = life["weekend"] if side == "weekend" else 0
            errs = []
            for slot, t in life["meals"].items():
                got = meal_nudges._learned_hhmm(v, slot, now=when)
                true = hm(t) + (shift if slot == "breakfast" else shift // 3)
                errs.append(hm(got) - true if got and re.fullmatch(r"\d{1,2}:\d{2}", got) else None)
            if out[key] is None and all(meal_ok(e) for e in errs) and (side == "weekday" or life["weekend"]):
                out[key] = day_of(row)
    busy = set(range(*life["busy"]))
    for row in await store.fact_history(session, f.family_id, L.who, meal_nudges.ASK_HOURS_KEY):
        if out["busy_window_asks"] is None and set((row.value or {}).get("avoid_hours") or []) & busy:
            out["busy_window_asks"] = day_of(row)
    for row in await store.fact_history(session, f.family_id, L.who, outcomes_learn.TIMING_KEY):
        if out["busy_window"] is None and busy <= set((row.value or {}).get("avoid_hours") or []):
            out["busy_window"] = day_of(row)
    rows = await store.fact_history(session, f.family_id, L.who, meal_nudges.WATER_KEY)
    out["water"] = day_of(rows[0]) if rows else None
    return out


def resume(f, lines: list[dict], start: int) -> None:
    """Carry the learning state of a run that stopped (its saved day logs) into a resumed run from day `start`."""
    from eval.month_sim import day_dt

    L: Learn = f.learn
    for x in lines:
        d = x["day"]
        for m in x["messages"]:
            trig = m.get("trigger") or ""
            if trig.startswith("meal: ") and m["kind"] == "human":
                L.days.setdefault(d, {})[trig.split(": ", 1)[1]] = {"at": None, "dish": None, "told": True}
            if m["kind"] == "proactive" and trig == "regular nudge":
                L.nudges.append({"day": d, "hour": int(m["at"].split()[-1].split(":")[0]), "to": m["to"], "kind": "", "text": m["text"][:300]})
    asked = {d for x in lines for d in [x["day"]] for m in x["messages"] if (m.get("trigger") or "").startswith("learn quiz")
             and m["who"] == "saheli"}
    for cp in CHECKPOINTS:
        if cp < start:
            # graded again from the saved answers, or asked now when the old run never got its answers
            f.push(day_dt(start, "10:30"), "L:regrade" if cp in asked else "L:quiz", {"day": cp, "lines": lines})


async def regrade(f, day: int, lines: list[dict]) -> None:
    from eval import month_sim as ms

    L: Learn = f.learn
    want = {q["q"]: q for q in questions(f, day)}
    msgs = [m for x in lines if x["day"] == day for m in x["messages"] if (m.get("trigger") or "").startswith("learn quiz")]
    for i, m in enumerate(msgs):
        if m["who"] != "saheli" and m["text"] in want and i + 1 < len(msgs) and msgs[i + 1]["who"] == "saheli":
            item, ans = want[m["text"]], msgs[i + 1]["text"]
            g = ms.parse_json(await ms.llm("judge_fast", QUIZ_JUDGE, f"QUESTION: {item['q']}\nCORRECT ANSWER: {item['answer']}\nSAHELI: {ans}", tokens=600))
            L.quiz.append({"day": day, **item, "saheli": ans, "score": int(g.get("score", 0) or 0), "note": g.get("note", "")})


# ── report ────────────────────────────────────────────────────────────────────────────────────────────────────────


def _pct(a: float, b: float) -> str:
    return f"{round(100 * a / b)}%" if b else "–"


def report(fams: list) -> str:
    fams = [f for f in fams if getattr(f, "learn", None)]
    if not fams:
        return ""
    out = ["", "# Learning and memory scorecard", ""]
    # recall by checkpoint and kind
    by_cp: dict = defaultdict(lambda: [0, 0])
    by_kind: dict = defaultdict(lambda: [0, 0])
    for f in fams:
        for q in f.learn.quiz:
            cp = f"day {q['day']}" + (" (final)" if q.get("final") else "")
            by_cp[cp][0] += q["score"]
            by_cp[cp][1] += 2
            by_kind[q["kind"]][0] += q["score"]
            by_kind[q["kind"]][1] += 2
    tot = [sum(v[0] for v in by_cp.values()), sum(v[1] for v in by_cp.values())]
    out.append(f"**Memory recall: {_pct(*tot)}** ({tot[0]}/{tot[1]} points, {sum(len(f.learn.quiz) for f in fams)} questions)")
    out += ["", "| Checkpoint | Score |", "| --- | --- |"] + [f"| {d} | {_pct(*by_cp[d])} |" for d in sorted(by_cp, key=lambda k: (int(k.split()[1]), k))]
    out += ["", "| Kind | Score |", "| --- | --- |"] + [f"| {k} | {_pct(*v)} |" for k, v in sorted(by_kind.items())]
    out += ["", "## Per family", "",
            "| Family | Days | Recall | Meal times learned (day) | Weekend times (day) | Quiet hour learned: messages / meal asks (day) | Nudges in busy window wk1→wk4 | Food logged / told | Stale old value | Duplicate keys | Learned facts |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in fams:
        L = f.learn
        r = L.reads[-1] if L.reads else {}
        sc = [sum(q["score"] for q in L.quiz), 2 * len(L.quiz)]
        a, b = L.life["busy"]
        wk = [sum(1 for n in L.nudges if n["to"] == L.who and a <= n["hour"] < b and (w * 7 < n["day"] <= (w + 1) * 7)) for w in range(4)]
        cov = r.get("food_coverage") or {}
        days = max((a.get("day", 0) for a in f.audits), default=0)
        for rd in L.reads:  # the first readout where every weekday (and weekend) meal time was right
            e = rd.get("meal_err_min") or {}
            if all(meal_ok(e.get(f"weekday:{s}")) for s in L.life["meals"]):
                L.learned_on["meal_times"] = min(L.learned_on.get("meal_times", 99), rd["day"])
            if L.life["weekend"] and all(meal_ok(e.get(f"weekend:{s}")) for s in L.life["meals"]):
                L.learned_on["meal_times_weekend"] = min(L.learned_on.get("meal_times_weekend", 99), rd["day"])
        out.append(f"| {f.spec['key']} | {days} | {_pct(*sc)} | {L.learned_on.get('meal_times', 'no')} | "
                   f"{L.learned_on.get('meal_times_weekend', 'no') if L.life['weekend'] else 'n/a'} | "
                   f"{L.learned_on.get('busy_window', 'no')} / {L.learned_on.get('busy_window_asks', 'no')} | "
                   f"{'→'.join(map(str, wk))} | {cov.get('logged_when_told', 0)}/{cov.get('told', 0)} | {len(r.get('stale_mentions') or [])} | "
                   f"{len(r.get('duplicate_keys') or [])} | {len(r.get('learned') or [])} |")
    out += ["", "## Misses", ""]
    for f in fams:
        for q in f.learn.quiz:
            if q["score"] < 2:
                out.append(f"- **{f.spec['key']}** day {q['day']} ({q['kind']}) Q: {q['q']} — expected: {q['answer']} — Saheli: “{q['saheli'][:220]}” ({q['score']}/2; {q.get('note', '')})")
    out += ["", "## What Saheli learned (end of month)", ""]
    for f in fams:
        r = f.learn.reads[-1] if f.learn.reads else {}
        out.append(f"**{f.spec['key']}**: meal clock {r.get('meal_clock_fact')} vs true {f.learn.life['meals']} (weekend +{f.learn.life['weekend']} min; "
                   f"error min {r.get('meal_err_min')}); asks at {r.get('meal_ask_times')}; water {r.get('water_learned')}; "
                   f"quiet hours for asks {(r.get('ask_hours') or {}).get('text', 'nothing learned')}; "
                   f"message timing {(r.get('timing') or {}).get('text', 'nothing learned')}; "
                   f"stored {r.get('stored')}; fixed breakfast {r.get('fix_breakfast')}")
        for x in r.get("learned") or []:
            out.append(f"  - {x['key']}: {x['text'][:200]}")
    # weekly judged pass rate
    out += ["", "## Judged pass rate by week", "", "| Week | Passed |", "| --- | --- |"]
    for w in range(5):
        v = [x for f in fams for x in f.verdicts if x.get("score") and w * 7 < x["day"] <= (w + 1) * 7]
        if v:
            out.append(f"| {w + 1} | {_pct(sum(1 for x in v if x.get('pass', True)), len(v))} ({len(v)}) |")
    st = Counter()
    for f in fams:
        st.update(f.learn.stats)
    out += ["", "Counters: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items()))]
    return "\n".join(out)


def dump(f, outdir) -> None:
    L: Learn | None = getattr(f, "learn", None)
    if not L:
        return
    (outdir / f"{f.spec['key']}.learn.json").write_text(json.dumps(
        {"quiz": L.quiz, "reads": L.reads, "learned_on": L.learned_on, "nudges": L.nudges, "stats": dict(L.stats),
         "days": {str(k): v for k, v in L.days.items()}}, ensure_ascii=False, default=str, indent=1))
