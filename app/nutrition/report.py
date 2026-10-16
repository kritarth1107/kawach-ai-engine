"""Food and nutrition views (plan §N8, §N9, §N10): the dashboard page, the brain's food_log answers, the nightly brief's FOOD
block, and the doctor report. Data only: every line says what was logged and how sure it is; nothing here advises.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.llm import router
from app.nutrition import entries, jobs, lookup, pantry, settings, targets
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.models import NutritionEntry

SLOT_SHOWN = ("breakfast", "lunch", "snack", "dinner", "after_dinner", "late_night", "other")
CONF = ("label", "said", "photo", "estimate", "unquantified")
SLOT_END = {"breakfast": 11 * 60, "lunch": 15 * 60 + 30, "snack": 19 * 60, "dinner": 23 * 60}

ITEM_PROMPT = """Someone asks how many of a food a person had. FOODS lists what they logged, with ids. Which of these foods are
that food, or are mostly made of it, so they count towards it ("egg": boiled egg, omelette, egg curry; "tea": tea with milk,
green tea; "roti": chapati, phulka)? Return only JSON: {"ids": ["..."]}"""


def _by_day(rows: list[NutritionEntry]) -> dict[str, list[NutritionEntry]]:
    out: dict[str, list[NutritionEntry]] = defaultdict(list)
    for r in rows:
        out[r.day].append(r)
    return out


def _covered(rows: list[NutritionEntry]) -> int:
    return len(set(P.MAIN_MEALS) & entries.meals_covered(rows))


def _f(x) -> float | None:
    """Amounts are numbers (a target of 60 is 60.0 to the dashboard's types)."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _round(d: dict) -> dict:
    return {k: round(v, 1) for k, v in d.items()}


async def today_text(session: AsyncSession, family_id: str, subject_id: str, day: str) -> str | None:
    tg, _ = await targets.current(session, family_id, subject_id)
    if not tg:
        return None
    return targets.today_line(targets.progress(entries.totals(await entries.on_day(session, family_id, subject_id, day)), tg))


async def count_item(rows: list[NutritionEntry], item: str, *, essential: bool = True) -> dict:
    """How many of one food (the AI decides which logged foods count; code adds up the pieces)."""
    foods = {}
    for r in rows:
        if r.kind == "food" and r.status == "active":
            foods.setdefault(str(r.food_id or r.name), {"id": str(r.food_id or r.name), "name": r.name, "tags": (r.flags or {}).get("tags") or []})
    if not foods:
        return {"item": item, "count": 0, "entries": []}
    try:
        got = await lookup.ask("nutrition", ITEM_PROMPT, json.dumps({"ITEM": item, "FOODS": list(foods.values())}, ensure_ascii=False),
                               essential=essential, max_tokens=600)
    except router.AllModelsFailed:
        return {"item": item, "count": None, "why": "could not count right now"}
    ids = {str(x) for x in (got.get("ids") if isinstance(got, dict) else None) or []}
    hit = [r for r in rows if r.kind == "food" and r.status == "active" and str(r.food_id or r.name) in ids]
    count = sum((r.quantity or 1) if r.unit in (None, "piece", "cup", "glass") else 1 for r in hit)
    return {"item": item, "count": round(count, 1), "entries": [f"{r.day} {entries.line(r)}" for r in hit][:20]}


def _entry_numbers(nutrients: dict | None, *, pending: bool = False) -> dict:
    """The numbers a question about a logged food is answered from. kcal is energy_kcal. A missing nutrient is left out, not shown as 0."""
    if pending:
        return {"numbers": "pending"}
    n = nutrients or {}
    out: dict = {"kcal": N.kcal(n), "protein_g": N.grams_of(n, "protein_g")}
    for key in ("carbs_g", "sugar_g", "fat_g", "fibre_g", "sodium_mg"):
        if key in n:
            out[key] = N.grams_of(n, key)
    return out


async def for_brain(session: AsyncSession, family_id: str, subject_id: str, first: str, last: str, *, what: str, item: str | None,
                    name: str, essential: bool = True) -> dict:
    rows = await entries.between(session, family_id, subject_id, first, last)
    s = await settings.get(session, family_id, subject_id)
    by_day = _by_day(rows)
    days = entries.day_days(last, (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days + 1)
    out: dict = {"who": name, "from": first, "to": last, "tracking": s["tracking"]}
    if what == "entries":
        out["entries"] = [{**_entry_numbers(r.nutrients, pending=entries.numbers_pending(r)), "entry_id": str(r.id), "day": r.day,
                           "time": clock.ist(r.eaten_at).strftime("%H:%M"), "meal": r.meal, "what": entries.line(r),
                           "confidence": r.confidence} for r in rows][:60]
    elif what == "counts":
        out["counts"] = entries.counters(rows, s)
        out["labels"] = {k: f"{v[0]} ({v[1]})" if v[1] else v[0] for k, v in entries.COUNTER_LABELS.items()}
        if len(days) > 1:
            out["per_day"] = {d: entries.counters(by_day.get(d, []), s) for d in days}
    elif what == "item":
        if not item:
            out["error"] = "give item"
        else:
            out.update(await count_item(rows, item, essential=essential))
    else:
        tg, _ = await targets.current(session, family_id, subject_id)
        per_day = []
        for d in days:
            day_rows = by_day.get(d, [])
            t = entries.totals([r for r in day_rows if not entries.numbers_pending(r)])
            waiting = any(entries.numbers_pending(r) for r in day_rows) and "energy_kcal" not in t
            per_day.append({"day": d, "kcal": None if waiting else N.kcal(t), "numbers": "pending" if waiting else None,
                            "protein_g": None if waiting else N.grams_of(t, "protein_g"), "carbs_g": N.grams_of(t, "carbs_g"),
                            "fat_g": N.grams_of(t, "fat_g"), "fibre_g": N.grams_of(t, "fibre_g"), "sugar_g": N.grams_of(t, "sugar_g"),
                            "sodium_mg": N.grams_of(t, "sodium_mg"), "water_ml": N.grams_of(t, "water_ml"), "main_meals_logged": _covered(by_day.get(d, [])),
                            "items": len([r for r in by_day.get(d, []) if r.kind == "food"])})
        out["days"] = per_day
        if len(days) == 1 and tg:
            prog = targets.progress(entries.totals(by_day.get(days[0], [])), tg)
            out["targets"] = prog
            out["today"] = targets.today_line(prog)
        estimated = [entries.line(r) for r in rows if r.confidence in ("estimate", "unquantified")]
        if estimated:
            out["estimated"] = estimated[:10]
    if not rows:
        out["note"] = "Nothing logged in this period: say so plainly; never guess what they ate."
    elif any(entries.numbers_pending(r) for r in rows):
        out["note"] = "Some of this food is saved and its numbers are still being added. Do not say 0 calories or 0 g protein for it."
    return out


# ── the dashboard page ───────────────────────────────────────────────────────


def _carried(by_day: dict[str, list[NutritionEntry]], days: list[str], key: str) -> list[float]:
    """A nutrient's day totals over the days whose food carried it: a value nobody logged is unknown, not zero (saturated
    fat with no food that gives it must not read 0 g on a doctor's report)."""
    out = []
    for d in days:
        t = entries.totals(by_day.get(d, []))
        if key in t:
            out.append(float(t[key]))
    return out


def _day_meals(rows: list[NutritionEntry]) -> dict:
    """What a day had in each meal, for the week cards and the calendar's meal dots: {meals, skipped, logged}."""
    meals = []
    for slot in SLOT_SHOWN:
        # items of one log share eaten_at and created_at: the name breaks the tie, so the same day reads the same every time
        food = sorted((r for r in rows if r.meal == slot and r.kind == "food" and r.status == "active"),
                      key=lambda r: (r.eaten_at, r.created_at, r.name))
        if food:
            t = entries.totals(food)
            meals.append({"meal": slot, "at": food[0].eaten_at.isoformat(), "names": [r.name for r in food][:12], "kcal": N.kcal(t),
                          "protein_g": N.grams_of(t, "protein_g")})
    skipped = sorted({r.meal for r in rows if r.kind == "skipped" and r.status == "active"} - {m["meal"] for m in meals})
    return {"meals": meals, "skipped": skipped, "logged": [m["meal"] for m in meals] + skipped}


def _met(t: dict, totals: dict) -> bool | None:
    p = targets.progress(totals, [t])
    return p[0]["met"] if p else None


def _noticed(by_day: dict[str, list[NutritionEntry]], week: list[str], tg: list[dict], today: str) -> list[dict]:
    """Data-only lines (plan §N8.2 "What Saheli noticed"): counts over the last 7 days, never advice."""
    out: list[dict] = []
    past = [d for d in week if d != today]
    logged = [d for d in past if by_day.get(d)]
    if len(logged) >= 3:
        for slot in P.MAIN_MEALS:
            missing = [d for d in logged if slot not in entries.meals_covered(by_day.get(d, []))]
            skipped = [d for d in logged if any(r.meal == slot and r.kind == "skipped" for r in by_day.get(d, []))]
            if len(skipped) >= 2:
                out.append({"key": f"skipped:{slot}", "text": f"{slot.capitalize()} skipped on {len(skipped)} of the last {len(logged)} logged days",
                            "severity": "watch" if len(skipped) >= 3 else "info"})
            elif len(missing) >= 3:
                out.append({"key": f"missing:{slot}", "text": f"{slot.capitalize()} not logged on {len(missing)} of the last {len(logged)} logged days",
                            "severity": "info"})
        for t in tg:
            if t.get("status") != "active":
                continue
            met = [d for d in logged if _met(t, entries.totals(by_day.get(d, [])))]
            label = targets.target_text(t).split(" (")[0]
            out.append({"key": f"target:{t['field']}", "text": f"{label}: met on {len(met)} of the last {len(logged)} logged days",
                        "severity": "info"})
    devs = sum(1 for d in week for r in by_day.get(d, []) if (r.flags or {}).get("deviation"))
    if devs:
        out.append({"key": "deviations", "text": f"Past a strict target or rule {devs} time(s) in the last 7 days", "severity": "watch"})
    allergy = [r for d in week for r in by_day.get(d, []) if (r.flags or {}).get("allergy")]
    if allergy:
        out.append({"key": "allergy", "text": f"{len(allergy)} item(s) in the last 7 days may contain something on their allergy list",
                    "severity": "watch"})
    return out


def _meals(rows: list[NutritionEntry], day: str, now_local: datetime) -> list[dict]:
    out = []
    today = now_local.date().isoformat()
    minute = now_local.hour * 60 + now_local.minute
    for slot in SLOT_SHOWN:
        es = [r for r in rows if r.meal == slot]
        if not es and slot not in P.MAIN_MEALS:
            continue
        known = [r for r in es if not entries.numbers_pending(r)]
        t = entries.totals(known)
        waiting = any(entries.numbers_pending(r) for r in es) and "energy_kcal" not in t and "water_ml" not in t
        due = day < today or (day == today and minute >= SLOT_END.get(slot, 24 * 60))
        out.append({"meal": slot, "label": entries.MEAL_LABEL[slot], "entries": [entries.entry_json(r) for r in es],
                    "kcal": None if waiting else N.kcal(t), "numbersPending": waiting,
                    "protein_g": N.grams_of(t, "protein_g"), "skipped": bool(es) and all(r.kind == "skipped" for r in es),
                    "notLogged": not es and slot in P.MAIN_MEALS and due})
    return out


async def view(session: AsyncSession, family_id: str, subject_id: str, *, day: str | None = None, days: int = 7) -> dict:
    from app.nutrition import agent

    now_local = clock.ist()
    day = day or now_local.date().isoformat()
    days = days if days in (1, 7, 30) else 7
    s = await settings.get(session, family_id, subject_id)
    tg, rules = await targets.current(session, family_id, subject_id, pending=True)
    active_tg = [t for t in tg if t.get("status") == "active"]
    month_days = entries.day_days(day, 30)
    week = month_days[-7:]
    rows = await entries.between(session, family_id, subject_id, month_days[0], day)
    by_day = _by_day(rows)
    today_rows = by_day.get(day, [])
    tot = entries.totals([r for r in today_rows if not entries.numbers_pending(r)])
    today_waiting = any(entries.numbers_pending(r) for r in today_rows) and "energy_kcal" not in tot and "water_ml" not in tot
    glass = float(s["sizes"].get("glass_ml") or 250)
    water_t = next((t for t in active_tg if t["field"] == "water_ml"), None)
    counters_week = {d: entries.counters(by_day.get(d, []), s) for d in week}
    counters = []
    for c in s["counters"]:
        label, unit = entries.COUNTER_LABELS[c]
        past = [counters_week[d][c] for d in week[:-1] if by_day.get(d)]
        counters.append({"key": c, "label": label, "unit": unit, "today": counters_week[day][c] if day in counters_week else 0,
                         "usual": round(sum(past) / len(past), 1) if past else None, "week": [counters_week[d][c] for d in week]})

    def day_row(d: str) -> dict:
        rows_d = by_day.get(d, [])
        known = [r for r in rows_d if not entries.numbers_pending(r)]
        t = entries.totals(known)
        waiting = any(entries.numbers_pending(r) for r in rows_d) and "energy_kcal" not in t and "water_ml" not in t
        return {**_day_meals(rows_d), "day": d, "kcal": None if waiting else N.kcal(t), "numbersPending": waiting,
                "protein_g": None if waiting else N.grams_of(t, "protein_g"), "carbs_g": N.grams_of(t, "carbs_g"),
                "fat_g": N.grams_of(t, "fat_g"), "fibre_g": N.grams_of(t, "fibre_g"), "sugar_g": N.grams_of(t, "sugar_g"),
                "sodium_mg": N.grams_of(t, "sodium_mg"), "water_ml": N.grams_of(t, "water_ml"), "items": len([r for r in by_day.get(d, []) if r.kind == "food"]),
                "mealsLogged": _covered(by_day.get(d, [])), "complete": round(_covered(by_day.get(d, [])) / 3, 2),
                "met": {t2["field"]: bool(_met(t2, t)) for t2 in active_tg} if by_day.get(d) else {}}

    calendar = [day_row(d) for d in month_days]
    logged7 = [r for r in calendar[-7:] if r["items"]]
    logged30 = [r for r in calendar if r["items"]]
    nutrients = []
    for k in N.MACROS + ("sat_fat_g", "potassium_mg", "calcium_mg", "iron_mg", "water_ml"):
        vals7 = _carried(by_day, [r["day"] for r in logged7], k)
        vals30 = _carried(by_day, [r["day"] for r in logged30], k)
        t = next((x for x in active_tg if x["nutrient"] == k), None)
        label, unit = N.LABELS.get(k, (k, ""))
        nutrients.append({"key": k, "label": label, "unit": unit, "avg7": round(sum(vals7) / len(vals7), 1) if vals7 else None,
                          "avg30": round(sum(vals30) / len(vals30), 1) if vals30 else None, "target": _f(targets.goal_of(t)) if t else None,
                          "min": _f(t.get("min")) if t else None, "max": _f(t.get("max")) if t else None, "highlight": t is not None})
    waiting = await agent.order_waiting(session, family_id, subject_id)
    asked = [j for j in await jobs.open_for(session, family_id, [subject_id]) if j.state == "needs_input" and j.needs]
    proteins = next((t for t in active_tg if t["field"] == "protein_g"), None)
    protein_streak = 0
    if proteins:
        for r in reversed(calendar[:-1] if day == now_local.date().isoformat() else calendar):
            if r["met"].get("protein_g"):
                protein_streak += 1
            else:
                break
    return {
        "subjectId": subject_id, "day": day, "days": days, "tracking": s,
        "targets": [{"key": t["key"], "field": t["field"], "nutrient": t["nutrient"], "text": t["text"], "sense": t.get("sense"),
                     "value": _f(targets.goal_of(t)), "min": _f(t.get("min")), "max": _f(t.get("max")), "unit": t.get("unit"), "setBy": t.get("set_by"),
                     "how": t.get("how"), "status": t["status"]} for t in tg],
        "rules": [{"key": r["key"], "rule": r.get("rule"), "food": r.get("food"), "maxPerDay": r.get("max_per_day"), "none": bool(r.get("none")),
                   "when": r.get("when"), "setBy": r.get("set_by"), "how": r.get("how"), "status": r["status"], "text": r["text"]} for r in rules],
        "today": {"day": day, "meals": _meals(today_rows, day, now_local), "numbersPending": today_waiting,
                  "totals": _round({k: v for k, v in tot.items() if k in N.TOTAL_KEYS}),
                  "progress": targets.progress(tot, active_tg),
                  "water": {"ml": N.grams_of(tot, "water_ml"), "glasses": round(N.grams_of(tot, "water_ml") / glass, 1), "glassMl": glass,
                            "targetMl": _f(targets.goal_of(water_t)) if water_t else None}},
        "history": [day_row(d) for d in month_days[-days:]],
        "counters": counters,
        "week": calendar[-7:],
        "month": {"calendar": [{"day": r["day"], "complete": r["complete"], "kcal": r["kcal"], "items": r["items"], "logged": r["logged"]}
                               for r in calendar],
                  "trends": {k: [r[k] for r in calendar] for k in ("kcal", "protein_g", "sugar_g", "sodium_mg")}},
        "nutrients": nutrients,
        "streaks": {"allMeals": entries.all_meals_streak(by_day, month_days), "proteinMet": protein_streak},
        "pantry": [pantry.pantry_json(p) for p in await pantry.in_stock(session, family_id, subject_id)],
        "waiting": waiting,
        "pendingQuestions": [{"jobId": str(j.id), "reason": (j.needs or {}).get("reason"), "question": (j.needs or {}).get("question"),
                              "options": (j.needs or {}).get("options") or [], "since": j.updated_at.isoformat()} for j in asked],
        "noticed": _noticed(by_day, week, tg, day),
        "units": list(P.UNITS), "mealSlots": list(P.MEALS),
    }


# ── the nightly brief (plan §N9, founder Q7) ─────────────────────────────────


async def brief_block(session: AsyncSession, family_id: str, subject_id: str, day: str, *, now: datetime | None = None,
                      to: str | None = None, summary_defaults: list[str] | None = None) -> str:
    """The FOOD block of a nightly note: meals with items, totals against targets, water, meals not logged,
    delivered food not yet confirmed, strict rules gone past, and what was only estimated. Empty when tracking is off,
    when this recipient is not meant to hear it, or when there is nothing at all.

    ``to`` left out keeps the block (callers that have not chosen a recipient yet). With ``to`` set, an empty
    ``daily_summary_to`` includes the usual people in ``summary_defaults``; a list includes only those ids.
    """
    from app.nutrition import agent

    s = await settings.get(session, family_id, subject_id)
    if not settings.tracked(s):
        return ""
    if to is not None and not settings.wants_summary(s, to, defaults=summary_defaults or []):
        return ""
    rows = await entries.on_day(session, family_id, subject_id, day)
    tg, _ = await targets.current(session, family_id, subject_id)
    waiting = await agent.order_waiting(session, family_id, subject_id)
    if not rows and not waiting and not tg:
        return ""
    focus = await targets.focus(session, family_id, subject_id)
    lines = ["FOOD (from the food log; each item with how sure it is: said, photo, label, estimate, unquantified):"]
    for slot in SLOT_SHOWN:
        es = [r for r in rows if r.meal == slot]
        if es and all(r.kind == "skipped" for r in es):
            lines.append(f"  {slot}: skipped (they said)")
        elif es:
            known = [r for r in es if not entries.numbers_pending(r)]
            t = entries.totals(known)
            items = "; ".join(f"{entries.line(r)} [{r.confidence}]" for r in es if r.kind == "food")
            phrase = N.highlight_phrase(t, focus)
            extra = f"; about {phrase}" if phrase else ""
            if any(entries.numbers_pending(r) for r in es) and not phrase:
                extra = "; numbers still being added"
            lines.append(f"  {slot} ({clock.ist(es[0].eaten_at).strftime('%H:%M')}): {items}{extra}")
    missing = [m for m in P.MAIN_MEALS if m not in entries.meals_covered(rows)]
    if missing:
        lines.append(f"  not logged: {', '.join(missing)} (say 'not logged', never 'skipped')")
    known_rows = [r for r in rows if not entries.numbers_pending(r)]
    tot = entries.totals(known_rows)
    if rows and s["tracking"] == "full":
        phrase = N.highlight_phrase(tot, focus)
        if phrase:
            lines.append(f"  totals: {phrase}")
        elif any(entries.numbers_pending(r) for r in rows):
            lines.append("  totals: numbers still being added")
    glass = float(s["sizes"].get("glass_ml") or 250)
    if tot.get("water_ml"):
        lines.append(f"  water: {round(tot['water_ml'] / glass, 1)} glasses ({N.grams_of(tot, 'water_ml'):.0f} ml)")
    prog = targets.progress(tot, tg)
    if prog and s["tracking"] == "full":
        lines.append("  against targets: " + "; ".join(
            f"{p['label']} {N.fmt(p['so_far'])} / {N.fmt(float(p['target']))} {p['unit']} ({'met' if p['met'] else 'not met'}; set by "
            f"{targets.WHO.get(p.get('set_by') or '', 'someone')}, {p.get('how')})" for p in prog))
    devs = sorted({d for r in rows for d in (r.flags or {}).get("deviation") or []})
    if devs:
        lines.append("  strict rules or targets gone past: " + "; ".join(devs[:4]))
    allergy = [f"{r.name} ({', '.join(r.flags['allergy'])})" for r in rows if (r.flags or {}).get("allergy")]
    if allergy:
        lines.append("  may contain an allergen on their list: " + "; ".join(allergy[:3]))
    for w in waiting[:3]:
        lines.append(f"  delivered, not yet confirmed eaten: {w['what']}" + (f" from {w['restaurant']}" if w.get("restaurant") else "")
                     + " (nothing logged until someone says)")
    return "\n".join(lines)


def recap_prompt(person: dict, day: str, food: str) -> str:
    """What Saheli writes for a self-care user's own food. Numbers only from the block; no advice of her own."""
    name = person.get("name") or "them"
    return (f"[Food recap] {name}'s own food for {day} (they manage their own care). Write it to them with send_message: "
            f"one short note of what they logged, the kcal and the nutrients named in FOOD, water, meals not logged, and any strict target "
            f"gone past. Only the FOOD block below. No advice, no judgement, no medical opinion. Numbers only as written; "
            f"an estimate stays an estimate. If the block is empty, send nothing. Then reply none.\n\n{food}")


def recap_waiting_text(person: dict, day: str, food: str) -> str:
    """Held until they next write. Food only, so the waiting brief never carries the caregiver digest."""
    name = person.get("name") or "them"
    return (f"OWN FOOD RECAP for {name} on {day}. This is their own food, not a caregiver brief about someone else. "
            f"Write only a short food note from the FOOD block. No advice, no medicines, nothing that is not in the block.\n\n{food}")


# ── the doctor report (plan §N10, founder Q8: only when the caregiver shares it) ──


async def doctor(session: AsyncSession, family_id: str, subject_id: str, days: int = 7) -> dict:
    from app.care.models import CareEvent

    days = days if days in (7, 14, 30) else 7
    last = clock.ist_day()
    span = entries.day_days(last, days)
    rows = await entries.between(session, family_id, subject_id, span[0], last)
    by_day = _by_day(rows)
    logged = [d for d in span if any(r.kind == "food" for r in by_day.get(d, []))]
    daily = []
    for d in span:
        t = entries.totals(by_day.get(d, []))
        daily.append({"day": d, **{k: round(t.get(k, 0.0), 1) for k in N.MACROS + ("water_ml",)}, "mealsLogged": _covered(by_day.get(d, []))})
    avg = {}
    for k in N.MACROS + ("sat_fat_g", "potassium_mg", "calcium_mg", "iron_mg", "water_ml"):
        vals = _carried(by_day, logged, k)
        avg[k] = round(sum(vals) / len(vals), 1) if vals else None
    tg, rules = await targets.current(session, family_id, subject_id)
    vs = []
    for t in tg:
        met = sum(1 for d in logged if _met(t, entries.totals(by_day.get(d, []))))
        vs.append({"text": t["text"], "field": t["field"], "average": avg.get(t["nutrient"]), "daysMet": met, "daysLogged": len(logged),
                   "setBy": t.get("set_by"), "how": t.get("how")})
    pattern = {slot: sum(1 for d in span if any(r.meal == slot and r.kind == "food" for r in by_day.get(d, []))) for slot in P.MAIN_MEALS}
    skipped = {slot: sum(1 for d in span if any(r.meal == slot and r.kind == "skipped" for r in by_day.get(d, []))) for slot in P.MAIN_MEALS}
    # water is in the averages (water_ml); "most eaten" is food
    eaten = [r for r in rows if r.kind == "food" and "water" not in ((r.flags or {}).get("tags") or [])]
    names = Counter(r.name for r in eaten)
    kcal_by = Counter()
    for r in eaten:
        kcal_by[r.name] += N.kcal(r.nutrients)
    total_kcal = sum(kcal_by.values()) or 1
    conf = Counter(r.confidence for r in rows if r.kind == "food")
    weights = (await session.execute(select(CareEvent.day, CareEvent.payload).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "vital",
        CareEvent.day >= entries.day_days(last, 90)[0]).order_by(CareEvent.at))).all()
    weight = [{"day": d, "value": N.number((p or {}).get("value"))} for d, p in weights if str((p or {}).get("kind") or "").lower() == "weight"]
    return {
        "subjectId": subject_id, "days": days, "from": span[0], "to": last, "daysLogged": len(logged), "averages": avg, "daily": daily,
        "targets": vs, "rules": [{"text": r["text"], "setBy": r.get("set_by"), "how": r.get("how")} for r in rules],
        "mealPattern": pattern, "skipped": skipped,
        "topFoods": [{"name": n, "times": c, "kcalShare": round(100 * kcal_by[n] / total_kcal)} for n, c in names.most_common(10)],
        "confidence": {k: conf.get(k, 0) for k in CONF},
        "deviations": sorted({d for r in rows for d in (r.flags or {}).get("deviation") or []})[:10],
        "weight": [w for w in weight if w["value"] is not None][-20:],
        "note": "Numbers come from what was logged; items marked estimate or unquantified are rough (a usual portion, or no amount given).",
    }


def doctor_text(name: str, r: dict) -> str:
    a = r["averages"]
    if not r["daysLogged"]:
        return f"No food was logged for {name} in the last {r['days']} days."
    lines = [f"Food and nutrition for {name}, {r['from']} to {r['to']}: logged on {r['daysLogged']} of {r['days']} days."]
    if a.get("energy_kcal") is not None:
        lines.append(f"Daily average (logged days): {N.fmt(a['energy_kcal'])} kcal, protein {N.fmt(a.get('protein_g') or 0)} g, carbs "
                     f"{N.fmt(a.get('carbs_g') or 0)} g, fat {N.fmt(a.get('fat_g') or 0)} g, sugar {N.fmt(a.get('sugar_g') or 0)} g, sodium "
                     f"{N.fmt(a.get('sodium_mg') or 0)} mg.")
    for t in r["targets"]:
        lines.append(f"{t['text']}: met on {t['daysMet']} of {t['daysLogged']} logged days.")
    lines.append("Main meals logged: " + ", ".join(f"{k} {v}/{r['days']}" for k, v in r["mealPattern"].items()) + ".")
    if r["topFoods"]:
        lines.append("Most eaten: " + ", ".join(f"{f['name']} ({f['times']}x)" for f in r["topFoods"][:5]) + ".")
    c = r["confidence"]
    rough = c.get("estimate", 0) + c.get("unquantified", 0)
    if rough:
        lines.append(f"{rough} item(s) were estimates or had no amount; treat those numbers as rough.")
    return "\n".join(lines)
