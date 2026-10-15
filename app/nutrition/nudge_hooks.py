"""What Kavach NUdge reads to ask about meals (N5_MEAL_NUDGES_DRAFT, INTERFACES.md §2.4). SQL only, no model calls: safe to
call every minute. Kavach NUdge owns app/care/nudges.py and the meal-ask behaviour (learned:nudges_meals_mode); this module
only answers questions about the food log and writes the habit learnings that come from it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from statistics import median

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.core import clock
from app.nutrition import entries, jobs, pantry, settings, targets
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.models import NutritionEntry

SLOTS = ("breakfast", "lunch", "snack", "dinner")
STABLE_DAYS = 7
STABLE_WITHIN = 45  # minutes either side of the median
USUAL_DAYS = 3  # a portion is theirs once the same amount shows on this many days
USUAL_KEY = "learned:food_usuals"


async def meal_logged(session: AsyncSession, family_id: str, person_id: str, slot: str, day: str | None = None) -> bool:
    """Is this meal accounted for: an entry (food or skipped), an agent question about it still open, or (tracking off) a
    meal event inside the slot's hours."""
    day = day or clock.ist_day()
    got = (await session.execute(select(NutritionEntry.id).where(
        NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person_id, NutritionEntry.day == day,
        NutritionEntry.meal == slot, NutritionEntry.status == "active").limit(1))).first()
    if got:
        return True
    for j in await jobs.open_for(session, family_id, [person_id]):
        ask = j.ask or {}
        at = None
        try:
            from app.nutrition.agent import parse_at

            at = parse_at(ask.get("eaten_at")) or j.created_at
        except Exception:  # noqa: BLE001
            at = j.created_at
        named = ask.get("meal") if isinstance(ask.get("meal"), str) else None
        placed = await entries.assign_slot(session, family_id, person_id, at, named=named)
        if placed.day == day and placed.meal == slot:
            return True
    events = await store.events(session, family_id, person_id, day=day, kinds=["meal"])
    for e in events:
        p = e.payload or {}
        if p.get("nutrition"):
            continue  # the entries above already answered for food-log meals (a removed one does not count)
        if (await entries.assign_slot(session, family_id, person_id, e.at)).meal == slot:
            return True
    return False


def _hm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


async def usual_meal_times(session: AsyncSession, family_id: str, person_id: str, *, days: int = 14) -> dict:
    """The median time of the first entry per slot per day over the last `days`; stable when seen on 7+ days, all within
    45 minutes of the median."""
    since = (clock.ist() - timedelta(days=days)).date().isoformat()
    rows = (await session.execute(select(NutritionEntry.day, NutritionEntry.meal, NutritionEntry.eaten_at).where(
        NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person_id, NutritionEntry.day >= since,
        NutritionEntry.status == "active", NutritionEntry.kind == "food"))).all()
    first: dict[str, dict[str, int]] = defaultdict(dict)
    for day, meal, at in rows:
        local = clock.ist(at)
        m = local.hour * 60 + local.minute
        if meal in SLOTS and (day not in first[meal] or m < first[meal][day]):
            first[meal][day] = m
    out: dict = {"evidence": {}, "stable": {}}
    for slot in SLOTS:
        times = list(first.get(slot, {}).values())
        out["evidence"][slot] = len(times)
        if not times:
            out[slot] = None
            out["stable"][slot] = False
            continue
        med = median(times)
        out[slot] = _hm(med)
        out["stable"][slot] = len(times) >= STABLE_DAYS and all(abs(t - med) <= STABLE_WITHIN for t in times)
    return out


async def pending_confirmations(session: AsyncSession, family_id: str, person_id: str) -> list[dict]:
    """Delivered restaurant meals waiting for "did you eat it?", and the agent's open questions for this person."""
    from app.nutrition import agent

    out: list[dict] = []
    for w in await agent.order_waiting(session, family_id, person_id):
        out.append({"kind": "order_meal", "task_id": w["task_id"], "what": w["what"], "restaurant": w.get("restaurant"), "since": w["since"],
                    "ask": f"Did you eat the {w['what']} that came? All of it?"})
    for j in await jobs.open_for(session, family_id, [person_id]):
        if j.state == "needs_input" and j.needs:
            out.append({"kind": "question", "job_id": str(j.id), "reason": j.needs.get("reason"), "question": j.needs.get("question"),
                        "options": j.needs.get("options") or [], "since": j.updated_at.isoformat()})
    return out


async def progress(session: AsyncSession, family_id: str, person_id: str, day: str | None = None) -> dict:
    """The day so far against their targets and rules, and the only foods a goal nudge may suggest: their own."""
    day = day or clock.ist_day()
    s = await settings.get(session, family_id, person_id)
    rows = await entries.on_day(session, family_id, person_id, day)
    tot = entries.totals(rows)
    tg, rules = await targets.current(session, family_id, person_id)
    covered = entries.meals_covered(rows)
    rule_rows = []
    for r in rules:
        count = sum((e.quantity or 1) if e.unit in (None, "piece") else 1 for e in rows if r["key"] in (e.flags or {}).get("rules", []))
        rule_rows.append({"key": r["key"], "rule": r.get("rule"), "food": r.get("food"), "max_per_day": r.get("max_per_day"),
                          "none": bool(r.get("none")), "today": count, "set_by": r.get("set_by"), "how": r.get("how")})
    since = (clock.ist() - timedelta(days=30)).date().isoformat()
    names = Counter((await session.execute(select(NutritionEntry.name).where(
        NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person_id, NutritionEntry.day >= since,
        NutritionEntry.status == "active", NutritionEntry.kind == "food"))).scalars())
    own = [n for n, _ in names.most_common(15)]
    for p in await pantry.in_stock(session, family_id, person_id):
        if p.name not in own:
            own.append(p.name)
    return {"day": day, "tracking": s["tracking"], "meals_logged": len(set(P.MAIN_MEALS) & covered),
            "slots": {slot: slot in covered for slot in SLOTS},
            "totals": {k: round(v, 1) for k, v in tot.items() if k in N.MACROS or k == "water_ml"},
            "targets": targets.progress(tot, tg), "rules": rule_rows, "own_foods": own[:25]}


async def learn_meal_habits(session: AsyncSession, family_id: str, person_id: str) -> dict:
    """Write learned:nudges_meals_times (stable slots only) and learned:nudges_meals_portions (their 10 most-logged items with
    the usual quantity) through outcomes_learn.save_learned: visible to the family and undoable."""
    from app.care import outcomes_learn

    out: dict = {}
    times = await usual_meal_times(session, family_id, person_id)
    stable = {slot: times[slot] for slot in ("breakfast", "lunch", "dinner") if times["stable"].get(slot)}
    if stable:
        words = "; ".join(f"{slot} around {_twelve(t)}" for slot, t in stable.items())
        out["times"] = await outcomes_learn.save_learned(
            session, family_id=family_id, subject_id=person_id, key="learned:nudges_meals_times",
            value={**stable, "evidence": {k: times["evidence"][k] for k in stable}},
            text=f"Usually has {words}; Saheli asks after that.")
    since = (clock.ist() - timedelta(days=30)).date().isoformat()
    rows = (await session.execute(select(NutritionEntry.name, NutritionEntry.quantity, NutritionEntry.unit).where(
        NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person_id, NutritionEntry.day >= since,
        NutritionEntry.status == "active", NutritionEntry.kind == "food", NutritionEntry.quantity.is_not(None)))).all()
    by: dict[str, Counter] = defaultdict(Counter)
    for name, q, unit in rows:
        by[name][(q, unit)] += 1
    top = sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:10]
    items = {}
    for name, c in top:
        (q, unit), n = c.most_common(1)[0]
        if sum(c.values()) >= 3 and n >= 2:
            items[name] = {"quantity": q, "unit": unit}
    if items:
        text = "Usual portions: " + ", ".join(f"{name} {P.qty_label(v['quantity'], v['unit'])}".strip() for name, v in list(items.items())[:5])
        out["portions"] = await outcomes_learn.save_learned(session, family_id=family_id, subject_id=person_id,
                                                            key="learned:nudges_meals_portions", value={"items": items}, text=text)
    usuals = await _food_usuals(session, family_id, person_id)
    if usuals["items"]:
        text = str(usuals.pop("text"))
        out["usuals"] = await outcomes_learn.save_learned(
            session, family_id=family_id, subject_id=person_id, key=USUAL_KEY, value=usuals, text=text)
    return out


def norm_food(name: str) -> str:
    """The structured food name, compared as data (not the person's sentence)."""
    return " ".join(str(name or "").strip().lower().split())


def match_usual(value: dict | None, name: str, meal: str | None) -> dict | None:
    """Their learned amount for this food at this meal, or None."""
    key = norm_food(name)
    if not key or not isinstance(value, dict):
        return None
    hits = []
    for row in value.get("items") or []:
        if not isinstance(row, dict):
            continue
        names = {norm_food(row.get("name") or "")} | {norm_food(s) for s in row.get("said") or []}
        if key in names:
            hits.append(row)
    if meal:
        same = [r for r in hits if r.get("meal") == meal]
        return same[0] if same else None
    return hits[0] if len(hits) == 1 else None


def usual_for_meal(value: dict | None, meal: str) -> list[dict]:
    if not isinstance(value, dict) or meal not in SLOTS:
        return []
    return [r for r in value.get("items") or [] if isinstance(r, dict) and r.get("meal") == meal]


def as_item(row: dict) -> dict:
    """One learned food, ready for the log, marked as their usual."""
    said = [s for s in row.get("said") or [] if s]
    out = {"name": said[0] if said else row.get("name") or "food", "quantity": row.get("quantity"),
           "unit": row.get("unit"), "_usual": True, "_assumed": True}
    if row.get("food_id"):
        out["food_id"] = row["food_id"]
    return out


def stamp(value: dict | None, items: list[dict], meal: str | None, *, expand: bool) -> list[dict]:
    """Point a known usual at its catalogue food (no new lookup). With expand, add what they take with it."""
    out = [dict(it) for it in items]
    have = {norm_food(it.get("name") or "") for it in out} | {norm_food(it.get("said") or "") for it in out}
    extra: list[dict] = []
    for it in out:
        row = match_usual(value, str(it.get("name") or it.get("said") or ""), meal)
        if row and row.get("food_id") and not it.get("food_id"):
            it["food_id"] = row["food_id"]
        if not expand or row is None:
            continue
        for w in row.get("with") or []:
            comp = match_usual(value, str(w), meal)
            if comp is None or norm_food(comp.get("name") or "") in have:
                continue
            extra.append(as_item(comp))
            have.add(norm_food(comp.get("name") or ""))
    return out + extra


async def _food_usuals(session: AsyncSession, family_id: str, person_id: str) -> dict:
    """Stable foods from this person's own logs: the amount they repeat at a meal, the time when it clusters, and what
    they eat with it. Nothing is inferred from a preference or from a meal they did not log."""
    since = (clock.ist() - timedelta(days=30)).date().isoformat()
    rows = (await session.execute(select(
        NutritionEntry.name, NutritionEntry.name_said, NutritionEntry.quantity, NutritionEntry.unit, NutritionEntry.food_id,
        NutritionEntry.meal, NutritionEntry.eaten_at, NutritionEntry.day, NutritionEntry.meal_id).where(
            NutritionEntry.family_id == family_id, NutritionEntry.subject_id == person_id, NutritionEntry.day >= since,
            NutritionEntry.status == "active", NutritionEntry.kind == "food", NutritionEntry.quantity.is_not(None)))).all()
    buckets: dict[tuple[str, str], dict] = {}
    for name, said, q, unit, food_id, meal, at, day, meal_id in rows:
        if meal not in SLOTS or not name:
            continue
        key = (meal, norm_food(name))
        b = buckets.get(key)
        if b is None:
            b = {"name": name, "said": Counter(), "qty": Counter(), "days": set(), "foods": Counter(),
                 "hours": {}, "meals": set()}
            buckets[key] = b
        b["qty"][(q, unit)] += 1
        b["days"].add(day)
        if said and norm_food(said) != key[1]:
            b["said"][str(said)[:80]] += 1
        if food_id:
            b["foods"][str(food_id)] += 1
        if meal_id:
            b["meals"].add(meal_id)
        if at is not None and day not in b["hours"]:
            local = clock.ist(at)
            b["hours"][day] = local.hour * 60 + local.minute
    stable: list[dict] = []
    by_meal_id: dict = defaultdict(set)
    for (meal, _), b in buckets.items():
        (q, unit), n = b["qty"].most_common(1)[0]
        total = sum(b["qty"].values())
        if len(b["days"]) < USUAL_DAYS or n < 2 or n * 2 < total:
            continue
        mins = list(b["hours"].values())
        hour = None
        if len(mins) >= USUAL_DAYS:
            med = median(mins)
            if all(abs(m - med) <= STABLE_WITHIN for m in mins):
                hour = _hm(med)
        qty = int(q) if float(q).is_integer() else q
        row = {"name": b["name"], "said": [s for s, c in b["said"].most_common(4) if c >= 2], "meal": meal,
               "quantity": qty, "unit": unit, "days": len(b["days"]), "with": []}
        if hour:
            row["hour"] = hour
        if b["foods"]:
            row["food_id"] = b["foods"].most_common(1)[0][0]
        stable.append(row)
        for mid in b["meals"]:
            by_meal_id[mid].add(norm_food(b["name"]))
    names = {(r["meal"], norm_food(r["name"])): r for r in stable}
    for row in stable:
        key = norm_food(row["name"])
        own = {mid for mid, foods in by_meal_id.items() if key in foods}
        if len(own) < USUAL_DAYS:
            continue
        companions: Counter = Counter()
        for mid in own:
            for other in by_meal_id[mid]:
                if other != key:
                    companions[other] += 1
        row["with"] = [names[(row["meal"], o)]["name"] for o, c in companions.most_common(2)
                       if (row["meal"], o) in names and c >= USUAL_DAYS and c * 5 >= len(own) * 3]
    text = _usual_text(stable)
    return {"items": stable, "text": text}


def _usual_text(items: list[dict]) -> str:
    parts = []
    for meal in SLOTS:
        group = [r for r in items if r["meal"] == meal]
        if not group:
            continue
        hours = {str(r["hour"]) for r in group if r.get("hour")}
        when = f" around {_twelve(next(iter(hours)))}" if len(hours) == 1 else ""
        bits = [f"{P.qty_label(r['quantity'], r.get('unit'))} {r['name']}".strip() for r in group]
        parts.append(f"{meal}{when}: " + ", ".join(bits))
    return "Usual " + "; ".join(parts) if parts else ""


def _twelve(hhmm: str) -> str:
    h, m = (int(x) for x in hhmm.split(":"))
    return f"{(h % 12) or 12}:{m:02d} {'am' if h < 12 else 'pm'}"


async def learn_all(sessions) -> dict:
    """Nightly: the habit learnings for everyone who logged food in the last two weeks."""
    since = (clock.ist() - timedelta(days=14)).date().isoformat()
    async with sessions() as session:
        people = sorted({(f, s) for f, s in (await session.execute(select(NutritionEntry.family_id, NutritionEntry.subject_id).where(
            NutritionEntry.day >= since, NutritionEntry.status == "active").distinct())).all() if not f.startswith(("shadow:", "sim-"))})
    done = 0
    for fid, sid in people:
        async with sessions() as session:
            got = await learn_meal_habits(session, fid, sid)
            await session.commit()
        done += bool(got)
    return {"people": len(people), "learned": done}


