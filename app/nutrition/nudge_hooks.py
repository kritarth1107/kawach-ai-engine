"""What Kavach NUdge reads to ask about meals (N5_MEAL_NUDGES_DRAFT, INTERFACES.md §2.4). SQL only, no model calls: safe to
call every minute. Kavach NUdge owns app/care/nudges.py and the meal-ask behaviour (learned:nudges_meals_mode); this module
only answers questions about the food log and writes the two habit learnings that come from it.
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
        if clock.ist_day(at) == day and (ask.get("meal") or P.meal_for(at)) == slot:
            return True
    events = await store.events(session, family_id, person_id, day=day, kinds=["meal"])
    for e in events:
        p = e.payload or {}
        if p.get("nutrition"):
            continue  # the entries above already answered for food-log meals (a removed one does not count)
        if P.meal_for(e.at) == slot:
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
    return out


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


