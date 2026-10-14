"""Targets and food rules (care facts in domain `nutrition_target`; founder Q6 replaces plan §N7.2).

- Optional: there is no target by default and no general guide.
- Anyone responsible can set one: the caregiver, the person themselves, or (entered by them) a doctor, dietitian or gym
  trainer. Each records `set_by` (caregiver | self | doctor | dietitian | trainer | other) and `how` (strict | guide).
- A number per nutrient (kcal, protein, sugar max, …) is one fact each (`nutrition_target:protein_g`); a food rule is one
  fact each (`nutrition_target:rule_<slug>`: "max 2 eggs a day", "no sugar", "protein after gym").
- strict: Saheli points out a deviation when it is logged, and the caregiver's brief lists it. guide: progress only.
- A target Saheli infers from chat waits (pending) until the person or a caregiver confirms it.
- An extreme calorie target (under 1,200 or over 3,500 kcal) is saved with one line suggesting a check with the doctor.
- The domain is a health domain: a weaker source changing a caregiver's target waits for a caregiver (store rules).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import slug
from app.nutrition import nutrients as N

DOMAIN = "nutrition_target"
SET_BY = ("caregiver", "self", "doctor", "dietitian", "trainer", "other")
HOW = ("strict", "guide")
WHO = {"caregiver": "the caregiver", "self": "themselves", "doctor": "their doctor", "dietitian": "their dietitian",
       "trainer": "their trainer", "other": "someone"}


def key_for(field: str) -> str:
    return f"{DOMAIN}:{field}"


def rule_key(rule: dict) -> str:
    return f"{DOMAIN}:rule_{slug(str(rule.get('food') or rule.get('rule') or 'rule'))[:80]}"


def target_value(field: str, given, *, set_by: str, how: str) -> dict | None:
    """A number (its usual sense: protein at least, sugar at most, kcal about) or {min, max, value}. Keys with no value are left
    out (the care record's fact values share one shape across domains: no nulls, and "amount" never collides with "value")."""
    sense = N.TARGET_SENSE[field]
    v: dict = {"field": field, "nutrient": N.TARGETS[field], "unit": "kcal" if field == "kcal" else field.rsplit("_", 1)[-1],
               "set_by": set_by if set_by in SET_BY else "other", "how": how if how in HOW else "guide"}
    if isinstance(given, dict):
        lo, hi, val = N.number(given.get("min")), N.number(given.get("max")), N.number(given.get("value"))
        if lo is None and hi is None and val is None:
            return None
        v.update({"sense": "range" if lo is not None and hi is not None else ("min" if lo is not None else "max" if hi is not None else sense),
                  "min": lo, "max": hi, "amount": val if val is not None else (lo if hi is None else hi if lo is None else None)})
    else:
        n = N.number(given)
        if n is None or n <= 0:
            return None
        v.update({"sense": sense, "amount": n, "min": n if sense == "min" else None, "max": n if sense == "max" else None})
    return {k: x for k, x in v.items() if x is not None}


def goal_of(v: dict) -> float | None:
    return v.get("amount") or v.get("min") or v.get("max")


def target_text(v: dict) -> str:
    label, unit = N.LABELS.get(v["nutrient"], (v["field"], ""))
    unit = "ml" if v["field"] == "water_ml" else unit
    if v.get("sense") == "range":
        amount = f"{N.fmt(v['min'])}–{N.fmt(v['max'])} {unit}"
    elif v.get("sense") == "min":
        amount = f"at least {N.fmt(v.get('min') or goal_of(v) or 0)} {unit}"
    elif v.get("sense") == "max":
        amount = f"at most {N.fmt(v.get('max') or goal_of(v) or 0)} {unit}"
    else:
        amount = f"about {N.fmt(goal_of(v) or 0)} {unit}"
    return f"{'Water' if v['field'] == 'water_ml' else label} {amount} a day (set by {WHO.get(v['set_by'], 'someone')}, {v['how']})"


def rule_value(rule: dict, *, set_by: str, how: str) -> dict | None:
    text = str(rule.get("rule") or "").strip()
    food = str(rule.get("food") or "").strip()
    if not (text or food):
        return None
    mx = N.number(rule.get("max_per_day"))
    v = {"rule": text[:200] or food[:200], "food": food[:120], "max_per_day": mx, "none": bool(rule.get("none")),
         "when": str(rule.get("when") or "")[:120] or None, "set_by": set_by if set_by in SET_BY else "other",
         "how": how if how in HOW else "guide"}
    return {k: x for k, x in v.items() if x is not None}


def rule_text(v: dict) -> str:
    return f"Food rule: {v['rule']} (set by {WHO.get(v['set_by'], 'someone')}, {v['how']})"


async def current(session: AsyncSession, family_id: str, person_id: str, *, pending: bool = False) -> tuple[list[dict], list[dict]]:
    """(targets, rules) on the record: {key, status, ...value}."""
    rows = await store.facts(session, family_id, person_id, domains=[DOMAIN], statuses=("active", "pending") if pending else ("active",))
    targets, rules = [], []
    for f in rows:
        v = {**(f.value or {}), "key": f.key, "status": f.status, "text": f.text}
        (rules if f.key.startswith(f"{DOMAIN}:rule_") else targets).append(v)
    return [t for t in targets if t.get("nutrient")], rules


def progress(totals: dict, targets: list[dict]) -> list[dict]:
    """Each target against what is logged so far (display data only; never advice)."""
    out = []
    for t in targets:
        if t.get("status", "active") != "active":
            continue
        so_far = float((totals or {}).get(t["nutrient"]) or 0)
        goal = goal_of(t)
        if not goal:
            continue
        sense = t.get("sense")
        met = (so_far >= (t.get("min") or goal)) if sense in ("min", "target") else (
            so_far <= (t.get("max") or goal)) if sense == "max" else ((t.get("min") or 0) <= so_far <= (t.get("max") or goal))
        label, unit = N.LABELS.get(t["nutrient"], (t["field"], ""))
        out.append({"key": t["key"], "nutrient": t["nutrient"], "field": t["field"], "label": "Water" if t["field"] == "water_ml" else label,
                    "unit": "ml" if t["field"] == "water_ml" else unit, "so_far": round(so_far, 1), "target": float(goal),
                    "min": float(t["min"]) if t.get("min") is not None else None, "max": float(t["max"]) if t.get("max") is not None else None, "sense": sense, "left": round(max(0.0, float(goal) - so_far), 1) if sense in ("min", "target") else None,
                    "over": round(max(0.0, so_far - float(t.get("max") or goal)), 1) if sense in ("max", "range") and t.get("max") else None,
                    "pct": round(100 * so_far / float(goal)) if goal else None, "met": bool(met), "set_by": t.get("set_by"), "how": t.get("how")})
    return out


def today_line(prog: list[dict]) -> str | None:
    """'Today: 1,240 / 1,800 kcal, 46 / 90 g protein.' (only targets that exist; kcal and protein first)."""
    order = {"energy_kcal": 0, "protein_g": 1}
    bits = []
    for p in sorted(prog, key=lambda x: order.get(x["nutrient"], 5))[:3]:
        unit = p["unit"]
        name = "" if p["nutrient"] == "energy_kcal" else f" {p['label'].lower()}"
        bits.append(f"{N.fmt(p['so_far'])} / {N.fmt(float(p['target']))} {unit}{name}")
    return ("Today: " + ", ".join(bits) + ".") if bits else None


async def focus(session: AsyncSession, family_id: str, person_id: str) -> set[str]:
    """Nutrients this person's own record makes relevant, for breaking a near tie on a food line."""
    tgts, _ = await current(session, family_id, person_id)
    facts = await store.facts(session, family_id, person_id, domains=["condition"], statuses=("active",))
    keys = [f.key for f in facts if not str(f.key).endswith(":none")]
    return N.relevant_from(keys, [t["nutrient"] for t in tgts])


def extreme_note(v: dict) -> str | None:
    """Founder Q6: an extreme calorie target is saved, with one line suggesting a check with the doctor."""
    if v.get("field") != "kcal" or v.get("set_by") in ("doctor", "dietitian"):
        return None
    vals = [x for x in (v.get("amount"), v.get("min"), v.get("max")) if x]
    lo, hi = N.EXTREME_KCAL
    if any(x < lo or x > hi for x in vals):
        return (f"Saved. {N.fmt(vals[0])} kcal a day is outside the usual {lo:,}–{hi:,} range: add one gentle line suggesting they check this "
                "target with their doctor (no advice of your own).")
    return None
