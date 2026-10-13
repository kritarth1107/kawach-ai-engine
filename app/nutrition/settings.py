"""Who is tracked and how (care fact `setting:nutrition`, domain `setting`).

Founder Q1: everyone is tracked by default (no fact = full tracking); anyone can turn it off for themselves, and a
caregiver for the person they look after. Founder Q3: Saheli asks adaptively by default (Kavach NUdge reads `ask`).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store

KEY = "setting:nutrition"
TRACKING = ("off", "counts", "full")
ASK = ("adaptive", "per_meal", "recap", "off")
SIZES = ("katori_ml", "bowl_ml", "cup_ml", "glass_ml", "plate_g")
COUNTERS = ("egg", "water", "tea_coffee", "sugar", "fruit", "protein")
DEFAULT = {"tracking": "full", "ask": "adaptive", "daily_summary_to": [], "sizes": {}, "counters": list(COUNTERS)}


def normalise(value: dict | None) -> dict:
    v = dict(value or {})
    out = dict(DEFAULT)
    if v.get("tracking") in TRACKING:
        out["tracking"] = v["tracking"]
    if v.get("ask") in ASK:
        out["ask"] = v["ask"]
    out["daily_summary_to"] = [str(x)[:64] for x in v.get("daily_summary_to") or [] if x][:6]
    sizes = {}
    for k, x in (v.get("sizes") or {}).items():
        try:
            n = float(x)
        except (TypeError, ValueError):
            continue
        if k in SIZES and 20 <= n <= 1000:
            sizes[k] = n
    out["sizes"] = sizes
    counters = [c for c in v.get("counters") or [] if c in COUNTERS]
    out["counters"] = counters or list(COUNTERS)
    return out


async def get(session: AsyncSession, family_id: str, person_id: str) -> dict:
    """The person's setting with every default filled in (no fact: full tracking, adaptive asks)."""
    row = await store.active_fact(session, family_id, person_id, KEY)
    return normalise(row.value if row else None)


def tracked(s: dict) -> bool:
    return s.get("tracking") != "off"


def sentence(s: dict) -> str:
    level = {"off": "Food tracking is off", "counts": "Saheli keeps count of food (counts only)",
             "full": "Saheli keeps track of food and nutrition"}[s["tracking"]]
    ask = {"adaptive": "asks after meals and learns when", "per_meal": "asks after each main meal not logged",
           "recap": "asks once in the evening", "off": "does not ask about meals"}[s["ask"]]
    sizes = "; own sizes: " + ", ".join(f"{k.split('_')[0]} {v:g} {k.split('_')[1]}" for k, v in s["sizes"].items()) if s["sizes"] else ""
    return f"{level}; {ask}{sizes}"


async def save(session: AsyncSession, *, family_id: str, person_id: str, changes: dict, source_kind: str, stated_by: str | None,
               source_ref: str | None = None) -> tuple[str, dict]:
    cur = await get(session, family_id, person_id)
    merged = dict(cur)
    for k in ("tracking", "ask", "daily_summary_to", "counters"):
        if changes.get(k) not in (None, "", []):
            merged[k] = changes[k]
    if isinstance(changes.get("sizes"), dict):
        merged["sizes"] = {**cur.get("sizes", {}), **changes["sizes"]}
    value = normalise(merged)
    w = await store.write_fact(session, family_id=family_id, subject_id=person_id, domain="setting", key=KEY, value=value,
                               text=sentence(value), source_kind=source_kind, source_ref=source_ref, stated_by=stated_by, replace=True)
    return w.result, value
