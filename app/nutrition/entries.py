"""The food log (plan §N2 nutrition_entries): write, correct and delete entries, each change a memory version (kind "food",
so `undo_change` works on WhatsApp and the dashboard), day totals, counters and streaks.

- Nutrients are computed when an entry is saved and kept on it; a catalogue fix later never changes history silently.
- Each saved meal is also one `meal` event in the day's ledger (what the brain, the brief, the day calendar and the
  reports already read), kept in step when the meal is corrected.
- An elder's own answer about a main meal replaces a caregiver's estimate of the same meal (plan §N4.6, the dose rule).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import delete as sql_delete
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store, versions
from app.care.models import CareEvent
from app.core import clock
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.models import NutritionEntry, NutritionFood, NutritionJob, NutritionPantry

FIELDS = ("meal_id", "meal", "kind", "eaten_at", "day", "food_id", "name", "name_said", "quantity", "unit", "grams", "nutrients",
          "confidence", "source", "source_ref", "photo_ref", "consumed", "portion_eaten", "status", "flags", "note", "subject_id")
MEAL_LABEL = {"breakfast": "Breakfast", "lunch": "Lunch", "snack": "Snack", "dinner": "Dinner",
              "after_dinner": "After dinner", "late_night": "Late night", "other": "Food"}


@dataclass
class Item:
    """One food ready to save."""

    name: str
    food: NutritionFood | None
    quantity: float | None
    unit: str | None
    name_said: str = ""
    confidence: str = "said"  # label | said | photo | estimate | unquantified
    grams: float | None = None  # from a photo reading when the unit gives none
    flags: dict = field(default_factory=dict)
    note: str | None = None
    pantry: NutritionPantry | None = None


def compute(food: NutritionFood | None, quantity: float | None, unit: str | None, *, sizes: dict | None = None, portion: str | None = None,
            grams: float | None = None) -> tuple[float | None, dict, bool]:
    """(grams, nutrients, estimate) for one item. No food or no grams: no nutrients (counted, unquantified)."""
    if food is None:
        return None, {}, True
    estimate = False
    if grams is None or quantity is not None:
        p = P.grams_for(food, quantity, unit, sizes)
        grams, estimate = (p.grams if p.grams is not None else grams), p.estimate
    if grams is None:
        return None, {}, True
    factor = P.PORTION.get(portion or "all", 1.0)
    vals = N.scale(food.per_100g or {}, grams, factor)
    if "water" in (food.tags or []):
        vals["water_ml"] = round(grams * factor, 1)
    return round(grams * factor, 1), vals, estimate


def entry_json(e: NutritionEntry) -> dict:
    n = e.nutrients or {}
    return {"id": str(e.id), "mealId": str(e.meal_id), "meal": e.meal, "kind": e.kind, "eatenAt": e.eaten_at.isoformat(), "day": e.day,
            "name": e.name, "nameSaid": e.name_said, "quantity": e.quantity, "unit": e.unit, "qty": P.qty_label(e.quantity, e.unit),
            "grams": e.grams, "nutrients": n,
            "kcal": None if numbers_pending(e) else N.kcal(n),
            "protein_g": None if numbers_pending(e) else N.grams_of(n, "protein_g"),
            "confidence": "pending" if numbers_pending(e) else e.confidence,
            "source": e.source, "sourceRef": e.source_ref, "photoRef": e.photo_ref, "consumed": e.consumed, "portionEaten": e.portion_eaten,
            "status": e.status, "loggedBy": e.logged_by, "flags": e.flags or {}, "note": e.note, "version": e.version,
            "foodId": str(e.food_id) if e.food_id else None}


def line(e: NutritionEntry) -> str:
    """'4 boiled eggs' / 'dal (1 katori)'."""
    if e.kind == "skipped":
        return f"{e.meal} skipped"
    q = P.qty_label(e.quantity, e.unit)
    if not q:
        return e.name_said or e.name
    if e.unit in (None, "piece"):
        return f"{q} {e.name_said or e.name}"
    return f"{e.name_said or e.name} ({q})"


def snapshot(e: NutritionEntry) -> dict:
    out: dict = {}
    for f in FIELDS:
        v = getattr(e, f)
        out[f] = v.isoformat() if isinstance(v, datetime) else (str(v) if isinstance(v, uuid.UUID) else v)
    return out


def _apply(e: NutritionEntry, snap: dict) -> None:
    for f in FIELDS:
        if f not in snap or f == "subject_id":
            continue
        v = snap[f]
        if f == "eaten_at" and isinstance(v, str):
            v = datetime.fromisoformat(v)
        elif f in ("meal_id", "food_id") and isinstance(v, str):
            v = uuid.UUID(v)
        setattr(e, f, v)


def target(e: NutritionEntry) -> str:
    return f"entry:{e.id}"


async def record(session: AsyncSession, e: NutritionEntry, op: str, *, reason: str = "", undoes: int | None = None):
    n = e.nutrients or {}
    body = f"{line(e)}" + (f", {N.kcal(n)} kcal" if n else "") + f" ({e.meal}, {e.day})" + (" [removed]" if e.status == "deleted" else "")
    return await versions.add(session, family_id=e.family_id, subject_id=e.subject_id, kind="food", target=target(e), op=op,
                              title=e.name[:255], body=body, value=snapshot(e), status=e.status, ref=str(e.id),
                              reason=reason or None, undoes=undoes)


# ── the meal's line in the day's ledger ──────────────────────────────────────


def meal_summary(rows: list[NutritionEntry], relevant: set[str] | None = None) -> str:
    live = [r for r in rows if r.status == "active"]
    if not live:
        return f"{MEAL_LABEL.get(rows[0].meal, 'Food') if rows else 'Food'}: removed from the food log"
    if all(r.kind == "skipped" for r in live):
        return f"{MEAL_LABEL.get(live[0].meal, 'Meal')} skipped"
    food = [r for r in live if r.kind == "food"]
    tot: dict = {}
    for r in food:
        tot = N.add(tot, r.nutrients or {})
    phrase = N.highlight_phrase(tot, relevant)
    nums = f"; about {phrase}" if phrase else ""
    return f"{MEAL_LABEL.get(food[0].meal, 'Food')}: " + ", ".join(line(r) for r in food)[:300] + nums


async def sync_meal_event(session: AsyncSession, family_id: str, meal_id: uuid.UUID) -> None:
    rows = list((await session.execute(select(NutritionEntry).where(NutritionEntry.meal_id == meal_id).order_by(NutritionEntry.created_at))).scalars())
    if not rows:
        return
    first = rows[0]
    tot: dict = {}
    for r in rows:
        if r.status == "active" and r.kind == "food":
            tot = N.add(tot, r.nutrients or {})
    payload = {"nutrition": True, "meal_id": str(meal_id), "meal": first.meal, "entries": [str(r.id) for r in rows if r.status == "active"],
               "kcal": N.kcal(tot), "protein_g": N.grams_of(tot, "protein_g"), "skipped": all(r.kind == "skipped" for r in rows),
               "removed": not any(r.status == "active" for r in rows), "source": first.source}
    ref = f"nutrition:{meal_id}"
    from app.nutrition import targets

    summary = meal_summary(rows, await targets.focus(session, family_id, first.subject_id))
    got = await store.record_event(session, family_id=family_id, subject_id=first.subject_id, kind="meal", summary=summary, payload=payload,
                                   actor_id=first.logged_by if first.logged_by != "saheli" else None, ref=ref, at=first.eaten_at)
    if got is None:  # already there: keep it in step
        from sqlalchemy import func

        await session.execute(update(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == "meal", CareEvent.ref == ref)
                              .values(summary=summary, payload=payload, subject_id=first.subject_id, at=first.eaten_at, day=first.day,
                                      tsv=func.to_tsvector("simple", summary)))


# ── writing ──────────────────────────────────────────────────────────────────


async def day_start_minute(session: AsyncSession, family_id: str, subject_id: str) -> int:
    """When their day starts: the breakfast time learned from their own logs, else 06:00. Not a clock hour on its own."""
    from app.care import store

    fact = await store.active_fact(session, family_id, subject_id, "learned:nudges_meals_times")
    raw = (fact.value or {}).get("breakfast") if fact and isinstance(fact.value, dict) else None
    if not isinstance(raw, str) or ":" not in raw:
        return P.DAY_START
    hour, minute = raw.split(":", 1)
    try:
        got = int(hour) * 60 + int(minute[:2])
    except ValueError:
        return P.DAY_START
    if got < 0 or got > 18 * 60:
        return P.DAY_START
    return got


async def dinner_saved(session: AsyncSession, family_id: str, subject_id: str, day: str, *,
                       except_meal: uuid.UUID | None = None, before: datetime | None = None) -> bool:
    """That diary day already has a dinner (eaten or skipped). Water and a late-night snack do not count.

    except_meal leaves out the meal being placed, so a dinner does not count itself. before counts only an earlier one,
    so a later evening row is after dinner and the first one stays dinner.
    """
    q = select(NutritionEntry.id).where(
        NutritionEntry.family_id == family_id, NutritionEntry.subject_id == subject_id, NutritionEntry.day == day,
        NutritionEntry.meal == "dinner", NutritionEntry.status == "active")
    if except_meal is not None:
        q = q.where(NutritionEntry.meal_id != except_meal)
    if before is not None:
        q = q.where(NutritionEntry.eaten_at < before)
    return (await session.execute(q.limit(1))).first() is not None


async def assign_slot(session: AsyncSession, family_id: str, subject_id: str, at: datetime, *, named: str | None = None,
                      only_water: bool = False) -> P.Slot:
    """The diary slot for this time: their own breakfast time when we know it, and whether dinner is already saved."""
    local = clock.ist(at)
    minute = local.hour * 60 + local.minute
    start = await day_start_minute(session, family_id, subject_id)
    saved = False
    if minute < start or minute >= P.DINNER_FROM:
        food_day = local.date() - (timedelta(days=1) if minute < start else timedelta())
        saved = await dinner_saved(session, family_id, subject_id, food_day.isoformat())
    return P.resolve(at, dinner_saved=saved, only_water=only_water, named=named, day_start=start)


async def _meals_they_moved(session: AsyncSession, refs: list[str]) -> set[str]:
    """Entries whose meal was changed after the save, other than by this re-slot. Those stay as the person left them."""
    from app.care.versions import MemoryVersion

    chosen: set[str] = set()
    for i in range(0, len(refs), 500):
        chunk = refs[i:i + 500]
        got = (await session.execute(select(
            MemoryVersion.ref, MemoryVersion.op, MemoryVersion.value, MemoryVersion.reason).where(
            MemoryVersion.kind == "food", MemoryVersion.ref.in_(chunk)))).all()
        by: dict[str, list[tuple[str, dict, str]]] = {}
        for ref, op, value, reason in got:
            if ref:
                by.setdefault(ref, []).append((op, value or {}, reason or ""))
        for ref, vs in by.items():
            origin = next((v.get("meal") for op, v, _ in vs if op == "write" and v.get("meal")), None)
            for op, v, reason in vs:
                meal = v.get("meal")
                if op == "change" and not reason.startswith("night slot") and origin and meal and meal != origin:
                    chosen.add(ref)
                    break
    return chosen


async def reslot_nights(session: AsyncSession, *, days: int = 7) -> dict:
    """Re-slot the last `days` of meals the clock or Saheli filed, using the same rule as a new log.

    After a saved dinner until midnight is after dinner; before their day starts (else 06:00) is late night, or the
    previous day's dinner when none is saved. A meal the person later moved stays. Through-the-day rows in that window
    move too. Daytime rows stay. Each move is a memory version (reason starts with "night slot") so it can be undone.
    Returns counts and ids, never names.
    """
    since = clock.now() - timedelta(days=max(1, days))
    rows = list((await session.execute(select(NutritionEntry).where(
        NutritionEntry.status == "active", NutritionEntry.eaten_at >= since,
        NutritionEntry.family_id.notlike("shadow:%"), NutritionEntry.family_id.notlike("sim-%")))).scalars())
    groups: dict[uuid.UUID, list[NutritionEntry]] = {}
    for e in rows:
        groups.setdefault(e.meal_id, []).append(e)
    chosen = await _meals_they_moved(session, [str(e.id) for e in rows])
    starts: dict[tuple[str, str], int] = {}
    seen = changed = kept = 0
    moves: list[dict] = []
    ordered = sorted(groups.values(), key=lambda g: min(e.eaten_at for e in g))
    for group in ordered:
        at = min(e.eaten_at for e in group)
        local = clock.ist(at)
        minute = local.hour * 60 + local.minute
        key = (group[0].family_id, group[0].subject_id)
        if key not in starts:
            starts[key] = await day_start_minute(session, key[0], key[1])
        start = starts[key]
        if not (minute < start or minute >= P.DINNER_FROM):
            continue
        seen += 1
        if any(str(e.id) in chosen for e in group):
            kept += 1
            continue
        meal_id = group[0].meal_id
        if minute < start:
            food_day = (local.date() - timedelta(days=1)).isoformat()
            saved = await dinner_saved(session, key[0], key[1], food_day, except_meal=meal_id)
        else:
            saved = await dinner_saved(session, key[0], key[1], local.date().isoformat(), except_meal=meal_id, before=at)
        only_water = all("water" in ((e.flags or {}).get("tags") or []) for e in group)
        slot = P.place(at, dinner_saved=saved, only_water=only_water, day_start=start)
        if all(e.meal == slot.meal and e.day == slot.day for e in group):
            continue
        now = clock.now()
        for e in group:
            old = e.meal
            e.meal, e.day, e.version, e.updated_at = slot.meal, slot.day, e.version + 1, now
            await record(session, e, "change", reason=f"night slot: {old} → {slot.meal}")
            moves.append({"id": str(e.id), "from": old, "to": slot.meal})
            changed += 1
        await session.flush()
        await sync_meal_event(session, key[0], meal_id)
    return {"seen": seen, "changed": changed, "kept": kept, "moves": moves}


async def save_meal(session: AsyncSession, *, family_id: str, subject_id: str, logged_by: str, items: list[Item], meal: str, eaten_at: datetime,
                    source: str, source_ref: str | None = None, job_id: uuid.UUID | None = None, photo_ref: str | None = None,
                    consumed: str = "yes", portion: str | None = None, sizes: dict | None = None, reason: str = "",
                    day: str | None = None) -> list[NutritionEntry]:
    now = clock.now()
    meal_id = uuid.uuid4()
    rows: list[NutritionEntry] = []
    for it in items:
        if it.quantity is None and it.food is not None and it.grams is None:
            # no amount said: the food's usual portion, shown as such ("dal (1 katori)") and marked an estimate
            it.quantity = float(it.food.typical_qty or 1.0)
            it.unit = it.unit or (it.food.default_unit if it.food.default_unit in P.UNITS else None)
            if it.confidence in ("said", "label", "photo"):
                it.confidence = "estimate"
        grams, vals, est = compute(it.food, it.quantity, it.unit, sizes=sizes, portion=portion if consumed == "part" else None, grams=it.grams)
        conf = it.confidence
        if conf == "pending":
            # A slow look-up is not zero. The numbers are filled in after, and the diary says so.
            grams, vals = None, {}
        elif it.food is None or grams is None:
            conf = "unquantified"
        elif est and conf in ("said", "label"):
            conf = "estimate"
        flags = dict(it.flags)
        if it.food is not None and it.food.tags:
            flags["tags"] = list(it.food.tags)
        if it.food is not None and (it.food.checked or {}).get("rough"):
            flags["rough"] = True
        if it.food is not None and (it.food.source == "model" or (it.food.checked or {}).get("rough")):
            flags["refine"] = True  # a web look-up can replace this estimate after the person has been told
        facts = N.fact_tags(it.food.per_100g if it.food is not None else None)
        if facts:
            flags["facts"] = facts
        e = NutritionEntry(
            id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, logged_by=logged_by[:64], meal_id=meal_id, meal=meal, kind="food",
            eaten_at=eaten_at, day=day or clock.ist_day(eaten_at), food_id=it.food.id if it.food else None,
            name=(it.food.name if it.food else it.name)[:200], name_said=(it.name_said or it.name)[:200], quantity=it.quantity,
            unit=it.unit, grams=grams, nutrients=vals, confidence=conf, source=source, source_ref=source_ref, photo_ref=photo_ref,
            consumed=consumed if consumed in ("yes", "part") else "yes", portion_eaten=portion if consumed == "part" else None,
            status="active", job_id=job_id, flags=flags, note=it.note, version=1, created_at=now, updated_at=now)
        session.add(e)
        rows.append(e)
    await session.flush()
    for e in rows:
        await record(session, e, "write", reason=reason)
    await sync_meal_event(session, family_id, meal_id)
    return rows


async def save_skipped(session: AsyncSession, *, family_id: str, subject_id: str, logged_by: str, meal: str, eaten_at: datetime, source: str,
                       source_ref: str | None = None, job_id: uuid.UUID | None = None, day: str | None = None) -> NutritionEntry:
    """They said they skipped this meal: a marker (no food), so the meal is accounted for and the pattern can count it."""
    now = clock.now()
    e = NutritionEntry(id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, logged_by=logged_by[:64], meal_id=uuid.uuid4(), meal=meal,
                       kind="skipped", eaten_at=eaten_at, day=day or clock.ist_day(eaten_at), name=f"{meal} skipped", name_said="", nutrients={},
                       confidence="said", source=source, source_ref=source_ref, status="active", job_id=job_id, flags={}, version=1,
                       created_at=now, updated_at=now)
    session.add(e)
    await session.flush()
    await record(session, e, "write")
    await sync_meal_event(session, family_id, e.meal_id)
    return e


async def replace_caregiver_estimate(session: AsyncSession, family_id: str, subject_id: str, day: str, meal: str, *, by: str) -> list[str]:
    """The elder's own answer about a main meal replaces what a caregiver logged for that same meal (plan §N4.6)."""
    if meal not in P.MAIN_MEALS:
        return []
    rows = list((await session.execute(select(NutritionEntry).where(
        NutritionEntry.family_id == family_id, NutritionEntry.subject_id == subject_id, NutritionEntry.day == day, NutritionEntry.meal == meal,
        NutritionEntry.status == "active", NutritionEntry.source == "caregiver"))).scalars())
    meals = set()
    for e in rows:
        e.status, e.updated_at, e.version = "corrected", clock.now(), e.version + 1
        await record(session, e, "change", reason="replaced by their own answer about this meal")
        meals.add(e.meal_id)
    await session.flush()
    for m in meals:
        await sync_meal_event(session, family_id, m)
    return [line(e) for e in rows]


async def change(session: AsyncSession, e: NutritionEntry, *, quantity: float | None = None, unit: str | None = None,
                 eaten_at: datetime | None = None, meal: str | None = None, food: NutritionFood | None = None, name_said: str | None = None,
                 sizes: dict | None = None, reason: str = "") -> int:
    """A correction ("it was 3 eggs, not 4"): the nutrients follow, and the old version stays for undo. Returns the
    version id (undo_change takes it)."""
    if food is not None:
        e.food_id, e.name = food.id, food.name[:200]
        if name_said:
            e.name_said = name_said[:200]
        flags = dict(e.flags or {})
        flags["tags"] = list(food.tags or [])
        e.flags = flags
    if quantity is not None:
        e.quantity = quantity
    if unit is not None:
        e.unit = unit
    if eaten_at is not None:
        e.eaten_at, e.day = eaten_at, clock.ist_day(eaten_at)
    if meal is not None:
        e.meal = meal
    if quantity is not None or unit is not None or food is not None:
        f = food or (await session.get(NutritionFood, e.food_id) if e.food_id else None)
        grams, vals, est = compute(f, e.quantity, e.unit, sizes=sizes, portion=e.portion_eaten)
        e.grams, e.nutrients = grams, vals
        if f is None or grams is None:
            e.confidence = "unquantified"
        elif e.confidence in ("unquantified", "estimate", "photo") and quantity is not None:
            e.confidence = "estimate" if est or f.source == "model" else "said"
    e.version += 1
    e.updated_at = clock.now()
    await session.flush()
    v = await record(session, e, "change", reason=reason)
    await sync_meal_event(session, e.family_id, e.meal_id)
    return v.id


async def remove(session: AsyncSession, e: NutritionEntry, *, reason: str = "") -> int:
    """Remove an entry from the food log (kept in the history; undo puts it back). Returns the version id."""
    e.status, e.updated_at, e.version = "deleted", clock.now(), e.version + 1
    await session.flush()
    if e.photo_ref:
        # plan §N12: deleting an entry deletes its photo, once no other item of that meal still uses it (undo brings the
        # entry back without it)
        still = (await session.execute(select(NutritionEntry.id).where(NutritionEntry.family_id == e.family_id,
                                                                       NutritionEntry.photo_ref == e.photo_ref,
                                                                       NutritionEntry.status == "active").limit(1))).first()
        if still is None:
            from app.nutrition import photos

            await photos.delete(e.photo_ref)
    v = await record(session, e, "delete", reason=reason)
    await sync_meal_event(session, e.family_id, e.meal_id)
    return v.id


# ── undo and restore (versions kind "food") ──────────────────────────────────


async def _entry_of(session: AsyncSession, v) -> NutritionEntry | None:
    try:
        e = await session.get(NutritionEntry, uuid.UUID(str(v.ref)))
    except ValueError:
        return None
    return e if e and e.family_id == v.family_id else None


async def _state_before(session: AsyncSession, v) -> dict | None:
    prev = await versions.previous(session, v)
    return dict(prev.value or {}) if prev else None


async def undo_or_restore(session: AsyncSession, v, mode: str, *, reason: str = "") -> dict:
    e = await _entry_of(session, v)
    if e is None:
        raise versions.Refused("That food entry is gone.")
    if mode == "undo":
        last = await versions.latest(session, v.family_id, v.subject_id, "food", v.target)
        if last and last.id != v.id:
            raise versions.Refused("It changed again after that; undo the newest change first.")
        before = await _state_before(session, v)
        snap = before if before is not None else {**snapshot(e), "status": "deleted"}
        op, undoes = "undo", v.id
    else:
        snap, op, undoes = dict(v.value or {}), "restore", None
    if all(snapshot(e).get(k) == snap.get(k) for k in FIELDS if k in snap):
        return {"result": "nothing", "why": "the food log already says that"}
    old_meal = e.meal_id
    _apply(e, snap)
    e.version += 1
    e.updated_at = clock.now()
    await session.flush()
    await record(session, e, op, reason=reason or op, undoes=undoes)
    await sync_meal_event(session, e.family_id, e.meal_id)
    if old_meal != e.meal_id:
        await sync_meal_event(session, e.family_id, old_meal)
    return {"result": "done", "entry": entry_json(e), "now": line(e) if e.status == "active" else "removed from the food log"}


def view_bits(v) -> dict:
    val = v.value or {}
    word = {"write": "logged", "change": "corrected", "delete": "removed", "undo": "undone", "restore": "put back"}.get(v.op, v.op)
    return {"label": f"Food log: {v.title}", "body": v.body, "value": val, "summary": f"{word}: {v.body[:140]}"}


async def preview(session: AsyncSession, v, mode: str) -> dict:
    try:
        e = await _entry_of(session, v)
        if e is None:
            raise versions.Refused("That food entry is gone.")
        if mode == "undo":
            last = await versions.latest(session, v.family_id, v.subject_id, "food", v.target)
            if last and last.id != v.id:
                raise versions.Refused("It changed again after that; undo the newest change first.")
            before = await _state_before(session, v)
            snap = before if before is not None else {"status": "deleted"}
        else:
            snap = dict(v.value or {})
        if snap.get("status") == "deleted":
            return ({"effect": "nothing", "text": "Nothing would change: it is already removed.", "button": ""} if e.status == "deleted" else
                    {"effect": "remove", "text": f"This removes '{line(e)}' from the food log. It stays in the history.", "button": "Yes, remove it"})
        q = P.qty_label(N.number(snap.get("quantity")), snap.get("unit"))
        what = f"{snap.get('name_said') or snap.get('name')}" + (f" ({q})" if q else "")
        return {"effect": "change", "text": f"The food log goes back to: {what}, {snap.get('meal')} on {snap.get('day')}.",
                "button": "Yes, undo" if mode == "undo" else "Yes, put back"}
    except versions.Refused as exc:
        return {"effect": "refused", "text": str(exc), "button": ""}


# ── reading ──────────────────────────────────────────────────────────────────


async def between(session: AsyncSession, family_id: str, subject_id: str, first_day: str, last_day: str, *, active: bool = True) -> list[NutritionEntry]:
    q = select(NutritionEntry).where(NutritionEntry.family_id == family_id, NutritionEntry.subject_id == subject_id,
                                     NutritionEntry.day >= first_day, NutritionEntry.day <= last_day)
    if active:
        q = q.where(NutritionEntry.status == "active")
    return list((await session.execute(q.order_by(NutritionEntry.eaten_at, NutritionEntry.created_at))).scalars())


async def on_day(session: AsyncSession, family_id: str, subject_id: str, day: str) -> list[NutritionEntry]:
    return await between(session, family_id, subject_id, day, day)


async def last(session: AsyncSession, family_id: str, subject_id: str, *, within: timedelta = timedelta(days=2)) -> NutritionEntry | None:
    return (await session.execute(select(NutritionEntry).where(
        NutritionEntry.family_id == family_id, NutritionEntry.subject_id == subject_id, NutritionEntry.status == "active",
        NutritionEntry.kind == "food", NutritionEntry.created_at >= clock.now() - within)
        .order_by(NutritionEntry.created_at.desc()).limit(1))).scalar_one_or_none()


async def get(session: AsyncSession, family_id: str, entry_id: str) -> NutritionEntry | None:
    try:
        e = await session.get(NutritionEntry, uuid.UUID(str(entry_id)))
    except ValueError:
        return None
    return e if e and e.family_id == family_id else None


def numbers_pending(e: NutritionEntry) -> bool:
    """True when this food was saved without numbers. Water that already has its ml is not waiting."""
    if e.kind != "food" or e.status != "active":
        return False
    n = e.nutrients or {}
    if "water_ml" in n or "water" in ((e.flags or {}).get("tags") or []):
        return False
    if "energy_kcal" in n:
        return False
    return e.confidence in ("pending", "unquantified") or (e.flags or {}).get("numbers") == "pending"


async def pending_older_than(session: AsyncSession, *, minutes: int = 5) -> int:
    """Food still waiting for numbers, saved more than `minutes` ago. A real zero or a glass of water is not waiting."""
    cutoff = clock.now() - timedelta(minutes=minutes)
    rows = list((await session.execute(select(NutritionEntry).where(
        NutritionEntry.status == "active", NutritionEntry.kind == "food",
        NutritionEntry.created_at < cutoff,
        or_(NutritionEntry.confidence == "pending", NutritionEntry.flags["numbers"].astext == "pending"),
    ))).scalars())
    return sum(1 for e in rows if numbers_pending(e))


def totals(rows: list[NutritionEntry]) -> dict:
    tot: dict = {}
    for r in rows:
        if r.status == "active" and r.kind == "food":
            tot = N.add(tot, r.nutrients or {})
    return tot


def day_days(last_day: str, n: int) -> list[str]:
    d = datetime.fromisoformat(last_day).date()
    return [(d - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def counters(rows: list[NutritionEntry], s: dict | None = None) -> dict:
    """Counts per day-set of entries: eggs, water glasses, tea/coffee cups, sugar teaspoons, fruit portions, protein grams.
    From each food's tags (set by the AI look-up), never from names."""
    glass = float(((s or {}).get("sizes") or {}).get("glass_ml") or 250)
    out = {"egg": 0.0, "water": 0.0, "tea_coffee": 0.0, "sugar": 0.0, "fruit": 0.0, "protein": 0.0}
    for r in rows:
        if r.status != "active" or r.kind != "food":
            continue
        tags = set((r.flags or {}).get("tags") or [])
        n = r.nutrients or {}
        pieces = r.quantity if (r.quantity and r.unit in (None, "piece")) else 1.0
        cups = r.quantity if (r.quantity and r.unit in ("cup", "glass", "piece", None)) else 1.0
        if "egg" in tags:
            out["egg"] += pieces
        if "tea" in tags or "coffee" in tags:
            out["tea_coffee"] += cups
        if "fruit" in tags:
            out["fruit"] += pieces
        if n.get("added_sugar_g") is not None:
            out["sugar"] += float(n["added_sugar_g"]) / 4.0
        elif "sugar" in tags and r.grams:
            out["sugar"] += float(r.grams) / 4.0
        out["water"] += float(n.get("water_ml") or 0) / glass
        out["protein"] += float(n.get("protein_g") or 0)
    return {k: round(v, 1) for k, v in out.items()}


COUNTER_LABELS = {"egg": ("Eggs", ""), "water": ("Water", "glasses"), "tea_coffee": ("Tea / coffee", "cups"), "sugar": ("Sugar", "tsp"),
                  "fruit": ("Fruit", "portions"), "protein": ("Protein", "g")}


def meals_covered(rows: list[NutritionEntry]) -> set[str]:
    return {r.meal for r in rows if r.status == "active"}


def all_meals_streak(by_day: dict[str, list[NutritionEntry]], days: list[str]) -> int:
    """Days running (newest first, today may still be open) with breakfast, lunch and dinner each logged or skipped."""
    n = 0
    for i, d in enumerate(reversed(days)):
        full = set(P.MAIN_MEALS) <= meals_covered(by_day.get(d, []))
        if full:
            n += 1
        elif i == 0:
            continue  # today is not over
        else:
            break
    return n


# ── account deletion ─────────────────────────────────────────────────────────


def _count(res) -> int:
    return int(getattr(res, "rowcount", 0) or 0)


async def erase(session: AsyncSession, family_id: str, subject_id: str | None = None) -> dict:
    """Delete a person's (or the whole family's) food log, questions, pantry, the family's own foods (whole family only),
    the food-log history and stored photos. Generic foods hold no personal data and stay."""
    from app.nutrition import photos
    from app.care.versions import MemoryVersion

    q = select(NutritionEntry.photo_ref).where(NutritionEntry.family_id == family_id, NutritionEntry.photo_ref.is_not(None))
    if subject_id:
        q = q.where(NutritionEntry.subject_id == subject_id)
    for ref in set((await session.execute(q)).scalars()):
        await photos.delete(ref)
    out = {}
    for model in (NutritionEntry, NutritionJob):
        stmt = sql_delete(model).where(model.family_id == family_id)
        if subject_id:
            stmt = stmt.where(model.subject_id == subject_id)
        out[model.__tablename__] = _count(await session.execute(stmt))
    p = sql_delete(NutritionPantry).where(NutritionPantry.family_id == family_id)
    if subject_id:
        p = p.where(NutritionPantry.subject_id == subject_id)
    out["nutrition_pantry"] = _count(await session.execute(p))
    if not subject_id:
        out["nutrition_foods"] = _count(await session.execute(sql_delete(NutritionFood).where(NutritionFood.family_id == family_id)))
    v = sql_delete(MemoryVersion).where(MemoryVersion.family_id == family_id, MemoryVersion.kind == "food")
    if subject_id:
        v = v.where(MemoryVersion.subject_id == subject_id)
    out["memory_versions_food"] = _count(await session.execute(v))
    ev = sql_delete(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == "meal", CareEvent.ref.like("nutrition:%"))
    if subject_id:
        ev = ev.where(CareEvent.subject_id == subject_id)
    out["meal_events"] = _count(await session.execute(ev))
    await session.flush()
    return out
