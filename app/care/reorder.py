"""One offer to reorder something the stock count says will run out within 3 days.

A medicine uses the care record's count and its doses per day. A grocery uses the pantry count and how fast that
same stored product was used, and only when they have ordered that product before (a regular one). The offer is one
question; an order starts only after a yes, through the usual list and confirm, cash on delivery. One offer per
run-out (the count it was based on), and none while an order for that item is still moving. One person hears at most
one offer a day; a closed WhatsApp chat gets the approved order notice, and the question waits.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import slug
from app.care.models import OpenLoop
from app.core import clock
from app.nutrition.models import NutritionEntry, NutritionPantry
from app.tasks.models import Task

OFFER_DAYS = 3
REGULAR_MIN = 2  # placed orders of the same stored product: they order this, not a one-off
USE_WINDOW = timedelta(days=30)
ORDER_WINDOW = timedelta(days=180)


def _items(task: Task) -> list[dict]:
    """Stored product rows on an order (the cart, then what was asked). Never the person's message."""
    d, r = task.details or {}, task.result or {}
    out = []
    for bucket in (r.get("items"), d.get("chosen"), d.get("items")):
        for i in bucket or []:
            if isinstance(i, dict) and str(i.get("name") or "").strip():
                out.append(i)
    return out


def _hit(task: Task, key: str) -> dict | None:
    for i in _items(task):
        if slug(str(i["name"])) == key:
            return i
    return None


async def _owner(session: AsyncSession, family_id: str, subject_id: str) -> str:
    """Who is asked: the person, when it is their own self care; a caregiver when it is for the elder."""
    roster = await store.roster(session, family_id)
    elder = (roster.elder or {}).get("id") if roster else None
    if subject_id and subject_id != elder:
        return subject_id
    members = (roster.members or []) if roster else []
    primary = next((m.get("id") for m in members if m.get("role") == "primary caregiver" and m.get("id")), None)
    other = next((m.get("id") for m in members if m.get("id") and m.get("id") != elder), None)
    return primary or other or elder or subject_id


async def _placed(session: AsyncSession, family_id: str, subject_id: str) -> list[Task]:
    rows = (await session.execute(
        select(Task).where(
            Task.family_id == family_id, Task.subject_id == subject_id, Task.kind == "order",
            Task.created_at >= clock.now() - ORDER_WINDOW,
        ).order_by(Task.created_at.desc()).limit(80)
    )).scalars()
    return [t for t in rows if (t.result or {}).get("placed")]


def _usual(orders: list[Task], key: str, kind: str) -> dict:
    """The last placed order of this stored product: the store, the amount, the name the store used."""
    from app.tasks.skills import SKILLS

    want = "pharmacy" if kind == "medicine" else "grocery"
    for t in orders:
        if SKILLS.get(t.service, {}).get("kind") != want:
            continue
        hit = _hit(t, key)
        if hit is None:
            continue
        label = SKILLS[t.service]["label"]
        return {"service": t.service, "where": label, "qty": int(hit.get("qty") or 1), "item": str(hit["name"])}
    return {}


async def _moving(session: AsyncSession, family_id: str, subject_id: str, key: str) -> bool:
    from app.tasks.runtime import LIVE

    rows = (await session.execute(
        select(Task).where(
            Task.family_id == family_id, Task.subject_id == subject_id, Task.kind == "order",
            Task.status.in_((*LIVE, "scheduled")),
        )
    )).scalars()
    return any(_hit(t, key) for t in rows)


async def _offered(session: AsyncSession, family_id: str, subject_id: str, item_key: str, episode: str) -> bool:
    """An open offer for this item, or any offer already made for this same count (this run-out)."""
    dedupe = f"refill:{subject_id}:{item_key}"
    open_one = (await session.execute(
        select(OpenLoop.id).where(OpenLoop.family_id == family_id, OpenLoop.dedupe_key == dedupe, OpenLoop.status == "open")
    )).first()
    if open_one:
        return True
    same = (await session.execute(
        select(OpenLoop.id).where(
            OpenLoop.family_id == family_id, OpenLoop.subject_id == subject_id, OpenLoop.kind == "refill",
            OpenLoop.detail["key"].astext == item_key, OpenLoop.detail["episode"].astext == episode,
        )
    )).first()
    return same is not None


async def _medicine(session: AsyncSession) -> list[dict]:
    from app.care import features
    from app.care.models import CareFact

    pairs = (await session.execute(
        select(CareFact.family_id, CareFact.subject_id).where(
            CareFact.domain == "medicine", CareFact.status == "active", ~CareFact.family_id.startswith("shadow:"),
        ).distinct()
    )).all()
    out = []
    for fid, sid in pairs:
        orders = await _placed(session, fid, sid)
        for row in await features.stock(session, fid, sid):
            days = row.get("daysLeft")
            if days is None or days > OFFER_DAYS or not row.get("asOf"):
                continue
            key = slug(row["name"])
            usual = _usual(orders, key, "medicine")
            out.append({
                "family_id": fid, "subject_id": sid, "item_key": row["key"], "episode": str(row["asOf"]),
                "name": row["name"], "days": days, "stock": row["stock"], "category": "medicine",
                "item": usual.get("item") or row["name"], **usual,
            })
    return out


def _use_rate(entries: list[NutritionEntry], key: str, unit: str) -> float | None:
    """Pieces (or packs) per day, from logged uses of this stored product. None without two uses."""
    rows = [e for e in entries if slug(e.name) == key and (e.unit or unit) == unit and (e.quantity or 0) > 0]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda e: e.eaten_at)
    span = (rows[-1].eaten_at - rows[0].eaten_at).total_seconds() / 86400
    return sum(float(e.quantity or 0) for e in rows) / max(span, 1.0)


async def _groceries(session: AsyncSession) -> list[dict]:
    from app.tasks.skills import SKILLS

    families = (await session.execute(
        select(NutritionPantry.family_id, NutritionPantry.subject_id).where(
            NutritionPantry.status == "in_stock", NutritionPantry.count > 0,
            ~NutritionPantry.family_id.startswith("shadow:"),
        ).distinct()
    )).all()
    out = []
    for fid, sid in families:
        if not sid:
            continue
        orders = await _placed(session, fid, sid)
        by_key: dict[str, set] = defaultdict(set)
        for t in orders:
            if SKILLS.get(t.service, {}).get("kind") != "grocery":
                continue
            for i in _items(t):
                by_key[slug(str(i["name"]))].add(t.id)
        regular = {k for k, ids in by_key.items() if len(ids) >= REGULAR_MIN}
        if not regular:
            continue
        stock_rows = list((await session.execute(
            select(NutritionPantry).where(
                NutritionPantry.family_id == fid, NutritionPantry.status == "in_stock", NutritionPantry.count > 0,
                or_(NutritionPantry.subject_id == sid, NutritionPantry.subject_id.is_(None)),
            )
        )).scalars())
        grouped: dict[str, list[NutritionPantry]] = defaultdict(list)
        for row in stock_rows:
            grouped[slug(row.name)].append(row)
        entries = list((await session.execute(
            select(NutritionEntry).where(
                NutritionEntry.family_id == fid, NutritionEntry.subject_id == sid, NutritionEntry.status == "active",
                NutritionEntry.eaten_at >= clock.now() - USE_WINDOW,
            ).order_by(NutritionEntry.eaten_at.desc()).limit(400)
        )).scalars())
        for key in regular:
            rows = grouped.get(key) or []
            if not rows:
                continue
            unit = rows[0].unit or "piece"
            have = sum(float(r.count or 0) for r in rows if (r.unit or unit) == unit)
            rate = _use_rate(entries, key, unit)
            if not rate or have < 0:
                continue
            days = math.floor(have / rate)
            if days > OFFER_DAYS:
                continue
            newest = max(rows, key=lambda r: r.added_at)
            usual = _usual(orders, key, "grocery")
            out.append({
                "family_id": fid, "subject_id": sid, "item_key": f"grocery:{key}", "episode": str(newest.id),
                "name": newest.name, "days": days, "stock": have, "category": "grocery",
                "item": usual.get("item") or newest.name, **usual,
            })
    return out


def _today_start():
    """Midnight IST of the current day, so one person is asked at most once before the next morning."""
    day = clock.ist()
    return day.replace(hour=0, minute=0, second=0, microsecond=0)


async def _asked_today(session: AsyncSession, family_id: str, owner_id: str) -> bool:
    row = (await session.execute(
        select(OpenLoop.id).where(
            OpenLoop.family_id == family_id, OpenLoop.owner_id == owner_id, OpenLoop.kind == "refill",
            OpenLoop.created_at >= _today_start(),
        ).limit(1)
    )).first()
    return row is not None


async def open_offers(session: AsyncSession) -> int:
    """Open reorder offers for stock that runs out within 3 days. One per person per day. Returns how many opened."""
    opened = 0
    asked: set[tuple[str, str]] = set()
    offers = [*(await _medicine(session)), *(await _groceries(session))]
    offers.sort(key=lambda o: (o["days"], 0 if o["category"] == "medicine" else 1, o["name"]))
    for offer in offers:
        if await _offered(session, offer["family_id"], offer["subject_id"], offer["item_key"], offer["episode"]):
            continue
        if await _moving(session, offer["family_id"], offer["subject_id"], slug(offer["item"])):
            continue
        owner = await _owner(session, offer["family_id"], offer["subject_id"])
        who = (offer["family_id"], owner)
        if who in asked or await _asked_today(session, offer["family_id"], owner):
            continue
        asked.add(who)
        await store.open_loop(
            session, family_id=offer["family_id"], subject_id=offer["subject_id"], kind="refill",
            title=(f"Reorder {offer['name']}: about {offer['days']} days left ({offer['stock']} left) "
                   f"for person {offer['subject_id']}"),
            detail={
                "key": offer["item_key"], "episode": offer["episode"], "days_left": offer["days"], "stock": offer["stock"],
                "item": offer["item"], "where": offer.get("where") or "no usual store yet",
                "qty": offer.get("qty") if offer.get("qty") is not None else "their usual amount",
                "service": offer.get("service") or "", "category": offer["category"], "pay": "cod", "max_wakes": 1,
            },
            owner_id=owner, wake_at=clock.now(), alert_rule="dashboard",
            dedupe_key=f"refill:{offer['subject_id']}:{offer['item_key']}",
        )
        opened += 1
    return opened
