"""Connector first, browser as fallback.

A store the family has linked (Swiggy, Instamart, Zepto) can be ordered through its official connector:
faster, cheaper, no login codes. Anything the connector cannot do (more than one item, a store it does
not cover, a broken or switched-off connector) goes to the browser agent. Health is tracked per service
and channel; three failures in a row take a channel out for 30 minutes.
"""

from __future__ import annotations

import os

from datetime import datetime, timedelta

from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core import clock
from app.db.session import Base
from app.tasks.models import Task

CONNECTOR_SERVICES = {"swiggy", "instamart", "zepto"}
# Grocery stores take several items in one cart (order lab 2026-10-10: "biscuits and munchies"); food stays one dish.
MULTI_ITEM = {"instamart", "zepto", "swiggy"}  # Swiggy Food: several dishes of one restaurant (live run 2026-10-11)
MAX_ITEMS = 6
DEGRADE_AFTER = 3
DEGRADED_FOR = timedelta(minutes=30)


class ChannelHealth(Base):
    __tablename__ = "agent_channel_health"

    service: Mapped[str] = mapped_column(String(24), primary_key=True)
    channel: Mapped[str] = mapped_column(String(16), primary_key=True)  # connector | browser
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    successes: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    degraded_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


async def _row(session: AsyncSession, service: str, channel: str) -> ChannelHealth:
    row = await session.get(ChannelHealth, (service, channel), with_for_update=True)
    if not row:
        row = ChannelHealth(service=service, channel=channel, consecutive_failures=0, successes=0, failures=0)
        session.add(row)
        await session.flush()
    return row


async def record(session: AsyncSession, service: str, channel: str, ok: bool, error: str | None = None) -> None:
    row = await _row(session, service, channel)
    row.last_at = clock.now()
    if ok:
        row.successes += 1
        row.consecutive_failures = 0
        row.degraded_until = None
    else:
        row.failures += 1
        row.consecutive_failures += 1
        row.last_error = (error or "")[:300]
        if row.consecutive_failures >= DEGRADE_AFTER:
            row.degraded_until = clock.now() + DEGRADED_FOR


async def healthy(session: AsyncSession, service: str, channel: str) -> bool:
    row = await session.get(ChannelHealth, (service, channel))
    return not (row and row.degraded_until and row.degraded_until > clock.now())


def browser_only(family_id: str) -> bool:
    """Ops testing setting BROWSER_ONLY_FAMILIES (comma-separated family ids): their orders skip the store connectors, so
    the browser path can be tested for a family whose stores are linked (repo variable, kept by every deploy)."""
    ids = {x.strip() for x in os.getenv("BROWSER_ONLY_FAMILIES", "").split(",") if x.strip()}
    return family_id.removeprefix("shadow:") in ids


async def pick_channel(session: AsyncSession, host, task: Task) -> tuple[str, str]:
    """(channel, why). Connector only for a linked store, one item, healthy connector."""
    d = task.details or {}
    if browser_only(task.family_id):
        return "browser", "this family's orders use the browser only (testing setting)"
    if task.kind != "order" or task.service not in CONNECTOR_SERVICES:
        return "browser", "no connector for this service"
    n = len(d.get("items") or [])
    if n == 0 or (n > 1 and task.service not in MULTI_ITEM):
        return "browser", "connector orders one item at a time here"
    if n > MAX_ITEMS:
        return "browser", f"more than {MAX_ITEMS} items"
    if not await healthy(session, task.service, "connector"):
        return "browser", "connector degraded after repeated failures"
    if host is None:
        return "browser", "no backend connection"
    try:
        st = await host.call("connector_status", {"store": task.service}, family_id=task.family_id.removeprefix("shadow:"),
                             subject_id=task.subject_id, actor_id=task.requested_by)
    except Exception as exc:  # noqa: BLE001
        return "browser", f"connector status failed: {str(exc)[:80]}"
    if not st.get("connected"):
        return "browser", "store not linked by the family"
    if not st.get("enabled"):
        return "browser", st.get("why") or "connector ordering switched off"
    return "connector", "linked store"


async def connector_search(host, task: Task, item: dict | None = None) -> dict:
    """The look-up through the linked store for one asked item: options with prices, no browser and no login code."""
    item = item or (task.details or {}).get("items")[0]
    out = await host.call(
        "connector_search",
        {"store": task.service, "item": item.get("name"), "restaurant": item.get("restaurant"),
         "placeId": ((task.details or {}).get("limits") or {}).get("place", {}).get("addressId")},
        family_id=task.family_id.removeprefix("shadow:"), subject_id=task.subject_id, actor_id=task.requested_by,
    )
    return out or {}


def _qty_for(items: list[dict], chosen: dict) -> int:
    asked = str(chosen.get("for_item") or "").lower()
    hit = next((i for i in items if str(i.get("name") or "").lower() == asked), None) or (items[0] if len(items) == 1 else None)
    return int((hit or {}).get("qty") or 1)


async def connector_prepare(host, task: Task) -> dict:
    d = task.details or {}
    items = d.get("items") or [{}]
    item = items[0]
    picks = [c for c in (d.get("chosen") or []) if c.get("ref")]  # the products picked from the options, one per item
    first = picks[0] if picks else None
    more = [{"pick": c["ref"], "qty": _qty_for(items, c)} for c in picks[1:]]
    out = await host.call(
        "connector_prepare",
        {"store": task.service, "item": item.get("name"), "qty": _qty_for(items, first) if first else int(item.get("qty") or 1),
         "restaurant": item.get("restaurant"), "placeId": (d.get("limits") or {}).get("place", {}).get("addressId"),
         **({"pick": first["ref"]} if first else {}), **({"more": more} if more else {})},
        family_id=task.family_id.removeprefix("shadow:"), subject_id=task.subject_id, actor_id=task.requested_by,
    )
    return out or {}


async def connector_place(host, task: Task) -> dict:
    out = await host.call(
        "connector_place", {"card": (task.details or {}).get("connector_card")},
        family_id=task.family_id.removeprefix("shadow:"), subject_id=task.subject_id, actor_id=task.requested_by,
    )
    return out or {}


async def health_table(session: AsyncSession) -> list[dict]:
    rows = (await session.execute(select(ChannelHealth).order_by(ChannelHealth.service, ChannelHealth.channel))).scalars()
    return [
        {"service": r.service, "channel": r.channel, "successes": r.successes, "failures": r.failures,
         "consecutiveFailures": r.consecutive_failures, "lastError": r.last_error,
         "degraded": bool(r.degraded_until and r.degraded_until > clock.now()), "lastAt": r.last_at.isoformat() if r.last_at else None}
        for r in rows
    ]
