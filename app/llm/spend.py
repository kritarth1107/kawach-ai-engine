"""What model calls cost, and a daily cap so a runaway can never repeat 3 Oct 2026.

On 3 Oct a simulation ran Vertex AI to ~₹18.6K in a day (normal ~₹313) and Google suspended the project as
"hijacked resources". Every call through app.llm.router is now priced and counted per IST day:

- below LLM_SOFT_CAP_INR: normal.
- above the soft cap: each role uses only its cheapest model, and non-essential roles (judge, sim, extract,
  classify, worker) are refused.
- above LLM_HARD_CAP_INR: only essential calls (a real person writing to Saheli) still go through, on the
  cheapest model. Medicine reminders do not use models, so they keep working.

Prices are estimates per 1M tokens in INR (LLM_PRICES_INR overrides them as JSON {"model": [in, out]}).
Thinking tokens are billed as output. The ledger lives in Postgres (table llm_spend) when configured, so the cap
holds across Cloud Run instances; tests and scripts without a database use the in-process counter only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import date, datetime

from sqlalchemy import BigInteger, Date, Float, Integer, String, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, mapped_column

from app.core import clock
from app.db.session import Base

logger = logging.getLogger(__name__)

USD_INR = 88.0
# USD per 1M tokens (input, output). Estimates; correct them from the invoice with LLM_PRICES_INR.
_USD = {
    "flash": (0.50, 3.00),
    "pro": (2.00, 12.00),
    "opus": (5.00, 25.00),
    "sonnet": (3.00, 15.00),
    "haiku": (1.00, 5.00),
}
CACHE_DISCOUNT = 0.25  # cached input tokens cost about a quarter
ESSENTIAL_ROLES = {"brain", "vision"}
CHEAP_ONLY_NOTE = "spend cap reached: cheapest model only"


class SpendCapReached(Exception):
    """The call is not essential and today's spend is over the cap."""


class LlmSpend(Base):
    __tablename__ = "llm_spend"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    role: Mapped[str] = mapped_column(String(24), primary_key=True)
    model: Mapped[str] = mapped_column(String(80), primary_key=True)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(BigInteger, default=0)
    tokens_out: Mapped[int] = mapped_column(BigInteger, default=0)
    cost_inr: Mapped[float] = mapped_column(Float, default=0.0)


def soft_cap() -> float:
    return float(os.getenv("LLM_SOFT_CAP_INR", "1500"))


def hard_cap() -> float:
    return float(os.getenv("LLM_HARD_CAP_INR", "3000"))


def _family(model: str) -> str:
    m = model.lower()
    for k in ("flash", "haiku", "sonnet", "opus", "pro"):
        if k in m:
            return k
    return "pro"  # unknown: assume the expensive end


def price_inr(model: str) -> tuple[float, float]:
    """(input, output) INR per 1M tokens."""
    override = os.getenv("LLM_PRICES_INR", "").strip()
    if override:
        try:
            table = json.loads(override)
            if model in table:
                a, b = table[model]
                return float(a), float(b)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("LLM_PRICES_INR is not valid JSON {model: [in, out]}")
    i, o = _USD[_family(model)]
    return i * USD_INR, o * USD_INR


def cost_inr(model: str, usage: dict) -> float:
    pin, pout = price_inr(model)
    cached = int(usage.get("cache_read") or 0)
    fresh = max(0, int(usage.get("in") or 0) - cached)
    out = int(usage.get("out") or 0) + int(usage.get("think") or 0)
    return (fresh * pin + cached * pin * CACHE_DISCOUNT + out * pout) / 1_000_000


# ── ledger ─────────────────────────────────────────────────────────────────────

_sessions = None  # async_sessionmaker, set by configure()
_mem: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0, 0, 0, 0.0])  # (day, role, model) -> calls, in, out, inr
_cache: dict[str, tuple[float, float]] = {}  # day -> (total, read at monotonic)
CACHE_SECONDS = 20.0
LEDGER_TIMEOUT = 2.0  # seconds; past this the in-process counter is used


def configure(sessions) -> None:
    """Persist the ledger (and read the cap across instances) through this sessionmaker."""
    global _sessions
    _sessions = sessions


def reset() -> None:
    global _sessions, _process_total
    _sessions = None
    _process_total = 0.0
    _mem.clear()
    _cache.clear()


_process_total = 0.0


def today() -> str:
    # Real wall-clock day (billing is real), not the simulation clock.
    return datetime.now(clock.IST).strftime("%Y-%m-%d")


def process_spent() -> float:
    """Spent by this process since it started (a simulation run checks its own budget with this)."""
    return _process_total


async def record(role: str, model: str, usage: dict) -> float:
    global _process_total
    inr = cost_inr(model, usage)
    _process_total += inr
    day = today()
    row = _mem[(day, role, model)]
    row[0] += 1
    row[1] += int(usage.get("in") or 0)
    row[2] += int(usage.get("out") or 0) + int(usage.get("think") or 0)
    row[3] += inr
    if day in _cache:
        total, at = _cache[day]
        _cache[day] = (total + inr, at)
    if _sessions is not None:
        task = asyncio.create_task(_write(day, role, model, usage, inr))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
    return inr


_pending: set = set()


async def _write(day: str, role: str, model: str, usage: dict, inr: float) -> None:
    """Ledger write in the background, with a short timeout, so a busy connection pool never delays a reply."""
    try:
        async with asyncio.timeout(LEDGER_TIMEOUT):
            async with _sessions() as s:
                stmt = insert(LlmSpend).values(
                    day=date.fromisoformat(day), role=role, model=model, calls=1,
                    tokens_in=int(usage.get("in") or 0), tokens_out=int(usage.get("out") or 0) + int(usage.get("think") or 0), cost_inr=inr,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[LlmSpend.day, LlmSpend.role, LlmSpend.model],
                    set_={"calls": LlmSpend.calls + 1, "tokens_in": LlmSpend.tokens_in + stmt.excluded.tokens_in,
                          "tokens_out": LlmSpend.tokens_out + stmt.excluded.tokens_out, "cost_inr": LlmSpend.cost_inr + stmt.excluded.cost_inr},
                )
                await s.execute(stmt)
                await s.commit()
    except Exception:  # noqa: BLE001 — accounting must never cost a person their reply
        logger.exception("llm spend ledger write failed")


async def flush() -> None:
    """Wait for background ledger writes (tests, shutdown)."""
    if _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


async def spent_today() -> float:
    day = today()
    hit = _cache.get(day)
    if hit and time.monotonic() - hit[1] < CACHE_SECONDS:
        return hit[0]
    total = sum(v[3] for k, v in _mem.items() if k[0] == day)
    if _sessions is not None:
        try:
            async with asyncio.timeout(LEDGER_TIMEOUT), _sessions() as s:
                db_total = (await s.execute(select(func.coalesce(func.sum(LlmSpend.cost_inr), 0.0)).where(LlmSpend.day == date.fromisoformat(day)))).scalar_one()
                total = max(total, float(db_total))
        except Exception:  # noqa: BLE001
            logger.exception("llm spend ledger read failed")
    _cache[day] = (total, time.monotonic())
    return total


async def gate(role: str, essential: bool | None = None) -> str:
    """'ok', 'cheap' (cheapest models first, low effort) or raise SpendCapReached.

    essential: a real person is waiting for this answer. Defaults to the role (brain, vision). A brain call on a
    scheduler turn (wake-up, task update, weekly check-in) passes essential=False: it still runs cheap above the
    soft cap, and stops above the hard cap."""
    in_role = role in ESSENTIAL_ROLES
    essential = in_role if essential is None else essential
    spent = await spent_today()
    if spent >= hard_cap():
        if not essential:
            raise SpendCapReached(f"₹{spent:.0f} spent today, hard cap ₹{hard_cap():.0f}: only people's messages are answered")
        logger.error("LLM HARD CAP: ₹%.0f spent today; answering an essential %s call on the cheapest model", spent, role)
        return "cheap"
    if spent >= soft_cap():
        if not (essential or in_role):
            raise SpendCapReached(f"₹{spent:.0f} spent today, soft cap ₹{soft_cap():.0f}: non-essential calls are paused")
        logger.warning("LLM SOFT CAP: ₹%.0f spent today; %s on the cheapest model", spent, role)
        return "cheap"
    return "ok"


def cheapest(routes: list) -> list:
    """The role's routes, cheapest first. All are kept as fallbacks: a capped day must not also lose failover."""
    return sorted(routes, key=lambda r: price_inr(r.model)[1])


async def summary(sessions, days: int = 7) -> list[dict]:
    async with sessions() as s:
        rows = (await s.execute(
            select(LlmSpend).where(LlmSpend.day >= func.current_date() - days).order_by(LlmSpend.day.desc(), LlmSpend.cost_inr.desc())
        )).scalars()
        return [{"day": r.day.isoformat(), "role": r.role, "model": r.model, "calls": r.calls, "tokensIn": r.tokens_in,
                 "tokensOut": r.tokens_out, "costInr": round(r.cost_inr, 2)} for r in rows]
