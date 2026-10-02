"""One clock for the brain and memory, so a simulation can run a month of a family's life in minutes."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_frozen: ContextVar[datetime | None] = ContextVar("kavach_clock", default=None)


def now() -> datetime:
    return _frozen.get() or datetime.now(timezone.utc)


def set_now(at: datetime | None):
    """Pin the clock for this task (sim runs and tests). Returns a token for reset_now."""
    if at is not None and at.tzinfo is None:
        raise ValueError("clock needs an aware datetime")
    return _frozen.set(at)


def reset_now(token) -> None:
    _frozen.reset(token)


def ist(at: datetime | None = None) -> datetime:
    return (at or now()).astimezone(IST)


def ist_day(at: datetime | None = None) -> str:
    return ist(at).strftime("%Y-%m-%d")
