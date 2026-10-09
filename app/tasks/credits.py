"""Browser service credit: checked every 10 minutes; the founder gets an email before it runs out.

Live 2026-10-09: the Browser Use account reached $0.04 and every order failed until someone noticed. Now: below
BU_CREDIT_WARN_USD (default $3) an ops alert email goes out (at most every 12 hours), and the admin console shows the
balance (GET /v2/agents/credits).
"""

from __future__ import annotations

import logging
import os

from app.care import store
from app.core import clock

logger = logging.getLogger(__name__)

OPS = "ops"


def warn_at() -> float:
    return float(os.getenv("BU_CREDIT_WARN_USD", "3"))


async def balance(agent) -> float | None:
    """USD left on the browser service, or None if it cannot be read."""
    try:
        data = await agent._req("GET", "/billing/account")
        return float(data.get("totalCreditsBalanceUsd"))
    except Exception as exc:  # noqa: BLE001 — a check must never break the sweep
        logger.warning("browser credit check failed: %s", exc)
        return None


async def check(sessions, agent, host) -> dict:
    usd = await balance(agent)
    out = {"usd": usd, "warn_at": warn_at(), "alerted": False}
    if usd is None or usd >= warn_at():
        return out
    now = clock.now()
    half = "am" if clock.ist(now).hour < 12 else "pm"
    async with sessions() as session:
        new = await store.record_event(session, family_id=OPS, subject_id=OPS, kind="credit_low",
                                       summary=f"Browser Use credit ${usd:.2f}", payload={"usd": usd}, ref=f"credit_low:{clock.ist_day(now)}:{half}")
        await session.commit()
    logger.error("BROWSER SERVICE CREDIT LOW: $%.2f left", usd)
    if new is None:  # already told in this half day
        return out
    try:
        await host.call("ops_alert", {
            "subject": f"Browser Use credit low: ${usd:.2f} left",
            "text": (f"The Browser Use account has ${usd:.2f} left (alert below ${warn_at():.2f}). When it reaches zero every "
                     "online order and ride stops. Top up at https://cloud.browser-use.com/settings?tab=billing"),
        }, family_id=OPS, subject_id=OPS, actor_id=OPS)
        out["alerted"] = True
    except Exception:  # noqa: BLE001
        logger.exception("credit alert email failed")
    return out
