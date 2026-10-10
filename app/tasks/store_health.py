"""Daily store health check: one guest look-up on each store with a saved look-up step (no login, nothing ordered), so a
store that changed its site is found and fixed before a family's order hits it. Runs in the daily job; one founder
alert lists what broke. Costs one short guest cloud browser (about ₹1-2 a day).
"""

from __future__ import annotations

import asyncio
import logging
import os

from app.tasks import fastpath

logger = logging.getLogger(__name__)

# A reference place and an everyday thing per kind of store (not any family's address).
PLACE = {"lat": float(os.getenv("STORE_HEALTH_LAT", "12.9716")), "lng": float(os.getenv("STORE_HEALTH_LNG", "77.5946")),
         "pincode": os.getenv("STORE_HEALTH_PINCODE", "560001")}
QUERY = {"blinkit": "milk", "instamart": "milk", "zepto": "milk", "apollo": "paracetamol", "1mg": "paracetamol",
         "pharmeasy": "paracetamol", "swiggy": "biryani", "zomato": "biryani"}
RIDE_ENDS = ({"lat": 12.9716, "lng": 77.5946, "label": "MG Road"}, {"lat": 12.9352, "lng": 77.6245, "label": "Koramangala"})


async def _look(service: str, cdp: str) -> str:
    site = fastpath.SITES[service]
    try:
        if site.kind == "ride":
            got = await fastpath.fares(service, cdp, pickup=RIDE_ENDS[0], drop=RIDE_ENDS[1])
            return "ok" if got.get("options") or got.get("login_required") else "no fares"
        got = await fastpath.search(service, cdp, QUERY.get(service, "milk"), lat=PLACE["lat"], lon=PLACE["lng"], pincode=PLACE["pincode"])
        return "ok" if got.get("items") or got.get("deliverable") is False else "no products"
    except Exception as exc:  # noqa: BLE001 — one store's failure is the finding
        return f"failed: {str(exc)[:140]}"


async def check(agent, alert=None, *, retry_wait_s: float = 5.0) -> dict:
    """{store: "ok" | problem}. alert(subject, text) is called once when any store failed twice."""
    if not hasattr(agent, "open_session"):
        return {"skipped": "no cloud browser"}
    results: dict[str, str] = {}
    sid = await agent.open_session(None)
    try:
        cdp = await agent.cdp_url(sid)
        for service in fastpath.SITES:
            results[service] = await _look(service, cdp)
        failed = [k for k, v in results.items() if v != "ok"]
        if failed:  # a cold first page load fails now and then (2026-10-10: instamart 202, 1mg): one more try before alerting
            await asyncio.sleep(retry_wait_s)
            for service in failed:
                results[service] = await _look(service, cdp)
    finally:
        try:
            await agent.stop_session(sid)
        except Exception:  # noqa: BLE001 — the sweeper stops it
            logger.warning("store health: stop session failed")
    broken = {k: v for k, v in results.items() if v != "ok"}
    if broken and alert is not None:
        try:
            await alert("Store check: saved look-up broke on " + ", ".join(broken),
                        "\n".join(f"{k}: {v}" for k, v in broken.items()) + "\n\nOrders on these stores fall back to the slower browser agent until fixed.")
        except Exception:  # noqa: BLE001
            logger.exception("store health alert failed")
    return results
