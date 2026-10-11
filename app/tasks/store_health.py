"""Daily store health check: one guest look-up on each store with a saved look-up step (no login, nothing ordered), so a
store that changed its site is found and fixed before a family's order hits it. Runs in the daily job; one founder
alert lists what broke. Costs one short guest cloud browser (about ₹1-2 a day).

The reference place moves through several cities, one a day (founder 2026-10-11: test for every city, not one); a store
that fails is tried once more in a fresh browser before anyone is alerted, and the alert carries the store's own error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from app.core import clock
from app.tasks import fastpath

logger = logging.getLogger(__name__)

# City centres (public landmarks, never a family's address): one is checked each day, in turn. STORE_HEALTH_PLACES
# (a JSON list of {city, lat, lng, pincode}) replaces the list; STORE_HEALTH_LAT/LNG pins one place.
CITIES = [
    {"city": "Bengaluru", "lat": 12.9716, "lng": 77.5946, "pincode": "560001"},
    {"city": "Delhi", "lat": 28.6315, "lng": 77.2167, "pincode": "110001"},
    {"city": "Mumbai", "lat": 19.0596, "lng": 72.8295, "pincode": "400050"},
    {"city": "Chennai", "lat": 13.0418, "lng": 80.2341, "pincode": "600017"},
    {"city": "Kolkata", "lat": 22.5726, "lng": 88.3639, "pincode": "700001"},
    {"city": "Hyderabad", "lat": 17.385, "lng": 78.4867, "pincode": "500001"},
    {"city": "Raipur", "lat": 21.2514, "lng": 81.6296, "pincode": "492001"},
]
QUERY = {"blinkit": "milk", "instamart": "milk", "zepto": "milk", "apollo": "paracetamol", "1mg": "paracetamol",
         "pharmeasy": "paracetamol", "swiggy": "biryani", "zomato": "biryani"}


def place_for_today() -> dict:
    """Today's reference place (a different city each day)."""
    if os.getenv("STORE_HEALTH_LAT") and os.getenv("STORE_HEALTH_LNG"):
        return {"city": os.getenv("STORE_HEALTH_CITY", "reference"), "lat": float(os.environ["STORE_HEALTH_LAT"]),
                "lng": float(os.environ["STORE_HEALTH_LNG"]), "pincode": os.getenv("STORE_HEALTH_PINCODE", "")}
    try:
        cities = json.loads(os.getenv("STORE_HEALTH_PLACES", "")) or CITIES
    except ValueError:
        cities = CITIES
    return cities[clock.ist().toordinal() % len(cities)]


def ride_ends(place: dict) -> tuple[dict, dict]:
    """A short ride in the same city: from the centre to a point about 4 km south-east."""
    return ({"lat": place["lat"], "lng": place["lng"], "label": f"{place['city']} centre"},
            {"lat": round(place["lat"] - 0.03, 4), "lng": round(place["lng"] + 0.03, 4), "label": f"{place['city']} south-east"})


async def _look(service: str, cdp: str, place: dict) -> str:
    site = fastpath.SITES[service]
    try:
        if site.kind == "ride":
            pickup, drop = ride_ends(place)
            got = await fastpath.fares(service, cdp, pickup=pickup, drop=drop)
            return "ok" if got.get("options") or got.get("login_required") else "no fares"
        got = await fastpath.search(service, cdp, QUERY.get(service, "milk"), lat=place["lat"], lon=place["lng"], pincode=place.get("pincode"))
        if got.get("deliverable") is False:
            return "ok"  # the store answered properly: it does not serve that point
        return "ok" if got.get("items") else f"no products for '{QUERY.get(service, 'milk')}' (the store answered without an error)"
    except Exception as exc:  # noqa: BLE001 — one store's failure is the finding
        return f"failed: {str(exc)[:160]}"


async def _round(agent, services: list[str], place: dict) -> dict[str, str]:
    sid = await agent.open_session(None)
    try:
        cdp = await agent.cdp_url(sid)
        return {s: await _look(s, cdp, place) for s in services}
    finally:
        try:
            await agent.stop_session(sid)
        except Exception:  # noqa: BLE001 — the sweeper stops it
            logger.warning("store health: stop session failed")


async def check(agent, alert=None, *, retry_wait_s: float = 5.0, place: dict | None = None) -> dict:
    """{store: "ok" | problem} at today's reference place. alert(subject, text) is called once when any store failed in
    two browsers."""
    if not hasattr(agent, "open_session"):
        return {"skipped": "no cloud browser"}
    place = place or place_for_today()
    results = await _round(agent, list(fastpath.SITES), place)
    failed = [k for k, v in results.items() if v != "ok"]
    if failed:
        # A cold first page or one blocked browser fails now and then (2026-10-10: instamart 202, 1mg; 2026-10-11:
        # instamart "no products"): once more in a fresh browser before alerting.
        await asyncio.sleep(retry_wait_s)
        results.update(await _round(agent, failed, place))
    broken = {k: v for k, v in results.items() if v != "ok"}
    if broken and alert is not None:
        try:
            await alert("Store check: saved look-up broke on " + ", ".join(broken),
                        f"Guest look-up near {place['city']} ({place['lat']}, {place['lng']}), tried in two browsers.\n"
                        + "\n".join(f"{k}: {v}" for k, v in broken.items())
                        + "\n\nOrders on these stores fall back to the slower browser agent until fixed.")
        except Exception:  # noqa: BLE001
            logger.exception("store health alert failed")
    return {**results, "_place": place["city"]}
