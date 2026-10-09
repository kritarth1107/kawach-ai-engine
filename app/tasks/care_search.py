"""Home lab tests and doctor appointments: live prices, report times and slots from several sites at once.

Same method as the store searches (app/tasks/fastpath.py): each site's own web API, called with fetch() from a tab on
its origin inside a guest cloud browser (no login, nothing booked). Snippets and how they were found:
agent workspace sitetests/fast/labs_*.md and appt_*.md (2026-10-09). Booking itself needs the family's login on the
chosen site; Saheli shows the options, the family picks, and the booking goes through the same consent rules as orders.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from app.tasks import fastpath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CareSite:
    label: str
    start_url: str
    settle_s: float = 1.0
    ready: str | None = None
    note: str = ""
    solo: bool = False    # needs its tab in front (waits on the page's own token/scripts): run on its own lane
    backup: bool = False  # slow or rate-limited: asked only when fewer than 3 sites answered


LABS: dict[str, CareSite] = {
    "labs_1mg": CareSite("Tata 1mg Labs", "https://www.1mg.com/labs"),
    "labs_healthians": CareSite("Healthians", "https://www.healthians.com/", ready="!!document.querySelector('meta[name=csrf-token]')",
                                note="₹500 minimum booking"),
    "labs_redcliffe": CareSite("Redcliffe Labs", "https://redcliffelabs.com/"),
    "labs_pharmeasy": CareSite("PharmEasy Labs", "https://pharmeasy.in/diagnostics", solo=True),
    "labs_apollo": CareSite("Apollo 24|7 Labs", "https://www.apollo247.com/", solo=True, backup=True),
    "labs_orange": CareSite("Orange Health", "https://www.orangehealth.in/", note="metros only"),
}
DOCTORS: dict[str, CareSite] = {
    "appt_practo": CareSite("Practo", "https://www.practo.com/", ready="/practo/i.test(document.title)"),
    "appt_apollo": CareSite("Apollo 24|7", "https://www.apollo247.com/", solo=True),
}
# what people say → words the sites use (a short query also pulls unrelated tests, so results are filtered by these)
TEST_WORDS = {
    "cbc": ["cbc", "complete blood count", "hemogram", "haemogram"],
    "hba1c": ["hba1c", "glycosylated", "glycated", "a1c"],
    "sugar": ["glucose", "sugar", "fasting blood sugar", "fbs", "ppbs"],
    "thyroid": ["thyroid", "tsh", "t3", "t4"],
    "lipid": ["lipid", "cholesterol"],
    "vitamin d": ["vitamin d", "25 oh", "25-oh", "vit d"],
    "vitamin b12": ["b12", "cobalamin"],
    "kidney": ["kidney", "kft", "rft", "creatinine", "renal"],
    "liver": ["liver", "lft", "sgpt", "sgot"],
    "full body": ["full body", "complete health", "health checkup", "wellness", "full health"],
    "urine": ["urine"],
    "iron": ["iron", "ferritin"],
}


def _wanted(query: str) -> list[str]:
    q = query.lower()
    words = []
    for key, syn in TEST_WORDS.items():
        if key in q or any(s in q for s in syn):
            words += syn
    return words or [w for w in re.findall(r"[a-z0-9]{3,}", q)]


def _matches(name: str, wanted: list[str]) -> bool:
    n = (name or "").lower()
    return any(w in n for w in wanted)


async def _all(sites: dict[str, CareSite], cdp_url: str, values: dict) -> list[dict]:
    """Sites in parallel tabs; 'solo' sites (they wait on the page's own scripts, which a background tab throttles) one
    after another on their own lane at the same time; 'backup' sites only if fewer than 3 answered. A site that failed
    gets one quiet retry on its own (live: 1mg, PharmEasy and Apollo failed in a crowd of tabs and answered alone)."""
    keys = [k for k, s in sites.items() if not s.backup]
    crowd = [k for k in keys if not sites[k].solo]
    lane = [k for k in keys if sites[k].solo]

    async def in_lane() -> dict[str, dict]:
        got = {}
        for k in lane:
            got[k] = await _one(k, sites[k], cdp_url, values)
        return got

    crowd_rows, lane_rows = await asyncio.gather(asyncio.gather(*(_one(k, sites[k], cdp_url, values) for k in crowd)), in_lane())
    got = {**dict(zip(crowd, crowd_rows)), **lane_rows}
    for k in keys:
        if not got[k].get("ok"):
            got[k] = await _one(k, sites[k], cdp_url, values, extra_settle=1.5)
    if sum(1 for k in keys if got[k].get("ok")) < 3:
        for k in (k for k, s in sites.items() if s.backup):
            got[k] = await _one(k, sites[k], cdp_url, values)
    return [got.get(k) or {"site": s.label, "ok": False, "error": "not asked (enough answers)", "skipped": True} for k, s in sites.items()]


async def _one(site_key: str, site: CareSite, cdp_url: str, values: dict, extra_settle: float = 0.0) -> dict:
    js = fastpath._snippet(site_key, values)
    try:
        out = await fastpath.cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s + extra_settle, ready=site.ready, timeout_s=40)
        if isinstance(out, dict) and out.get("status") == 0:
            await asyncio.sleep(1.5)
            out = await fastpath.cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s + 2, ready=site.ready, timeout_s=40)
    except Exception as exc:  # noqa: BLE001 — one site down must not stop the others
        logger.warning("care search %s failed: %s", site_key, exc)
        return {"site": site.label, "ok": False, "error": str(exc)[:120]}
    if not isinstance(out, dict) or out.get("status") != 200:
        logger.warning("care search %s answered %s", site_key, (out.get("status"), out.get("error")) if isinstance(out, dict) else repr(out)[:80])
        return {"site": site.label, "ok": False, "error": f"answered {out.get('status') if isinstance(out, dict) else out!r}"[:120]}
    return {"site": site.label, "ok": True, **out}


async def labs(cdp_url: str, test: str, *, lat=None, lon=None, pincode=None, per_site: int = 3) -> list[dict]:
    """Per lab: serviceable, the best matching tests or packages (price, MRP, fasting, report time) and the earliest slots."""
    wanted = _wanted(test)
    values = {"q": test, "lat": lat, "lon": lon, "pincode": pincode}
    rows = await _all(LABS, cdp_url, values)
    out = []
    for (key, site), r in zip(LABS.items(), rows):
        if r.get("skipped"):
            continue
        if not r.get("ok"):
            out.append({"lab": site.label, "available": None, "problem": r.get("error")})
            continue
        if r.get("serviceable") is False:
            out.append({"lab": site.label, "available": False, "problem": "does not collect at this address"})
            continue
        items = [i for i in r.get("items") or [] if _matches(i.get("name"), wanted)] or (r.get("items") or [])[:1]
        items = sorted(items, key=lambda i: (i.get("kind") != "test", float(i.get("price") or 1e9)))[:per_site]
        out.append({"lab": site.label, "available": True, "note": site.note or None, "slots": (r.get("slots") or [])[:3] or None,
                    "options": [{"name": i.get("name"), "price": fastpath._rupees(i.get("price")), "mrp": fastpath._rupees(i.get("mrp")),
                                 "fasting": i.get("fasting"), "report": i.get("report_time"), "tests": i.get("tests_included"),
                                 "url": i.get("url")} for i in items]})
    return out


async def doctors(cdp_url: str, need: str, *, city: str, lat=None, lon=None, per_site: int = 5) -> list[dict]:
    """Doctors for a specialty or a problem near the family: fee, experience, clinic, next free slot, video or clinic."""
    values = {"q": need, "city": city, "lat": lat, "lon": lon}
    rows = await _all(DOCTORS, cdp_url, values)
    out = []
    for (key, site), r in zip(DOCTORS.items(), rows):
        if not r.get("ok"):
            out.append({"site": site.label, "available": None, "problem": r.get("error")})
            continue
        items = r.get("items") or []
        # in-clinic doctors in the family's city first, then video consults (Apollo tops up with doctors elsewhere)
        where = lambda i: f"{i.get('clinic') or ''} {i.get('area') or ''}".lower()  # noqa: E731
        local = [i for i in items if city.lower() in where(i) or (key == "appt_practo" and not i.get("video_consult"))]
        rest = [i for i in items if i not in local and i.get("video_consult")]  # another city only as a video consult
        out.append({"site": site.label, "available": bool(items), "doctors": [
            {"doctor": i.get("doctor"), "specialty": i.get("specialty"), "years": i.get("experience_years"),
             "fee": fastpath._rupees(i.get("fee")) if i.get("fee") not in (None, 0) else ("free" if i.get("fee") == 0 else None),
             "where": ", ".join(x for x in (i.get("clinic"), i.get("area")) if x) or None, "next_slot": i.get("next_slot"),
             "video": i.get("video_consult"), "url": i.get("url")} for i in (local + rest)[:per_site]]})
    return out


def city_of(place: dict | None) -> str:
    """The family's city from the saved place ('…, Labhandih, Raipur, Chhattisgarh 492001' → Raipur)."""
    p = place or {}
    if p.get("city"):
        return str(p["city"])
    parts = [x.strip() for x in re.split(r",", str(p.get("full") or p.get("nickname") or "")) if x.strip()]
    parts = [re.sub(r"\b\d{6}\b", "", x).strip() for x in parts]
    states = {"chhattisgarh", "maharashtra", "karnataka", "tamil nadu", "west bengal", "uttar pradesh", "madhya pradesh", "rajasthan",
              "gujarat", "delhi", "haryana", "punjab", "telangana", "andhra pradesh", "kerala", "odisha", "bihar", "jharkhand", "assam", "india"}
    for x in reversed(parts):
        if x and x.lower() not in states:
            return x
    return "Raipur"
