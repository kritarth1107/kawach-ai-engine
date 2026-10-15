"""Whether each airline's check-in page is up, read with a plain fetch. No Browser Use credit and no booking is sent.

A page counts as having the form only when its own text shows a booking reference and a last name. A timeout is the result,
not a guess that the form is there.
"""

from __future__ import annotations

import logging

from app.tasks.checkin_desk import fields_present

logger = logging.getLogger(__name__)

PAGES = {
    "indigo": "https://www.goindigo.in/web-check-in.html",
    "airindia": "https://www.airindia.com/in/en/manage/web-check-in.html",
    "airindiaexpress": "https://www.airindiaexpress.com/check-in",
    "akasa": "https://www.akasaair.com/check-in",
    "spicejet": "https://www.spicejet.com/checkin",
}


async def check(fetch=None) -> dict[str, dict]:
    """One row per airline: loaded, and whether the form was in the page text."""
    import httpx

    async def _get(url: str) -> tuple[int, str]:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            return r.status_code, r.text[:80_000]

    get = fetch or _get
    out: dict[str, dict] = {}
    for name, url in PAGES.items():
        try:
            status, text = await get(url)
        except Exception as exc:  # noqa: BLE001 — a site that does not answer is the finding
            logger.warning("check-in page %s: %s", name, exc)
            out[name] = {"loaded": False, "fields": False, "error": type(exc).__name__}
            continue
        out[name] = {"loaded": status == 200, "fields": status == 200 and fields_present(text), "status": status}
    return out
