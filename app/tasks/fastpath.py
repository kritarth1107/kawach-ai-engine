"""Fixed fast steps for stores whose web app we know, run inside the task's own Browser Use browser over CDP.

The browser agent reads pages and clicks like a person: a Blinkit look-up took about 3 minutes. Blinkit's web app gets
its search results from one request (/v1/layout/search, with the delivery point's latitude and longitude); calling it
from a blinkit.com tab in the same cloud browser (same India proxy, same profile) takes seconds. When anything here
fails or looks wrong, the caller falls back to the browser agent, so a changed site costs time, never a wrong order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx
import websockets

logger = logging.getLogger(__name__)

FAST_SERVICES = {"blinkit"}


class FastPathError(RuntimeError):
    pass


async def cdp_evaluate(cdp_url: str, url: str, expression: str, *, settle_s: float = 3.0, timeout_s: float = 30.0):
    """Open url in a new tab of the cloud browser, run expression (awaited), return its value, close the tab."""
    async def run():
        async with httpx.AsyncClient(timeout=15) as c:
            ws_url = (await c.get(cdp_url.rstrip("/") + "/json/version")).json()["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            counter = 0

            async def send(method: str, params: dict | None = None, session: str | None = None) -> dict:
                nonlocal counter
                counter += 1
                mid = counter
                await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}, **({"sessionId": session} if session else {})}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == mid:
                        if "error" in msg:
                            raise FastPathError(f"{method}: {msg['error']}")
                        return msg.get("result") or {}

            target = (await send("Target.createTarget", {"url": url}))["targetId"]
            try:
                sid = (await send("Target.attachToTarget", {"targetId": target, "flatten": True}))["sessionId"]
                await asyncio.sleep(settle_s)
                out = await send("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True}, session=sid)
                if out.get("exceptionDetails"):
                    raise FastPathError(str(out["exceptionDetails"].get("text") or "script error")[:200])
                return (out.get("result") or {}).get("value")
            finally:
                try:
                    await send("Target.closeTarget", {"targetId": target})
                except Exception:  # noqa: BLE001 — the tab goes with the browser anyway
                    pass

    return await asyncio.wait_for(run(), timeout=timeout_s)


_BLINKIT_SEARCH = """(async () => {
  const r = await fetch('/v1/layout/search?q=' + encodeURIComponent(%(q)s) + '&search_type=type_to_search', {
    method: 'POST', body: '{}',
    headers: {'content-type': 'application/json', 'lat': %(lat)s, 'lon': %(lon)s, 'app_client': 'consumer_web'}});
  if (!r.ok) return {status: r.status};
  const j = await r.json();
  const snips = (j.response && j.response.snippets) || [];
  const t = (x) => (x && (x.text || x.title && x.title.text)) || null;
  return {status: r.status, products: snips.filter(s => s.data && s.data.name).map(s => ({
    name: t(s.data.name), pack: t(s.data.variant), price: t(s.data.normal_price), mrp: t(s.data.mrp),
    stock: typeof s.data.inventory === 'number' ? s.data.inventory : null, id: s.data.identity && s.data.identity.id}))};
})()"""

GENERIC = {"the", "and", "for", "with", "pack", "of", "ml", "ltr", "kg", "gm", "pcs", "soft", "drink"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if (len(w) > 2 or w.isdigit()) and w not in GENERIC}


async def blinkit_search(cdp_url: str, query: str, lat: float, lon: float) -> list[dict]:
    """Products for the query at that point: name, pack, price, in stock, exact_match (every asked word in the name)."""
    js = _BLINKIT_SEARCH % {"q": json.dumps(query), "lat": json.dumps(str(lat)), "lon": json.dumps(str(lon))}
    out = await cdp_evaluate(cdp_url, "https://blinkit.com/", js)
    if not isinstance(out, dict) or out.get("status") != 200 or not isinstance(out.get("products"), list):
        raise FastPathError(f"blinkit search answered {out if not isinstance(out, dict) else out.get('status')}")
    asked = _words(query)
    items = []
    for p in out["products"]:
        name = str(p.get("name") or "")
        if not name or (asked and not asked & _words(name)):
            continue  # the search mixes in unrelated products (chips with cola): keep only ones sharing a word asked for
        has = _words(f"{name} {p.get('pack') or ''}") | _words(re.sub(r"(\d+)\s*(ml|g|kg|l|ltr)", r"\1", str(p.get("pack") or "")))
        items.append({"name": name, "pack": p.get("pack"), "price": p.get("price"), "available": (p.get("stock") or 0) > 0,
                      "exact_match": asked <= has if asked else None, "store_id": p.get("id")})
    return items[:6]
