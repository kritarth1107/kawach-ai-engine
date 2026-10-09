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
from dataclasses import dataclass
from pathlib import Path

import httpx
import websockets

logger = logging.getLogger(__name__)



class FastPathError(RuntimeError):
    pass


async def cdp_evaluate(cdp_url: str, url: str, expression: str, *, settle_s: float = 3.0, timeout_s: float = 30.0,
                       copy_headers: str | None = None, ready: str | None = None):
    """Open url in a new tab of the cloud browser, run expression (awaited), return its value, close the tab.

    copy_headers: a URL pattern. The headers the site's own page sent on its last matching request (app version, device
    and session ids, its login tokens) are put into the expression as %(page_headers)s, so a call made from the snippet
    looks exactly like the app's own, whatever version the site runs. They are never logged or returned."""
    async def run():
        async with httpx.AsyncClient(timeout=15) as c:
            ws_url = (await c.get(cdp_url.rstrip("/") + "/json/version")).json()["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            counter = 0

            seen: dict = {}

            def on_event(msg: dict) -> None:
                if copy_headers and msg.get("method") == "Network.requestWillBeSent":
                    req = (msg.get("params") or {}).get("request") or {}
                    if re.search(copy_headers, req.get("url", "")):
                        seen.update({k.lower(): v for k, v in (req.get("headers") or {}).items()})

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
                    on_event(msg)

            target = (await send("Target.createTarget", {"url": "about:blank" if copy_headers else url}))["targetId"]
            try:
                sid = (await send("Target.attachToTarget", {"targetId": target, "flatten": True}))["sessionId"]
                if copy_headers:
                    await send("Network.enable", {}, session=sid)
                    await send("Page.navigate", {"url": url}, session=sid)
                deadline = asyncio.get_running_loop().time() + settle_s
                while asyncio.get_running_loop().time() < deadline:  # keep reading events while the page loads
                    try:
                        on_event(json.loads(await asyncio.wait_for(ws.recv(), timeout=max(0.05, deadline - asyncio.get_running_loop().time()))))
                    except asyncio.TimeoutError:
                        break
                expr = expression
                if copy_headers:
                    keep = {k: v for k, v in seen.items() if not k.startswith(("sec-", ":")) and k not in ("referer", "user-agent", "content-length", "accept-encoding", "cookie", "origin")}
                    if not keep:
                        raise FastPathError("the page made no request to copy headers from")
                    expr = expr.replace("%(page_headers)s", json.dumps(json.dumps(keep)))
                if ready:  # e.g. a bot check that reloads the page: wait until the real page is there
                    for _ in range(40):
                        try:
                            r = await send("Runtime.evaluate", {"expression": f"!!({ready})", "returnByValue": True}, session=sid)
                            if (r.get("result") or {}).get("value") is True:
                                break
                        except FastPathError:
                            pass
                        await asyncio.sleep(0.5)
                for attempt in range(3):
                    try:
                        out = await send("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True}, session=sid)
                        break
                    except FastPathError as exc:
                        if "navigated" not in str(exc) or attempt == 2:
                            raise
                        await asyncio.sleep(2.0)  # the page reloaded itself (bot check): run again on the new page
                if out.get("exceptionDetails"):
                    raise FastPathError(str(out["exceptionDetails"].get("text") or "script error")[:200])
                return (out.get("result") or {}).get("value")
            finally:
                try:
                    await send("Target.closeTarget", {"targetId": target})
                except Exception:  # noqa: BLE001 — the tab goes with the browser anyway
                    pass

    return await asyncio.wait_for(run(), timeout=timeout_s)


GENERIC = {"the", "and", "for", "with", "pack", "of", "ml", "ltr", "kg", "gm", "pcs", "soft", "drink"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if (len(w) > 2 or w.isdigit()) and w not in GENERIC}


@dataclass(frozen=True)
class Site:
    start_url: str  # the page whose origin the snippet runs on (cookies, same-site requests)
    kind: str = "shop"  # shop | ride
    settle_s: float = 3.0  # wait after opening the page before the snippet runs
    ready: str | None = None  # JS condition that is true once the real page is there (after a bot-check reload)


# One entry per store with a working snippet in app/tasks/fastjs/<service>.js. A snippet returns
# shop: {status, serviceable?, eta?, products: [{name, pack, price, mrp, available, rx_required?, restaurant?, eta?, id}]}
# ride: {status, login_required?, options: [{type, fare, eta}]}
SITES: dict[str, Site] = {
    "blinkit": Site("https://blinkit.com/"),
    "rapido": Site("https://m.rapido.bike/", kind="ride", settle_s=1.5),
    # AWS WAF answers a new browser with a challenge page that reloads into the real one
    # (ready = the real page's own state is there; readyState 'complete' waits for analytics, ~10 s more)
    "instamart": Site("https://www.swiggy.com/instamart", settle_s=1.0, ready="!!window.___INITIAL_STATE___"),
    "swiggy": Site("https://www.swiggy.com/", settle_s=1.0, ready="!!window.___INITIAL_STATE__"),
    "zepto": Site("https://www.zepto.com/", settle_s=1.0, ready="document.cookie.includes('XSRF-TOKEN=') && document.readyState !== 'loading'"),
    "zomato": Site("https://www.zomato.com/", settle_s=3.0),
}
FAST_SERVICES = set(SITES)
JS_DIR = Path(__file__).parent / "fastjs"


def _snippet(service: str, values: dict) -> str:
    js = (JS_DIR / f"{service}.js").read_text()
    # %(name)s tokens only (a snippet may contain other % signs)
    # page_headers is filled later by cdp_evaluate (headers copied from the page's own request)
    return re.sub(r"%\((\w+)\)s", lambda m: m.group(0) if m.group(1) not in values else json.dumps("" if values[m.group(1)] is None else str(values[m.group(1)])), js)


def _rupees(v) -> str | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return f"₹{v:g}"
    t = str(v).strip()
    return t if t.startswith("₹") else (f"₹{t}" if re.fullmatch(r"\d+(\.\d+)?", t) else t)


async def search(service: str, cdp_url: str, query: str, *, lat=None, lon=None, pincode=None) -> dict:
    """A store's own search, run in the task's cloud browser. Returns {deliverable, eta, items}; raises FastPathError when
    the answer is not usable (so the caller falls back to the browser agent)."""
    site = SITES[service]
    js = _snippet(service, {"q": query, "lat": lat, "lon": lon, "pincode": pincode})
    out = await cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s, ready=site.ready, timeout_s=45)
    if not isinstance(out, dict) or out.get("status") != 200:
        raise FastPathError(f"{service} search answered {out.get('status') if isinstance(out, dict) else out!r}"[:160])
    if out.get("serviceable") is False:
        return {"deliverable": False, "eta": None, "items": []}
    if not isinstance(out.get("products"), list):
        raise FastPathError(f"{service} search returned no product list")
    asked = _words(query)
    items = []
    for p in out["products"]:
        name = str(p.get("name") or "").strip()
        label = f"{name} {p.get('restaurant') or ''}"
        if not name or (asked and not asked & _words(label)):
            continue  # searches mix in unrelated products: keep only ones sharing a word asked for
        pack = p.get("pack")
        has = _words(f"{label} {pack or ''}") | _words(re.sub(r"(\d+)\s*(ml|g|kg|l|ltr|mg)", r"\1", str(pack or "")))
        items.append({k: v for k, v in {
            "name": name, "pack": pack, "price": _rupees(p.get("price")), "mrp": _rupees(p.get("mrp")),
            "available": bool(p.get("available", True)), "exact_match": asked <= has if asked else None,
            "rx_required": p.get("rx_required"), "restaurant": p.get("restaurant"), "eta": p.get("eta"), "store_id": p.get("id"),
        }.items() if v is not None})
    return {"deliverable": True, "eta": out.get("eta"), "items": items[:6]}


async def fares(service: str, cdp_url: str, *, pickup: dict, drop: dict) -> dict:
    """Ride options with fares for pickup → drop ({lat, lng, label}), through the service's own request. Raises
    FastPathError when the answer is not usable (the agent looks instead)."""
    site = SITES[service]
    js = _snippet(service, {"plat": pickup["lat"], "plon": pickup["lng"], "dlat": drop["lat"], "dlon": drop["lng"],
                            "pickup": pickup.get("label") or "", "drop": drop.get("label") or ""})
    out = await cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s, timeout_s=40)
    if isinstance(out, dict) and out.get("status") == 0:  # the app was still starting: once more
        out = await cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s + 3, timeout_s=40)
    if not isinstance(out, dict) or out.get("status") != 200:
        raise FastPathError(f"{service} fares answered {out.get('status') if isinstance(out, dict) else out!r}"[:160])
    options = [{"type": o.get("type"), "fare": _rupees(o.get("fare")), "eta": o.get("eta")}
               for o in out.get("options") or [] if o.get("type") and o.get("fare")]
    if not options and not out.get("login_required"):
        raise FastPathError(f"{service} returned no fares")
    return {"login_required": bool(out.get("login_required")) and not options, "options": options}


async def blinkit_search(cdp_url: str, query: str, lat: float, lon: float) -> list[dict]:
    return (await search("blinkit", cdp_url, query, lat=lat, lon=lon))["items"]


CART_SITES = {"blinkit": ("https://blinkit.com/", r"blinkit\.com/v\d")}


async def blinkit_cart(cdp_url: str, products: list[dict], lat=None, lon=None) -> dict:
    """Logged-in Blinkit: build the server cart with these products ({store_id, qty}) at the account's saved address and
    return an agent-style cart report. Raises FastPathError (not logged in, item gone, bad answer) so the agent does it."""
    items = [{"product_id": str(p["store_id"]), "quantity": int(p.get("qty") or 1)} for p in products if p.get("store_id")]
    if not items:
        raise FastPathError("no store product ids to put in the cart")
    js = _snippet("blinkit_cart", {"items": json.dumps(items), "address_id": "", "lat": lat, "lon": lon})
    url, pattern = CART_SITES["blinkit"]
    out = await cdp_evaluate(cdp_url, url, js, settle_s=4.0, copy_headers=pattern)
    if not isinstance(out, dict) or not out.get("logged_in"):
        raise FastPathError("not logged in")
    if out.get("status") != 200 or not out.get("raw"):
        raise FastPathError(f"blinkit cart answered {out.get('status')}: {str(out.get('error'))[:120]}")
    data = json.loads(out["raw"]).get("cart_data") or {}
    bill, addr = data.get("bill_details") or {}, data.get("address") or {}
    lines = [{"name": i.get("name"), "qty": i.get("quantity"), "price": _rupees(i.get("total_price") or i.get("price")),
              "available": not i.get("is_forced_oos") and (i.get("unavailable_quantity") or 0) == 0} for i in data.get("items") or []]
    if len(lines) != len(items) or not all(l["available"] for l in lines):
        raise FastPathError("an item is not available in the cart")
    fees = sum(float(c.get("amount") or 0) for c in data.get("additional_charges") or []) + float(bill.get("delivery_charge") or 0)
    cod = '"value": "cash_on_delivery", "eta": null, "serviceable": true' in json.dumps(json.loads(out["raw"])) or (
        "cash_on_delivery" not in json.dumps(data.get("disabled_payment_modes") or []))
    return {"items": lines, "total": _rupees(bill.get("payable_amount")), "fees": _rupees(fees) if fees else None,
            "cod_available": bool(cod), "logged_in": True, "cart_id": str(json.loads(out["raw"]).get("cart_id") or ""),
            "address_used": f"{addr.get('label') or ''}: {addr.get('line1') or ''}, {addr.get('city') or ''} {addr.get('pincode') or ''}".strip(": ")}
