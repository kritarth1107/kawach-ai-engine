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


class ItemsUnavailable(FastPathError):
    """Products the store's cart refused (not sold at the serving store, out of stock): ids = their store ids."""

    def __init__(self, ids: list[str]):
        super().__init__(f"an item is not available in the cart ({', '.join(ids) or 'unknown'})")
        self.ids = ids


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


# Words a store's product name often leaves out ("Parle G Glucose" for "Parle-G biscuit"): not needed for an exact match
GENERIC = {"the", "and", "for", "with", "pack", "of", "ml", "ltr", "kg", "gm", "pcs", "soft", "drink", "biscuit", "biscuits",
           "cookie", "cookies", "tablet", "tablets", "strip", "capsule", "capsules", "packet", "pouch", "bottle", "can", "cans"}


def _words(text: str) -> set[str]:
    # "650mg" counts as 650 (numbers and letters split)
    return {w for w in re.findall(r"[a-z]+|\d+", str(text or "").lower()) if (len(w) > 2 or w.isdigit()) and w not in GENERIC}


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
    # pharmacies search by pincode
    "apollo": Site("https://www.apollopharmacy.in/", settle_s=1.5),
    "1mg": Site("https://www.1mg.com/", settle_s=1.0),
    "pharmeasy": Site("https://pharmeasy.in/", settle_s=1.5),
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


async def search(service: str, cdp_url: str, query: str, *, lat=None, lon=None, pincode=None, limit: int = 6) -> dict:
    """A store's own search, run in the task's cloud browser. Returns {deliverable, eta, items}; raises FastPathError when
    the answer is not usable (so the caller falls back to the browser agent)."""
    site = SITES[service]
    js = _snippet(service, {"q": query, "lat": lat, "lon": lon, "pincode": pincode})
    out = await cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s, ready=site.ready, timeout_s=45)
    if isinstance(out, dict) and out.get("status") == 0:
        # the page was not really there yet (a bot-check reload with old cookies in the profile): once more
        await asyncio.sleep(2.0)
        out = await cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s + 3, ready=site.ready, timeout_s=45)
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
            # ids a logged-in cart step needs (Instamart: product, variant, item)
            "cart_ref": p.get("cart_item") or {k: p[k] for k in ("product_id", "spin", "item_id") if p.get(k)} or None,
        }.items() if v is not None})
    return {"deliverable": True, "eta": out.get("eta"), "items": items[:limit], "logged_in": bool(out.get("logged_in"))}


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


CART_SITES = {"blinkit": ("https://blinkit.com/", r"blinkit\.com/v\d"), "instamart": ("https://www.swiggy.com/instamart", None)}


async def cart(service: str, cdp_url: str, products: list[dict], *, place: dict | None = None, lat=None, lon=None) -> dict:
    """The store's logged-in cart step for these picked products; raises FastPathError so the agent does it instead."""
    if service == "instamart":
        return await instamart_cart(cdp_url, products, place or {})
    return await blinkit_cart(cdp_url, products, lat=lat, lon=lon)


async def instamart_cart(cdp_url: str, products: list[dict], place: dict) -> dict:
    """Logged-in Swiggy Instamart (order lab 2026-10-10, recorded from the web app): the family place among the account's
    saved addresses, select it, sync the cart to exactly these products, read the bill and COD. An agent-style report."""
    items = [{"product_id": r.get("product_id"), "spin": r.get("spin"), "item_id": r.get("item_id"), "qty": int(p.get("qty") or 1)}
             for p in products for r in [p.get("cart_ref") or {}] if r.get("spin") and r.get("item_id")]
    if not items or len(items) != len(products):
        raise FastPathError("no Instamart cart ids for every product")
    if not place.get("pincode"):
        raise FastPathError("no delivery pincode to find the saved address")
    js = _snippet("instamart_cart", {"items": json.dumps(items),
                                     "place": json.dumps({k: place.get(k) for k in ("pincode", "line1", "full")})})
    site = SITES["instamart"]
    out = await cdp_evaluate(cdp_url, site.start_url, js, settle_s=site.settle_s, ready=site.ready, timeout_s=40)
    if not isinstance(out, dict) or not out.get("logged_in"):
        raise FastPathError("not logged in")
    if out.get("problem") == "address_missing":
        raise FastPathError(f"the family address is not saved on the account ({out.get('addresses_seen')} saved)")
    if out.get("problem") or out.get("error"):
        raise FastPathError(f"instamart cart: {out.get('problem') or out.get('error')}")
    lines = out.get("lines") or []
    want = {i["item_id"]: i["qty"] for i in items}
    got = {l.get("item_id"): l.get("qty") for l in lines}
    if got != want or not all(l.get("available") for l in lines) or out.get("unavailable"):
        raise FastPathError(f"the cart does not match the picked items ({got} vs {want}; {out.get('unavailable') or ''})")
    if not out.get("address_matches"):
        raise FastPathError("the cart is not on the family address")
    # cash on delivery, read on the cart page (the app sets its session headers there); it also shows the same cart
    cod = await cdp_evaluate(cdp_url, CART_PAGE, _snippet("instamart_cod", {}), settle_s=2.0, ready=site.ready, timeout_s=40)
    if not isinstance(cod, dict) or cod.get("items") != want:
        raise FastPathError(f"the cart page shows a different cart ({(cod or {}).get('items') if isinstance(cod, dict) else cod})")
    out["cod_available"] = cod.get("cod_available")
    fee_total = round((out.get("total") or 0) - (out.get("item_total") or 0))
    fees = ", ".join(f"{f['label']} ₹{round(f['value'])}" for f in out.get("fees") or [])
    fees = f"₹{fee_total} ({fees})" if fee_total > 0 and fees else (f"₹{fee_total}" if fee_total > 0 else "none")
    return {"items": [{"name": l["name"], "qty": l["qty"], "price": _rupees(round(l["price"])) if l.get("price") else None, "available": True} for l in lines],
            "total": _rupees(round(out.get("total") or 0)) if out.get("total") else None, "fees": fees or None,
            # None = could not be read: the place step then finds out (the store refuses cash it does not offer)
            "cod_available": out.get("cod_available"), "logged_in": True, "cart_id": out.get("cart_id") or "",
            "address_used": out.get("address_used"), "eta": out.get("eta") or None,
            # what the place step re-checks before it sends the order
            "place_check": {"items": want, "total": out.get("total"), "address_id": out.get("address_id")}}


PLACE_SITES = {"instamart", "blinkit"}
CART_PAGE = "https://www.swiggy.com/instamart/cart"


async def place(service: str, cdp_url: str, check: dict) -> dict:
    """Place the confirmed cart, cash on delivery, with the store's own requests. Returns an agent-style report:
    {placed, order_id, total, payment_method, eta} | {placed: False, unclear: True} (sent, no clear answer: never retry) |
    raises FastPathError (nothing was sent: the cart changed or the page could not be used)."""
    if service == "blinkit":
        out = await blinkit_place(cdp_url, check)
        if out.get("unclear") and check.get("total"):
            # Pay Now was pressed but no order page came: look in the account's order list before calling it unclear
            # (lab 2026-10-10: order 2 went through and was reported unclear).
            for wait in (0, 8):
                await asyncio.sleep(wait)
                try:
                    found = await blinkit_recent_order(cdp_url, check["total"])
                except Exception as exc:  # noqa: BLE001 — still unclear: never retried
                    logger.warning("blinkit order list failed: %s", exc)
                    found = None
                if found:
                    return {"placed": True, "order_id": found, "total": _rupees(check.get("total")), "payment_method": "Cash on Delivery"}
        return out
    if service != "instamart":
        raise FastPathError(f"no place step for {service}")
    if not (check or {}).get("items") or not check.get("total") or not check.get("address_id"):
        raise FastPathError("nothing to check the cart against")
    js = _snippet("instamart_place", {"want": json.dumps(check)})
    site = SITES["instamart"]
    try:
        out = await cdp_evaluate(cdp_url, CART_PAGE, js, settle_s=2.0, ready=site.ready, timeout_s=50)
    except FastPathError:
        raise
    except Exception as exc:  # noqa: BLE001 — the tab broke: we cannot tell whether the order call left the page
        return {"placed": False, "unclear": True, "problem": f"the browser stopped while placing ({str(exc)[:80]}); check the app before trying again"}
    if not isinstance(out, dict):
        return {"placed": False, "unclear": True, "problem": "no answer from the place step; check the app before trying again"}
    if not out.get("sent"):
        raise FastPathError(f"not placed: {out.get('problem') or out.get('error') or 'cart check failed'}")
    if out.get("placed") and out.get("order_id"):
        return {"placed": True, "order_id": out["order_id"], "total": _rupees(round(out.get("total") or 0)), "payment_method": "Cash on Delivery",
                "eta": out.get("eta")}
    return {"placed": False, "unclear": True, "order_id": out.get("order_id") or None,
            "problem": f"the store's answer was not clear ({out.get('status') or out.get('problem') or out.get('error')}); check the app before trying again"}


async def blinkit_cart(cdp_url: str, products: list[dict], lat=None, lon=None) -> dict:
    """Logged-in Blinkit: build the server cart with these products ({store_id, qty}) at the account's saved address and
    return an agent-style cart report. Raises FastPathError (not logged in, item gone, bad answer) so the agent does it."""
    items = [{"product_id": str(p["store_id"]), "quantity": int(p.get("qty") or 1)} for p in products if p.get("store_id")]
    if not items:
        raise FastPathError("no store product ids to put in the cart")
    local = [{**(p.get("cart_ref") or {}), "quantity": int(p.get("qty") or 1)} for p in products if (p.get("cart_ref") or {}).get("product_id")]
    js = _snippet("blinkit_cart", {"items": json.dumps(items), "address_id": "", "lat": lat, "lon": lon,
                                   "local": json.dumps(local if len(local) == len(items) else [])})
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
        got = {str(i.get("product_id")): i for i in data.get("items") or []}
        bad = [it["product_id"] for it in items if it["product_id"] not in got
               or got[it["product_id"]].get("is_forced_oos") or (got[it["product_id"]].get("unavailable_quantity") or 0) > 0]
        raise ItemsUnavailable(bad or [it["product_id"] for it in items] if len(items) == 1 else bad)
    fees = sum(float(c.get("amount") or 0) for c in data.get("additional_charges") or []) + float(bill.get("delivery_charge") or 0)
    cod = '"value": "cash_on_delivery", "eta": null, "serviceable": true' in json.dumps(json.loads(out["raw"])) or (
        "cash_on_delivery" not in json.dumps(data.get("disabled_payment_modes") or []))
    report = {"items": lines, "total": _rupees(bill.get("payable_amount")), "fees": _rupees(fees) if fees else None,
              "cod_available": bool(cod), "logged_in": True, "cart_id": str(json.loads(out["raw"]).get("cart_id") or ""),
              "address_used": f"{addr.get('label') or ''}: {addr.get('line1') or ''}, {addr.get('city') or ''} {addr.get('pincode') or ''}".strip(": ")}
    if out.get("local_written"):
        # the place step checks the checkout page against this before it presses Pay Now
        report["place_check"] = {"count": sum(i["quantity"] for i in items), "prices": sorted(int(round(float(re.sub(r"[^\d.]", "", str(l.get("price") or 0)) or 0))) for l in lines),
                                 "total": int(round(float(bill.get("payable_amount") or 0))), "address": str(addr.get("line1") or "")[:40]}
    return report



def blinkit_orders_from(body, now=None) -> list[dict]:
    """Blinkit's order list (layout answer) → [{order_id, status, amount, minutes_ago}] for orders placed today."""
    from datetime import datetime, timedelta, timezone

    ist = timezone(timedelta(hours=5, minutes=30))
    now = (now or datetime.now(timezone.utc)).astimezone(ist)
    found: list[dict] = []

    def walk(o) -> None:
        if isinstance(o, dict):
            attrs = ((o.get("tracking") or {}).get("common_attributes") or {}) if isinstance(o.get("tracking"), dict) else {}
            oid = str(attrs.get("order_id") or "")
            if oid and not any(f["order_id"] == oid for f in found):
                text = " ".join(_texts(o))
                amount = re.search(r"₹\s*([\d,]+)", text)
                when = re.search(r"Today,\s*(\d{1,2}):(\d{2})\s*([ap]m)", text, re.I)
                minutes = None
                if when:
                    h = int(when.group(1)) % 12 + (12 if when.group(3).lower() == "pm" else 0)
                    at = now.replace(hour=h, minute=int(when.group(2)), second=0, microsecond=0)
                    minutes = int((now - at).total_seconds() // 60)
                found.append({"order_id": oid, "status": attrs.get("order_status"), "minutes_ago": minutes,
                              "amount": int(amount.group(1).replace(",", "")) if amount else None})
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(body)
    return found


async def blinkit_recent_order(cdp_url: str, total, within_min: int = 15) -> str | None:
    """The id of an order placed in the last few minutes for this total (no other order of that total then), else None."""
    out = None
    for settle in (7.0, 12.0):  # a fresh browser's first page can be slow to make its own requests
        try:
            out = await cdp_evaluate(cdp_url, "https://blinkit.com/account/orders", (JS_DIR / "blinkit_orders.js").read_text(),
                                     settle_s=settle, timeout_s=60, copy_headers=r"blinkit\.com/v1/")
            break
        except FastPathError:
            continue
    if not isinstance(out, dict) or out.get("status") != 200:
        return None
    want = round(float(re.sub(r"[^\d.]", "", str(total)) or 0))
    hits = [o for o in blinkit_orders_from(out.get("body")) if o["amount"] == want and o["minutes_ago"] is not None
            and 0 <= o["minutes_ago"] <= within_min and str(o.get("status") or "").upper() not in ("CANCELLED", "FAILED")]
    return hits[0]["order_id"] if len(hits) == 1 else None


def _texts(o, out: list | None = None) -> list[str]:
    """Every "text" string in a layout answer, in order."""
    out = [] if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "text" and isinstance(v, str):
                out.append(v)
            else:
                _texts(v, out)
    elif isinstance(o, list):
        for v in o:
            _texts(v, out)
    return out


_ETA_TEXT = re.compile(r"\b(?:arriving|arrives|reaching|delivering|on (?:the|its) way)\b[^.\n]{0,30}?\bin\s+(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*(?:min|mins|minutes)\b", re.I)
_AWAY = re.compile(r"\b(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*(?:min|mins|minutes)\s+away\b", re.I)


def blinkit_eta(texts: list[str]) -> str | None:
    """When a Blinkit order will come, from its order page ("Arriving in 9 minutes", "8 mins away"). Not the page header's
    store time ("Delivery in 10 minutes"), and nothing once it has arrived."""
    for t in texts:
        if re.search(r"\barrived\b|\bdelivered\b", t, re.I):
            continue
        m = _ETA_TEXT.search(t) or _AWAY.search(t)
        if m:
            return f"{m.group(1)}-{m.group(2)} min" if m.group(2) else f"{m.group(1)} min"
    return None


async def blinkit_place(cdp_url: str, check: dict, *, dry: bool = False) -> dict:
    """Blinkit: the place call goes out from Zomato's payment frame (zomato.com/zpaykit, cross-origin), so this opens the
    checkout page, checks the cart lines and address, makes Cash the open option inside that frame (its own CDP session),
    then presses the page's Pay Now. Same report shape as place()."""
    if not (check or {}).get("count") or not check.get("prices"):
        raise FastPathError("nothing to check the checkout page against")
    want = json.dumps({k: check.get(k) for k in ("count", "prices", "address")})
    js_check = _snippet("blinkit_checkout_check", {"want": want})
    js_frame, js_press = (JS_DIR / "blinkit_cash_frame.js").read_text(), (JS_DIR / "blinkit_paynow.js").read_text()
    pressed = False

    async def run():
        nonlocal pressed
        async with httpx.AsyncClient(timeout=15) as c:
            ws_url = (await c.get(cdp_url.rstrip("/") + "/json/version")).json()["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            counter, frames = 0, {}  # targetId -> {"session", "url"}
            track = {}  # the order page's own tracking answer (crystal_track_order): requestId, done

            def on_event(msg: dict) -> None:
                m, p = msg.get("method"), msg.get("params") or {}
                if m == "Network.responseReceived" and "crystal_track_order" in ((p.get("response") or {}).get("url") or ""):
                    track["id"] = p.get("requestId")
                elif m == "Network.loadingFinished" and p.get("requestId") and p.get("requestId") == track.get("id"):
                    track["done"] = True
                elif m == "Target.attachedToTarget":
                    ti = p.get("targetInfo") or {}
                    frames[ti.get("targetId")] = {"session": p.get("sessionId"), "url": ti.get("url", "")}
                elif m == "Target.targetInfoChanged":
                    ti = p.get("targetInfo") or {}
                    if ti.get("targetId") in frames:
                        frames[ti["targetId"]]["url"] = ti.get("url", "")

            async def send(method: str, params: dict | None = None, session: str | None = None, timeout: float = 30) -> dict:
                nonlocal counter
                counter += 1
                mid = counter
                await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}, **({"sessionId": session} if session else {})}))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                    if msg.get("id") == mid:
                        if "error" in msg:
                            raise FastPathError(f"{method}: {msg['error']}")
                        return msg.get("result") or {}
                    on_event(msg)

            async def evaluate(expr: str, session: str, timeout: float = 30):
                r = await send("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True}, session=session, timeout=timeout)
                if r.get("exceptionDetails"):
                    raise FastPathError(str(r["exceptionDetails"].get("text") or "script error")[:200])
                return (r.get("result") or {}).get("value")

            async def order_page_eta() -> str | None:
                for _ in range(20):  # up to ~10 s for the page's tracking answer
                    if track.get("done"):
                        break
                    await asyncio.sleep(0.5)
                    await send("Target.getTargetInfo", {"targetId": target}, timeout=10)  # lets events in
                if track.get("done"):
                    got = await send("Network.getResponseBody", {"requestId": track["id"]}, session=sid, timeout=10)
                    raw = got.get("body") or ""
                    if got.get("base64Encoded"):
                        import base64
                        raw = base64.b64decode(raw).decode("utf-8", "replace")
                    try:
                        eta = blinkit_eta(_texts(json.loads(raw)))
                    except ValueError:
                        eta = blinkit_eta([raw])
                    if eta:
                        return eta
                text = await evaluate("document.body ? document.body.innerText : ''", sid, timeout=10)
                return blinkit_eta([str(text or "")])

            target = (await send("Target.createTarget", {"url": "about:blank"}))["targetId"]
            try:
                sid = (await send("Target.attachToTarget", {"targetId": target, "flatten": True}))["sessionId"]
                await send("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True}, session=sid)
                await send("Page.enable", {}, session=sid)
                await send("Network.enable", {}, session=sid)
                await send("Page.navigate", {"url": "https://blinkit.com/checkout"}, session=sid)
                await asyncio.sleep(3.0)
                page = await evaluate(js_check, sid, timeout=40)
                if not (page or {}).get("ok"):
                    return {"clicked": False, "problem": (page or {}).get("problem") or "checkout check failed", "page": (page or {}).get("page")}
                pay = None
                for _ in range(40):  # the payment frame attaches as its own target
                    pay = next((f for f in frames.values() if "zpaykit" in (f.get("url") or "")), None)
                    if pay:
                        break
                    try:
                        on_event(json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5)))
                    except asyncio.TimeoutError:
                        pass
                if not pay:
                    return {"clicked": False, "problem": "the payment frame did not load"}
                cash = await evaluate(js_frame, pay["session"], timeout=20)
                if not (cash or {}).get("cash"):
                    return {"clicked": False, "problem": "cash on delivery could not be selected", "frame": cash}
                if dry:
                    return {"clicked": False, "ready": True, "page": page.get("page")}
                pressed = True
                press = await evaluate(js_press, sid, timeout=20)
                if not (press or {}).get("clicked"):
                    pressed = False
                    return {"clicked": False, "problem": (press or {}).get("problem") or "Pay Now not found"}
                # the order page comes with a full navigation: watch the tab's address
                for _ in range(60):
                    await asyncio.sleep(0.5)
                    try:
                        url = ((await send("Target.getTargetInfo", {"targetId": target}, timeout=10)).get("targetInfo") or {}).get("url", "")
                    except FastPathError:
                        continue
                    m = re.search(r"/track/(\d+)/(\d+)", url)
                    if m:
                        placed = {"clicked": True, "placed": True, "cart_id": m.group(1), "order_id": m.group(2)}
                        try:  # when it comes: the order page's own tracking answer, else its text
                            placed["eta"] = await order_page_eta()
                        except Exception:  # noqa: BLE001 — the order is placed either way
                            pass
                        try:  # the app keeps the ordered items in its own cart copy: empty it
                            await evaluate((JS_DIR / "blinkit_clear_cart.js").read_text(), sid, timeout=10)
                        except Exception:  # noqa: BLE001 — the next cart step overwrites it anyway
                            pass
                        return placed
                return {"clicked": True, "placed": False, "problem": "no order page within 30 s"}
            finally:
                try:
                    await send("Target.closeTarget", {"targetId": target}, timeout=10)
                except Exception:  # noqa: BLE001
                    pass

    try:
        out = await asyncio.wait_for(run(), timeout=70)
    except Exception as exc:  # noqa: BLE001
        if not pressed:
            raise FastPathError(f"not placed: {str(exc)[:120]}") from exc
        return {"placed": False, "unclear": True, "problem": f"the browser stopped after Pay Now ({str(exc)[:80]}); check the app before trying again"}
    if dry:
        return out
    if not out.get("clicked"):
        raise FastPathError(f"not placed: {out.get('problem') or 'checkout check failed'}")
    if out.get("placed") and out.get("order_id"):
        return {"placed": True, "order_id": out["order_id"], "total": _rupees(check.get("total")) if check.get("total") else None,
                "payment_method": "Cash on Delivery", **({"eta": out["eta"]} if out.get("eta") else {})}
    return {"placed": False, "unclear": True, "problem": f"Pay Now was pressed but no order page came ({out.get('problem')}); check the app before trying again"}
