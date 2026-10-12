"""The login code typed into the store's own code boxes by real key presses, as a person types it.

PharmEasy, live 2026-10-13: its code is four boxes; the browser agent's typing put three digits in, a script set the
wrong ones, it restarted the login by itself and asked for new codes until PharmEasy blocked the number ("Error in
verification"); another family was asked for a code six times. Here the code goes in one key at a time into the open tab
(the site's own key handlers move between the boxes), every box is cleared first, and what the boxes then hold is read
back: only the exact code counts. The agent then only presses Continue. Nothing here clicks a button or submits.
"""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urlparse

import httpx
import websockets

logger = logging.getLogger(__name__)

# The visible inputs that take a login code, in page order (HTML conventions, not a store's words): a one-time-code
# field, a field named or labelled otp / code / pin, or single-character boxes.
FIND = r"""(() => {
  const vis = e => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
  const ok = e => vis(e) && !e.disabled && !['hidden','checkbox','radio','submit','button','email','password'].includes((e.type||'').toLowerCase());
  const said = e => [e.name, e.id, e.getAttribute('aria-label'), e.placeholder, e.className].join(' ');
  const all = [...document.querySelectorAll('input')].filter(ok);
  let boxes = all.filter(e => (e.autocomplete||'').includes('one-time-code') || /otp|one.?time|verif|code|pin/i.test(said(e)));
  const singles = all.filter(e => e.maxLength === 1);
  if (singles.length >= 4 && (!boxes.length || boxes.every(b => singles.includes(b)))) boxes = singles;
  window.__kvCode = boxes;
  return boxes.length;
})()"""
READ = "(() => (window.__kvCode || []).map(e => e.value || '').join(''))()"
CLEAR_AND_FOCUS = r"""((i) => {
  const b = window.__kvCode || []; if (!b[i]) return false;
  b[i].focus(); if (b[i].select) { try { b[i].select(); } catch (e) {} }
  return document.activeElement === b[i];
})(%d)"""


def _key(d: str) -> list[dict]:
    code = 48 + int(d)
    return [{"type": "keyDown", "key": d, "code": f"Digit{d}", "text": d, "unmodifiedText": d,
             "windowsVirtualKeyCode": code, "nativeVirtualKeyCode": code},
            {"type": "keyUp", "key": d, "code": f"Digit{d}", "windowsVirtualKeyCode": code, "nativeVirtualKeyCode": code}]


BACKSPACE = [{"type": "rawKeyDown", "key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8, "nativeVirtualKeyCode": 8},
             {"type": "keyUp", "key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8, "nativeVirtualKeyCode": 8}]


async def type_code(cdp_url: str, code: str, *, hosts: tuple[str, ...] = (), pause_s: float = 0.15, timeout_s: float = 25.0) -> bool:
    """Type the code into the store's open code boxes. True only when the boxes then hold exactly this code."""
    digits = "".join(c for c in str(code) if c.isdigit())
    if not 4 <= len(digits) <= 8:
        return False

    async def run() -> bool:
        async with httpx.AsyncClient(timeout=10) as c:
            ws_url = (await c.get(cdp_url.rstrip("/") + "/json/version")).json()["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            n = 0

            async def send(method: str, params: dict | None = None, session: str | None = None) -> dict:
                nonlocal n
                n += 1
                mid = n
                await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}, **({"sessionId": session} if session else {})}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == mid:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result") or {}

            async def js(expr: str, sid: str):
                r = await send("Runtime.evaluate", {"expression": expr, "returnByValue": True}, session=sid)
                return (r.get("result") or {}).get("value")

            pages = [t for t in (await send("Target.getTargets"))["targetInfos"] if t.get("type") == "page"]
            mine = [t for t in pages if not hosts or any((urlparse(t.get("url") or "").hostname or "").endswith(h) for h in hosts)]
            for t in mine:
                sid = (await send("Target.attachToTarget", {"targetId": t["targetId"], "flatten": True}))["sessionId"]
                try:
                    boxes = int(await js(FIND, sid) or 0)
                    if not boxes:
                        continue
                    many = boxes > 1
                    # empty every box first (a half-typed code stays in them), from the last one back
                    for i in reversed(range(boxes if many else 1)):
                        await js(CLEAR_AND_FOCUS % i, sid)
                        for _ in range(1 if many else 10):
                            for ev in BACKSPACE:
                                await send("Input.dispatchKeyEvent", ev, session=sid)
                    for i, d in enumerate(digits):
                        if many and i < boxes:
                            await js(CLEAR_AND_FOCUS % i, sid)  # most sites move on by themselves; this keeps the rest right
                        elif not many and i == 0:
                            await js(CLEAR_AND_FOCUS % 0, sid)
                        for ev in _key(d):
                            await send("Input.dispatchKeyEvent", ev, session=sid)
                        await asyncio.sleep(pause_s)
                    await asyncio.sleep(0.4)
                    got = "".join(ch for ch in str(await js(READ, sid) or "") if ch.isdigit())
                    if got == digits:
                        return True
                    logger.warning("code boxes hold %d of %d digits after typing", len(got), len(digits))
                    return False
                finally:
                    try:
                        await send("Target.detachFromTarget", {"sessionId": sid})
                    except Exception:  # noqa: BLE001
                        pass
            return False

    try:
        return await asyncio.wait_for(run(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 — the agent types it instead, told how
        logger.warning("typing the login code failed: %s", str(exc)[:160])
        return False
