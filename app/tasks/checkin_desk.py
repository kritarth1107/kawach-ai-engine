"""A local check-in desk for tests. One path per airline: the booking form, the seat map, paid add-ons, and a boarding pass.

Nothing here calls Browser Use or a live airline. A fake booking reference is not found. A free window or aisle near the front
is the only seat the walk takes. A priced seat or add-on is a stop, and the pay step is never posted.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

AIRLINES = ("indigo", "airindia", "airindiaexpress", "akasa", "spicejet")
KNOWN = {("TESTPN", "Devi")}


def fields_present(html: str) -> bool:
    """The page shows a booking reference and a last name. Code checks the page text."""
    low = (html or "").lower()
    ref = any(m in low for m in ("field name=pnr", "booking reference", "record locator", "name=\"pnr\"", "name='pnr'"))
    last = "field name=last_name" in low or "last name" in low
    return ref and last


def choose_seat(page: str) -> str | None:
    """A free window or aisle, nearest the front. A priced seat is ignored."""
    found = []
    for line in (page or "").splitlines():
        if not line.startswith("SEAT "):
            continue
        bit = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
        if bit.get("free") != "yes" or bit.get("price") != "0":
            continue
        if bit.get("side") not in ("window", "aisle") or not bit.get("id"):
            continue
        if bit["id"] not in page:
            continue
        found.append(bit)
    if not found:
        return None
    found.sort(key=lambda s: (int(s.get("row") or "99"), 0 if s.get("side") == "window" else 1))
    return found[0]["id"]


def _seats(paid_only: bool) -> str:
    lines = ["Pick a seat. A price is a paid seat: do not take it."]
    for row, letter, side in ((1, "A", "window"), (1, "B", "middle"), (1, "C", "aisle"), (2, "A", "window")):
        paid = paid_only or letter == "B"
        lines.append(f"SEAT id={row}{letter} row={row} side={side} free={'no' if paid else 'yes'} price={'400' if paid else '0'}")
    return "\n".join(lines)


def _page(title: str, body: str) -> bytes:
    return f"<!doctype html><title>{title}</title><pre>{body}</pre>".encode()


class _Desk(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002 — the base method's name
        return

    def _send(self, body: bytes, status: int = 200, kind: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _form(self) -> dict[str, str]:
        n = int(self.headers.get("Content-Length") or 0)
        raw = parse_qs(self.rfile.read(n).decode())
        return {k: v[0] for k, v in raw.items()}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path)
        parts = [p for p in path.path.split("/") if p]
        if len(parts) == 2 and parts[0] in AIRLINES and parts[1] == "check-in":
            self._send(_page(parts[0], "FIELD name=pnr\nFIELD name=last_name\nBooking reference\nLast name"))
            return
        if len(parts) == 2 and parts[0] in AIRLINES and parts[1] == "pass.pdf":
            q = parse_qs(path.query)
            pnr, seat = (q.get("pnr") or [""])[0], (q.get("seat") or [""])[0]
            body = f"%PDF-1.1\nPASS pnr={pnr} seat={seat} paid=no\n%%EOF\n".encode()
            self._send(body, kind="application/pdf")
            return
        self._send(b"not found", 404)

    def do_POST(self) -> None:  # noqa: N802
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) != 2 or parts[0] not in AIRLINES:
            self._send(b"not found", 404)
            return
        form, step = self._form(), parts[1]
        if step == "check-in":
            if (form.get("pnr"), form.get("last_name")) not in KNOWN:
                self._send(_page(parts[0], "NOT_FOUND\nThat booking was not found."))
                return
            self._send(_page(parts[0], _seats(form.get("mode") == "paidonly")))
            return
        if step == "seat":
            seat = form.get("seat") or ""
            if seat in ("1B", "2B") or form.get("mode") == "paidonly":
                self._send(_page(parts[0], f"PAY amount=400 seat={seat}\nThis seat costs money. Do not pay."))
                return
            self._send(_page(parts[0], "ADDON id=meal price=350\nADDON id=bag price=500\nADDON id=skip price=0"))
            return
        if step == "addon":
            if form.get("addon") != "skip":
                self._send(_page(parts[0], "PAY amount=350\nThat add-on costs money. Do not pay."))
                return
            seat = form.get("seat") or ""
            self._send(_page(parts[0], f"PASS pnr={form.get('pnr')} seat={seat} paid=no\n/pass.pdf?pnr={form.get('pnr')}&seat={seat}"))
            return
        self._send(b"not found", 404)


class Desk:
    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Desk)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> str:
        self._thread.start()
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=2)


async def walk(base: str, airline: str = "indigo", *, pnr: str = "TESTPN", last_name: str = "Devi", mode: str = "") -> dict:
    """The check-in path against the desk. A missing booking stops. A free seat is taken. Pay is never posted."""
    import httpx

    calls: list[str] = []
    async with httpx.AsyncClient(timeout=5) as client:
        form = (await client.get(f"{base}/{airline}/check-in")).text
        calls.append("GET /check-in")
        if not fields_present(form):
            return {"ok": False, "problem": "the booking form was not on the page", "calls": calls}
        missing = await client.post(f"{base}/{airline}/check-in", data={"pnr": "XXXXXX", "last_name": last_name})
        calls.append("POST /check-in missing")
        if "NOT_FOUND" not in missing.text:
            return {"ok": False, "problem": "a booking that is not there was accepted", "calls": calls}
        seats = await client.post(f"{base}/{airline}/check-in", data={"pnr": pnr, "last_name": last_name, "mode": mode})
        calls.append("POST /check-in")
        if "NOT_FOUND" in seats.text:
            return {"ok": False, "pnr_found": False, "calls": calls}
        seat = choose_seat(seats.text)
        if not seat:
            return {"ok": False, "pnr_found": True, "free_seat": False, "paid": False, "problem": "no free seat", "calls": calls}
        taken = await client.post(f"{base}/{airline}/seat", data={"seat": seat, "mode": mode})
        calls.append("POST /seat")
        if "PAY" in taken.text:
            return {"ok": False, "paid": False, "problem": "the seat costs money", "calls": calls}
        if "ADDON id=skip price=0" not in taken.text:
            return {"ok": False, "problem": "no way to skip the paid add-ons", "calls": calls}
        done = await client.post(f"{base}/{airline}/addon", data={"addon": "skip", "pnr": pnr, "seat": seat})
        calls.append("POST /addon skip")
        if "PAY" in done.text or "paid=no" not in done.text or f"seat={seat}" not in done.text:
            return {"ok": False, "paid": False, "problem": "no boarding pass", "calls": calls}
        pdf = await client.get(f"{base}/{airline}/pass.pdf", params={"pnr": pnr, "seat": seat})
        calls.append("GET /pass.pdf")
    paid = b"paid=no" in pdf.content and f"seat={seat}".encode() in pdf.content
    return {"ok": paid, "pnr_found": True, "free_seat": True, "seat": seat, "paid": False, "boarding_pass": paid, "calls": calls}
