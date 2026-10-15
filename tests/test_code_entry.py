"""The login code typed key by key into a store's code boxes (PharmEasy, live 2026-10-13: four boxes took three digits
from the agent, it restarted the login itself, and the number was blocked). Run against a real headless Chrome with the
kinds of code boxes stores use; skipped where no Chrome is installed."""

import asyncio
import shutil
import socket
import subprocess
import tempfile
import time

import httpx
import pytest

from app.tasks import codeentry

CHROME = shutil.which("google-chrome") or shutil.which("chromium")
pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome here")

AUTO = """<div>Enter OTP<input name=otp1 maxlength=1><input name=otp2 maxlength=1><input name=otp3 maxlength=1><input name=otp4 maxlength=1></div>
<script>const b=[...document.querySelectorAll('input')];b.forEach((e,i)=>{e.addEventListener('input',()=>{if(e.value&&b[i+1])b[i+1].focus()});
e.addEventListener('keydown',k=>{if(k.key==='Backspace'&&!e.value&&b[i-1])b[i-1].focus()})});b[0].value='6';b[1].value='2';b[2].value='7';</script>"""
PLAIN = """<input aria-label="digit" maxlength=1><input maxlength=1><input maxlength=1><input maxlength=1><input maxlength=1><input maxlength=1>"""
SINGLE = """<label>Code <input id=verify-code autocomplete="one-time-code" maxlength=4 value="62"></label>"""
# keys only: a component that ignores pasted or scripted values and takes each key press itself
KEYS_ONLY = """<div id=w></div><script>const w=document.getElementById('w');const b=[];for(let i=0;i<4;i++){const e=document.createElement('input');
e.name='otp';e.maxLength=1;e.addEventListener('keydown',k=>{if(/^[0-9]$/.test(k.key)){k.preventDefault();e.value=k.key;if(b[i+1])b[i+1].focus()}
else if(k.key==='Backspace'){k.preventDefault();e.value='';}});e.addEventListener('input',()=>{e.value=''});b.push(e);w.appendChild(e)}
b[0].value='6';b[1].value='2';b[2].value='7';b[3].value='2';</script>"""


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _launch() -> tuple[subprocess.Popen, str]:
    """A fresh headless Chrome whose debug port is answering. One start can exit before the port is up (seen on the
    engine test job: the page open got ConnectError and the code was never typed)."""
    port, home = _port(), tempfile.mkdtemp(prefix="kv-chrome-")
    proc = subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={home}",
         "--no-first-run", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            httpx.get(url + "/json/version", timeout=0.5)
            return proc, url
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    proc.kill()
    proc.wait(timeout=5)
    raise RuntimeError(f"Chrome debug port never answered (exit {proc.returncode})")


@pytest.fixture
def chrome():
    last = None
    proc = url = None
    for _ in range(3):
        try:
            proc, url = _launch()
            break
        except RuntimeError as e:
            last = e
    else:
        pytest.fail(str(last))
    yield url
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _open(cdp: str, html: str) -> None:
    target = f"{cdp}/json/new?" + "data:text/html," + html.replace("#", "%23").replace("\n", " ")
    last = None
    for _ in range(5):
        try:
            async with httpx.AsyncClient() as c:
                t = (await c.put(target)).json()
            if t.get("id"):
                await asyncio.sleep(0.8)
                return
            last = t
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
            last = e
        await asyncio.sleep(0.3)
    raise AssertionError(f"could not open the code page: {last}")


async def _boxes(cdp: str) -> str:
    import json

    import websockets

    async with httpx.AsyncClient() as c:
        pages = [p for p in (await c.get(cdp + "/json/list")).json() if p["url"].startswith("data:")]
    async with websockets.connect(pages[0]["webSocketDebuggerUrl"]) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
            "expression": "[...document.querySelectorAll('input')].map(e=>e.value).join('')", "returnByValue": True}}))
        while True:
            m = json.loads(await ws.recv())
            if m.get("id") == 1:
                return m["result"]["result"]["value"]


@pytest.mark.parametrize("html,code", [(AUTO, "4821"), (PLAIN, "905172"), (SINGLE, "3307"), (KEYS_ONLY, "6274")])
async def test_the_code_goes_in_whole_whatever_the_boxes_and_what_was_half_typed_before(chrome, html, code):
    await _open(chrome, html)
    assert await codeentry.type_code(chrome, code, pause_s=0.05) is True
    assert await _boxes(chrome) == code


async def test_no_code_boxes_on_the_page_is_never_a_success(chrome):
    await _open(chrome, "<p>Your cart</p><input name=search placeholder='Search medicines'>")
    assert await codeentry.type_code(chrome, "4821", pause_s=0.05) is False
    assert await codeentry.type_code(chrome, "12", pause_s=0.05) is False, "not a code"
