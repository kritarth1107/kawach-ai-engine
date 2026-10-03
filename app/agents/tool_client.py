"""HTTP client for kavach-backend Saheli tool executor."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {502, 503, 504}
# Safe to retry only when the backend surely did nothing: it was unavailable (503) or we never connected.
SAFE_RETRY_STATUS = {503}

_clients: dict[int, httpx.AsyncClient] = {}


def _client() -> httpx.AsyncClient:
    """One pooled client per event loop (a new TLS connection per tool call was slow)."""
    loop = asyncio.get_running_loop()
    c = _clients.get(id(loop))
    if c is None or c.is_closed:
        for k in [k for k, v in _clients.items() if v.is_closed]:
            _clients.pop(k, None)
        c = httpx.AsyncClient(timeout=35.0, limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
        _clients[id(loop)] = c
    return c


async def close_clients() -> None:
    for c in list(_clients.values()):
        if not c.is_closed:
            await c.aclose()
    _clients.clear()


def _is_write(tool: str) -> bool:
    from app.brain.host import WRITE_TOOLS

    return tool in WRITE_TOOLS


async def execute_backend_tool(
    *,
    tool: str,
    args: dict,
    family_id: str,
    elder_id: str,
    actor_user_id: str,
) -> dict:
    settings = get_settings()
    base = (settings.kavach_backend_url or "http://localhost:5000").rstrip("/")
    url = f"{base}/internal/v1/saheli/tools/execute"
    headers = {"X-Kavach-Secret": settings.kawach_api_secret}
    payload = {
        "tool": tool,
        "args": args,
        "family_id": family_id,
        "recipient_user_id": elder_id,
        "actor_user_id": actor_user_id,
    }
    # A write (send a WhatsApp, mark a dose, notify caregivers) that timed out may already have happened:
    # retrying it could send twice. Writes retry only when the backend surely did nothing.
    write = _is_write(tool)
    retry_status = SAFE_RETRY_STATUS if write else RETRYABLE_STATUS

    last_error: Exception | None = None
    for attempt in range(2):
        started = time.monotonic()
        try:
            res = await _client().post(url, json=payload, headers=headers)
            latency_ms = int((time.monotonic() - started) * 1000)
            if res.status_code in retry_status and attempt == 0:
                logger.warning("saheli_tool retry tool=%s status=%s latency_ms=%s", tool, res.status_code, latency_ms)
                await asyncio.sleep(0.5)
                continue
            res.raise_for_status()
            body = res.json()
            logger.info("saheli_tool tool=%s latency_ms=%s", tool, latency_ms)
            return body.get("result", body)
        except httpx.ConnectError as exc:  # never reached the backend: safe to retry any tool
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(0.5)
                continue
            raise
        except httpx.TimeoutException as exc:
            last_error = exc
            logger.warning("saheli_tool timeout tool=%s attempt=%s write=%s", tool, attempt, write)
            if attempt == 0 and not write:
                await asyncio.sleep(0.5)
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError(f"Tool {tool} failed after retry")
