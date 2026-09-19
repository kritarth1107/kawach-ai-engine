"""HTTP client for kavach-backend Saheli tool executor."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {502, 503, 504}


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

    last_error: Exception | None = None
    for attempt in range(2):
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                res = await client.post(url, json=payload, headers=headers)
                latency_ms = int((time.monotonic() - started) * 1000)
                if res.status_code in RETRYABLE_STATUS and attempt == 0:
                    logger.warning(
                        "saheli_tool retry tool=%s status=%s latency_ms=%s",
                        tool,
                        res.status_code,
                        latency_ms,
                    )
                    await asyncio.sleep(0.5)
                    continue
                res.raise_for_status()
                body = res.json()
                logger.info("saheli_tool tool=%s latency_ms=%s", tool, latency_ms)
                return body.get("result", body)
        except httpx.TimeoutException as exc:
            last_error = exc
            logger.warning("saheli_tool timeout tool=%s attempt=%s", tool, attempt)
            if attempt == 0:
                await asyncio.sleep(0.5)
                continue
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in RETRYABLE_STATUS and attempt == 0:
                last_error = exc
                await asyncio.sleep(0.5)
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError(f"Tool {tool} failed after retry")
