"""HTTP client for kavach-backend Saheli tool executor."""

from __future__ import annotations

import httpx

from app.core.config import get_settings


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
    async with httpx.AsyncClient(timeout=35.0) as client:
        res = await client.post(
            url,
            json={
                "tool": tool,
                "args": args,
                "family_id": family_id,
                "recipient_user_id": elder_id,
                "actor_user_id": actor_user_id,
            },
            headers={"X-Kavach-Secret": settings.kawach_api_secret},
        )
        res.raise_for_status()
        payload = res.json()
        return payload.get("result", payload)
