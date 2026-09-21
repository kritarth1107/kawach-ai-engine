"""Fetch Kavach backend memory context for the dream job."""

from __future__ import annotations

import logging
import uuid

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def fetch_dream_context(
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> tuple[str, str]:
    settings = get_settings()
    base = (settings.kavach_backend_url or "").rstrip("/")
    secret = settings.kavach_job_secret or settings.kawach_api_secret or ""
    if not base or not secret:
        return "", ""

    url = f"{base}/internal/v1/saheli/memory-context"
    params = {"aiFamilyId": str(family_id), "aiElderId": str(elder_id)}
    headers = {"X-Kavach-Job-Secret": secret}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(url, params=params, headers=headers)
            if res.status_code != 200:
                logger.warning("Dream context fetch failed: %s", res.status_code)
                return "", ""
            data = res.json().get("data") or {}
            life = data.get("companion_profile") or {}
            roster = data.get("family_roster") or []
            life_lines = [f"- Child name: {life.get('child_name') or life.get('childName') or 'Saheli'}"]
            if roster:
                life_lines.append(
                    "Family: "
                    + ", ".join(f"{m.get('name', 'Member')}" for m in roster[:8])
                )
            life_context = "\n".join(life_lines)
            care_context = str(data.get("care_record_context") or "")
            return life_context, care_context
    except Exception:
        logger.exception("Dream context fetch error")
        return "", ""
