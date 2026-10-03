"""The learned playbook: short lessons and real example replies per situation, shown to Saheli before she replies.

Versions move draft -> canary (10% of families) -> live, or are rejected/blocked/retired. A family is in the
canary arm by a stable hash of its id. The block shown in a turn holds only the lessons and 2-3 examples for
that turn's situation and the person's way of writing, and always sits under the fixed rules: "the rules above
always win".
"""

from __future__ import annotations

import hashlib
import os
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.learn.models import PlaybookVersion

CACHE_SECONDS = 60.0
MAX_CHARS = 1600
_cache: dict[str, tuple[float, PlaybookVersion | None, PlaybookVersion | None]] = {}


def canary_percent() -> int:
    return max(0, min(50, int(os.getenv("LEARN_CANARY_PERCENT", "10"))))


def arm_for(family_id: str) -> str:
    h = int(hashlib.sha256(family_id.removeprefix("shadow:").encode()).hexdigest()[:8], 16) % 100
    return "canary" if h < canary_percent() else "live"


async def versions(session: AsyncSession) -> tuple[PlaybookVersion | None, PlaybookVersion | None]:
    """(live, canary), cached for a minute per process (one small row each)."""
    hit = _cache.get("v")
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1], hit[2]
    rows = list((await session.execute(select(PlaybookVersion).where(PlaybookVersion.status.in_(("live", "canary"))))).scalars())
    live = max((r for r in rows if r.status == "live"), key=lambda r: r.version, default=None)
    canary = max((r for r in rows if r.status == "canary"), key=lambda r: r.version, default=None)
    _cache["v"] = (time.monotonic(), live, canary)
    return live, canary


def reset_cache() -> None:
    _cache.clear()


async def for_family(session: AsyncSession, family_id: str) -> tuple[PlaybookVersion | None, str]:
    forced = os.getenv("LEARN_FORCE_VERSION", "").strip()
    if forced:  # training runs compare playbooks on the same simulated families
        if forced == "0":
            return None, "live"
        return await session.get(PlaybookVersion, int(forced)), "forced"
    live, canary = await versions(session)
    if canary and arm_for(family_id) == "canary":
        return canary, "canary"
    return live, "live"


def _lang_match(ex_lang: str, lang: str) -> int:
    if not lang or not ex_lang:
        return 0
    if ex_lang == lang:
        return 2
    return 1 if ex_lang.split("-")[0] == lang.split("-")[0] else 0


def block(pb: PlaybookVersion | None, situation: str, lang: str = "") -> str:
    if not pb:
        return ""
    lessons = list((pb.lessons or {}).get(situation) or []) + list((pb.lessons or {}).get("general") or [])
    examples = sorted((pb.examples or {}).get(situation) or [], key=lambda e: -_lang_match(e.get("lang", ""), lang))[:3]
    if not lessons and not examples:
        return ""
    lines = [f"LEARNED FROM OTHER FAMILIES (how replies like this went well; the rules above always win; situation: {situation}):"]
    lines += [f"  - {x}" for x in lessons[:6]]
    if examples:
        lines.append("  Examples of replies that worked (names replaced; write your own, never copy details):")
        for e in examples:
            lines.append(f"    They: {e.get('context', '')[:200]}\n    Saheli: {e.get('reply', '')[:300]}")
    text = "\n".join(lines)
    return text[:MAX_CHARS]
