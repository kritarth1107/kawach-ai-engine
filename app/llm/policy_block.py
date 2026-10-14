"""A provider's policy refusal is never retried (KP, 14 Oct 2026).

On 14 Oct Google started refusing every Vertex call in the test project with 403 "Request rejected in accordance with our
terms of use and policies", and the router kept calling: each request tried every Gemini route of the same project, and
when all were tripped it "tried anyway"; 3,872 refused calls went out in one 5-minute window. More refused traffic can turn
the block into a project or billing-account suspension (production shares the billing account).

Now one policy refusal blocks that provider (every route of it in this process: they share the project) for POLICY_BLOCK_S;
during the window nothing is sent to it, and after it a single call tests it again.
"""

from __future__ import annotations

import os
import re
import time

_blocked_until: dict[str, float] = {}
_why: dict[str, str] = {}

POLICY = re.compile(r"in accordance with our terms of use|terms of use and polic|violat\w* (?:our|the) (?:terms|usage polic)", re.I)


def window_s() -> float:
    return float(os.getenv("POLICY_BLOCK_S", "900"))


def is_policy_refusal(status: int | None, text: str) -> bool:
    return (status in (None, 400, 403)) and bool(POLICY.search(text or ""))


def note(provider: str, text: str) -> None:
    _blocked_until[provider] = time.monotonic() + window_s()
    _why[provider] = (text or "")[:200]


def blocked(provider: str) -> str | None:
    """Why this provider is blocked right now, or None."""
    if _blocked_until.get(provider, 0.0) > time.monotonic():
        return _why.get(provider) or "policy refusal"
    return None


def reset() -> None:
    _blocked_until.clear()
    _why.clear()
