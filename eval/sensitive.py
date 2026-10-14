"""Crisis, self-harm and illegal-substance test content in the test project (KP's decision, 14 Oct 2026).

Since about 01:45 IST on 14 Oct, Google has rejected every Vertex call in sunny-ship with 403 "Request rejected in
accordance with our terms of use and policies"; the likely trigger is the volume of such test messages (every crisis probe
every night, the founder's drug cases on every deploy). Until the founder has answered Google:
- SENSITIVE_EVALS=pause (the default): no crisis probes in the nightly world, no illegal-substance founder cases anywhere.
- SENSITIVE_EVALS=sample (after the answer): a rotating sample of CRISIS_SAMPLE crisis probes a night, the founder's drug
  cases only in the nightly (founder_points_eval), never on a deploy (the replay cases).
Crisis handling stays covered by the engine's own tests (no model) on every push.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date

ILLEGAL_PREFIX = "illegal"  # founder cases illegal_info, illegal_effects


def mode() -> str:
    m = os.getenv("SENSITIVE_EVALS", "pause").strip().lower()
    return m if m in ("pause", "sample") else "pause"


def crisis_sample() -> int:
    return max(0, int(os.getenv("CRISIS_SAMPLE", "30") or 0))


def crisis_tonight(probes: list[dict], day: date | None = None) -> list[dict]:
    """Tonight's crisis probes from all of them: none while paused, else a sample that rotates night by night."""
    if mode() == "pause" or not probes:
        return []
    n = crisis_sample()
    night = (day or date.today()).toordinal()
    ranked = sorted(probes, key=lambda q: hashlib.sha256(f"{night}:{q['id']}".encode()).hexdigest())
    return ranked[:n]


def founder_case_allowed(key: str, *, where: str) -> bool:
    """where: 'deploy' (the replay cases before a deploy, and the learn gate) or 'nightly'."""
    if not key.startswith(ILLEGAL_PREFIX):
        return True
    return where == "nightly" and mode() == "sample"
