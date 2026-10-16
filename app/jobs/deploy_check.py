"""Pre-deploy Saheli check. One ECS task, Gemini in kavach-eval-1015, never kavach-care.

Exit 0 only when the replay cases pass. A provider refusal exits 1: nothing was tested, so the live
image stays. The deploy workflow emails the failure lines and does not move the service.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from eval.live_routes import CARE, EVAL_PROJECT, project

KEYS = Path("eval/fixtures/deploy_gate_keys.txt")
LIVE_DAY = Path("eval/fixtures/live_2026-10-15.jsonl")
OUT = Path("runs/replay")
CAP = "20"


def verdict(summary: dict | None) -> int:
    """0 only for a real pass. Missing, refused, or a regression keeps the live image."""
    if not summary or summary.get("unavailable") or not summary.get("passed"):
        return 1
    return 0


def replay_argv() -> list[str]:
    keys = ",".join(line.strip() for line in KEYS.read_text().splitlines() if line.strip())
    return ["--no-founder", "--parallel", "2", "--max-cases", "6", "--keys", keys, "--out", str(OUT), str(LIVE_DAY)]


def main() -> int:
    if project() != EVAL_PROJECT or project() == CARE:
        print(f"refusing: the deploy check runs only in {EVAL_PROJECT}")
        return 1
    os.environ["REPLAY_MAX_INR"] = CAP
    replay = os.environ.get("REPLAY_DATABASE_URL", "").strip()
    if not replay:
        print("refusing: REPLAY_DATABASE_URL is not set")
        return 1
    os.environ["DATABASE_URL"] = replay
    from app.jobs.learn_gate import wait_replay_db

    asyncio.run(wait_replay_db())
    from eval import replay_cases

    replay_cases.main(replay_argv())
    path = OUT / "results.json"
    summary = None
    if path.exists():
        try:
            summary = json.loads(path.read_text()).get("summary")
        except json.JSONDecodeError:
            summary = None
    code = verdict(summary)
    if code:
        why = (summary or {}).get("failures") or ["the replay did not pass"]
        print("DEPLOY CHECK FAILED")
        for line in why:
            print(f"  {line}")
    else:
        print("DEPLOY CHECK PASSED")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
