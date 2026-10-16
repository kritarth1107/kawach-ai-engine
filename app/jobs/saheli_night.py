"""The nightly world, inside AWS, after the dream. Gemini in kavach-eval-1015. Cap ₹150.

One process: probes, judge, recall, the message week, nudge quality, the report, then the score
posted to the live engine so learning and the morning report can read it. A provider refusal is
unavailable (exit 0). A broken bar exits 1.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.live_routes import CARE, EVAL_PROJECT, project

# The scheduled night. A dispatch on GitHub used to allow more. This task does not.
CAPS = {
    "PROBES_MAX_INR": "60",
    "JUDGE_MAX_INR": "40",
    "RECALL_MAX_INR": "20",
    "NUDGE_EVAL_MAX_INR": "20",
    "POINTS_MAX_INR": "10",
    "NIGHT_CAP_INR": "150",
    "SHARDS": "1",
    "SENSITIVE_EVALS": "pause",
    "WEB_SEARCH_PROVIDER": "off",
    "MEMORY_EMBEDDINGS": "off",
    "LLM_PROVIDER": "router",
    "DEBUG": "false",
}
OUT = Path("runs/nightly")


def night_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    env.update(CAPS)
    env["NIGHTLY_OUT"] = str(OUT)
    replay = env.get("REPLAY_DATABASE_URL", "").strip()
    if replay:
        env["DATABASE_URL"] = replay
    env.setdefault("ENGINE_URL", "http://engine.kavach.local:8000")
    return env


def _run(args: list[str], env: dict[str, str]) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.call([sys.executable, "-m", *args], env=env)


def main() -> int:
    if project() != EVAL_PROJECT or project() == CARE:
        print(f"refusing: the night runs only in {EVAL_PROJECT}")
        return 1
    env = night_env()
    if not env.get("DATABASE_URL"):
        print("refusing: REPLAY_DATABASE_URL is not set")
        return 1
    from app.jobs.learn_gate import wait_replay_db

    asyncio.run(wait_replay_db())
    OUT.mkdir(parents=True, exist_ok=True)
    if _run(["eval.world.make"], env):
        return 1
    if _run(["eval.live_routes", "--preflight"], env):
        return 1
    if (OUT / "provider_down.json").exists():
        print("night unavailable: the model refused the preflight")
        scores = OUT / "scores.json"
        if not scores.exists():
            scores.write_text('{"unavailable": true, "passed": false, "failures": ["the model refused the preflight"]}\n')
        _post(env)
        return 0
    steps = [
        ["eval.world.probes", "full", "--parallel", "2", "--every", "night"],
        ["eval.world.judge", str(OUT), "--sample", "1"],
        ["eval.world.recall", "full", "--families", "25", "--parallel", "2"],
        ["eval.world.budget", "--families", "200", "--days", "7", "--tick", "5"],
        ["eval.nudge_day_sim", "--days", "2", "--tick", "5"],
        ["eval.nudge_quality", "--per-language", "10", "--out", "runs/nudge_quality"],
        ["eval.founder_points_eval", "--repeat", "2"],
    ]
    for args in steps:
        code = _run(args, env)
        if code:
            print(f"night step failed ({code}): {' '.join(args)}")
            (OUT / "scores.json").write_text(json.dumps({
                "passed": False, "failures": [f"step failed: {' '.join(args)}"],
            }))
            _post(env)
            return code
    code = _run(["eval.world.report", str(OUT), "--out", str(OUT / "report.md"), "--scores", str(OUT / "scores.json"),
                 "--extras", "runs"], env)
    _post(env)
    return code


def _post(env: dict[str, str]) -> None:
    scores = OUT / "scores.json"
    if not scores.exists():
        return
    cmd = [sys.executable, "scripts/post_eval_scores.py", str(scores), "--kind", "nightly", "--branch", "master",
           "--commit", env.get("IMAGE_SHA", ""), "--run-id", env.get("NIGHT_RUN_ID", "aws-night")]
    print("+", " ".join(cmd), flush=True)
    subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
