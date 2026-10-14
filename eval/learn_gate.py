"""The self-learning replay gate, run where the replay cases can run (trust plan §10.2 step 2; .github/workflows/learn-gate.yml).

Production's nightly learning makes a playbook draft (lessons, approved rules) but cannot run the replay cases itself (no
eval/, no test project there), so the draft waits. This job, on the live Bedrock routes after the 02:00 night:
1. asks the engine for the waiting draft and the live playbook (GET /v2/admin/learn/waiting-draft);
2. runs every replay case twice on Bedrock in ap-south-1, with the live playbook and with the draft in its place
   (app.learn.replay.run_for_playbook, on a local test database; the repo's cases);
3. posts the result (POST /v2/admin/learn/playbooks/{version}/replay): no regression starts its canary on 20% of families,
   a regression rejects it, an error leaves it waiting (never a silent pass).

    ENGINE_URL=https://… ID_TOKEN=… REPLAY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/kawach_replay \
    LEARN_REPLAY_MAX_INR=150 WEB_SEARCH_PROVIDER=off \
    PYTHONPATH=. python -m eval.learn_gate [cases.jsonl] [--dry-run]
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")
os.environ.setdefault("PERSON_SYNC", "off")

import asyncio
import json
import sys
import urllib.request
from types import SimpleNamespace


def call(method: str, path: str, body: dict | None = None) -> dict:
    url = os.environ["ENGINE_URL"].rstrip("/") + path
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {os.environ['ID_TOKEN']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 — the engine's own URL
        return json.loads(r.read() or b"{}")


def playbook(d: dict | None):
    """The engine's playbook as the replay runner reads it (version, lessons, examples)."""
    if not d:
        return None
    return SimpleNamespace(version=d["version"], lessons=d.get("lessons") or {}, examples=d.get("examples") or {}, scope=d.get("scope") or [],
                           status=d.get("status"))


async def gate(draft: dict, live: dict | None, case_files: list[str], budget: float) -> dict:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.learn import replay
    from eval.world.seed_engine import assert_local, prepare

    url = os.environ["REPLAY_DATABASE_URL"]
    assert_local(url)
    engine = create_async_engine(url)
    await prepare(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        return await replay.run_for_playbook(sessions, playbook(draft), budget_inr=budget, live=playbook(live), case_files=tuple(case_files))
    finally:
        await engine.dispose()


def main(argv: list[str]) -> int:
    from eval.live_routes import apply

    apply()
    waiting = call("GET", "/v2/admin/learn/waiting-draft")
    draft = waiting.get("draft")
    if not draft:
        print("no playbook draft is waiting for its replay gate")
        return 0
    print(f"draft v{draft['version']} ({draft.get('note') or ''}) against live v{(waiting.get('live') or {}).get('version')}", flush=True)
    budget = float(os.getenv("LEARN_REPLAY_MAX_INR", "150"))
    run_id = f"learn-gate-{os.getenv('GITHUB_RUN_ID', 'local')}"
    try:
        res = asyncio.run(gate(draft, waiting.get("live"), [a for a in argv if a.endswith(".jsonl")], budget))
        body = {"run_id": run_id, "regressions": res.get("regressions") or [], "fixed": res.get("fixed") or [], "skipped": res.get("skipped") or [],
                "n": int(res.get("n") or 0), "cost_inr": float(res.get("cost_inr") or 0), "live": res.get("live")}
    except Exception as exc:  # noqa: BLE001 — reported as an error: the draft keeps waiting
        body = {"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"[:400]}
    print(json.dumps(body, ensure_ascii=False, indent=1))
    if "--dry-run" in argv:
        return 0
    out = call("POST", f"/v2/admin/learn/playbooks/{draft['version']}/replay", body)
    print(f"engine: {out}")
    return 1 if body.get("error") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
