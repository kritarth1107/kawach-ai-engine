"""The self-learning gate's replay run (trust plan section 10.2 step 2, on the cases of C3).

run_for_playbook(sessions, pb, budget_inr=...) runs every replay case twice on real Gemini, once with the live playbook and
once with the candidate `pb` in its place, and reports:
    {"regressions": [cases that pass with the live playbook and fail with pb], "fixed": [fail live, pass with pb],
     "n": cases run, "cost_inr": ₹ spent, "skipped": [cases that could not run]}
An empty regressions list means the candidate passed (lessons.replay_gate decides).

Paid runs go to the sunny-ship test project only, on a local test database (REPLAY_DATABASE_URL), with the runner in
eval/ (eval/replay_cases.py), so it runs where the tests run (GitHub Actions, a laptop), never inside the production
engine: anywhere else it raises ReplayUnavailable and drafts wait. Cases: the repo's invented families plus every open and
fixed case in `sessions`' replay_cases (anonymised).
"""

from __future__ import annotations

import os

TEST_PROJECT = "sunny-ship-508214-q1"


class ReplayUnavailable(RuntimeError):
    """The replay run cannot happen here (not the test project, no test database, or no runner)."""


_FROM_DB = object()


async def run_for_playbook(sessions, pb, *, budget_inr: float = 150.0, max_cases: int = 40, live=_FROM_DB, case_files: tuple[str, ...] = ()) -> dict:
    """live: the playbook the candidate is compared with (default: the live one in `sessions`; the learn-gate workflow passes
    production's). case_files: exported cases (jsonl) to add to the repo's."""
    if os.getenv("GCP_PROJECT_ID") != TEST_PROJECT or os.getenv("GOOGLE_CLOUD_PROJECT", TEST_PROJECT) != TEST_PROJECT:
        raise ReplayUnavailable("paid replay runs go to the sunny-ship test project only")
    url = os.getenv("REPLAY_DATABASE_URL", "").strip()
    if not url:
        raise ReplayUnavailable("set REPLAY_DATABASE_URL to a local test database")
    try:
        from eval import replay_cases as runner
    except ImportError as exc:  # the production image has no eval/
        raise ReplayUnavailable("the replay runner (eval/) is not here") from exc
    from sqlalchemy import select

    from app.learn import playbook as learned
    from app.llm import spend
    from app.quality.cases import as_row
    from app.quality.models import ReplayCase

    cases = runner.load_cases(list(case_files))
    async with sessions() as s:
        try:
            rows = list((await s.execute(select(ReplayCase).where(ReplayCase.status.in_(("open", "fixed"))))).scalars())
        except Exception:  # noqa: BLE001 — a database without the table: the repo's cases only
            await s.rollback()
            rows = []
        cases += [as_row(r) for r in rows if r.anonymised or not r.family_id]
        if live is _FROM_DB:
            live, _canary = await learned.versions(s)
    cases = runner.choose(cases, max_cases)
    start = spend.process_spent()
    half = budget_inr / 2
    with_live = {r["id"]: r for r in await runner.run_cases(cases, url, half, playbook=live)}
    with_pb = {r["id"]: r for r in await runner.run_cases(cases, url, half, playbook=pb)}
    if not any(r["passed"] is not None for r in with_pb.values()):
        raise ReplayUnavailable("no replay case could run (time, budget or the cases)")  # never a silent pass
    regressions = [cid for cid, r in with_live.items() if r["passed"] is True and with_pb.get(cid, {}).get("passed") is False]
    fixed = [cid for cid, r in with_live.items() if r["passed"] is False and with_pb.get(cid, {}).get("passed") is True]
    skipped = sorted({cid for cid, r in [*with_live.items(), *with_pb.items()] if r["passed"] is None})
    return {"regressions": regressions, "fixed": fixed, "n": len(cases), "cost_inr": round(spend.process_spent() - start, 2), "skipped": skipped,
            "playbook": getattr(pb, "version", None), "live": getattr(live, "version", None)}
