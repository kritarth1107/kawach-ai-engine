"""The admin's telemetry: tokens, model calls, money, cache and time per turn, by role, model, tool and family, and the
scheduled jobs' runs. Synthetic turns only; no text is returned."""

from datetime import timedelta

from app.api import admin
from app.care import why
from app.core import clock


async def test_tokens_calls_cost_and_cache_per_turn(db, at):
    at("2026-10-17 12:00")
    for i in range(4):
        calls = [("brain", "gemini-3.5-flash", 0.5, 10_000, 200), ("brain", "gemini-3.5-flash", 0.5, 12_000, 100), ("judge_fast", "gemini-3.5-flash", 0.1, 2_000, 20)]
        timing = {"contextMs": 300, "cacheRead": 6_000, "promptTok": 24_000, "model": [["brain", 4000], ["brain", 3000], ["judge_fast", 800]], "tool": [["recall", 200]]}
        await why.record_turn(db, trace_id=f"tm{i}", family_id="fam-tel-1", thread_id="p1", kind="message", at=clock.now() - timedelta(minutes=i),
                              ms=8_000 + 1000 * i, text="secret words", reply="secret reply", calls=calls, timing=timing)
    await why.record_turn(db, trace_id="tm-own", family_id="fam-tel-1", thread_id="p1", kind="system", ms=3000,
                          calls=[("worker", "gemini-3.5-flash", 0.2, 5_000, 50)])
    await why.record_turn(db, trace_id="tm-sim", family_id="sim-x", thread_id="p", kind="message", ms=1, calls=[("brain", "m", 99.0, 1, 1)])
    await db.commit()
    out = await admin.admin_telemetry(db, hours=24)
    m = out["messages"]
    assert m["turns"] == 4 and m["callsPerTurn"]["avg"] == 3 and m["tokensInPerTurn"]["avg"] == 24_000
    assert m["costPerTurn"]["avg"] == 1.1 and m["costInr"] == 4.4
    assert m["cache"] == {"readTokens": 24_000, "promptTokens": 96_000, "hitPct": 25.0}
    assert out["proactive"]["turns"] == 1 and out["proactive"]["costInr"] == 0.2
    brain = next(r for r in out["byRole"] if r["role"] == "brain")
    assert brain["calls"] == 8 and brain["p50Ms"] == 3000 and brain["p90Ms"] == 4000
    assert out["byTool"][0] == {"tool": "recall", "calls": 4, "p50Ms": 200, "p90Ms": 200}
    assert [f["family"] for f in out["byFamily"]] == ["fam-tel-"]
    assert out["slowest"][0]["ms"] == 11_000
    assert "secret" not in str(out)


class _Same:
    """The test's own session as a session factory (record_job opens one per run)."""

    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, *exc):
        return False


async def test_job_runs_latest_and_last_day(db, at):
    at("2026-10-17 12:00")
    sessions = lambda: _Same(db)  # noqa: E731
    now = clock.now()
    await why.record_job(sessions, "tasks", now - timedelta(minutes=2), 120, 200)
    await why.record_job(sessions, "tasks", now - timedelta(minutes=1), 90, 500)
    await why.record_job(sessions, "dream", now - timedelta(hours=30), 600_000, 200)
    jobs = {j["job"]: j for j in (await admin.admin_job_runs(db, days=3))["jobs"]}
    assert jobs["tasks"]["lastOk"] is False and jobs["tasks"]["lastCode"] == 500
    assert jobs["tasks"]["runs24h"] == 2 and jobs["tasks"]["failed24h"] == 1
    assert jobs["dream"]["runs24h"] == 0 and jobs["dream"]["lastOk"] is True
