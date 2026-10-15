"""The Browser Use comparison stays off until the founder says to start it."""

import os
from types import SimpleNamespace

import pytest

import scripts.browser_use_bench as bench


def test_the_bench_does_not_run_until_the_founder_says_so(monkeypatch):
    monkeypatch.delenv("BROWSER_USE_BENCH", raising=False)
    assert bench.main(["--run"]) == 2
    assert all("Do not pay" in task for _, _, task in bench.RUNS)
    assert {api for api, _ in bench.CASES} == {"v2", "v4"}
    assert {service for service, _, _ in bench.RUNS} >= {"blinkit", "indigo", "akasa", "spicejet"}


def test_the_phrase_only_prints_the_plan(monkeypatch, capsys):
    monkeypatch.setenv("BROWSER_USE_BENCH", bench.PHRASE)
    monkeypatch.setenv("BROWSER_USE_API_KEY", "would-spend")
    assert bench.main([]) == 0
    said = capsys.readouterr().out
    assert "Nothing is ordered or booked" in said and "blinkit" in said and "indigo" in said
    assert "Nothing was called" in said


def test_run_without_a_key_calls_nothing(monkeypatch, capsys):
    monkeypatch.setenv("BROWSER_USE_BENCH", bench.PHRASE)
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    assert bench.main(["--run"]) == 2
    assert "Nothing was called" in capsys.readouterr().err


class _Agent:
    def __init__(self, status: str = "finished") -> None:
        self.status = status
        self.runs: list[dict] = []
        self.stopped: list = []
        self.polls = 0

    async def run(self, **kw):
        self.runs.append({**kw, "_api": os.environ.get("BROWSER_API"), "_model": os.environ.get("BROWSER_MODEL")})
        return SimpleNamespace(task_id=f"t{len(self.runs)}", session_id="sess", status=self.status, output={"paid": False})

    async def poll(self, task_id):
        self.polls += 1
        return SimpleNamespace(task_id=task_id, session_id="sess", status="started", output=None)

    async def stop(self, task_id, *, end_session=False):
        self.stopped.append(("stop", task_id, end_session))

    async def stop_session(self, session_id):
        self.stopped.append(("session", session_id))


@pytest.mark.asyncio
async def test_the_go_path_uses_no_profile_and_stops_the_session(monkeypatch):
    monkeypatch.delenv("BROWSER_API", raising=False)
    monkeypatch.delenv("BROWSER_MODEL", raising=False)
    agent = _Agent()
    rows = await bench.execute(agent, polls=1)
    assert len(rows) == len(bench.RUNS) * len(bench.CASES)
    assert all(run["profile_id"] is None and "Do not pay" in run["goal"] for run in agent.runs)
    assert {run["_api"] for run in agent.runs} == {"v2", "v4"}
    assert os.environ.get("BROWSER_API") is None
    assert all(bit[0] == "session" for bit in agent.stopped)


@pytest.mark.asyncio
async def test_a_run_that_does_not_finish_is_stopped(monkeypatch):
    agent = _Agent("started")

    async def _no_wait(_seconds):
        return None

    rows = await bench.execute(agent, sleep=_no_wait, polls=2)
    assert all(row["status"] == "stopped" for row in rows)
    assert agent.polls == len(rows) * 2
    assert ("stop", "t1", True) in agent.stopped
