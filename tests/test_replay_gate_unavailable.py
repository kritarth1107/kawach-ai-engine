"""KP's decision of 14 Oct, after Google started rejecting every Vertex call in the test project (403, terms of use):
- a replay run in which no case got a real verdict (every case crashed on the provider: 403, auth, 5xx; or none ran) means
  the gate is UNAVAILABLE, not that a fixed case failed: the deploy goes on if the test job passed, loudly (the run summary,
  an email to the team when the streak starts, the admin Daily score page, the morning email), and the gate restores
  itself on the next run that reaches the model; a mix (some real verdicts, a fixed case failing) still blocks;
- crisis, self-harm and illegal-substance test content is paused (eval/sensitive.py), then sampled: crisis probes a sample
  a night, the founder's drug cases only in the nightly, never before a deploy."""

import json
from datetime import date

import pytest

from app.core import clock
from app.quality import report
from eval import replay_cases as runner
from eval import sensitive

DOWN = "crashed: AllModelsFailed: gemini:gemini-3.5-flash@asia-south1: 403 403 PERMISSION_DENIED. {'error': {'code': 403, 'message': 'Request rejected'"


def _r(cid, status, passed, whys):
    return {"id": cid, "status": status, "kind": "k", "source": "founder", "passed": passed, "runs": [{"passed": False, "why": w} for w in whys]}


def test_every_case_refused_by_the_provider_is_unavailable_not_a_regression():
    res = [_r("a", "fixed", False, [DOWN, DOWN]), _r("b", "open", False, [DOWN, DOWN])]
    why = runner.unavailable(res)
    assert why and "PERMISSION_DENIED" in why
    assert runner.unavailable([]) == "no replay case ran"


def test_a_mix_still_blocks_and_a_quota_429_is_not_the_provider_being_down():
    real_fail = {"id": "c", "status": "fixed", "kind": "k", "source": "founder", "passed": False, "runs": [{"passed": False, "why": "judge: advice"}] * 2}
    assert runner.unavailable([_r("a", "fixed", False, [DOWN, DOWN]), real_fail]) is None
    quota = _r("q", "fixed", False, ["crashed: AllModelsFailed: 429 RESOURCE_EXHAUSTED"] * 2)
    assert runner.unavailable([quota]) is None


async def _answers():
    return None


async def test_the_runner_reports_unavailable_and_lets_the_deploy_go(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "preflight", _answers)

    async def fake_run(cases, url, cap, *, parallel=20, playbook=None, deadline_s=None):
        return [_r(c["id"], c["status"], False, [DOWN, DOWN]) for c in cases]

    monkeypatch.setattr(runner, "run_cases", fake_run)
    code = await runner.main_async([{"id": "x", "status": "fixed", "kind": "k"}], 2, "postgresql+asyncpg://localhost/x", 10, tmp_path)
    assert code == 0
    out = json.loads((tmp_path / "results.json").read_text())["summary"]
    assert out["unavailable"] is True and "PERMISSION_DENIED" in out["why"] and out["passed"] is False
    assert "GATE UNAVAILABLE" in (tmp_path / "summary.md").read_text()


async def test_a_real_fixed_failure_still_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "preflight", _answers)

    async def fake_run(cases, url, cap, *, parallel=20, playbook=None, deadline_s=None):
        return [{"id": "x", "status": "fixed", "kind": "k", "source": "founder", "passed": False, "runs": [{"passed": False, "why": "judge: advice"}] * 2}]

    monkeypatch.setattr(runner, "run_cases", fake_run)
    assert await runner.main_async([{"id": "x", "status": "fixed", "kind": "k"}], 2, "postgresql+asyncpg://localhost/x", 10, tmp_path) == 1


async def test_a_refused_preflight_sends_no_case_at_all(monkeypatch, tmp_path):
    ran = []

    async def refused():
        return "the preflight call was refused: 403 PERMISSION_DENIED Request rejected in accordance with our terms of use"

    async def fake_run(*a, **k):
        ran.append(1)
        return []

    monkeypatch.setattr(runner, "preflight", refused)
    monkeypatch.setattr(runner, "run_cases", fake_run)
    assert await runner.main_async([{"id": "x", "status": "fixed", "kind": "k"}], 2, "postgresql+asyncpg://localhost/x", 10, tmp_path) == 0
    assert not ran and json.loads((tmp_path / "results.json").read_text())["summary"]["why"].startswith("the preflight call was refused")


async def test_preflight_reads_a_policy_refusal_and_lets_a_429_through(monkeypatch):
    from app.llm import router

    async def refuse(role, **kw):
        raise router.AllModelsFailed("gemini:gemini-3.5-flash@asia-south1: 403 Request rejected in accordance with our terms of use and policies.")

    monkeypatch.setattr(router, "complete", refuse)
    assert (await runner.preflight()).startswith("the preflight call was refused")

    async def busy(role, **kw):
        raise router.AllModelsFailed("gemini:gemini-3.5-flash@asia-south1: 429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(router, "complete", busy)
    assert await runner.preflight() is None


async def test_the_team_hears_when_the_streak_starts_and_the_page_shows_since(db, at, monkeypatch):
    from app.api import quality
    from app.brain import host as host_mod

    sent = []

    class Live:
        async def call(self, tool, args, **kw):
            sent.append((tool, args))
            return {"ok": True}

    monkeypatch.setattr(host_mod, "LiveHost", Live)
    at("2026-10-14 01:50")
    ok = quality.ReplayResultsIn(commit="c0", branch="master", results=[], summary={"run_id": "replay-0", "passed": True, "cases": 21})
    await quality.post_replay_results(ok, db, "ci@x")
    for i, when in enumerate(("2026-10-14 02:00", "2026-10-14 02:20")):
        at(when)
        body = quality.ReplayResultsIn(commit=f"c{i + 1}", branch="master", results=[],
                                       summary={"run_id": f"replay-{i + 1}", "unavailable": True, "why": "the model provider refused every case (403)",
                                                "passed": False})
        await quality.post_replay_results(body, db, "ci@x")
    assert len(sent) == 1 and sent[0][0] == "team_report" and "UNAVAILABLE since 14 Oct 02:00" in sent[0][1]["subject"]
    gate = await report.replay_gate(db)
    assert gate["unavailable"] and gate["runs"] == 2 and gate["since"] == "14 Oct 02:00 IST"
    at("2026-10-14 03:00")  # the provider answers again: the gate is back by itself
    back = quality.ReplayResultsIn(commit="c3", branch="master", results=[], summary={"run_id": "replay-3", "passed": True, "cases": 21})
    await quality.post_replay_results(back, db, "ci@x")
    assert (await report.replay_gate(db))["unavailable"] is False


def test_sensitive_content_is_paused_then_sampled(monkeypatch):
    crisis = [{"id": f"w{i:04d}-q09", "topic": "crisis"} for i in range(100)]
    monkeypatch.delenv("SENSITIVE_EVALS", raising=False)
    assert sensitive.mode() == "pause" and sensitive.crisis_tonight(crisis) == []
    assert not sensitive.founder_case_allowed("illegal_info", where="nightly") and sensitive.founder_case_allowed("beer", where="deploy")
    monkeypatch.setenv("SENSITIVE_EVALS", "sample")
    monkeypatch.setenv("CRISIS_SAMPLE", "30")
    a, b = sensitive.crisis_tonight(crisis, date(2026, 10, 15)), sensitive.crisis_tonight(crisis, date(2026, 10, 16))
    assert len(a) == 30 and {q["id"] for q in a} != {q["id"] for q in b}  # a different sample each night
    assert sensitive.founder_case_allowed("illegal_info", where="nightly") and not sensitive.founder_case_allowed("illegal_info", where="deploy")


def test_no_drug_case_before_a_deploy_and_no_crisis_probe_while_paused(monkeypatch):
    from eval.world import probes

    monkeypatch.delenv("SENSITIVE_EVALS", raising=False)
    assert not any(c["id"].startswith("founder-illegal") for c in runner.load_cases([]))
    fams = [{"id": "w0001", "probes": [{"id": "w0001-q01", "topic": "small_talk"}, {"id": "w0001-q09", "topic": "crisis"}]}]
    assert all(q["topic"] != "crisis" for qs in probes.tonight(fams, 1).values() for q in qs)
    assert all(q["topic"] != "crisis" for qs in probes.tonight(fams, 11).values() for q in qs)


@pytest.fixture(autouse=True)
def _clock():
    yield
    clock.set_now(None)
