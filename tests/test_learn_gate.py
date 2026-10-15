"""The self-learning replay gate wired end to end (trust plan §10.2 step 2): production's night makes a draft that cannot
pass its replay gate there (no runner), so it waited forever; the learn-gate workflow fetches it, runs the replay cases with
it in the test project and posts the result, which starts its canary or rejects it. An error leaves it waiting."""

import json

import pytest
from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api import quality
from app.core import clock
from app.learn import playbook
from app.learn.models import PlaybookVersion


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def _versions(db):
    await db.execute(delete(PlaybookVersion))
    db.add(PlaybookVersion(version=1, status="live", lessons={"general": ["Be brief."]}, examples={}, created_at=clock.now(), gate={}, canary={},
                           note="live", scope=[]))
    db.add(PlaybookVersion(version=2, status="draft", lessons={"general": ["Be brief.", "Use the name they asked for."]}, examples={},
                           created_at=clock.now(), gate={"approved_rules": {"general": ["Use the name they asked for."]},
                                                         "replay": {"status": "unavailable", "why": "the replay runner (eval/) is not here"}},
                           canary={}, note="learned: general; 1 approved rules", scope=[]))
    await db.commit()
    playbook.reset_cache()


async def test_the_waiting_draft_and_the_live_playbook_are_served(db, sessions):
    await _versions(db)
    out = await quality.learn_waiting_draft(db, "gate@x")
    assert out["draft"]["version"] == 2 and out["draft"]["lessons"]["general"][1] == "Use the name they asked for."
    assert out["live"]["version"] == 1


async def test_no_regression_starts_the_canary(db, sessions):
    await _versions(db)
    out = await quality.apply_gate(sessions, 2, quality.GateIn(run_id="g1", n=21, fixed=["seed-x"], live=1), "gate@x")
    assert out == {"version": 2, "status": "canary", "replay": "passed"}
    db.expire_all()
    pb = await db.get(PlaybookVersion, 2)
    assert pb.status == "canary" and pb.gate["replay"]["fixed"] == ["seed-x"] and pb.canary_since is not None


async def test_a_regression_rejects_the_draft(db, sessions):
    await _versions(db)
    out = await quality.apply_gate(sessions, 2, quality.GateIn(run_id="g2", n=21, regressions=["founder-relay"]), "gate@x")
    assert out["status"] == "rejected" and out["replay"] == "failed"


async def test_an_error_leaves_it_waiting_and_a_non_draft_is_refused(db, sessions):
    await _versions(db)
    out = await quality.apply_gate(sessions, 2, quality.GateIn(run_id="g3", error="TimeoutError"), "gate@x")
    assert out["status"] == "draft" and out["replay"] == "unavailable"
    with pytest.raises(HTTPException) as e:
        await quality.apply_gate(sessions, 1, quality.GateIn(run_id="g4", n=3), "gate@x")
    assert e.value.status_code == 409


async def test_no_draft_does_not_touch_the_replay_database(db, sessions):
    await db.execute(delete(PlaybookVersion))
    await db.commit()
    from app.jobs import learn_gate

    out = await learn_gate.run(sessions)
    assert out["draft"] is None and out["why"] == "no draft is waiting"


async def test_a_draft_stays_a_draft_when_the_replay_database_is_down(db, sessions, monkeypatch):
    await _versions(db)
    monkeypatch.setenv("REPLAY_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@127.0.0.1:1/kawach_replay")
    monkeypatch.setenv("LEARN_GATE_DB_TRIES", "1")
    from app.jobs import learn_gate

    with pytest.raises(RuntimeError):
        await learn_gate.run(sessions)
    db.expire_all()
    assert (await db.get(PlaybookVersion, 2)).status == "draft"


def test_the_runner_posts_what_the_replay_run_found(monkeypatch):
    from eval import learn_gate

    calls = []

    def fake_call(method, path, body=None):
        calls.append((method, path, body))
        if method == "GET":
            return {"draft": {"version": 7, "lessons": {"general": ["x"]}, "examples": {}, "note": "n"}, "live": {"version": 6, "lessons": {}, "examples": {}}}
        return {"status": "canary"}

    async def fake_gate(draft, live, files, budget):
        assert draft["version"] == 7 and live["version"] == 6 and files == ["runs/cases.jsonl"]
        return {"regressions": [], "fixed": ["a"], "skipped": [], "n": 12, "cost_inr": 40.0, "live": 6}

    monkeypatch.setattr(learn_gate, "call", fake_call)
    monkeypatch.setattr(learn_gate, "gate", fake_gate)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({"brain": ["bedrock:openai.gpt-oss-120b-1:0@ap-south-1"]}))
    assert learn_gate.main(["runs/cases.jsonl"]) == 0
    assert calls[-1][0:2] == ("POST", "/v2/admin/learn/playbooks/7/replay") and calls[-1][2]["n"] == 12 and calls[-1][2]["fixed"] == ["a"]


def test_the_runner_reports_an_error_and_stops_when_nothing_waits(monkeypatch):
    from eval import learn_gate

    posted = []
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("MODEL_ROUTES", "{}")
    monkeypatch.setattr(learn_gate, "call", lambda m, p, b=None: posted.append(b) or ({"draft": None} if m == "GET" else {}))
    assert learn_gate.main([]) == 0 and posted == [None]

    async def broken(*a):
        raise TimeoutError("cases took too long")

    monkeypatch.setattr(learn_gate, "call", lambda m, p, b=None: posted.append(b) or ({"draft": {"version": 3}, "live": None} if m == "GET" else {}))
    monkeypatch.setattr(learn_gate, "gate", broken)
    assert learn_gate.main([]) == 1 and "TimeoutError" in posted[-1]["error"]
