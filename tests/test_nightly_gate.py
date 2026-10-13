"""Trust plan C2.3, founder D8: engine deploys wait while the latest full nightly on master failed (scripts/nightly_gate.py)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("nightly_gate", Path(__file__).parents[1] / "scripts" / "nightly_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)  # type: ignore[union-attr]


def run(conclusion, at="2026-10-14T19:30:00Z", title="Nightly full", branch="master", event="schedule", rid=1):
    return {"id": rid, "conclusion": conclusion, "created_at": at, "display_title": title, "head_branch": branch, "event": event}


def no_events(n):
    return []


def decide(runs, **kw):
    kw.setdefault("message", "A change")
    kw.setdefault("prs", [])
    kw.setdefault("events_of", no_events)
    kw.setdefault("founder", "kritarth1107")
    return gate.decide(runs, **kw)


def test_green_latest_night_allows_red_blocks():
    assert decide([run("success")])[0] is True
    ok, why = decide([run("failure", rid=9), run("success", at="2026-10-13T19:30:00Z")])
    assert ok is False and "FAILED" in why and "run 9" in why


def test_only_full_master_nights_count():
    runs = [run("failure", title="Nightly ci", at="2026-10-15T01:00:00Z"), run("failure", title="Nightly full fixture", at="2026-10-15T02:00:00Z"),
            run("failure", branch="daily-scores", at="2026-10-15T03:00:00Z"), run("cancelled", at="2026-10-15T04:00:00Z"),
            run("failure", event="push", at="2026-10-15T05:00:00Z"), run("success")]
    assert decide(runs)[0] is True
    assert decide([])[0] is True  # no full night yet


def test_the_fix_itself_goes_through():
    assert decide([run("failure")], message="Language fix for Malayalam [fixes-nightly]")[0] is True


def test_the_founders_label_overrides_but_nobody_elses():
    prs = [{"number": 7, "labels": [{"name": "nightly-override"}]}]
    by_founder = lambda n: [{"event": "labeled", "label": {"name": "nightly-override"}, "actor": {"login": "kritarth1107"}}]  # noqa: E731
    by_other = lambda n: [{"event": "labeled", "label": {"name": "nightly-override"}, "actor": {"login": "someone"}}]  # noqa: E731
    assert decide([run("failure")], prs=prs, events_of=by_founder)[0] is True
    assert decide([run("failure")], prs=prs, events_of=by_other)[0] is False
    assert decide([run("failure")], prs=[{"number": 8, "labels": []}], events_of=by_founder)[0] is False


def test_a_manual_override_only_by_the_founder():
    assert decide([run("failure")], actor="kritarth1107", override_input=True)[0] is True
    assert decide([run("failure")], actor="kuchhu-bot", override_input=True)[0] is False


def test_dry_run_with_a_red_night_reports_blocked(monkeypatch, capsys):
    monkeypatch.setenv("DRY_LATEST", "failure")
    monkeypatch.setenv("COMMIT_MESSAGE", "an ordinary change")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert gate.main([]) == 1 and "DEPLOY BLOCKED" in capsys.readouterr().out
    assert gate.main(["--dry-run"]) == 0
    monkeypatch.setenv("DRY_LATEST", "success")
    assert gate.main([]) == 0
