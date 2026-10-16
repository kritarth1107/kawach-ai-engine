"""The pre-deploy check moves the service only when the replay actually passed."""

from app.jobs.deploy_check import verdict
from app.jobs.saheli_night import CAPS, night_env


def test_a_pass_is_the_only_zero():
    assert verdict({"passed": True}) == 0


def test_a_regression_blocks():
    assert verdict({"passed": False, "failures": ["case a"]}) == 1


def test_a_refused_model_blocks():
    assert verdict({"unavailable": True, "passed": False}) == 1


def test_a_missing_summary_blocks():
    assert verdict(None) == 1


def test_the_night_cap_stays_150():
    env = night_env({"PROBES_MAX_INR": "9999", "REPLAY_DATABASE_URL": "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/kawach_replay"})
    assert env["PROBES_MAX_INR"] == CAPS["PROBES_MAX_INR"]
    assert env["NIGHT_CAP_INR"] == "150"
    assert int(env["PROBES_MAX_INR"]) + int(env["JUDGE_MAX_INR"]) + int(env["RECALL_MAX_INR"]) + int(env["NUDGE_EVAL_MAX_INR"]) + int(env["POINTS_MAX_INR"]) == 150
