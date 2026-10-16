"""The night spends at most ₹150, and the owner check requires a review outside your area."""

from app.jobs.saheli_night import CAPS, night_env


def test_the_night_cap_stays_150():
    env = night_env({"PROBES_MAX_INR": "9999", "REPLAY_DATABASE_URL": "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/kawach_replay"})
    assert env["PROBES_MAX_INR"] == CAPS["PROBES_MAX_INR"]
    assert env["NIGHT_CAP_INR"] == "150"
    assert int(env["PROBES_MAX_INR"]) + int(env["JUDGE_MAX_INR"]) + int(env["RECALL_MAX_INR"]) + int(env["NUDGE_EVAL_MAX_INR"]) + int(env["POINTS_MAX_INR"]) == 150
