"""The reply guards on a frozen sample of real simulated replies (round 3), graded by the judge.

Each false alarm on a good reply costs a rewrite call and delay; each miss on a bad one reaches a family.
If a guard change moves these rates past the margins, regenerate the sample with eval/replay_guards.py
--freeze only after reading why (journal/tests)."""

import json
from pathlib import Path

from eval.replay_guards import rates

FIXTURE = Path(__file__).parent / "fixtures" / "guard_replay.json"
MARGIN = 0.02


def test_guards_do_not_get_noisier_or_blinder():
    data = json.loads(FIXTURE.read_text())
    frozen, now = data["rates_at_freeze"], rates(data["cases"])
    for check, r in now.items():
        assert r["passed"] <= frozen[check]["passed"] + MARGIN, f"{check} now fires on {r['passed']:.1%} of good replies (was {frozen[check]['passed']:.1%})"
        assert r["failed"] >= frozen[check]["failed"] - MARGIN, f"{check} now catches {r['failed']:.1%} of bad replies (was {frozen[check]['failed']:.1%})"


def test_false_alarm_ceiling():
    data = json.loads(FIXTURE.read_text())
    now = rates(data["cases"])
    assert sum(r["passed"] for r in now.values()) <= 0.20  # at most ~1 in 5 good replies gets a rewrite
