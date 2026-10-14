"""A missing fact is data, and a follow-up wake is the clock time they gave."""

from datetime import datetime
from types import SimpleNamespace

from app.brain.tools import wake_from_clock
from app.care.gaps import _slept, asked_keys, food_gaps, ledger_topics, nudge_covers, pick, render
from app.core import clock


def test_water_is_the_gap_after_a_meal_and_an_asked_one_drops_out():
    assert food_gaps(True, False, 20) == [("water", "no water logged today, and a meal is")]
    assert food_gaps(False, False, 20) == [("meal", "no meal logged today")]
    assert food_gaps(False, False, 9) == []
    items = food_gaps(True, False, 20) + [("sleep", "when they sleep is not on record"), ("habit:gym", "today's gym has no result yet")]
    assert pick(items, asked_keys(["water"], ["meal:dinner", "activity:gym:done"])) == [("sleep", "when they sleep is not on record")]
    assert "meal:dinner" in ledger_topics("meal") and "activity:gym:done" in ledger_topics("habit:gym")
    assert ledger_topics("water") == ["water", "water:count"] and ledger_topics("sleep") == ["sleep", "sleep:when"]


def test_a_nudge_sent_today_covers_that_meal_water_or_sleep():
    today = "2026-10-15"
    assert nudge_covers("meal", "meal:dinner", today, today) == "meal"
    assert nudge_covers("water", "water:count", today, today) == "water"
    assert nudge_covers("sleep", "sleep:when", today, today) == "sleep"
    assert nudge_covers("water", "water:count", "2026-10-14", today) is None
    assert nudge_covers("collect", "sleep", today, today) is None
    assert _slept(SimpleNamespace(kind="sleep", payload={}))
    assert _slept(SimpleNamespace(kind="activity_done", payload={"type": "sleep"}))
    assert not _slept(SimpleNamespace(kind="routine", payload={"key": "routine:cook"}))
    left = [("water", "no water"), ("sleep", "when")]
    assert pick(left, {"water", "sleep"}) == []
    assert "gap_asked" in render("Riyan", left)


def test_a_follow_up_wakes_at_their_clock_time():
    token = clock.set_now(datetime.fromisoformat("2026-10-15T14:00:00+05:30"))
    try:
        later = wake_from_clock("22:15")
        assert clock.ist(later).strftime("%H:%M") == "22:15" and clock.ist(later).date().isoformat() == "2026-10-15"
        nxt = wake_from_clock("10:00")
        assert clock.ist(nxt).date().isoformat() == "2026-10-16"
        assert wake_from_clock("tonight") is None
    finally:
        clock.reset_now(token)
