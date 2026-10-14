"""A missing fact is data, and a follow-up wake is the clock time they gave."""

from datetime import datetime

from app.brain.tools import wake_from_clock
from app.care.gaps import asked_keys, food_gaps, ledger_topics, pick, render
from app.core import clock


def test_water_is_the_gap_after_a_meal_and_an_asked_one_drops_out():
    assert food_gaps(True, False, 20) == [("water", "no water logged today, and a meal is")]
    assert food_gaps(False, False, 20) == [("meal", "no meal logged today")]
    assert food_gaps(False, False, 9) == []
    items = food_gaps(True, False, 20) + [("sleep", "when they sleep is not on record"), ("habit:gym", "today's gym has no result yet")]
    assert pick(items, asked_keys(["water"], ["meal:dinner", "activity:gym:done"])) == [("sleep", "when they sleep is not on record")]
    assert "meal:dinner" in ledger_topics("meal") and "activity:gym:done" in ledger_topics("habit:gym")
    assert "gap_asked" in render("Riyan", items)


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
