"""A hundred clock cases for where a log sits in the diary. The time decides, not a food's name."""

from datetime import datetime, timedelta

from app.core import clock
from app.nutrition import portions as P

# (hour, minute, dinner already saved, meal, day offset from the clock date)
NIGHT_SAVED = [(0, 0), (0, 15), (0, 30), (1, 0), (1, 15), (1, 30), (2, 0), (2, 15), (2, 30), (2, 36),
               (3, 0), (3, 15), (3, 30), (3, 45), (4, 0), (4, 30), (4, 58), (5, 0), (5, 30), (5, 59)]
NIGHT_OPEN = [(0, 0), (0, 15), (0, 30), (1, 0), (1, 20), (2, 0), (2, 15), (2, 36), (3, 0), (3, 15),
              (4, 0), (4, 58), (5, 0), (5, 59)]
DAY = [
    (6, 0, "breakfast"), (6, 30, "breakfast"), (7, 30, "breakfast"), (8, 0, "breakfast"), (8, 30, "breakfast"),
    (9, 15, "breakfast"), (10, 0, "breakfast"), (10, 59, "breakfast"),
    (11, 0, "lunch"), (12, 0, "lunch"), (12, 30, "lunch"), (13, 0, "lunch"), (14, 45, "lunch"), (15, 0, "lunch"), (15, 29, "lunch"),
    (15, 30, "snack"), (16, 0, "snack"), (16, 30, "snack"), (17, 20, "snack"), (18, 0, "snack"), (18, 30, "snack"), (18, 59, "snack"),
]
EVENING = [(19, 0), (19, 45), (20, 0), (20, 30), (21, 0), (21, 30), (22, 0), (22, 15), (23, 0), (23, 40), (23, 59)]

CASES: list[tuple[int, int, bool, str, int]] = []
for h, m in NIGHT_SAVED:
    CASES.append((h, m, True, "late_night", 0))
for h, m in NIGHT_OPEN:
    CASES.append((h, m, False, "dinner", -1))
for h, m, meal in DAY:
    CASES.append((h, m, True, meal, 0))
    CASES.append((h, m, False, meal, 0))
for h, m in EVENING:
    CASES.append((h, m, True, "after_dinner", 0))
    CASES.append((h, m, False, "dinner", 0))


def test_one_hundred_clock_cases_land_on_the_right_diary_slot():
    assert len(CASES) == 100
    assert len({(h, m, saved) for h, m, saved, _, _ in CASES}) == 100
    # the founder's night: 02:36 and 04:58, dinner already eaten, stay on the 15th as late night
    assert (2, 36, True, "late_night", 0) in CASES
    assert (4, 58, True, "late_night", 0) in CASES
    clock_day = datetime(2026, 10, 15).date()
    for h, m, saved, meal, offset in CASES:
        at = datetime(2026, 10, 15, h, m, tzinfo=clock.IST)
        got = P.place(at, dinner_saved=saved)
        assert got.meal == meal, (h, m, saved, got)
        assert got.day == (clock_day + timedelta(days=offset)).isoformat()
        # a breakfast label from the brain does not move a night log onto breakfast
        if h < 6 or h >= 19:
            named = P.resolve(at, dinner_saved=saved, named="breakfast")
            assert named.meal == meal and named.day == got.day


def test_water_at_night_is_not_a_dinner_and_not_breakfast():
    at = datetime(2026, 10, 15, 3, 0, tzinfo=clock.IST)
    got = P.place(at, dinner_saved=False, only_water=True)
    assert got.meal == "late_night" and got.day == "2026-10-15"
    evening = P.place(datetime(2026, 10, 15, 21, 0, tzinfo=clock.IST), dinner_saved=False, only_water=True)
    assert evening.meal == "other" and evening.day == "2026-10-15"
