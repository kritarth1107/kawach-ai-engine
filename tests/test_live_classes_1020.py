"""Classes of live mistakes (20 Oct), each tested beyond the one example: an answer binds to the newest question;
exact counts, sums and dates are computed; web results never spoil a story; a fixed line names every dose and reading;
background model work waits after a 429. Synthetic data only."""

import asyncio
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.brain import confirm, guards
from app.brain.tools.exact import arithmetic, compute, count
from app.brain.tools.trips import SEARCH_NOTE
from app.brain.tools.base import ToolRefused
from app.core import clock
from app.llm import router


def turn(role, minutes_ago, text="", pending=None):
    return SimpleNamespace(role=role, text=text, at=clock.now() - timedelta(minutes=minutes_ago),
                           meta={"pending_saves": pending} if pending else {})


SAVE = [{"id": "s1", "tool": "log_food", "sig": "log_food:x", "args": {}, "what": "rasam rice", "for": "Vish"}]


def test_an_answer_belongs_to_the_newest_question(at):
    at("2026-10-20 17:40")
    # a read-back hours ago, then a new question: "Yup" is about the new question
    assert confirm.pending_from([turn("assistant", 410, "Log rasam rice for lunch?", SAVE), turn("assistant", 25, "What did you have for a snack?")]) == []
    # the read-back is her latest message: it is waiting
    assert [p["id"] for p in confirm.pending_from([turn("assistant", 1, "Log rasam rice for lunch?", SAVE)])] == ["s1"]
    # a short aside after the read-back (no question) keeps it waiting
    assert confirm.pending_from([turn("assistant", 3, "Log it?", SAVE), turn("assistant", 2, "Get well soon.")])
    # older than the window: not waiting, whatever came after
    assert confirm.pending_from([turn("assistant", 45, "Log it?", SAVE)]) == []


@pytest.mark.parametrize("text,of,n", [("petting zoo", "r", 0), ("strawberry", "r", 3), ("Mississippi", "s", 4),
                                       ("banana", "an", 2), ("Durga Puja", "a", 2)])
def test_counts_are_computed(text, of, n):
    assert count(text, of) == n


def test_sums_and_dates_are_computed():
    assert arithmetic("12*7+5") == 89 and arithmetic("(2+3)**2") == 25 and arithmetic("1,500 / 3") == 500
    for bad in ("__import__('os')", "a+1", "2**1000"):
        with pytest.raises((ValueError, SyntaxError)):
            arithmetic(bad)
    out = asyncio.run(compute(SimpleNamespace(), {"kind": "days_between", "text": "2026-10-20", "of": "2026-12-25"}))
    assert out == {"days": 66}
    assert asyncio.run(compute(SimpleNamespace(), {"kind": "weekday", "text": "2026-10-20"}))["weekday"] == "Tuesday"
    with pytest.raises(ToolRefused):
        asyncio.run(compute(SimpleNamespace(), {"kind": "math", "text": "1/0"}))


def test_every_web_result_carries_the_no_spoiler_rule():
    for word in ("show", "film", "book", "match", "ending", "twist", "cameo", "later season", "ask"):
        assert word in SEARCH_NOTE


def test_a_fixed_line_names_every_dose():
    line = guards.doses_logged({"saved": {"language": "en"}}, [("Thyronorm", ["07:30"]), ("Ecosprin 75", ["13:00"]), ("Atorva", [])])
    assert line == "Logged as taken: Thyronorm (07:30), Ecosprin 75 (13:00), Atorva."
    assert "Ecosprin 75" in guards.doses_logged({"saved": {"language": "bn"}, "now": "bengali"}, [("Thyronorm", ["07:30"]), ("Ecosprin 75", ["13:00"])])
    assert guards.readable("water_intake_glasses 6") == "water intake glasses 6"


def test_background_work_waits_after_a_429_and_replies_do_not(monkeypatch):
    slept = []

    async def sleep(s):
        slept.append(s)

    monkeypatch.setattr(router.asyncio, "sleep", sleep)
    monkeypatch.setattr(router, "_throttled_at", time.monotonic())
    monkeypatch.setattr(router, "BACKGROUND_PAUSE_S", 30.0)
    asyncio.run(router._yield_to_live("brain"))
    assert slept == []
    asyncio.run(router._yield_to_live("extract"))
    assert slept and 0 < slept[0] <= router.BACKGROUND_PAUSE_S


def test_a_skip_asks_why_and_tells_family_only_for_a_care_recipient():
    from app.brain.tools.health import skip_next

    elder = SimpleNamespace(elder_id="e1", members=[{"id": "e1", "role": "elder"}, {"id": "c1", "role": "primary caregiver"}])
    out = skip_next(elder, "e1", "Metformin", "skipped")
    assert "ask" in out and "why" in out and "alert_caregiver" in out and "encourage once" in out
    own = skip_next(elder, "c1", "Magnesium", "skipped")
    assert "nobody else is told" in own and "alert_caregiver" not in own
    assert "They said: doctor paused it" in skip_next(elder, "e1", "Metformin", "doctor paused it")


def test_a_repeated_nudge_topic_gives_way_to_another(monkeypatch):
    from app.care.nudges import compose as C

    drafts = iter([{"option_id": "o1", "text": "Who is your favourite character?"}, {"option_id": "o2", "text": "How many glasses of water so far?"}])
    seen = []

    async def fake_compose(ctx, problems=None):
        seen.append(problems)
        return next(drafts)

    async def fine(ctx, text):
        return []

    monkeypatch.setattr(C, "compose", fake_compose)
    monkeypatch.setattr(C, "grounded", fine)
    monkeypatch.setattr(C, "verify", fine)
    monkeypatch.setattr(C, "check", lambda d, ctx: ["this question was already asked today: 'Who is'"] if d["option_id"] == "o1" else [])
    ctx = SimpleNamespace(options=[SimpleNamespace(id="o1"), SimpleNamespace(id="o2")], off=set())
    draft, problems = asyncio.run(C.write_and_check(ctx))
    assert draft["option_id"] == "o2" and not problems
    assert "o1" in seen[1][-1] and "different option_id" in seen[1][-1]
