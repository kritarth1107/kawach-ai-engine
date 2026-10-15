"""The day's judge sample: a few ordinary turns a family, a fixed cap for everyone, and the turns that are never skipped."""

from app.quality import sample


def test_three_ordinary_turns_and_the_day_stops_at_the_cap(monkeypatch):
    monkeypatch.setenv("JUDGE_PER_FAMILY", "3")
    monkeypatch.setenv("JUDGE_DAILY_CAP", "90")
    rows = list(range(10))
    assert sample.pick(rows, mine=0, total=0) == [0, 1, 2]
    assert sample.pick(rows, mine=2, total=0) == [0]
    assert sample.pick(rows, mine=0, total=89) == [0]
    assert sample.pick(rows, mine=0, total=90) == []


def test_a_small_day_includes_every_family_and_a_large_day_does_not(monkeypatch):
    monkeypatch.setenv("JUDGE_PER_FAMILY", "3")
    monkeypatch.setenv("JUDGE_DAILY_CAP", "90")
    assert sample.admitted("fam-a", "2026-10-16", 5) is True
    admitted = [fid for fid in (f"fam-{i}" for i in range(5000)) if sample.admitted(fid, "2026-10-16", 5000)]
    assert 15 <= len(admitted) <= 50  # about 30 slots; the hard cap is what stops the spend


def test_an_alert_a_guard_and_a_founder_report_are_not_ordinary():
    class Row:
        def __init__(self, **kw):
            self.judge_band = kw.get("band")
            self.situation = kw.get("situation", "chit_chat")
            self.tools = kw.get("tools", [])
            self.corrected = kw.get("corrected", False)
            self.thread_id = kw.get("thread", "p")

    assert sample.always_logged(Row(situation="emergency"), set()) is True
    assert sample.always_logged(Row(tools=["alert_caregiver"]), set()) is True
    assert sample.always_logged(Row(band="always"), set()) is True
    assert sample.always_logged(Row(corrected=True), set()) is True
    assert sample.always_logged(Row(), {"p"}) is True
    assert sample.always_logged(Row(), set()) is False
    assert sample.always_live(voice_unsure=True, alerts=False, health=False, tools=[], situation="chit_chat") is True
    assert sample.always_live(voice_unsure=False, alerts=False, health=False, tools=[], situation="chit_chat") is False
