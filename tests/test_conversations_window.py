"""The conversations window never says 'closed since yesterday' on a day they already answered."""

from datetime import datetime, timedelta, timezone

from app.care.conversations import opener, window_open

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 15, 10, 0, tzinfo=IST)
YESTERDAY = datetime(2026, 10, 14, 14, 0, tzinfo=IST)


def test_an_answer_today_opens_a_closed_gate():
    assert window_open(gate_open=False, last_in=YESTERDAY, answered_today=True, now=NOW) is True


def test_no_reply_since_yesterday_stays_closed():
    assert window_open(gate_open=False, last_in=YESTERDAY, answered_today=False, now=NOW) is False


def test_a_recent_message_with_an_open_gate_is_open():
    assert window_open(gate_open=True, last_in=NOW - timedelta(hours=2), answered_today=False, now=NOW) is True


def test_a_missing_gate_still_opens_for_a_recent_message():
    assert window_open(gate_open=None, last_in=NOW - timedelta(hours=2), answered_today=False, now=NOW) is True


def test_reopen_template_only_when_a_row_says_so():
    assert opener([{"reason": "window_closed", "at": "a", "text": ""}]) is None
    sent = opener([{"reason": "waiting:opener", "at": "08:00", "text": "Good morning"}])
    assert sent is not None and sent["at"] == "08:00" and sent["reason"] == "waiting:opener"
    update = opener([{"reason": "waiting:update", "at": "08:05", "text": ""}])
    assert update is not None and update["reason"] == "waiting:update"
