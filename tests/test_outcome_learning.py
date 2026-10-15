"""Saheli learns from what happens and shows it as a fact the family can see and undo (trust plan A5; founder 2026-10-13:
"If the person always skips a nudge at 2 pm, she learns to try a different time. Save it as a fact the caregiver can see
and undo.")."""

from datetime import timedelta

from app.brain import tools
from app.care import outcomes_learn, store
from app.core import clock
from app.learn import timing
from app.learn.models import ReplyLog
from app.sim.world import SimHost

FAM = "fam-learn"
ME = {"id": "self-l", "name": "Anita Rao", "role": "primary caregiver"}


async def seed_replies(db, at, *, person=ME["id"], bad=(14, 9, 1), good=(16, 9, 7)):
    """bad/good: (IST hour, sent, answered) over the last weeks."""
    now = at("2026-10-28 22:00")
    for hour, sent, answered in (bad, good):
        for i in range(sent):
            when = clock.ist(now - timedelta(days=i + 1)).replace(hour=hour, minute=10)
            db.add(ReplyLog(family_id=FAM, thread_id=person, at=when, kind="proactive", situation="checkin", text="hi",
                            sent_hour=hour, replied=i < answered))
    # other families: an all-family prior
    for i in range(20):
        db.add(ReplyLog(family_id="other", thread_id=f"o{i}", at=now - timedelta(days=2), kind="proactive", situation="checkin",
                        text="hi", sent_hour=11, replied=i % 2 == 0))
    await db.flush()
    return now


async def test_a_time_they_rarely_answer_becomes_a_visible_learning(db, at):
    await store.save_roster(db, FAM, ME, [ME])
    await seed_replies(db, at)
    out = await outcomes_learn.run_family(db, FAM, [ME["id"]])
    assert out["learned"] == 1
    f = await store.active_fact(db, FAM, ME["id"], outcomes_learn.TIMING_KEY)
    assert f.domain == "learned" and f.source_kind == "learned" and f.pinned_at is None
    assert f.value["avoid_hours"] == [14, 15] and f.value["prefer_hours"] == [16, 17]
    assert "2–4 pm" in f.text and "4–6 pm" in f.text and "1 of 9" in f.text
    assert await outcomes_learn.new_since(db, FAM, ME["id"]) == [f.text]


async def test_check_ins_stay_out_of_the_learned_hours(db, at):
    await store.save_roster(db, FAM, ME, [ME])
    await seed_replies(db, at)
    await outcomes_learn.run_family(db, FAM, [ME["id"]])
    at("2026-10-29 14:30")
    why = await timing.quiet_reason(db, FAM, ME["id"], "checkin")
    assert why and "learned" in why
    at("2026-10-29 16:30")
    assert not (await outcomes_learn.avoid_now(db, FAM, ME["id"]))


async def test_too_few_tries_teach_nothing(db, at):
    await store.save_roster(db, FAM, ME, [ME])
    await seed_replies(db, at, bad=(14, 3, 0), good=(16, 3, 3))
    assert await outcomes_learn.timing(db, FAM, ME["id"]) is None


async def test_a_silent_window_is_learned_without_a_better_hour(db, at):
    """Four starts in one two-hour window, answered only hours later, are enough. No other window is required."""
    await store.save_roster(db, FAM, ME, [ME])
    now = at("2026-10-28 22:00")
    for i in range(4):
        when = clock.ist(now - timedelta(days=i + 1)).replace(hour=10, minute=15)
        db.add(ReplyLog(family_id=FAM, thread_id=ME["id"], at=when, kind="proactive", situation="nudge", text="hi",
                        sent_hour=10 + (i % 2), replied=True, reply_delay_s=3 * 3600))
    await db.flush()
    t = await outcomes_learn.timing(db, FAM, ME["id"])
    assert t is not None
    assert t["value"]["avoid_hours"] == [10, 11] and t["value"]["prefer_hours"] == []
    assert "10–12 am" in t["text"] and "stops starting a conversation" in t["text"]
    out = await outcomes_learn.run_family(db, FAM, [ME["id"]])
    assert out["learned"] == 1
    f = await store.active_fact(db, FAM, ME["id"], outcomes_learn.TIMING_KEY)
    assert f and "10–12" in f.text


async def test_undo_is_respected_for_60_days(db, at):
    await store.save_roster(db, FAM, ME, [ME])
    await seed_replies(db, at)
    await outcomes_learn.run_family(db, FAM, [ME["id"]])
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ME, speaker=ME, members=[ME], message_ref="u1")
    await tools.stop(c, {"domain": "learned", "name": "timing_messages", "reason": "I do read at 2 pm", "wrong": True})
    assert await store.active_fact(db, FAM, ME["id"], outcomes_learn.TIMING_KEY) is None
    out = await outcomes_learn.run_family(db, FAM, [ME["id"]])
    assert out == {"learned": 0, "cooling": 1}


async def test_a_caregiver_change_is_pinned_and_learning_never_overwrites_it(db, at):
    await store.save_roster(db, FAM, ME, [ME])
    await seed_replies(db, at)
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="learned", key=outcomes_learn.TIMING_KEY,
                           value={"avoid_hours": [], "prefer_hours": [19]}, text="Best to write in the evening", source_kind="dashboard",
                           stated_by=ME["id"])
    await outcomes_learn.run_family(db, FAM, [ME["id"]])
    f = await store.active_fact(db, FAM, ME["id"], outcomes_learn.TIMING_KEY)
    assert f.text == "Best to write in the evening" and f.pinned_at is not None


async def test_a_habit_skipped_on_the_same_weekday_is_learned(db, at):
    await store.save_roster(db, FAM, ME, [ME])
    now = at("2026-10-28 22:00")
    for i in range(7):
        monday = clock.ist(now).replace(hour=19) - timedelta(days=(clock.ist(now).weekday() - 0) % 7 + 7 * i)
        await store.record_event(db, family_id=FAM, subject_id=ME["id"], kind="activity_skipped" if i < 6 else "activity_done",
                                 summary="walk", payload={"key": "activity:walk", "label": "Evening walk"}, at=monday)
    out = await outcomes_learn.run_family(db, FAM, [ME["id"]])
    assert out["learned"] == 1
    f = await store.active_fact(db, FAM, ME["id"], "learned:habit_walk_monday")
    assert f and "Mondays" in f.text and f.value["skipped"] == 6


async def test_the_brain_cannot_write_a_learning_from_chat(db, at):
    import pytest

    await store.save_roster(db, FAM, ME, [ME])
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ME, speaker=ME, members=[ME], message_ref="u2")
    with pytest.raises(tools.ToolRefused):
        await tools.remember(c, {"domain": "learned", "name": "x", "details": {}, "sentence": "x"})
