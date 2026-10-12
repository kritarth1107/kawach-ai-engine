"""Activities and habits (founder 2026-10-13): a stated routine is saved at once as an activity, not a reminder; Saheli
nudges before (fitness), checks in after (did it happen, what did they do), logs what they say with readings going to
health data, counts streaks; something given up gets an evening check; a routine sends nothing. 100+ catalogue types."""

import json
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.brain import tools, wake
from app.care import activities, store
from app.care.models import CareEvent, OpenLoop
from app.sim.world import SimHost

FAM = "fam-act"
ME = {"id": "self-a", "name": "Rahul", "role": "primary caregiver"}


def ctx(db, host=None, said=""):
    return tools.TurnCtx(session=db, host=host or SimHost(), family_id=FAM, elder=ME, speaker=ME, members=[ME], user_text=said)


async def call(c, name, args):
    out, err = await tools.run(c, name, args)
    return json.loads(out), err


async def loops(db) -> list[OpenLoop]:
    return list((await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "activity").order_by(OpenLoop.wake_at))).scalars())


def test_the_catalogue_has_over_a_hundred_activities_across_life():
    assert len(activities.CATALOG) >= 100
    cats = {a["category"] for a in activities.CATALOG.values()}
    assert {"fitness", "health", "mind", "spiritual", "food", "sleep", "social", "work", "home", "quit", "hobby", "care"} <= cats
    assert activities.CATALOG["gym"]["kind"] == "habit" and activities.CATALOG["lunch"]["kind"] == "routine"
    assert activities.CATALOG["no_smoking"]["kind"] == "quit" and activities.CATALOG["bills"]["kind"] == "task"


async def test_rahuls_gym_is_saved_as_an_activity_with_a_nudge_before_and_a_check_in_after(db, at):
    """Live 2026-10-13: he told his gym schedule and had to ask for reminders separately."""
    at("2026-10-13 07:00")  # a Tuesday
    await store.save_roster(db, FAM, ME, [ME])
    out, err = await call(ctx(db, said="I have gym at 9:30, Monday to Saturday"), "activity",
                          {"action": "add", "type": "gym", "days": [0, 1, 2, 3, 4, 5], "start": "09:30"})
    assert not err and out["key"] == "activity:gym" and out["saved"] == "Gym (Mon–Sat, 09:30–10:30)"
    assert "check in after" in out["what_happens"] and "30 min before" in out["what_happens"]
    ls = await loops(db)
    phases = [(l.detail["date"], l.detail["phase"], l.wake_at) for l in ls]
    assert ("2026-10-13", "before", ls[0].wake_at) == phases[0] and ls[0].wake_at.astimezone().strftime("%H:%M") and len(ls) == 4
    today = [l for l in ls if l.detail["date"] == "2026-10-13"]
    assert sorted(l.detail["phase"] for l in today) == ["after", "before"]
    from app.core import clock

    assert {l.detail["phase"]: clock.ist(l.wake_at).strftime("%H:%M") for l in today} == {"before": "09:00", "after": "10:45"}
    # planning again (the wake job) adds nothing new
    assert await activities.plan_for(db, FAM, ME["id"], await store.active_fact(db, FAM, ME["id"], "activity:gym")) == 0


async def test_the_after_check_in_asks_and_log_activity_records_it_with_readings_and_closes_it(db, at):
    at("2026-10-13 07:00")
    await store.save_roster(db, FAM, ME, [ME])
    await call(ctx(db), "activity", {"action": "add", "type": "gym", "days": [0, 1, 2, 3, 4, 5], "start": "09:30", "goal": "lose 5 kg"})
    after = next(l for l in await loops(db) if l.detail["phase"] == "after" and l.detail["date"] == "2026-10-13")
    at("2026-10-13 10:50")
    prompt = await wake.wake_prompt(db, after, ME["id"])
    assert "[Activity: after]" in prompt and "log_activity" in prompt and "lose 5 kg" in prompt and "exercises" in prompt
    host = SimHost()
    out, err = await call(ctx(db, host, said="haan aa gaya, chest and triceps kiya 50 min, weight 89.8"), "log_activity",
                          {"key": "activity:gym", "done": True, "summary": "Chest and triceps, 50 minutes", "duration_min": 50,
                           "details": {"exercises": ["bench press", "dips"]}, "readings": [{"kind": "weight", "value": "89.8", "unit": "kg"}]})
    assert not err and out["streak"] == 1 and out["readings"][0]["kind"] == "weight"
    assert any(v.get("kind") == "weight" for v in host.world.vitals), "the weight reached health data"
    ev = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "activity_done"))).scalars().one()
    assert ev.payload["details"]["duration_min"] == 50 and ev.payload["key"] == "activity:gym"
    await db.refresh(after)
    assert after.status == "done", "the check-in is answered"
    # a done day makes the next prompt for today nothing
    before_tomorrow = next(l for l in await loops(db) if l.detail["date"] == "2026-10-14" and l.detail["phase"] == "before")
    assert await wake.wake_prompt(db, before_tomorrow, ME["id"]) is not None


async def test_streak_counts_scheduled_days_and_skips_days_off(db, at):
    at("2026-10-10 09:00")  # Saturday
    await store.save_roster(db, FAM, ME, [ME])
    await call(ctx(db), "activity", {"action": "add", "type": "gym", "days": [0, 1, 2, 3, 4, 5], "start": "09:30"})
    for day in ("2026-10-10", "2026-10-12", "2026-10-13"):  # Sat, (Sun off), Mon, Tue
        at(f"{day} 11:00")
        await call(ctx(db), "log_activity", {"key": "activity:gym", "done": True, "summary": "done"})
    rows = await activities.status(db, FAM, ME["id"])
    assert rows[0]["streak"] == 3 and rows[0]["today"] == "done" and rows[0]["week_done"] == 3


async def test_something_given_up_gets_an_evening_check_and_a_routine_sends_nothing(db, at):
    at("2026-10-13 08:00")
    await store.save_roster(db, FAM, ME, [ME])
    out, _ = await call(ctx(db), "activity", {"action": "add", "type": "no_smoking"})
    assert "every evening at 20:00" in out["what_happens"]
    q = [l for l in await loops(db) if l.detail["activity_key"] == "activity:no_smoking"]
    assert q and all(l.detail["phase"] == "after" for l in q)
    prompt = await wake.wake_prompt(db, q[0], ME["id"])
    assert "[Activity: evening check]" in prompt and "no shame" in prompt
    out, _ = await call(ctx(db), "activity", {"action": "add", "type": "lunch", "start": "13:30"})
    assert "time my messages" in out["what_happens"]
    assert not [l for l in await loops(db) if l.detail["activity_key"] == "activity:lunch"]


async def test_change_pause_and_stop_replan_or_drop_the_check_ins(db, at):
    at("2026-10-13 07:00")
    await store.save_roster(db, FAM, ME, [ME])
    await call(ctx(db), "activity", {"action": "add", "type": "evening_walk", "start": "18:00"})
    out, err = await call(ctx(db), "activity", {"action": "change", "key": "activity:evening_walk", "start": "19:00"})
    assert not err and out["saved"] == "Evening walk (every day, 19:00–19:30)"
    live = [l for l in await loops(db) if l.status == "open"]
    from app.core import clock

    assert {clock.ist(l.wake_at).strftime("%H:%M") for l in live if l.detail["date"] == "2026-10-13"} == {"18:30", "19:45"}, "a nudge 30 min before and the check-in after"
    await call(ctx(db), "activity", {"action": "pause", "key": "activity:evening_walk"})
    assert not [l for l in await loops(db) if l.status == "open"]
    await call(ctx(db), "activity", {"action": "stop", "key": "activity:evening_walk"})
    assert not await activities.active(db, FAM, ME["id"])


async def test_a_habit_needs_a_time_and_an_unknown_one_needs_a_label(db, at):
    at("2026-10-13 07:00")
    await store.save_roster(db, FAM, ME, [ME])
    out, err = await call(ctx(db), "activity", {"action": "add", "type": "gym"})
    assert err and "what time" in out["refused"]
    out, err = await call(ctx(db), "activity", {"action": "add", "type": "kite_flying", "start": "17:00"})
    assert err and "label" in out["refused"]
    out, err = await call(ctx(db), "activity", {"action": "add", "type": "custom", "label": "Kite flying", "start": "17:00"})
    assert not err and out["key"] == "activity:kite_flying"


async def test_routines_already_told_become_activities_and_their_own_reminder_replaces_the_nudge(db, at, monkeypatch):
    """Live 2026-10-13: Rahul's gym schedule sat in the record as a routine and was never scheduled."""
    from app.llm import router
    from app.llm.router import LLMReply

    class Reader:
        async def complete(self, route, **kw):
            return LLMReply(text=json.dumps({"activities": [
                {"type": "gym", "label": "Gym", "kind": "habit", "days": [0, 1, 2, 3, 4, 5], "start": "09:30", "end": None},
                {"type": "wake_up", "label": "Wake up", "kind": "routine", "days": [], "start": "08:30", "end": None},
                {"type": "gym", "label": "Gym again", "kind": "habit", "days": [], "start": None},
            ]}), tool_calls=[], model="fake")

    router.register_provider("fake", Reader())
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        at("2026-10-13 07:00")
        await store.save_roster(db, FAM, ME, [ME])
        await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="routine", key="routine:daily_schedule",
                               value={"gym": "09:30", "wake": "08:30"}, text="Wakes up at 08:30, gym at 09:30", source_kind="elder_said")
        host = SimHost()
        await host.call("create_reminder", {"text": "Leave for the gym", "schedule": {"times": ["09:00"], "repeat": "daily"}},
                        family_id=FAM, subject_id=ME["id"], actor_id=ME["id"])
        made = await activities.adopt(db, host, FAM, ME["id"])
        assert made == ["activity:gym", "activity:wake_up"]
        gym = activities.normalise((await store.active_fact(db, FAM, ME["id"], "activity:gym")).value)
        assert gym["days"] == [0, 1, 2, 3, 4, 5] and gym["start"] == "09:30" and gym["nudge_before"] == 0, "his 09:00 reminder already nudges"
        assert [l.detail["phase"] for l in await loops(db) if l.detail["date"] == "2026-10-13"] == ["after"]
        assert await activities.adopt(db, host, FAM, ME["id"]) == [], "running again adds nothing"
    finally:
        router._providers.pop("fake", None)


async def test_their_own_reminder_before_a_habit_means_no_second_nudge_when_saheli_saves_it(db, at):
    """Live 2026-10-13: Rahul and Kritarth already had 'leave for the gym' reminders; the backend lists times at the top level."""
    at("2026-10-13 07:00")
    await store.save_roster(db, FAM, ME, [ME])
    host = SimHost()
    await host.call("create_reminder", {"text": "Leave for the gym", "schedule": {"times": ["09:00"], "repeat": "daily"}},
                    family_id=FAM, subject_id=ME["id"], actor_id=ME["id"])
    out, err = await call(ctx(db, host), "activity", {"action": "add", "type": "gym", "days": [0, 1, 2, 3, 4, 5], "start": "09:30"})
    assert not err and "nudge" not in out["what_happens"]
    assert [l.detail["phase"] for l in await loops(db) if l.detail["date"] == "2026-10-13"] == ["after"]


async def test_an_activity_adopted_wrongly_is_refreshed_and_one_they_told_is_left_alone(db, at, monkeypatch):
    from app.llm import router
    from app.llm.router import LLMReply

    class Reader:
        async def complete(self, route, **kw):
            return LLMReply(text=json.dumps({"activities": [
                {"type": "gym", "label": "goes to the gym", "kind": "habit", "days": [0, 1, 2, 3, 4, 5], "start": "20:00", "end": "21:00"},
                {"type": "evening_walk", "label": "walks", "kind": "habit", "days": [], "start": "18:00"},
            ]}), tool_calls=[], model="fake")

    router.register_provider("fake", Reader())
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        at("2026-10-13 07:00")
        await store.save_roster(db, FAM, ME, [ME])
        await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="routine", key="routine:gym", value={"time": "20:00"},
                               text="Goes to the gym 8-9 pm, Mon-Sat", source_kind="elder_said")
        # adopted before the fix: the record's words as the label and a nudge despite their own 19:30 reminder
        await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="activity", key="activity:gym",
                               value=activities.normalise({"type": "gym", "label": "goes to the gym", "kind": "habit", "days": [0, 1, 2, 3, 4, 5],
                                                           "start": "20:00", "end": "21:00"}), text="goes to the gym", source_kind="inferred")
        # the walk they told Saheli themselves, at another time
        await call(ctx(db), "activity", {"action": "add", "type": "evening_walk", "start": "19:00"})
        host = SimHost()
        await host.call("create_reminder", {"text": "Head out to the gym", "schedule": {"times": ["19:30"], "repeat": "daily"}},
                        family_id=FAM, subject_id=ME["id"], actor_id=ME["id"])
        made = await activities.adopt(db, host, FAM, ME["id"])
        assert made == ["activity:gym"]
        gym = activities.normalise((await store.active_fact(db, FAM, ME["id"], "activity:gym")).value)
        assert gym["label"] == "Gym" and gym["nudge_before"] == 0
        walk = activities.normalise((await store.active_fact(db, FAM, ME["id"], "activity:evening_walk")).value)
        assert walk["start"] == "19:00", "what they told Saheli stays"
    finally:
        router._providers.pop("fake", None)


async def test_quiet_hours_follow_their_own_night_when_they_told_saheli(db, at):
    """Live 2026-10-13: Kritarth sleeps at 3 and wakes at 1:30 pm; his 9:15 pm gym check-in must not wait till morning.
    Without their own sleep times the household night (21:00–08:00) stays."""
    from app.brain import wake
    from app.core import clock

    at("2026-10-13 21:15")
    await store.save_roster(db, FAM, ME, [ME])
    assert (await wake.quiet_for(db, FAM, ME["id"]))[0], "no sleep times known: the usual night"
    await call(ctx(db), "activity", {"action": "add", "type": "sleep", "start": "03:00", "kind": "routine"})
    await call(ctx(db), "activity", {"action": "add", "type": "wake_up", "start": "13:30", "kind": "routine"})
    assert (await wake.quiet_for(db, FAM, ME["id"])) == (False, None), "9:15 pm is his evening"
    at("2026-10-14 05:00")
    quiet, morning = await wake.quiet_for(db, FAM, ME["id"])
    assert quiet and clock.ist(morning).strftime("%d %H:%M") == "14 14:00", "his morning: wake-up plus 30 min"
