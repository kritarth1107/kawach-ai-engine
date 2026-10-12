"""Saheli's own check-ins (founder 2026-10-09): today's unmarked schedule after a chat ends, health questions now and then,
weight about every 20 days, never at night; the same for a person caring for themselves."""

from datetime import timedelta

from app.care import checkins, store
from app.core import clock

FAM = "fam-ck"
MAA = {"id": "maa-ck", "name": "Sarla", "role": "elder"}
SELF = {"id": "self-ck", "name": "Asha", "role": "primary caregiver"}


class Host:
    def __init__(self, items=None):
        self.items = items or []
        self.calls = []

    async def call(self, tool, args, **kw):
        self.calls.append(tool)
        if tool == "get_today_schedule":
            return {"items": self.items}
        return {}


DUE = [{"scheduleId": "s1", "title": "BP tablet", "time": "08:00", "status": "due", "markedBy": None},
       {"scheduleId": "s2", "title": "Shelcal 500mg", "time": "14:00", "status": "due", "markedBy": None},
       {"scheduleId": "s3", "title": "Folvite", "time": "13:00", "status": "completed", "markedBy": "maa-ck"}]


async def _chat(db, who, at):
    await store.add_turn(db, family_id=FAM, thread_id=who, role="user", text="ok", speaker_id=who)
    t = (await store.recent_turns(db, FAM, who))[-1]
    t.at = at
    await db.commit()


async def test_after_a_chat_ends_saheli_asks_about_todays_unmarked_items(db, at):
    now = at("2026-10-09 19:15")
    await store.save_roster(db, FAM, MAA, [MAA])
    await _chat(db, MAA["id"], now - timedelta(minutes=10))
    got = await checkins.plan(db, Host(DUE), FAM, MAA["id"], now=now)
    assert got and got[0] == "doses:s1,s2" and "BP tablet at 08:00" in got[1] and "Shelcal 500mg at 14:00" in got[1] and "Folvite" not in got[1]
    # still talking (2 min ago): wait for the conversation to end
    await _chat(db, MAA["id"], now - timedelta(minutes=2))
    assert await checkins.plan(db, Host(DUE), FAM, MAA["id"], now=now) is None
    # an order is moving for them: not now
    assert await checkins.plan(db, Host(DUE), FAM, MAA["id"], now=now + timedelta(minutes=10), busy=True) is None


async def test_asked_once_a_day_per_set_and_never_at_night(db, at):
    now = at("2026-10-09 15:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    await _chat(db, MAA["id"], now - timedelta(minutes=10))
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": "doses:s1,s2"})
    await db.commit()
    got = await checkins.plan(db, Host(DUE), FAM, MAA["id"], now=now)
    assert got is None or not got[0].startswith("doses")
    night = at("2026-10-09 22:40")
    await _chat(db, MAA["id"], night - timedelta(minutes=10))
    assert await checkins.plan(db, Host(DUE[:1] + [{**DUE[1], "scheduleId": "s9"}]), FAM, MAA["id"], now=night) is None


async def test_health_questions_rotate_and_weight_waits_20_days(db, at):
    now = at("2026-10-09 11:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="condition", key="condition:hypertension", value={"name": "hypertension"},
                         text="Has hypertension (high BP)", source_kind="caregiver_said", confidence=0.9)
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="vital", summary="weight 62 kg", payload={"kind": "weight", "value": "62"})
    await db.commit()
    got = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now)
    assert got and got[0] in ("reading:bp", "feelings", "reports") and got[0] != "weight", "weight logged today: not asked"
    # weight 21 days old → it comes up; one health question a day
    ev = (await store.events(db, FAM, MAA["id"], kinds=["vital"]))[-1]
    ev.at = now - timedelta(days=21)
    for topic in ("reading:bp", "feelings", "reports"):
        await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": topic})
        e = (await store.events(db, FAM, MAA["id"], kinds=["checkin"]))[-1]
        e.at = now - timedelta(days=1)
    await db.commit()
    got = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now)
    assert got and got[0] == "weight" and "62 kg" in got[1]
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": "weight"})
    await db.commit()
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=now + timedelta(hours=1)) is None, "one per window"
    evening = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now + timedelta(hours=6))
    assert evening and evening[0] not in ("weight", "reading:bp", "feelings", "reports"), "the evening window: a personal one"
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": evening[0]})
    await db.commit()
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=now + timedelta(hours=8)) is None, "two a day at most"
    # outside the gentle windows (14:00): no health question
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=at("2026-10-10 14:00")) is None


async def test_first_health_question_is_how_they_feel_not_weight(db, at):
    now = at("2026-10-09 11:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    got = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now)
    assert got and got[0] == "feelings"


async def test_a_reported_problem_is_followed_up_first(db, at):
    """Live 2026-10-09 19:17: Maa said she felt uneasy and dizzy. The next check-in asks about that before anything else."""
    now = at("2026-10-10 11:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": "feelings"})
    (await store.events(db, FAM, MAA["id"], kinds=["checkin"]))[-1].at = now - timedelta(days=1)
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="symptom", summary="चक्कर आ रहे हैं (dizziness)")
    (await store.events(db, FAM, MAA["id"], kinds=["symptom"]))[-1].at = now - timedelta(hours=16)
    await db.commit()
    got = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now)
    assert got and got[0] == "feelings" and "dizziness" in got[1] and "Ask how that is now" in got[1]


async def test_order_updates_do_not_use_up_the_days_check_ins(db, at):
    now = at("2026-10-09 17:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    for i in range(3):
        await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text=f"order update {i}",
                             meta={"proactive": True, "ref": f"task:t{i}"})
    await db.commit()
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=now) is not None
    for i in range(3):
        await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text=f"nudge {i}", meta={"proactive": True})
    await db.commit()
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=now) is None, "three own messages today: enough"


async def test_self_care_person_gets_the_same_check_ins(db, at):
    now = at("2026-10-09 17:00")
    await store.save_roster(db, FAM, SELF, [SELF])  # caring for herself: the subject is herself
    got = await checkins.plan(db, Host([]), FAM, SELF["id"], now=now)
    assert got and f"person {SELF['id']}" in got[1]


async def test_members_with_their_own_care_record_are_checked_in_too(db, at):
    from app.care.models import FamilyRoster
    from sqlalchemy import select

    at("2026-10-09 17:00")
    await store.save_roster(db, FAM, MAA, [MAA, SELF])
    await store.write_fact(db, family_id=FAM, subject_id=SELF["id"], domain="medicine", key="medicine:thyronorm", value={"name": "Thyronorm 50"},
                           text="Takes Thyronorm 50 mcg in the morning", source_kind="caregiver_said")
    await db.commit()
    roster = (await db.execute(select(FamilyRoster).where(FamilyRoster.family_id == FAM))).scalar_one()
    assert await checkins.cared_for(db, roster) == [MAA["id"], SELF["id"]]


# ── founder 2026-10-12: nudges for everyone, personal and from memory, and never silently held back ──

SON = {"id": "son-ck", "name": "Ravi", "role": "primary caregiver"}


async def _said(db, who, text, at):
    await store.add_turn(db, family_id=FAM, thread_id=who, role="user", text=text, speaker_id=who)
    (await store.recent_turns(db, FAM, who))[-1].at = at
    await db.commit()


async def test_a_caregiver_who_talks_to_saheli_gets_personal_nudges_too(db, at):
    from sqlalchemy import select

    from app.care.models import FamilyRoster

    now = at("2026-10-12 11:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    roster = (await db.execute(select(FamilyRoster).where(FamilyRoster.family_id == FAM))).scalar_one()
    assert await checkins.nudge_people(db, roster) == [(MAA["id"], True)], "a caregiver who never wrote gets nothing"
    await _said(db, SON["id"], "Going to the gym at 8 tonight, trying to quit smoking too", now - timedelta(days=1))
    assert await checkins.nudge_people(db, roster) == [(MAA["id"], True), (SON["id"], False)]
    got = await checkins.plan(db, Host([]), FAM, SON["id"], now=now, cared=False, name="Ravi")
    assert got and got[0] == "follow_up", got
    assert "trying to quit smoking" in got[1] and "Pick ONE thing" in got[1] and "person son-ck (Ravi)" in got[1]
    await store.record_event(db, family_id=FAM, subject_id=SON["id"], kind="checkin", summary="x", payload={"topic": "follow_up"})
    await db.commit()
    # a caregiver: one gentle nudge a day, never a health check-in about weight or readings
    assert await checkins.plan(db, Host([]), FAM, SON["id"], now=now + timedelta(hours=6), cared=False) is None
    nxt = await checkins.plan(db, Host([]), FAM, SON["id"], now=now + timedelta(days=1), cared=False)
    assert nxt and nxt[0].startswith("know:"), nxt
    a_doctor = {"id": "doc-ck", "name": "Dr Rao", "role": "family doctor"}
    await store.save_roster(db, FAM, MAA, [MAA, SON, a_doctor])
    await _said(db, a_doctor["id"], "Please share her last BP readings with me", now - timedelta(hours=5))
    roster = (await db.execute(select(FamilyRoster).where(FamilyRoster.family_id == FAM))).scalar_one()
    assert a_doctor["id"] not in [p for p, _ in await checkins.nudge_people(db, roster)], "never a doctor"


async def test_getting_to_know_them_asks_one_missing_thing_at_a_time(db, at):
    now = at("2026-10-12 11:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="naming", key="naming:address", value={"address_as": "Maa"},
                           text="Call her Maa", source_kind="caregiver_said")
    await db.commit()
    asked = {"feelings": now - timedelta(hours=12), "reports": now - timedelta(hours=12)}  # last night: not due again yet
    for topic, at_ in asked.items():
        await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": topic})
        (await store.events(db, FAM, MAA["id"], kinds=["checkin"]))[-1].at = at_
    await db.commit()
    got = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now)
    assert got and got[0] == "know:routine", "naming is known: the next missing thing"
    assert "ONE light, natural question" in got[1] and "remember routine" in got[1] and "Call her Maa" in got[1]
    # asked and not answered: not again for three weeks; the next one comes after two days
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": "know:routine"})
    await db.commit()
    later = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now + timedelta(days=2, hours=6))
    assert later and later[0] == "know:interests", later


async def test_a_check_in_that_did_not_go_out_is_tried_again_and_does_not_use_up_the_day(db, at):
    now = at("2026-10-12 11:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="checkin", summary="x", payload={"topic": "feelings", "sent": False})
    await db.commit()
    got = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now + timedelta(minutes=10))
    assert got and got[0] != "feelings", "the unsent one waits a few hours, the window is still free"
    again = await checkins.plan(db, Host([]), FAM, MAA["id"], now=now + timedelta(hours=6))
    assert again and again[0] == "feelings", again


async def test_an_order_for_one_person_holds_back_only_their_nudges_and_not_for_long(db, at):
    from types import SimpleNamespace

    now = at("2026-10-12 11:00")
    waiting = SimpleNamespace(requested_by=SON["id"], subject_id=SON["id"], status="awaiting_confirm", updated_at=now - timedelta(minutes=5), created_at=now)
    assert checkins.moving_for([waiting], SON["id"], now) and not checkins.moving_for([waiting], MAA["id"], now)
    waiting.updated_at = now - timedelta(hours=2)
    assert not checkins.moving_for([waiting], SON["id"], now), "left waiting for two hours: no longer holds them back"
    running = SimpleNamespace(requested_by=SON["id"], subject_id=MAA["id"], status="running", updated_at=now - timedelta(hours=2), created_at=now)
    assert checkins.moving_for([running], MAA["id"], now) and checkins.moving_for([running], SON["id"], now)


async def test_order_and_ride_updates_do_not_use_up_the_limit_of_conversations_saheli_starts(db, at):
    """Live 2026-10-12: eleven ride and order updates in the night counted as conversations Saheli started, so his
    check-in was refused all day ('their limit is 4')."""
    from app.learn import timing
    from app.learn.models import ReplyLog

    now = at("2026-10-12 11:00")
    for i in range(11):
        db.add(ReplyLog(family_id=FAM, thread_id=SON["id"], at=now - timedelta(hours=8, minutes=i), kind="proactive", situation="order",
                        text=f"update {i}", sent_hour=2))
    await db.commit()
    assert await timing.quiet_reason(db, FAM, SON["id"], "checkin") is None
    for i in range(4):
        db.add(ReplyLog(family_id=FAM, thread_id=SON["id"], at=now - timedelta(hours=1, minutes=i), kind="proactive", situation="checkin",
                        text=f"nudge {i}", sent_hour=10))
    await db.commit()
    assert await timing.quiet_reason(db, FAM, SON["id"], "checkin"), "his own unanswered nudges still count"


def test_an_orders_follow_up_is_about_the_order_not_a_nudge():
    from app.learn import situations

    assert situations.tag(text="", role="system", tools=[], prompt="[Order follow-up] The Swiggy Instamart order for person x") == "order"
    assert situations.tag(text="", role="system", tools=[], prompt="[Order feedback] The order for person x came") == "order"
    assert situations.tag(text="", role="system", tools=[], prompt="[Check-in: how they are] Ask person x") == "checkin"


async def test_an_unanswered_order_follow_up_does_not_hold_back_a_check_in(db, at):
    """Live 2026-10-12: Maa's 10:30 check-in was refused because the 08:30 'has it come?' about an order was unanswered.
    Only Saheli's own unanswered nudge asks for patience."""
    from app.brain import tools

    at("2026-10-12 08:30")
    await store.save_roster(db, FAM, MAA, [MAA])
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text="बिस्कुट मिल गए?",
                         meta={"proactive": True, "delivered": True, "ref": "task:wake:e42ee742:1"})
    await db.commit()
    at("2026-10-12 10:00")
    nudge = tools.TurnCtx(session=db, host=None, family_id=FAM, elder=MAA, speaker={"id": "saheli-scheduler", "role": "system"},
                          members=[MAA], message_ref="checkin:feelings:maa-ck:2026-10-12")
    assert not [p for p in await tools.send_problems(nudge, MAA["id"], "आज तबीयत कैसी है?") if "give them time" in p]
    # her own unanswered nudge still asks for patience
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text="आज तबीयत कैसी है?",
                         meta={"proactive": True, "delivered": True, "ref": "checkin:feelings:maa-ck:2026-10-12"})
    await db.commit()
    assert [p for p in await tools.send_problems(nudge, MAA["id"], "आज खाना खाया?") if "give them time" in p]


async def test_nudges_follow_their_own_day(db, at):
    """His record says he wakes at 13:30–14:00 and sleeps at 03:00: no morning nudge; Maa sleeps at 21:00: none at 20:15."""
    now = at("2026-10-12 11:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="routine", key="routine:wake", value={"times": ["13:30", "14:00"]},
                           text="Wakes up around 13:30 - 14:00", source_kind="caregiver_said")
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="routine", key="routine:sleep", value={"times": ["03:00", "04:00"]},
                           text="Sleeps around 03:00 - 04:00", source_kind="caregiver_said")
    await db.commit()
    assert not await checkins.awake(db, FAM, MAA["id"], now)
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=now) is None, "asleep at 11:00"
    assert await checkins.awake(db, FAM, MAA["id"], at("2026-10-12 17:00"))
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=at("2026-10-12 17:00")) is not None
    early = {"id": "early-ck", "name": "Kamla", "role": "elder"}
    await store.write_fact(db, family_id=FAM, subject_id=early["id"], domain="routine", key="routine:sleep", value={"time": "21:00"},
                           text="Sleeps around 21:00", source_kind="caregiver_said")
    await db.commit()
    assert await checkins.awake(db, FAM, early["id"], at("2026-10-12 19:30")) and not await checkins.awake(db, FAM, early["id"], at("2026-10-12 20:15"))


async def test_a_reply_in_a_chat_they_ended_does_not_hold_back_a_nudge(db, at):
    """Live 2026-10-12: he said "Thanks", Saheli answered "You're welcome!" at 15:34, and every 16:30 nudge was refused as
    'messaged them and they have not answered'. Only her own nudges wait for an answer."""
    from app.brain import tools

    at("2026-10-12 15:33")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text="Thanks", speaker_id=SON["id"])
    at("2026-10-12 15:34")
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="assistant", text="You're welcome!", meta={"model": "gemini"})
    await db.commit()
    at("2026-10-12 16:30")
    nudge = tools.TurnCtx(session=db, host=None, family_id=FAM, elder=MAA, speaker={"id": "saheli-scheduler", "role": "system"},
                          members=[MAA, SON], message_ref="checkin:follow_up:son-ck:2026-10-12")
    assert not [p for p in await tools.send_problems(nudge, SON["id"], "How did the call with Yash go?") if "give them time" in p]
