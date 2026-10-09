"""Saheli's own check-ins (founder 2026-10-09): today's unmarked schedule after a chat ends, health questions now and then,
weight about every 20 days, never at night; the same for a person caring for themselves."""

from datetime import timedelta

from app.care import checkins, store
from app.core import clock

FAM = "fam-ck"
MAA = {"id": "maa-ck", "name": "Vasundara", "role": "elder"}
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
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=now + timedelta(hours=6)) is None, "one health question a day"
    # outside the gentle windows (14:00): no health question
    assert await checkins.plan(db, Host([]), FAM, MAA["id"], now=at("2026-10-10 14:00")) is None


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
