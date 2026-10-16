"""Crisis talk reaches the Kavach care team, and an alert never goes to the person it is about (founder 2026-10-13).

Live 2026-10-13, 01:28-01:34 IST: a person caring for himself wrote of jumping from a roof, then of a gun. Saheli sent
five "urgent care alert" WhatsApps, each under a new issue name, to his family's caregivers: only himself. She told him she
was informing his family and emergency services (there was no one), four follow-ups were set to ask him about himself,
and no one at Kavach knew. Now: reason crisis emails the care team at once (later lines batched), the family other than
the person is told once per half hour, no follow-up is ever owned by the person, and the reply says who was told."""


import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import careteam, store
from app.care.models import CareEvent, OpenLoop
from app.sim.world import SimHost

FAM = "fam-crisis"
SELF = {"id": "self-cr", "name": "Rahul", "role": "primary caregiver"}
ELDER = {"id": "elder-cr", "name": "Kamla", "role": "elder"}
SON = {"id": "son-cr", "name": "Ravi", "role": "primary caregiver"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


def ctx_for(db, host, *, person, members, said, family=FAM):
    return tools.TurnCtx(session=db, host=host, family_id=family, elder=person, speaker=person, members=members, user_text=said)


def crisis(issue: str, message: str) -> dict:
    return {"reason": "crisis", "issue": issue, "message": message, "confidence": 0.95}


async def _loops(db) -> list[OpenLoop]:
    return list((await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM))).scalars())


async def test_his_case_the_team_is_emailed_at_once_and_nothing_goes_to_him_or_asks_him_about_himself(db, at):
    at("2026-10-13 01:28")
    await store.save_roster(db, FAM, SELF, [SELF])
    host = SimHost()
    host.world.other_caregivers = 0  # caring for himself: no one else in the family
    said = "मैं छत में जा रहा हूं और मैं छत से कूद रहा हूं"
    out = await tools.alert_caregiver(ctx_for(db, host, person=SELF, members=[SELF], said=said),
                                      crisis("suicide-threat", "Rahul says he is going to jump from the roof."))
    assert out["sent"] and "no one else in the family to tell" in out["told"] and "Kavach care team has been emailed" in out["told"]
    [mail] = host.world.team_alerts
    assert mail["kind"] == "crisis" and not mail["update"] and mail["lines"][0]["said"] == said
    assert mail["family_told"] == 0
    assert not await _loops(db), "no follow-up asks him about himself"


async def test_a_crisis_over_six_minutes_is_one_email_then_one_update_and_one_whatsapp_to_the_family(db, at, sessions):
    at("2026-10-13 01:28")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    host = SimHost()
    for minute, issue, msg in [(28, "suicide-threat", "Kamla ji says she will jump from the roof."),
                               (29, "no-reason-to-live", "She says she has no reason to live."),
                               (31, "gun", "She says she has a gun and will shoot someone and herself."),
                               (33, "shot", "She says she was shot in the chest.")]:
        at(f"2026-10-13 01:{minute}")
        await tools.alert_caregiver(ctx_for(db, host, person=ELDER, members=[ELDER, SON], said=msg), crisis(issue, msg))
    await db.commit()
    # the family (her son) hears once in the half hour; the team gets the first line at once
    assert len(host.world.alerts) == 1 and host.world.alerts[0]["kind"] == "crisis" and host.world.alerts[0]["urgency"] == "high"
    assert [len(m["lines"]) for m in host.world.team_alerts] == [1, 2], "01:28 at once; 01:31 with the 01:29 line"
    # 01:33 waited (two minutes after the last email): the wake job sends it as one update once the gap has passed
    at("2026-10-13 01:37")
    assert await careteam.flush(sessions, lambda fid: host) == 1
    last = host.world.team_alerts[-1]
    assert last["update"] and [l["summary"] for l in last["lines"]] == ["She says she was shot in the chest."]
    assert await careteam.flush(sessions, lambda fid: host) == 0, "nothing waits any more"
    # the follow-up next morning goes to her son, never to her
    loops = await _loops(db)
    assert loops and all(l.owner_id == SON["id"] for l in loops)


async def test_the_same_crisis_issue_again_still_reaches_the_team(db, at):
    at("2026-10-13 01:28")
    await store.save_roster(db, FAM, SELF, [SELF])
    host = SimHost()
    host.world.other_caregivers = 0
    c = ctx_for(db, host, person=SELF, members=[SELF], said="I want to die")
    await tools.alert_caregiver(c, crisis("suicide", "He says he wants to die."))
    at("2026-10-13 01:35")
    c2 = ctx_for(db, host, person=SELF, members=[SELF], said="I have the pills in my hand")
    out = await tools.alert_caregiver(c2, crisis("suicide", "He says he has pills in his hand."))
    assert not out["sent"] and "Kavach care team" in out["told"]
    assert [m["lines"][-1]["said"] for m in host.world.team_alerts] == ["I want to die", "I have the pills in my hand"]


async def test_a_red_flag_with_no_one_else_to_tell_goes_to_the_team_and_with_family_it_does_not(db, at):
    at("2026-10-13 10:00")
    red = {"reason": "red_flag", "issue": "chest-pain", "message": "Severe chest pain and breathlessness.", "confidence": 0.9}
    await store.save_roster(db, FAM, SELF, [SELF])
    alone = SimHost()
    alone.world.other_caregivers = 0
    out = await tools.alert_caregiver(ctx_for(db, alone, person=SELF, members=[SELF], said="I have severe chest pain and can't breathe"), red)
    assert alone.world.team_alerts and alone.world.team_alerts[0]["kind"] == "emergency_alone"
    assert "no one else in the family to tell" in out["told"]
    await store.save_roster(db, "fam-cr-2", ELDER, [ELDER, SON])
    family = SimHost()
    out = await tools.alert_caregiver(ctx_for(db, family, person=ELDER, members=[ELDER, SON], said="seene mein bahut dard hai, saans nahi aa rahi",
                                              family="fam-cr-2"), red)
    assert not family.world.team_alerts and "1 family member(s) told" in out["told"]


def test_a_saved_name_is_said_in_their_language():
    from app.brain import guards

    hindi = {"saved": {"language": "hi", "script": "roman"}}
    assert guards.call_you(hindi, "Simran") == "Main aapko Simran kahungi."
    assert "35 words" not in guards.call_you(hindi, "Simran")


def test_who_was_told_is_already_in_the_persons_language():
    from app.brain import guards

    hindi = {"saved": {"language": "hi"}, "now": "devanagari"}
    line = guards.who_was_told(hindi, n=1)
    assert line == "वॉट्सऐप पर परिवार के 1 सदस्य को बता दिया"
    assert "family member" not in line
    assert guards.who_was_told(None, n=1) == "1 family member(s) told on WhatsApp"
    assert guards.who_was_told(None, team="sent", already_today=True) == (
        "the Kavach care team has been emailed and will check on them")
    pa = guards.who_was_told({"saved": {"language": "pa"}, "now": "gurmukhi"}, n=1, team="sent")
    assert "family member" not in pa and "1" in pa and "Kavach" in pa


async def test_a_shadow_family_never_emails_the_team(db, at):
    at("2026-10-13 01:28")
    host = SimHost()
    host.world.other_caregivers = 0
    await store.save_roster(db, "shadow:" + FAM, SELF, [SELF])
    await tools.alert_caregiver(ctx_for(db, host, person=SELF, members=[SELF], said="I will jump", family="shadow:" + FAM),
                                crisis("suicide", "He says he will jump."))
    assert not host.world.team_alerts
    rows = (await db.execute(select(CareEvent).where(CareEvent.kind == careteam.KIND))).scalars().all()
    assert not rows


def test_crisis_is_a_whatsapp_alert_reason():
    from app.brain import policy

    assert policy.alert_decision("crisis", 0.5).whatsapp, "a crisis never waits on confidence"
