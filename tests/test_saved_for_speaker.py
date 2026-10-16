"""Whatever is saved belongs to the person writing, unless the message is about someone else.

Live 16 Oct 04:04: a caregiver wrote "Good night"; log_activity was called with no about, and a blank about meant the
care recipient, so his sleep was saved as his mother's and Saheli said "Logged sleep for <mother>". A "forget" of
"Sleep at 04:04 AM" then hid every memory about her sleep (her routine, an older log, her diary and setup notes),
because the matching dropped the numbers. Founder: log for another person only when asked, else for the sender, for
everything.
"""

import json

import pytest

from app.brain import tools
from sqlalchemy import select

from app.care import extract, memory_upkeep, store
from app.care.models import CareEvent
from app.llm import router
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM, ELDER, CG, OUTSIDER = "fam-own", "e-own", "c-own", "x-own"
E = {"id": ELDER, "name": "Babita", "role": "elder"}
C = {"id": CG, "name": "Kittu", "role": "primary caregiver"}

SAVES = sorted(tools.SAVES_FOR_SPEAKER)


async def all_events(db, subject):
    """Every event, the hidden ones too (store.events leaves forgotten ones out)."""
    return list((await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.subject_id == subject)
                                  .order_by(CareEvent.id))).scalars())


def ctx(db, speaker=C, **kw):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=speaker, members=[E, C], **kw)


def test_every_saving_tool_says_blank_is_the_speakers_own():
    specs = {s.name: s for s in tools.specs()}
    for name in SAVES:
        about = specs[name].schema["properties"]["about"]["description"]
        assert "speaker's own" in about, name
        assert "Leave out for the care recipient" not in about, name


async def test_a_caregivers_good_night_is_his_own_sleep(db, at):
    at("2026-10-16 04:04")
    out, err = await tools.run(ctx(db), "log_activity", {"key": "sleep", "done": True, "summary": "Going to sleep after a late night"})
    assert not err
    data = json.loads(out)
    assert data["subject_id"] == CG and data["for"] == "Kittu" and not data.get("for_other")
    assert [e.kind for e in await store.events(db, FAM, CG)] == ["activity_done"]
    assert not await store.events(db, FAM, ELDER)


@pytest.mark.parametrize("tool,args", [
    ("log_vital", {"kind": "bp", "value": "130/85"}),
    ("log_event", {"kind": "symptom", "summary": "headache since the meeting", "severity": "info"}),
    ("log_event", {"kind": "sleep", "summary": "slept at 4 after a late night"}),
    ("note", {"topic": "work", "text": "Meeting with the tourism department went well"}),
    ("remember", {"domain": "preference", "name": "short replies", "details": {}, "sentence": "Likes short replies"}),
])
async def test_a_caregivers_own_save_never_lands_on_the_elder(db, at, tool, args):
    at("2026-10-16 09:00")
    out, err = await tools.run(ctx(db), tool, args)
    assert not err, out
    assert json.loads(out)["subject_id"] == CG
    assert not await store.events(db, FAM, ELDER)
    assert not await store.facts(db, FAM, ELDER)
    assert not await store.notes(db, FAM, [ELDER])


async def test_a_save_about_the_elder_names_her(db, at):
    at("2026-10-16 22:00")
    out, err = await tools.run(ctx(db), "log_activity", {"key": "sleep", "done": True, "summary": "Maa went to sleep", "about": ELDER})
    assert not err
    data = json.loads(out)
    assert data["subject_id"] == ELDER and data["for_other"] is True
    assert [e.kind for e in await store.events(db, FAM, ELDER)] == ["activity_done"] and not await store.events(db, FAM, CG)


async def test_the_elders_own_words_and_a_scheduled_turn_and_the_dashboard_stay_on_her(db, at):
    at("2026-10-16 09:00")
    her, err = await tools.run(ctx(db, E), "log_vital", {"kind": "sugar", "value": "140"})
    system, err2 = await tools.run(ctx(db, {"id": "saheli-scheduler", "role": "system"}), "log_event", {"kind": "other", "summary": "check-in sent"})
    page, err3 = await tools.run(ctx(db, channel="dashboard"), "log_vital", {"kind": "weight", "value": "61", "unit": "kg"})
    assert not (err or err2 or err3)
    assert {json.loads(x)["subject_id"] for x in (her, system, page)} == {ELDER}


async def test_a_speaker_outside_the_household_saves_nothing_as_theirs(db, at):
    at("2026-10-16 09:00")
    out, err = await tools.run(ctx(db, {"id": OUTSIDER, "name": "Visitor", "role": "caregiver"}), "log_vital", {"kind": "bp", "value": "120/80"})
    assert err and "not in HOUSEHOLD" in out
    assert not await store.events(db, FAM, ELDER) and not await store.events(db, FAM, OUTSIDER)


async def test_a_dose_with_no_about_for_a_medicine_only_the_elder_takes_asks_whose(db, at):
    at("2026-10-16 09:00")
    _, err = await tools.run(ctx(db), "remember", {"about": ELDER, "domain": "medicine", "name": "Metformin",
                                                   "details": {"dose": "500 mg", "times": ["09:00"]}, "sentence": "Metformin 500 mg at 9"})
    assert not err
    out, err = await tools.run(ctx(db), "log_dose", {"medicine": "Metformin", "outcome": "taken"})
    assert err and "Babita" in out and ELDER in out and "about" in out
    assert not [e for e in await store.events(db, FAM, ELDER) if e.kind.startswith("dose_")]
    out, err = await tools.run(ctx(db), "log_dose", {"medicine": "Metformin", "outcome": "taken", "about": ELDER})
    assert not err and json.loads(out)["subject_id"] == ELDER
    stock, err = await tools.run(ctx(db), "set_stock", {"medicine": "Metformin", "count": 10})
    assert err and "Babita" in stock


async def test_forget_keeps_the_numbers_and_never_hides_other_memories(db, at):
    at("2026-10-13 10:00")
    _, err = await tools.run(ctx(db), "remember", {"about": ELDER, "domain": "routine", "name": "sleep", "details": {"time": "22:00"},
                                                   "sentence": "Sleeps around 22:00"})
    assert not err
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="sleep", summary="Going to sleep very late at 02:38 AM")
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="diary", title="Diary",
                            body_md="- 2026-10-13: Babita went to sleep very late last night but felt fine")
    at("2026-10-16 04:04")
    wrong = await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="activity_done",
                                     summary="Sleep: Going to sleep at 04:04 AM after a late night.")
    out, err = await tools.run(ctx(db), "forget", {"what": "Sleep at 04:04 AM", "about": ELDER})
    assert not err and json.loads(out)["forgotten"] == 1
    db.expire_all()
    kept = {e.id: bool((e.payload or {}).get("forgotten")) for e in await all_events(db, ELDER) if e.kind != "memory_forgotten"}
    wrong_id = wrong if isinstance(wrong, int) else wrong.id
    assert kept[wrong_id] is True and sum(kept.values()) == 1
    assert (await store.active_fact(db, FAM, ELDER, "routine:sleep")) is not None
    assert "went to sleep very late" in (await store.notes(db, FAM, [ELDER]))[0].body_md


async def test_forget_asks_which_when_the_words_match_many_memories(db, at):
    at("2026-10-12 21:00")
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="sleep", summary="Slept late after the wedding")
    at("2026-10-13 21:00")
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="sleep", summary="Slept late, knee pain")
    at("2026-10-16 09:00")
    out, err = await tools.run(ctx(db), "forget", {"what": "slept late", "about": ELDER})
    data = json.loads(out)
    assert not err and data["forgotten"] == 0 and len(data["matches"]) == 2 and "refs" in data["note"]
    db.expire_all()
    assert not any((e.payload or {}).get("forgotten") for e in await all_events(db, ELDER))
    ref = next(m["ref"] for m in data["matches"] if "knee" in m["what"])
    out, err = await tools.run(ctx(db), "forget", {"what": "slept late", "refs": [ref], "about": ELDER})
    assert not err and json.loads(out)["forgotten"] == 1
    db.expire_all()
    hidden = [e.summary for e in await all_events(db, ELDER) if (e.payload or {}).get("forgotten") and e.kind == "sleep"]
    assert hidden == ["Slept late, knee pain"]


async def test_the_dashboard_forget_box_still_removes_every_match(db, at):
    at("2026-10-12 21:00")
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="sleep", summary="Slept late after the wedding")
    at("2026-10-13 21:00")
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="sleep", summary="Slept late, knee pain")
    out = await memory_upkeep.forget(db, FAM, [ELDER], "slept late", by=CG)
    assert out["forgotten"] == 2


class Extractor:
    out: dict = {}

    async def complete(self, route, **kw):
        return LLMReply(text=json.dumps(self.out), tool_calls=[], model="fake")


@pytest.fixture
def fake_extract(monkeypatch):
    fake = Extractor()
    router.register_provider("fake", fake)
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()
    yield fake
    router._providers.pop("fake", None)


async def test_the_re_read_files_a_caregivers_own_words_on_him(db, at, fake_extract):
    at("2026-10-16 09:00")
    await store.save_roster(db, FAM, E, [E, C])
    await store.add_turn(db, family_id=FAM, thread_id=CG, role="user", text="Started going to the gym at 7 every morning")
    fake_extract.out = {"facts": [{"domain": "routine", "name": "gym", "details": {"time": "07:00"}, "sentence": "Gym at 7 every morning",
                                   "evidence": "gym at 7"}], "stops": [], "notes": []}
    await extract.extract_family(db, FAM)
    assert await store.active_fact(db, FAM, CG, "routine:gym") is not None
    assert await store.active_fact(db, FAM, ELDER, "routine:gym") is None


async def test_the_re_read_never_files_an_unnamed_item_on_the_elder_when_two_people_spoke(db, at, fake_extract):
    at("2026-10-16 09:00")
    await store.save_roster(db, FAM, E, [E, C])
    await store.add_turn(db, family_id=FAM, thread_id=CG, role="user", text="I walk at 6")
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="Aaj mandir gayi thi")
    fake_extract.out = {"facts": [{"domain": "routine", "name": "walk", "details": {"time": "06:00"}, "sentence": "Walks at 6",
                                   "evidence": "I walk at 6"}], "stops": [], "notes": []}
    counts = await extract.extract_family(db, FAM)
    assert counts.get("not_member") == 1
    assert await store.active_fact(db, FAM, ELDER, "routine:walk") is None


async def test_a_scheduled_turn_about_one_person_saves_for_that_person(db, at):
    """Live 16 Oct 21:15: the scheduled gym check-in for a caregiver opened its follow-up loop on his mother."""
    at("2026-10-16 21:15")
    system = {"id": "saheli-scheduler", "role": "system"}
    ctx_for_him = ctx(db, system)
    ctx_for_him.system_about = CG
    out, err = await tools.run(ctx_for_him, "open_loop", {"kind": "watch", "title": "Gym workout follow-up"})
    assert not err and json.loads(out)["subject_id"] == CG
    plain = ctx(db, system)  # a scheduled turn with no person stays on the care recipient
    out, err = await tools.run(plain, "log_event", {"kind": "other", "summary": "check-in sent"})
    assert not err and json.loads(out)["subject_id"] == ELDER
