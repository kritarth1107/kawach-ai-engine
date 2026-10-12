import json

import pytest

from app.care import extract, store
from app.care.redact import scrub_secrets
from app.llm import router
from app.llm.router import LLMReply

FAM = "fam-ext"
ELDER = {"id": "elder-x", "name": "Kamla", "role": "elder"}
SON = {"id": "cg-x", "name": "Ravi", "role": "caregiver"}

OUT = {
    "facts": [
        {"domain": "medicine", "name": "Ecosprin", "details": {"name": "Ecosprin", "dose": "75 mg", "times": ["14:00"]},
         "sentence": "Ecosprin 75 mg after lunch", "evidence": "lunch ke baad ecosprin"},
        {"domain": "home", "name": "cook", "details": {"name": "Radha", "time": "11:00"}, "sentence": "Cook Radha comes at 11",
         "evidence": "Radha 11 baje aati hai"},
        {"domain": "nonsense", "name": "x", "details": {}, "sentence": "ignored"},
    ],
    "stops": [],
    "notes": [{"about": "family", "topic": "Grandchildren", "text": "Granddaughter Pihu turns 7 on 12 Nov"}],
}


class Extractor:
    async def complete(self, route, **kw):
        return LLMReply(text="Here you go:\n" + json.dumps(OUT), tool_calls=[], model="fake")


@pytest.fixture
def fake(monkeypatch):
    router.register_provider("fake", Extractor())
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()
    yield
    router._providers.pop("fake", None)


def test_scrub_secrets():
    assert "482913" not in scrub_secrets("Your OTP is 482913, do not share")
    assert "4829" not in scrub_secrets("4829 is your OTP for Swiggy")
    assert scrub_secrets("BP 140/90 aur sugar 180") == "BP 140/90 aur sugar 180"


async def test_extract_puts_health_facts_on_hold(db, at, fake):
    at("2026-10-02 15:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="Lunch ke baad ecosprin leti hoon. OTP 556677")
    counts = await extract.extract_family(db, FAM)
    assert counts == {"facts": 1, "pending": 1, "stops": 0, "notes": 1}
    facts = {f.key: f for f in await store.facts(db, FAM, ELDER["id"])}
    assert facts["medicine:ecosprin"].status == "pending"
    from app.care import versions

    hist = await versions.changes(db, FAM, [ELDER["id"]], kinds=("fact",), target="medicine:ecosprin")
    assert [v.op for v in hist] == ["pending"]  # a proposal heard in a re-read shows in memory history too
    assert facts["home:cook"].status == "active" and facts["home:cook"].source_kind == "inferred"
    loops = await store.live_loops(db, FAM, [ELDER["id"]])
    assert loops[0].kind == "confirm_fact"
    notes = await store.notes(db, FAM, ["family"])
    assert "Pihu" in notes[0].body_md
    turns = await store.recent_turns(db, FAM, ELDER["id"])
    assert "556677" not in turns[0].text and turns[0].extracted
    assert await extract.extract_family(db, FAM) == {"skipped": "nothing new"}


PLANS = {
    "facts": [{"about": "cg-x", "domain": "routine", "name": "gym", "details": {"time": "20:00"}, "sentence": "Goes to the gym at 8 pm",
               "evidence": "gym 8 baje jaata hoon"}],
    "stops": [], "notes": [],
    "plans": [
        {"about": "cg-x", "what": "Trainer will send a diet plan tonight", "when": "2026-10-13T21:00", "ask_after": "did the diet plan come?"},
        {"about": "elder-x", "what": "Eye check-up at the hospital", "when": None, "ask_after": "what did the doctor say?"},
        {"about": "cg-x", "what": "Old exam", "when": "2026-10-01T10:00"},
        {"about": "cg-x", "what": ""},
    ],
}


async def test_a_stated_plan_becomes_a_follow_up_and_a_timed_routine_becomes_an_activity(db, at, monkeypatch):
    """Founder 2026-10-13: "never forget anything which was said" — what they said will happen is asked about afterwards,
    and a routine with a time (gym at 8) gets check-ins without anyone asking for a reminder."""
    from app.care import activities
    from app.core import clock

    class Reader:
        async def complete(self, route, **kw):
            if "activities" in (kw.get("system_stable") or ""):
                return LLMReply(text=json.dumps({"activities": [{"type": "gym", "label": "Gym", "kind": "habit", "days": [], "start": "20:00"}]}),
                                tool_calls=[], model="fake")
            return LLMReply(text=json.dumps(PLANS), tool_calls=[], model="fake")

    router.register_provider("fake", Reader())
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()
    adopted = []
    real_adopt = activities.adopt

    async def adopt(session, host, fid, sid):
        made = await real_adopt(session, host, fid, sid)
        adopted.append((sid, made))
        return made

    monkeypatch.setattr(activities, "adopt", adopt)
    try:
        at("2026-10-13 15:00")
        await store.save_roster(db, FAM, ELDER, [ELDER, SON])
        await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text="gym 8 baje jaata hoon, trainer aaj raat diet plan bhejega")
        counts = await extract.extract_family(db, FAM)
        assert counts["plans"] == 2, "the past one and the empty one are dropped"
        loops = {l.title.split(" (ask")[0]: l for l in await store.live_loops(db, FAM, [ELDER["id"], SON["id"]]) if l.kind == "plan"}
        diet = loops["Trainer will send a diet plan tonight"]
        assert diet.owner_id == SON["id"] and clock.ist(diet.wake_at).strftime("%d %H:%M") == "13 23:00"
        assert "did the diet plan come" in diet.title
        eye = loops["Eye check-up at the hospital"]
        assert eye.owner_id == ELDER["id"] and clock.ist(eye.wake_at).strftime("%d %H:%M") == "14 11:00", "no time: asked next late morning"
        from app.brain import wake

        prompt = await wake.wake_prompt(db, diet, ELDER["id"])
        assert "[Follow-up on something they said]" in prompt and str(diet.id) in prompt
        assert any(sid == SON["id"] for sid, _ in adopted), "the re-read turns routines into activities"
    finally:
        router._providers.pop("fake", None)
