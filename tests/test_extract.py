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
