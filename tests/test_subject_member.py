"""A relative who is not a member is never given the speaker's medicines, doses or readings."""

import json

from app.brain import persona, tools
from app.brain.guards import saved_for_problems
from app.care import extract, store
from app.llm import router
from app.llm.router import LLMReply

FAM = "fam-who"
PRIYA = {"id": "me-priya", "name": "Priya", "role": "primary caregiver"}
GOPALAN_CONTACT = "gopalan"
KAMLA = {"id": "elder-who", "name": "कमला देवी", "role": "elder"}
RAVI = {"id": "cg-ravi", "name": "Ravi Sharma", "role": "primary caregiver"}
ANIL_A = {"id": "cg-anil-a", "name": "Anil Sharma", "role": "primary caregiver"}
ANIL_B = {"id": "cg-anil-b", "name": "Anil Gupta", "role": "co-caregiver"}


class Host:
    def __init__(self):
        self.calls = []

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args, kw))
        return {"ok": True}


def ctx(db, elder, speaker, members):
    return tools.TurnCtx(session=db, host=Host(), family_id=FAM, elder=elder, speaker=speaker, members=members)


async def test_a_non_member_medicine_is_not_saved_on_the_self_care_speaker(db, at):
    at("2026-10-07 09:00")
    c = ctx(db, PRIYA, PRIYA, [PRIYA])
    out, err = await tools.run(c, "remember", {
        "domain": "medicine", "name": "Amlong", "about": GOPALAN_CONTACT,
        "details": {"name": "Amlong", "dose": "5 mg", "times": ["08:00"]},
        "sentence": "Amlong 5 mg once daily in the morning",
    })
    assert err and "not a member" in json.loads(out)["refused"]
    assert await store.facts(db, FAM, PRIYA["id"]) == []
    assert not c.host.calls


async def test_a_caregiver_names_the_elder_and_the_elder_names_the_caregiver(db, at):
    at("2026-10-07 09:00")
    members = [KAMLA, RAVI]
    out, err = await tools.run(ctx(db, KAMLA, RAVI, members), "remember", {
        "domain": "medicine", "name": "Ecosprin", "about": "कमला",
        "details": {"dose": "75 mg", "times": ["14:00"]}, "sentence": "Ecosprin 75 mg after lunch",
    })
    got = json.loads(out)
    assert not err and got["subject_id"] == KAMLA["id"] and got["for"] == "कमला देवी" and got["for_other"] is True
    assert (await store.facts(db, FAM, KAMLA["id"]))[0].key == "medicine:ecosprin"
    assert await store.facts(db, FAM, RAVI["id"]) == []

    out, err = await tools.run(ctx(db, KAMLA, KAMLA, members), "log_vital",
                               {"kind": "bp", "value": "130/80", "about": "Ravi"})
    got = json.loads(out)
    assert not err and got["subject_id"] == RAVI["id"] and got["for"] == "Ravi Sharma" and got["for_other"] is True


async def test_a_reading_and_a_dose_for_someone_outside_the_family_are_refused(db, at):
    at("2026-10-07 09:00")
    c = ctx(db, KAMLA, RAVI, [KAMLA, RAVI])
    for name, args in (
        ("log_vital", {"kind": "sugar", "value": "140", "about": "Gopalan"}),
        ("log_dose", {"medicine": "Amlong", "outcome": "taken", "about": "Gopalan"}),
        ("set_reminder", {"text": "Amlong", "times": ["08:00"], "repeat": "daily", "about": "Gopalan"}),
        ("log_event", {"kind": "symptom", "summary": "headache", "about": "Gopalan"}),
        ("stop", {"domain": "medicine", "name": "Amlong", "reason": "not his", "about": "Gopalan"}),
        ("log_food", {"items": [{"name": "rice"}], "for_whom": "Gopalan"}),
        ("log_activity", {"key": "walk", "done": True, "summary": "a short walk", "about": "Gopalan"}),
        ("change_reminder", {"id": "medicine:amlong", "action": "change", "times": ["09:00"], "about": "Gopalan"}),
    ):
        out, err = await tools.run(c, name, args)
        assert err and "not a member" in json.loads(out)["refused"], name
    assert await store.facts(db, FAM, KAMLA["id"]) == []
    assert await store.events(db, FAM, KAMLA["id"]) == []


async def test_a_shared_first_name_is_not_guessed(db, at):
    at("2026-10-07 09:00")
    out, err = await tools.run(ctx(db, KAMLA, ANIL_A, [KAMLA, ANIL_A, ANIL_B]), "remember", {
        "domain": "preference", "name": "tea", "about": "Anil", "details": {}, "sentence": "Takes tea without sugar",
    })
    assert err and "More than one member" in json.loads(out)["refused"]


def test_the_reply_has_to_name_who_it_was_saved_for():
    assert "for someone other" in persona.PERSONA
    saved = {"for": "Priya", "for_other": True, "result": "created"}
    missing = saved_for_problems("Saved.", [{"tool": "remember", "ok": True, "result": json.dumps(saved)}])
    assert missing and "Priya" in missing[0]
    assert saved_for_problems("Saved for priya.", [{"tool": "remember", "ok": True, "result": json.dumps(saved)}]) == []
    assert saved_for_problems("Saved your lunch.", [{"tool": "log_event", "ok": True, "result": json.dumps({"for": "Priya"})}]) == []
    assert saved_for_problems("Saved.", [{"tool": "remember", "ok": False, "result": json.dumps({"refused": "no"})}]) == []


async def test_the_extractor_does_not_file_a_non_member_on_the_care_recipient(db, at, monkeypatch):
    body = {"facts": [
        {"domain": "medicine", "name": "Amlong", "about": "Gopalan", "details": {"dose": "5 mg", "times": ["08:00"]},
         "sentence": "Gopalan takes Amlong 5 mg in the morning", "evidence": "achan's prescription"},
        {"domain": "medicine", "name": "Ecosprin", "about": "Ravi", "details": {"dose": "75 mg"},
         "sentence": "Ravi takes Ecosprin 75 mg", "evidence": "I take ecosprin"},
    ], "stops": [], "notes": [{"about": "Gopalan", "topic": "Father", "text": "Gopalan takes Amlong"}],
        "readings": [{"about": "Gopalan", "kind": "sugar", "value": "140", "unit": "mg/dL"}], "plans": [], "moods": []}

    class Reader:
        async def complete(self, route, **kw):
            return LLMReply(text=json.dumps(body), tool_calls=[], model="fake")

    router.register_provider("fake", Reader())
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        at("2026-10-07 21:00")
        await store.save_roster(db, FAM, PRIYA, [PRIYA, RAVI])
        await store.add_turn(db, family_id=FAM, thread_id=PRIYA["id"], role="user", text="found achan's prescription")
        counts = await extract.extract_family(db, FAM)
        assert counts["not_member"] == 2 and counts["pending"] == 1
        assert await store.facts(db, FAM, PRIYA["id"]) == []
        assert [e.kind for e in await store.events(db, FAM, PRIYA["id"])] == ["memory_extract"]
        ravi = await store.facts(db, FAM, RAVI["id"])
        assert len(ravi) == 1 and ravi[0].status == "pending" and ravi[0].key == "medicine:ecosprin"
        notes = await store.notes(db, FAM, ["family"])
        assert notes and "Gopalan" in notes[0].body_md
    finally:
        router._providers.pop("fake", None)
