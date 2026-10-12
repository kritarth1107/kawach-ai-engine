"""The person-centred pages' one read (founder-approved design 2026-10-13): who they are, mood over two weeks, what was
learned with the words it came from, their people and day, and a week of doses. Only what the record holds."""

from app.brain import tools
from app.care import mood, portrait, store
from app.sim.world import SimHost

FAM = "fam-portrait"
ELDER = {"id": "elder-p", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "son-p", "name": "Rohit", "role": "primary caregiver"}


async def test_portrait_holds_the_person_mood_learned_words_people_day_and_doses(db, at):
    at("2026-10-12 09:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    host = SimHost()
    son = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker=SON, members=[ELDER, SON])
    for args in (
        {"domain": "profile", "name": "age", "details": {"age": 72}, "sentence": "72 years old"},
        {"domain": "home", "name": "address", "details": {"address": "14 Shastri Nagar, Jodhpur", "city": "Jodhpur"}, "sentence": "Lives at 14 Shastri Nagar"},
        {"domain": "naming", "name": "address_as", "details": {"name": "Maa ji", "avoid": ["Kamla"]}, "sentence": "Call her Maa ji"},
        {"domain": "medicine", "name": "Metformin", "details": {"dose": "500 mg", "times": ["08:30", "20:30"]}, "sentence": "Metformin 500 mg twice"},
        {"domain": "contact", "name": "Ramesh", "details": {"name": "Ramesh", "relation": "neighbour", "phone": "+919000000001"}, "sentence": "Ramesh ji, neighbour, has a key"},
        {"domain": "routine", "name": "temple", "details": {"time": "07:00"}, "sentence": "Walks to the temple at 7"},
        {"domain": "routine", "name": "walks", "details": {"note": "most days"}, "sentence": "Walks most days"},
        {"domain": "preference", "name": "likes", "details": {"topics": ["Bhajans"]}, "sentence": "Enjoys bhajans"},
    ):
        await tools.run(son, "remember", args)
    # something she said herself, saved from her message
    await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="Pihu ki birthday 12 November ko hai", speaker_id=ELDER["id"],
                         message_ref="wamid.P1")
    her = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER, SON], message_ref="wamid.P1")
    await tools.run(her, "remember", {"domain": "occasion", "name": "pihu_birthday", "details": {"date": "11-12"}, "sentence": "Pihu's birthday is on 12 Nov"})
    await mood.record(db, family_id=FAM, subject_id=ELDER["id"], score=2, word="worried", summary="Worried about her knee")
    at("2026-10-13 11:40")
    await mood.record(db, family_id=FAM, subject_id=ELDER["id"], score=4, word="calm", summary="Happy after a video call")
    await store.record_event(db, family_id=FAM, subject_id=ELDER["id"], kind="dose_taken", summary="Metformin taken", payload={"medicine": "Metformin"})

    p = await portrait.build(db, FAM, ELDER["id"])
    assert p["person"]["age"] == "72" and p["person"]["callAs"] == "Maa ji" and p["person"]["address"] == "14 Shastri Nagar, Jodhpur"
    assert p["person"]["city"] == "Jodhpur" and p["person"]["name"] == "Kamla Devi"
    assert [d["word"] for d in p["mood"]["days"]] == ["worried", "calm"] and p["mood"]["now"]["word"] == "calm"
    bday = next(x for x in p["learned"] if x["key"] == "occasion:pihu_birthday")
    assert bday["said"]["text"] == "Pihu ki birthday 12 November ko hai" and bday["statedBy"] == "Kamla Devi" and bday["said"]["heard"] is False
    assert any(x["name"] == "Rohit" and x["kind"] == "member" for x in p["people"])
    assert any(x["name"] == "Ramesh" and x["phone"] == "+919000000001" for x in p["people"])
    assert [d["time"] for d in p["day"]] == ["07:00"], "only routine with a time is on the day line"
    assert p["avoid"] == ['Never call them "Kamla"'] and any("bhajans" in e["text"] for e in p["enjoys"])
    assert len(p["week"]) == 7 and p["week"][-1]["day"] == "2026-10-13"
    today = p["week"][-1]["doses"]
    assert [d["status"] for d in today if d["time"] == "08:30"] == ["taken"]
    assert p["stats"]["known"] >= 8 and p["stats"]["pending"] == 0


async def test_portrait_of_an_empty_record_is_empty_not_invented(db, at):
    at("2026-10-13 09:00")
    await store.save_roster(db, FAM, ELDER, [ELDER])
    p = await portrait.build(db, FAM, ELDER["id"])
    assert p["person"]["age"] is None and p["person"]["address"] is None
    assert p["mood"] == {"now": None, "days": []} and p["learned"] == [] and p["day"] == [] and p["people"] == []
    assert all(not d["doses"] for d in p["week"])


def test_routine_time_reads_only_structured_clock_times():
    assert portrait.routine_time({"time": "07:00"}) == "07:00"
    assert portrait.routine_time({"start_time": "20:00", "end_time": "21:00"}) == "20:00"
    assert portrait.routine_time({"times": ["13:30", "14:00"]}) == "13:30"
    assert portrait.routine_time({"time": "morning"}) is None and portrait.routine_time({}) is None


async def test_the_emergency_card_never_shows_has_none_as_an_item_and_carries_the_home_address(db, at):
    """A self-care user answered "no allergies, no regular medicines": the card listed an allergen "none"."""
    from app.care import features

    at("2026-10-13 09:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    son = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ELDER, speaker=SON, members=[ELDER, SON])
    await tools.run(son, "remember", {"domain": "allergy", "name": "none", "none": True, "details": {}, "sentence": "No known allergies."})
    await tools.run(son, "remember", {"domain": "medicine", "name": "none", "none": True, "details": {}, "sentence": "Takes no regular medicines."})
    await tools.run(son, "remember", {"domain": "home", "name": "address", "details": {"address": "14 Shastri Nagar, Jodhpur"}, "sentence": "Home address"})
    card = await features.emergency(db, FAM, ELDER["id"])
    assert card["allergies"] == [] and card["medicines"] == []
    assert card["none"] == {"allergy": True, "condition": False, "medicine": True}
    assert card["home"] == "14 Shastri Nagar, Jodhpur"


async def test_words_the_re_read_heard_are_marked_heard_and_a_link_to_a_person_reads_their_name(db, at, api_names=None):
    """Live 2026-10-13: a re-read fact showed its evidence as if the person had said it, and a family link showed an id."""
    from app.api.dash import links_json
    from app.care import links
    from app.care.models import CareFact

    at("2026-10-13 09:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    w = await store.write_fact(db, family_id=FAM, subject_id=ELDER["id"], domain="family", key="family:son_rohit", value={"relation": "son"},
                               text="Rohit is her son.", source_kind="inferred", note='heard: "mera beta Rohit"')
    p = await portrait.build(db, FAM, ELDER["id"])
    son = next(x for x in p["learned"] if x["key"] == "family:son_rohit")
    assert son["said"] == {"text": "mera beta Rohit", "at": son["said"]["at"], "voice": False, "heard": True} and son["statedBy"] is None
    await links.save_many(db, family_id=FAM, subject_id=ELDER["id"], from_ref=w.fact.key,
                          items=[{"relation": "related_to", "to_domain": "person", "to_name": SON["id"], "as": "son"}],
                          source_kind="caregiver_said", stated_by=SON["id"], source_ref=None)
    rows = links.by_from(await links.links_for(db, FAM, ELDER["id"]))[w.fact.key]
    shown = links_json(rows, {SON["id"]: "Rohit"})
    assert shown and "Rohit" in shown[0]["text"] and SON["id"] not in shown[0]["text"]
