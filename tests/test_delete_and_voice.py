"""A saved event they take back is removed with forget, and a chosen line can go as a voice note.

Live 14 Oct: the brain called stop for a joked chest wound, then said an event cannot be deleted. forget already
hid events, but the tool told it to use stop, and the day's ledger still read the event back. A request to send
particular words as a voice note had no tool, so she said she could not.
"""

from app.brain import tools
from app.care import checkins, digest, memory_upkeep, rejected, store, symptom_patterns
from app.sim.world import SimHost

FAM, ELDER, CG = "fam-del", "e-del", "c-del"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
C = {"id": CG, "name": "Asha", "role": "primary caregiver"}


def ctx(db, speaker, host=None, **kw):
    return tools.TurnCtx(session=db, host=host or SimHost(), family_id=FAM, elder=E, speaker=speaker, members=[E, C], **kw)


def spec(name):
    return next(s for s in tools.specs() if s.name == name)


def test_delete_tools_point_at_forget_and_voice():
    forget = spec("forget")
    assert "not stop" in forget.description and "never_true" in forget.schema["properties"]
    assert "use stop instead" not in forget.description
    send = spec("send_message")
    assert "voice" in send.schema["properties"] and "voice note" in send.description


async def test_a_retracted_event_is_not_read_back(db, at):
    at("2026-10-14 11:00")
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="symptom", summary="chest wound, said as a joke")
    out, err = await tools.run(ctx(db, E), "forget", {"what": "chest wound", "never_true": True})
    assert not err and "Do not mention it again" in out
    db.expire_all()
    day = await store.events(db, FAM, ELDER, day="2026-10-14")
    assert not any("wound" in (e.summary or "") for e in day)
    assert "wound" not in digest.ledger(day)
    assert not [h for h in await store.recall(db, FAM, [ELDER], "chest wound") if "wound" in h.text]
    assert not await symptom_patterns.events(db, FAM, ELDER)
    assert not await checkins._symptoms_since(db, FAM, ELDER, None, at("2026-10-14 12:00"))


async def test_stop_on_an_event_says_to_call_forget(db, at):
    at("2026-10-14 11:00")
    out, err = await tools.run(ctx(db, E), "stop", {"domain": "condition", "name": "chest wound", "reason": "it was a joke"})
    assert not err and "forget" in out and "not_found" in out


async def test_forget_stops_a_matching_fact_and_a_joke_is_not_in_the_record(db, at):
    at("2026-10-14 11:00")
    saved, err = await tools.run(ctx(db, C), "remember", {
        "domain": "condition", "name": "chest wound", "details": {"name": "chest wound"}, "sentence": "Chest wound on file",
    })
    assert not err and "created" in saved
    out, err = await tools.run(ctx(db, E), "forget", {"what": "chest wound", "never_true": True})
    assert not err and "Do not mention it again" in out
    db.expire_all()
    active = await store.facts(db, FAM, ELDER, statuses=("active",))
    assert not any("wound" in f.text.lower() for f in active) or "facts_waiting" in out
    rows = await store.facts(db, FAM, ELDER)
    rec = digest.care_record("Kamla", rows, rejected=await rejected.rejected(db, FAM, ELDER))
    head, _, tail = rec.partition("Never say")
    assert "wound" not in head.lower()
    assert "wound" in tail.lower()
    card = await memory_upkeep.profile_card(db, FAM, E)
    assert "wound" not in card.lower()
    # a non-health fact ends at once
    await tools.run(ctx(db, C), "remember", {
        "domain": "preference", "name": "bhajans", "details": {}, "sentence": "Loves morning bhajans",
    })
    gone, err = await tools.run(ctx(db, C), "forget", {"what": "bhajans"})
    assert not err and '"facts"' in gone
    db.expire_all()
    assert not any(f.domain == "preference" and f.status == "active" for f in await store.facts(db, FAM, ELDER))


async def test_voice_note_goes_to_the_asker_or_a_family_member(db, at):
    at("2026-10-14 11:00")
    host = SimHost()
    said = "send a voice note saying I love you"
    me = ctx(db, E, host, user_text=said)
    out, err = await tools.run(me, "send_message", {"to": ELDER, "text": "I love you", "voice": True})
    assert not err and "voice note" in out
    assert host.world.sent[-1]["voice"] is True and host.world.sent[-1]["to"] == ELDER
    out, err = await tools.run(me, "send_message", {"to": CG, "text": "I love you", "voice": True})
    assert not err and host.world.sent[-1]["to"] == CG and host.world.sent[-1]["voice"] is True
    out, err = await tools.run(me, "send_message", {"to": ELDER, "text": "I love you"})
    assert err and "voice true" in out
    out, err = await tools.run(me, "send_message", {"to": "stranger", "text": "I love you", "voice": True})
    assert err and "HOUSEHOLD" in out
