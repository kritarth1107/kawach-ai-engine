"""The backend's delivery ledger (trust plan B1.1–B1.3, B3.4): every message Saheli sends says what it is (a check-in
counts towards the no-reply escalation), the wamid the backend returns is kept on the turn, a reminder's event carries
its wamid into the chat, and messages the backend already gave word for word are never repeated by Saheli."""

from app.api.brain import EventIn, TurnIn, push_event
from app.brain import loop, tools
from app.care import store
from app.sim.world import SimHost

FAM = "fam-ledger"
MAA = {"id": "maa-led", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "son-led", "name": "Ravi", "role": "primary caregiver"}
SYSTEM = {"id": "saheli-scheduler", "role": "system"}


def test_each_message_says_what_it_is():
    assert tools.delivery_kind("checkin:mood:maa:2026-10-14T10:00") == "checkin"
    assert tools.delivery_kind("brief:son:2026-10-14") == "brief"
    assert tools.delivery_kind("wake:123:1") == "message"
    assert tools.delivery_kind("task:wake:9:2") == "message"
    assert tools.delivery_kind(None) == "message"


async def test_a_scheduled_check_in_is_sent_as_a_check_in_and_keeps_its_wamid(db, at):
    at("2026-10-14 10:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="user", text="ok", speaker_id=MAA["id"])
    host = SimHost()
    ref = "checkin:mood:maa-led:2026-10-14T10:00"
    ctx = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker=SYSTEM, members=[MAA, SON], message_ref=ref)
    res = await tools.send_message(ctx, {"to": MAA["id"], "text": "Kamla ji, how are you feeling this morning?"})
    assert res.get("delivered"), res
    sent = host.world.sent[-1]
    assert sent["kind"] == "checkin" and sent["ref"] == ref
    turns = await store.recent_turns(db, FAM, MAA["id"])
    assert turns[-1].meta.get("wamid", "").startswith("wamid.sim")


async def test_a_relay_in_a_conversation_is_a_message(db, at):
    at("2026-10-14 11:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text="hi", speaker_id=SON["id"])
    host = SimHost()
    ctx = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker=MAA, members=[MAA, SON], message_ref="m1")
    res = await tools.send_message(ctx, {"to": SON["id"], "text": "Ravi, Kamla ji asked me to tell you she is fine."})
    assert res.get("delivered"), res
    assert host.world.sent[-1]["kind"] == "message" and "ref" not in host.world.sent[-1]


async def test_the_reminder_event_carries_its_wamid_into_the_chat(db, at):
    at("2026-10-14 18:00")
    payload = {"custom": True, "reminderId": "rem-l", "slot": "18:00", "dateKey": "2026-10-14", "preview": "Time for a short walk",
               "wamid": "wamid.HBgMOTE5"}
    await push_event(EventIn(family_id=FAM, subject_id=MAA["id"], kind="reminder_sent", summary="reminder: walk at 18:00",
                             payload=payload, ref="nudge:n1:sent"), db)
    events = await store.events(db, FAM, MAA["id"], kinds=["reminder_sent"])
    assert events[-1].payload["wamid"] == "wamid.HBgMOTE5"
    turns = await store.recent_turns(db, FAM, MAA["id"])
    assert turns[-1].meta.get("via") == "reminder" and turns[-1].meta.get("wamid") == "wamid.HBgMOTE5"


def test_what_the_backend_already_gave_is_never_repeated():
    given = [{"purpose": "alert", "text": "Kamla ji missed the evening dose.", "at": "2026-10-12T15:30:00Z", "delivered": True}]
    block = loop.waiting_block(loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], text="ok", waiting=given))
    assert block.startswith("JUST GIVEN TO THE SPEAKER") and "do not repeat them" in block
    assert "Kamla ji missed the evening dose." in block and "WAITING FOR THE SPEAKER" not in block
    both = given + [{"purpose": "message", "text": "Your order is on its way.", "at": "2026-10-12T16:00:00Z"}]
    block = loop.waiting_block(loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], text="ok", waiting=both))
    assert "JUST GIVEN TO THE SPEAKER" in block and "WAITING FOR THE SPEAKER" in block
    body = TurnIn(family_id=FAM, elder=MAA, speaker=SON, text="ok", waiting=given)
    assert body.waiting[0].delivered is True
