"""WhatsApp's 24-hour window (founder 2026-10-12: "WhatsApp has its limitation of a 24-hour window"). Saheli may start a
chat only within 24 hours of the person's last message; after that only an approved Utility template reaches them. The
backend keeps each number's window and sends the template; the message itself waits for their next message, when
Saheli gives it. A check-in has no template: it is not sent (or planned) while the chat is closed."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.brain import TurnIn
from app.brain import loop, tools
from app.care import brief, checkins, store
from app.care.models import CareEvent, OpenLoop
from app.sim.world import SimHost
from app.tasks.models import Task

FAM = "fam-window"
MAA = {"id": "maa-win", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "son-win", "name": "Ravi", "role": "primary caregiver"}
SYSTEM = {"id": "saheli-scheduler", "role": "system"}


def system_ctx(db, host, ref, **kw):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker=SYSTEM, members=[MAA, SON], message_ref=ref, **kw)


async def test_news_for_a_caregiver_whose_chat_is_closed_goes_as_the_alert_template_and_waits(db, at):
    at("2026-10-11 19:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text="ok thanks", speaker_id=SON["id"])
    at("2026-10-12 21:00")
    host = SimHost()
    host.world.closed.add(SON["id"])
    c = system_ctx(db, host, "wake:12:1", profiles={SON["id"]: {"saved": {"language": "hi"}}})
    res = await tools.send_message(c, {"to": SON["id"], "text": "कमला जी ने सुबह के बाद जवाब नहीं दिया। समय मिले तो उन्हें फ़ोन कर लीजिए।"})
    assert res["via"] == "template" and "short notice" in res["note"]
    t = host.world.templates[-1]
    assert t["purpose"] == "alert" and t["params"] == ["Kamla", "कमला जी ने सुबह के बाद जवाब नहीं दिया"] and t["lang"] == "hi"
    assert (await store.recent_turns(db, FAM, SON["id"]))[-1].meta.get("waiting") is True
    # he has not seen it: his next turn's history leaves it out (the WAITING block gives it, as it stands then)
    req = loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], text="hi")
    assert [m.get("role") for m in await loop.history(db, req)] == ["user"]


async def test_a_check_in_to_a_closed_chat_is_not_sent_and_not_counted(db, at):
    at("2026-10-12 17:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    host = SimHost()
    host.world.closed.add(MAA["id"])
    res = await tools.send_message(system_ctx(db, host, "checkin:feelings:maa-win:2026-10-12"), {"to": MAA["id"], "text": "Kaisi hain aap aaj?"})
    assert res["delivered"] is False and res["reason"] == "outside_window"
    assert not host.world.templates and not host.world.sent
    assert not await store.recent_turns(db, FAM, MAA["id"])  # not said: the check-in is tried again once she writes


async def test_an_order_update_to_a_closed_chat_names_the_order(db, at):
    now = at("2026-10-12 09:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    db.add(Task(family_id=FAM, subject_id=MAA["id"], requested_by=SON["id"], service="uber", kind="ride", goal="airport", status="done",
                phase="place", created_at=now, updated_at=now, deadline_at=now))
    await db.commit()
    host = SimHost()
    host.world.closed.add(SON["id"])
    await tools.send_message(system_ctx(db, host, "task:abc123"),
                             {"to": SON["id"], "text": "Your ride is booked for tomorrow morning. The driver details come an hour before."})
    t = host.world.templates[-1]
    assert t["purpose"] == "order" and t["params"] == ["Uber ride", "Your ride is booked for tomorrow morning"]


async def test_an_open_chat_gets_the_message_itself(db, at):
    at("2026-10-12 21:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    host = SimHost()
    res = await tools.send_message(system_ctx(db, host, "wake:12:1"), {"to": SON["id"], "text": "Kamla ji has not answered since the morning check."})
    assert res == {"delivered": True} and host.world.sent and not host.world.templates


async def test_a_brief_for_a_closed_chat_waits_and_the_summary_template_tells_them(db, at):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    at("2026-10-12 22:30")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.write_fact(db, family_id=FAM, subject_id=SON["id"], domain="language", key="language:preferred", value={"language": "en"},
                           text="Writes in English", source_kind="caregiver_said", replace=True)
    await db.commit()
    host = SimHost()
    host.world.closed.add(SON["id"])
    got = await brief.send(sessions, host, await store.roster(db, FAM), SON, "2026-10-12")
    assert got == "waiting"
    assert host.world.templates == [{"to": SON["id"], "purpose": "summary", "params": ["Kamla"], "lang": "en", "at": host.world.templates[0]["at"]}]
    waiting = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "brief"))).scalars().all()
    assert len(waiting) == 1 and waiting[0].owner_id == SON["id"]
    # open again: the brief itself goes, no template
    host.world.closed.clear()
    assert await brief.chat_open(host, db, FAM, SON["id"], None) is True


async def test_no_check_in_is_planned_for_someone_whose_chat_is_closed(db, at, monkeypatch):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    at("2026-10-12 17:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await db.commit()

    async def planned(*a, **k):
        return ("feelings", "[Check-in: feelings] ask how she is")

    turns = []

    async def turn(*a, **k):
        turns.append(a)

    monkeypatch.setattr(checkins, "plan", planned)
    monkeypatch.setattr("app.brain.wake.system_turn", turn)
    host = SimHost()
    host.world.closed.add(MAA["id"])
    await checkins.run(sessions, lambda fid: host)
    assert not turns  # no brain call for a message that cannot go
    assert not (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "checkin"))).scalars().all()
    host.world.closed.clear()
    await checkins.run(sessions, lambda fid: host)
    assert len(turns) == 1


def test_what_waited_is_given_first_in_their_next_turn():
    waited = [{"purpose": "alert", "text": "Kamla ji missed the evening dose.", "at": "2026-10-12T15:30:00Z"}]
    req = loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], text="hi", waiting=waited)
    block = loop.waiting_block(req)
    assert block.startswith("WAITING FOR THE SPEAKER") and "12 Oct 21:00 (alert): Kamla ji missed the evening dose." in block
    assert loop.waiting_block(loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SYSTEM, members=[MAA, SON], text="x", waiting=waited)) == ""
    body = TurnIn(family_id=FAM, elder=MAA, speaker=SON, text="hi", waiting=waited)
    assert body.waiting[0].text == "Kamla ji missed the evening dose."


def test_first_line_is_the_template_line():
    assert tools.first_line("*Kamla ji* ne dawa nahi li. Unhe call kar lijiye.") == "Kamla ji ne dawa nahi li"
    assert tools.first_line("आज सुबह की दवा छूट गई। शाम को फिर पूछूँगी।") == "आज सुबह की दवा छूट गई"
    assert len(tools.first_line("word " * 60)) <= 90
