"""The nightly brief (founder 2026-10-12): each caregiver gets the care recipient's whole day on WhatsApp, by default
30 minutes after the care recipient's bedtime, at a time they can change; once a day; never lost when WhatsApp's 24-hour
window is closed."""

import json
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import loop, tools
from app.care import brief, store

FAM = "fam-brief"
MAA = {"id": "maa-brief", "name": "Babli", "role": "elder"}
SON = {"id": "son-brief", "name": "Ravi", "role": "primary caregiver"}
NIECE = {"id": "niece-brief", "name": "Asha", "role": "caregiver"}
DOC = {"id": "doc-brief", "name": "Dr Rao", "role": "family doctor"}


class Host:
    def __init__(self, items=None):
        self.items = items or []
        self.calls = []

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args))
        if tool == "get_today_schedule":
            return {"items": self.items}
        return {"ok": True, "delivered": True}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def _family(db, members=(MAA, SON)):
    await store.save_roster(db, FAM, MAA, list(members))
    await db.commit()


async def _sleep(db, value):
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="routine", key="routine:sleep", value=value, text="sleep",
                           source_kind="caregiver_said", replace=True)
    await db.commit()


async def _wrote(db, who, at, text="ok"):
    await store.add_turn(db, family_id=FAM, thread_id=who, role="user", text=text, speaker_id=who)
    (await store.recent_turns(db, FAM, who))[-1].at = at
    await db.commit()


async def test_the_default_time_is_after_the_care_recipients_bedtime(db, at):
    at("2026-10-12 12:00")
    await _family(db)
    assert (await brief.default_time(db, FAM, MAA["id"])).strftime("%H:%M") == "22:00", "no bedtime saved"
    await _sleep(db, {"time": "21:30"})
    assert (await brief.default_time(db, FAM, MAA["id"])).strftime("%H:%M") == "22:00"
    # a caregiver whose night Saheli doesn't know: the default quiet hours (21:00-08:00) count, so the brief goes before
    # them instead of being refused as night (live 2026-10-13)
    assert (await brief.brief_time(db, FAM, SON["id"], MAA["id"])).strftime("%H:%M") == "20:30"
    await _sleep(db, {"times": ["23:30", "00:30"]})
    assert (await brief.default_time(db, FAM, MAA["id"])).strftime("%H:%M") == "23:45", "never past midnight by default"
    await _sleep(db, {"time": "01:00"})
    assert (await brief.default_time(db, FAM, MAA["id"])).strftime("%H:%M") == "22:00", "a bedtime after midnight: the default"


async def test_due_once_at_its_time_and_a_morning_time_brings_yesterday(db, at):
    t = brief._hm("22:30")
    assert brief.due_day(t, at("2026-10-12 22:25")) is None
    assert brief.due_day(t, at("2026-10-12 22:31")) == "2026-10-12"
    assert brief.due_day(t, at("2026-10-13 00:45")) is None, "more than two hours late: not in the middle of the night"
    assert brief.due_day(brief._hm("08:00"), at("2026-10-13 08:05")) == "2026-10-12"


async def test_the_brief_holds_the_whole_day_as_data(db, at):
    from app.tasks.models import Task

    now = at("2026-10-12 22:30")
    await _family(db)
    host = Host([{"scheduleId": "s1", "title": "Metformin", "dosage": "500 mg", "time": "08:00", "status": "completed", "markedBy": MAA["id"]},
                 {"scheduleId": "s2", "title": "Metformin", "dosage": "500 mg", "time": "20:00", "status": "due", "markedBy": None}])
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="vital", summary="BP 150/95", at=now - timedelta(hours=10))
    await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="symptom", summary="knee pain", at=now - timedelta(hours=9))
    await _wrote(db, MAA["id"], now - timedelta(hours=8), "aaj ghutne me dard hai")
    db.add(Task(family_id=FAM, subject_id=MAA["id"], requested_by=MAA["id"], service="blinkit", kind="order", goal="atta",
                details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}], "confirmed_total": "₹318"}, status="done", phase="place",
                result={"placed": True, "total": "₹318", "cart": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]}, history=[],
                created_at=now - timedelta(hours=6), updated_at=now - timedelta(hours=6), deadline_at=now))
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "times": ["08:00", "20:00"]}, text="Metformin 500 mg", source_kind="caregiver_said")
    await db.commit()
    text = await brief.digest(db, host, FAM, MAA, "2026-10-12", now=now)
    assert "08:00 Metformin 500 mg: taken" in text and "20:00 Metformin 500 mg: not marked" in text
    assert "vital: BP 150/95" in text and "symptom: knee pain" in text
    assert "1 messages" in text and "ghutne me dard" in text
    assert "blinkit order" in text and "placed" in text and "₹318" in text
    assert "TOMORROW: Metformin at 08:00, 20:00" in text
    p = brief.prompt_for(SON, MAA, "2026-10-12", text)
    assert p.startswith("[Daily brief]") and "never 'taken'" in p and "person son-brief" in p


async def test_each_caregiver_gets_it_once_and_a_closed_window_keeps_it_waiting(db, at, sessions, monkeypatch):
    now = at("2026-10-12 20:01")
    await _family(db, (MAA, SON, NIECE, DOC))
    await _sleep(db, {"time": "19:30"})
    await _wrote(db, SON["id"], now - timedelta(hours=3))  # inside WhatsApp's window; the niece last wrote two days ago
    await _wrote(db, NIECE["id"], now - timedelta(days=2))
    sent: list[tuple[str, str]] = []

    async def fake_turn(sessions_, host, family_id, prompt, ref, deliver_to=None, still_true=None):
        sent.append((deliver_to, ref))
        async with sessions_() as s:
            await store.add_turn(s, family_id=family_id, thread_id=deliver_to, role="assistant", text="Aaj Babli ji ka din…",
                                 meta={"proactive": True, "delivered": True, "ref": ref})
            await s.commit()

    from app.brain import wake

    monkeypatch.setattr(wake, "system_turn", fake_turn)
    out = await brief.run(sessions, lambda fid: Host(), now=now)
    assert out == {"due": 2, "sent": 1, "waiting": 1}, out
    assert [to for to, _ in sent] == [SON["id"]], "the doctor gets none; the niece's waits"
    assert await brief.run(sessions, lambda fid: Host(), now=now + timedelta(minutes=10)) == {"due": 0, "sent": 0, "waiting": 0}, "once a day"
    waiting = await brief.waiting_for(db, FAM, NIECE["id"])
    assert len(waiting) == 1 and "DAY: 2026-10-12" in waiting[0].detail["text"]
    # when she writes, the brief is in her turn's context; the care recipient never sees it
    req = loop.TurnRequest(family_id=FAM, elder=MAA, speaker=NIECE, members=[MAA, SON, NIECE], text="hi")
    dynamic, _ = await loop.turn_context(db, req)
    assert "A DAILY BRIEF IS WAITING FOR THE SPEAKER" in dynamic and f"close_loop {waiting[0].id}" in dynamic
    elder_req = loop.TurnRequest(family_id=FAM, elder=MAA, speaker=MAA, members=[MAA, SON, NIECE], text="hi")
    assert "DAILY BRIEF IS WAITING" not in (await loop.turn_context(db, elder_req))[0]


async def test_if_saheli_sends_nothing_the_raw_data_never_goes(db, at, sessions, monkeypatch):
    """Live 2026-10-13: the writer's message was refused and the raw data block went out as the brief. It is written for
    the writer, never for a person: she is asked once more, then the brief waits for their next message."""
    now = at("2026-10-12 20:31")
    await _family(db)
    await _wrote(db, SON["id"], now - timedelta(hours=1))
    from app.brain import wake

    tries: list[str] = []

    async def silent(sessions_, host, family_id, prompt, ref, deliver_to=None, still_true=None):
        tries.append(prompt[:60])

    monkeypatch.setattr(wake, "system_turn", silent)
    host = Host()
    roster = await store.roster(db, FAM)
    assert await brief.send(sessions, host, roster, SON, "2026-10-12", now=now) == "waiting"
    assert len(tries) == 2 and tries[1].startswith("(Your last try sent nothing")
    assert not [args for tool, args in host.calls if tool == "send_whatsapp"], "no raw data on WhatsApp"
    waiting = await brief.waiting_for(db, FAM, SON["id"])
    assert len(waiting) == 1 and "DAY: 2026-10-12" in waiting[0].detail["text"]
    assert await brief.sent_already(db, FAM, SON["id"], "2026-10-12"), "not tried again every five minutes"


async def test_at_their_night_the_brief_waits_instead_of_waking_them(db, at, sessions, monkeypatch):
    now = at("2026-10-12 22:31")
    await _family(db)
    await _wrote(db, SON["id"], now - timedelta(hours=1))
    from app.brain import wake

    tries: list[str] = []

    async def refused(sessions_, host, family_id, prompt, ref, deliver_to=None, still_true=None):
        tries.append(ref)  # send_message refuses at night, so nothing goes out

    monkeypatch.setattr(wake, "system_turn", refused)
    host = Host()
    roster = await store.roster(db, FAM)
    assert await brief.send(sessions, host, roster, SON, "2026-10-12", now=now) == "waiting"
    assert len(tries) == 1, "no second try at night"
    assert not [a for t, a in host.calls if t in ("send_whatsapp", "send_template")]
    assert len(await brief.waiting_for(db, FAM, SON["id"])) == 1


async def test_the_brief_never_goes_to_the_person_it_is_about(db, at, sessions, monkeypatch):
    now = at("2026-10-12 20:31")
    await _family(db, (MAA, SON))
    from app.brain import wake

    called: list[str] = []

    async def turn(sessions_, host, family_id, prompt, ref, deliver_to=None, still_true=None):
        called.append(deliver_to)

    monkeypatch.setattr(wake, "system_turn", turn)
    host = Host()
    roster = await store.roster(db, FAM)
    assert await brief.send(sessions, host, roster, MAA, "2026-10-12", now=now) == "empty"
    assert not called and not host.calls


async def test_a_brief_time_the_caregiver_chose_is_not_refused_as_night(db, at):
    from app.brain import tools as brain_tools

    now = at("2026-10-12 21:20")
    await _family(db)
    await _wrote(db, SON["id"], now - timedelta(hours=1))
    host = Host()
    sys_ctx = brain_tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker={"id": "saheli-scheduler", "name": "Saheli", "role": "system"},
                                  members=[MAA, SON], message_ref=f"brief:{SON['id']}:2026-10-12")
    out, _ = await brain_tools.run(sys_ctx, "send_message", {"to": SON["id"], "text": "Here is Babli ji's day: morning tablets taken."})
    assert "night for them" in out, "without a chosen time 21:20 is inside the default quiet hours"
    ctx = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON])
    await tools.run(ctx, "daily_brief", {"action": "set", "time": "21:15"})
    await db.commit()
    sys_ctx = brain_tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker={"id": "saheli-scheduler", "name": "Saheli", "role": "system"},
                                  members=[MAA, SON], message_ref=f"brief:{SON['id']}:2026-10-12")
    out, err = await brain_tools.run(sys_ctx, "send_message", {"to": SON["id"], "text": "Here is Babli ji's day: morning tablets taken."})
    assert "night for them" not in out, out


async def test_a_caregiver_sets_their_own_time_turns_it_off_or_asks_now(db, at):
    at("2026-10-12 18:00")
    await _family(db)
    host = Host([{"scheduleId": "s1", "title": "BP tablet", "time": "09:00", "status": "completed", "markedBy": MAA["id"]}])

    def ctx(speaker):
        return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker=speaker, members=[MAA, SON])

    out, err = await tools.run(ctx(SON), "daily_brief", {"action": "set", "time": "21:15"})
    assert not err and json.loads(out) == {"result": "on", "time": "21:15", "default": False}
    assert (await brief.brief_time(db, FAM, SON["id"], MAA["id"])).strftime("%H:%M") == "21:15"
    out, err = await tools.run(ctx(SON), "daily_brief", {"action": "off"})
    assert not err and json.loads(out)["result"] == "off" and await brief.brief_time(db, FAM, SON["id"], MAA["id"]) is None
    out, err = await tools.run(ctx(SON), "daily_brief", {"action": "on"})
    assert not err and json.loads(out)["time"] == "21:15", "back on at their own time"
    out, err = await tools.run(ctx(SON), "daily_brief", {"action": "now"})
    assert not err and "09:00 BP tablet: taken" in json.loads(out)["brief_data"]
    _, err = await tools.run(ctx(MAA), "daily_brief", {"action": "set", "time": "20:00"})
    assert err, "the care recipient has no brief to set"
    _, err = await tools.run(ctx(SON), "daily_brief", {"action": "set", "time": "nine"})
    assert err


class DownHost(Host):
    """The backend restarting (a deploy): every call is a 503."""

    async def call(self, tool, args, **kw):
        import httpx

        self.calls.append((tool, args))
        req = httpx.Request("POST", "http://backend/internal/v1/saheli/tools/execute")
        raise httpx.HTTPStatusError("503", request=req, response=httpx.Response(503, request=req))


async def test_a_backend_restart_at_brief_time_defers_it_and_the_next_run_sends_it_whole(db, at, sessions, monkeypatch):
    # live 2026-10-14 20:30: two briefs were built during a backend restart, without the doses, and held
    now = at("2026-10-12 20:01")
    await _family(db, (MAA, SON))
    await _sleep(db, {"time": "19:30"})
    await _wrote(db, SON["id"], now - timedelta(hours=3))
    sent: list[str] = []

    async def fake_turn(sessions_, host, family_id, prompt, ref, deliver_to=None, still_true=None):
        sent.append(prompt)
        async with sessions_() as s:
            await store.add_turn(s, family_id=family_id, thread_id=deliver_to, role="assistant", text="Aaj Babli ji ka din…",
                                 meta={"proactive": True, "delivered": True, "ref": ref})
            await s.commit()

    from app.brain import wake

    monkeypatch.setattr(wake, "system_turn", fake_turn)
    out = await brief.run(sessions, lambda fid: DownHost(), now=now)
    assert out.get("deferred") == 1 and out["sent"] == 0 and out["waiting"] == 0, out
    assert not sent and not await brief.waiting_for(db, FAM, SON["id"]), "nothing held without its schedule"
    host = Host([{"scheduleId": "s1", "title": "Metformin", "dosage": "500 mg", "time": "08:00", "status": "completed"}])
    assert await brief.run(sessions, lambda fid: host, now=now + timedelta(minutes=5)) == {"due": 1, "sent": 1, "waiting": 0}
    assert "08:00 Metformin 500 mg: taken" in sent[0]
