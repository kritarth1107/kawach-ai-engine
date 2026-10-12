"""The rest of the founder's report of 2026-10-12, 21:52 IST (his Ola for the next afternoon):
- "should have asked which type of cab I want": the fares are shown by type and the person picks; the type they named, or
  the one they took on that app last time, is the suggestion; their pick is saved for next time;
- "it said will require OTP, even when the session was still active": the login is spoken of from the family's saved login;
- the pickup: a saved place with only its area's point is asked for as a location (per pickup), and a Google Maps link they
  send is an exact point like a WhatsApp location."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import store
from app.sim.agent import FARES, FakeAgent
from app.sim.world import SimHost
from app.tasks import matcher, runtime, sandbox

FAM, ME = "fam-rt", "self-rt"
M = {"id": ME, "name": "Vikram", "role": "elder"}
OLA_FARES = {**FARES, "options": [{"type": "Auto", "fare": "₹278", "eta": "2 min"}, {"type": "Mini", "fare": "₹515", "eta": "3 min"},
                                  {"type": "Bike", "fare": "₹131", "eta": "4 min"}, {"type": "Prime Sedan", "fare": "₹540", "eta": "3 min"}]}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class H:
    def __init__(self, sessions, agent):
        self.sessions, self.agent, self.told = sessions, agent, []

    async def profile_for(self, task):
        return {"profileId": f"prof-{task.family_id}-{task.service}", "loginPhone": "9000011111"}

    async def notify(self, family_id, requested_by, prompt):
        self.told.append(prompt)

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def _ride(db, **details):
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="ola", kind="ride", goal="Cab to the office",
                             details={"pickup": "Home", "drop": "Office", "for_set": True, **details})
    await db.commit()
    return t


async def _fares_offered(h, db, t):
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "choice"
    return h.told[-1]


async def test_the_types_are_shown_with_fares_the_pick_is_asked_and_saved_for_next_time(db, at, sessions):
    at("2026-10-12 21:52")
    h = H(sessions, FakeAgent(script={"prepare": [OLA_FARES]}))
    t = await _ride(db)
    said = await _fares_offered(h, db, t)
    for line in ("Auto ₹278", "Mini ₹515", "Bike ₹131", "Prime Sedan ₹540"):
        assert line in said, said
    assert "ask which one to book" in said
    out = await runtime.provide_input(db, t, kind="choice", value="2", by=ME, by_is_elder=True)
    assert "Mini" in out
    await db.commit()
    usual = [f for f in await store.facts(db, FAM, ME, domains=["preference"]) if f.key == "ride_vehicle:ola"]
    assert usual and usual[0].value["vehicle"] == "Mini"


async def test_the_type_they_took_last_time_is_the_suggestion_not_the_cheapest(db, at, sessions):
    at("2026-10-12 21:52")
    h = H(sessions, FakeAgent(script={"prepare": [OLA_FARES]}))
    t = await _ride(db, usual_vehicle="Mini")
    said = await _fares_offered(h, db, t)
    suggested = [o for o in runtime._ride_offers(t) if o["suggested"]]
    assert [o["text"].split(" ₹")[0] for o in suggested] == ["Mini"] and "Bike" not in suggested[0]["text"]
    assert "Mini ₹515" in said


async def test_a_type_they_named_is_the_suggestion(db, at, sessions):
    at("2026-10-12 21:52")
    h = H(sessions, FakeAgent(script={"prepare": [OLA_FARES]}))
    t = await _ride(db, vehicle="sedan", usual_vehicle="Mini")
    await _fares_offered(h, db, t)
    assert [o["text"].split(" ₹")[0] for o in runtime._ride_offers(t) if o["suggested"]] == ["Prime Sedan"]


async def test_start_task_reads_the_riders_usual_type(db, at):
    at("2026-10-12 21:50")
    await store.save_roster(db, FAM, M, [M])
    await store.write_fact(db, family_id=FAM, subject_id=ME, domain="preference", key="ride_vehicle:ola",
                           value={"service": "Ola", "vehicle": "Mini"}, text="Takes Mini on Ola", source_kind="elder_said", stated_by=ME)
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=M, speaker=M, members=[M], message_ref="m1")
    ctx.location = {"lat": 12.93, "lng": 77.58, "name": "", "address": ""}
    await tools.start_task(ctx, {"kind": "ride", "goal": "Cab to the office", "pickup": "here", "drop": "Office", "service": "ola",
                                 "for_whom": ME, "pickup_shared": True})
    t = (await runtime.live_tasks(db, FAM))[0]
    assert t.details["usual_vehicle"] == "Mini"


@pytest.mark.parametrize("live", [True, False])
async def test_the_login_is_spoken_of_from_the_saved_login(db, at, sessions, monkeypatch, live):
    """Founder: "it said it will require an OTP, even when the session was still active"."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 21:51")
    if live:
        await sandbox.note_login(db, FAM, "ola", {"logged_in": True}, phase="prepare")
        await db.commit()
    h = H(sessions, FakeAgent(script={"browse": [{**FARES, "options": [], "logged_in": False, "login_required": True}]}))
    t = await _ride(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    said = h.told[-1]
    assert t.input_needed == "go" and "a login code will come" not in said
    if live:
        assert "the family's Ola login is still active, so no login code should be needed" in said
    else:
        assert "if Ola sends a login code, it goes to" in said and "is asked for only then" in said


class Places(SimHost):
    def __init__(self, precision="pincode"):
        super().__init__()
        self.precision = precision

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "list_places":
            return {"places": [{"name": "Home", "address": "C-5 Sample Park, Indore 452001", "isDefault": True}]}
        if tool == "delivery_place":
            if args.get("exact") and args.get("words") == "Home":
                return {"addressId": "h1", "nickname": "Home", "full": "C-5 Sample Park, Indore 452001", "pincode": "452001",
                        "lat": 22.72, "lng": 75.86, "matched": True, "geoPrecision": self.precision, "defaultFor": [ME]}
            return {"matched": False}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


@pytest.fixture
def home_words(monkeypatch):
    async def which(words, places):
        return "Home" if "home" in words.lower() else None  # the AI's answer, faked

    monkeypatch.setattr(matcher, "which_place", which)


def _ctx(db, host, ref, **kw):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=M, speaker=M, members=[M], message_ref=ref, **kw)


async def test_his_corrected_pickup_is_asked_for_as_a_location_again(db, at, home_words):
    """The first ask was for another pickup; his corrected one, his saved home with only its pincode's point, is asked for
    as a location too (asked once for that pickup)."""
    at("2026-10-12 21:49")
    await store.save_roster(db, FAM, M, [M])
    host = Places()
    out = await tools.start_task(_ctx(db, host, "m1"), {"kind": "ride", "goal": "Cab", "pickup": "City Hospital", "drop": "Office",
                                                       "service": "ola", "for_whom": ME})
    assert out["ask_location"] == "pickup"
    ctx = _ctx(db, host, "m2")
    out = await tools.start_task(ctx, {"kind": "ride", "goal": "Cab", "pickup": "my home", "drop": "City Hospital", "service": "ola",
                                       "for_whom": ME})
    assert out["ask_location"] == "pickup" and ctx.location_request
    assert "Home is saved without its exact point" in out["next"] and "a Google Maps link works too" in out["next"]
    out = await tools.start_task(_ctx(db, host, "m3"), {"kind": "ride", "goal": "Cab", "pickup": "my home", "drop": "City Hospital",
                                                       "service": "ola", "for_whom": ME})
    assert out["status"] == "started", "asked once for that pickup: their answer goes ahead"


async def test_a_saved_home_with_its_exact_point_is_not_asked(db, at, home_words):
    at("2026-10-12 21:49")
    await store.save_roster(db, FAM, M, [M])
    out = await tools.start_task(_ctx(db, Places(precision="user"), "m1"), {"kind": "ride", "goal": "Cab", "pickup": "my home",
                                                                           "drop": "City Hospital", "service": "ola", "for_whom": ME})
    assert out["status"] == "started"


async def test_a_google_maps_link_is_an_exact_point(db, at, monkeypatch, home_words):
    at("2026-10-12 21:49")
    await store.save_roster(db, FAM, M, [M])

    async def follow(link):
        assert link == "https://maps.app.goo.gl/AbCdEf123"
        return ("https://www.google.com/maps/place/City+Hospital/@22.7001,75.8701,17z/data=!3m1!4b1!8m2!3d22.70015!4d75.87022", "")

    monkeypatch.setattr(tools, "_follow", follow)
    host = Places(precision="user")
    out = await tools.start_task(_ctx(db, host, "m1"), {"kind": "ride", "goal": "Cab", "pickup": "my home", "drop": "the link I sent",
                                                       "drop_link": "https://maps.app.goo.gl/AbCdEf123", "service": "ola", "for_whom": ME})
    assert out["status"] == "started"
    t = (await runtime.live_tasks(db, FAM))[0]
    drop = t.details["drop_place"]
    assert (drop["lat"], drop["lng"], drop["precision"], drop["nickname"]) == (22.70015, 75.87022, "user", "City Hospital")


async def test_a_link_that_cannot_be_read_is_asked_again_never_typed_into_the_app(db, at, monkeypatch, home_words):
    at("2026-10-12 21:49")
    await store.save_roster(db, FAM, M, [M])

    async def follow(link):
        return ("https://www.google.com/sorry/index", "<html>unusual traffic</html>")

    monkeypatch.setattr(tools, "_follow", follow)
    out, err = await tools.run(_ctx(db, Places(precision="user"), "m1"), "start_task", {
        "kind": "ride", "goal": "Cab", "pickup": "my home", "drop": "the link", "drop_link": "https://maps.app.goo.gl/x1",
        "service": "ola", "for_whom": ME})
    assert err and "could not be read" in out and not await runtime.live_tasks(db, FAM)
    assert await tools.maps_link_pin("https://example.com/maps?q=1,2") is None, "only Google Maps links"


async def test_his_ride_for_later_is_looked_at_now_and_booked_only_at_its_time(db, at, sessions, monkeypatch):
    """D: Ola shows no fares without a login; the logged-in look-up at the request only looks (never books: the guard);
    the pick goes into the ONE question that says when it will be booked; nothing is booked before then."""
    from datetime import datetime

    from app.core import clock
    from app.sim.agent import BOOKED
    from app.tasks import later, matcher

    async def same(asked, shown, *, saved):
        return True

    monkeypatch.setattr(matcher, "same_ride_end", same)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 21:50")
    tomorrow3 = datetime(2026, 10, 13, 15, 0, tzinfo=clock.IST)
    agent = FakeAgent(script={"browse": [{**FARES, "options": [], "logged_in": False, "login_required": True}],
                              "prepare": [OLA_FARES], "place": [{**BOOKED, "ride_id": "CRN-7"}]})
    h = H(sessions, agent)
    t = await _ride(db, later={"at": tomorrow3.isoformat(), "means": "arrive", "book_at": later.book_time(tomorrow3, "arrive").isoformat()})
    for _ in range(3):
        await h.tick()
    await db.refresh(t)
    assert t.input_needed == "go"
    await runtime.provide_input(db, t, kind="go", value="yes", by=ME, by_is_elder=True)
    await db.commit()
    for _ in range(3):
        await h.tick()
    await db.refresh(t)
    look = [r for r in agent.runs if r["phase"] == "prepare"][0]["goal"]
    assert "Do NOT book" in look and "never press a book, confirm, request, place-order or pay button" in look
    assert t.input_needed == "choice" and not any(r["phase"] == "place" for r in agent.runs)
    ask = await runtime.provide_input(db, t, kind="choice", value="2", by=ME, by_is_elder=True)
    await db.commit()
    assert "nothing is booked now: Mini on Ola" in ask and "booked at 14:50" in ask, ask
    await runtime.provide_input(db, t, kind="later", value="yes", by=ME, by_is_elder=True, max_amount=600)
    await db.commit()
    await db.refresh(t)
    assert t.status == "scheduled"
    at("2026-10-13 14:00")
    await h.tick()
    assert not any(r["phase"] == "place" for r in agent.runs), "nothing is booked before its time"
    at("2026-10-13 14:26")  # the fare is looked at before its time (within their limit: nothing to say)
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert not any(r["phase"] == "place" for r in agent.runs), (t.status, later.of(t))
    book_at = later.parse_at(later.of(t)["book_at"])  # the app's own pickup time moves it a little
    assert clock.ist(book_at).strftime("%d %H:%M") >= "13 14:50"
    at(clock.ist(book_at).strftime("%Y-%m-%d %H:%M"))
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert t.status == "done" and [r["phase"] for r in agent.runs].count("place") == 1, (t.status, later.of(t), [h_["note"][:120] for h_ in t.history[-6:]], [r["phase"] for r in agent.runs])
