"""Book at a time (founder 2026-10-12): "book a cab at 6 am Friday from home to the airport", "order milk at 7 tomorrow".
Options now, ONE question with everything and the most they will pay, then booked at its time without asking; the login
code is asked by how long the app keeps a login (never at the last minute); a fare above their limit plus the leeway is
asked once, in time; any app, any store, both roles."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import boundaries, store
from app.core import clock
from app.sim.agent import BOOKED, CART, OTP, PLACED, FakeAgent
from app.sim.world import SimHost
from app.specialists import guard
from app.tasks import later, runtime
from app.tasks.sandbox import ServiceLogin, StoreLogin

FAM, ELDER, SON = "fam-later", "elder-later", "son-later"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
S = {"id": SON, "name": "Ravi", "role": "primary caregiver"}
GUEST = {"logged_in": False, "needs_otp": False, "blocked": False, "problem": ""}
LOGGED = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": ""}
OLA_NOW = {**GUEST, "options": [{"type": "Mini", "fare": "₹548", "eta": "4 min"}, {"type": "Auto", "fare": "₹364", "eta": "6 min"}],
           "pickup_used": "Home, 12 Civil Lines", "drop_used": "Airport"}


def fares(mini: str, eta: str = "6 min") -> dict:
    return {**LOGGED, "options": [{"type": "Mini", "fare": mini, "eta": eta}, {"type": "Auto", "fare": "₹380", "eta": "5 min"}]}


@pytest.fixture(autouse=True)
def ends_ok(monkeypatch):
    """The app set the ride's ends where they were meant (the AI check, faked)."""
    from app.tasks import matcher

    async def same(asked, shown, *, saved):
        return True

    monkeypatch.setattr(matcher, "same_ride_end", same)


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class Harness:
    def __init__(self, sessions, agent):
        self.sessions, self.agent, self.told = sessions, agent, []

    async def profile_for(self, task):
        return {"profileId": f"prof-{task.service}", "loginPhone": "9000012345"}

    async def notify(self, family_id, person, prompt):
        self.told.append((person, prompt))

    async def tick(self, n=1):
        for _ in range(n):
            await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def _until(h, db, t, ok, most=8):
    for _ in range(most):
        await h.tick()
        await db.refresh(t)
        if ok(t):
            return t
    return t


async def _ride_for_later(db, at: datetime, means="arrive") -> runtime.Task:
    await store.save_roster(db, FAM, E, [E, S])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="ola", kind="ride", goal="Cab to the airport",
                             details={"pickup": "Home", "drop": "Airport", "later": {"at": at.isoformat(), "means": means,
                                                                                      "book_at": later.book_time(at, means).isoformat()}})
    await db.commit()
    return t


async def test_a_cab_for_friday_6_am_is_set_once_logged_in_ahead_and_booked_then_without_asking(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 10:00")
    friday = datetime(2026, 10, 16, 6, 0, tzinfo=clock.IST)
    agent = FakeAgent(script={"browse": [OLA_NOW], "prepare": [OTP, fares("₹560")], "place": [{**BOOKED, "ride_id": "CRN-1", "driver": "Ajay",
                              "vehicle": "White Dzire", "plate": "CG04 AB 0001", "pickup_eta": "6 min", "ride_pin": "4321"}]})
    h = Harness(sessions, agent)
    t = await _ride_for_later(db, friday)
    await _until(h, db, t, lambda t: t.input_needed == "go")
    assert "Fares on Ola" in h.told[-1][1]
    ask = await runtime.provide_input(db, t, kind="go", value="1", by=SON, by_is_elder=False)
    await db.commit()
    # ONE question: what, the route as the app set it, when, today's fare, cash, and the most they will pay
    assert "nothing is booked now: Mini on Ola" in ask and "[the app set: Home, 12 Civil Lines]" in ask
    assert "for Fri 16 Oct, 06:00 (booked at 05:50 so it reaches the pickup around then)" in ask
    assert "Today's fare is about ₹548" in ask and "suggest ₹640" in ask and "above about ₹656 I will ask you first" in ask
    assert (t.status, t.input_needed) == ("awaiting_confirm", "later") and not [r for r in agent.runs if r["phase"] == "prepare"]
    said = await runtime.provide_input(db, t, kind="later", value="yes", by=SON, by_is_elder=False, max_amount=640)
    await db.commit()
    # never logged in, 12 h logins by default: the code is asked 9 h before the 05:50 booking (Thu 20:50, before the night)
    assert t.status == "scheduled" and later.of(t)["max"] == 640
    assert "A login code will be asked on Thu 15 Oct, 20:50" in said and "booked on Fri 16 Oct, 05:50" in said
    assert "SET FOR LATER: for Fri 16 Oct, 06:00" in runtime.describe(t)
    await h.tick()
    assert t.status == "scheduled" and len(agent.runs) == 1, "nothing runs until it is due"
    # the login ahead: the code is asked then (with why)
    at("2026-10-15 20:51")
    await _until(h, db, t, lambda t: t.input_needed == "otp")
    assert "logging in now, ahead of it" in h.told[-1][1]
    await runtime.provide_input(db, t, kind="otp", value="123456", by=SON, by_is_elder=False)
    await db.commit()
    agent.script["prepare"] = [{**LOGGED}]
    await _until(h, db, t, lambda t: t.status == "scheduled")
    assert later.of(t)["state"] == "logged_in" and "all set" in h.told[-1][1]
    # 25 minutes before: the fare is looked at; ₹560 is within the limit: nothing to say
    agent.script["prepare"] = [fares("₹560", eta="8 min")]
    before = len(h.told)
    at("2026-10-16 05:26")
    await _until(h, db, t, lambda t: later.of(t).get("state") == "ready")
    assert len(h.told) == before, "a fare within their limit is not news"
    assert later.of(t)["book_at"].startswith("2026-10-16T05:52"), "booked 8 minutes ahead: the app's pickup time"
    # its time: booked without asking, every detail goes out
    at("2026-10-16 05:52")
    await _until(h, db, t, lambda t: t.status == "done")
    place = [r for r in agent.runs if r["phase"] == "place"]
    assert len(place) == 1 and "If the fare shown for it is above ₹656" in place[0]["goal"]
    sent = h.told[-1][1]
    for bit in ("Booked: Mini ₹560", "ride id CRN-1", "car White Dzire CG04 AB 0001", "ride PIN 4321"):
        assert bit in sent, bit


async def test_a_fare_above_their_limit_at_the_check_is_asked_once_in_time(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 10:00")
    t = await _ride_for_later(db, datetime(2026, 10, 12, 15, 0, tzinfo=clock.IST))
    agent = FakeAgent(script={"browse": [OLA_NOW], "prepare": [{**LOGGED}], "place": [{**BOOKED, "ride_id": "CRN-2"}]})
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "go")
    await runtime.provide_input(db, t, kind="go", value="1", by=SON, by_is_elder=False)
    said = await runtime.provide_input(db, t, kind="later", value="yes", by=SON, by_is_elder=False, max_amount=600)
    await db.commit()
    assert "A login code is needed now" in said, "a login made now lasts until 14:50"
    await _until(h, db, t, lambda t: t.status == "scheduled")
    agent.script["prepare"] = [fares("₹720")]
    at("2026-10-12 14:26")
    await _until(h, db, t, lambda t: t.input_needed == "fare")
    ask = h.told[-1][1]
    assert "is ₹720 now, above the ₹600 they set" in ask and "task_input fare yes" in ask
    assert runtime.plain_news([t], SON, {SON: "Ravi"})[0].startswith("For your ride on Mon 12 Oct, 14:54: it costs ₹720 on Ola now")
    out = await runtime.provide_input(db, t, kind="fare", value="yes", by=SON, by_is_elder=False)
    await db.commit()
    assert out.startswith("agreed: it will be booked at Mon 12 Oct, 14:54") and t.status == "scheduled"
    at("2026-10-12 14:54")
    await _until(h, db, t, lambda t: t.status == "done")
    assert "If the fare shown for it is above ₹738" in [r for r in agent.runs if r["phase"] == "place"][0]["goal"]


async def test_a_login_that_ran_out_before_the_check_is_asked_at_once_saying_why(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 10:00")
    t = await _ride_for_later(db, datetime(2026, 10, 12, 13, 0, tzinfo=clock.IST))
    agent = FakeAgent(script={"browse": [OLA_NOW], "prepare": [{**LOGGED}]})
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "go")
    await runtime.provide_input(db, t, kind="go", value="1", by=SON, by_is_elder=False)
    await runtime.provide_input(db, t, kind="later", value="yes", by=SON, by_is_elder=False, max_amount=600)
    await db.commit()
    await _until(h, db, t, lambda t: t.status == "scheduled")
    agent.script["prepare"] = [OTP]
    at("2026-10-12 12:26")
    await _until(h, db, t, lambda t: t.input_needed == "otp")
    assert "For the booking on Mon 12 Oct, 12:50: the Ola login ran out" in h.told[-1][1]


async def test_milk_for_7_tomorrow_is_ordered_then_within_the_limit(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 21:00")
    await store.save_roster(db, FAM, E, [E, S])
    # the store is logged in already (a login seen alive 2 h ago, made 3 h ago): no code is needed for tomorrow 06:48
    db.add(ServiceLogin(family_id=FAM, service="blinkit", profile_id="prof-blinkit", state="ok", last_login_ok_at=clock.now() - timedelta(hours=2)))
    db.add(StoreLogin(family_id=FAM, service="blinkit", login_at=clock.now() - timedelta(hours=3), last_ok_at=clock.now() - timedelta(hours=2)))
    db.add(StoreLogin(family_id="other-fam", service="blinkit", login_at=clock.now() - timedelta(days=9),
                      last_ok_at=clock.now() - timedelta(days=6), died_at=clock.now() - timedelta(days=5)))
    seven = datetime(2026, 10, 13, 7, 0, tzinfo=clock.IST)
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Milk",
                             details={"items": [{"name": "milk", "qty": 2}], "later": {"at": seven.isoformat(), "means": "arrive",
                                                                                        "book_at": later.book_time(seven, "arrive").isoformat()}})
    await db.commit()
    milk = {**GUEST, "deliverable": True, "items": [{"name": "Amul Taaza Toned Milk", "pack": "500 ml", "price": "₹28", "available": True}]}
    cart = {**CART, "items": [{"name": "Amul Taaza Toned Milk 500 ml", "qty": 2, "price": "₹56", "available": True}], "total": "₹71"}
    agent = FakeAgent(script={"browse": [milk], "prepare": [cart], "place": [{**PLACED, "order_id": "B-7", "total": "₹71", "eta": "11 min"}]})
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "go")
    ask = await runtime.provide_input(db, t, kind="go", value="1", by=ELDER, by_is_elder=True)
    await db.commit()
    assert "nothing is ordered now: 2 x Amul Taaza Toned Milk (500 ml) ₹28 on Blinkit, to their saved place" in ask
    assert "ordered at 06:50 so it arrives around then" in ask and "Today's price is about ₹56" in ask and "suggest ₹70" in ask
    said = await runtime.provide_input(db, t, kind="later", value="yes", by=ELDER, by_is_elder=True)  # a plain yes: the suggestion
    await db.commit()
    assert later.of(t)["max"] == 70 and "No login code is needed" in said and t.status == "scheduled"
    at("2026-10-13 06:26")
    await _until(h, db, t, lambda t: later.of(t).get("state") == "ready")
    assert "₹71" == t.details["confirmed_total"], "₹71 is within ₹70 + the leeway"
    at("2026-10-13 06:50")
    await _until(h, db, t, lambda t: t.status == "done")
    assert "Placed: 2 x Amul Taaza Toned Milk 500 ml" in h.told[-1][1] and "order id B-7" in h.told[-1][1]


async def test_the_approvers_yes_comes_up_front_and_a_cancel_before_its_time_stops_it(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 10:00")
    t = await _ride_for_later(db, datetime(2026, 10, 14, 9, 0, tzinfo=clock.IST))
    t.requested_by = ELDER
    await boundaries.save(db, FAM, {"elder_ride_limit": 500, "approvers": [SON]}, by=SON)
    await db.commit()
    agent = FakeAgent(script={"browse": [OLA_NOW]})
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "go")
    await runtime.provide_input(db, t, kind="go", value="1", by=ELDER, by_is_elder=True)
    said = await runtime.provide_input(db, t, kind="later", value="yes", by=ELDER, by_is_elder=True, max_amount=640)
    await db.commit()
    assert said.startswith("needs approval first") and t.input_needed == "approve"
    out = await runtime.provide_input(db, t, kind="approve", value="yes", by=SON, by_is_elder=False)
    await db.commit()
    assert out.startswith("approved. Set for later") and t.status in ("scheduled", "queued")
    stop = await runtime.request_cancel(db, agent, t, by=ELDER, reason="plans changed")
    await db.commit()
    assert "set for later is cancelled; nothing was booked" in stop and t.status == "cancelled"


async def test_a_planned_login_nobody_answers_keeps_the_booking(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 10:00")
    t = await _ride_for_later(db, datetime(2026, 10, 14, 9, 0, tzinfo=clock.IST))
    agent = FakeAgent(script={"browse": [OLA_NOW], "prepare": [OTP]})
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "go")
    await runtime.provide_input(db, t, kind="go", value="1", by=SON, by_is_elder=False)
    await runtime.provide_input(db, t, kind="later", value="yes", by=SON, by_is_elder=False, max_amount=640)
    await db.commit()
    at(clock.ist(later.parse_at(later.of(t)["login_at"])).strftime("%Y-%m-%d %H:%M"))
    await _until(h, db, t, lambda t: t.input_needed == "otp")
    clock.set_now(clock.now() + timedelta(minutes=70))
    await _until(h, db, t, lambda t: t.status == "scheduled")
    assert t.status == "scheduled" and later.of(t)["state"] == "waiting" and later.of(t)["login_at"] is None


def test_when_the_code_is_asked_follows_how_long_the_login_lasts():
    now = datetime(2026, 10, 12, 10, 0, tzinfo=clock.IST)
    book = datetime(2026, 10, 16, 5, 50, tzinfo=clock.IST)
    assert later.plan_login(now, book, timedelta(hours=12), since=None) == datetime(2026, 10, 15, 20, 50, tzinfo=clock.IST)
    # 14 h: 11 h before 05:50 is 18:50; 10 h logins: 7 h before is 22:50, at night, so 20:30 (the login still lasts until then)
    assert later.plan_login(now, book, timedelta(hours=14), since=None) == datetime(2026, 10, 15, 18, 50, tzinfo=clock.IST)
    assert later.plan_login(now, book, timedelta(hours=10), since=None) == datetime(2026, 10, 15, 20, 30, tzinfo=clock.IST)
    assert later.plan_login(now, book, timedelta(days=5), since=None) == now, "a login made now lasts until then: now"
    assert later.plan_login(now, book, timedelta(days=5), since=now - timedelta(hours=1)) is None, "logged in until then: no code"
    noon = datetime(2026, 10, 16, 12, 0, tzinfo=clock.IST)
    # 10 h logins: 7 h before noon is 05:00, at night: 08:00 (the login made then still lasts past noon)
    assert later.plan_login(now, noon, timedelta(hours=10), since=None) == datetime(2026, 10, 16, 8, 0, tzinfo=clock.IST)
    assert later.suggest_max(548) == 640 and round(later.cap(900)) == 922 and later.cap(200) == 210


async def test_how_long_a_login_lasts_is_measured_from_the_runs(db, at):
    from app.tasks import sandbox

    at("2026-10-12 10:00")
    assert await later.login_life(db, "uber") == later.DEFAULT_LOGIN_LIFE
    await sandbox.note_login(db, "f1", "uber", {"logged_in": True}, fresh=True, phase="prepare")
    at("2026-10-13 22:00")
    await sandbox.note_login(db, "f1", "uber", {"logged_in": True}, phase="prepare")
    assert await later.login_life(db, "uber") == timedelta(hours=36), "seen alive for 36 h"
    await sandbox.note_login(db, "f1", "uber", {"logged_in": False}, phase="browse")
    assert await later.login_life(db, "uber") == timedelta(hours=36), "a guest look is not a lost login"
    at("2026-10-14 10:00")
    await sandbox.note_login(db, "f1", "uber", {"needs_otp": True}, phase="prepare")
    assert await later.login_life(db, "uber") == timedelta(hours=48) * 0.9, "gone by 48 h: a little less than that"


def test_a_price_that_moved_after_the_yes_goes_ahead_within_the_leeway():
    assert guard.within_leeway(920, 900) and not guard.within_leeway(925, 900)
    assert guard.within_leeway(109, 100) and not guard.within_leeway(111, 100), "never less than ₹10"


async def test_the_brain_sets_a_ride_for_later_with_a_time(db, at):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, E, [E, S])
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=S, members=[E, S], message_ref="m1")
    ride = {"kind": "ride", "goal": "Cab to the airport", "pickup": "Home", "drop": "Airport", "service": "ola", "at": "2026-10-16T06:00",
            "for_whom": ELDER}
    assert (await tools.start_task(ctx, ride))["ask_location"] == "pickup", "the pickup as a location first, for later too"
    ctx.message_ref = "m2"
    out = await tools.start_task(ctx, ride)
    t = (await runtime.shown_tasks(db, FAM))[0]
    assert later.of(t)["book_at"] == "2026-10-16T05:50:00+05:30" and later.of(t)["means"] == "arrive"
    out, err = await tools.run(ctx, "start_task", {"kind": "ride", "goal": "cab", "pickup": "Home", "drop": "Station", "service": "uber", "for_whom": ELDER,
                                                    "at": "2026-10-11T06:00"})
    assert err and "has passed" in out


async def test_no_limit_still_keeps_the_familys_approval_limit_and_an_unanswered_fare_is_said_even_at_dawn(db, at, sessions, monkeypatch):
    """"Book whatever it costs": no limit is recorded, but the family's approval limit still holds at the fare the check
    finds (the approver is asked then, before its time). A booking nobody answered for by its time is always said, also in
    the early morning when other nudges wait."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 10:00")
    t = await _ride_for_later(db, datetime(2026, 10, 13, 6, 0, tzinfo=clock.IST))
    t.requested_by = ELDER
    await boundaries.save(db, FAM, {"elder_ride_limit": 600, "approvers": [SON]}, by=SON)
    await db.commit()
    agent = FakeAgent(script={"browse": [OLA_NOW], "prepare": [{**LOGGED}]})
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "go")
    await runtime.provide_input(db, t, kind="go", value="1", by=ELDER, by_is_elder=True)
    said = await runtime.provide_input(db, t, kind="later", value="yes", by=ELDER, by_is_elder=True, no_max=True)
    await db.commit()
    assert later.of(t)["no_max"] and not said.startswith("needs approval"), "₹548 today is within her ₹600 limit"
    await _until(h, db, t, lambda t: t.status == "scheduled")
    agent.script["prepare"] = [fares("₹720")]
    at("2026-10-13 05:26")
    await _until(h, db, t, lambda t: t.input_needed == "approve")
    assert "needs approval first" in h.told[-1][1] or any("[Approval needed]" in p for _, p in h.told)
    # nobody answers by its time: said plainly at 06:10, though it is early morning
    at("2026-10-13 06:15")
    await _until(h, db, t, lambda t: t.status == "failed")
    assert "NOT booked: the Ola ride set for Tue 13 Oct, 06:00" in h.told[-1][1]


async def test_a_booking_that_could_not_run_by_its_time_is_said(db, at, sessions):
    at("2026-10-12 10:00")
    t = await _ride_for_later(db, datetime(2026, 10, 12, 18, 0, tzinfo=clock.IST))
    t.status = "scheduled"
    t.details = {**t.details, "later": {**later.of(t), "by": SON, "max": 600, "state": "waiting", "login_at": None}}
    await db.commit()
    h = Harness(sessions, FakeAgent())
    at("2026-10-12 18:30")  # the engine was down at the check and at its time
    await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and "could not be done by its time" in h.told[-1][1] and h.told[-1][0] == SON


async def test_the_question_for_later_has_one_tap_answers_that_carry_the_limit(db, at, sessions, monkeypatch):
    """Sim 2026-10-12: the brain left the limit out of the question and recorded "no limit" on a plain yes. The question
    carries buttons: yes (up to the suggested limit), any amount, no; a tap is data, in every Indian script."""
    from app.brain import loop
    from app.brain.loop import TurnRequest
    from app.care import outcomes

    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 10:00")
    t = await _ride_for_later(db, datetime(2026, 10, 14, 9, 0, tzinfo=clock.IST))
    h = Harness(sessions, FakeAgent(script={"browse": [OLA_NOW]}))
    await _until(h, db, t, lambda t: t.input_needed == "go")
    await runtime.provide_input(db, t, kind="go", value="1", by=SON, by_is_elder=False)
    await db.commit()
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=S, members=[E, S])
    out, err = await tools.run(ctx, "offer_buttons", {"kind": "later", "key": str(t.id)})
    assert not err and [b["id"] for b in ctx.buttons] == [f"v2:lt:yes:{t.id}:640", f"v2:lt:any:{t.id}:640", f"v2:lt:no:{t.id}:640"]
    assert [b["title"] for b in outcomes.buttons("later", "k", {"script": "bengali"})] == ["হ্যাঁ, করে দাও", "যত হোক", "না"]
    news, spec = runtime.plain_news([t], SON, {SON: "Ravi"})
    assert spec == ("later", f"{t.id}:640") and "paying up to ₹640" in news
    note = await outcomes.handle_later_button(db, family_id=FAM, speaker_id=SON, elder_id=ELDER, code="yes", key=f"{t.id}:640")
    await db.commit()
    assert note.startswith(f"(tapped yes, up to ₹640 on the booking for later {t.id}: Set for later") and later.of(t)["max"] == 640
    # any amount, on another one
    t2 = await _ride_for_later(db, datetime(2026, 10, 15, 9, 0, tzinfo=clock.IST))
    t2.service = "uber"
    await db.commit()
    agent2 = FakeAgent(script={"browse": [OLA_NOW]})
    h2 = Harness(sessions, agent2)
    await _until(h2, db, t2, lambda t: t.input_needed == "go")
    await runtime.provide_input(db, t2, kind="go", value="1", by=SON, by_is_elder=False)
    await db.commit()
    note = await outcomes.handle_later_button(db, family_id=FAM, speaker_id=SON, elder_id=ELDER, code="any", key=f"{t2.id}:640")
    await db.commit()
    assert later.of(t2)["no_max"] is True and later.of(t2)["max"] is None
    assert outcomes.parse_button(f"v2:lt:no:{t2.id}:640") == ("lt", "no", f"{t2.id}:640")


async def test_an_order_whose_total_moved_after_the_yes_goes_ahead_within_the_leeway_and_asks_above_it(db, at):
    """Founder 2026-10-12 (the same leeway everywhere a price moves after the confirm): the cart is rebuilt after a place
    step could not send; the same products at ₹715 go on a ₹700 yes; at ₹720 (above ₹717.5) they are asked again."""
    at("2026-10-12 12:30")
    await store.save_roster(db, FAM, E, [E, S])
    cart = {"items": [{"name": "Aashirvaad Atta 5kg", "qty": 2, "price": "₹580"}], "total": "₹700", "cod_available": True}
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="zepto", kind="order", goal="atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 2}]})
    t.status, t.phase, t.input_needed, t.result = "awaiting_confirm", "prepare", "confirm", dict(cart)
    t.details = {**t.details, "cart_fp": guard.cart_fingerprint(cart)}
    runtime._start_placing(t, SON)
    d = t.details
    for total, placed in (("₹715", True), ("₹720", False)):
        t.phase, t.status = "prepare", "running"
        t.details = {**d, "replace_confirmed": {"fp": d["confirmed_fp"], "total": d["confirmed_total"], "by": SON,
                                                 "items_fp": d["confirmed_items_fp"]}, "auto_replaced": 0, "total_relook": True}
        status, msg = runtime._outcome(t, {**cart, "total": total})
        assert (status == "queued" and t.phase == "place") is placed, (total, status, msg[:120])
    # the agent's place goal uses the same leeway
    from app.specialists.agents import specialist_for

    t.phase, t.details = "place", {**t.details, "confirmed_total": "₹700"}
    assert "the total is now more than ₹718" in specialist_for("zepto").goal(t)


async def test_the_later_buttons_go_only_on_the_question_for_later(db, at, sessions):
    """Sim 2026-10-12: the brain put the later buttons on a ride's fare confirm for now; they are dropped there (the message
    still goes), and on the question for later they carry the suggested limit."""
    at("2026-10-12 12:40")
    await store.save_roster(db, FAM, E, [E, S])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="uber", kind="ride", goal="cab",
                             details={"pickup": "Home", "drop": "Station"})
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "choice"
    await db.commit()
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker={"id": "saheli-scheduler", "role": "system"}, members=[E, S])
    out, err = await tools.run(ctx, "send_message", {"to": SON, "text": "Book Mini for ₹213?", "buttons": {"kind": "later", "key": str(t.id)}})
    assert not err and "buttons" not in ctx.host.world.sent[-1], "no later buttons on a confirm for now"
