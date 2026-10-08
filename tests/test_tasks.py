from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.sim.agent import BOOKED, CANCELLED, CART, FARES, OTP, PLACED, FakeAgent
from app.tasks import runtime
from app.tasks.models import Task

FAM, ELDER, SON = "fam-t", "elder-t", "cg-t"


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class Harness:
    def __init__(self, sessions, agent):
        self.sessions, self.agent, self.told = sessions, agent, []

    async def profile_for(self, task):
        return f"prof-{task.family_id}-{task.service}"

    async def notify(self, family_id, requested_by, prompt):
        self.told.append(prompt)

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def _order(db, **details) -> Task:
    t = await runtime.create(
        db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="Atta for Amma",
        details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}], **details},
    )
    await db.commit()
    return t


async def test_order_prepare_confirm_place(db, at, sessions):
    at("2026-10-02 10:00")
    h = Harness(sessions, FakeAgent(script={"prepare": [CART], "place": [PLACED]}))
    t = await _order(db)
    await h.tick()  # starts prepare
    await h.tick()  # cart ready
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.result["total"] == "₹318"
    assert "confirm" in h.told[-1]
    assert "Do NOT place" in h.agent.runs[0]["goal"] and h.agent.runs[0]["profile"] == "prof-fam-t-instamart"
    assert "Cash on Delivery" in h.agent.runs[0]["hints"]
    msg = await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    assert msg.startswith("confirmed; placing it now")
    await db.commit()
    await h.tick()  # starts place in the same session
    assert h.agent.runs[1]["session"] == "s-t1" and h.agent.runs[1]["phase"] == "place"  # same browser, same login
    await h.tick()
    await db.refresh(t)
    assert t.status == "done" and t.result["order_id"] == "IM-55821"
    assert "Placed" in h.told[-1]


async def test_otp_then_continue(db, at, sessions):
    at("2026-10-02 10:00")
    h = Harness(sessions, FakeAgent(script={"prepare": [OTP, CART]}))
    t = await _order(db)
    await h.tick()
    await h.tick()
    await db.refresh(t)
    assert t.status == "needs_input" and t.input_needed == "otp" and "9888" in h.told[-1]
    assert await runtime.provide_input(db, t, kind="otp", value="hello", by=ELDER, by_is_elder=True) != "code received; continuing"
    assert await runtime.provide_input(db, t, kind="otp", value="4821", by=ELDER, by_is_elder=True) == "code received; continuing"
    await db.commit()
    await h.tick()
    assert "4821" in h.agent.runs[-1]["goal"]
    await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm"


async def test_cancel_before_placing_stops_agent(db, at, sessions):
    at("2026-10-02 10:00")
    agent = FakeAgent(script={"prepare": [CART]}, finish_after_polls=5)
    h = Harness(sessions, agent)
    t = await _order(db)
    await h.tick()
    await db.refresh(t)
    assert t.status == "running"
    msg = await runtime.request_cancel(db, agent, t, by=ELDER, reason="changed mind")
    assert msg == "stopped; nothing was placed" and t.status == "cancelled" and agent.stopped


async def test_cancel_after_placing_runs_cancel_and_asks_about_fee(db, at, sessions):
    at("2026-10-02 10:00")
    fee = {**CANCELLED, "cancelled": False, "cancel_fee": "₹25"}
    h = Harness(sessions, FakeAgent(script={"prepare": [CART], "place": [PLACED], "cancel": [fee, CANCELLED]}))
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=SON, by_is_elder=False); await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "done"
    assert await runtime.request_cancel(db, h.agent, t, by=ELDER, reason="wrong item") == "already placed; cancelling it on the service now"
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "needs_input" and t.input_needed == "fee" and "₹25" in h.told[-1]
    await runtime.provide_input(db, t, kind="fee", value="yes", by=SON, by_is_elder=False); await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "cancelled"


async def test_elder_over_limit_needs_caregiver(db, at, sessions):
    at("2026-10-02 10:00")
    big = {**CART, "total": "₹2,480"}
    h = Harness(sessions, FakeAgent(script={"prepare": [big]}))
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    msg = await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    assert "caregiver must confirm" in msg and t.status == "awaiting_confirm"
    assert (await runtime.provide_input(db, t, kind="confirm", value="yes", by=SON, by_is_elder=False)).startswith("confirmed; placing it now")


async def test_no_cod_or_blocked_fails_without_placing(db, at, sessions):
    at("2026-10-02 10:00")
    h = Harness(sessions, FakeAgent(script={"prepare": [{**CART, "cod_available": False}]}))
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and "Cash on delivery" in h.told[-1]
    assert not [r for r in h.agent.runs if r["phase"] == "place"]


async def test_ride_options_choice_book(db, at, sessions):
    at("2026-10-02 10:00")
    h = Harness(sessions, FakeAgent(script={"prepare": [FARES], "place": [BOOKED]}))
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="uber", kind="ride", goal="Cab to clinic",
                             details={"pickup": "Home", "drop": "Dr Iyer clinic"})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "choice"
    await runtime.provide_input(db, t, kind="choice", value="Auto ₹142", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    assert "Auto ₹142" in h.agent.runs[-1]["goal"]
    await h.tick()
    await db.refresh(t)
    assert t.status == "done" and t.result["ride_id"] == "UB-9921"


async def test_waiting_too_long_expires(db, at, sessions):
    now = at("2026-10-02 10:00")
    h = Harness(sessions, FakeAgent(script={"prepare": [CART]}))
    t = await _order(db)
    await h.tick(); await h.tick()
    at("2026-10-02 11:00")
    await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and "nothing was placed" in h.told[-1]
    assert t.deadline_at < now + timedelta(hours=1)


class LoginHarness(Harness):
    """The live host's answer: the profile and the number of the person who asked (or none)."""

    def __init__(self, sessions, agent, phone):
        super().__init__(sessions, agent)
        self.phone = phone

    async def profile_for(self, task):
        return {"profileId": f"prof-{task.family_id}-{task.service}", "loginPhone": self.phone}


async def test_agent_logs_in_with_the_askers_number(db, at, sessions):
    at("2026-10-08 14:43")
    h = LoginHarness(sessions, FakeAgent(script={"prepare": [OTP, CART]}), "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    goal = h.agent.runs[0]["goal"]
    assert "enter the mobile number 9000012345" in goal and h.agent.runs[0]["profile"] == "prof-fam-t-instamart"
    assert t.status == "needs_input" and "phone ending 2345" in h.told[-1]
    assert t.details["login"] == "2345" and "9000012345" not in str(t.details)


async def test_code_claimed_without_a_number_is_not_believed(db, at, sessions):
    """Live 2026-10-08: Blinkit's login box was empty, yet the agent reported a code 'sent to phone' and the person
    waited for a code that never came. Without a number to log in with, that report is a failure, said plainly."""
    at("2026-10-08 14:43")
    h = LoginHarness(sessions, FakeAgent(script={"prepare": [{**OTP, "otp_sent_to": "phone"}]}), None)
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert "do not try" in h.agent.runs[0]["goal"]
    assert t.status == "failed" and "no code was sent" in h.told[-1]


FOUND = {"logged_in": False, "needs_otp": False, "blocked": False, "problem": "", "deliverable": True, "eta": "12 minutes",
         "items": [{"name": "Aashirvaad Atta 5kg", "price": "₹245", "available": True}]}


async def test_order_looks_first_then_logs_in_after_go_ahead(db, at, sessions, monkeypatch):
    """Founder's order of steps: find the product, price and delivery to their place without logging in; only after
    the person's go-ahead log in (code to their phone), build the cart, confirm, place."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [FOUND], "prepare": [OTP, CART]}), "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    browse = h.agent.runs[0]
    assert browse["phase"] == "browse" and "without logging in" in browse["goal"] and "9000012345" not in browse["goal"]
    assert t.status == "awaiting_confirm" and t.input_needed == "go"
    assert "₹245" in h.told[-1] and "Nothing is ordered" in h.told[-1] and "phone ending 2345" in h.told[-1]
    assert "waiting for go" in await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    assert (await runtime.provide_input(db, t, kind="go", value="haan", by=ELDER, by_is_elder=True)).startswith("going ahead")
    await db.commit()
    await h.tick()
    prep = h.agent.runs[1]
    assert prep["phase"] == "prepare" and "enter the mobile number 9000012345" in prep["goal"] and "Aashirvaad Atta 5kg ₹245" in prep["goal"]
    await h.tick()
    await db.refresh(t)
    assert t.status == "needs_input" and t.input_needed == "otp" and "phone ending 2345" in h.told[-1]
    assert await runtime.provide_input(db, t, kind="otp", value="4821", by=ELDER, by_is_elder=True) == "code received; continuing"
    await db.commit()
    await h.tick()
    code_run = h.agent.runs[-1]["goal"]
    assert "4821" in code_run and "enter the mobile number" not in code_run  # enter the code given; never ask for a new one
    await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm"


async def test_look_up_says_plainly_when_it_does_not_deliver(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [{**FOUND, "items": [], "deliverable": False, "location_set": "Raipur 492001"}]}), "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and "does not deliver" in h.told[-1] and len(h.agent.runs) == 1


async def test_already_logged_in_goes_straight_to_the_cart(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [{**FOUND, "logged_in": True}], "prepare": [CART]}), "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "queued" and t.phase == "prepare" and h.told == []  # nothing to ask: no code needed
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm" and len(h.told) == 1


async def test_ride_shows_fares_before_login_then_logs_in(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    guest = {**FARES, "logged_in": False}
    h = LoginHarness(sessions, FakeAgent(script={"browse": [guest], "prepare": [OTP, FARES]}), "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="rapido", kind="ride", goal="Auto to the station",
                             details={"pickup": "Home", "drop": "Raipur Railway Station"})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert h.agent.runs[0]["phase"] == "browse" and "without logging in" in h.agent.runs[0]["goal"]
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and "Auto ₹142" in h.told[-1] and "Nothing is booked" in h.told[-1]
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.input_needed == "otp" and "enter the mobile number 9000012345" in h.agent.runs[1]["goal"]


async def test_ride_site_that_needs_login_for_fares_asks_to_go_ahead(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    wall = {"logged_in": False, "needs_otp": False, "blocked": False, "problem": "login needed to view fares", "login_required": True}
    h = LoginHarness(sessions, FakeAgent(script={"browse": [wall]}), "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="ola", kind="ride", goal="Cab to the station",
                             details={"pickup": "Home", "drop": "Raipur Railway Station"})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and "only after a login" in h.told[-1]
