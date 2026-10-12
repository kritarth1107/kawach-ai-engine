"""Nothing is booked or placed before the final confirm, and nothing booked is ever denied (founder 2026-10-12, 21:52 IST).

His Ola ride for the next afternoon: Ola was still logged in, the fare look-up pressed "Confirm & Book" "to make the login
appear", an auto came to his door, the agent's report left the booking out, his cancel wrote "stopped before placing" and
Saheli told him "nothing was booked". Now a look-up runs without the family's login; a run before the confirm is stopped
on the page with the final button or on a page that only follows a booking; a booking it made anyway is owned at once and
cancelled on the store; and no cancel says "nothing was booked" before the store's own list was read."""

import dataclasses
from dataclasses import dataclass, field

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.sim.agent import CANCELLED, CART, CHECKED, FARES, FakeAgent
from app.specialists import guard
from app.tasks import runtime
from app.tasks.models import Task

FAM, ME = "fam-nb", "self-nb"
OLA = "https://book.olacabs.com/"
# his run's own pages (Browser Use steps of 2026-10-12 16:22 UTC), the agent's report without the booking
BOOKED_PAGES = [OLA + "confirm-ride-p2p?pickup_name=x", OLA + "?pickup_name=x", OLA + "confirm-ride-p2p?pickup_name=x",
                OLA + "ride-detail/CRN11420826289?pickup_name=x", OLA + "cancel-ride?pickup_name=x", OLA + "?pickup_name=x"]
HIDDEN = {**FARES, "booked": None, "ride_id": None}


@dataclass
class PathAgent(FakeAgent):
    """A fake agent whose runs are on the given pages (per phase), also while they are still running."""

    paths: dict = field(default_factory=dict)

    async def poll(self, task_id: str):
        run = await super().poll(task_id)
        phase = next((r["phase"] for r in self.runs if r["task"] == task_id), "prepare")
        return dataclasses.replace(run, path=list(self.paths.get(phase, self.path)))


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class H:
    def __init__(self, sessions, agent):
        self.sessions, self.agent, self.told = sessions, agent, []

    async def profile_for(self, task):
        return f"prof-{task.family_id}-{task.service}"

    async def notify(self, family_id, requested_by, prompt):
        self.told.append(prompt)

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def _ride(db, **details) -> Task:
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="ola", kind="ride",
                             goal="Cab from home to the office", details={"pickup": "Home", "drop": "Office", "for_set": True, **details})
    await db.commit()
    return t


async def test_his_case_the_fare_look_up_that_booked_is_owned_and_cancelled_on_the_app(db, at, sessions):
    at("2026-10-12 21:52")
    agent = PathAgent(script={"prepare": [HIDDEN], "cancel": [CANCELLED]}, paths={"prepare": BOOKED_PAGES})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert (t.phase, t.status) == ("cancel", "queued"), "its own pages show the booking, though its report left it out"
    assert t.result["booked"] and t.result["by_mistake"] and t.result["ride_id"] == "CRN11420826289"
    assert t.details["unauthorized_place"] and t.details["cancel_ask"]["state"] == "cancelling"
    said = h.told[-1]
    assert "MISTAKE" in said and "was booked before anyone said yes" in said and "Never say nothing was booked" in said
    assert "CRN11420826289" in said and "cancelling it on Ola right now" in said
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "cancelled" and agent.runs[-1]["phase"] == "cancel" and "Cancelled on Ola" in h.told[-1]
    assert sum(1 for r in agent.runs if r["phase"] == "place") == 0


async def test_a_run_on_the_page_with_the_final_button_is_stopped_checked_and_run_again_away_from_it(db, at, sessions):
    at("2026-10-12 21:52")
    agent = PathAgent(script={"prepare": [FARES]}, finish_after_polls=99, paths={"prepare": [OLA, OLA + "confirm-ride-p2p?x"]})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick(); await h.tick()  # started; on the next look it is on the "Confirm & Book" page
    await db.refresh(t)
    assert agent.stopped == ["t1"] and (t.phase, t.status) == ("check", "queued")
    agent.finish_after_polls = 1
    await h.tick()
    assert agent.runs[-1]["phase"] == "check" and "check only: do NOT book" in agent.runs[-1]["goal"]
    await h.tick()  # the list shows nothing of this ride: the fare look-up again, kept away from that page
    await db.refresh(t)
    assert (t.phase, t.status) == ("prepare", "queued") and not t.result.get("booked")
    agent.paths["prepare"] = [OLA, OLA + "?pickup_name=x"]
    await h.tick()
    goal = agent.runs[-1]["goal"]
    assert agent.runs[-1]["phase"] == "prepare" and "/confirm-ride-p2p, a page this step must never open" in goal
    assert "never press a book, confirm, request, place-order or pay button" in goal
    await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "choice", "the fares go to them to pick"
    assert not any("MISTAKE" in x for x in h.told)


async def test_a_run_on_a_page_after_booking_is_stopped_at_once_and_the_ride_cancelled(db, at, sessions):
    at("2026-10-12 21:52")
    agent = PathAgent(script={"prepare": [FARES], "cancel": [CANCELLED]}, finish_after_polls=99,
                      paths={"prepare": [OLA, OLA + "ride-detail/CRN99887766?x"]})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert agent.stopped == ["t1"] and t.result["ride_id"] == "CRN99887766" and t.phase == "cancel"
    assert "MISTAKE" in h.told[-1]


async def test_his_cancel_while_it_books_never_hears_nothing_was_booked(db, at, sessions):
    """He cancelled while the run was going (the auto was at his door): its pages already show the booking."""
    at("2026-10-12 21:53")
    agent = PathAgent(script={"prepare": [HIDDEN], "cancel": [CANCELLED]}, finish_after_polls=99, paths={"prepare": BOOKED_PAGES[:4]})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick()
    await db.refresh(t)
    said = await runtime.request_cancel(db, agent, t, by=ME, reason="said they cancelled it themselves in the app")
    assert said.startswith("It WAS booked by mistake") and "CRN11420826289" in said and "never say nothing was booked" in said
    assert t.phase == "cancel" and t.result["by_mistake"]


async def test_a_cancel_after_a_logged_in_run_reads_the_store_first_and_tells_the_truth(db, at, sessions):
    """Nothing on its pages yet: the store's own list is read; it shows the ride, which he cancelled in the app himself."""
    at("2026-10-12 21:53")
    shown = {**CHECKED, "this_task": True, "ride_id": "CRN11420826289", "status": "Cancelled by you", "cancelled": True}
    agent = PathAgent(script={"prepare": [FARES], "check": [shown]}, finish_after_polls=99, paths={"prepare": [OLA]})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick()
    await db.refresh(t)
    said = await runtime.request_cancel(db, agent, t, by=ME, reason="said they cancelled it themselves in the app")
    assert said.startswith("STOPPED; checking on Ola") and "never say nothing was booked" in said and t.phase == "check"
    await db.commit()
    agent.finish_after_polls = 1
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "cancelled" and t.result["by_mistake"] and t.result["ride_id"] == "CRN11420826289"
    assert "now shows it cancelled" in h.told[-1] and "Never say nothing was booked" in h.told[-1]
    assert not any(r["phase"] == "cancel" for r in agent.runs), "already cancelled: nothing more to do on the app"


async def test_a_list_that_cannot_be_read_is_never_nothing_was_booked(db, at, sessions):
    at("2026-10-12 21:53")
    agent = PathAgent(script={"prepare": [FARES], "check": [{**CHECKED, "this_task": None, "logged_in": False, "problem": "login page"}]},
                      finish_after_polls=99, paths={"prepare": [OLA]})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick()
    await db.refresh(t)
    await runtime.request_cancel(db, agent, t, by=ME, reason="not needed")
    await db.commit()
    agent.finish_after_polls = 1
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and "It is NOT known whether anything was booked" in h.told[-1]


async def test_a_cart_run_that_placed_on_any_store_is_owned_and_cancelled(db, at, sessions):
    """Every store, not only Ola: a Blinkit cart run that reached its payment-status page."""
    at("2026-10-12 21:55")
    agent = PathAgent(script={"prepare": [CART], "cancel": [CANCELLED]},
                      paths={"prepare": ["https://blinkit.com/s/", "https://blinkit.com/checkout", "https://blinkit.com/payment-status"]})
    h = H(sessions, agent)
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="blinkit", kind="order", goal="atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}], "for_set": True})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.phase == "cancel" and t.result["placed"] and t.result["by_mistake"] and "was placed before anyone said yes" in h.told[-1]


async def test_a_look_up_never_runs_where_the_family_is_logged_in(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 21:50")
    agent = PathAgent(script={"browse": [{**FARES, "logged_in": False, "login_required": True}], "prepare": [FARES]})
    h = H(sessions, agent)
    t = await _ride(db)
    t.agent_session = "fast-on-profile"  # a fast look-up's browser, on the family's profile (logged in)
    await db.commit()
    await h.tick()
    assert agent.runs[0]["phase"] == "browse" and agent.runs[0]["profile"] is None and agent.runs[0]["session"] is None
    assert "fast-on-profile" in agent.sessions_stopped
    await h.tick()
    await db.refresh(t)
    assert t.details["guest_session"] == t.agent_session
    await runtime.provide_input(db, t, kind="go", value="yes", by=ME, by_is_elder=True)
    await db.commit()
    await h.tick()
    assert agent.runs[-1]["phase"] == "prepare" and agent.runs[-1]["profile"] == "prof-fam-nb-ola" and agent.runs[-1]["session"] is None


async def test_a_cancel_of_a_look_up_without_login_is_stopped_at_once(db, at, sessions, monkeypatch):
    """A look-up has no login, so nothing can be booked there: "stopped" is true without a check."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-12 21:50")
    agent = PathAgent(script={"browse": [FARES]}, finish_after_polls=99)
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick()
    await db.refresh(t)
    said = await runtime.request_cancel(db, agent, t, by=ME, reason="not needed")
    assert said == "stopped; nothing was booked" and t.status == "cancelled" and "without the family's login" in t.history[-1]["note"]


@pytest.mark.parametrize("service,page,kind", [
    ("ola", "https://book.olacabs.com/confirm-ride-p2p?x=1", "final"),
    ("ola", "https://book.olacabs.com/ride-detail/CRN11420826289", "after"),
    ("ola", "https://book.olacabs.com/cancel-ride", "after"),
    ("uber", "https://m.uber.com/go/trip/abc123", "after"),
    ("blinkit", "https://blinkit.com/payment-status", "after"),
    ("blinkit", "https://blinkit.com/account/orders/track/123456/7", "after"),
    ("1mg", "https://www.1mg.com/cart/payment", "final"),
    ("1mg", "https://www.1mg.com/cart/checkout/success", "after"),
    ("instamart", "https://www.swiggy.com/my-account/order-details/12345678", "after"),
    ("zepto", "https://www.zepto.com/order-success/abc123", "suspect"),
    ("apollo", "https://www.apollopharmacy.in/thank-you?order=1", "suspect"),
    ("rapido", "https://m.rapido.bike/track/xyz", "suspect"),
])
def test_every_store_s_final_and_after_pages(service, page, kind):
    assert guard.past_confirm(service, ["https://x.test/", page])[0] == kind


@pytest.mark.parametrize("service,page", [
    ("blinkit", "https://blinkit.com/prn/fitness-tracker-band/prid/123456"),
    ("blinkit", "https://blinkit.com/checkout"),
    ("zomato", "https://www.zomato.com/hi/raipur/some-restaurant/order"),
    ("instamart", "https://www.swiggy.com/instamart/cart"),
    ("ola", "https://book.olacabs.com/?pickup_name=track%20road"),
    ("1mg", "https://www.1mg.com/cart/summary-page"),
])
def test_pages_a_cart_or_fare_run_needs_are_not_stopped(service, page):
    assert guard.past_confirm(service, [page]) is None


async def test_after_every_run_with_the_login_the_store_is_read_before_the_options_go_out(db, at, sessions, monkeypatch):
    """B, founder 2026-10-12: "after every run that must not place, check the store's live state"."""
    monkeypatch.setenv("STORE_CHECK_AFTER_RUNS", "on")
    at("2026-10-12 21:52")
    agent = PathAgent(script={"prepare": [FARES]}, paths={"prepare": [OLA, OLA + "?pickup_name=x"]})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert (t.phase, t.status) == ("check", "queued") and not h.told, "nothing goes out before the store's list is read"
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert [r["phase"] for r in agent.runs] == ["prepare", "check"]
    assert t.status == "awaiting_confirm" and t.input_needed == "choice" and t.result["options"] == FARES["options"]
    assert "this_task" not in t.result and any("checked on Ola: nothing was booked" in x["note"] for x in t.history)
    assert len(h.told) == 1 and "MISTAKE" not in h.told[0]


async def test_the_check_after_a_run_finds_what_it_booked_and_it_is_cancelled(db, at, sessions, monkeypatch):
    monkeypatch.setenv("STORE_CHECK_AFTER_RUNS", "on")
    at("2026-10-12 21:52")
    found = {**CHECKED, "this_task": True, "ride_id": "CRN555666", "status": "Driver on the way", "cancelled": False}
    agent = PathAgent(script={"prepare": [HIDDEN], "check": [found], "cancel": [CANCELLED]}, paths={"prepare": [OLA]})
    h = H(sessions, agent)
    t = await _ride(db)
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert t.phase == "cancel" and t.result["ride_id"] == "CRN555666" and "MISTAKE" in h.told[-1]
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "cancelled" and "Cancelled on Ola" in h.told[-1]


async def test_a_run_that_asked_for_a_login_code_had_no_login_and_is_not_checked(db, at, sessions, monkeypatch):
    from app.sim.agent import OTP

    monkeypatch.setenv("STORE_CHECK_AFTER_RUNS", "on")
    at("2026-10-12 21:52")
    agent = PathAgent(script={"prepare": [OTP]}, paths={"prepare": [OLA + "login"]})
    h = H(sessions, agent)
    t = await _ride(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "needs_input" and t.input_needed == "otp" and [r["phase"] for r in agent.runs] == ["prepare"]
