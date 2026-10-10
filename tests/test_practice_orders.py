"""Practice orders (founder 2026-10-11): an order goes through every step up to the store's final place-order button and stops
there; nothing is placed, the cart is emptied, and the person hears it was a practice. A practice that reports a placed
order raises the alarm."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care.models import OpenLoop
from app.sim.agent import CART, PLACED, FakeAgent
from app.tasks import fastpath, practice, runtime
from tests.test_tasks import LinkedHost

FAM, ELDER = "fam-pr", "elder-pr"


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


@pytest.fixture
def practice_on(monkeypatch):
    monkeypatch.setenv("PRACTICE_ORDER_FAMILIES", f"other-fam, {FAM}")


class Harness:
    def __init__(self, sessions, agent, host=None):
        self.sessions, self.agent, self.told, self.host = sessions, agent, [], host

    async def profile_for(self, task):
        return {"profileId": f"prof-{task.service}", "loginPhone": "9000012345"}

    async def notify(self, family_id, requested_by, prompt):
        self.told.append(prompt)

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify,
                                  host_for=(lambda f: self.host) if self.host else None)


async def _order(db, service="swiggy"):
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=service, kind="order",
                             goal="Paneer butter masala", details={"items": [{"name": "Paneer Butter Masala", "qty": 1}]})
    await db.commit()
    return t


def test_only_listed_families_practise(monkeypatch):
    monkeypatch.setenv("PRACTICE_ORDER_FAMILIES", "fam-a,fam-b")
    assert practice.family_listed("fam-b") and practice.family_listed("shadow:fam-a")
    assert not practice.family_listed("fam-c")
    monkeypatch.setenv("PRACTICE_ORDER_FAMILIES", "")
    assert not practice.family_listed("fam-a")


async def test_browser_practice_goes_to_the_last_button_and_never_presses_it(db, at, sessions, practice_on):
    at("2026-10-11 01:00")
    stop = {**CART, "placed": False, "reached_final_step": True, "final_button": "Place Order ₹318", "cart_emptied": True,
            "payment_method": "Pay on Delivery"}
    h = Harness(sessions, FakeAgent(script={"prepare": [CART], "place": [stop]}))
    t = await _order(db)
    assert t.details["practice"] is True, "marked when created"
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm"
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    await db.commit()
    await h.tick()
    run = h.agent.runs[-1]
    assert run["phase"] == "place" and "THIS IS A PRACTICE RUN" in run["goal"] and "do NOT click it" in run["goal"]
    assert "remove every item" in run["goal"] and "Never report placed=true" in run["goal"]
    assert "Place the order now" not in run["goal"]
    await h.tick()
    await db.refresh(t)
    assert t.status == "cancelled" and not t.result.get("placed")
    assert "Practice order" in h.told[-1] and "'Place Order ₹318'" in h.told[-1] and "It was NOT placed" in h.told[-1]
    loops = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "delivery"))).scalars().all()
    assert not loops, "no 'has it come?' for a practice"


async def test_a_practice_that_reports_placed_raises_the_alarm(db, at, sessions, practice_on):
    at("2026-10-11 01:00")
    h = Harness(sessions, FakeAgent(script={"prepare": [CART], "place": [PLACED]}))
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and t.details["practice_accident"] is True
    assert "may have placed a real order (order id IM-55821)" in h.told[-1] and "cancel it there" in h.told[-1]


async def test_a_task_marked_practice_stays_practice_when_the_setting_goes(db, at, sessions, monkeypatch):
    monkeypatch.setenv("PRACTICE_ORDER_FAMILIES", FAM)
    at("2026-10-11 01:00")
    stop = {**CART, "placed": False, "reached_final_step": True, "final_button": "Place Order"}
    h = Harness(sessions, FakeAgent(script={"prepare": [CART], "place": [stop]}))
    t = await _order(db)
    monkeypatch.setenv("PRACTICE_ORDER_FAMILIES", "")
    await h.tick(); await h.tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    await db.commit()
    await h.tick()
    assert "THIS IS A PRACTICE RUN" in h.agent.runs[-1]["goal"]


class PracticeLinked(LinkedHost):
    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "connector_status":
            self.calls.append((tool, args))
            return {"connected": True, "enabled": True}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


async def test_connector_practice_rebuilds_the_cart_and_never_calls_place(db, at, sessions, practice_on, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")  # looked up through the connector first, as in production
    at("2026-10-11 01:00")
    host = PracticeLinked()
    h = Harness(sessions, FakeAgent(script={}), host)
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order",
                             goal="Diet Coke", details={"items": [{"name": "Diet Coke", "qty": 1}]})
    await db.commit()
    for _ in range(3):
        await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm", t.history
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    await db.commit()
    await h.tick()
    await db.refresh(t)
    tools_called = [c[0] for c in host.calls]
    assert "connector_place" not in tools_called and tools_called.count("connector_prepare") == 2, tools_called
    assert t.status == "cancelled" and "Practice order" in h.told[-1] and "₹49" in h.told[-1]


async def test_blinkit_practice_stops_before_pay_now(db, at, sessions, practice_on, monkeypatch):
    at("2026-10-11 01:00")
    seen = {}

    async def dry_place(cdp, check, *, dry=False):
        seen["dry"] = dry
        return {"clicked": False, "ready": True}

    async def real_place(*a, **kw):
        raise AssertionError("never placed")

    monkeypatch.setattr(fastpath, "blinkit_place", dry_place)
    monkeypatch.setattr(fastpath, "place", real_place)
    t = await _order(db, service="blinkit")
    t.phase, t.agent_session = "place", "s1"
    t.details = {**t.details, "fast_place_check": {"count": 1, "prices": [60], "total": 75, "address": "12 Civil"}}
    await db.commit()

    class A:
        async def cdp_url(self, sid):
            return "ws://x"

    out = await runtime._fast_place(db, A(), t, None, None)
    assert seen == {"dry": True} and out["practice"] and out["reached_final_step"] and out["final_button"] == "Pay Now"


async def test_a_food_look_up_with_every_place_closed_says_so_with_opening_times(db, at):
    at("2026-10-11 01:00")
    t = await _order(db)
    t.phase = "browse"
    out = {"items": [{"name": "Paneer Butter Masala", "price": 240, "available": False, "restaurant": "Haldiram's", "closed": True,
                      "opens": "Opens next at 9 am, tomorrow"},
                     {"name": "Paneer Butter Masala Thali", "price": 199, "available": False, "restaurant": "Sagar Ratna", "closed": True}],
           "deliverable": True}
    status, message = runtime._browse_outcome(t, out)
    assert status == "failed" and "closed now (Haldiram's (Opens next at 9 am, tomorrow); Sagar Ratna)" in message


def test_the_food_picker_sees_the_restaurant_and_that_it_is_closed():
    from app.tasks import matcher

    line = matcher._listing(1, {"name": "Masala Dosa", "price": "₹120", "restaurant": "Sagar Ratna", "closed": True, "opens": "Opens at 7 am"})
    assert line == "1. Masala Dosa | ₹120 | from Sagar Ratna | [restaurant closed now, Opens at 7 am]"
