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
    assert "needs approval first" in msg and t.status == "awaiting_confirm" and t.input_needed == "approve"
    assert "approved. confirmed; placing it now" in await runtime.provide_input(db, t, kind="confirm", value="yes", by=SON, by_is_elder=False)


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


async def _compare(db, item="Diet Coke") -> list[Task]:
    out = []
    for s in runtime.COMPARE_STORES["grocery"]:
        out.append(await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=s, kind="order", goal=f"{item} for Amma",
                                        details={"items": [{"name": item, "qty": 1}], "compare": "g1"}))
    await db.commit()
    return out


COKE_BLINKIT = {**FOUND, "eta": "11 mins", "items": [
    {"name": "Diet Coke Can (300 ml)", "price": "₹40", "available": True, "for_item": "Diet Coke"},
    {"name": "Diet Coke (6 x 300 ml)", "price": "₹220", "available": True, "for_item": "Diet Coke"}]}
COKE_INSTAMART = {**FOUND, "eta": "15 mins", "items": [{"name": "Coca-Cola Diet Coke Can 300 ml", "price": "₹42", "available": True, "for_item": "Diet Coke"}]}
NO_ZEPTO = {**FOUND, "items": [], "deliverable": False, "location_set": "Raipur 492001"}


async def test_no_store_named_looks_on_every_store_and_offers_all_options(db, at, sessions, monkeypatch):
    """Founder 2026-10-09: 'how is it choosing Blinkit? it should go for other options like Instamart and Zepto, and give
    the available options as we had earlier'. One look-up per store at once, one update with every option."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")  # a comparison looks first even when single orders do not
    at("2026-10-09 15:03")
    h = LoginHarness(sessions, FakeAgent(script={"browse:blinkit": [COKE_BLINKIT], "browse:instamart": [COKE_INSTAMART], "browse:zepto": [NO_ZEPTO],
                                                 "prepare": [OTP, CART]}, finish_after_polls=1), "9000012345")
    b, i, z = await _compare(db)
    await h.tick()  # all three look at once, none logs in
    assert sorted(r["service"] for r in h.agent.runs) == ["blinkit", "instamart", "zepto"]
    assert all(r["phase"] == "browse" and "9000012345" not in r["goal"] and "up to 3 matching products" in r["goal"] for r in h.agent.runs)
    await h.tick()
    assert len(h.told) == 1, "one update for the whole comparison, not one per store"
    msg = h.told[0]
    assert msg.startswith("[Task update] Looked on Blinkit, Swiggy Instamart, Zepto without logging in")
    assert "Diet Coke Can (300 ml) ₹40" in msg and "Diet Coke (6 x 300 ml) ₹220" in msg and "₹42" in msg
    assert "Zepto: Zepto does not deliver" in msg and f"[task {b.id}]" in msg and f"[task {i.id}]" in msg and f"[task {z.id}]" not in msg
    assert "numbered list" in msg and "phone ending 2345" in msg
    for t in (b, i, z):
        await db.refresh(t)
    assert (b.status, b.input_needed, i.status, z.status) == ("awaiting_confirm", "go", "awaiting_confirm", "failed")
    # she picks the single can on Blinkit: that is the go-ahead, and Instamart is dropped
    said = await runtime.provide_input(db, b, kind="go", value="Diet Coke Can (300 ml)", by=ELDER, by_is_elder=True)
    assert said.startswith("going ahead on Blinkit with Diet Coke Can (300 ml) ₹40; the other stores are dropped")
    await db.commit()
    await db.refresh(i)
    assert i.status == "cancelled"
    await h.tick()
    prep = h.agent.runs[-1]
    assert prep["service"] == "blinkit" and prep["phase"] == "prepare" and "enter the mobile number 9000012345" in prep["goal"]
    assert "Diet Coke Can (300 ml) ₹40" in prep["goal"] and "6 x 300" not in prep["goal"]


async def test_comparison_waits_a_little_for_a_slow_store_then_sends_what_it_has(db, at, sessions, monkeypatch):
    at("2026-10-09 15:03")
    agent = FakeAgent(script={"browse:blinkit": [COKE_BLINKIT], "browse:instamart": [COKE_INSTAMART], "browse:zepto": [NO_ZEPTO]})
    h = LoginHarness(sessions, agent, "9000012345")
    b, i, z = await _compare(db)
    await h.tick()
    agent.finish_after_polls = 99  # Instamart and Zepto keep loading
    agent._polls[[r["task"] for r in agent.runs if r["service"] == "blinkit"][0]] = 98
    await h.tick()
    assert h.told == [], "the first answer waits for the others"
    at("2026-10-09 15:08")
    await h.tick()
    assert len(h.told) == 1 and "₹40" in h.told[0] and "Instamart: still looking" in h.told[0]
    # Instamart answers later: told on its own as another option
    agent.finish_after_polls = 1
    await h.tick()
    late = [m for m in h.told[1:] if "Instamart" in m]
    assert late and "Another store answered after the options were sent" in late[0] and "₹42" in late[0]
    assert not any("Zepto" in m for m in h.told[1:]), "a late 'does not deliver' is not worth a message"


async def test_comparison_declined_drops_every_store_and_expires_once(db, at, sessions):
    at("2026-10-09 15:03")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [COKE_INSTAMART]}), "9000012345")
    b, i, z = await _compare(db)
    await h.tick(); await h.tick()
    for t in (b, i, z):
        await db.refresh(t)
    assert await runtime.provide_input(db, i, kind="go", value="nahi", by=ELDER, by_is_elder=True) == "declined on every store; nothing was ordered"
    await db.commit()
    for t in (b, z):
        await db.refresh(t)
    assert b.status == z.status == "cancelled"
    # a fresh comparison nobody answers: one "nobody picked" message, not three
    b2, i2, z2 = await _compare(db, item="Atta")
    for t in (b2, i2, z2):
        t.details = {**t.details, "compare": "g2"}
    await db.commit()
    await h.tick(); await h.tick()
    n = len(h.told)
    at("2026-10-09 16:00")
    await h.tick()
    assert len(h.told) == n + 1 and "Nobody picked one of the options" in h.told[-1]


async def test_go_ahead_with_several_products_found_uses_the_best_match_or_the_named_one(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 15:03")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [COKE_BLINKIT], "prepare": [OTP]}), "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert "ask which one" in h.told[-1]
    assert "more than one product" in await runtime.provide_input(db, t, kind="go", value="diet coke", by=ELDER, by_is_elder=True)
    said = await runtime.provide_input(db, t, kind="go", value="6 x 300 ml pack", by=ELDER, by_is_elder=True)
    assert "Diet Coke (6 x 300 ml) ₹220" in said
    await db.commit()
    await h.tick()
    assert "Diet Coke (6 x 300 ml) ₹220" in h.agent.runs[-1]["goal"] and "Can (300 ml)" not in h.agent.runs[-1]["goal"]


async def test_start_task_without_a_store_compares_the_usual_ones(db, at):
    from app.brain import tools
    from app.sim.world import SimHost

    at("2026-10-09 15:03")
    e = {"id": ELDER, "name": "Vasundara", "role": "elder"}
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=e, speaker=e, members=[e])
    with pytest.raises(tools.ToolRefused):
        await tools.start_task(ctx, {"kind": "ride", "goal": "Cab to the clinic"})
    old = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Atta",
                               details={"items": [{"name": "Atta", "qty": 1}]})
    ask = {"kind": "order", "category": "grocery", "goal": "Diet Coke for Vasundara ji", "items": [{"name": "Diet Coke"}]}
    ctx.message_ref = "wamid.1"
    first = await tools.start_task(ctx, ask)
    assert "confirm_address" in first and first["status"] == "not started yet", "the address is confirmed before any search"
    assert (await tools.start_task(ctx, ask))["status"].startswith("waiting"), "not in the same turn"
    assert len([t for t in await runtime.live_tasks(db, FAM) if runtime.compare_group(t)]) == 0
    ctx.message_ref = "wamid.2"  # her "haan" in the next message
    out = await tools.start_task(ctx, ask)
    assert out["looking_on"] == ["Swiggy Instamart", "Zepto"] and len(out["task_ids"]) == 2 and str(old.id) in out["already_running"][0]
    group = [t for t in await runtime.live_tasks(db, FAM) if runtime.compare_group(t)]
    assert len({runtime.compare_group(t) for t in group}) == 1 and {t.service for t in group} == {"instamart", "zepto"}


class ClosingAgent(FakeAgent):
    """Browser Use refuses a step on a browser that already stopped."""

    async def run(self, *, session_id, **kw):
        if session_id and session_id in self.sessions_stopped:
            raise RuntimeError('browser-use POST tasks 400: {"detail":"Browser session is stopped. Please start a new session and try again."}')
        return await super().run(session_id=session_id, **kw)


async def test_go_ahead_after_the_look_up_browser_closed_starts_a_fresh_one(db, at, sessions, monkeypatch):
    """Live 2026-10-09 15:48: Maa said ok, the login step tried the look-up's closed browser and failed every minute."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 15:45")
    agent = ClosingAgent(script={"browse": [FOUND], "prepare": [OTP]})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    agent.sessions_stopped.append(t.agent_session)  # Browser Use closed it on its own
    await runtime.provide_input(db, t, kind="go", value="ok", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    await db.refresh(t)
    assert t.status == "running" and agent.runs[-1]["phase"] == "prepare" and agent.runs[-1]["session"] is None
    assert agent.runs[-1]["profile"] == "prof-fam-t-instamart" and "fresh one" in t.history[-2]["note"]


def test_pick_among_packs_with_the_same_name():
    """Live 2026-10-09 15:59: two 'Pepsi Zero Sugar Soft Drink' packs (₹20, ₹40); 'the ₹20 one' was asked again and again."""
    found = [{"name": "Pepsi Zero Sugar Soft Drink", "price": "₹40", "pack": "300 ml"},
             {"name": "Pepsi Zero Sugar Soft Drink", "price": "₹20", "pack": "160 ml"}]
    assert runtime._match_product(found, "Pepsi Zero Sugar Soft Drink") == "ambiguous"
    assert runtime._match_product(found, "Pepsi Zero Sugar Soft Drink ₹20")["price"] == "₹20"
    assert runtime._match_product(found, "Pepsi Zero Sugar Soft Drink (160 ml) ₹20")["pack"] == "160 ml"
    assert runtime._match_product(found, "Pepsi Zero Sugar Soft Drink (300 ml)")["price"] == "₹40"
    assert runtime._option({"name": "Coke Zero", "price": "₹39", "exact_match": False}).endswith("[similar, not exactly what was asked]")


async def test_code_for_a_closed_login_page_asks_for_a_new_one_and_is_not_kept(db, at, sessions):
    """Live 2026-10-09 16:08: the login page closed before the code came; the fresh browser typed the code as the phone
    number. Now it logs in again, the person hears why a new code came, and no code is kept on the task."""
    at("2026-10-09 16:03")
    agent = ClosingAgent(script={"prepare": [OTP, OTP]})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.input_needed == "otp"
    agent.sessions_stopped.append(t.agent_session)
    await runtime.provide_input(db, t, kind="otp", value="998877", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    await db.refresh(t)
    goal = agent.runs[-1]["goal"]
    assert "998877" not in goal and "enter the mobile number 9000012345" in goal and agent.runs[-1]["session"] is None
    assert "998877" not in str(t.details)
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "otp" and "closed before the code could be used" in h.told[-1] and "new code" in h.told[-1]


class LinkedHost:
    """The backend for a family that linked Instamart (connector) but not Blinkit."""

    def __init__(self):
        self.calls = []

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        self.calls.append((tool, args))
        if tool == "connector_status":
            ok = args["store"] == "instamart"
            return {"connected": ok, "enabled": ok}
        if tool == "connector_search":
            return {"ok": True, "items": [{"name": "Coca-Cola Diet Coke Can 300 ml", "price": "₹40", "ref": {"store": "instamart", "name": "Coca-Cola Diet Coke Can 300 ml", "spinId": "S1"}},
                                          {"name": "Coca-Cola Zero Sugar 300 ml", "price": "₹40", "ref": {"store": "instamart", "name": "Coca-Cola Zero Sugar 300 ml", "spinId": "S2"}}]}
        if tool == "connector_prepare":
            return {"ok": True, "items": [{"name": args["pick"]["name"], "qty": 1, "price": "₹40"}], "total": "₹49", "fees": "₹9",
                    "cod_available": True, "address_used": "Home", "card": {"cardId": "c1", "totalPaise": 4900}}
        if tool == "connector_place":
            return {"status": "placed", "orderId": "IM-1", "total": "₹49"}
        return {}


async def test_linked_store_looks_up_and_orders_through_its_connector_without_a_code(db, at, sessions):
    """Founder 2026-10-09 'make it fast': Instamart is linked for Maa's family, so its look-up, cart and order go through
    the connector (seconds, no browser, no login code); Blinkit still uses the browser."""
    at("2026-10-09 17:00")
    host = LinkedHost()
    agent = FakeAgent(script={"browse:blinkit": [COKE_BLINKIT]})
    told = []
    for s in ("blinkit", "instamart"):
        await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=s, kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}], "compare": "g9"})
    await db.commit()

    async def notify(f, r, p):
        told.append(p)

    async def tick():
        return await runtime.tick(sessions, agent, profile_for=lambda t: _profile(t), notify=notify, host_for=lambda f: host)

    async def _profile(t):
        return {"profileId": f"prof-{t.service}", "loginPhone": "9000012345"}

    await tick(); await tick()
    assert [r["service"] for r in agent.runs] == ["blinkit"], "Instamart was looked up through its connector, not a browser"
    assert len(told) == 1 and "Swiggy Instamart [task" in told[0] and "(linked account: no login code)" in told[0]
    assert "Coca-Cola Zero Sugar 300 ml ₹40 [similar" in told[0] and "Coca-Cola Diet Coke Can 300 ml ₹40" in told[0]
    im = next(t for t in await runtime.live_tasks(db, FAM) if t.service == "instamart")
    said = await runtime.provide_input(db, im, kind="go", value="Coca-Cola Diet Coke Can 300 ml ₹40", by=ELDER, by_is_elder=True)
    assert "linked account (no login code" in said
    await db.commit()
    await tick()
    await db.refresh(im)
    assert im.status == "awaiting_confirm" and im.input_needed == "confirm" and im.result["total"] == "₹49"
    assert ("connector_prepare", ) == tuple(c[0] for c in host.calls if c[0] == "connector_prepare")[:1]
    assert next(c[1] for c in host.calls if c[0] == "connector_prepare")["pick"]["spinId"] == "S1"
    await runtime.provide_input(db, im, kind="confirm", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await tick()
    await db.refresh(im)
    assert im.status == "done" and im.result["order_id"] == "IM-1" and [r["service"] for r in agent.runs] == ["blinkit"]


class ParleHost(LinkedHost):
    """Instamart linked; the search lists several Parle-G packs, the first is not the one asked for."""

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "connector_search":
            self.calls.append((tool, args))
            names = ["Parle Parle G Gold Biscuits Pouch — 1 kg x 2", "Parle Parle G Gold Biscuits Pouch — 1 kg", "Parle Parle-G Gold Biscuits — 475 g"]
            return {"ok": True, "items": [{"name": n, "price": "₹75", "ref": {"store": "instamart", "name": n, "spinId": f"P{i}"}} for i, n in enumerate(names)]}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


async def _connector_order(db, sessions, host, asked):
    told = []
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal=asked,
                             details={"items": [{"name": asked, "qty": 1}]})
    await db.commit()

    async def notify(f, r, p):
        told.append(p)

    async def no_profile(t):
        return None

    for _ in range(3):
        await runtime.tick(sessions, FakeAgent(), profile_for=no_profile, notify=notify, host_for=lambda f: host)
    await db.refresh(t)
    return t, told


async def test_connector_builds_the_cart_for_the_pack_asked_not_the_first_hit(db, at, sessions, monkeypatch):
    """Order lab 2026-10-10: "Parle-G biscuit 475 g" on the linked Instamart got the first hit, "1 kg x 2" ₹250."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 14:30")
    host = ParleHost()
    t, _ = await _connector_order(db, sessions, host, "Parle-G biscuit 475 g")
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm"
    assert next(c[1] for c in host.calls if c[0] == "connector_prepare")["pick"]["spinId"] == "P2"


async def test_connector_asks_which_one_when_several_packs_fit(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 14:30")
    host = ParleHost()
    t, told = await _connector_order(db, sessions, host, "Parle-G biscuit")
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and t.phase == "browse"
    assert not any(c[0] == "connector_prepare" for c in host.calls)
    assert "no login code" in told[-1] and "475 g" in told[-1] and "1 kg x 2" in told[-1]
    said = await runtime.provide_input(db, t, kind="go", value="Parle Parle G Gold Biscuits Pouch — 1 kg", by=ELDER, by_is_elder=True)
    assert "linked account" in said


def test_asked_words_match_plurals_and_pack_numbers():
    assert runtime.asked_match("Parle-G biscuit 475 g", "Parle Parle-G Gold Biscuits — 475 g")
    assert not runtime.asked_match("Parle-G biscuit 475 g", "Parle Parle G Gold Biscuits Pouch — 1 kg x 2")
    assert not runtime.asked_match("Diet Coke", "Coca-Cola Zero Sugar 300 ml")
    assert runtime.asked_match("", "anything") is None
    assert runtime.asked_match("RiteBite Max Protein wafer bar 240 g", "Rite Bite Max Protein Assorted 10g Protein Millet Wafer Bar — 240 g")


def test_single_pack_ask_prefers_the_single_pack_over_a_multipack():
    box, two = "Rite Bite Max Protein Wafer Bar — 240 g", "Rite Bite Max Protein Wafer Bar — 240 g x 2"
    found = [{"name": two, "exact_match": True}, {"name": box, "exact_match": True}]
    assert runtime.best_exact("RiteBite wafer bar 240 g", found)["name"] == box
    assert runtime.best_exact("RiteBite wafer bar 240 g x 2", found) is None  # both fit a multipack ask: they pick
    assert runtime.best_exact("anything", [{"name": box}])["name"] == box


class SnackHost(LinkedHost):
    """Instamart linked; biscuits and munchies looked up separately, both go in one cart."""

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "connector_search":
            self.calls.append((tool, args))
            if "biscuit" in args["item"].lower():
                names = ["Parle-G Gold Biscuits — 475 g", "Britannia Good Day Cashew Biscuits — 200 g"]
            else:
                names = ["Kurkure Masala Munch — 90 g", "Lay's India's Magic Masala Chips — 50 g"]
            return {"ok": True, "items": [{"name": n, "price": "₹40", "ref": {"store": "instamart", "name": n, "spinId": n[:6]}} for n in names]}
        if tool == "connector_prepare":
            self.calls.append((tool, args))
            lines = [{"name": args["pick"]["name"], "qty": args["qty"]}] + [{"name": m["pick"]["name"], "qty": m["qty"]} for m in args.get("more", [])]
            return {"ok": True, "items": lines, "total": "₹150", "fees": "₹30", "cod_available": True, "address_used": "Home",
                    "card": {"cardId": "c2", "totalPaise": 15000}}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


async def test_connector_takes_several_items_in_one_cart(db, at, sessions, monkeypatch):
    """Order lab 2026-10-10: "add biscuits and munchies" — two items, one Instamart cart, each picked from its own look-up."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 15:30")
    host = SnackHost()
    told = []
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order",
                             goal="biscuits and munchies", details={"items": [{"name": "biscuits", "qty": 1}, {"name": "munchies", "qty": 2}]})
    await db.commit()

    async def notify(f, r, p):
        told.append(p)

    async def no_profile(t):
        return None

    async def tick():
        await runtime.tick(sessions, FakeAgent(), profile_for=no_profile, notify=notify, host_for=lambda f: host)

    await tick()
    await db.refresh(t)
    assert [c[1]["item"] for c in host.calls if c[0] == "connector_search"] == ["biscuits", "munchies"]
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and "For biscuits:" in told[-1] and "For munchies:" in told[-1]
    said = await runtime.provide_input(db, t, kind="go", value="Parle-G Gold Biscuits — 475 g | Kurkure Masala Munch — 90 g",
                                       by=ELDER, by_is_elder=True)
    assert "linked account" in said
    await db.commit()
    await tick()
    await db.refresh(t)
    prep = next(c[1] for c in host.calls if c[0] == "connector_prepare")
    assert prep["pick"]["name"].startswith("Parle-G") and prep["qty"] == 1
    assert prep["more"] == [{"pick": {"store": "instamart", "name": "Kurkure Masala Munch — 90 g", "spinId": "Kurkur"}, "qty": 2}]
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm"
    assert "not in the cart" not in told[-1]  # the cart is checked against the picked products, not the words asked


async def test_a_go_naming_a_product_not_found_is_asked_again(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 15:30")
    host = SnackHost()
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order",
                             goal="biscuits and munchies", details={"items": [{"name": "biscuits", "qty": 1}, {"name": "munchies", "qty": 1}]})
    await db.commit()

    async def no_profile(t):
        return None

    async def notify(f, r, p):
        return None

    await runtime.tick(sessions, FakeAgent(), profile_for=no_profile, notify=notify, host_for=lambda f: host)
    await db.refresh(t)
    said = await runtime.provide_input(db, t, kind="go", value="Parle-G Gold Biscuits — 475 g | Haldiram bhujia", by=ELDER, by_is_elder=True)
    assert "not one of the products found" in said and t.phase == "browse"


def test_a_product_name_with_bars_in_it_is_still_picked_whole():
    potazos = "Britannia 5050 Potazos Masti Masala Spicy Flavoured Crisps | 71.5 g | Potato and Biscuit Ka Tasty Combination — 71.5 g"
    groups = [("biscuits", [{"name": "Sunfeast Bourbon Dark Fantasy Biscuit — 99 g", "price": "₹14"}, {"name": "Let's Try Atta Jeera Cookies — 200 g", "price": "₹92"}]),
              ("munchies", [{"name": potazos + " x 3", "price": "₹84"}, {"name": potazos + " x 2", "price": "₹56"}, {"name": potazos, "price": "₹28"}])]
    got = runtime._named_in(f"Sunfeast Bourbon Dark Fantasy Biscuit — 99 g ₹14 | {potazos} ₹28", groups)
    assert [g["price"] for g in got] == ["₹14", "₹28"]
    got = runtime._named_in(f"Sunfeast Bourbon Dark Fantasy Biscuit — 99 g | {potazos} x 2 ₹56", groups)
    assert [g["price"] for g in got] == ["₹14", "₹56"]
    assert runtime._named_in("Dark Fantasy | Potazos", groups) is None  # short names fall back to the per-part match


def test_food_orders_stay_one_dish_on_the_connector():
    from app.specialists import channels

    assert "instamart" in channels.MULTI_ITEM and "swiggy" not in channels.MULTI_ITEM


class FastAgent(FakeAgent):
    async def open_session(self, profile_id):
        self.opened = profile_id
        return "fs-1"

    async def cdp_url(self, session_id):
        return "https://fs-1.cdp.test"


async def test_blinkit_looks_up_through_its_own_web_request_in_seconds_then_the_agent_logs_in_there(db, at, sessions, monkeypatch):
    """Founder 2026-10-09 'make it fast': the agent's Blinkit look-up took 3 minutes; the fixed search runs Blinkit's own
    request in the task's cloud browser. The same browser then serves the login and cart."""
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    asked = {}

    async def fake_search(service, cdp, query, lat=None, lon=None, pincode=None):
        asked.update(cdp=cdp, query=query, lat=lat, lon=lon)
        return {"deliverable": True, "eta": None, "items": [
            {"name": "Coca-Cola Diet Coke Soft Drink No Caffeine", "pack": "330 ml", "price": "₹209", "available": True, "exact_match": True}]}

    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 17:30")
    agent = FastAgent(script={"prepare": [OTP]})
    h = LoginHarness(sessions, agent, "9000012345")
    place = {"addressId": "a1", "nickname": "Home", "pincode": "492001", "full": "Sunita Park, Raipur", "lat": 21.238, "lng": 81.6858}
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]}, limits=Limits(place=place))
    await db.commit()
    await h.tick()
    await db.refresh(t)
    assert agent.runs == [] and asked == {"cdp": "https://fs-1.cdp.test", "query": "Diet Coke", "lat": 21.238, "lon": 81.6858}
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and t.agent_session == "fs-1"
    assert "Coca-Cola Diet Coke Soft Drink No Caffeine (330 ml) ₹209" in h.told[-1]
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    run = agent.runs[-1]
    assert run["phase"] == "prepare" and run["session"] == "fs-1" and "enter the mobile number 9000012345" in run["goal"]


async def test_fast_look_up_that_breaks_falls_back_to_the_agent(db, at, sessions, monkeypatch):
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    async def broken(*a, **k):
        raise fastpath.FastPathError("blinkit search answered 403")

    monkeypatch.setattr(fastpath, "search", broken)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 17:30")
    agent = FastAgent(script={"browse": [FOUND]})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]}, limits=Limits(place={"lat": 21.2, "lng": 81.6}))
    await db.commit()
    await h.tick()
    await db.refresh(t)
    assert [r["phase"] for r in agent.runs] == ["browse"] and any("fast look-up did not work" in x["note"] for x in t.history)


def test_fast_search_exact_match_counts_the_pack():
    from app.tasks.fastpath import _words
    asked = _words("Amul Taaza milk 500 ml")
    assert asked == {"amul", "taaza", "milk", "500"}
    assert asked <= (_words("Amul Taaza Toned Milk 500 ml") | _words("500"))


def test_fast_snippets_fill_placeholders_safely_and_results_are_cleaned():
    from app.tasks import fastpath
    js = fastpath._snippet("blinkit", {"q": 'diet "coke" 50%', "lat": 21.2, "lon": 81.6})
    assert 'encodeURIComponent("diet \\"coke\\" 50%")' in js and "%(q)s" not in js
    assert fastpath._rupees(40) == "₹40" and fastpath._rupees("24.86") == "₹24.86" and fastpath._rupees("₹209") == "₹209"


async def test_logged_in_blinkit_builds_the_cart_through_its_own_request(db, at, sessions, monkeypatch):
    """Founder 2026-10-09 trial: with the family's Blinkit profile already logged in, the cart is built through Blinkit's
    own cart request (11 s live) instead of the agent (3 min); not logged in → the agent logs in as before."""
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    async def fake_search(service, cdp, query, lat=None, lon=None, pincode=None):
        return {"deliverable": True, "eta": None, "items": [
            {"name": "Coca-Cola Diet Coke Soft Drink No Caffeine", "pack": "330 ml", "price": "₹209", "available": True, "exact_match": True, "store_id": "746124"}]}

    built = {}

    async def fake_cart(cdp, products, lat=None, lon=None):
        built["products"] = products
        return {"items": [{"name": "Coca-Cola Diet Coke Soft Drink No Caffeine", "qty": 1, "price": "₹209", "available": True}],
                "total": "₹218", "fees": "₹9", "cod_available": True, "logged_in": True, "address_used": "Home: 504, Sunita Park, Raipur 492001"}

    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setattr(fastpath, "blinkit_cart", fake_cart)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 18:00")
    agent = FastAgent(script={})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]}, limits=Limits(place={"lat": 21.24, "lng": 81.69, "pincode": "492001"}))
    await db.commit()
    await h.tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    await db.refresh(t)
    assert agent.runs == [], "no browser agent at all: look-up and cart both through the store's requests"
    assert built["products"][0]["store_id"] == "746124"
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm" and t.result["total"] == "₹218" and "₹218" in h.told[-1]


async def test_not_logged_in_the_agent_only_logs_in_then_the_cart_is_built_fast(db, at, sessions, monkeypatch):
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    async def fake_search(service, cdp, query, lat=None, lon=None, pincode=None):
        return {"deliverable": True, "eta": None, "items": [{"name": "Diet Coke", "pack": "330 ml", "price": "₹209", "available": True, "store_id": "746124"}]}

    calls = {"n": 0}

    async def fake_cart(cdp, products, lat=None, lon=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise fastpath.FastPathError("not logged in")
        return {"items": [{"name": "Diet Coke", "qty": 1, "price": "₹209", "available": True}], "total": "₹218", "cod_available": True, "logged_in": True}

    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setattr(fastpath, "blinkit_cart", fake_cart)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 18:10")
    logged = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": ""}
    agent = FastAgent(script={"prepare": [OTP, logged]})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]}, limits=Limits(place={"lat": 21.24, "lng": 81.69}))
    await db.commit()
    await h.tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert agent.runs, [x["note"] for x in t.history]
    goal = agent.runs[0]["goal"]
    assert "only log in" in goal and "enter the mobile number 9000012345" in goal and "Put exactly these in the cart" not in goal
    assert t.input_needed == "otp"
    await runtime.provide_input(db, t, kind="otp", value="4821", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick(); await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm" and t.result["total"] == "₹218" and calls["n"] == 2


async def test_rapido_fares_come_through_its_own_request_with_places_resolved(db, at, sessions, monkeypatch):
    from app.tasks import fastpath

    class Host:
        async def call(self, tool, args, **kw):
            assert tool == "ride_place"
            return {"lat": 21.24, "lng": 81.69, "label": args["words"]}

    async def fake_fares(service, cdp, pickup, drop):
        assert pickup["label"] == "Home" and drop["label"] == "Raipur Railway Station"
        return {"login_required": False, "options": [{"type": "Auto", "fare": "₹135 - ₹164", "eta": None}]}

    monkeypatch.setattr(fastpath, "fares", fake_fares)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 18:30")
    agent = FastAgent(script={})
    told = []
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="rapido", kind="ride", goal="Auto to station",
                             details={"pickup": "Home", "drop": "Raipur Railway Station"})
    await db.commit()

    async def notify(f, r, p):
        told.append(p)

    async def prof(t):
        return {"profileId": "p", "loginPhone": "9000012345"}

    await runtime.tick(sessions, agent, profile_for=prof, notify=notify, host_for=lambda f: Host())
    await db.refresh(t)
    assert agent.runs == [] and t.status == "awaiting_confirm" and t.input_needed == "go" and "Auto ₹135 - ₹164" in told[-1]


def test_medicine_strength_written_with_its_unit_still_matches():
    from app.tasks.fastpath import _words
    assert _words("Dolo 650") <= _words("Dolo 650Mg Strip Of 15 Tablets")
    assert not _words("Telma 40") <= _words("Telma 80 Tablet")
