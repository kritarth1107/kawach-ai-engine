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


async def _pick(db, t, value="yes", by=ELDER) -> str:
    """The person picks from the options listed (the brain passes the number shown, or yes for the suggested one)."""
    await db.refresh(t)
    out = await runtime.provide_input(db, t, kind="go", value=value, by=by, by_is_elder=by == ELDER)
    await db.commit()
    return out


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
    assert t.status == "needs_input" and t.input_needed == "otp" and "2002" in h.told[-1]
    assert await runtime.provide_input(db, t, kind="otp", value="hello", by=ELDER, by_is_elder=True) != "code received; continuing"
    assert await runtime.provide_input(db, t, kind="otp", value="4821", by=ELDER, by_is_elder=True) == "code received; continuing"
    await db.commit()
    await h.tick()
    assert "4821" in h.agent.runs[-1]["goal"]
    await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm"


async def test_cancel_before_placing_stops_agent(db, at, sessions):
    from app.tasks import sandbox

    at("2026-10-02 10:00")
    await sandbox.note_login(db, FAM, "instamart", {"logged_in": True}, phase="prepare")  # the account is logged in
    agent = FakeAgent(script={"prepare": [CART]}, finish_after_polls=5)
    h = Harness(sessions, agent)
    t = await _order(db)
    await h.tick()
    await db.refresh(t)
    assert t.status == "running"
    msg = await runtime.request_cancel(db, agent, t, by=ELDER, reason="changed mind")
    # the cart run had the family's login: the store's own list is read before "nothing was placed" (founder 2026-10-12)
    assert msg.startswith("STOPPED; checking on Swiggy Instamart") and t.phase == "check" and agent.stopped
    assert "never say nothing was placed" in msg
    await db.commit()
    agent.finish_after_polls = 1
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "cancelled" and agent.runs[-1]["phase"] == "check"
    assert any("checked on swiggy instamart: nothing" in h_["note"].lower() for h_ in t.history[-2:])


async def test_a_self_care_ride_cancel_is_done_on_the_app_and_a_free_one_is_not_asked_again(db, at, sessions):
    """Founder 2026-10-12: someone ordering for themselves asks to cancel a booked ride: the ride app is asked to cancel it; a
    ₹0 fee is not asked about again; the one result says it is cancelled."""
    at("2026-10-02 10:00")
    free = {**CANCELLED, "cancelled": False, "cancel_fee": "₹0"}
    h = Harness(sessions, FakeAgent(script={"prepare": [FARES], "place": [BOOKED], "cancel": [free, CANCELLED]}))
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="uber", kind="ride", goal="Cab to clinic",
                             details={"pickup": "Home", "drop": "Dr Iyer clinic"})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.input_needed == "choice" and "1. Auto ₹142 (pickup in 4 min) ← suggested" in h.told[-1] and "2. Uber Go ₹236" in h.told[-1]
    assert "pass its number" in await runtime.provide_input(db, t, kind="choice", value="the cheap one", by=ELDER, by_is_elder=True)
    await runtime.provide_input(db, t, kind="choice", value="1", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "done" and "Auto ₹142" in h.agent.runs[-1]["goal"]
    said = await runtime.request_cancel(db, h.agent, t, by=ELDER, reason="not needed")
    await db.commit()
    assert said.startswith("NOT cancelled yet: it is already BOOKED on Uber (ride UB-9921), driver Ramesh") and "asking Uber to cancel it" in said
    told = len(h.told)
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert t.status == "cancelled" and len([r for r in h.agent.runs if r["phase"] == "cancel"]) == 2
    assert len(h.told) == told + 1 and "Cancelled on Uber" in h.told[-1] and "costs" not in h.told[-1]


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


async def test_order_looks_first_lists_what_it_found_then_logs_in_and_asks_once(db, at, sessions, monkeypatch):
    """Founder 2026-10-11: options at all times. The look-up (no login) finds the product; the person gets what was found,
    numbered with Saheli's suggestion marked, and picks; the login code is asked; then the one confirm with the amount."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [FOUND], "prepare": [OTP, CART]}), "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    browse = h.agent.runs[0]
    assert browse["phase"] == "browse" and "without logging in" in browse["goal"] and "9000012345" not in browse["goal"]
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and len(h.told) == 1
    assert "1. Aashirvaad Atta 5kg ₹245 ← suggested" in h.told[0] and "nothing is ordered yet" in h.told[0]
    assert f"task_input on task {t.id}, kind go, value = the number(s) they picked" in h.told[0]
    assert "options sent, waiting for their pick" in runtime.describe(t) and "1. Aashirvaad Atta 5kg ₹245 (suggested)" in runtime.describe(t)
    assert (await _pick(db, t, "1")).startswith("going ahead on Swiggy Instamart with Aashirvaad Atta 5kg ₹245")
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
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm" and len(h.told) == 3
    assert "ONE short question" in h.told[-1] and "₹318" in h.told[-1] and "shall I order" in h.told[-1]


async def test_a_store_that_does_not_deliver_looks_on_the_other_stores_instead(db, at, sessions, monkeypatch):
    """Founder 2026-10-10: keep going until it is placed, asking as little as possible: the store named does not deliver
    there, so the other usual stores look (no question); nowhere at all → one message saying so."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    nope = {**FOUND, "items": [], "deliverable": False, "location_set": "Raipur 492001"}
    h = LoginHarness(sessions, FakeAgent(script={"browse": [nope]}), "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and h.told == [], "not asked: the other stores look instead"
    others = await runtime.live_tasks(db, FAM)
    assert sorted(x.service for x in others) == ["blinkit", "zepto"] and len({runtime.compare_group(x) for x in others}) == 1
    await h.tick(); await h.tick()
    assert len(h.told) == 1 and "not available on any of them" in h.told[0] and "They asked for Swiggy Instamart" in h.told[0]
    assert "Blinkit: Blinkit does not deliver" in h.told[0] and "Zepto: Zepto does not deliver" in h.told[0], "every store is reported"


async def test_already_logged_in_needs_no_code_after_the_pick(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-08 20:00")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [{**FOUND, "logged_in": True}], "prepare": [CART]}), "9000012345")
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and len(h.told) == 1
    await _pick(db, t)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm" and len(h.told) == 2  # no code was needed


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


async def test_no_store_named_looks_on_every_store_and_reports_every_one_with_the_best_suggested(db, at, sessions, monkeypatch):
    """Founder 2026-10-11 (options at all times; report every store): every usual store looks at once; ONE update lists
    what each store found, numbered, the best one suggested, and why a store has nothing. The person picks; that store
    goes on to the login and the cart, the others wait in case it cannot build the cart."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")  # a comparison looks first even when single orders do not
    at("2026-10-09 15:03")
    h = LoginHarness(sessions, FakeAgent(script={"browse:blinkit": [COKE_BLINKIT], "browse:instamart": [COKE_INSTAMART], "browse:zepto": [NO_ZEPTO],
                                                 "prepare": [OTP, CART]}, finish_after_polls=1), "9000012345")
    b, i, z = await _compare(db)
    await h.tick()  # all three look at once, none logs in
    assert sorted(r["service"] for r in h.agent.runs) == ["blinkit", "instamart", "zepto"]
    assert all(r["phase"] == "browse" and "9000012345" not in r["goal"] and "up to 3 matching products" in r["goal"] for r in h.agent.runs)
    await h.tick()
    assert len(h.told) == 1, "one update for all three stores"
    summary = h.told[0]
    for t in (b, i, z):
        await db.refresh(t)
    assert (b.status, b.input_needed, i.status, i.input_needed, z.status) == ("awaiting_confirm", "go", "awaiting_confirm", "go", "failed")
    assert f"- Blinkit [task {b.id}], delivers in about 11 mins:" in summary and "1. Diet Coke Can (300 ml) ₹40 ← suggested" in summary
    assert "2. Diet Coke (6 x 300 ml) ₹220" in summary and "3. Coca-Cola Diet Coke Can 300 ml ₹42" in summary
    assert "- Zepto: Zepto does not deliver" in summary and summary.index("Blinkit") < summary.index("Swiggy Instamart [task")
    said = await _pick(db, i, "1")
    assert "option 1 is on Blinkit's list" in said and f"task {b.id}" in said, "the number belongs to another store's task"
    assert (await _pick(db, b, "1")).startswith("going ahead on Blinkit with Diet Coke Can (300 ml) ₹40")
    await db.refresh(i)
    assert i.status == "cancelled" and i.details["dropped_for"] == str(b.id)
    await h.tick()
    prep = h.agent.runs[-1]
    assert prep["service"] == "blinkit" and prep["phase"] == "prepare" and "enter the mobile number 9000012345" in prep["goal"]
    assert "Diet Coke Can (300 ml) ₹40" in prep["goal"] and "6 x 300" not in prep["goal"]


async def test_comparison_waits_a_little_for_a_slow_store_then_reports_it_as_too_slow(db, at, sessions, monkeypatch):
    at("2026-10-09 15:03")
    agent = FakeAgent(script={"browse:blinkit": [COKE_BLINKIT], "browse:instamart": [COKE_INSTAMART], "browse:zepto": [NO_ZEPTO]})
    h = LoginHarness(sessions, agent, "9000012345")
    b, i, z = await _compare(db)
    await h.tick()
    agent.finish_after_polls = 99  # Instamart and Zepto keep loading
    agent._polls[[r["task"] for r in agent.runs if r["service"] == "blinkit"][0]] = 98
    await h.tick()
    await db.refresh(b)
    assert b.status == "awaiting_confirm" and h.told == [], "the first answer waits for the others"
    at("2026-10-09 15:08")
    await h.tick()
    for t in (b, i, z):
        await db.refresh(t)
    assert (b.status, b.input_needed) == ("awaiting_confirm", "go") and i.status == z.status == "cancelled"
    assert i.details["too_slow"] and len(agent.stopped) == 2
    assert len(h.told) == 1 and "1. Diet Coke Can (300 ml) ₹40 ← suggested" in h.told[0]
    assert "- Swiggy Instamart: its website did not answer in time" in h.told[0] and "- Zepto: its website did not answer in time" in h.told[0]


async def test_cancelling_a_look_up_on_several_stores_stops_every_store(db, at, sessions):
    at("2026-10-09 15:03")
    agent = FakeAgent(script={"browse": [COKE_INSTAMART]}, finish_after_polls=99)
    h = LoginHarness(sessions, agent, "9000012345")
    b, i, z = await _compare(db)
    await h.tick()
    for t in (b, i, z):
        await db.refresh(t)
    said = await runtime.request_cancel(db, agent, i, by=ELDER, reason="nahi chahiye")
    await db.commit()
    for t in (b, i, z):
        await db.refresh(t)
    assert b.status == i.status == z.status == "cancelled" and "nothing was placed" in said and len(agent.stopped) == 3


async def test_several_products_found_are_listed_with_the_single_pack_suggested(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 15:03")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [COKE_BLINKIT], "prepare": [OTP]}), "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.input_needed == "go" and t.details["chosen"][0]["name"] == "Diet Coke Can (300 ml)"
    assert "1. Diet Coke Can (300 ml) ₹40 ← suggested" in h.told[0] and "2. Diet Coke (6 x 300 ml) ₹220" in h.told[0]
    assert (await _pick(db, t, "you choose")).startswith("pass the number"), "words are never matched; the brain passes yes"
    await _pick(db, t, "yes")
    await h.tick()
    assert "Diet Coke Can (300 ml) ₹40" in h.agent.runs[-1]["goal"] and "6 x 300" not in h.agent.runs[-1]["goal"]


async def test_start_task_without_a_store_compares_the_usual_ones(db, at):
    from app.brain import tools
    from app.sim.world import SimHost

    at("2026-10-09 15:03")
    e = {"id": ELDER, "name": "Shanti", "role": "elder"}
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=e, speaker=e, members=[e])
    with pytest.raises(tools.ToolRefused):
        await tools.start_task(ctx, {"kind": "ride", "goal": "Cab to the clinic", "for_whom": ELDER})
    old = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Atta",
                               details={"items": [{"name": "Atta", "qty": 1}]})
    ask = {"kind": "order", "category": "grocery", "goal": "Diet Coke for Shanti ji", "items": [{"name": "Diet Coke"}], "for_whom": ELDER}
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
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    await db.refresh(t)
    assert t.status == "running" and agent.runs[-1]["phase"] == "prepare" and agent.runs[-1]["session"] is None
    # the look-up ran without the family's login; the go-ahead's step opens the family's own browser (founder 2026-10-12)
    assert agent.runs[-1]["profile"] == "prof-fam-t-instamart" and any("opens the family's own" in h_["note"] or "fresh one" in h_["note"]
                                                                       for h_ in t.history[-3:])


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
    the connector (seconds, no browser, no login code); Blinkit still uses the browser. Same price on both: the linked
    store wins (no login code), and the exact product, not the near one."""
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
    assert len(told) == 1 and "(linked account: no login code)" in told[0]
    assert "1. Coca-Cola Diet Coke Can 300 ml ₹40 ← suggested" in told[0] and "Zero Sugar" not in told[0]
    im = next(t for t in await runtime.live_tasks(db, FAM) if t.service == "instamart")
    await runtime.provide_input(db, im, kind="go", value="1", by=ELDER, by_is_elder=True); await db.commit()
    assert im.phase == "prepare" and all(t.service == "instamart" for t in await runtime.live_tasks(db, FAM))
    await tick()
    await db.refresh(im)
    assert im.status == "awaiting_confirm" and im.input_needed == "confirm" and im.result["total"] == "₹49"
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

    await runtime.tick(sessions, FakeAgent(), profile_for=no_profile, notify=notify, host_for=lambda f: host)
    await db.refresh(t)
    assert t.input_needed == "go", "what was found goes to the person first"
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True)
    await db.commit()
    for _ in range(2):
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


async def test_connector_cart_uses_the_ais_first_exact_listing(db, at, sessions, monkeypatch, fake_matcher):
    """The AI ranks the exact listings (a single pack first); the cart takes its first one and asks once."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 14:30")
    host = ParleHost()
    fake_matcher.script["Parle-G biscuit"] = {"exact": ["Parle Parle G Gold Biscuits Pouch — 1 kg", "Parle Parle-G Gold Biscuits — 475 g"]}
    t, told = await _connector_order(db, sessions, host, "Parle-G biscuit")
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm"
    assert next(c[1] for c in host.calls if c[0] == "connector_prepare")["pick"]["spinId"] == "P1"
    assert len(told) == 2 and "1. Parle Parle G Gold Biscuits Pouch — 1 kg ₹75 ← suggested" in told[0] and "ONE short question" in told[1]
    assert fake_matcher.calls[0][0] == ["Parle-G biscuit"] and len(fake_matcher.calls[0][1][0]) == 3, "the AI saw every listing"


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


async def test_connector_takes_several_items_in_one_cart(db, at, sessions, monkeypatch, fake_matcher):
    """Order lab 2026-10-10: "add biscuits and munchies" — two items, one Instamart cart, each from its own look-up;
    Saheli picks each herself and asks once."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 15:30")
    host = SnackHost()
    fake_matcher.script["munchies"] = {"exact": ["Kurkure Masala Munch — 90 g"]}
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
    assert t.input_needed == "go" and len(told) == 1
    assert "for biscuits:" in told[0] and "1. Parle-G Gold Biscuits — 475 g ₹40 ← suggested" in told[0]
    assert "for munchies:" in told[0] and "3. Kurkure Masala Munch — 90 g ₹40 each; 2 = ₹80 ← suggested" in told[0] and "(one per item)" in told[0]
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await tick()
    await db.refresh(t)
    prep = next(c[1] for c in host.calls if c[0] == "connector_prepare")
    assert prep["pick"]["name"].startswith("Parle-G") and prep["qty"] == 1
    assert prep["more"] == [{"pick": {"store": "instamart", "name": "Kurkure Masala Munch — 90 g", "spinId": "Kurkur"}, "qty": 2}]
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm"
    assert "not in the cart" not in told[-1]  # the cart is checked against the picked products, not the words asked


async def test_a_pick_is_an_option_id_never_a_name(db, at, sessions, monkeypatch, fake_matcher):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 15:30")
    host = SnackHost()
    fake_matcher.script["munchies"] = {"exact": ["Kurkure Masala Munch — 90 g"]}
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order",
                             goal="biscuits and munchies", details={"items": [{"name": "biscuits", "qty": 1}, {"name": "munchies", "qty": 1}]})
    await db.commit()

    async def no_profile(t):
        return None

    async def notify(f, r, p):
        return None

    async def tick():
        await runtime.tick(sessions, FakeAgent(), profile_for=no_profile, notify=notify, host_for=lambda f: host)

    await tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="go", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await tick()
    await db.refresh(t)
    assert t.input_needed == "confirm"
    assert "looking for more" in await runtime.provide_input(db, t, kind="more", target=2, by=ELDER, by_is_elder=True)
    await db.commit()
    await tick()
    await db.refresh(t)
    assert t.input_needed == "go"
    said = await runtime.provide_input(db, t, kind="go", value="Lay's India's Magic Masala Chips — 50 g", by=ELDER, by_is_elder=True)
    assert "pass the number" in said and t.phase == "browse"
    lays = next(n for n, o in t.details["offer"].items() if o["text"].startswith("Lay's"))
    said = await runtime.provide_input(db, t, kind="go", value=lays, by=ELDER, by_is_elder=True)
    assert said.startswith("going ahead") and [c["name"][:6] for c in t.details["chosen"]] == ["Parle-", "Lay's "]


def test_food_orders_take_several_dishes_of_one_restaurant_on_the_connector():
    """Live run 2026-10-11: Swiggy's connector takes several dishes of one restaurant in one cart (zomato has no connector)."""
    from app.specialists import channels

    assert {"instamart", "swiggy"} <= channels.MULTI_ITEM and "zomato" not in channels.MULTI_ITEM


class FastAgent(FakeAgent):
    async def open_session(self, profile_id):
        self.opened = profile_id
        return "fs-1"

    async def cdp_url(self, session_id):
        return "https://fs-1.cdp.test"


async def test_blinkit_looks_up_through_its_own_web_request_in_seconds_then_the_agent_logs_in_there(db, at, sessions, monkeypatch):
    """Founder 2026-10-09 'make it fast': the agent's Blinkit look-up took 3 minutes; the fixed search runs Blinkit's own
    request in the task's cloud browser. The same browser then serves the login and cart once the person picked."""
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
    place = {"addressId": "a1", "nickname": "Home", "pincode": "492001", "full": "Sample Park, Raipur", "lat": 21.251, "lng": 81.6296}
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]}, limits=Limits(place=place))
    await db.commit()
    await h.tick()
    await db.refresh(t)
    assert agent.runs == [] and asked == {"cdp": "https://fs-1.cdp.test", "query": "Diet Coke", "lat": 21.251, "lon": 81.6296}
    assert t.input_needed == "go" and t.agent_session == "fs-1" and "1. Coca-Cola Diet Coke Soft Drink No Caffeine (330 ml) ₹209" in h.told[0]
    await _pick(db, t, "1")
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
                "total": "₹218", "fees": "₹9", "cod_available": True, "logged_in": True, "address_used": "Home: 12, Sample Park, Raipur 492001"}

    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setattr(fastpath, "blinkit_cart", fake_cart)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-09 18:00")
    agent = FastAgent(script={})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Diet Coke",
                             details={"items": [{"name": "Diet Coke", "qty": 1}]}, limits=Limits(place={"lat": 21.25, "lng": 81.63, "pincode": "492001"}))
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
                             details={"items": [{"name": "Diet Coke", "qty": 1}]}, limits=Limits(place={"lat": 21.25, "lng": 81.63}))
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
            return {"lat": 21.25, "lng": 81.63, "label": args["words"]}

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


async def _instamart_browser_order(db, h, monkeypatch, place_result, second_total=None):
    """Instamart through the browser with the saved steps (order lab 2026-10-10): look-up, cart and place by the store's own
    requests in the family's logged-in cloud browser; the agent is never started."""
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    calls = {"cart": 0, "place": []}

    async def fake_search(service, cdp, query, lat=None, lon=None, pincode=None):
        return {"deliverable": True, "eta": "15 mins", "items": [
            {"name": "Vachan Butter", "pack": "200 g", "price": "₹117", "available": True, "exact_match": True,
             "cart_ref": {"product_id": "P1", "spin": "S1", "item_id": "I1"}}]}

    async def fake_cart(service, cdp, products, place=None, lat=None, lon=None):
        calls["cart"] += 1
        assert service == "instamart" and products[0]["cart_ref"]["item_id"] == "I1" and place["pincode"] == "492001"
        total = second_total if second_total and calls["cart"] > 1 else 129
        return {"items": [{"name": "Vachan Butter", "qty": 1, "price": "₹117", "available": True}], "total": f"₹{total}", "fees": "₹12",
                "cod_available": True, "logged_in": True, "address_used": "Kavach Home: B12 Sample Park, Raipur 492001",
                "place_check": {"items": {"I1": 1}, "total": total, "address_id": "A1"}}

    async def fake_place(service, cdp, check):
        calls["place"].append(check)
        if isinstance(place_result, Exception):
            raise place_result
        return place_result

    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setattr(fastpath, "cart", fake_cart)
    monkeypatch.setattr(fastpath, "place", fake_place)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    place = {"addressId": "a1", "nickname": "Home", "pincode": "492001", "full": "B12 Sample Park, Raipur", "line1": "B12, Sample Park", "lat": 21.25, "lng": 81.63}
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="butter",
                             details={"items": [{"name": "butter", "qty": 1}], "channel": "browser"}, limits=Limits(place=place))
    await db.commit()
    await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "go", "what was found goes to the person"
    await _pick(db, t, "1")
    await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm" and t.result["total"] == "₹129" and calls["cart"] == 1
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    await db.refresh(t)
    return t, calls


async def test_instamart_browser_order_places_with_the_saved_steps(db, at, sessions, monkeypatch):
    at("2026-10-10 16:00")
    h = LoginHarness(sessions, FastAgent(), "9000012345")
    t, calls = await _instamart_browser_order(db, h, monkeypatch, {"placed": True, "order_id": "IM-77", "total": "₹129", "payment_method": "Cash on Delivery", "eta": "15-16 min"})
    assert t.status == "done" and t.result["order_id"] == "IM-77" and h.agent.runs == []
    assert calls["place"] == [{"items": {"I1": 1}, "total": 129, "address_id": "A1"}]


async def test_instamart_cart_that_changed_before_placing_sends_nothing_and_asks_again(db, at, sessions, monkeypatch):
    from app.tasks import fastpath

    at("2026-10-10 16:00")
    h = LoginHarness(sessions, FastAgent(), "9000012345")
    t, calls = await _instamart_browser_order(db, h, monkeypatch, fastpath.FastPathError("not placed: total_changed"), second_total=140)
    await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.input_needed == "confirm" and calls["cart"] == 2 and len(calls["place"]) == 1
    assert not t.result.get("order_id") and h.agent.runs == [] and "₹140" in h.told[-1]


async def test_a_saved_place_step_that_breaks_is_placed_by_the_agent_on_the_same_yes(db, at, sessions, monkeypatch):
    """Founder 2026-10-10: even if the website changes, it should place. The saved place step fails before sending (site
    changed): the cart is built again, it is the same cart at the same total, so the browser agent places it on the yes
    already given; nobody is asked twice, and the founder gets one alert to fix the step."""
    from app.sim.agent import PLACED
    from app.tasks import fastpath

    at("2026-10-10 16:00")
    alerts = []

    class Host:
        async def call(self, tool, args, **kw):
            alerts.append((tool, args.get("subject")))
            return {}

    agent = FastAgent(script={"place": [{**PLACED, "total": "₹129"}]})
    h = LoginHarness(sessions, agent, "9000012345")
    h.host = Host()
    orig = h.tick

    async def tick():
        return await runtime.tick(h.sessions, h.agent, profile_for=h.profile_for, notify=h.notify, host_for=lambda f: h.host)

    h.tick = tick
    t, calls = await _instamart_browser_order(db, h, monkeypatch, fastpath.FastPathError("not placed: Pay button not found"))
    asks = len(h.told)
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert calls["cart"] == 2 and len(calls["place"]) == 1, "the saved place step is not tried again"
    assert [r["phase"] for r in agent.runs] == ["place"] and "Cash / Pay on Delivery" in agent.runs[0]["goal"]
    assert t.status == "done" and t.result["order_id"] == "IM-55821" and len(h.told) == asks + 1 and "Placed" in h.told[-1]
    assert alerts == [("ops_alert", "Saved place step broke on Swiggy Instamart")]
    del orig


async def test_instamart_place_without_a_clear_answer_is_never_retried(db, at, sessions, monkeypatch):
    at("2026-10-10 16:00")
    h = LoginHarness(sessions, FastAgent(), "9000012345")
    t, calls = await _instamart_browser_order(db, h, monkeypatch, {"placed": False, "unclear": True, "problem": "check the app"})
    for _ in range(2):
        await h.tick()
    await db.refresh(t)
    assert t.status != "done" and len(calls["place"]) == 1 and h.agent.runs == []


class _FakeBlinkitTab:
    """Stand-in for the cloud browser's CDP socket while Blinkit's checkout runs: the payment frame (Zomato's zpaykit)
    attaches as its own target, and Pay Now loads the order page with a full navigation (order lab 2026-10-10)."""

    def __init__(self, press_error=None, order_url="https://blinkit.com/account/orders/track/9001/3837182613",
                 track_texts=("Order is on the way", "Arriving in 9 minutes")):
        from app.tasks import fastpath
        self.js = {n: (fastpath.JS_DIR / f"{n}.js").read_text() for n in ("blinkit_cash_frame", "blinkit_paynow", "blinkit_clear_cart")}
        self.out, self.press_error, self.order_url = [], press_error, order_url
        self.pressed, self.cleared, self.closed = False, False, False
        self.track_texts, self.track_sent = track_texts, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, raw):
        import json as _j
        m = _j.loads(raw)
        method, p, sid, res = m["method"], m.get("params") or {}, m.get("sessionId"), {}
        if method == "Target.createTarget":
            res = {"targetId": "T"}
        elif method == "Target.attachToTarget":
            res = {"sessionId": "S"}
        elif method == "Target.setAutoAttach":
            self.out.append({"method": "Target.attachedToTarget", "params": {"sessionId": "FS", "targetInfo": {"targetId": "F", "url": "https://www.zomato.com/zpaykit/init?x=1"}}})
        elif method == "Target.getTargetInfo":
            res = {"targetInfo": {"url": self.order_url if self.pressed else "https://blinkit.com/checkout"}}
            if self.pressed and not self.track_sent:  # the order page asks for its tracking layout
                self.track_sent = True
                url = "https://blinkit.com/v1/layout/crystal_track_order?cart_id=9001&order_id=3837182613"
                self.out.append({"method": "Network.responseReceived", "sessionId": "S", "params": {"requestId": "R1", "response": {"url": url}}})
                self.out.append({"method": "Network.loadingFinished", "sessionId": "S", "params": {"requestId": "R1"}})
        elif method == "Network.getResponseBody":
            body = {"response": {"snippets": [{"data": {"title": {"text": t}}} for t in self.track_texts]}}
            res = {"body": _j.dumps(body), "base64Encoded": False}
        elif method == "Target.closeTarget":
            self.closed = True
        elif method == "Runtime.evaluate":
            e = p["expression"]
            if sid == "FS":
                res = {"result": {"value": {"cash": True}}}
            elif e == self.js["blinkit_paynow"]:
                if self.press_error:
                    self.out.append({"id": m["id"], "error": {"code": -32000, "message": self.press_error}})
                    return
                self.pressed = True
                res = {"result": {"value": {"clicked": True}}}
            elif e == self.js["blinkit_clear_cart"]:
                self.cleared = True
                res = {"result": {"value": True}}
            else:  # the checkout check on the main page
                res = {"result": {"value": {"ok": True, "page": {"count": 2, "total": 180}}}}
        self.out.append({"id": m["id"], "result": res})

    async def recv(self):
        import asyncio
        import json as _j

        while not self.out:  # noqa: ASYNC110 — a fake socket waiting for its next scripted answer
            await asyncio.sleep(0.05)
        return _j.dumps(self.out.pop(0))


def _fake_cdp(monkeypatch, tab):
    from app.tasks import fastpath

    class Resp:
        def json(self):
            return {"webSocketDebuggerUrl": "ws://fake"}

    class Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return Resp()

    monkeypatch.setattr(fastpath.httpx, "AsyncClient", Client)
    monkeypatch.setattr(fastpath.websockets, "connect", lambda *a, **k: tab)


BLINKIT_CHECK = {"count": 2, "prices": [78, 63], "address": "12, Block D", "total": 180}


async def test_blinkit_place_reads_the_order_id_after_the_full_page_load_and_empties_the_app_cart(monkeypatch):
    """Lab order 2 (₹180, 2026-10-10) went through but was reported unclear: Pay Now reloads the whole tab."""
    from app.tasks import fastpath
    tab = _FakeBlinkitTab()
    _fake_cdp(monkeypatch, tab)
    out = await fastpath.place("blinkit", "http://cdp", BLINKIT_CHECK)
    assert out == {"placed": True, "order_id": "3837182613", "total": "₹180", "payment_method": "Cash on Delivery", "eta": "9 min"}
    assert tab.cleared and tab.closed


async def test_blinkit_place_without_a_delivery_time_on_the_order_page_gives_none(monkeypatch):
    """The page header's store time ("Delivery in 10 minutes") is not the order's; an arrived order has no time to come."""
    from app.tasks import fastpath
    tab = _FakeBlinkitTab(track_texts=("Delivery in 10 minutes", "Order arrived in 20 minutes"))
    _fake_cdp(monkeypatch, tab)
    out = await fastpath.place("blinkit", "http://cdp", BLINKIT_CHECK)
    assert out["placed"] is True and "eta" not in out


def test_blinkit_delivery_time_words():
    from app.tasks.fastpath import blinkit_eta
    assert blinkit_eta(["Order is on the way", "Arriving in 9 minutes"]) == "9 min"
    assert blinkit_eta(["Ajay is 6 mins away"]) == "6 min"
    assert blinkit_eta(["Delivery in 10 minutes Home"]) is None
    assert blinkit_eta(["Order arrived in 20 minutes"]) is None


async def test_blinkit_place_dry_run_stops_before_pay_now(monkeypatch):
    from app.tasks import fastpath
    tab = _FakeBlinkitTab()
    _fake_cdp(monkeypatch, tab)
    out = await fastpath.blinkit_place("http://cdp", BLINKIT_CHECK, dry=True)
    assert out["ready"] is True and not tab.pressed


async def test_blinkit_place_that_breaks_after_pay_now_is_unclear_never_not_placed(monkeypatch):
    from app.tasks import fastpath
    tab = _FakeBlinkitTab(press_error="Inspected target navigated or closed")
    _fake_cdp(monkeypatch, tab)
    out = await fastpath.place("blinkit", "http://cdp", BLINKIT_CHECK)
    assert out["placed"] is False and out["unclear"] is True


async def test_blinkit_place_without_a_checked_cart_sends_nothing():
    from app.tasks import fastpath
    with pytest.raises(fastpath.FastPathError):
        await fastpath.place("blinkit", "http://cdp", {"count": 0})


def _fake_cart_from(products_seen):
    async def fake_cart(service, cdp, products, place=None, lat=None, lon=None):
        products_seen.append([f"{p['name']} ({p.get('pack')})" for p in products])
        return {"items": [{"name": p["name"], "qty": p.get("qty") or 1, "price": p.get("price"), "available": True} for p in products],
                "total": "₹300", "fees": "₹9", "cod_available": True, "logged_in": True}
    return fake_cart


async def test_more_options_for_one_item_looks_it_up_again_with_a_longer_list(db, at, sessions, monkeypatch):
    """Order lab 2026-10-10 (Blinkit): "Maggi more" — the 840 g pack was not among the ones listed; at the confirm the
    person can ask to see more of one item, pick one, and the other item keeps Saheli's pick."""
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    asked, carts = [], []

    async def fake_search(service, cdp, query, lat=None, lon=None, pincode=None, limit=6):
        asked.append((query, limit))
        if "mushroom" in query:
            return {"deliverable": True, "eta": None, "items": [{"name": "Button Mushroom", "pack": "180 g", "price": "₹68", "available": True, "store_id": "M1"}]}
        packs = [(f"{70 + n} g", "₹20") for n in range(6)] + [("840 g", "₹167")]  # the big pack is 7th: cut at 6
        return {"deliverable": True, "eta": None, "items": [{"name": "Maggi 2 Minutes Noodles", "pack": p, "price": r, "available": True,
                                                              "store_id": f"G{p}"} for p, r in packs][:limit]}

    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setattr(fastpath, "cart", _fake_cart_from(carts))
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 17:30")
    h = LoginHarness(sessions, FastAgent(script={}), "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="mushroom, maggi",
                             details={"items": [{"name": "mushroom", "qty": 1}, {"name": "maggi", "qty": 1}]},
                             limits=Limits(place={"lat": 21.25, "lng": 81.63, "pincode": "492001"}))
    await db.commit()
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "go" and "for mushroom:" in h.told[0] and "for maggi:" in h.told[0]
    await _pick(db, t, "yes")
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "confirm" and carts == [["Button Mushroom (180 g)", "Maggi 2 Minutes Noodles (70 g)"]] and len(h.told) == 2
    assert "pass target" in await runtime.provide_input(db, t, kind="more", by=ELDER, by_is_elder=True)
    out = await runtime.provide_input(db, t, kind="more", target=2, by=ELDER, by_is_elder=True); await db.commit()
    assert "looking for more maggi" in out
    await h.tick()
    await db.refresh(t)
    assert ("maggi", runtime.MORE_LIMIT) in asked and ("mushroom", 6) in asked[2:], asked
    assert t.input_needed == "go" and "840 g" in h.told[-1] and "Kept as picked before: Button Mushroom (180 g) ₹68" in h.told[-1]
    assert "for mushroom" not in h.told[-1], "only the item they asked to see more of is listed"
    assert "pass the number" in await runtime.provide_input(db, t, kind="go", value="Maggi 840 g", by=ELDER, by_is_elder=True)
    big = next(n for n, o in t.details["offer"].items() if "840 g" in o["text"])
    out = await runtime.provide_input(db, t, kind="go", value=big, by=ELDER, by_is_elder=True)
    assert "going ahead" in out, out
    await db.commit()
    await h.tick()
    await db.refresh(t)
    assert carts[-1] == ["Button Mushroom (180 g)", "Maggi 2 Minutes Noodles (840 g)"] and t.input_needed == "confirm" and h.agent.runs == []


async def test_change_add_remove_at_the_confirm_are_data_from_the_brain(db, at, sessions, monkeypatch):
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    asked, carts = [], []

    async def fake_search(service, cdp, query, lat=None, lon=None, pincode=None, limit=6):
        asked.append((query, limit))
        size = "1 kg" if "1 kg" in query else "200 g"
        name = "Bread" if "bread" in query else "Amul Paneer"
        return {"deliverable": True, "eta": None, "items": [{"name": name, "pack": size, "price": "₹90", "available": True, "store_id": name + size}]}

    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setattr(fastpath, "cart", _fake_cart_from(carts))
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 17:40")
    h = LoginHarness(sessions, FastAgent(script={}), "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="paneer",
                             details={"items": [{"name": "paneer", "qty": 2}]}, limits=Limits(place={"lat": 21.25, "lng": 81.63, "pincode": "492001"}))
    await db.commit()
    await h.tick()
    await _pick(db, t)
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "confirm"
    assert "updating" in await runtime.provide_input(db, t, kind="change", items=[{"name": "paneer 1 kg", "qty": 2}], by=ELDER, by_is_elder=True)
    await db.commit()
    await db.refresh(t)
    assert t.details["items"] == [{"name": "paneer 1 kg", "qty": 2}] and t.phase == "prepare" and not t.details.get("cart_fp")
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "go" and "1. Amul Paneer (1 kg) ₹90 each; 2 = ₹180 ← suggested" in h.told[-1], "the changed item's options"
    await _pick(db, t, "1")
    await h.tick()
    await db.refresh(t)
    assert asked[-1] == ("paneer 1 kg", 6) and carts[-1] == ["Amul Paneer (1 kg)"] and t.input_needed == "confirm"
    await runtime.provide_input(db, t, kind="add", items=[{"name": "bread", "qty": 2}], by=ELDER, by_is_elder=True); await db.commit()
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "go" and "1. Bread (200 g) ₹90 each; 2 = ₹180" in h.told[-1] and "Kept as picked before: Amul Paneer (1 kg) ₹90" in h.told[-1]
    await _pick(db, t, "1")
    await h.tick()
    await db.refresh(t)
    assert t.details["items"][-1] == {"name": "bread", "qty": 2} and carts[-1] == ["Amul Paneer (1 kg)", "Bread (200 g)"]
    assert "only item" not in await runtime.provide_input(db, t, kind="remove", target=1, by=ELDER, by_is_elder=True)
    await db.commit()
    await db.refresh(t)
    assert t.details["items"] == [{"name": "bread", "qty": 2}]
    told = len(h.told)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.input_needed == "confirm" and carts[-1] == ["Bread (200 g)"] and len(h.told) == told + 1, "nothing new to pick: the cart"
    assert "only item" in await runtime.provide_input(db, t, kind="remove", target=1, by=ELDER, by_is_elder=True)

def test_blinkit_order_list_finds_the_order_a_place_step_lost():
    """Lab order 2 (2026-10-10): it went through at 5:06 pm and was reported unclear; the order list shows it."""
    from datetime import datetime, timezone

    from app.tasks import fastpath

    def card(oid, title, sub, status):
        return {"data": {"title": {"text": title}, "subtitle": {"text": sub}},
                "tracking": {"common_attributes": {"order_id": oid, "order_status": status}}}

    body = {"response": {"snippets": [card("3837182613", "Order is confirmed", "₹180 • Today, 5:06 pm", "CONFIRMED"),
                                      card("3837132565", "Order is on the way", "₹169 • Today, 4:58 pm", "ON_THE_WAY"),
                                      card("3829904192", "Arrived in 12 minutes", "₹218 • 09 Oct, 4:33 pm", "DELIVERED")]}}
    now = datetime(2026, 10, 10, 11, 38, tzinfo=timezone.utc)  # 5:08 pm IST
    got = fastpath.blinkit_orders_from(body, now=now)
    assert {k: got[0][k] for k in ("order_id", "status", "minutes_ago", "amount")} == {"order_id": "3837182613", "status": "CONFIRMED", "minutes_ago": 2, "amount": 180}
    assert got[2]["minutes_ago"] is None


async def test_blinkit_unclear_place_is_resolved_from_the_order_list(monkeypatch):
    from app.tasks import fastpath

    async def unclear(cdp, check, dry=False):
        return {"placed": False, "unclear": True, "problem": "no order page"}

    async def recent(cdp, total, within_min=15):
        assert total == 180
        return "3837182613"

    async def no_wait(s):
        return None

    monkeypatch.setattr(fastpath, "blinkit_place", unclear)
    monkeypatch.setattr(fastpath, "blinkit_recent_order", recent)
    monkeypatch.setattr(fastpath.asyncio, "sleep", no_wait)
    out = await fastpath.place("blinkit", "http://cdp", BLINKIT_CHECK)
    assert out == {"placed": True, "order_id": "3837182613", "total": "₹180", "payment_method": "Cash on Delivery"}


KHAKHRA = [
    {"name": "Prolicious High Protein & Fiber Thin Khakhra - Assorted Flavours", "pack": "7 x 50 g", "price": "₹412", "store_id": "676442",
     "available": True, "exact_match": True, "cart_ref": {"product_id": 676442, "merchant_id": 46743}},
    {"name": "Jabsons Roasted Wheat Khakhra (Methi)", "pack": "180 g", "price": "₹80", "store_id": "546359", "available": True,
     "exact_match": True, "cart_ref": {"product_id": 546359, "merchant_id": 37622}},
    {"name": "Charliee Methi Khakhra", "pack": "150 g", "price": "₹65", "store_id": "56720", "available": True, "exact_match": True,
     "cart_ref": {"product_id": 56720, "merchant_id": 37622}},
    {"name": "Jabsons Roasted Wheat Khakhra Jeera", "pack": "180 g", "price": "₹80", "store_id": "546362", "available": True,
     "exact_match": True, "cart_ref": {"product_id": 546362, "merchant_id": 37622}},
]


async def test_a_product_the_store_refuses_is_swapped_for_the_next_best_and_the_cart_built_fast_again(db, at, sessions, monkeypatch, fake_matcher):
    from app.specialists.contract import Limits
    from app.tasks import fastpath

    async def fake_search(service, cdp, query, lat=None, lon=None, pincode=None, limit=6):
        return {"deliverable": True, "eta": None, "items": [dict(i) for i in KHAKHRA]}

    carts = []

    async def fake_cart(service, cdp, products, place=None, lat=None, lon=None):
        carts.append([p["store_id"] for p in products])
        if products[0]["store_id"] == "56720":
            raise fastpath.ItemsUnavailable(["56720"])
        return {"items": [{"name": products[0]["name"], "qty": 1, "price": products[0]["price"], "available": True}], "total": "₹95",
                "cod_available": True, "logged_in": True}

    alerts = []

    class Host:
        async def call(self, tool, args, **kw):
            alerts.append(tool)
            return {}

    fake_matcher.script["Khakhra"] = {"exact": ["Charliee Methi Khakhra", "Jabsons Roasted Wheat Khakhra (Methi)", "Jabsons Roasted Wheat Khakhra Jeera"]}
    monkeypatch.setattr(fastpath, "search", fake_search)
    monkeypatch.setattr(fastpath, "cart", fake_cart)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-10 20:08")
    agent = FastAgent(script={})
    told = []

    async def notify(f, r, p):
        told.append(p)

    async def prof(t):
        return {"profileId": "p", "loginPhone": "9000012345"}

    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Khakhra",
                             details={"items": [{"name": "Khakhra", "qty": 1}]}, limits=Limits(place={"lat": 21.25, "lng": 81.63, "pincode": "492001"}))
    await db.commit()
    await runtime.tick(sessions, agent, profile_for=prof, notify=notify, host_for=lambda f: Host())
    await _pick(db, t, "yes")  # Saheli's suggestion: a refused one is swapped for her next exact one, not asked again
    for _ in range(3):
        await runtime.tick(sessions, agent, profile_for=prof, notify=notify, host_for=lambda f: Host())
    await db.refresh(t)
    assert carts == [["56720"], ["546359"]] and agent.runs == [] and alerts == []
    assert t.input_needed == "confirm" and "Jabsons Roasted Wheat Khakhra (Methi)" in t.details["chosen"][0]["name"]


async def test_a_slow_store_is_never_silent_the_stall_watchdog_says_what_is_happening(db, at, sessions):
    """Founder 2026-10-13 ("Saheli's message got stuck"): every 4 minutes of silence while it works, one honest line of what
    is happening now; then the run's own time limit ends it, said plainly. Never a promise left hanging."""
    at("2026-10-10 20:08")
    agent = FakeAgent(script={"prepare": [CART]}, finish_after_polls=999)
    h = Harness(sessions, agent)
    t = await _order(db)
    await h.tick()
    at("2026-10-10 20:11")
    await h.tick()
    assert h.told == []
    at("2026-10-10 20:12:30")
    await h.tick(); await h.tick()
    assert len(h.told) == 1 and "STILL WORKING on Swiggy Instamart after 4 min" in h.told[0] and "Do not ask anything" in h.told[0]
    assert "getting it ready on the store" in h.told[0] and "nothing is placed yet" in h.told[0]
    at("2026-10-10 20:15")
    await h.tick()
    assert len(h.told) == 1, "not again before another 4 quiet minutes"
    at("2026-10-10 20:16:40")
    await h.tick()
    assert len(h.told) == 2 and "after 8 min" in h.told[1]
    at("2026-10-10 20:21")
    await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and "took too long" in h.told[-1] and len(h.told) == 3


async def test_an_agent_cart_with_a_product_it_could_not_add_is_not_asked_it_picks_the_next(db, at, sessions, monkeypatch):
    """Live 2026-10-10: the agent's cart said '0 x Prolicious Khakhra' and Maa was asked to choose among four ₹246 packs."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    at("2026-10-10 20:13")
    zero = {**CART, "items": [{"name": "Prolicious High Protein & Fiber Thin Khakhra - Assorted Flavours (7 x 50 g)", "qty": 0, "price": "₹412"}],
            "total": None, "alternatives": [{"name": "Prolicious Jeera Khakhra (2 pcs)", "price": "₹246"}]}
    ok = {**CART, "items": [{"name": "Jabsons Roasted Wheat Khakhra (Methi)", "qty": 1, "price": "₹80"}], "total": "₹139"}
    h = Harness(sessions, FakeAgent(script={"prepare": [zero, ok]}))
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="Khakhra",
                             details={"items": [{"name": "Khakhra", "qty": 1}]})
    t.result = {"items": [dict(i) for i in KHAKHRA]}
    t.details = {**t.details, "browsed": True, "auto_picked": True, "chosen": [dict(KHAKHRA[0])], "fast_cart_tried": True,
                 "found": [dict(i) for i in KHAKHRA],
                 "matched": {"Khakhra": {"exact": ["676442", "56720", "546359"], "closest": None}}}
    await db.commit()
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert [c["name"] for c in t.details["chosen"]] == ["Charliee Methi Khakhra"] and t.input_needed == "confirm"
    assert len(h.told) == 1 and "ONE short question" in h.told[0] and "₹246" not in h.told[0]


async def test_change_to_several_products_makes_each_its_own_item(db, at, sessions):
    """Live 2026-10-10: Maa's 'Noice methi aur masala khakhra, peri peri muruku' became one item. The brain now passes each
    product as its own item (data); the runtime keeps them apart."""
    at("2026-10-10 20:29")
    t = await _order(db)
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "confirm"
    t.details = {**t.details, "cart_fp": "fp", "channel": "browser"}
    await db.commit()
    new = [{"name": "NOICE Methi Khakhra", "qty": 1}, {"name": "NOICE Masala Khakhra", "qty": 1},
           {"name": "Peri Peri Muruku", "qty": 1, "must_match": ["peri peri"]}]
    out = await runtime.provide_input(db, t, kind="change", target=1, items=new, by=ELDER, by_is_elder=True)
    assert "NOICE Methi Khakhra, NOICE Masala Khakhra, Peri Peri Muruku" in out
    assert [i["name"] for i in t.details["items"]] == ["NOICE Methi Khakhra", "NOICE Masala Khakhra", "Peri Peri Muruku"]
    assert t.details["items"][2]["must_match"] == ["peri peri"] and t.details["channel"] is None and t.phase == "prepare"


def test_two_items_do_not_get_the_same_product():
    t = Task(details={"items": [{"name": "methi khakhra"}, {"name": "masala khakhra"}],
                      "matched": {"methi khakhra": {"exact": ["M", "T"]}, "masala khakhra": {"exact": ["M"]}}}, result={})
    found = [{"name": "NOICE Masala Khakhra", "price": "₹59", "store_id": "M", "for_item": "methi khakhra"},
             {"name": "NOICE Methi Khakhra", "price": "₹62", "store_id": "T", "for_item": "methi khakhra"},
             {"name": "NOICE Masala Khakhra", "price": "₹59", "store_id": "M", "for_item": "masala khakhra"}]
    t.service, t.kind, t.phase = "instamart", "order", "browse"
    runtime._browse_outcome(t, {"items": found, "deliverable": True, "logged_in": True})
    assert sorted(c["store_id"] for c in t.details["chosen"]) == ["M", "T"]


MURUKU = [{"name": "Modern Kitchens Butter Muruku", "pack": "150 g", "price": "₹35", "store_id": "B", "available": True, "exact_match": False}]


def test_when_the_ai_finds_only_another_variant_it_is_offered_as_similar_and_other_stores_look():
    """Founder 2026-10-10: 'asked peri peri muruku, you got butter muruku; take care next time'. Never passed off as what
    they asked: offered marked similar, and the other stores look too (tick: _fall_back_to_other_stores)."""
    t = Task(details={"items": [{"name": "peri peri muruku"}], "matched": {"peri peri muruku": {"exact": [], "closest": "B"}}},
             result={}, service="instamart", kind="order", phase="browse")
    status, msg = runtime._browse_outcome(t, {"items": [dict(MURUKU[0])], "deliverable": True, "logged_in": True})
    assert status == "awaiting_confirm" and t.details["similar_only"] and "Nothing on Swiggy Instamart is exactly what they asked" in msg
    assert "1. Modern Kitchens Butter Muruku (150 g) ₹35 [similar: not exactly peri peri muruku] ← suggested" in msg
    assert t.details["chosen"][0]["exact_match"] is False


def test_an_item_that_is_only_another_variant_stays_out_and_the_confirm_says_so():
    t = Task(details={"items": [{"name": "methi khakhra"}, {"name": "peri peri muruku"}],
                      "matched": {"methi khakhra": {"exact": ["T"]}, "peri peri muruku": {"exact": [], "closest": "B"}}},
             result={}, service="instamart", kind="order", phase="browse")
    found = [{"name": "NOICE Methi Khakhra", "price": "₹69", "store_id": "T", "for_item": "methi khakhra", "available": True},
             {**MURUKU[0], "for_item": "peri peri muruku"}]
    status, msg = runtime._browse_outcome(t, {"items": found, "deliverable": True, "logged_in": True})
    assert status == "awaiting_confirm" and [c["store_id"] for c in t.details["chosen"]] == ["T"]
    assert "1. NOICE Methi Khakhra ₹69 ← suggested" in msg
    assert "2. Modern Kitchens Butter Muruku (150 g) ₹35 [similar: not exactly peri peri muruku]" in msg and "muruku] ←" not in msg
    chosen, left = runtime._picked(t, None)  # a plain yes: the suggestion, the similar one stays out
    assert [c["store_id"] for c in chosen] == ["T"] and left == ["peri peri muruku"]
    t.details = {**t.details, "chosen": chosen, "missing_items": left}
    ask = runtime._confirm_ask(t, {"total": "₹120"})
    assert "NOT in the cart: peri peri muruku" in ask and "never call another kind the same thing" in ask

async def test_the_stores_they_named_are_each_looked_at_and_kept_through_the_address_question(db, at):
    """Founder 2026-10-11: 'when the person asks to search several stores, report the results from every store they named'.
    The stores they named are exactly the ones searched, also when they come back only with the address answer."""
    from app.brain import tools
    from app.sim.world import SimHost

    at("2026-10-11 21:12")
    e = {"id": ELDER, "name": "Shanti", "role": "elder"}
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=e, speaker=e, members=[e])
    ask = {"kind": "order", "goal": "Karachi Bakery biscuits", "items": [{"name": "Karachi Bakery fruit biscuits"}],
           "services": ["blinkit", "zepto", "blinkit"], "for_whom": ELDER}
    ctx.message_ref = "wamid.10"
    first = await tools.start_task(ctx, ask)
    assert first["status"] == "not started yet" and "services blinkit, zepto" in first["next"]
    ctx.message_ref = "wamid.11"  # "haan, ghar pe": the brain leaves the stores out this time
    out = await tools.start_task(ctx, {k: v for k, v in ask.items() if k != "services"})
    assert out["looking_on"] == ["Blinkit", "Zepto"] and len(out["task_ids"]) == 2 and "what every store found" in out["next"]
    group = [t for t in await runtime.live_tasks(db, FAM) if runtime.compare_group(t)]
    assert {t.service for t in group} == {"blinkit", "zepto"} and len({runtime.compare_group(t) for t in group}) == 1


async def test_one_store_named_with_only_similar_products_joins_the_other_stores_and_every_option_goes_out(db, at, sessions, monkeypatch, fake_matcher):
    """The store they named has only a close variant: it is offered (marked similar) together with what the other usual
    stores found, in one update; nothing is passed off as what they asked."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 21:40")
    fake_matcher.script["Amul Taaza milk 1 litre"] = {"exact": ["Amul Taaza Toned Milk 1 L"], "closest": "Amul Gold Milk 1 L"}
    gold = {"name": "Amul Gold Milk 1 L", "price": "₹68", "available": True}
    taaza = {"name": "Amul Taaza Toned Milk 1 L", "price": "₹56", "available": True}
    agent = FakeAgent(script={"browse:instamart": [{**FOUND, "items": [gold]}], "browse:blinkit": [{**FOUND, "items": [taaza]}],
                              "browse:zepto": [{**FOUND, "items": [], "deliverable": False}]})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await _order(db, items=[{"name": "Amul Taaza milk 1 litre", "qty": 1}])
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert (t.status, t.input_needed) == ("awaiting_confirm", "go") and h.told == [], "its options wait for the other stores"
    assert t.details["similar_only"] and runtime.compare_group(t) and t.details["fell_back"] == ["Blinkit", "Zepto"]
    for _ in range(3):
        await h.tick()
    assert len(h.told) == 1
    summary = h.told[0]
    assert summary.startswith("[Task update] They asked for Swiggy Instamart: it has only similar products, not exactly this")
    assert "1. Amul Taaza Toned Milk 1 L ₹56 ← suggested" in summary
    assert "2. Amul Gold Milk 1 L ₹68 [similar: not exactly Amul Taaza milk 1 litre]" in summary and "- Zepto: Zepto does not deliver" in summary


async def test_a_light_tick_leaves_the_housekeeping_to_the_minutes_run(db, at, sessions, monkeypatch):
    at("2026-10-11 21:45")
    seen = []

    async def release(*a, **k):
        seen.append("release")
        return 0

    monkeypatch.setattr(runtime, "release_sessions", release)
    h = Harness(sessions, FakeAgent(script={"prepare": [CART]}))
    await runtime.tick(sessions, h.agent, profile_for=h.profile_for, notify=h.notify, light=True)
    assert seen == []
    await h.tick()
    assert seen == ["release"]
