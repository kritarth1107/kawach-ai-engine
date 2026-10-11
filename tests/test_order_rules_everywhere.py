"""The founder's order rules hold for every service, store mix and person (2026-10-11: "do not take this as only just this
order fix ... think of universal issue and fix"): options at all times (rides too: the fares are the options), every
store or app asked is reported, asks that wait on a person go out at once, a cancel says only what is true at every
stage, and no two messages about one task contradict each other."""

import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import loop, tools
from app.brain.loop import TurnRequest
from app.care import outcomes, store
from app.sim.agent import BOOKED, CART, OTP, FakeAgent
from app.sim.world import SimHost
from app.tasks import runtime

FAM, ELDER, SON = "fam-rules", "elder-rules", "son-rules"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
S = {"id": SON, "name": "Ravi", "role": "primary caregiver"}
GUEST = {"logged_in": False, "needs_otp": False, "blocked": False, "problem": ""}
UBER = {**GUEST, "options": [{"type": "Uber Go", "fare": "₹236", "eta": "6 min"}, {"type": "Auto", "fare": "₹142", "eta": "4 min"}]}
OLA = {**GUEST, "options": [{"type": "Auto", "fare": "₹128", "eta": "5 min"}, {"type": "Mini", "fare": "₹210", "eta": "7 min"}]}
NO_RAPIDO = {**GUEST, "options": [], "problem": "no captains nearby"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class Harness:
    def __init__(self, sessions, agent, phone="9000012345"):
        self.sessions, self.agent, self.phone, self.told = sessions, agent, phone, []

    async def profile_for(self, task):
        return {"profileId": f"prof-{task.service}", "loginPhone": self.phone}

    async def notify(self, family_id, person, prompt):
        self.told.append((person, prompt))

    async def tick(self, n=1):
        for _ in range(n):
            await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def _rides(db, services=("uber", "ola", "rapido"), by=ELDER):
    out = {}
    for s in services:
        out[s] = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=by, service=s, kind="ride", goal="Auto to AIIMS",
                                      details={"pickup": "Home", "drop": "AIIMS gate 2", "compare": "r1"})
    await db.commit()
    return out


async def test_every_ride_app_asked_reports_its_fares_numbered_and_the_pick_gets_one_confirm(db, at, sessions, monkeypatch):
    """Rides like orders: one update with every app's fares, numbered, the cheapest suggested and an app with no rides said
    with why; the person picks a number; that app logs in and the one confirm names the fare now; yes books it."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 21:30")
    fresh = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "",
             "options": [{"type": "Auto", "fare": "₹131", "eta": "4 min"}, {"type": "Mini", "fare": "₹214", "eta": "7 min"}]}
    agent = FakeAgent(script={"browse:uber": [UBER], "browse:ola": [OLA], "browse:rapido": [NO_RAPIDO], "prepare:ola": [fresh],
                              "place:ola": [{**BOOKED, "ride_id": "OLA-7"}]})
    h = Harness(sessions, agent)
    g = await _rides(db)
    await h.tick(2)
    assert len(h.told) == 1, "one update for every app"
    summary = h.told[0][1]
    assert "Looked on Ola, Rapido, Uber, from Home to AIIMS gate 2 without logging in; nothing is booked." in summary
    assert "1. Auto ₹128 (pickup in 5 min) ← suggested" in summary and "2. Mini ₹210" in summary
    assert "3. Uber Go ₹236 (pickup in 6 min)" in summary and "4. Auto ₹142 (pickup in 4 min)" in summary
    assert "- Rapido: No ride options came up on Rapido" in summary, "the app with no rides is reported with why"
    assert summary.index("- Ola [task") < summary.index("- Uber [task"), "the cheapest app first"
    for t in g.values():
        await db.refresh(t)
    said = await runtime.provide_input(db, g["uber"], kind="go", value="2", by=ELDER, by_is_elder=True)
    assert "option 2 is on Ola's list" in said and f"task {g['ola'].id}" in said
    said = await runtime.provide_input(db, g["ola"], kind="go", value="yes", by=ELDER, by_is_elder=True)
    await db.commit()
    assert said.startswith("going ahead on Ola with Auto (about ₹128 before the login)")
    await h.tick(2)
    for t in g.values():
        await db.refresh(t)
    assert g["uber"].status == "cancelled" and g["uber"].details["dropped_for"] == str(g["ola"].id), "set aside, not lost"
    assert (g["ola"].status, g["ola"].input_needed) == ("awaiting_confirm", "choice")
    confirm = h.told[-1][1]
    assert "They picked Auto (₹128 earlier); now it is ₹131" in confirm and "book Auto for ₹131, from Home to AIIMS gate 2" in confirm
    assert "task_input choice value 1" in confirm
    out = await runtime.provide_input(db, g["ola"], kind="choice", value="1", by=ELDER, by_is_elder=True)
    await db.commit()
    assert out.startswith("booking Auto ₹131")
    await h.tick(2)
    await db.refresh(g["ola"])
    assert g["ola"].status == "done" and g["ola"].result["ride_id"] == "OLA-7"


async def test_one_ride_app_lists_its_guest_fares_for_the_pick(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 21:40")
    h = Harness(sessions, FakeAgent(script={"browse": [UBER]}))
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="uber", kind="ride", goal="cab",
                             details={"pickup": "Home", "drop": "Clinic"})
    await db.commit()
    await h.tick(2)
    await db.refresh(t)
    assert (t.status, t.input_needed) == ("awaiting_confirm", "go")
    msg = h.told[-1][1]
    assert "Fares on Uber, from Home to Clinic (before the login" in msg and "2. Auto ₹142 (pickup in 4 min) ← suggested" in msg
    assert "options sent, waiting for their pick (task_input go" in runtime.describe(t)


async def test_a_ride_with_no_app_named_compares_every_ride_app_and_named_apps_are_exactly_those(db, at):
    at("2026-10-11 21:45")
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=E, members=[E, S])
    out = await tools.start_task(ctx, {"kind": "ride", "goal": "Auto to AIIMS", "pickup": "Home", "drop": "AIIMS"})
    assert out["looking_on"] == ["Uber", "Ola", "Rapido"] and "fares from every app" in out["next"]
    for t in await runtime.live_tasks(db, FAM):
        t.status = "cancelled"
    await db.commit()
    out = await tools.start_task(ctx, {"kind": "ride", "goal": "cab", "pickup": "Home", "drop": "Station", "services": ["ola", "uber", "blinkit"]})
    assert out["looking_on"] == ["Ola", "Uber"], "only the ride apps they named"


# ── a ride's pickup and drop are the places meant ────────────────────────────

HOME = {"nickname": "Ghar", "full": "12 Civil Lines, Raipur 492001", "pincode": "492001", "lat": 21.25, "lng": 81.63}


class PlacesHost(SimHost):
    """The family's address book: one saved place."""

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "list_places":
            return {"places": [{"name": "Ghar", "address": HOME["full"], "isDefault": True}]}
        if tool == "delivery_place":
            return {**HOME, "addressId": "a1", "matched": True} if args.get("exact") and args.get("words") == "Ghar" else {"matched": False}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


async def test_a_ride_from_a_saved_place_gives_the_app_its_full_address(db, at, monkeypatch):
    """Live 2026-10-12: a saved home went to the ride app as words, the app chose the city and the ride was booked there.
    A saved place they mean (an AI decision) goes as its full address; a landmark goes as their words."""
    from app.specialists.agents import specialist_for
    from app.tasks import matcher

    at("2026-10-12 08:00")

    async def which(words, places):
        return "Ghar" if "home" in words.lower() else None  # the AI's answer, faked

    monkeypatch.setattr(matcher, "which_place", which)
    ctx = tools.TurnCtx(session=db, host=PlacesHost(), family_id=FAM, elder=E, speaker=E, members=[E, S])
    await tools.start_task(ctx, {"kind": "ride", "goal": "Auto to the station", "pickup": "my home", "drop": "Raipur railway station",
                                 "services": ["ola"]})
    t = (await runtime.live_tasks(db, FAM))[0]
    assert t.details["pickup_place"]["full"] == HOME["full"] and "drop_place" not in t.details
    t.phase = "prepare"
    goal = specialist_for("ola").goal(t)
    assert f"Set the pickup to exactly '{HOME['full']}' and the drop to exactly 'Raipur railway station'" in goal
    assert "never a city or area alone" in goal and "pickup_used" in goal
    assert f"from Ghar ({HOME['full']}) to Raipur railway station" in runtime.describe(t)


async def _ride_from_home(db, script, monkeypatch):
    from app.tasks import matcher

    async def same(asked, shown, *, saved):
        return "Civil Lines" in shown or shown == "Raipur Railway Station"  # the AI's answer, faked

    monkeypatch.setattr(matcher, "same_ride_end", same)
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="ola", kind="ride", goal="Auto to the station",
                             details={"pickup": "my home", "drop": "Raipur railway station", "pickup_place": HOME})
    await db.commit()
    return t, FakeAgent(script=script)


async def _until(h, db, t, ok, most=8):
    for _ in range(most):
        await h.tick()
        await db.refresh(t)
        if ok(t):
            return


RIGHT = {"pickup_used": "Ghar, 12 Civil Lines, Raipur", "drop_used": "Raipur Railway Station"}
CITY = {"pickup_used": "Raipur, Chhattisgarh", "drop_used": "Raipur Railway Station"}
FARES_ = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "options": [{"type": "Auto", "fare": "₹142", "eta": "4 min"}]}


async def test_a_ride_the_app_set_to_a_city_is_set_again_then_stopped_never_offered_from_there(db, at, sessions, monkeypatch):
    at("2026-10-12 08:05")
    t, agent = await _ride_from_home(db, {"prepare": [{**FARES_, **CITY}, {**FARES_, **CITY}]}, monkeypatch)
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.status == "failed")
    goals = [r["goal"] for r in agent.runs if r["phase"] == "prepare"]
    assert len(goals) == 2 and "The last run set the pickup to 'Raipur, Chhattisgarh' instead of '12 Civil Lines" in goals[1]
    assert t.status == "failed" and "Nothing was booked: Ola kept setting the pickup to 'Raipur, Chhattisgarh'" in h.told[-1][1]
    assert not any("₹142" in p for _, p in h.told), "no fare from the wrong pickup was offered"


async def test_a_ride_is_never_booked_from_a_pickup_the_app_shows_wrong_and_the_booking_is_tried_once_more(db, at, sessions, monkeypatch):
    at("2026-10-12 08:10")
    mismatch = {**FARES_, "booked": False, "pickup_mismatch": True, **CITY}
    t, agent = await _ride_from_home(db, {"prepare": [{**FARES_, **RIGHT}], "place": [mismatch, {**BOOKED, "ride_id": "OLA-9", **RIGHT}]}, monkeypatch)
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "choice")
    assert "[the app set: Ghar, 12 Civil Lines, Raipur]" in h.told[-1][1], "the confirm says where the app set the pickup"
    await runtime.provide_input(db, t, kind="choice", value="1", by=ELDER, by_is_elder=True)
    await db.commit()
    await _until(h, db, t, lambda t: t.status == "done")
    places = [r["goal"] for r in agent.runs if r["phase"] == "place"]
    assert len(places) == 2 and "The last run set the pickup to 'Raipur, Chhattisgarh'" in places[1]
    assert "do NOT book: report pickup_mismatch=true" in places[0]
    assert t.status == "done" and t.result["ride_id"] == "OLA-9" and "Booked: Auto" in h.told[-1][1]
    assert "[the app set: Ghar, 12 Civil Lines, Raipur]" in runtime.describe(t)


async def test_a_ride_booked_from_another_place_is_said_first(db, at, sessions, monkeypatch):
    at("2026-10-12 08:15")
    t, agent = await _ride_from_home(db, {"prepare": [{**FARES_, **RIGHT}], "place": [{**BOOKED, "ride_id": "OLA-10", **CITY}]}, monkeypatch)
    h = Harness(sessions, agent)
    await _until(h, db, t, lambda t: t.input_needed == "choice")
    await runtime.provide_input(db, t, kind="choice", value="1", by=ELDER, by_is_elder=True)
    await db.commit()
    await _until(h, db, t, lambda t: t.status == "done")
    said = h.told[-1][1]
    assert "Booked, BUT Ola set the pickup to 'Raipur, Chhattisgarh' instead of '12 Civil Lines, Raipur 492001'" in said
    assert "Never say it is from the place they meant" in said
    assert "[the app set: Raipur, Chhattisgarh]" in runtime.describe(t)


# ── one true status ──────────────────────────────────────────────────────────


async def test_an_update_whose_task_moved_on_before_delivery_is_dropped_there(db, at, sessions):
    """Live 2026-10-11: three statuses for one order. An update is written for the task as it is; the person answers or
    cancels before it is delivered; at delivery (under the family's turn lock) it no longer holds, so nothing is said."""
    at("2026-10-11 21:50")
    await store.save_roster(db, FAM, E, [E, S])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="zepto", kind="order", goal="atta",
                             details={"items": [{"name": "atta"}]})
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "confirm"
    t.details = {**t.details, "cart_fp": "abc123"}
    await db.commit()
    token = runtime._UPDATE_FOR.set({str(t.id): runtime._stage(t)})  # "the cart is ready: shall I order?"
    check = runtime.delivery_check()
    runtime._UPDATE_FOR.reset(token)
    assert await check(db), "still true: delivered"
    said = await runtime.request_cancel(db, FakeAgent(), t, by=ELDER, reason="nahi chahiye")  # she says cancel meanwhile
    await db.commit()
    assert said == "stopped; nothing was placed" and not await check(db)
    host = SimHost()
    async with sessions() as s:
        res = await loop.run_turn(s, host, TurnRequest(family_id=FAM, elder=E, speaker={"id": "saheli", "role": "system"}, members=[E, S],
                                                       text="[Task update] The cart is ready...", message_ref="task:stale1", channel="scheduler",
                                                       still_true=check))
    assert res.stale and res.reply == "none" and host.world.sent == [], "nothing said, no model call"


def test_the_stage_moves_with_what_the_person_was_last_told():
    from app.tasks.models import Task

    t = Task(status="awaiting_confirm", phase="prepare", input_needed="confirm", details={"cart_fp": "aaaa1111"})
    a = runtime._stage(t)
    t.details = {"cart_fp": "bbbb2222"}
    assert runtime._stage(t) != a, "a changed cart is a new confirm"
    t.status, t.input_needed = "queued", None
    b = runtime._stage(t)
    t.status = "running"
    assert runtime._stage(t) == b, "queued and running are both 'working'"
    t.status, t.phase = "done", "place"
    c = runtime._stage(t)
    t.details = {"cancel_ask": {"state": "placed", "told_by": "reply"}}
    assert runtime._stage(t) != c, "a cancel answered after it was placed"


async def test_a_cancel_tap_says_cancelled_only_when_it_was_stopped(db, at, sessions, monkeypatch):
    """The confirm's Cancel button said "cancelled, nothing was ordered" whatever happened; a tap while it is being placed
    (or after) now goes to the brain with the truth, and the caregiver is told."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    at("2026-10-11 21:55")
    await store.save_roster(db, FAM, E, [E, S])
    agent = FakeAgent(script={"prepare": [CART], "place": [{**CART, "placed": True, "order_id": "IM-9", "total": "₹318"}]}, finish_after_polls=2)
    monkeypatch.setattr(tools, "task_agent", lambda: agent)
    h = Harness(sessions, agent)
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    for _ in range(4):
        await h.tick()
        await db.refresh(t)
        if t.status == "awaiting_confirm":
            break
    assert (await runtime.provide_input(db, t, kind="confirm", value="yes", by=SON, by_is_elder=False)).startswith("confirmed")
    await db.commit()
    await h.tick()  # placing now
    await db.refresh(t)
    assert (t.status, t.phase) == ("running", "place")
    reply, note = await outcomes.handle_order_button(db, family_id=FAM, speaker_id=ELDER, elder_id=ELDER, code="cancel",
                                                     task_id=f"{t.id}:{t.details['cart_fp'][:8]}", profile=None)
    await db.commit()
    assert reply is None and "NOT cancelled yet: the order is being placed on Swiggy Instamart right now" in note
    await h.tick(2)
    await db.refresh(t)
    assert t.status == "done" and [p for who, p in h.told if who == SON and p.startswith("[Order needs you]")]


# ── asks go out at once ──────────────────────────────────────────────────────


async def test_an_approval_waiting_goes_out_before_any_other_task_moves(db, at, sessions):
    at("2026-10-11 22:00")
    await store.save_roster(db, FAM, E, [E, S])
    waiting = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="atta",
                                   details={"items": [{"name": "atta"}]})
    waiting.status, waiting.input_needed = "awaiting_confirm", "approve"
    waiting.details = {**waiting.details, "approval_needed": {"reasons": ["over ₹500"], "approvers": [SON], "notified": []}}
    other = await runtime.create(db, family_id="fam-other", subject_id="e2", requested_by="e2", service="zepto", kind="order", goal="milk",
                                 details={"items": [{"name": "milk"}]})
    await db.commit()
    h = Harness(sessions, FakeAgent(script={"prepare": [OTP]}))
    await h.tick()
    assert h.told and h.told[0][0] == SON and h.told[0][1].startswith("[Approval needed]"), "the approval first"
    del other


async def test_a_store_that_shows_nothing_before_a_login_logs_in_then_lists(db, at, sessions, monkeypatch):
    """No options and straight to a cart was the old way for a store that shows nothing to a guest."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 22:05")
    nothing = {**GUEST, "items": [], "login_required": True, "deliverable": True}
    found = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "deliverable": True,
             "items": [{"name": "Aashirvaad Atta 5kg", "price": "₹245", "available": True}]}
    agent = FakeAgent(script={"browse": [nothing, found], "prepare": [{"logged_in": True, "needs_otp": False, "blocked": False, "problem": ""}]})
    h = Harness(sessions, agent)
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="zepto", kind="order", goal="atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    await h.tick(6)
    await db.refresh(t)
    assert [r["phase"] for r in agent.runs] == ["browse", "prepare", "browse"], "look, log in, look again"
    assert (t.status, t.input_needed) == ("awaiting_confirm", "go") and "1. Aashirvaad Atta 5kg ₹245 ← suggested" in h.told[-1][1]


async def test_a_store_whose_website_failed_is_reported_as_down_not_as_not_found(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 22:10")
    found = {**GUEST, "deliverable": True, "items": [{"name": "Amul Taaza Milk 500 ml", "price": "₹28", "available": True}]}
    agent = FakeAgent(script={"browse:blinkit": [found], "browse:instamart": [found], "browse:zepto": [{"__fail__": "browser crashed"}]})
    h = Harness(sessions, agent)
    for s in ("blinkit", "instamart", "zepto"):
        await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=s, kind="order", goal="milk",
                             details={"items": [{"name": "Amul Taaza milk", "qty": 1}], "compare": "c9"})
    await db.commit()
    await h.tick(6)
    summary = next(p for _, p in h.told if "Looked on" in p)
    assert "- Zepto: its website did not work just now" in summary


async def test_what_a_store_has_instead_is_offered_marked_similar(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 22:15")
    nope = {**GUEST, "deliverable": True, "items": [], "problem": "not found",
            "alternatives": [{"name": "Nandini Toned Milk 500 ml", "price": "₹26"}]}
    h = Harness(sessions, FakeAgent(script={"browse": [nope]}))
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="zepto", kind="order", goal="milk",
                             details={"items": [{"name": "Aavin milk", "qty": 1}], "only_store": True})
    await db.commit()
    await h.tick(2)
    await db.refresh(t)
    assert (t.status, t.input_needed) == ("awaiting_confirm", "go")
    assert "1. Nandini Toned Milk 500 ml ₹26 [similar: not exactly Aavin milk] ← suggested" in h.told[-1][1]
    assert json.dumps(t.details["offer"])


# ── name what was bought; never re-ask a step the order has left (founder 2026-10-12, Tata 1mg) ─────────────────────────


async def test_what_was_bought_is_named_never_the_words_of_the_request(db, at, sessions):
    """Live 2026-10-12: the confirm named the cart's product, but "placed" said "Magnesium Citrate tablets" (his words). The
    order line, the placed outcome and the delivery updates name what was bought, and say when it is not exactly that."""
    from app.tasks import tracking

    at("2026-10-12 01:47")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="1mg", kind="order", goal="Magnesium citrate tablets",
                             details={"items": [{"name": "Magnesium citrate tablets", "qty": 1}]})
    calcium = {"name": "Calcium and Vitamin D3 Veg Tablets with Zinc, Magnesium & Alfalfa", "price": "₹369", "store_id": "c1",
               "exact_match": False, "for_item": "Magnesium citrate tablets"}
    t.status, t.phase = "done", "place"
    t.details = {**t.details, "chosen": [calcium], "cart_fp": "fp1"}
    t.result = {"placed": True, "order_id": "PO1", "total": "₹384",
                "items": [{"name": calcium["name"], "qty": 1, "price": "₹369"}]}
    await db.commit()
    line = runtime.describe(t)
    assert "BOUGHT: 1 x Calcium and Vitamin D3 Veg Tablets with Zinc, Magnesium & Alfalfa" in line
    assert "NOT exactly what was asked: Calcium and Vitamin D3" in line and "is not exactly 'Magnesium citrate tablets'" in line
    assert "never the ask" in line
    status, msg = runtime._outcome(t.__class__(**{c: getattr(t, c) for c in ("id", "family_id", "subject_id", "requested_by", "service",
                                                                                 "kind", "goal", "status", "phase", "result")},
                                              details={**t.details, "confirmed_total": "₹384"}),
                                   {"placed": True, "order_id": "PO1", "total": "₹384", "payment_method": "Cash on Delivery", "eta": "1 day"})
    assert status == "done" and msg.startswith("Placed: 1 x Calcium and Vitamin D3") and "never the words they asked with" in msg
    said = tracking._say(t, "Tata 1mg", "on_the_way", 30, "")
    assert "Tata 1mg order of 1 x Calcium and Vitamin D3" in said and "Magnesium citrate tablets (" not in said.split("NOT exactly")[0]


async def test_confirm_buttons_go_only_on_an_order_still_waiting_for_that_confirm(db, at):
    """The buttons carry the cart check; an order already being placed, placed or changed gets none (no re-asked confirm)."""
    at("2026-10-12 01:50")
    await store.save_roster(db, FAM, E, [E, S])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="milk",
                             details={"items": [{"name": "milk"}]})
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "confirm"
    t.details = {**t.details, "cart_fp": "abcdef123456"}
    await db.commit()
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=E, members=[E, S])
    out, err = await tools.run(ctx, "offer_buttons", {"kind": "order", "key": str(t.id)})
    assert not err and ctx.buttons and all(f"{t.id}:abcdef12" in b["id"] for b in ctx.buttons), "the cart check rides on the buttons"
    t.status, t.phase, t.input_needed = "running", "place", None
    await db.commit()
    ctx2 = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=E, members=[E, S])
    out, err = await tools.run(ctx2, "offer_buttons", {"kind": "order", "key": str(t.id)})
    assert err and "not waiting for a yes any more" in out and not ctx2.buttons


async def test_a_second_yes_tap_says_where_it_stands_never_could_not_be_placed(db, at):
    at("2026-10-12 01:52")
    await store.save_roster(db, FAM, E, [E, S])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="milk",
                             details={"items": [{"name": "milk"}]})
    t.status, t.phase, t.input_needed = "queued", "place", None
    t.details = {**t.details, "cart_fp": "abcdef123456", "place_started": "2026-10-12T01:51:00+05:30"}
    await db.commit()
    reply, note = await outcomes.handle_order_button(db, family_id=FAM, speaker_id=ELDER, elder_id=ELDER, code="yes",
                                                     task_id=f"{t.id}:abcdef12", profile=None)
    assert reply == outcomes.ORDER_TAP["english"]["yes"] and note is None, "it is being placed: that is the truth"
    t.status, t.result = "done", {"placed": True, "order_id": "B1"}
    await db.commit()
    reply, note = await outcomes.handle_order_button(db, family_id=FAM, speaker_id=ELDER, elder_id=ELDER, code="yes",
                                                     task_id=f"{t.id}:abcdef12", profile=None)
    assert reply is None and "already placed" in note


def test_a_ride_fares_pickup_time_reads_naturally():
    from app.tasks.models import Task

    t = Task(kind="ride", details={}, result={"options": [{"type": "Auto", "fare": "₹364", "eta": "Available"},
                                                          {"type": "Mini", "fare": "₹548", "eta": "4 min"}]})
    texts = [o["text"] for o in runtime._ride_offers(t)]
    assert texts == ["Auto ₹364 (Available)", "Mini ₹548 (pickup in 4 min)"]


# ── after booking: every detail, and a cancel done in the app ─────────────────


async def test_a_booked_ride_and_a_placed_order_give_every_detail(db, at, sessions, monkeypatch):
    """Founder 2026-10-12: the booking details never reached him. The update carries the ride id, driver, car and plate,
    the pickup time and the ride PIN; an order its id, total and delivery time. The agent is asked for all of them."""
    from app.specialists.agents import specialist_for

    at("2026-10-12 08:20")
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    booked = {**BOOKED, "ride_id": "CRN-7", "driver": "Ajay", "vehicle": "White Dzire", "plate": "CG04 AB 0001", "pickup_eta": "4 min",
              "ride_pin": "4321"}
    agent = FakeAgent(script={"prepare": [{"logged_in": True, "needs_otp": False, "blocked": False, "problem": "",
                                           "options": [{"type": "Mini", "fare": "₹548", "eta": "4 min"}]}], "place": [booked]})
    h = Harness(sessions, agent)
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="ola", kind="ride", goal="Cab to the airport",
                             details={"pickup": "Home", "drop": "Airport"})
    await db.commit()
    await _until(h, db, t, lambda t: t.input_needed == "choice")
    await runtime.provide_input(db, t, kind="choice", value="1", by=SON, by_is_elder=False)
    await db.commit()
    assert "pickup_eta, ride_pin" in specialist_for("ola").goal(t)
    await _until(h, db, t, lambda t: t.status == "done")
    said = h.told[-1][1]
    for bit in ("ride id CRN-7", "driver Ajay", "car White Dzire CG04 AB 0001", "pickup in about 4 min", "ride PIN 4321"):
        assert bit in said, bit
    o = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="blinkit", kind="order", goal="milk",
                             details={"items": [{"name": "milk"}], "confirmed_total": "₹71"})
    # as the tick does: the report is on the task before its outcome is written
    o.phase, o.result = "place", {"items": [{"name": "Amul Taaza Toned Milk", "qty": 2, "price": "₹56"}], "placed": True, "order_id": "B-9",
                                  "total": "₹71", "eta": "12 min"}
    status, msg = runtime._outcome(o, {"placed": True, "order_id": "B-9", "total": "₹71", "eta": "12 min"})
    assert status == "done" and "order id B-9; total ₹71, cash on delivery; delivery in about 12 min" in msg


async def test_a_cancel_they_did_themselves_in_the_app_is_recorded_and_nothing_more_is_asked(db, at, sessions):
    """Live 2026-10-12: he cancelled the booked Ola ride in the app himself; the task still read booked (counted in the
    month's spend) and "did the cab come?" was due in the morning. Recorded by the brain's done_in_app, for every store and
    ride app and both roles."""
    from app.care import boundaries
    from app.care.models import OpenLoop
    from app.tasks import aftercare

    at("2026-10-12 02:07")
    await store.save_roster(db, FAM, E, [E, S])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="ola", kind="ride", goal="Cab to the airport",
                             details={"pickup": "Home", "drop": "Airport", "confirmed_total": "₹548", "choice": "Mini ₹548"})
    t.status, t.phase, t.result = "done", "place", {"booked": True, "ride_id": "CRN-8"}
    await aftercare.open_followup(db, t)
    await db.commit()
    assert await boundaries.month_spent(db, FAM) == 548
    # her son says he cancelled it in the Ola app: recorded, and she hears who cancelled it
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=S, members=[E, S])
    out, err = await tools.run(ctx, "done_in_app", {"task_id": str(t.id), "what": "cancelled"})
    await db.commit()
    r = json.loads(out)
    assert not err and r["result"].startswith("Recorded: they cancelled the Ola ride themselves in the Ola app")
    assert f"Tell person {ELDER}" in r["tell"] and "Ravi cancelled this ride themselves in the Ola app" in r["tell"]
    await db.refresh(t)
    assert t.status == "cancelled" and t.details["cancel_ask"]["state"] == "cancelled_in_app"
    assert await boundaries.month_spent(db, FAM) == 0, "it left the month's spend"
    loop = (await db.execute(select_loop(OpenLoop, t))).scalar_one()
    assert loop.status != "open", "no 'did the cab come?' question"
    assert "cancelled it themselves in the Ola app" in runtime.describe(t)
    # a change made in the app is recorded too, and the order stays
    o = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="zepto", kind="order", goal="milk",
                             details={"items": [{"name": "milk"}]})
    o.status, o.phase, o.result = "done", "place", {"placed": True, "order_id": "Z-1"}
    await db.commit()
    out, _ = await tools.run(ctx, "done_in_app", {"task_id": str(o.id), "what": "changed", "detail": "removed the bread"})
    await db.refresh(o)
    assert o.status == "done" and "removed the bread" in runtime.describe(o)
    # one Saheli never placed: nothing to record
    q = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="blinkit", kind="order", goal="eggs",
                             details={"items": [{"name": "eggs"}]})
    await db.commit()
    out, _ = await tools.run(ctx, "done_in_app", {"task_id": str(q.id), "what": "cancelled"})
    assert "nothing to record" in json.loads(out)["result"]


def select_loop(OpenLoop, t):
    from sqlalchemy import select

    return select(OpenLoop).where(OpenLoop.dedupe_key == f"delivery:{t.id}")
