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
    assert reply is None and "NOT cancelled: the order is being placed on Swiggy Instamart right now" in note
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
