"""Who is asked what during an order (founder 2026-10-10): the family sets who gives store login codes for a person's
orders (the care recipient, or a caregiver when the elder cannot); the code is asked from that person with the order;
when a caregiver cancels, the care recipient is told who cancelled."""

import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import boundaries, store
from app.sim.agent import CART, OTP, FakeAgent
from app.tasks import runtime

FAM, ELDER, SON = "fam-op", "elder-op", "son-op"
E = {"id": ELDER, "name": "Shanti", "role": "elder"}
S = {"id": SON, "name": "Mohan", "role": "primary caregiver"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class Harness:
    def __init__(self, sessions, phones):
        self.sessions, self.agent, self.told, self.phones, self.asked_numbers = sessions, FakeAgent(script={"prepare": [OTP, CART]}), [], phones, []

    async def profile_for(self, task):
        who = (task.details or {}).get("code_from") or task.requested_by
        self.asked_numbers.append(who)
        return {"profileId": "prof", "loginPhone": self.phones.get(who)}

    async def notify(self, family_id, person, prompt):
        self.told.append((person, prompt))

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


def test_whose_phone_gets_the_code():
    p = {"login_codes": {ELDER: SON}}
    assert boundaries.code_person({}, ELDER, ELDER) == ELDER, "not set: whoever asked (as before)"
    assert boundaries.code_person({}, ELDER, SON) == SON
    assert boundaries.code_person(p, ELDER, ELDER) == SON
    assert boundaries.code_person({"login_codes": {ELDER: "self"}}, ELDER, SON) == ELDER
    assert "login codes for Shanti's orders go to Mohan" in boundaries.describe(p, {ELDER: "Shanti", SON: "Mohan"})


async def _order(db, requested_by=ELDER):
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=requested_by, service="instamart", kind="order",
                             goal="Atta for Maa", details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    return t


async def test_the_code_is_asked_from_the_caregiver_the_family_set_with_the_order(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    at("2026-10-10 19:00")
    await boundaries.save(db, FAM, {"login_codes": {ELDER: SON}}, by=SON)
    t = await _order(db)
    assert t.details["code_from"] == SON
    h = Harness(sessions, {ELDER: "9000011111", SON: "9000022222"})
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert h.asked_numbers[0] == SON and "enter the mobile number 9000022222" in h.agent.runs[0]["goal"], "the store logs in with his number"
    assert t.status == "needs_input" and t.input_needed == "otp"
    person, prompt = h.told[-1]
    assert person == SON and prompt.startswith("[Login code needed]") and "Atta for Maa" in prompt and "ending 2222" in prompt
    assert f"tell person {ELDER}" in prompt and "cancel" in prompt
    assert not any(p == ELDER for p, _ in h.told), "Maa is not asked for a code she cannot give"
    # his code continues the order; the cart then goes back to Maa for the one confirm
    await runtime.provide_input(db, t, kind="otp", value="4821", by=SON, by_is_elder=False); await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.input_needed == "confirm" and h.told[-1][0] == ELDER


async def test_self_means_the_care_recipient_even_when_a_caregiver_orders(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    at("2026-10-10 19:00")
    await boundaries.save(db, FAM, {"login_codes": {ELDER: "self"}}, by=SON)
    t = await _order(db, requested_by=SON)
    assert t.details["code_from"] == ELDER
    h = Harness(sessions, {ELDER: "9000011111", SON: "9000022222"})
    await h.tick(); await h.tick()
    person, prompt = h.told[-1]
    assert person == ELDER and "[Login code needed]" in prompt and "enter the mobile number 9000011111" in h.agent.runs[0]["goal"]


async def test_a_caregiver_cancelling_tells_the_care_recipient_who_cancelled(db, at):
    at("2026-10-10 19:00")
    t = await _order(db)
    t.status, t.input_needed = "needs_input", "otp"
    await db.commit()
    ctx = tools.TurnCtx(session=db, host=None, family_id=FAM, elder=E, speaker=S, members=[E, S])
    out, err = await tools.run(ctx, "cancel_task", {"task_id": str(t.id), "reason": "not now"})
    r = json.loads(out)
    assert not err and "nothing was placed" in r["result"] and f"Tell person {ELDER}" in r["tell"] and "Mohan asked to cancel" in r["tell"]
    # Maa cancelling her own order: nobody else to tell
    t2 = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="milk",
                              details={"items": [{"name": "milk"}]})
    t2.status, t2.input_needed = "awaiting_confirm", "confirm"
    await db.commit()
    out, _ = await tools.run(tools.TurnCtx(session=db, host=None, family_id=FAM, elder=E, speaker=E, members=[E, S]),
                             "task_input", {"task_id": str(t2.id), "kind": "confirm", "value": "nahi"})
    assert "tell" not in json.loads(out)


async def test_only_an_approver_sets_who_gives_codes_and_only_to_family_members(db, at):
    at("2026-10-10 19:00")
    await store.save_roster(db, FAM, E, [E, S])
    son = tools.TurnCtx(session=db, host=None, family_id=FAM, elder=E, speaker=S, members=[E, S])
    _, err = await tools.run(son, "set_boundaries", {"login_codes": {ELDER: "stranger"}})
    assert err
    out, err = await tools.run(son, "set_boundaries", {"login_codes": {ELDER: SON}})
    assert not err and "login codes for Shanti's orders go to Mohan" in json.loads(out)["saved"]
    assert boundaries.code_person(await boundaries.get(db, FAM), ELDER, ELDER) == SON
    _, err = await tools.run(tools.TurnCtx(session=db, host=None, family_id=FAM, elder=E, speaker=E, members=[E, S]),
                             "set_boundaries", {"login_codes": {ELDER: "self"}})
    assert err, "the care recipient does not change the family's limits"


async def test_onboarding_saves_who_gives_codes_before_the_household_is_on_file(db, at):
    from fastapi import HTTPException

    from app.api import dash

    at("2026-10-10 19:30")
    body = dash.LoginCodesIn(actor=dash.Actor(id=SON, name="Mohan"), codes={ELDER: SON})
    out = await dash.login_codes_save(FAM, ELDER, body, db)
    assert out["login_codes"] == {ELDER: SON}
    with pytest.raises(HTTPException):
        await dash.login_codes_save(FAM, ELDER, dash.LoginCodesIn(actor=dash.Actor(id=SON, name="K"), codes={ELDER: "stranger"}), db)
    await store.save_roster(db, FAM, E, [E, S])
    with pytest.raises(HTTPException):  # once the household is on file, only an approver changes it
        await dash.login_codes_save(FAM, ELDER, dash.LoginCodesIn(actor=dash.Actor(id=ELDER, name="V"), codes={ELDER: "self"}), db)


class MailHost:
    def __init__(self):
        self.emails = []

    async def call(self, tool, args, **kw):
        if tool == "email_member":
            self.emails.append((args["to"], args["subject"], args["text"]))
            return {"sent": True}
        return {}


async def _waiting_order(db, sessions):
    """An order whose cart waits for Maa's yes (she has not answered)."""
    await store.save_roster(db, FAM, E, [E, S])
    t = await _order(db)
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "confirm"
    t.result = {"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1, "price": "₹289"}], "total": "₹318"}
    t.details = {**t.details, "cart_fp": "fp1"}
    await db.commit()
    told, host = [], MailHost()

    async def notify(f, person, prompt):
        told.append((person, prompt))

    async def tick():
        await runtime.tick(sessions, FakeAgent(), profile_for=lambda t: None, notify=notify, host_for=lambda f: host)

    return t, told, host, tick


async def test_an_order_not_placed_in_20_minutes_asks_the_caregiver_then_emails_then_cancels(db, at, sessions):
    at("2026-10-10 20:00")
    t, told, host, tick = await _waiting_order(db, sessions)
    at("2026-10-10 20:19")
    await tick()
    assert told == [] and host.emails == []
    at("2026-10-10 20:21")
    await tick()
    person, prompt = told[-1]
    assert person == SON and prompt.startswith("[Order needs you]") and "a yes to the cart (total ₹318)" in prompt and "task_input keep" in prompt
    at("2026-10-10 20:24")
    await tick()
    assert host.emails == [] and len(told) == 1, "nothing more within the 5 minutes"
    at("2026-10-10 20:26")
    await tick()
    assert len(host.emails) == 1 and host.emails[0][0] == SON and "reply to saheli on whatsapp" in host.emails[0][2].lower()
    at("2026-10-10 20:31")
    await tick()
    await db.refresh(t)
    assert t.status == "cancelled" and "30 minutes" in t.history[-1]["note"]
    last = {p: m for p, m in told}
    assert "cancelled properly" in last[ELDER] and "cancelled properly" in last[SON]


async def test_keep_trying_or_any_answer_starts_the_20_minutes_again(db, at, sessions):
    at("2026-10-10 20:00")
    t, told, host, tick = await _waiting_order(db, sessions)
    at("2026-10-10 20:21")
    await tick()
    assert told and told[-1][0] == SON
    await runtime.provide_input(db, t, kind="keep", value="yes", by=SON, by_is_elder=False); await db.commit()
    at("2026-10-10 20:35")
    await tick()
    await db.refresh(t)
    assert host.emails == [] and t.status == "awaiting_confirm", "kept: no email, no cancel"
    at("2026-10-10 20:42")
    await tick()
    assert len(told) == 2 and told[-1][1].startswith("[Order needs you]"), "asked again 20 minutes after keep"
    # Maa answers the cart herself: the ladder starts over and her yes places it
    await db.refresh(t)
    out = await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True); await db.commit()
    await db.refresh(t)
    assert "placing" in out and not t.details.get("ladder_asked_at")


async def test_an_order_the_store_may_be_taking_is_never_cancelled_by_the_ladder(db, at, sessions):
    at("2026-10-10 20:00")
    t, told, host, tick = await _waiting_order(db, sessions)
    t.status, t.phase, t.input_needed = "queued", "place", None
    t.details = {**t.details, "place_started": "2026-10-10T14:30:00+00:00"}
    await db.commit()
    at("2026-10-10 20:45")
    await runtime.escalate(sessions, FakeAgent(), lambda *a: None, None)
    await db.refresh(t)
    assert t.status == "queued" and not t.details.get("ladder_asked_at")


# ── a cancel after the order is placed (founder 2026-10-11, and 2026-10-12 02:35 IST): one true status; the care recipient →
# the caregiver is told; a caregiver or a self-care person → the store cancels it (free: no second ask; a fee: one ask) ──
PLACED_IM = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "placed": True, "order_id": "IM-55821",
             "payment_method": "Cash on Delivery", "eta": "17 min", "total": "₹318"}


CANCELLED_OK = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "cancelled": True}


async def _confirmed_order(db, sessions, monkeypatch, *, members=(E, S), place=PLACED_IM, polls=1, cancel=(CANCELLED_OK,)):
    """An order Shanti asked for, with its cart confirmed: the next tick starts placing it."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    await store.save_roster(db, FAM, E, list(members))
    agent = FakeAgent(script={"prepare": [CART], "place": [place], "cancel": list(cancel)}, finish_after_polls=polls)
    told = []

    async def notify(f, person, prompt):
        told.append((person, prompt))

    async def profile_for(t):
        return {"profileId": "prof", "loginPhone": "9000011111"}

    async def tick():
        await runtime.tick(sessions, agent, profile_for=profile_for, notify=notify)

    t = await _order(db)
    for _ in range(4):
        await tick()
        await db.refresh(t)
        if t.status == "awaiting_confirm":
            break
    assert t.input_needed == "confirm"
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    await db.commit()
    return t, agent, told, tick


async def test_the_care_recipient_asking_to_cancel_a_placed_order_hears_the_caregiver_is_told_and_he_is(db, at, sessions, monkeypatch):
    """Live 2026-10-11: Maa asked to cancel a placed order and heard 'cancelled', then 'cancelling is in progress', then
    'placed, arriving in 14 minutes', and a cancel run on the store failed. Now (founder 2026-10-12): it is already placed and
    cannot be cancelled from here; her caregiver is being told, and he is, once, with the order; he can have it cancelled."""
    at("2026-10-11 21:15")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch)
    await tick(); await tick()
    await db.refresh(t)
    assert t.status == "done" and t.result["order_id"] == "IM-55821"
    token = runtime._UPDATE_FOR.set({str(t.id): runtime._stage(t)})  # the plain "placed" update, written before her cancel
    placed_update_still_true = runtime.delivery_check()
    runtime._UPDATE_FOR.reset(token)
    said = await runtime.request_cancel(db, agent, t, by=ELDER, reason="galti se order hua")
    await db.commit()
    assert said.startswith("NOT cancelled: it is ALREADY PLACED on Swiggy Instamart: 1 x Aashirvaad Atta 5kg (order IM-55821)")
    assert f"you are telling person {SON} right now" in said and "Never say cancelled" in said
    await db.refresh(t)
    assert t.status == "done" and t.phase == "place" and t.details["cancel_ask"]["elder"] is True
    assert not await placed_update_still_true(db), "the plain 'placed' update is dropped at delivery: her reply told the truth"
    assert "NOT cancelled; the caregiver was told" in runtime.describe(t)
    before = len(told)
    await tick()
    to_son = [p for who, p in told[before:] if who == SON]
    assert len(to_son) == 1 and to_son[0].startswith("[Order needs you]") and "if they say cancel, call cancel_task" in to_son[0]
    assert "IM-55821" in to_son[0] and "cancel it in the Swiggy Instamart app" in to_son[0]
    again = await runtime.request_cancel(db, agent, t, by=ELDER, reason="galti se order hua")
    await db.commit()
    assert again.startswith("NOT cancelled") and f"person {SON}" in again, "asked again: the same truth"
    await tick()
    assert len([p for who, p in told if who == SON]) == 1, "the caregiver is told once"
    assert not [r for r in agent.runs if r["phase"] == "cancel"], "her own ask runs no cancel on the store"
    # her son then asks Saheli to cancel it: the store is asked and cancels it; she hears too
    await db.refresh(t)
    said = await runtime.request_cancel(db, agent, t, by=SON, reason="Maa does not want it")
    await db.commit()
    assert said.startswith("NOT cancelled yet: it is already PLACED") and "asking Swiggy Instamart to cancel it now" in said
    await tick(); await tick()
    await db.refresh(t)
    assert t.status == "cancelled" and [r for r in agent.runs if r["phase"] == "cancel"]
    to_son = [p for who, p in told if who == SON]
    assert "Cancelled on Swiggy Instamart" in to_son[-1] and f"Also tell person {ELDER}" in to_son[-1]


async def test_the_caregiver_hears_of_the_care_recipients_cancel_once_whatever_the_brain_also_calls(db, at, sessions, monkeypatch):
    """Sims 2026-10-12: Maa asked to cancel a placed order; the brain called cancel_task and also messaged her son, who got
    two messages (and in another, it messaged him instead of calling cancel_task). The task system's "[Order needs you]"
    is the one message: a message or a non-emergency alert to him in that turn is refused; one sent before the cancel
    in the same turn counts as telling him."""
    from app.sim.world import SimHost

    at("2026-10-12 09:10")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch)
    await tick(); await tick()
    await db.refresh(t)
    assert t.status == "done"
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=E, members=[E, S])
    out, err = await tools.run(ctx, "cancel_task", {"task_id": str(t.id), "reason": "galti se"})
    assert not err and json.loads(out)["result"].startswith("NOT cancelled") and ctx.told_by_tasks == {SON}
    out, err = await tools.run(ctx, "send_message", {"to": SON, "text": "Maa wants to cancel the atta order"})
    assert err and "task system is telling this person" in json.loads(out)["refused"]
    out, err = await tools.run(ctx, "alert_caregiver", {"reason": "safety", "issue": "order-cancel", "message": "cancel it", "confidence": 0.9})
    assert err and "telling the caregivers" in json.loads(out)["refused"]
    await db.commit()
    before = len(told)
    await tick()
    assert len([p for who, p in told[before:] if who == SON]) == 1, "the one [Order needs you]"

    # the other order of calls: a message to him first, then cancel_task: that message was the one
    t.details = {**t.details, "cancel_ask": None}
    await db.commit()
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=E, members=[E, S])
    out, err = await tools.run(ctx, "send_message", {"to": SON, "text": "Maa wants to cancel the atta order (IM-55821)"})
    assert not err, out
    await tools.run(ctx, "cancel_task", {"task_id": str(t.id), "reason": "galti se"})
    await db.commit()
    await db.refresh(t)
    assert t.details["cancel_ask"]["state"] == "told_carers" and t.details["cancel_ask"].get("carers_told")
    before = len(told)
    await tick()
    assert not [p for who, p in told[before:] if who == SON], "no second message to him"


async def test_a_self_care_persons_cancel_after_placing_goes_to_the_store(db, at, sessions, monkeypatch):
    """Sim 2026-10-12 (Mumbai, self-care): she asked to cancel her own placed order and heard 'cannot be cancelled from
    here; I told the family'. Someone ordering for themselves (no caregiver) has the store cancel it."""
    at("2026-10-12 09:20")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch, members=(E,))
    await tick(); await tick()
    await db.refresh(t)
    said = await runtime.request_cancel(db, agent, t, by=ELDER, reason="chukun zali")
    await db.commit()
    assert "asking Swiggy Instamart to cancel it now" in said
    await tick(); await tick()
    await db.refresh(t)
    assert t.status == "cancelled" and [r for r in agent.runs if r["phase"] == "cancel"]
    assert told[-1][0] == ELDER and "Cancelled on Swiggy Instamart" in told[-1][1]


async def test_a_caregiver_asking_has_the_store_cancel_it_and_a_free_cancel_is_not_asked_again(db, at, sessions, monkeypatch):
    """Founder 2026-10-12 (Tata 1mg): 'after cancellation it asked me to confirm' for a ₹0 fee. A caregiver's cancel goes to the
    store; a free one goes through without a second question, and the one result follows."""
    at("2026-10-11 21:20")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch, cancel=({**CANCELLED_OK, "cancelled": False, "cancel_fee": "₹0"}, CANCELLED_OK))
    await tick(); await tick()
    await db.refresh(t)
    said = await runtime.request_cancel(db, agent, t, by=SON, reason="not needed")
    await db.commit()
    assert said.startswith("NOT cancelled yet") and "telling person" not in said
    before = len(told)
    for _ in range(4):
        await tick()
    await db.refresh(t)
    assert t.status == "cancelled" and len([r for r in agent.runs if r["phase"] == "cancel"]) == 2
    after = [p for who, p in told[before:]]
    assert len(after) == 1 and "Cancelled on Swiggy Instamart" in after[0] and "costs" not in after[0], "one result, no fee question"
    assert not [p for _, p in told if p.startswith("[Order needs you]")]
    from app.tasks import aftercare

    assert await aftercare.loop_for(db, t) is None, "cancelled: no 'has it come?' left open (live 2026-10-12)"


async def test_a_real_cancellation_fee_is_asked_once_with_its_amount(db, at, sessions, monkeypatch):
    at("2026-10-11 21:22")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch, cancel=({**CANCELLED_OK, "cancelled": False, "cancel_fee": "₹25"}, CANCELLED_OK))
    await tick(); await tick()
    await db.refresh(t)
    await runtime.request_cancel(db, agent, t, by=SON, reason="not needed")
    await db.commit()
    await tick(); await tick()
    await db.refresh(t)
    assert (t.status, t.input_needed) == ("needs_input", "fee") and "cancelling costs ₹25" in told[-1][1]
    assert told[-1][0] == SON, "the one who asked to cancel is asked"
    assert "asking Swiggy Instamart to cancel it with the fee of ₹25" in await runtime.provide_input(db, t, kind="fee", value="yes", by=SON, by_is_elder=False)
    await db.commit()
    await tick(); await tick()
    await db.refresh(t)
    assert t.status == "cancelled" and "a cancellation fee of ₹25, as they agreed" in told[-1][1]


async def test_a_store_that_will_not_cancel_sends_a_caregiver_or_self_care_person_to_its_app(db, at, sessions, monkeypatch):
    at("2026-10-11 21:24")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch, cancel=({**CANCELLED_OK, "cancelled": False, "problem": "the order is already packed"},))
    await tick(); await tick()
    await db.refresh(t)
    await runtime.request_cancel(db, agent, t, by=SON, reason="not needed")
    await db.commit()
    await tick(); await tick()
    await db.refresh(t)
    assert t.status == "done" and t.details["cancel_ask"]["state"] == "refused"
    assert "Swiggy Instamart would not cancel it (the order is already packed)" in told[-1][1] and "cancel it in the Swiggy Instamart app" in told[-1][1]
    assert not [p for _, p in told if p.startswith("[Order needs you]")]
    again = await runtime.request_cancel(db, agent, t, by=SON, reason="please")
    assert again.startswith("NOT cancelled: Swiggy Instamart would not cancel it"), "asked again: the same truth, no new run"
    # self care: the person's own ask goes to the store too (nobody else to tell)
    t2, agent2, told2, tick2 = await _confirmed_order(db, sessions, monkeypatch, members=(E,))
    await tick2(); await tick2()
    await db.refresh(t2)
    said = await runtime.request_cancel(db, agent2, t2, by=ELDER, reason="not needed")
    assert said.startswith("NOT cancelled yet") and "asking Swiggy Instamart to cancel it now" in said


async def test_a_cancel_while_it_is_being_placed_gets_one_true_outcome(db, at, sessions, monkeypatch):
    at("2026-10-11 21:25")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch, polls=2)
    await tick()
    await db.refresh(t)
    assert t.status == "running" and t.phase == "place"
    said = await runtime.request_cancel(db, agent, t, by=ELDER, reason="ruko")
    await db.commit()
    assert said.startswith("NOT cancelled yet: the order is being placed on Swiggy Instamart right now and that cannot be stopped")
    assert "never say cancelled" in said and "the caregiver is told at once" in said
    await tick(); await tick()
    await db.refresh(t)
    to_maa = [p for who, p in told if who == ELDER]
    assert t.status == "done" and "it went through: it is PLACED on Swiggy Instamart: 1 x Aashirvaad Atta 5kg (order IM-55821" in to_maa[-1]
    assert f"you are telling person {SON} right now" in to_maa[-1]
    assert len([p for who, p in told if who == SON]) == 1 and not [r for r in agent.runs if r["phase"] == "cancel"]


async def test_a_caregivers_cancel_while_it_is_being_placed_has_the_store_cancel_it_once_placed(db, at, sessions, monkeypatch):
    at("2026-10-11 21:27")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch, polls=2)
    await tick()
    await db.refresh(t)
    said = await runtime.request_cancel(db, agent, t, by=SON, reason="ruko")
    await db.commit()
    assert "Swiggy Instamart is asked to cancel it, and ONE task update comes with the result" in said
    before = len(told)
    for _ in range(6):
        await tick()
    await db.refresh(t)
    after = [p for who, p in told[before:]]
    assert t.status == "cancelled" and len(after) == 1 and "Cancelled on Swiggy Instamart" in after[0], "no 'placed' update in between"


async def test_a_cancel_while_placing_that_did_not_go_through_says_nothing_was_ordered(db, at, sessions, monkeypatch):
    at("2026-10-11 21:30")
    nope = {"logged_in": True, "needs_otp": False, "blocked": False, "placed": False, "pressed_place": False, "problem": "the address page did not load"}
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch, place=nope, polls=2)
    await tick()
    await db.refresh(t)
    assert t.status == "running" and t.phase == "place"
    await runtime.request_cancel(db, agent, t, by=ELDER, reason="ruko")
    await db.commit()
    await tick(); await tick()
    await db.refresh(t)
    assert t.status == "failed" and "it did NOT go through, so nothing was ordered" in told[-1][1]
    assert not [p for who, p in told if who == SON]


async def test_an_order_that_may_have_gone_through_is_never_called_failed_when_they_ask_to_cancel(db, at, sessions, monkeypatch):
    at("2026-10-11 21:35")
    t, agent, told, tick = await _confirmed_order(db, sessions, monkeypatch)
    t.status, t.details = "failed", {**t.details, "place_unclear": True}
    await db.commit()
    said = await runtime.request_cancel(db, agent, t, by=ELDER, reason="ruko")
    await db.commit()
    assert said.startswith("NOT cancelled: it is not clear whether it went through") and "it may already be placed" in said
    await tick()
    to_son = [p for who, p in told if who == SON]
    assert len(to_son) == 1 and "may have gone through" in to_son[0]


async def test_every_stores_options_waiting_for_one_pick_get_one_ask_and_one_cancel(db, at, sessions):
    """The options of every store wait for one pick: after 20 minutes the caregiver is asked once (not once per store), and
    after 30 the whole comparison is cancelled together."""
    at("2026-10-11 20:00")
    await store.save_roster(db, FAM, E, [E, S])
    group = []
    for service in ("blinkit", "instamart", "zepto"):
        t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=service, kind="order", goal="atta",
                                 details={"items": [{"name": "atta"}], "compare": "g-ladder", "compare_announced": True,
                                          "compare_seen": "2026-10-11T14:30:00+00:00"})
        t.status, t.phase, t.input_needed = "awaiting_confirm", "browse", "go"
        group.append(t)
    await db.commit()
    told = []

    async def notify(f, person, prompt):
        told.append((person, prompt))

    at("2026-10-11 20:21")
    await runtime.escalate(sessions, FakeAgent(), notify, None)
    asks = [p for who, p in told if p.startswith("[Order needs you]")]
    assert len(asks) == 1 and "their pick from the options found" in asks[0]
    at("2026-10-11 20:26")
    await runtime.escalate(sessions, FakeAgent(), notify, None)
    at("2026-10-11 20:31")
    await runtime.escalate(sessions, FakeAgent(), notify, None)
    for t in group:
        await db.refresh(t)
    assert {t.status for t in group} == {"cancelled"}
    assert len([p for who, p in told if who == ELDER and "cancelled properly" in p]) == 1
