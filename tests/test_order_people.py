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
E = {"id": ELDER, "name": "Vasundara", "role": "elder"}
S = {"id": SON, "name": "Kritarth", "role": "primary caregiver"}


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
    assert "login codes for Vasundara's orders go to Kritarth" in boundaries.describe(p, {ELDER: "Vasundara", SON: "Kritarth"})


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
    assert not err and "nothing was placed" in r["result"] and f"Tell person {ELDER}" in r["tell"] and "Kritarth asked to cancel" in r["tell"]
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
    assert not err and "login codes for Vasundara's orders go to Kritarth" in json.loads(out)["saved"]
    assert boundaries.code_person(await boundaries.get(db, FAM), ELDER, ELDER) == SON
    _, err = await tools.run(tools.TurnCtx(session=db, host=None, family_id=FAM, elder=E, speaker=E, members=[E, S]),
                             "set_boundaries", {"login_codes": {ELDER: "self"}})
    assert err, "the care recipient does not change the family's limits"


async def test_onboarding_saves_who_gives_codes_before_the_household_is_on_file(db, at):
    from fastapi import HTTPException

    from app.api import dash

    at("2026-10-10 19:30")
    body = dash.LoginCodesIn(actor=dash.Actor(id=SON, name="Kritarth"), codes={ELDER: SON})
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
    await runtime.provide_input(db, t, kind="keep", value="haan", by=SON, by_is_elder=False); await db.commit()
    at("2026-10-10 20:35")
    await tick()
    await db.refresh(t)
    assert host.emails == [] and t.status == "awaiting_confirm", "kept: no email, no cancel"
    at("2026-10-10 20:42")
    await tick()
    assert len(told) == 2 and told[-1][1].startswith("[Order needs you]"), "asked again 20 minutes after keep"
    # Maa answers the cart herself: the ladder starts over and her yes places it
    await db.refresh(t)
    out = await runtime.provide_input(db, t, kind="confirm", value="haan", by=ELDER, by_is_elder=True); await db.commit()
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
