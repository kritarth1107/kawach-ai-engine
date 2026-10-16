"""Whom an order or a ride is for (founder 2026-10-12, from his mother's phone: "the Karachi Bakery biscuits was the order
of the caregiver, not the elder, so what is this, it's confusing"). His own order, from his own chat, to his own home, was
recorded as his mother's (every order used to name the care recipient), and she was asked twice whether it came. Now the
brain names the person on start_task (no default), the delivery place is checked against whose home it is, and everything
after (the care record checks, the login code, approvals, cancels, who hears about trouble, the follow-up) follows it."""

import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import boundaries, store
from app.sim.agent import CART, FakeAgent
from app.sim.world import SimHost
from app.tasks import aftercare, runtime

FAM, ELDER, SON, DAU = "fam-fw", "elder-fw", "son-fw", "dau-fw"
E = {"id": ELDER, "name": "Kamla Devi", "role": "elder"}
S = {"id": SON, "name": "Ravi", "role": "primary caregiver"}
D = {"id": DAU, "name": "Asha", "role": "co caregiver"}
MAA_HOME = {"addressId": "p-maa", "nickname": "Ghar", "full": "12 Sample Colony, Lucknow 226001", "pincode": "226001", "lat": 26.85,
            "lng": 80.95, "defaultFor": [ELDER]}
HIS_HOME = {"addressId": "p-son", "nickname": "Indore flat", "full": "C-5 Sample Park, Indore 452001", "pincode": "452001",
            "lat": 22.72, "lng": 75.86, "defaultFor": [SON]}


class Book(SimHost):
    """The family's address book: Maa's home in Lucknow, her son's in Indore; each person's default is their own home."""

    def __init__(self):
        super().__init__()
        self.place_calls: list[tuple[str, str]] = []  # (words, the person whose places were looked in)

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "list_places":
            return {"places": [{"name": "Ghar", "address": MAA_HOME["full"], "isDefault": subject_id == ELDER},
                               {"name": "Indore flat", "address": HIS_HOME["full"], "isDefault": subject_id == SON}]}
        if tool == "delivery_place":
            words = str(args.get("words") or "")
            self.place_calls.append((words, subject_id))
            if not words:
                return {**(HIS_HOME if subject_id == SON else MAA_HOME), "matched": None}
            hit = {"Ghar": MAA_HOME, "Indore flat": HIS_HOME}.get(words)
            return {**hit, "matched": True} if hit else {"matched": False, "saved": ["Ghar (Lucknow)", "Indore flat (Indore)"]}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


@pytest.fixture(autouse=True)
def place_words(monkeypatch):
    from app.tasks import matcher

    async def which(words, places):  # the AI's answer, faked: a place by its name; "home" → the person's own default place
        w = words.lower()
        if "indore" in w:
            return "Indore flat"
        if "ghar" in w:
            return "Ghar"
        return next((p["name"] for p in places if p.get("isDefault")), None) if "home" in w else None

    monkeypatch.setattr(matcher, "which_place", which)


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


def ctx(db, speaker, host, ref="m1"):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=E, speaker=speaker, members=[E, S, D], message_ref=ref)


ORDER = {"kind": "order", "goal": "Karachi Bakery fruit biscuits", "items": [{"name": "Karachi Bakery fruit biscuits"}], "service": "instamart"}


async def test_whom_it_is_for_is_never_a_default(db, at):
    """Not clear whom it is for: asked in the same line as the address (the place of the person asking), never the care
    recipient by default; "for me" goes ahead without a second address question (sim 2026-10-12: the brain asked whom by
    itself, then start_task asked the address again)."""
    at("2026-10-12 15:00")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    first = await tools.start_task(ctx(db, S, host), dict(ORDER))
    assert first["status"] == "not started yet" and first["confirm_address"].startswith("Indore flat"), "his own place, not Maa's"
    assert host.place_calls[-1] == ("", SON) and first["for_whom"] == "not known yet: ask"
    assert "In the same line, ask whom it is for: them or Kamla or Asha." in first["next"]
    assert not await runtime.live_tasks(db, FAM)
    out = await tools.start_task(ctx(db, S, host, ref="m2"), {**ORDER, "for_whom": SON})
    assert out["status"] == "started", out
    t = (await runtime.live_tasks(db, FAM))[0]
    assert (t.subject_id, t.details["limits"]["place"]["nickname"]) == (SON, "Indore flat")


async def test_whom_still_not_said_is_asked_and_a_ride_is_never_guessed(db, at):
    at("2026-10-12 15:01")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    await tools.start_task(ctx(db, S, host), dict(ORDER))
    out, err = await tools.run(ctx(db, S, host, ref="m2"), "start_task", {**ORDER, "for_whom": ""})
    r = json.loads(out)
    assert err and f"{SON} (Ravi)" in r["refused"] and "never assume the care recipient" in r["refused"], "the address was answered, not whom"
    out, err = await tools.run(ctx(db, S, host, ref="m3"), "start_task", {**ORDER, "for_whom": "Ravi"})
    assert err, "a name is not an id"
    out, err = await tools.run(ctx(db, S, host, ref="m4"), "start_task", {"kind": "ride", "goal": "Cab to the station",
                                                                        "pickup": "home", "drop": "the station", "service": "ola"})
    assert err and "for_whom" in json.loads(out)["refused"]
    assert not await runtime.live_tasks(db, FAM)


async def test_a_missing_goal_or_kind_is_asked_for_not_a_crash(db, at):
    """Lang eval 2026-10-12: start_task without goal reached the address question and failed with a KeyError."""
    at("2026-10-12 15:04")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    first = await tools.start_task(ctx(db, S, host), {k: v for k, v in ORDER.items() if k != "goal"})
    assert first["status"] == "not started yet" and first["confirm_address"]
    out, err = await tools.run(ctx(db, S, host, ref="m2"), "start_task", {"kind": "order", "for_whom": SON, "service": "instamart"})
    assert err and "Give goal" in json.loads(out)["refused"]
    out, err = await tools.run(ctx(db, S, host, ref="m3"), "start_task", {"goal": "milk", "for_whom": SON})
    assert err and "Give kind" in json.loads(out)["refused"]


async def test_for_someone_living_elsewhere_their_own_place_is_confirmed_once(db, at):
    """Asked at his place, he says it is for Maa (her own home is elsewhere): her place is confirmed once, then it goes."""
    at("2026-10-12 15:02")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    await tools.start_task(ctx(db, S, host), dict(ORDER))
    again = await tools.start_task(ctx(db, S, host, ref="m2"), {**ORDER, "for_whom": ELDER})
    assert again["status"] == "not started yet" and again["confirm_address"].startswith("Ghar: 12 Sample Colony")
    assert "Say it is for Kamla" in again["next"]
    out = await tools.start_task(ctx(db, S, host, ref="m3"), {**ORDER, "for_whom": ELDER})
    assert out["status"] == "started", out
    t = (await runtime.live_tasks(db, FAM))[0]
    assert (t.subject_id, t.details["limits"]["place"]["nickname"]) == (ELDER, "Ghar")


async def test_no_question_about_whose_home_for_someone_with_no_home_of_their_own(db, at):
    """His sister lives with Maa (no place of her own on record): her own order to Maa's home is not asked about; and a
    household of one is never asked whom it is for."""
    at("2026-10-12 15:03")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    first = await tools.start_task(ctx(db, D, host), {**ORDER, "for_whom": DAU})
    assert first["confirm_address"].startswith("Ghar") and "is the home of" not in first["next"] and "Say it is for" not in first["next"]
    out = await tools.start_task(ctx(db, D, host, ref="m2"), {**ORDER, "for_whom": DAU})
    assert out["status"] == "started", out
    assert (await runtime.live_tasks(db, FAM))[0].subject_id == DAU
    me = {"id": "self-fw", "name": "Sunita", "role": "elder"}
    await store.save_roster(db, "fam-one", me, [me])
    one = tools.TurnCtx(session=db, host=host, family_id="fam-one", elder=me, speaker=me, members=[me], message_ref="o1")
    first = await tools.start_task(one, dict(ORDER))
    assert "ask whom it is for" not in first["next"] and first["for_whom"] == "Sunita"


async def test_a_caregivers_own_order_is_his_from_his_own_places_to_the_follow_up(db, at, sessions, monkeypatch):
    """His own biscuits: his default place (not Maa's), his care record (not her allergy list), his login number, and the
    follow-up asks him, never her, though she talks to Saheli."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    at("2026-10-12 15:05")
    await store.save_roster(db, FAM, E, [E, S, D])
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="allergy", key="allergy:biscuits", value={"allergen": "biscuits"},
                           text="Allergic to biscuits", source_kind="caregiver_said")
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="ram ram", meta={})
    await boundaries.save(db, FAM, {"login_codes": {ELDER: SON}}, by=SON)
    await db.commit()
    host = Book()
    first = await tools.start_task(ctx(db, S, host), {**ORDER, "for_whom": SON})
    assert first["confirm_address"].startswith("Indore flat: C-5 Sample Park"), "his own default place, not Maa's"
    assert host.place_calls[-1] == ("", SON)
    out = await tools.start_task(ctx(db, S, host, ref="m2"), {**ORDER, "for_whom": SON})
    assert out["status"] == "started", out
    t = (await runtime.live_tasks(db, FAM))[0]
    assert (t.subject_id, t.requested_by, t.details["for_set"]) == (SON, SON, True)
    assert t.details["limits"]["allergies"] == [], "her allergy list does not stop his own order"
    assert t.details["code_from"] == SON
    assert f"person {SON}'s own" in runtime.describe(t)
    # placed: the follow-up asks him
    t.status, t.phase = "done", "place"
    t.result = {"placed": True, "order_id": "IM-1", "total": "₹170", "eta": "12 min", "items": [{"name": "Karachi Bakery Fruit Biscuits", "qty": 1}]}
    await runtime.delivery_followup(db, t)
    await db.commit()
    loop = await aftercare.loop_for(db, t)
    assert loop.owner_id == SON


async def test_the_founders_case_his_own_home_named_for_her_order_is_asked_once(db, at):
    """The brain guessed the order was for Maa; he answered the address question with "my Indore flat" (his own home). Whom
    it is for is asked once; his answer goes ahead, and the order is his."""
    at("2026-10-12 15:10")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    first = await tools.start_task(ctx(db, S, host), {**ORDER, "for_whom": ELDER})
    assert first["confirm_address"].startswith("Ghar: 12 Sample Colony, Lucknow"), "her default place"
    assert "Say it is for Kamla" in first["next"], "the question says whom it is for, so he can correct it"
    out = await tools.start_task(ctx(db, S, host, ref="m2"), {**ORDER, "for_whom": ELDER, "area": "my indore flat"})
    assert out["status"] == "not started yet" and out["check_for_whom"]
    assert "Indore flat is the home of Ravi, not Kamla's" in out["next"] and "for_whom set to the person they say" in out["next"]
    assert not await runtime.live_tasks(db, FAM)
    # "for me": his own order, to his own home
    out = await tools.start_task(ctx(db, S, host, ref="m3"), {**ORDER, "for_whom": SON, "area": "my indore flat"})
    assert out["status"] == "started"
    t = (await runtime.live_tasks(db, FAM))[0]
    assert (t.subject_id, t.details["limits"]["place"]["nickname"]) == (SON, "Indore flat")


async def test_for_her_to_his_home_on_purpose_goes_ahead_after_the_one_question(db, at):
    """Maa is staying with him: for her, to his home. Asked once; "yes, for Maa" goes ahead."""
    at("2026-10-12 15:15")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    first = await tools.start_task(ctx(db, S, host), {**ORDER, "for_whom": ELDER, "area": "my indore flat"})
    assert "Indore flat is the home of Ravi, not Kamla's: ask in the same line" in first["next"], "said in the address question itself"
    out = await tools.start_task(ctx(db, S, host, ref="m2"), {**ORDER, "for_whom": ELDER, "area": "my indore flat"})
    assert out["status"] == "started", out
    t = (await runtime.live_tasks(db, FAM))[0]
    assert t.subject_id == ELDER and t.requested_by == SON
    t.status, t.phase, t.result = "done", "place", {"placed": True, "order_id": "IM-2", "items": [{"name": "Biscuits", "qty": 1}]}
    await runtime.delivery_followup(db, t)
    await db.commit()
    assert (await aftercare.loop_for(db, t)).owner_id == SON, "it went to his home: he is asked whether it came, not Maa"


async def test_a_caregivers_own_ride_from_home_is_from_his_home(db, at):
    at("2026-10-12 15:20")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    c = ctx(db, S, host)
    c.location_request = False
    out = await tools.start_task(c, {"kind": "ride", "goal": "Cab to the airport", "pickup": "home", "drop": "Indore airport",
                                     "service": "ola", "for_whom": SON})
    out = out if out.get("status") == "started" else await tools.start_task(ctx(db, S, host, ref="m2"), {
        "kind": "ride", "goal": "Cab to the airport", "pickup": "home", "drop": "Indore airport", "service": "ola", "for_whom": SON})
    t = (await runtime.live_tasks(db, FAM))[0]
    assert t.subject_id == SON and t.details["pickup_place"]["full"] == HIS_HOME["full"], "home = the rider's own home"
    assert ("Ghar", SON) not in host.place_calls and any(subj == SON for _, subj in host.place_calls)


async def test_his_own_orders_trouble_and_cancel_are_his(db, at, sessions, monkeypatch):
    """Who hears about a stuck or late order, and how a cancel is handled, follow whom it is for: his own order is not
    treated as the care recipient's (no "caregiver told" path, no other caregiver asked)."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "off")
    at("2026-10-12 15:30")
    await store.save_roster(db, FAM, E, [E, S, D])
    t = await runtime.create(db, family_id=FAM, subject_id=SON, requested_by=SON, service="instamart", kind="order", goal="biscuits",
                             details={"items": [{"name": "biscuits"}], "for_set": True})
    await db.commit()
    assert await runtime._carers(db, t) == [SON], "his own order: him, not his sister, never Maa"
    maa = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=SON, service="zepto", kind="order", goal="milk",
                               details={"items": [{"name": "milk"}], "for_set": True})
    assert set(await runtime._carers(db, maa)) == {SON}, "Maa's order: the approvers (the primary caregiver)"
    t.status, t.phase = "done", "place"
    t.result = {"placed": True, "order_id": "IM-3", "items": [{"name": "biscuits", "qty": 1}]}
    await db.commit()
    said = await runtime.request_cancel(db, FakeAgent(), t, by=SON, reason="not needed")
    assert "asking Swiggy Instamart to cancel it now" in said and "telling person" not in said


async def test_a_self_care_persons_big_order_is_not_left_waiting_for_an_approver_who_does_not_exist(db, at, monkeypatch):
    """Found on the way: someone ordering for themselves with nobody else in the family went over the care recipient's own
    limit (₹1500) and waited for an approval nobody could give."""
    at("2026-10-12 15:40")
    me = {"id": "self-fw", "name": "Sunita", "role": "elder"}
    await store.save_roster(db, "fam-self", me, [me])
    t = await runtime.create(db, family_id="fam-self", subject_id="self-fw", requested_by="self-fw", service="blinkit", kind="order",
                             goal="groceries", details={"items": [{"name": "rice"}], "for_set": True})
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "confirm"
    t.result = {**CART, "total": "₹2,480"}
    t.details = {**t.details, "cart_fp": "fp1"}
    await db.commit()
    out = await runtime.provide_input(db, t, kind="confirm", value="yes", by="self-fw", by_is_elder=True)
    assert out.startswith("confirmed; placing it now"), out


async def test_a_caregivers_own_order_leaves_nothing_in_the_elders_brief(db, at):
    """His own snacks and his own cab: the asks are on his care record. Her daily brief is built from her events only, so
    it does not say she was asked to confirm a delivery, or a pickup, for something that is his."""
    at("2026-10-16 21:34")
    await store.save_roster(db, FAM, E, [E, S, D])
    host = Book()
    first = await tools.start_task(ctx(db, S, host), {**ORDER, "for_whom": SON, "goal": "evening snacks"})
    assert first["status"] == "not started yet" and first["confirm_address"]
    ride = await tools.start_task(ctx(db, S, host, ref="r1"), {
        "kind": "ride", "goal": "Cab from the cafe", "pickup": "the cafe", "drop": "home", "service": "ola", "for_whom": SON})
    assert ride.get("ask_location") == "pickup"
    blocked = await tools.start_task(ctx(db, S, host, ref="x1"), {
        "kind": "order", "goal": "a tobacco item", "for_whom": SON,
        "items": [{"name": "a tobacco item", "restricted": "tobacco"}]})
    assert blocked["status"] == "not started"
    kinds = ["order_address_asked", "order_for_whom_asked", "order_refused", "ride_pin_asked"]
    assert await store.events(db, FAM, ELDER, kinds=kinds) == []
    his = {e.kind for e in await store.events(db, FAM, SON, kinds=kinds)}
    assert his == {"order_address_asked", "ride_pin_asked", "order_refused"}
    from app.care.brief import digest

    text = await digest(db, host, FAM, E, "2026-10-16")
    assert "order_address_asked" not in text and "evening snacks" not in text and "ride_pin_asked" not in text and "order_refused" not in text
