"""Saved delivery places from chat (order lab 2026-10-10: Saheli had no tool to save a new delivery place; dashboard only).
The address book rules live in the backend; here: who may call what, and that an order to a place that is not saved asks
instead of going to the default place."""

import json

from app.brain import persona, tools
from app.brain.host import WRITE_TOOLS

FAM = "fam-places"
ELDER = {"id": "e-pl", "name": "Vasundara", "role": "elder"}
SON = {"id": "c-pl", "name": "Kritarth", "role": "primary caregiver"}
NIECE = {"id": "v-pl", "name": "Asha", "role": "view only"}

HOME = {"addressId": "a1", "nickname": "Home", "pincode": "492001", "full": "504, Sunita Park, Raipur 492001", "lat": 21.2, "lng": 81.6,
        "saved": ["Home (Raipur 492001)", "Vish's Home (Bangalore 560092)"]}


class Host:
    def __init__(self):
        self.calls = []

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args, kw))
        if tool == "delivery_place":
            words = (args.get("words") or "").lower()
            if not words:
                return {**HOME, "matched": None}
            if "vish" in words:
                return {**HOME, "addressId": "a2", "nickname": "Vish's Home", "full": "74, Amruthahalli, Bangalore 560092", "matched": True}
            return {**HOME, "matched": False}
        return {"ok": True, "tool": tool}


def c(db, speaker, host, ref="m1"):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker=speaker, members=[ELDER, SON, NIECE], message_ref=ref)


async def test_the_person_or_a_caregiver_saves_a_place_from_chat(db, at):
    at("2026-10-10 18:00")
    host = Host()
    out, err = await tools.run(c(db, ELDER, host), "save_place", {"address": "74, Amruthahalli, Bangalore 560092", "name": "Vish's Home",
                                                                  "receiver_name": "Vish", "receiver_phone": "9980531439"})
    assert not err, out
    tool, args, kw = host.calls[-1]
    assert tool == "save_place" and args["name"] == "Vish's Home" and args["receiver_phone"] == "9980531439"
    assert kw["subject_id"] == ELDER["id"] and kw["actor_id"] == ELDER["id"]
    out, err = await tools.run(c(db, SON, host), "save_place", {"place": "Vish's Home", "make_default": True})
    assert not err and host.calls[-1][1] == {"place": "Vish's Home", "make_default": True} and host.calls[-1][2]["actor_id"] == SON["id"]
    _, err = await tools.run(c(db, NIECE, host), "save_place", {"address": "1, MG Road, Raipur 492001", "name": "X"})
    assert err and len(host.calls) == 2
    system = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker={"id": "saheli", "role": "system"}, members=[ELDER, SON])
    _, err = await tools.run(system, "save_place", {"address": "1, MG Road, Raipur 492001", "name": "X"})
    assert err


async def test_only_a_caregiver_removes_a_place_and_the_yes_is_passed_on(db, at):
    at("2026-10-10 18:00")
    host = Host()
    _, err = await tools.run(c(db, ELDER, host), "remove_place", {"place": "Home", "confirmed": True})
    assert err and not host.calls
    out, err = await tools.run(c(db, SON, host), "remove_place", {"place": "Home"})
    assert not err and host.calls[-1][1] == {"place": "Home", "confirmed": False}
    await tools.run(c(db, SON, host), "remove_place", {"place": "Home", "confirmed": True})
    assert host.calls[-1][1]["confirmed"] is True
    out, err = await tools.run(c(db, NIECE, host), "places", {})
    assert not err and host.calls[-1][0] == "list_places"


async def test_an_order_to_a_place_that_is_not_saved_asks_for_it(db, at):
    at("2026-10-10 18:05")
    host = Host()
    out, err = await tools.run(c(db, ELDER, host), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk",
                                                                   "items": [{"name": "milk"}], "area": "Clinic"})
    r = json.loads(out)
    assert not err and r["unknown_place"] == "Clinic" and "save_place" in r["next"] and "Vish's Home" in r["saved_places"]
    assert "confirm_address" not in r
    out, _ = await tools.run(c(db, ELDER, host), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk",
                                                                 "items": [{"name": "milk"}], "area": "Vish ka ghar"})
    assert json.loads(out)["confirm_address"].startswith("Vish's Home: 74, Amruthahalli")
    at("2026-10-10 18:30")  # a later order asks for its address again
    out, _ = await tools.run(c(db, ELDER, host, ref="m3"), "start_task", {"kind": "order", "service": "instamart", "goal": "bread",
                                                                           "items": [{"name": "bread"}]})
    assert json.loads(out)["confirm_address"].startswith("Home: 504")


def test_place_writes_are_not_run_in_shadow_and_saheli_knows_the_rule():
    assert {"save_place", "remove_place"} <= WRITE_TOOLS
    assert "save_place" in persona.PERSONA
    names = {s.name for s in tools.specs()}
    assert {"places", "save_place", "remove_place"} <= names


async def test_a_named_place_is_never_ordered_to_when_the_saved_places_cannot_be_read(db, at):
    """Shadow check 2026-10-10: the place look-up failed, the address step said "near Clinic", and the next turn started
    the order with no saved place (it would have gone to the store account's own address)."""
    at("2026-10-10 18:40")

    class Down(Host):
        async def call(self, tool, args, **kw):
            if tool == "delivery_place":
                raise RuntimeError("backend down")
            return await super().call(tool, args, **kw)

    host = Down()
    out, err = await tools.run(c(db, ELDER, host), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk",
                                                                   "items": [{"name": "milk"}], "area": "Clinic"})
    r = json.loads(out)
    assert not err and r["status"] == "not started yet" and "try again" in r["next"] and "confirm_address" not in r
    out, _ = await tools.run(c(db, ELDER, host, ref="m2"), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk",
                                                                           "items": [{"name": "milk"}], "area": "Clinic"})
    assert "task_id" not in json.loads(out), "still nothing started on the next turn"


async def test_a_place_named_after_the_address_was_confirmed_must_still_be_saved(db, at):
    """Shadow check 2026-10-10: within 10 minutes of a confirmed address, "send it to Clinic" skipped the check and the
    order would have gone to the default place."""
    at("2026-10-10 19:00")
    host = Host()
    await tools.run(c(db, ELDER, host), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk", "items": [{"name": "milk"}]})
    out, err = await tools.run(c(db, ELDER, host, ref="m2"), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk",
                                                                             "items": [{"name": "milk"}], "area": "Clinic"})
    r = json.loads(out)
    assert not err and r.get("unknown_place") == "Clinic" and "task_id" not in r
    out, _ = await tools.run(c(db, ELDER, host, ref="m3"), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk",
                                                                           "items": [{"name": "milk"}], "area": "Vish ka ghar"})
    r = json.loads(out)
    assert r.get("task_id"), r
    from app.tasks.models import Task
    task = await db.get(Task, __import__("uuid").UUID(r["task_id"]))
    assert ((task.details or {}).get("limits") or {}).get("place", {}).get("nickname") == "Vish's Home"


async def test_the_store_they_named_stays_with_the_order_after_the_address_question(db, at):
    """Sim 2026-10-11: "Blinkit se …" → address asked → "haan ghar": the second start_task left the store out and three stores
    were searched instead of Blinkit."""
    at("2026-10-11 10:30")
    host = Host()
    out, _ = await tools.run(c(db, ELDER, host), "start_task", {"kind": "order", "service": "blinkit", "goal": "milk", "items": [{"name": "milk"}]})
    assert "service blinkit" in json.loads(out)["next"]
    out, err = await tools.run(c(db, ELDER, host, ref="m2"), "start_task", {"kind": "order", "category": "grocery", "goal": "milk",
                                                                             "items": [{"name": "milk"}]})
    r = json.loads(out)
    assert not err and r.get("task_id") and "looking_on" not in r, r
    from app.tasks.models import Task
    task = await db.get(Task, __import__("uuid").UUID(r["task_id"]))
    assert task.service == "blinkit"
