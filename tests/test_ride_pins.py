"""A ride's pickup as a WhatsApp location (founder 2026-10-12: "ask for coordinates location from WhatsApp ... because it
might miss the exact pickup location"): asked with the "Send location" button unless a saved place already has its exact
point; the pin goes to every ride app as that point, and it is saved to the place they named for next time."""

import json

import pytest

from app.brain import tools
from app.care import store
from app.sim.world import SimHost
from app.specialists.agents import specialist_for
from app.tasks import matcher, runtime

FAM, ELDER, SON = "fam-pin", "elder-pin", "son-pin"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
S = {"id": SON, "name": "Ravi", "role": "primary caregiver"}
HOME = {"nickname": "Ghar", "full": "12 Civil Lines, Raipur 492001", "pincode": "492001", "lat": 21.2500, "lng": 81.6300}


class PlacesHost(SimHost):
    """The family's address book, with how exact Ghar's point is; save_place calls are kept."""

    def __init__(self, precision="area"):
        super().__init__()
        self.precision, self.saves = precision, []

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "list_places":
            return {"places": [{"name": "Ghar", "address": HOME["full"], "isDefault": True}]}
        if tool == "delivery_place":
            if args.get("exact") and args.get("words") == "Ghar":
                return {**HOME, "addressId": "a1", "matched": True, "geoPrecision": self.precision}
            return {"matched": False}
        if tool == "save_place":
            self.saves.append(args)
            return {"ok": True}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


@pytest.fixture(autouse=True)
def home_words(monkeypatch):
    async def which(words, places):
        return "Ghar" if "home" in words.lower() or "ghar" in words.lower() else None  # the AI's answer, faked

    monkeypatch.setattr(matcher, "which_place", which)


async def _ctx(db, host, *, ref, location=None):
    await store.save_roster(db, FAM, E, [E, S])
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=E, speaker=S, members=[E, S], message_ref=ref, location=location)


RIDE = {"kind": "ride", "goal": "Cab to the station", "pickup": "home", "drop": "Raipur railway station", "service": "ola", "for_whom": SON}


async def test_the_pickup_is_asked_as_a_location_then_the_pin_is_used_and_saved_to_home(db, at):
    at("2026-10-12 08:00")
    host = PlacesHost()
    ctx = await _ctx(db, host, ref="wamid.p1")
    out = await tools.start_task(ctx, dict(RIDE))
    assert out["ask_location"] == "pickup" and "Send location" in out["next"] and ctx.location_request
    assert not await runtime.live_tasks(db, FAM), "nothing is searched before the pin"
    # the pin comes in the next message (100 m from where Ghar was thought to be)
    at("2026-10-12 08:01")
    pin = {"lat": 21.2509, "lng": 81.6301, "name": "", "address": ""}
    ctx = await _ctx(db, host, ref="wamid.p2", location=pin)
    out = await tools.start_task(ctx, {**RIDE, "pickup_shared": True})
    t = (await runtime.live_tasks(db, FAM))[0]
    place = t.details["pickup_place"]
    assert (place["lat"], place["lng"], place["precision"], place["pin"]) == (21.2509, 81.6301, "user", True)
    assert host.saves == [{"place": "Ghar", "lat": 21.2509, "lng": 81.6301}] and "Ghar's exact point" in out["pin_saved"]
    t.phase = "prepare"
    goal = specialist_for("ola").goal(t)
    assert "The pickup is the map point 21.250900, 81.630100: set the pickup there" in goal


async def test_a_pin_far_from_the_saved_place_is_used_but_not_saved(db, at):
    at("2026-10-12 09:00")
    host = PlacesHost()
    ctx = await _ctx(db, host, ref="wamid.f1", location={"lat": 21.30, "lng": 81.70, "name": "Office", "address": "Telibandha"})
    out = await tools.start_task(ctx, {**RIDE, "pickup_shared": True})
    t = (await runtime.live_tasks(db, FAM))[0]
    assert t.details["pickup_place"]["full"] == "Office, Telibandha" and not host.saves and "pin_saved" not in out


async def test_a_place_with_its_exact_point_is_not_asked_and_they_may_go_on_without_a_pin(db, at):
    at("2026-10-12 10:00")
    ctx = await _ctx(db, PlacesHost(precision="user"), ref="wamid.u1")
    out = await tools.start_task(ctx, dict(RIDE))
    assert "ask_location" not in out and (await runtime.live_tasks(db, FAM))[0].details["pickup_place"]["full"] == HOME["full"]
    for t in await runtime.live_tasks(db, FAM):
        t.status = "cancelled"
    await db.commit()
    # asked once; they would rather use the saved address: the next message goes ahead with it
    host = PlacesHost()
    ctx = await _ctx(db, host, ref="wamid.u2")
    assert (await tools.start_task(ctx, dict(RIDE)))["ask_location"] == "pickup"
    ctx = await _ctx(db, host, ref="wamid.u3")
    out = await tools.start_task(ctx, dict(RIDE))
    assert out.get("status") == "started" and (await runtime.live_tasks(db, FAM))[0].details["pickup_place"]["full"] == HOME["full"]


async def test_a_pin_shared_a_little_earlier_is_used_and_asking_works_on_scheduled_turns(db, at):
    at("2026-10-12 11:00")
    host = PlacesHost()
    await store.save_roster(db, FAM, E, [E, S])
    await store.add_turn(db, family_id=FAM, thread_id=SON, role="user", text="[location lat=21.25 lng=81.63]", speaker_id=SON,
                         message_ref="wamid.e1", meta={"location": {"lat": 21.25, "lng": 81.63, "name": "", "address": ""}})
    await db.commit()
    at("2026-10-12 11:20")
    ctx = await _ctx(db, host, ref="wamid.e2")
    await tools.start_task(ctx, {**RIDE, "pickup_shared": True})
    assert (await runtime.live_tasks(db, FAM))[0].details["pickup_place"]["lat"] == 21.25
    # a scheduled turn asks with send_message; the message carries the "Send location" button
    sys_ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker={"id": "saheli-scheduler", "role": "system"},
                            members=[E, S])
    out, err = await tools.run(sys_ctx, "send_message", {"to": SON, "text": "Pickup ki location bhejiye", "ask_location": True})
    assert not err and sys_ctx.host.world.sent[-1]["location_request"] is True
    out, err = await tools.run(sys_ctx, "ask_location", {"purpose": "pickup"})
    assert err and "send_message with ask_location" in json.loads(out)["refused"]


async def test_save_place_takes_the_shared_pin(db, at):
    at("2026-10-12 12:00")
    host = PlacesHost()
    ctx = await _ctx(db, host, ref="wamid.s1", location={"lat": 21.2511, "lng": 81.6299, "name": "", "address": ""})
    out, err = await tools.run(ctx, "save_place", {"place": "Ghar", "use_shared_location": True})
    assert not err and host.saves == [{"place": "Ghar", "lat": 21.2511, "lng": 81.6299}]


async def test_a_bare_current_location_gets_its_address_for_the_app(db, at):
    """A "current location" pin has no name or address: its address in words (the backend's reverse geocoding) goes to the
    app with the point, and what the app sets can be checked against it."""

    class ReverseHost(PlacesHost):
        async def call(self, tool, args, *, family_id, subject_id, actor_id):
            if tool == "ride_place" and "lat" in args:
                return {"lat": args["lat"], "lng": args["lng"], "label": "Sample Park", "address": "14, Sample Park Road, Raipur 492001"}
            return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)

    at("2026-10-12 13:00")
    ctx = await _ctx(db, ReverseHost(), ref="wamid.r1", location={"lat": 21.26, "lng": 81.64, "name": "", "address": ""})
    await tools.start_task(ctx, {**RIDE, "pickup_shared": True})
    t = (await runtime.live_tasks(db, FAM))[0]
    place = t.details["pickup_place"]
    assert place["full"] == "14, Sample Park Road, Raipur 492001" and (place["lat"], place["lng"]) == (21.26, 81.64)
    t.phase = "prepare"
    goal = specialist_for("ola").goal(t)
    assert "Set the pickup to exactly '14, Sample Park Road, Raipur 492001'" in goal and "map point 21.260000, 81.640000" in goal
