"""A blocked ride, or a fare they will not pay, offers only ways the look-up actually wrote down."""

from datetime import datetime

from app.brain import tools
from app.care import store, ways
from app.core import clock
from app.sim.world import SimHost
from app.tasks import later, runtime

RAW = """Testpur airport prepaid taxi desk: sedan 1450 rupees, booked at the desk. Door to door.
Airport bus route TP-4 until 01:00, fare 210, cash or UPI on the bus. Stops at the city bus stand, not the house.
A cab on your own app for this route is usually 900.
"""
PROPOSED = """OPTION kind=airport_taxi fare=1450 door=yes route=none pay=at the desk steps=Book at the airport prepaid taxi desk
OPTION kind=airport_bus fare=210 door=no route=TP-4 pay=cash or UPI on the bus steps=Take route TP-4 from the airport bus stop
OPTION kind=own_app fare=900 door=yes route=none pay=in their app steps=Book it in their own cab app
OPTION kind=metro fare=50 door=no route=MADE-UP pay=token steps=Take the metro
"""

FAM, ELDER, SON = "fam-ways", "elder-ways", "son-ways"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
S = {"id": SON, "name": "Ravi", "role": "primary caregiver"}


class _Task:
    def __init__(self, **kw):
        self.kind = "ride"
        self.service = "uber"
        self.phase = "browse"
        self.details = {"pickup": "Terminal 1", "drop": "Home"}
        self.result = {}
        self.input_needed = None
        self.status = "running"
        for k, v in kw.items():
            setattr(self, k, v)


def test_a_fare_the_page_did_not_write_is_dropped_and_one_option_is_marked():
    at = datetime(2026, 10, 16, 0, 30, tzinfo=clock.IST)
    options = ways.options_from(PROPOSED, RAW, at=at, elder=True)
    fares = {o["kind"]: o["fare"] for o in options}
    assert fares == {"airport_taxi": 1450, "airport_bus": 210, "own_app": 900}
    marked = [o for o in options if o.get("recommended")]
    assert len(marked) == 1 and marked[0]["kind"] == "own_app" and marked[0]["why"] == "lowest door-to-door fare at that hour"
    bus = next(o for o in options if o["kind"] == "airport_bus")
    assert bus["safety"] == "An onward ride is needed at that hour."
    said = ways.messages(options)
    assert len(said) == 4 and said[-1] == "Which feels right?"
    assert "₹50" not in " ".join(said) and "MADE-UP" not in " ".join(said)
    assert "₹900" in said[2] and "I'd take this one" in said[2]


def test_in_the_day_the_lowest_fare_is_the_one_marked():
    at = datetime(2026, 10, 16, 11, 0, tzinfo=clock.IST)
    options = ways.options_from(PROPOSED, RAW, at=at, elder=False)
    marked = next(o for o in options if o.get("recommended"))
    assert marked["kind"] == "airport_bus" and marked["fare"] == 210
    assert all(not o.get("safety") for o in options)


def test_a_blocked_ride_says_nothing_was_booked_and_asks_for_the_other_ways():
    status, said = runtime._outcome(_Task(), {"blocked": True, "problem": "security check"})
    assert status == "failed"
    assert "Nothing was booked and nothing was paid" in said
    assert "ways_there" in said and "Terminal 1" in said and "Home" in said
    assert "security check" in said


def test_a_status_page_moves_the_landing_only_when_it_writes_the_time():
    raw = "Flight 6E-204 arrival 00:45. Departure was 22:00."
    assert ways.checked_arrival("ARRIVES=00:45", raw, "6E-204") == "00:45"
    assert ways.checked_arrival("ARRIVES=01:10", raw, "6E-204") is None
    assert ways.checked_arrival("ARRIVES=00:45", raw, "6E-999") is None
    land = datetime(2026, 10, 16, 0, 30, tzinfo=clock.IST)
    assert ways.retimed(land, "00:45") == datetime(2026, 10, 16, 0, 45, tzinfo=clock.IST)


def test_a_landing_time_sets_the_cab_45_minutes_later():
    land = datetime(2026, 10, 16, 0, 30, tzinfo=clock.IST)
    assert ways.pickup_after_landing(land) == datetime(2026, 10, 16, 1, 15, tzinfo=clock.IST)


async def test_the_brain_sets_the_cab_from_the_landing_time(db, at):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, E, [E, S])
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=S, members=[E, S], message_ref="m1")
    ride = {"kind": "ride", "goal": "Cab from the airport", "pickup": "Terminal 1", "drop": "Home", "service": "uber",
            "landing": "2026-10-16T00:30", "for_whom": ELDER}
    assert (await tools.start_task(ctx, ride))["ask_location"] == "pickup"
    ctx.message_ref = "m2"
    await tools.start_task(ctx, ride)
    t = (await runtime.shown_tasks(db, FAM))[0]
    assert later.of(t)["at"].startswith("2026-10-16T01:15") and later.of(t)["means"] == "arrive"
    assert t.details["landing_at"].startswith("2026-10-16T00:30")


async def test_ways_there_sends_only_what_the_page_wrote(db, monkeypatch):
    await store.save_roster(db, FAM, E, [E, S])
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=S, members=[E, S], message_ref="m1")

    async def fetch(origin, destination, at):
        return RAW, [{"title": "Testpur airport", "url": "https://example.test/taxi"}]

    async def judge(raw):
        return PROPOSED

    monkeypatch.setattr(ways, "fetch", fetch)
    monkeypatch.setattr(ways, "judge", judge)
    out = await tools.ways_there(ctx, {"origin": "Testpur airport", "destination": "Home", "at": "2026-10-16T00:30", "elder": True})
    assert out["found"] is True
    text = " ".join(out["messages"])
    assert "₹1450" in text and "₹210" in text and "₹900" in text and "TP-4" in text
    assert "₹50" not in text and "MADE-UP" not in text
    assert out["messages"][-1] == "Which feels right?"
    assert "Nothing was booked and nothing was paid" in out["note"]


async def test_a_forwarded_flight_is_a_trip_and_the_cab_follows_the_page(db, at, monkeypatch):
    at("2026-10-15 18:00")
    await store.save_roster(db, FAM, E, [E, S])
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=S, members=[E, S], message_ref="m1")

    async def fetch_text(query):
        return "Flight 6E-204 arrival 00:45.", []

    async def judge_time(raw, flight):
        return "ARRIVES=00:45"

    monkeypatch.setattr(ways, "fetch_text", fetch_text)
    monkeypatch.setattr(ways, "judge_time", judge_time)
    out = await tools.note_trip(ctx, {"airline": "IndiGo", "flight": "6E-204", "origin": "Testpur", "destination": "Homepur",
                                     "departs": "2026-10-16T22:00", "arrives": "2026-10-16T00:30", "pnr": "TESTPN", "about": ELDER})
    assert out["landing_moved"] is True and out["arrives"].startswith("2026-10-16T00:45")
    assert out["cab_at"].startswith("2026-10-16T01:30") and out["reminder"]
    assert "start_task" in out["next"] and "TESTPN" not in out["saved"]
    saved = await store.facts(db, FAM, ELDER, domains=["trip"])
    assert saved and saved[0].value["pnr"] == "TESTPN" and saved[0].value["flight"] == "6E-204"
