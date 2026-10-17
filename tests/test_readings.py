"""Every reading reaches Health Records (founder 2026-10-13: "if someone has sent weight or BMI or anything those should go
to health records, which never went for Kritarth and Rahul"). Live: an InBody scan's BMI and body fat were kept only as a
note; only the weight was logged."""

import json

from sqlalchemy import select

from app.brain import tools
from app.care import extract, readings, store
from app.care.models import CareEvent, MemoryNote, OpenLoop
from app.core import clock
from app.llm import router
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-read"
ME = {"id": "self-r", "name": "Kittu", "role": "primary caregiver"}


def ctx(db, host):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ME, speaker=ME, members=[ME], user_text="")


async def vitals(db):
    return list((await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "vital").order_by(CareEvent.at))).scalars())


class Fake:
    def __init__(self, out):
        self.out = out

    async def complete(self, route, **kw):
        return LLMReply(text=json.dumps(self.out), tool_calls=[], model="fake")


def use(monkeypatch, out):
    router.register_provider("fake", Fake(out))
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()


async def test_a_scan_is_logged_value_by_value_on_the_day_it_was_measured(db, at):
    at("2026-10-12 19:00")
    await store.save_roster(db, FAM, ME, [ME])
    host = SimHost()
    for a in ({"kind": "weight", "value": "73.5", "unit": "kg", "taken_on": "2026-10-11"},
              {"kind": "bmi", "value": "25.4", "taken_on": "2026-10-11"},
              {"kind": "body_fat", "value": "23.2", "unit": "%", "taken_on": "2026-10-11"},
              {"kind": "other", "name": "Visceral fat", "value": "8"}):
        out, err = await tools.run(ctx(db, host), "log_vital", a)
        assert not err and json.loads(out)["health_records"] == "saved"
    rows = await vitals(db)
    assert [clock.ist(r.at).date().isoformat() for r in rows] == ["2026-10-11"] * 3 + ["2026-10-12"]
    assert [v["kind"] for v in host.world.vitals] == ["weight", "bmi", "body_fat", "other"]
    assert host.world.vitals[0]["taken_on"] == "2026-10-11" and host.world.vitals[3]["name"] == "Visceral fat"
    out, err = await tools.run(ctx(db, host), "log_vital", {"kind": "other", "value": "8"})
    assert err and "name" in out
    out, err = await tools.run(ctx(db, host), "log_vital", {"kind": "weight", "value": "70", "taken_on": "2026-10-20"})
    assert err and "future" in out


async def test_an_old_reading_is_history_not_an_alarm(db, at):
    at("2026-10-12 19:00")
    await store.save_roster(db, FAM, ME, [ME])
    out, _ = await tools.run(ctx(db, SimHost()), "log_vital", {"kind": "bp", "value": "190/125", "taken_on": "2026-09-01"})
    assert json.loads(out)["red_flag"] is None
    out, _ = await tools.run(ctx(db, SimHost()), "log_vital", {"kind": "bp", "value": "190/125"})
    assert json.loads(out)["red_flag"]


async def test_the_re_read_asks_about_readings_said_in_chat_and_logs_none(db, at, monkeypatch):
    # Review 17 Oct: nothing the background re-read finds saves health on its own. A reading the chat did not log is put to a
    # caregiver once (memory check); one already logged is skipped; one with no number is dropped.
    at("2026-10-13 15:00")
    await store.save_roster(db, FAM, ME, [ME])
    host = SimHost()
    monkeypatch.setattr(extract, "host_for_readings", host)
    await tools.run(ctx(db, host), "log_vital", {"kind": "weight", "value": "90.3", "unit": "kg"})
    use(monkeypatch, {"facts": [], "stops": [], "notes": [], "readings": [
        {"about": ME["id"], "kind": "weight", "value": "90.3 kg", "unit": "kg"},  # already logged live
        {"about": ME["id"], "kind": "height", "value": "178", "unit": "cm"},
        {"about": ME["id"], "kind": "sugar", "value": "not sure"},  # no number
    ]})
    try:
        await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="user", text="90.3 kg, I am 5 ft 10")
        counts = await extract.extract_family(db, FAM)
        assert counts.get("readings_to_confirm") == 1 and not counts.get("readings")
        assert [r.payload["kind"] for r in await vitals(db)] == ["weight"]
        loops = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "memory_check"))).scalars().all()
        assert len(loops) == 1 and "height 178" in loops[0].title
    finally:
        router._providers.pop("fake", None)


async def test_readings_kept_only_in_notes_are_recovered_once(db, at, monkeypatch):
    at("2026-10-13 15:00")
    await store.save_roster(db, FAM, ME, [ME])
    db.add(MemoryNote(family_id=FAM, subject_id=ME["id"], slug="fitness", title="Fitness",
                      body_md="- 2026-10-11: InBody scan: Weight 73.5 kg, BMI 25.4, Body Fat 23.2%", updated_at=clock.now()))
    await db.flush()
    host = SimHost()
    await tools.run(ctx(db, host), "log_vital", {"kind": "weight", "value": "73.5", "unit": "kg", "taken_on": "2026-10-11"})
    use(monkeypatch, {"readings": [
        {"kind": "weight", "value": "73.5", "unit": "kg", "taken_on": "2026-10-11"},
        {"kind": "BMI", "value": "25.4", "taken_on": "2026-10-11"},  # live: the model wrote its own labels
        {"kind": "body fat", "value": "23.2", "unit": "%", "taken_on": "2026-10-11"},
    ]})
    try:
        listed = await readings.recover(db, FAM, ME["id"], apply=False)
        assert len(listed) == 3 and len(await vitals(db)) == 1, "a dry run changes nothing"
        got = await readings.recover(db, FAM, ME["id"], apply=True, host=host)
        assert [r["kind"] for r in got] == ["bmi", "body_fat"], "the weight was already there"
        assert {clock.ist(r.at).date().isoformat() for r in await vitals(db)} == {"2026-10-11"}
        assert await readings.recover(db, FAM, ME["id"], apply=True, host=host) == [], "running again adds nothing"
    finally:
        router._providers.pop("fake", None)
