"""Whatever they say about their doses is recorded, also corrections and earlier days (founder 2026-10-09).

Live 2026-10-09 19:12: Maa said she took everything but Folvite, then "no, I made a mistake: I took nothing today;
yesterday I took everything but Folvite". The dashboard kept the first answer and yesterday could not be marked."""

import json

from app.brain import tools
from app.care import doses, store
from app.core import clock
from app.sim.world import SimHost

FAM, MAA = "fam-dose", "maa-dose"
MEDS = (("BP", "1", ["08:00"]), ("Folvite 5mg", "½ tablet", ["13:00"]), ("Shelcal 500mg", "1 tablet", ["14:00"]))


def ctx(db, host):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder={"id": MAA, "name": "Vasundara"},
                         speaker={"id": MAA, "name": "Vasundara", "role": "elder"}, members=[{"id": MAA, "name": "Vasundara"}],
                         message_ref=None)


async def call(c, name, args):
    out, err = await tools.run(c, name, args)
    assert not err, out
    return json.loads(out)


async def _meds(db, c):
    for name, dose, times in MEDS:
        await call(c, "remember", {"domain": "medicine", "name": name, "details": {"dose": dose, "times": times}, "sentence": f"{name} at {times[0]}"})


async def _standing(db, day):
    rows = await store.events(db, FAM, MAA, day=day, kinds=list(doses.KINDS))
    return {r.payload["medicine"].split()[0].lower(): r.kind.removeprefix("dose_") for r in rows}


async def test_a_correction_replaces_the_answer_and_yesterday_can_be_fixed(db, at):
    host = SimHost()
    at("2026-10-08 08:29")
    c = ctx(db, host)
    await _meds(db, c)
    await call(c, "log_dose", {"medicine": "folvite", "outcome": "taken"})  # yesterday morning, as it was live
    at("2026-10-09 19:12")
    for med, outcome in (("bp", "taken"), ("shelcal", "taken"), ("folvite", "skipped")):
        await call(c, "log_dose", {"medicine": med, "outcome": outcome})
    assert await _standing(db, "2026-10-09") == {"bp": "taken", "shelcal": "taken", "folvite": "skipped"}
    # "no, I made a mistake: nothing today"
    at("2026-10-09 19:13")
    for med in ("bp", "shelcal", "folvite"):
        await call(c, "log_dose", {"medicine": med, "outcome": "missed"})
    assert await _standing(db, "2026-10-09") == {"bp": "missed", "shelcal": "missed", "folvite": "missed"}
    # "yesterday everything except Folvite"
    for med, outcome in (("bp", "taken"), ("shelcal", "taken"), ("folvite", "missed")):
        out = await call(c, "log_dose", {"medicine": med, "outcome": outcome, "day": "yesterday"})
        assert "for 2026-10-08" in out["logged"]
    assert await _standing(db, "2026-10-08") == {"bp": "taken", "shelcal": "taken", "folvite": "missed"}
    # the backend marks the right day, and the older answers stay on record as corrected
    assert [m["dateKey"] for m in host.world.marks][-3:] == ["2026-10-08"] * 3
    corrected = await store.events(db, FAM, MAA, kinds=[doses.CORRECTED])
    assert len(corrected) == 4 and {e.payload["was"] for e in corrected} == {"dose_taken", "dose_skipped"}
    # what the brain reads as today: only the latest answers
    from app.care import digest

    today = digest.ledger(await store.events(db, FAM, MAA, day=clock.ist_day()))
    assert "dose_taken" not in today and "dose_skipped" not in today and today.count("dose_missed") == 3


async def test_two_doses_of_one_medicine_are_kept_apart(db, at):
    host = SimHost()
    at("2026-10-09 08:30")
    c = ctx(db, host)
    await call(c, "remember", {"domain": "medicine", "name": "Metformin", "details": {"times": ["08:00", "20:00"]}, "sentence": "Metformin twice"})
    await call(c, "log_dose", {"medicine": "Metformin", "outcome": "taken"})
    at("2026-10-09 20:30")
    await call(c, "log_dose", {"medicine": "Metformin", "outcome": "missed"})
    rows = await store.events(db, FAM, MAA, day="2026-10-09", kinds=list(doses.KINDS))
    assert [(r.payload["time"], r.kind) for r in rows] == [("08:00", "dose_taken"), ("20:00", "dose_missed")]
    # correcting the morning one by its time leaves the evening one alone
    await call(c, "log_dose", {"medicine": "Metformin", "outcome": "missed", "time": "8:00 AM"})
    rows = await store.events(db, FAM, MAA, day="2026-10-09", kinds=list(doses.KINDS))
    assert sorted((r.payload["time"], r.kind) for r in rows) == [("08:00", "dose_missed"), ("20:00", "dose_missed")]


async def test_days_and_times_are_read_safely(at):
    at("2026-10-09 10:00")
    assert doses.resolve_day("yesterday") == "2026-10-08" and doses.resolve_day(None) == "2026-10-09"
    assert doses.resolve_day("2026-10-10") is None and doses.resolve_day("2026-09-30") is None and doses.resolve_day("soon") is None
    assert [doses.hhmm(t) for t in ("8:00 AM", "08:00", "9 pm", "12:30 am", "noon")] == ["08:00", "08:00", "21:00", "00:30", None]


async def test_log_dose_refuses_a_day_out_of_range(db, at):
    at("2026-10-09 10:00")
    out, err = await tools.run(ctx(db, SimHost()), "log_dose", {"medicine": "BP", "outcome": "taken", "day": "2026-10-11"})
    assert err and "last 7 days" in out


async def test_a_dashboard_tick_for_yesterday_replaces_saheli_s_answer(db, at):
    """The backend pushes a dose marked on the dashboard (any of the last 7 days); it replaces the earlier answer."""
    import httpx

    from app.core.security import verify_api_secret
    from app.db.session import get_db
    from app.main import app

    host = SimHost()
    at("2026-10-08 13:30")
    c = ctx(db, host)
    await _meds(db, c)
    await call(c, "log_dose", {"medicine": "Folvite", "outcome": "taken"})
    at("2026-10-09 09:00")

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post("/v2/events", json={
                "family_id": FAM, "subject_id": MAA, "kind": "dose_missed", "summary": "Folvite 5mg ½ tablet (1:00 PM): missed (marked by Kritarth)",
                "payload": {"scheduleId": "s2", "dateKey": "2026-10-08", "time": "1:00 PM", "status": "missed", "medicine": "Folvite 5mg"},
                "ref": "completion:s2:2026-10-08:missed:1"})
            assert r.status_code == 200 and r.json()["recorded"]
    finally:
        app.dependency_overrides.clear()
    assert await _standing(db, "2026-10-08") == {"folvite": "missed"}
