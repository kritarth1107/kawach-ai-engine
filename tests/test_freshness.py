"""Weekly medicines count only on their day, and old facts are re-checked before acting (founder 2026-10-09)."""

import json
from datetime import date, timedelta

from app.brain import tools
from app.care import checkins, digest, doses, freshness, store
from app.core import clock
from app.sim.world import SimHost

FAM, MAA = "fam-fresh", "maa-fresh"


def ctx(db, host=None):
    return tools.TurnCtx(session=db, host=host or SimHost(), family_id=FAM, elder={"id": MAA, "name": "Vasundara"},
                         speaker={"id": MAA, "name": "Vasundara", "role": "elder"}, members=[{"id": MAA, "name": "Vasundara"}])


async def call(c, name, args, ok=True):
    out, err = await tools.run(c, name, args)
    assert err != ok, out
    return json.loads(out) if ok else out


def test_weekday_numbering_converts_both_ways():
    # care record counts from Monday (0), the reminder service from Sunday (0)
    assert doses.to_backend_days([6]) == [0] and doses.from_backend_days([0]) == [6]
    assert doses.to_backend_days([0, 2]) == [1, 3] and doses.from_backend_days([1, 3]) == [0, 2]
    assert doses.weekdays(["Sunday"]) == [6] and doses.weekdays(list(range(7))) is None and doses.weekdays([]) is None
    assert doses.due_on({"days": [6]}, date(2026, 10, 11)) and not doses.due_on({"days": [6]}, "2026-10-09")
    assert doses.due_on({}, "2026-10-09")


async def test_weekly_medicine_is_not_marked_on_other_days(db, at):
    at("2026-10-09 19:13")  # a Friday
    c = ctx(db)
    await call(c, "remember", {"domain": "medicine", "name": "Vitamin D3 60000", "details": {"times": ["10:00"], "days": [6]}, "sentence": "Vitamin D3 on Sundays"})
    await call(c, "remember", {"domain": "medicine", "name": "Shelcal", "details": {"times": ["14:00"]}, "sentence": "Shelcal at 2"})
    # "I took nothing today" → Shelcal missed; Vitamin D3 was not due, so nothing to mark
    await call(c, "log_dose", {"medicine": "Shelcal", "outcome": "missed"})
    out = await call(c, "log_dose", {"medicine": "Vitamin D3", "outcome": "skipped"}, ok=False)
    assert "only on Sunday" in out
    # taken on a day it was not due: recorded, and Saheli is told to check it was not taken early by mistake
    out = await call(c, "log_dose", {"medicine": "Vitamin D3", "outcome": "taken"})
    assert "not on 2026-10-09" in out["not_due"]
    # the reminder service gets Sunday as 0
    syncs = [x for x in c.host.world.calls if x["tool"] == "sync_medicine_schedule" and "Vitamin" in x["args"]["name"]]
    assert syncs and syncs[-1]["args"]["days"] == [0]


async def test_old_facts_are_marked_and_confirming_refreshes_them(db, at):
    at("2026-06-01 10:00")
    c = ctx(db)
    await call(c, "remember", {"domain": "medicine", "name": "Thyronorm", "details": {"times": ["07:00"]}, "sentence": "Thyronorm 50 at 7"})
    await call(c, "remember", {"domain": "allergy", "name": "penicillin", "details": {}, "sentence": "Allergic to penicillin"})
    now = at("2026-10-09 11:00")
    facts = await store.facts(db, FAM, MAA)
    stale = await freshness.stale_facts(db, FAM, MAA, facts)
    assert set(stale) == {"medicine:thyronorm"} and stale["medicine:thyronorm"] == "over 4 months ago", "allergies never go stale"
    record = digest.care_record("Vasundara", facts, stale)
    assert "Thyronorm 50 at 7 (elder said, unconfirmed; last confirmed over 4 months ago, check it is still right" in record
    # the check-in asks about it (one gentle question in the health window)
    got = await checkins.plan(db, _Host(), FAM, MAA, now=now)
    assert got and got[0] == "confirm:medicine:thyronorm" and "fact_still_true medicine:thyronorm" in got[1]
    await call(c, "fact_still_true", {"key": "medicine:thyronorm"})
    assert await freshness.stale_facts(db, FAM, MAA, facts) == {}


async def test_stalest_prefers_medicines(db, at):
    at("2025-09-01 10:00")
    c = ctx(db)
    await call(c, "remember", {"domain": "doctor", "name": "Dr Mehta", "details": {"name": "Dr Mehta"}, "sentence": "Dr Mehta is her doctor"})
    at("2026-06-01 10:00")
    await call(c, "remember", {"domain": "medicine", "name": "Thyronorm", "details": {"times": ["07:00"]}, "sentence": "Thyronorm 50 at 7"})
    at("2026-10-09 11:00")
    row = await freshness.stalest(db, FAM, MAA, await store.facts(db, FAM, MAA))
    assert row and row.key == "medicine:thyronorm"


class _Host:
    async def call(self, tool, args, **kw):
        return {"items": []}


async def test_home_counts_a_weekly_medicine_only_on_its_day(db, at):
    from app.api import dash

    at("2026-10-08 09:00")  # Thursday
    c = ctx(db)
    await call(c, "remember", {"domain": "medicine", "name": "Vitamin D3 60000", "details": {"times": ["10:00"], "days": [6]}, "sentence": "Vitamin D3 on Sundays"})
    await call(c, "remember", {"domain": "medicine", "name": "Shelcal", "details": {"times": ["14:00"]}, "sentence": "Shelcal at 2"})
    at("2026-10-08 14:30")
    await call(c, "log_dose", {"medicine": "Shelcal", "outcome": "taken"})
    at("2026-10-09 09:00")  # Friday
    home = await dash.home(FAM, MAA, db)
    assert [d["name"] for d in home["doses"]] == ["Shelcal"], "Vitamin D3 is not today's dose"
    assert home["week"]["adherence"][-2] == 100, "Thursday: the one due dose was taken"
    at("2026-10-11 09:00")  # Sunday
    home = await dash.home(FAM, MAA, db)
    assert [d["name"] for d in home["doses"]] == ["Vitamin D3 60000", "Shelcal"]
