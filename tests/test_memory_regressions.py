"""Bugs found by the 30-day family simulation; each must stay fixed."""

import json

from app.brain import tools
from app.care import store
from app.care.domains import fact_key
from app.sim.world import SimHost

FAM, AAI, BABA, RAHUL = "fam-r", "aai", "baba", "rahul"


def ctx(db, speaker=RAHUL, role="caregiver", elder=AAI):
    return tools.TurnCtx(
        session=db, host=SimHost(), family_id=FAM, elder={"id": elder, "name": "Aai"},
        speaker={"id": speaker, "name": speaker, "role": role},
        members=[{"id": AAI, "name": "Aai"}, {"id": BABA, "name": "Baba"}, {"id": RAHUL, "name": "Rahul"}],
    )


async def call(c, name, args):
    out, err = await tools.run(c, name, args)
    assert not err, out
    return json.loads(out)


def test_one_key_per_medicine():
    assert fact_key("medicine", "Thyronorm 50") == fact_key("medicine", "Thyronorm 50 mcg") == "medicine:thyronorm"
    assert fact_key("medicine", "Insulin Glargine 14 units") == "medicine:insulin_glargine"


async def test_restating_a_medicine_without_times_keeps_its_times(db, at):
    at("2026-10-05 10:00")
    c = ctx(db)
    await call(c, "remember", {"domain": "medicine", "name": "Thyronorm 50", "details": {"name": "Thyronorm", "dose": "50 mcg", "times": ["07:00"]}, "sentence": "Thyronorm 50 mcg at 7"})
    out = await call(c, "remember", {"domain": "medicine", "name": "Thyronorm", "details": {"name": "Thyronorm", "dose": "50 mcg"}, "sentence": "Thyronorm 50 mcg"})
    assert out["result"] == "unchanged"
    rows = await store.facts(db, FAM, AAI, domains=["medicine"])
    assert [(r.key, r.value["times"], r.status) for r in rows] == [("medicine:thyronorm", ["07:00"], "active")]
    assert sorted(r["time"] for r in c.host.world.schedules.values() if r["active"]) == ["07:00"]


async def test_approving_a_change_never_erases_times(db, at):
    at("2026-10-05 10:00")
    await call(ctx(db), "remember", {"domain": "medicine", "name": "Telma", "details": {"name": "Telma", "dose": "40 mg", "times": ["09:00"]}, "sentence": "Telma 40 at 9", "about": BABA})
    # The elder reports a new dose (weaker source): it waits as pending, without times.
    w = await store.write_fact(db, family_id=FAM, subject_id=BABA, domain="medicine", key=fact_key("medicine", "Telma 80"),
                               value={"name": "Telma", "dose": "80 mg"}, text="Telma 80", source_kind="elder_said")
    assert w.result == "pending" and w.fact.value["times"] == ["09:00"]
    row = await store.resolve_pending(db, fact_id=w.fact.id, approve=True, by=RAHUL)
    assert row.status == "active" and row.value == {"name": "Telma", "dose": "80 mg", "times": ["09:00"]}


async def test_logging_the_same_dose_twice_is_flagged(db, at):
    at("2026-10-05 08:00")
    e = ctx(db, AAI, "elder")
    await call(e, "log_dose", {"medicine": "Ultracet", "outcome": "taken"})
    at("2026-10-05 08:14")
    out = await call(e, "log_dose", {"medicine": "Ultracet", "outcome": "taken"})
    assert "possible_double_dose" in out
