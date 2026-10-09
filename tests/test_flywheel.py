"""The flywheel numbers (founder's architecture slide: completion, reopen rate, cost per task) add up."""

from datetime import timedelta

from app.brain import tools
from app.care import store
from app.learn import flywheel
from app.sim.world import SimHost
from app.tasks.models import Task

FAM, MAA, SON = "fam-fly", "maa-fly", "son-fly"
ROSTER = [{"id": MAA, "name": "Kamla", "role": "elder"}, {"id": SON, "name": "Ravi", "role": "primary caregiver"}]


def _task(at, status, placed=False, service="blinkit", cost=2.0, when=None):
    t0 = at(when or "2026-10-08 10:00")
    return Task(family_id=FAM, subject_id=MAA, requested_by=MAA, service=service, kind="order", goal="atta", status=status, phase="place",
                details={"metrics": {"cost_inr": cost}}, result={"placed": placed}, history=[], created_at=t0,
                updated_at=t0 + timedelta(minutes=12), deadline_at=t0 + timedelta(hours=1))


async def test_flywheel_numbers(db, at):
    at("2026-10-01 09:00")
    await store.save_roster(db, FAM, ROSTER[0], ROSTER)
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ROSTER[0], speaker=ROSTER[1], members=ROSTER)
    await tools.run(c, "remember", {"domain": "medicine", "name": "Shelcal", "details": {"times": ["14:00"]}, "sentence": "Shelcal at 2"})
    db.add(_task(at, "done", placed=True))
    db.add(_task(at, "failed", when="2026-10-08 11:00"))
    db.add(_task(at, "done", placed=True, when="2026-10-08 11:30"))  # asked again after the failure: a reopen
    for d in ("2026-10-07", "2026-10-08"):
        at(f"{d} 14:10")
        await tools.run(tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ROSTER[0], speaker=ROSTER[0], members=ROSTER),
                        "log_dose", {"medicine": "Shelcal", "outcome": "taken"})
    await db.commit()
    at("2026-10-09 10:00")
    out = await flywheel.compute(db, days=14)
    t = out["tasks"]
    assert t["completed"] == 2 and t["failed"] == 1 and t["completionRate"] == 0.667 and t["reopenRate"] == 1.0
    assert t["medianMinutesToComplete"] == 12.0 and t["browserInrPerCompleted"] == 3.0
    fam = next(f for f in out["perFamily"] if f["family"] == FAM)
    assert fam["adherence7d"] is not None and fam["delegated7d"] == 3
