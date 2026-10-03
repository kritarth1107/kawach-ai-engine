"""Each person's own normal, and whether their reminders work."""

from datetime import datetime

from app.care import baselines, patterns, store
from app.brain import tools
from app.core import clock
from app.sim.world import SimHost

FAM, ELDER, CG = "fam-b", "elder-b", "cg-b"


def ist(day, hhmm, month=10):
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, month, day, h, m, tzinfo=clock.IST)


async def ev(db, event_kind, at, summary, **payload):
    clock.set_now(at)
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind=event_kind, summary=summary, payload=payload)


async def med(db):
    clock.set_now(ist(1, "07:00"))
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder={"id": ELDER, "name": "Kamla"},
                      speaker={"id": CG, "name": "Asha", "role": "caregiver"}, members=[{"id": ELDER, "name": "Kamla"}, {"id": CG, "name": "Asha"}])
    await tools.run(c, "remember", {"domain": "medicine", "name": "Metformin", "details": {"dose": "500 mg", "times": ["08:00", "21:00"]},
                                    "sentence": "Metformin 500 mg at 08:00 and 21:00"})


async def test_own_normal_bp_and_above_it(db, at):
    for d in range(1, 22):
        await ev(db, "vital", ist(d, "09:00"), f"bp {118 + d % 4}/{76 + d % 3}", kind="bp", value=f"{118 + d % 4}/{76 + d % 3}")
    for d in (23, 25, 27):
        await ev(db, "vital", ist(d, "09:00"), "bp 136/86", kind="bp", value="136/86")  # high for her, though below 140
    clock.set_now(ist(28, "08:00"))
    b = await baselines.save(db, FAM, ELDER)
    assert 118 <= b["vitals"]["bp"]["systolic"]["median"] <= 122
    line = baselines.usual_line("Kamla", b)
    assert "BP usually ~1" in line
    kinds = {p.kind for p in await patterns.find(db, FAM, ELDER)}
    assert "above_usual" in kinds and "vital_high" not in kinds  # high for her, not by the general rule


async def test_reminder_time_suggestion_when_always_late(db, at):
    await med(db)
    for d in range(1, 15):
        await ev(db, "reminder_sent", ist(d, "21:00"), "Metformin 21:00 reminder")
        await ev(db, "dose_taken", ist(d, "21:50"), "Metformin: taken", medicine="Metformin")
        await ev(db, "dose_taken", ist(d, "08:05"), "Metformin: taken", medicine="Metformin")
    clock.set_now(ist(15, "10:00"))
    b = await baselines.save(db, FAM, ELDER)
    night = b["doses"]["Metformin@21:00"]
    assert night["delay_min"] == 50 and night["after_reminder"] == 1.0
    sug = b["suggestions"]
    assert sug and sug[0]["suggest"] == "21:45"
    assert any(p.kind == "reminder_time" and "21:45" in p.title for p in await patterns.find(db, FAM, ELDER))


async def test_reminder_that_does_not_help(db, at):
    await med(db)
    for d in range(1, 15):
        await ev(db, "reminder_sent", ist(d, "08:00"), "Metformin 08:00 reminder")
        if d % 3 == 0:
            await ev(db, "dose_taken", ist(d, "08:20"), "Metformin: taken", medicine="Metformin")
        else:
            await ev(db, "dose_taken", ist(d, "11:30"), "Metformin: taken", medicine="Metformin")  # hours later, not because of the reminder
    clock.set_now(ist(15, "12:00"))
    b = await baselines.save(db, FAM, ELDER)
    assert b["doses"]["Metformin@08:00"]["after_reminder"] < 0.4
    assert any(s["suggest"] is None for s in b["suggestions"])


async def test_rhythm_and_no_data(db, at):
    clock.set_now(ist(20, "10:00"))
    assert baselines.usual_line("Kamla", await baselines.save(db, FAM, ELDER)) == ""
    for d in range(1, 15):
        for h in (7, 12, 20):
            clock.set_now(ist(d, f"{h:02d}:30"))
            await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="hi", speaker_id=ELDER)
    clock.set_now(ist(15, "10:00"))
    r = (await baselines.save(db, FAM, ELDER))["rhythm"]
    assert r["messages_per_day"] == 3 and r["active_from"] == "07:30" and r["active_to"] == "20:30"
