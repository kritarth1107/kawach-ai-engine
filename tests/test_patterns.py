"""Pattern spotting: two weeks of a family's logs, and what Saheli should notice on her own."""

from datetime import timedelta

import pytest

from app.brain import tools
from app.care import patterns, store
from app.core import clock
from app.sim.world import SimHost

FAM, ELDER, CG = "fam-p", "elder-p", "cg-p"


def ctx(db, speaker=CG):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder={"id": ELDER, "name": "Kamla"},
                         speaker={"id": speaker, "name": "Asha", "role": "caregiver"},
                         members=[{"id": ELDER, "name": "Kamla"}, {"id": CG, "name": "Asha"}])


async def ev(db, event_kind, at, summary, **payload):
    clock.set_now(at)
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind=event_kind, summary=summary, payload=payload)


def ist(day, hhmm):
    """day: 1..15 of Oct 2026, IST."""
    from datetime import datetime

    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, day, h, m, tzinfo=clock.IST)


async def medicine(db, start):
    clock.set_now(start)
    await tools.run(ctx(db), "remember", {"domain": "medicine", "name": "Metformin", "details": {"dose": "500 mg", "times": ["08:00", "21:00"]},
                                          "sentence": "Metformin 500 mg at 08:00 and 21:00"})


async def test_evening_dose_and_sunday_pattern(db, at):
    await medicine(db, ist(1, "07:00"))
    for d in range(1, 15):
        await ev(db, "dose_taken", ist(d, "08:10"), "Metformin: taken", medicine="Metformin")
        day = ist(d, "21:00")
        if day.weekday() == 6 or d % 3 == 0:  # every Sunday and some other evenings
            continue
        await ev(db, "dose_taken", ist(d, "21:20"), "Metformin: taken", medicine="Metformin")
    clock.set_now(ist(15, "10:00"))
    found = {p.kind: p for p in await patterns.find(db, FAM, ELDER)}
    assert "dose_slot" in found and "21:00" in found["dose_slot"].title and "08:00" not in found["dose_slot"].title
    assert "dose_weekday" in found and "Sundays" in found["dose_weekday"].title


async def test_regular_doses_no_pattern(db, at):
    await medicine(db, ist(1, "07:00"))
    for d in range(1, 15):
        await ev(db, "dose_taken", ist(d, "08:05"), "Metformin: taken", medicine="Metformin")
        await ev(db, "dose_taken", ist(d, "21:10"), "Metformin: taken", medicine="Metformin")
    clock.set_now(ist(15, "10:00"))
    assert not [p for p in await patterns.find(db, FAM, ELDER) if p.kind.startswith("dose")]


async def test_late_doses(db, at):
    await medicine(db, ist(1, "07:00"))
    for d in range(1, 15):
        await ev(db, "dose_taken", ist(d, "10:00"), "Metformin: taken", medicine="Metformin")  # 2 h late every morning
        await ev(db, "dose_taken", ist(d, "21:05"), "Metformin: taken", medicine="Metformin")
    clock.set_now(ist(15, "10:00"))
    late = [p for p in await patterns.find(db, FAM, ELDER) if p.kind == "dose_late"]
    assert late and "08:00" in late[0].title and "120" in late[0].title


async def test_bp_creeping_up_and_high(db, at):
    for d, v in [(1, "128/80"), (3, "130/82"), (5, "126/78"), (6, "129/80"), (9, "146/92"), (11, "150/94"), (13, "148/90")]:
        await ev(db, "vital", ist(d, "09:00"), f"bp {v}", kind="bp", value=v)
    clock.set_now(ist(14, "10:00"))
    found = {p.key: p for p in await patterns.find(db, FAM, ELDER)}
    assert "vital:bp:up" in found and "vital:bp:high" in found


async def test_symptom_comes_back_and_low_mood_and_meals(db, at):
    for d, t in [(3, "ghutne mein dard"), (6, "knee pain again"), (9, "ghutna sooja hua"), (12, "knee hurts on stairs")]:
        await ev(db, "symptom", ist(d, "11:00"), t, severity="watch")
    for d, t in [(9, "feeling lonely today"), (11, "bahut udaas"), (13, "akeli lag rahi hoon")]:
        await ev(db, "mood", ist(d, "18:00"), t, severity="watch")
    for d in (10, 12, 13):
        await ev(db, "meal", ist(d, "14:00"), "skipped lunch, no appetite")
    clock.set_now(ist(14, "10:00"))
    kinds = {p.kind for p in await patterns.find(db, FAM, ELDER)}
    assert {"symptom_repeat", "mood_low", "meals_skipped"} <= kinds


async def test_going_quiet_and_late_nights(db, at):
    for d in range(1, 11):
        for h in (8, 11, 15, 19):
            clock.set_now(ist(d, f"{h:02d}:00"))
            await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="hello", speaker_id=ELDER)
    for d in (8, 9, 10):
        clock.set_now(ist(d, "02:30"))
        await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="neend nahi aa rahi", speaker_id=ELDER)
    clock.set_now(ist(14, "10:00"))  # nothing on days 11-13
    kinds = {p.kind for p in await patterns.find(db, FAM, ELDER)}
    assert {"quiet", "night_awake"} <= kinds


async def test_recorded_once_a_week_and_shown_to_brain(db, at):
    for d in (3, 6, 9, 12):
        await ev(db, "symptom", ist(d, "11:00"), "chakkar aa raha hai", severity="watch")
    clock.set_now(ist(14, "10:00"))
    first = await patterns.record_new(db, FAM, ELDER)
    again = await patterns.record_new(db, FAM, ELDER)
    assert first and not again
    items = await patterns.recent(db, FAM, [ELDER])
    block = patterns.context_block(items, {ELDER: "Kamla"})
    assert "Dizziness keeps coming back" in block and "NEW" in block
    out, err = await tools.run(ctx(db), "patterns", {})
    assert not err and "Dizziness" in out


async def test_nothing_to_notice(db, at):
    clock.set_now(ist(14, "10:00"))
    assert await patterns.find(db, FAM, ELDER) == []
    out, err = await tools.run(ctx(db), "patterns", {})
    assert "Nothing stands out" in out


@pytest.mark.parametrize("minutes,name", [(8 * 60, "morning"), (13 * 60, "afternoon"), (18 * 60, "evening"), (21 * 60, "night"), (2 * 60, "night")])
def test_slot_names(minutes, name):
    assert patterns.slot_name(minutes) == name


async def test_only_sundays_missed_is_still_noticed(db, at):
    await medicine(db, ist(1, "07:00"))
    for d in range(1, 15):
        await ev(db, "dose_taken", ist(d, "08:05"), "Metformin: taken", medicine="Metformin")
        if ist(d, "21:00").weekday() != 6:
            await ev(db, "dose_taken", ist(d, "21:10"), "Metformin: taken", medicine="Metformin")
    clock.set_now(ist(15, "10:00"))
    found = {p.kind: p for p in await patterns.find(db, FAM, ELDER)}
    assert "dose_weekday" in found and "dose_slot" not in found


@pytest.mark.parametrize("raw,want", [(None, None), (3, [3]), ([0, 2], [0, 2]), (["Mon", "thu"], [0, 3]), ("sun", [6]), (9, None)])
def test_weekdays_from_any_shape(raw, want):
    assert patterns._weekdays(raw) == want


async def test_odd_medicine_record_does_not_hide_other_patterns(db, at):
    clock.set_now(ist(1, "07:00"))
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="medicine", key="medicine:weird",
                           value={"name": "Weird", "times": "08:00", "days": 7}, text="Weird", source_kind="caregiver_said", stated_by=CG)
    for d in (3, 6, 9, 12):
        await ev(db, "symptom", ist(d, "11:00"), "knee pain", severity="watch")
    clock.set_now(ist(14, "10:00"))
    assert any(p.kind == "symptom_repeat" for p in await patterns.find(db, FAM, ELDER))
