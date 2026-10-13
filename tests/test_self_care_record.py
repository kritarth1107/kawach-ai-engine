"""An adult managing their own care is their own record keeper (world probe judge, 2026-10-13).

A self-care user's own statements were filed as "elder said": a dose their doctor changed waited for a caregiver's
confirmation that nobody could give, and they could not confirm it themselves. Their words now carry a caregiver's
weight. An elder's own change still waits for a caregiver, and alerts still never go to a person about themselves."""

from app.brain import tools
from app.care import store
from app.sim.world import SimHost

FAM = "fam-self"
ME = {"id": "self-1", "name": "Rhea", "role": "primary caregiver"}
MAA = {"id": "elder-1", "name": "Kamla", "role": "elder"}
SON = {"id": "son-1", "name": "Ravi", "role": "primary caregiver"}


def ctx(db, speaker, elder, members):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=elder, speaker=speaker, members=members, message_ref="m1")


async def test_who_counts_as_their_own_record_keeper(db):
    me = ctx(db, ME, ME, [ME])
    assert me.self_care and me.source_kind == "caregiver_said" and not me.is_caregiver  # not their own alert target
    elder = ctx(db, MAA, MAA, [MAA, SON])
    assert not elder.self_care and elder.source_kind == "elder_said"
    son = ctx(db, SON, MAA, [MAA, SON])
    assert not son.self_care and son.source_kind == "caregiver_said" and son.is_caregiver


async def test_a_self_care_users_dose_change_is_saved_not_left_waiting(db, at):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, ME, [ME])
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg", "times": ["08:30"]}, text="Metformin 500 mg at 08:30", source_kind="caregiver_said")
    out = await tools.remember(ctx(db, ME, ME, [ME]), {"domain": "medicine", "name": "Metformin",
                                                       "details": {"name": "Metformin", "dose": "1000 mg", "times": ["08:30"]},
                                                       "sentence": "Doctor changed my Metformin to 1000 mg at 08:30"})
    assert out["result"] == "superseded"
    active = await store.active_fact(db, FAM, ME["id"], "medicine:metformin")
    assert active.value["dose"] == "1000 mg"


async def test_an_elders_own_dose_change_still_waits_for_a_caregiver(db, at):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg", "times": ["08:30"]}, text="Metformin 500 mg at 08:30", source_kind="caregiver_said")
    out = await tools.remember(ctx(db, MAA, MAA, [MAA, SON]), {"domain": "medicine", "name": "Metformin",
                                                              "details": {"name": "Metformin", "dose": "1000 mg", "times": ["08:30"]},
                                                              "sentence": "Doctor ne Metformin 1000 kar di"})
    assert out["result"] == "pending"


async def test_a_self_care_user_can_confirm_their_own_waiting_change(db, at):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, ME, [ME])
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="medicine", key="medicine:amlodipine",
                           value={"name": "Amlodipine", "dose": "5 mg"}, text="Amlodipine 5 mg", source_kind="prescription")
    w = await store.stop_fact(db, family_id=FAM, subject_id=ME["id"], key="medicine:amlodipine", reason="feels dizzy", source_kind="elder_said")
    assert w.result == "pending"  # left waiting before this fix
    out = await tools.confirm_change(ctx(db, ME, ME, [ME]), {"key": "medicine:amlodipine", "approve": True})
    assert out["result"] == "stopped"
