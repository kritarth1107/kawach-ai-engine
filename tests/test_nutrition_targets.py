"""Targets and food rules (founder Q6) and the tracking setting (founder Q1): who set them, strict or guide, pending when
inferred, the extreme-calorie caution line, progress against what was logged."""

import json

from app.brain import tools
from app.care import store
from app.nutrition import settings, targets
from app.sim.world import SimHost
from tests import nutrition_fakes

nfake = nutrition_fakes.nfake  # the fixture

FAM = "fam-tgt"
PAPA = {"id": "e-tgt", "name": "Suresh", "role": "elder"}
DAUGHTER = {"id": "c-tgt", "name": "Meera", "role": "primary caregiver"}
SELF = {"id": "s-tgt", "name": "Anil", "role": "primary caregiver"}  # manages his own care


def ctx(db, speaker, elder=PAPA, members=None):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=elder, speaker=speaker, members=members or [PAPA, DAUGHTER], message_ref="t1")


async def call(db, speaker, name, args, **kw):
    out, err = await tools.run(ctx(db, speaker, **kw), name, args)
    return json.loads(out), err


async def test_a_target_records_who_set_it_and_how_and_is_pinned(db, at):
    at("2026-10-14 10:00")
    r, err = await call(db, DAUGHTER, "nutrition_target", {"for_whom": PAPA["id"], "set": {"protein_g": 60, "sugar_g": 25}, "set_by": "doctor", "how": "strict"})
    assert not err and {s["result"] for s in r["saved"]} == {"created"}
    tg, _ = await targets.current(db, FAM, PAPA["id"])
    by = {t["field"]: t for t in tg}
    assert by["protein_g"]["sense"] == "min" and by["sugar_g"]["sense"] == "max" and by["sugar_g"]["set_by"] == "doctor" and by["sugar_g"]["how"] == "strict"
    fact = await store.active_fact(db, FAM, PAPA["id"], "nutrition_target:sugar_g")
    assert fact.pinned_at is not None and "set by their doctor, strict" in fact.text


async def test_a_target_saheli_only_inferred_waits_for_confirmation(db, at):
    at("2026-10-14 10:00")
    r, _ = await call(db, SELF, "nutrition_target", {"set": {"kcal": 1500}, "set_by": "self", "inferred": True}, elder=SELF, members=[SELF])
    assert r["pending"] and not r["saved"] and "confirm" in r["note"]
    tg, _ = await targets.current(db, FAM, SELF["id"])
    assert tg == []  # not in effect
    r, err = await call(db, SELF, "confirm_change", {"key": "nutrition_target:kcal", "approve": True}, elder=SELF, members=[SELF])
    assert not err and r["result"] == "active"


async def test_an_extreme_calorie_target_is_saved_with_one_caution_line(db, at):
    at("2026-10-14 10:00")
    r, _ = await call(db, SELF, "nutrition_target", {"set": {"kcal": 1000}, "set_by": "trainer"}, elder=SELF, members=[SELF])
    assert r["saved"] and "check this target with their doctor" in r["caution"]
    r, _ = await call(db, SELF, "nutrition_target", {"set": {"kcal": 1000}, "set_by": "doctor"}, elder=SELF, members=[SELF])
    assert "caution" not in r  # the doctor's own number


async def test_a_high_protein_target_with_conditions_asks_the_brain_to_judge_kidneys(db, at):
    at("2026-10-14 10:00")
    await store.write_fact(db, family_id=FAM, subject_id=SELF["id"], domain="condition", key="condition:ckd", value={}, text="Chronic kidney disease stage 3",
                           source_kind="caregiver_said")
    r, _ = await call(db, SELF, "nutrition_target", {"set": {"protein_g": 150}, "set_by": "trainer"}, elder=SELF, members=[SELF])
    assert "Chronic kidney disease" in r["caution"] and "affects the kidneys" in r["caution"]


async def test_food_rules_and_removal(db, at):
    at("2026-10-14 10:00")
    r, _ = await call(db, DAUGHTER, "nutrition_target", {"for_whom": PAPA["id"], "rules": [{"rule": "no sugar", "food": "sugar", "none": True}],
                                                         "set_by": "caregiver", "how": "guide"})
    key = r["saved"][0]["key"]
    assert key == "nutrition_target:rule_sugar"
    _, rules = await targets.current(db, FAM, PAPA["id"])
    assert rules[0]["none"] and rules[0]["how"] == "guide"
    r, _ = await call(db, DAUGHTER, "nutrition_target", {"for_whom": PAPA["id"], "remove": [key]})
    assert r["removed"][0]["result"] == "stopped"


async def test_the_elder_cannot_quietly_change_a_caregivers_target(db, at):
    at("2026-10-14 10:00")
    await call(db, DAUGHTER, "nutrition_target", {"for_whom": PAPA["id"], "set": {"sugar_g": 25}, "set_by": "doctor", "how": "strict"})
    r, _ = await call(db, PAPA, "nutrition_target", {"set": {"sugar_g": 60}, "set_by": "self"})
    assert r["pending"]  # a health domain: a weaker source's change waits for a caregiver


def test_progress_and_the_today_line():
    tg = [targets.target_value("kcal", 1800, set_by="self", how="guide"), targets.target_value("protein_g", 90, set_by="trainer", how="guide")]
    for t, f in zip(tg, ("kcal", "protein_g")):
        t.update(key=f"nutrition_target:{f}", status="active")
    prog = targets.progress({"energy_kcal": 1240, "protein_g": 46}, tg)
    assert targets.today_line(prog) == "Today: 1,240 / 1,800 kcal, 46 / 90 g protein."
    assert [p["met"] for p in prog] == [False, False]


async def test_everyone_is_tracked_by_default_and_can_turn_it_off(db, at):
    at("2026-10-14 10:00")
    assert (await settings.get(db, FAM, PAPA["id"]))["tracking"] == "full"
    r, err = await call(db, PAPA, "nutrition_settings", {"tracking": "off"})
    assert not err and r["setting"]["tracking"] == "off"
    r, _ = await call(db, DAUGHTER, "nutrition_settings", {"for_whom": PAPA["id"], "tracking": "full", "sizes": {"katori_ml": 200}})
    assert r["setting"]["tracking"] == "full" and r["setting"]["sizes"] == {"katori_ml": 200.0}


async def test_settings_and_targets_are_not_written_through_remember(db, at):
    at("2026-10-14 10:00")
    r, err = await call(db, DAUGHTER, "remember", {"domain": "nutrition_target", "name": "kcal", "details": {}, "sentence": "1500 kcal"})
    assert err and "nutrition_target tool" in r["refused"]
