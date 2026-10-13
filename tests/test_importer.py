from app.care import importer, store
from app.sim.world import SimHost, World

FAM, ELDER = "fam-imp", "elder-imp"

RECORD = {
    "nameToUse": "Leela ji",
    "avoidMaa": True,
    "allergies": ["milk"],
    "dietRules": ["Low salt please", "No onion on Tuesdays"],
    "profileMedicines": [{"name": "Thyroxine", "dose": "50 mcg", "time": "06:30"}],
    "schedules": [
        {"scheduleId": "s1", "type": "MEDICINE", "title": "Metformin", "time": "08:30", "dosage": "500 mg", "instructions": "After food", "daysOfWeek": [], "sourceKey": None},
        {"scheduleId": "s2", "type": "MEDICINE", "title": "Metformin", "time": "8:30 PM", "dosage": "500 mg", "instructions": "After food", "daysOfWeek": [], "sourceKey": None},
        {"scheduleId": "s3", "type": "CHECK_IN", "title": "Evening walk", "time": "18:00", "dosage": None, "instructions": None, "daysOfWeek": [], "sourceKey": None},
    ],
    "learned": [{"category": "food", "text": "Loves besan chilla", "confirmed": True, "confidence": 0.9}],
}


async def test_import_seeds_record_once(db, at):
    at("2026-10-02 08:00")
    host = SimHost(World(backend_record=RECORD))
    assert not await importer.already_imported(db, FAM, ELDER)
    counts = await importer.import_family(db, host, family_id=FAM, backend_family_id=FAM, elder_id=ELDER)
    assert counts["created"] >= 2
    facts = {f.key: f for f in await store.facts(db, FAM, ELDER)}
    assert facts["medicine:metformin"].value["times"] == ["08:30", "20:30"]
    assert "routine:evening_walk" in facts
    claims = [c for c in host.world.calls if c["tool"] == "claim_schedule_rows"]
    assert claims and sorted(claims[0]["args"]["scheduleIds"]) == ["s1", "s2"]
    assert await importer.already_imported(db, FAM, ELDER)


async def test_old_profile_fields_are_not_read_at_the_first_turn_any_more(db, at):
    """Trust plan A1: the old profile (name to use, allergies, diet rules, profile medicines, learned lines) moved once
    with the memory move (app.care.memory_unify); the first-turn import reads the care schedules only, so a frozen old
    profile can never seed the record again, and diet rules are never read with a word list."""
    at("2026-10-02 08:00")
    host = SimHost(World(backend_record=RECORD))
    await importer.import_family(db, host, family_id=FAM, backend_family_id=FAM, elder_id=ELDER)
    keys = {f.key for f in await store.facts(db, FAM, ELDER)}
    assert not {"naming:address_as", "allergy:milk", "medicine:thyroxine"} & keys
    assert not [k for k in keys if k.startswith("diet:")]
    assert not await store.notes(db, FAM, [ELDER])
    assert not hasattr(importer, "DIET_WORDS")


async def test_shadow_import_does_not_claim_rows(db, at):
    at("2026-10-02 08:00")
    host = SimHost(World(backend_record=RECORD))
    await importer.import_family(db, host, family_id="shadow:" + FAM, backend_family_id=FAM, elder_id=ELDER)
    assert not [c for c in host.world.calls if c["tool"] == "claim_schedule_rows"]


def test_to_hhmm():
    assert importer.to_hhmm("1:00 PM") == "13:00"
    assert importer.to_hhmm("10:00 AM") == "10:00"
    assert importer.to_hhmm("12:30 am") == "00:30"
    assert importer.to_hhmm("8 pm") == "20:00"
    assert importer.to_hhmm("08:30") == "08:30"
    assert importer.to_hhmm("after lunch") is None
    assert importer.to_hhmm("8") is None
