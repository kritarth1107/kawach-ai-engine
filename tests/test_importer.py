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
    assert counts["created"] >= 7
    facts = {f.key: f for f in await store.facts(db, FAM, ELDER)}
    assert facts["medicine:metformin"].value["times"] == ["08:30", "20:30"]
    assert facts["medicine:thyroxine"].value["times"] == ["06:30"]
    assert "diet:low_salt" in facts and "diet:no_onion" in facts
    assert facts["naming:address_as"].value == {"name": "Leela ji", "avoid": ["maa"]}
    assert "routine:evening_walk" in facts
    claims = [c for c in host.world.calls if c["tool"] == "claim_schedule_rows"]
    assert claims and sorted(claims[0]["args"]["scheduleIds"]) == ["s1", "s2"]
    notes = await store.notes(db, FAM, [ELDER])
    assert "besan chilla" in notes[0].body_md
    assert await importer.already_imported(db, FAM, ELDER)


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
