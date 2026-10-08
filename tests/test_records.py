"""Health records Saheli remembers only after a person chose so; forgetting removes them; prescription courses end."""
from app.care import memory_index, records, store

from tests.test_dash import ACTOR, ELDER, FAM, client  # noqa: F401 — fixture


async def test_remember_then_forget_a_record(client, db, at):  # noqa: F811
    at("2026-10-08 22:00")
    body = {"actor": ACTOR, "document_id": "doc-1", "title": "Blood count, kidney, liver", "date": "2026-10-05",
            "points": ["Haemoglobin 9.6 g/dL (low) on 5 Oct 2026", "Creatinine 1.41 mg/dL (high) on 5 Oct 2026"]}
    assert (await client.post(f"/v2/dash/{FAM}/{ELDER}/records/remember", json=body)).json() == {"remembered": 2}
    note = next(n for n in await store.notes(db, FAM, [ELDER]) if n.slug == records.NOTE_SLUG)
    assert "5 Oct 2026 · Blood count, kidney, liver: Haemoglobin 9.6" in note.body_md
    hits = await memory_index.search(db, FAM, [ELDER], "haemoglobin", limit=5)
    assert any("Haemoglobin 9.6" in h.text for h in hits)

    # Saving the same record again replaces it (no duplicate lines)
    body["points"] = ["Haemoglobin 9.6 g/dL (low) on 5 Oct 2026"]
    await client.post(f"/v2/dash/{FAM}/{ELDER}/records/remember", json=body)
    note = next(n for n in await store.notes(db, FAM, [ELDER]) if n.slug == records.NOTE_SLUG)
    assert note.body_md.count("Haemoglobin") == 1 and "Creatinine" not in note.body_md

    r = (await client.post(f"/v2/dash/{FAM}/{ELDER}/records/forget", json={"actor": ACTOR, "document_id": "doc-1"})).json()
    assert r["forgotten"] == 1
    note = next(n for n in await store.notes(db, FAM, [ELDER]) if n.slug == records.NOTE_SLUG)
    assert "Haemoglobin" not in note.body_md
    hits = await memory_index.search(db, FAM, [ELDER], "haemoglobin", limit=5)
    assert not any("Haemoglobin 9.6" in h.text for h in hits)


async def test_prescription_course_ends_after_its_last_day(client, db, at):  # noqa: F811
    at("2026-10-08 10:00")
    body = {"actor": ACTOR, "domain": "medicine", "name": "Pantocid",
            "details": {"name": "Pantocid", "dose": "40 mg", "times": ["07:30"], "ends_on": "2026-11-07", "from_record": "doc-2"},
            "sentence": "Pantocid 40 mg at 07:30 until 7 Nov 2026"}
    r = (await client.post(f"/v2/dash/{FAM}/{ELDER}/facts", json=body)).json()
    assert r["reminders"]["active"] == ["07:30"]
    host = client.host
    assert await records.end_finished_courses(db, lambda fid: host, "2026-11-07") == 0  # still the last day
    assert await records.end_finished_courses(db, lambda fid: host, "2026-11-08") == 1
    f = next(x for x in await store.facts(db, FAM, ELDER, statuses=("stopped",)) if x.key == "medicine:pantocid")
    assert f.status == "stopped"
    rows = [r for r in host.world.schedules.values() if r["sourceKey"] == "medicine:pantocid"]
    assert rows and not any(r["active"] for r in rows)  # its reminders are off too
