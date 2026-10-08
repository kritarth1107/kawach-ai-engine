import httpx
import pytest

from app.api import dash
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.main import app
from app.sim.world import SimHost, World

FAM, ELDER, CG = "fam-d", "elder-d", "cg-d"
ACTOR = {"id": CG, "name": "Asha"}


@pytest.fixture
async def client(db, monkeypatch):
    host = SimHost(World(backend_record={"allergies": ["milk"], "schedules": [
        {"scheduleId": "s1", "type": "MEDICINE", "title": "Metformin", "time": "8:30 AM", "dosage": "500 mg", "daysOfWeek": [], "sourceKey": None}]}))
    monkeypatch.setattr(dash, "make_host", lambda: host)

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        c.host = host
        yield c
    app.dependency_overrides.clear()


async def test_overview_imports_then_edit_and_history(client, at):
    at("2026-10-02 09:00")
    r = (await client.get(f"/v2/dash/{FAM}/{ELDER}/overview")).json()
    keys = {f["key"] for f in r["facts"]}
    assert {"allergy:milk", "medicine:metformin"} <= keys
    body = {"actor": ACTOR, "domain": "medicine", "name": "Metformin", "details": {"name": "Metformin", "dose": "1000 mg", "times": ["08:30", "20:30"]},
            "sentence": "Metformin 1000 mg at 08:30 and 20:30"}
    r = (await client.post(f"/v2/dash/{FAM}/{ELDER}/facts", json=body)).json()
    assert r["result"] == "superseded" and r["reminders"]["active"] == ["08:30", "20:30"]
    h = (await client.get(f"/v2/dash/{FAM}/{ELDER}/history", params={"key": "medicine:metformin"})).json()
    assert [v["status"] for v in h["versions"]] == ["superseded", "active"]
    ev = (await client.get(f"/v2/dash/{FAM}/{ELDER}/events", params={"day": "2026-10-02"})).json()["events"]
    assert any(e["kind"] == "dashboard_edit" for e in ev)


async def test_bad_domain_and_resolve_pending(client, db, at):
    at("2026-10-02 09:00")
    assert (await client.post(f"/v2/dash/{FAM}/{ELDER}/facts", json={"actor": ACTOR, "domain": "x", "name": "a", "sentence": "b"})).status_code == 400
    from app.care import store

    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="medicine", key="medicine:amlodipine", value={"dose": "5 mg"},
                           text="Amlodipine 5 mg", source_kind="prescription")
    w = await store.stop_fact(db, family_id=FAM, subject_id=ELDER, key="medicine:amlodipine", reason="dizzy", source_kind="elder_said")
    assert w.result == "pending"
    r = await client.post(f"/v2/dash/{FAM}/{ELDER}/facts/resolve", json={"actor": ACTOR, "key": "medicine:amlodipine", "approve": True})
    assert r.json()["result"] == "stopped"


async def test_note_scrubs_otp(client, at):
    at("2026-10-02 09:00")
    r = await client.put(f"/v2/dash/{FAM}/{ELDER}/notes", json={"actor": ACTOR, "subject_id": ELDER, "slug": "food", "title": "Food", "body": "Likes poha. OTP 556677"})
    assert r.json()["saved"]
    notes = (await client.get(f"/v2/dash/{FAM}/{ELDER}/overview")).json()["notes"]
    assert "556677" not in notes[0]["body"]


async def test_home_summary(client, db, at):
    from app.care import store

    at("2026-10-02 07:00")
    await client.get(f"/v2/dash/{FAM}/{ELDER}/overview")  # imports Metformin 08:30
    at("2026-10-02 08:30")
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="reminder_sent", summary="dose due: Metformin 500 mg at 08:30", ref="r1")
    at("2026-10-02 08:40")
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="dose_taken", summary="metformin: taken", payload={"medicine": "metformin"})
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="vital", summary="bp 140/90", payload={"kind": "bp", "value": "140/90"})
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="vital", summary="bp 132/84", payload={"kind": "bp", "value": "132/84"})
    at("2026-10-02 12:00")
    r = (await client.get(f"/v2/dash/{FAM}/{ELDER}/home")).json()
    assert r["doses"] == [{"id": "medicine:metformin@08:30", "time": "08:30", "name": "Metformin", "dose": "500 mg", "status": "taken"}]
    assert r["week"]["adherence"][-1] == 100 and r["week"]["scheduled"] == 7
    assert r["vitals"]["bp"]["value"] == "132/84" and r["vitals"]["bp"]["change"] == {"pct": 6, "dir": "down"}
    assert r["timeline"][0]["kind"] == "vital"


async def test_home_needs_refill_and_appointment_and_views(client, at):
    at("2026-10-02 07:00")
    await client.get(f"/v2/dash/{FAM}/{ELDER}/overview")  # imports Metformin 08:30
    r = await client.post(f"/v2/dash/{FAM}/{ELDER}/stock", json={"actor": ACTOR, "key": "medicine:metformin", "count": 3})
    assert r.status_code == 200 and r.json()["daysLeft"] == 3 and r.json()["low"]
    r = await client.post(f"/v2/dash/{FAM}/{ELDER}/facts", json={"actor": ACTOR, "domain": "appointment", "name": "Dr Iyer 2026-10-03",
                          "details": {"doctor": "Dr Iyer", "when": "2026-10-03T11:00", "place": "Iyer Clinic"}, "sentence": "Dr Iyer on 3 Oct at 11"})
    assert r.status_code == 200
    home = (await client.get(f"/v2/dash/{FAM}/{ELDER}/home")).json()
    kinds = {n["kind"]: n for n in home["needsYou"]}
    assert kinds["refill"]["title"] == "Metformin: 3 days left"
    assert kinds["appointment"]["title"].startswith("Dr Iyer, 03/10 at 11:00")
    assert home["nextAppointment"]["doctor"] == "Dr Iyer"
    assert not [l for l in home["followUps"] if l["kind"] in ("refill", "appointment")]
    q = await client.post(f"/v2/dash/{FAM}/{ELDER}/appointments/question", json={"actor": ACTOR, "key": "Dr Iyer", "question": "Knee pain?"})
    assert q.json()["questions"] == ["Knee pain?"]
    team = (await client.get(f"/v2/dash/{FAM}/{ELDER}/care-team")).json()
    assert team["upcoming"][0]["questions"] == ["Knee pain?"]
    t = (await client.post(f"/v2/dash/{FAM}/{ELDER}/family-tasks", json={"actor": ACTOR, "title": "Call Maa", "assignee": CG, "due": "2026-10-02T20:00"})).json()
    done = (await client.post(f"/v2/dash/{FAM}/{ELDER}/family-tasks/{t['id']}/done", json={"actor": ACTOR})).json()
    assert done["status"] == "done"
    assert (await client.get(f"/v2/dash/{FAM}/{ELDER}/spending?month=2026-13")).status_code == 400
    card = (await client.get(f"/v2/dash/{FAM}/{ELDER}/emergency")).json()
    assert card["allergies"][0]["allergen"] == "milk"


async def test_speech_view(client, at):
    at("2026-10-02 09:00")
    assert (await client.get(f"/v2/dash/{FAM}/{ELDER}/speech")).json() == {}
    body = {"actor": ACTOR, "domain": "language", "name": "preferred", "details": {"language": "hi", "dialect": "mwr"}, "sentence": "Speaks Marwari"}
    assert (await client.post(f"/v2/dash/{FAM}/{ELDER}/facts", json=body)).status_code == 200
    assert (await client.get(f"/v2/dash/{FAM}/{ELDER}/speech")).json() == {"dialect": "mwr", "language": "hi"}
