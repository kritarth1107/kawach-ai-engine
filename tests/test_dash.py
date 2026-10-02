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
