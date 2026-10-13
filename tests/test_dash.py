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
    host = SimHost(World(backend_record={"schedules": [
        {"scheduleId": "s1", "type": "MEDICINE", "title": "Metformin", "time": "8:30 AM", "dosage": "500 mg", "daysOfWeek": [], "sourceKey": None}]}))
    monkeypatch.setattr(dash, "make_host", lambda: host)
    from app.care import store

    # the first-turn import reads the care schedules only; what the old profile held moved with app.care.memory_unify
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="allergy", key="allergy:milk", value={"allergen": "milk"},
                           text="Allergic to milk", source_kind="caregiver_said", stated_by=CG)

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


async def test_the_language_and_voice_setting_lives_in_the_care_record_and_syncs_to_the_backend(client, at):
    """Trust plan A1: the engine language/voice fact is the truth; the backend keeps a copy written by the sync."""
    at("2026-10-02 09:00")
    r = await client.put(f"/v2/dash/{FAM}/{ELDER}/speech", json={"actor": ACTOR, "language": "Bengali", "voice": "always"})
    assert r.status_code == 200
    person = (await client.get(f"/v2/person/{FAM}/{ELDER}")).json()
    assert person["language"] == "bn" and person["voice"] == "always" and person["known"] is True
    synced = [c["args"] for c in client.host.world.calls if c["tool"] == "set_voice_preference"]
    assert {"mode": "always"} in synced and any(a.get("language") == "bn" for a in synced)
    # a later language change keeps the voice setting
    await client.put(f"/v2/dash/{FAM}/{ELDER}/speech", json={"actor": ACTOR, "language": "Hindi"})
    person = (await client.get(f"/v2/person/{FAM}/{ELDER}")).json()
    assert person["language"] == "hi" and person["voice"] == "always"
    assert (await client.get(f"/v2/dash/{FAM}/{ELDER}/speech")).json() == {"language": "hi", "voice": "always"}


async def test_a_fact_the_caregiver_says_was_never_true_goes_on_the_never_use_list(client, at, db):
    """The dashboard's "What Saheli learned" card: Not true → the rejected list, so it is never proposed or used again."""
    from app.care import rejected, store

    at("2026-10-02 09:00")
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="preference", key="preference:ginger_tea", value={},
                           text="Likes ginger tea", source_kind="inferred")
    r = await client.post(f"/v2/dash/{FAM}/{ELDER}/facts/stop", json={"actor": ACTOR, "domain": "preference", "name": "ginger_tea",
                                                                      "reason": "not true", "wrong": True})
    assert r.status_code == 200
    assert [f.key for f in await rejected.rejected(db, FAM, ELDER)] == ["preference:ginger_tea"]
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="routine", key="routine:walk", value={}, text="Evening walk",
                           source_kind="inferred")
    await client.post(f"/v2/dash/{FAM}/{ELDER}/facts/stop", json={"actor": ACTOR, "domain": "routine", "name": "walk", "reason": "stopped"})
    assert [f.key for f in await rejected.rejected(db, FAM, ELDER)] == ["preference:ginger_tea"]  # an ended fact is not "wrong"


# ── 14 Oct audit (Dashboard Builder): roster duplicates and care-team helpers ──

async def test_a_member_listed_twice_is_saved_once(db):
    from app.care import store

    elder = {"id": "e-dup", "name": "Asha", "role": "elder"}
    cg = {"id": "c-dup", "name": "Ravi", "role": "primary caregiver"}
    await store.save_roster(db, "fam-dup", elder, [elder, cg, {"id": "c-dup", "name": "", "role": "caregiver", "relation": "son"}])
    r = await store.roster(db, "fam-dup")
    assert [m["id"] for m in r.members] == ["e-dup", "c-dup"]
    assert r.members[1]["name"] == "Ravi" and r.members[1]["relation"] == "son", "the first entry wins; a later one only fills gaps"


async def test_care_team_helpers_are_people_only_with_their_key(db):
    from app.care import features, store

    fam, sid = "fam-help", "e-help"
    await store.write_fact(db, family_id=fam, subject_id=sid, domain="home", key="home:home_address",
                           value={"address": "12 Park Road", "city": "Pune"}, text="Lives at 12 Park Road", source_kind="caregiver_said")
    await store.write_fact(db, family_id=fam, subject_id=sid, domain="home", key="home:lives_with", value={"with": "family"},
                           text="Lives with family", source_kind="caregiver_said")
    await store.write_fact(db, family_id=fam, subject_id=sid, domain="home", key="home:cook_meena", value={"name": "Meena", "role": "cook"},
                           text="Meena cooks lunch", source_kind="caregiver_said")
    out = await features.care_team(db, fam, sid)
    assert [h["name"] for h in out["helpers"]] == ["Meena"]
    assert out["helpers"][0]["key"] == "home:cook_meena" and out["helpers"][0]["domain"] == "home"


async def test_report_counts_unmarked_doses_and_gives_vital_units(db, at):
    from app.care import features, store

    at("2026-10-14 20:00")
    fam, sid = "fam-rep", "e-rep"
    await store.write_fact(db, family_id=fam, subject_id=sid, domain="medicine", key="medicine:amlo",
                           value={"name": "Amlodipine", "dose": "5 mg", "times": ["08:00", "21:00"]}, text="Amlodipine 5 mg at 08:00 and 21:00",
                           source_kind="caregiver_said")
    await store.record_event(db, family_id=fam, subject_id=sid, kind="vital", summary="BP 140/90", payload={"kind": "bp", "value": "140/90"})
    out = await features.report(db, fam, sid, days=2)
    today = out["adherence"]["byDay"][-1]
    assert today["expected"] == 2 and today["unmarked"] == 1, "only the 08:00 dose is due by 20:00, and nothing was marked"
    assert out["adherence"]["byDay"][0]["unmarked"] == 2
    assert out["vitals"][0]["unit"] == "mmHg"
