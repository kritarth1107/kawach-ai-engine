"""The engine's dashboard routes for Food & nutrition (INTERFACES.md §3), against their contracts."""

import json
import uuid

import httpx
import pytest

from app import contracts
from app.api import dash
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.main import app
from app.sim.world import SimHost, World
from tests import nutrition_fakes

nfake = nutrition_fakes.nfake  # the fixture

FAM = "fam-api-food"
NANI = "e-api-food"
SON = {"id": "c-api-food", "name": "Arjun"}


@pytest.fixture
async def client(db, monkeypatch):
    host = SimHost(World())
    monkeypatch.setattr(dash, "make_host", lambda: host)

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=30) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def url(path=""):
    return f"/v2/dash/{FAM}/{NANI}/nutrition{path}"


async def test_add_change_delete_undo_and_the_view_keeps_its_contract(client, db, at, nfake):
    at("2026-10-14 09:00")
    r = await client.post(url("/entries"), json={"actor": SON, "items": [{"name": "boiled egg", "quantity": 2, "unit": "piece"},
                                                                         {"name": "tea", "quantity": 1, "unit": "cup"}], "meal": "breakfast"})
    assert r.status_code == 200 and r.json()["result"] == "saved", r.text
    eid = r.json()["saved"][0]["id"]
    r = await client.post(url(f"/entries/{eid}"), json={"actor": SON, "set": {"quantity": 3}})
    assert r.status_code == 200 and r.json()["result"] == "changed"
    r = await client.post(url(f"/entries/{eid}/delete"), json={"actor": SON})
    assert r.status_code == 200 and r.json()["result"] == "removed"
    vid = r.json()["version_id"]
    h = await client.get(f"/v2/dash/{FAM}/{NANI}/memory-history", params={"kind": "food"})
    assert h.status_code == 200 and h.json()["changes"][0]["id"] == vid and h.json()["changes"][0]["canUndo"]
    p = await client.get(f"/v2/dash/{FAM}/{NANI}/memory-history/{vid}/preview")
    assert p.json()["effect"] == "change"
    u = await client.post(f"/v2/dash/{FAM}/{NANI}/memory-history/{vid}", json={"actor": SON, "mode": "undo"})
    assert u.status_code == 200 and u.json()["result"] == "done"
    r = await client.post(url("/targets"), json={"actor": SON, "set": {"protein_g": 60}, "set_by": "doctor", "how": "strict"})
    assert r.status_code == 200 and r.json()["saved"]
    r = await client.post(url("/settings"), json={"actor": SON, "sizes": {"glass_ml": 300}, "counters": ["egg", "water"]})
    assert r.status_code == 200 and r.json()["setting"]["counters"] == ["egg", "water"]
    r = await client.post(url("/pantry"), json={"actor": SON, "action": "add", "item": "Dates", "count": 10})
    assert r.status_code == 200 and r.json()["result"] == "added"
    view = await client.get(url(), params={"days": 7})
    assert view.status_code == 200
    body = view.json()
    egg = next(e for e in body["today"]["meals"][0]["entries"] if e["name"] == "Egg, whole, boiled")
    assert egg["qty"] == "3" and body["pantry"][0]["name"] == "Dates"
    assert [c["key"] for c in body["counters"]] == ["egg", "water"]
    schema = contracts.load("dash.nutrition")
    assert schema is not None
    errs = contracts.validate(body, schema)
    open("/tmp/nutri_errs.json", "w").write(json.dumps(errs))
    assert errs == [], errs
    rep = await client.get(url("/report"), params={"days": 14})
    assert rep.status_code == 200 and rep.json()["days"] == 14
    rschema = contracts.load("dash.nutrition-report")
    assert rschema is not None and contracts.validate(rep.json(), rschema) == []


async def test_autocomplete_finds_the_familys_and_shared_foods(client, db, at, nfake):
    at("2026-10-14 09:00")
    await client.post(url("/entries"), json={"actor": SON, "items": [{"name": "boiled egg", "quantity": 2, "unit": "piece"}]})
    r = await client.get(url("/foods"), params={"q": "egg"})
    assert r.status_code == 200 and r.json()["foods"][0]["name"] == "Egg, whole, boiled" and r.json()["foods"][0]["kcalPer100g"] == 155
    assert (await client.get(url("/foods"), params={"q": ""})).json() == {"foods": []}


async def test_waiting_did_they_eat_it_yes_part_no_on_the_dashboard(client, db, at, nfake):
    from tests.test_nutrition_log import _order

    at("2026-10-14 13:00")
    t = _order(subject=NANI)
    t.family_id = FAM
    db.add(t)
    await db.flush()
    view = (await client.get(url())).json()
    assert view["waiting"][0]["task_id"] == str(t.id)
    r = await client.post(url("/consumed"), json={"actor": SON, "task_id": str(t.id), "consumed": "yes"})
    assert r.status_code == 200 and r.json()["result"] == "saved"
    view = (await client.get(url())).json()
    assert view["waiting"] == [] and view["today"]["meals"]
    assert (await client.post(url("/consumed"), json={"actor": SON, "task_id": str(uuid.uuid4()), "consumed": "yes"})).status_code == 404


async def test_bad_input_is_refused_plainly(client, db, at, nfake):
    at("2026-10-14 09:00")
    assert (await client.get(url(), params={"day": "yesterday"})).status_code == 400
    assert (await client.post(url("/entries/not-an-id"), json={"actor": SON, "set": {"quantity": 2}})).status_code == 404
    r = await client.post(url("/entries"), json={"actor": SON, "items": []})
    assert r.status_code == 422
