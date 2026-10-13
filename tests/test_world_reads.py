"""Plan B1.2: every dashboard and admin read, for every family and person of the world's ci slice (120 families in 30
languages, every household type and data shape). No AI. A page must never break on a real family's data."""

import time

import httpx
import pytest

from app.api import admin as admin_api, dash
from app.core import clock
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.main import app
from app.sim.world import SimHost, World
from eval.world import make
from eval.world.seed_engine import Seeded, ist, seed_family
from eval.world.slice import slice_ids

PAST = make.day(-3)
ROUTES = ["overview", f"overview?day={PAST}", "home", "portrait", "emergency", "stock", "memory-health", "memory-history", "skills", "care-team",
          "report", "wellbeing", "patterns", "family-tasks", "spending", "boundaries", f"events?day={PAST}", "outcomes", "work"]
SLOW_S = 1.5


@pytest.fixture
async def world_client(db, monkeypatch):
    host = SimHost(World())
    monkeypatch.setattr(dash, "make_host", lambda: host)

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    app.dependency_overrides[admin_api.verify_admin_caller] = lambda: None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=30) as c:
            c.host = host
            yield c
    finally:
        app.dependency_overrides.clear()
        clock.set_now(None)


async def test_every_read_for_every_family_of_the_ci_slice(world_client, db):
    world = make.build()
    fams = [f for f in world if f["id"] in set(slice_ids("ci", world))]
    out = Seeded()
    for fam in fams:
        await seed_family(db, fam, out)
    clock.set_now(ist(make.TODAY.isoformat(), "12:00"))
    failures, slow, calls = [], [], 0
    for fam in fams:
        world_client.host.world.backend_record = {"schedules": fam["backend"]["schedules"]}
        fid = fam["familyId"]
        subjects = list(fam["truth"]) or [fam["people"][0]["id"]]
        for sid in subjects:
            for route in ROUTES:
                url = f"/v2/dash/{fid}/{sid}/{route}"
                t0 = time.monotonic()
                r = await world_client.get(url)
                dt = time.monotonic() - t0
                calls += 1
                if r.status_code != 200 or not isinstance(r.json(), dict):
                    failures.append(f"{fam['id']} ({fam['household']}, {','.join(fam['shapes']) or 'plain'}) {route}: {r.status_code} {r.text[:160]}")
                elif dt > SLOW_S:
                    slow.append(f"{fam['id']} {route}: {dt:.2f}s")
        r = await world_client.get(f"/v2/admin/families/{fid}/care-record", params={"subject": subjects[0]})
        calls += 1
        if r.status_code != 200:
            failures.append(f"{fam['id']} admin care-record: {r.status_code} {r.text[:160]}")
    r = await world_client.get("/v2/admin/live", params={"minutes": 120})
    assert r.status_code == 200, r.text[:300]
    assert not failures, f"{len(failures)} of {calls} reads failed:\n" + "\n".join(failures[:25])
    assert len(slow) <= calls * 0.01, f"{len(slow)} reads slower than {SLOW_S}s:\n" + "\n".join(slow[:15])
    assert out.families == len(fams) and calls > 2000
