"""Export the dashboard's page-test fixture (plan B1.6): the ui slice (40 families with the most different shapes),
each family's people and, per person cared for, the engine's answer for every dashboard read.

Seeds a throwaway local database, calls the engine routes in-process (no AI), writes one gzipped JSON.

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_export \
    PYTHONPATH=. python -m eval.world.export_dashboard ../kavach-dashboard/scripts/fixtures/world-ui.json.gz
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import gzip
import json
import sys
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import clock
from eval.world import make
from eval.world.seed_engine import Seeded, assert_local, ist, prepare, seed_family
from eval.world.slice import slice_ids

ROUTES = ["overview", "home", "portrait", "emergency", "stock", "memory-health", "memory-history", "skills", "care-team", "report", "wellbeing",
          "patterns", "family-tasks", "spending", "boundaries", "events", "outcomes", "work"]
PEOPLE_KEYS = ("id", "key", "first", "surname", "name", "role", "backendRole", "relation", "age", "gender", "language", "phone", "city", "careRecipient",
               "selfCare", "usesWhatsapp")


async def export(url: str, out: Path, slice_name: str = "ui") -> int:
    assert_local(url)
    engine = create_async_engine(url)
    await prepare(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    world = make.build()
    fams = [f for f in world if f["id"] in set(slice_ids(slice_name, world))]
    from app.api import admin as admin_api, dash
    from app.core.security import verify_api_secret
    from app.db.session import get_db
    from app.main import app
    from app.sim.world import SimHost, World

    host = SimHost(World())
    dash.make_host = lambda: host

    async def _db():
        async with sessions() as s:
            yield s

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    app.dependency_overrides[admin_api.verify_admin_caller] = lambda: None
    rows = []
    try:
        async with sessions() as s:
            out_seed = Seeded()
            for f in fams:
                await seed_family(s, f, out_seed)
            await s.commit()
        clock.set_now(ist(make.TODAY.isoformat(), "12:00"))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=30) as c:
            for f in fams:
                host.world.backend_record = {"schedules": f["backend"]["schedules"]}
                subjects = sorted(f["truth"]) or [f["people"][0]["id"]]
                engine_answers = {}
                for sid in subjects:
                    answers = {}
                    for route in ROUTES:
                        r = await c.get(f"/v2/dash/{f['familyId']}/{sid}/{route}")
                        if r.status_code == 200:
                            answers[route] = r.json()
                    engine_answers[sid] = answers
                rows.append({"id": f["id"], "familyId": f["familyId"], "household": f["household"], "language": f["language"], "shapes": f["shapes"],
                             "people": [{k: p.get(k) for k in PEOPLE_KEYS} for p in f["people"]], "places": f["places"], "engine": engine_answers})
    finally:
        app.dependency_overrides.clear()
        clock.set_now(None)
        await engine.dispose()
    lock = json.loads((make.DATA / "world.lock").read_text())
    data = json.dumps({"world": lock["sha256"], "today": make.TODAY.isoformat(), "slice": slice_name, "families": rows}, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":")).encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(out, "wb", mtime=0) as fh:
        fh.write(data)
    print(f"wrote {len(rows)} families ({len(data) // 1024} KB raw, {out.stat().st_size // 1024} KB gz) to {out}")
    return 0


def main(argv: list[str]) -> int:
    url = os.getenv("DATABASE_URL") or ""
    out = Path(argv[0] if argv else "../kavach-dashboard/scripts/fixtures/world-ui.json.gz")
    return asyncio.run(export(url, out))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
