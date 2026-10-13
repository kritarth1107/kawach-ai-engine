"""Load world families into an engine Postgres through the store APIs (never raw SQL), so every invariant holds.

Refuses any database that is not local (localhost, 127.0.0.1, or a CI service container): the world must never
reach production or a shared database.

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_world \
    PYTHONPATH=. python -m eval.world.seed_engine ci          # a slice name, or family ids (w0001 w0002 …)
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")  # seeding never calls a model

import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.care import store
from app.core import clock
from eval.world.make import TODAY, load
from eval.world.slice import slice_ids

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "db", "postgres"}
BRAIN_ROLE = {"elder": "elder", "primary caregiver": "primary caregiver", "caregiver": "caregiver", "family member": "family member"}


def assert_local(url: str) -> None:
    host = (urlparse(url.replace("+asyncpg", "")).hostname or "").lower()
    if host not in LOCAL_HOSTS:
        raise SystemExit(f"refusing to seed the world into {host or url!r}: only a local or CI database")


def ist(day: str, hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=clock.IST).astimezone(timezone.utc)


def relations(fam: dict) -> dict[str, tuple[str, str]]:
    """Who each member is to the primary caregiver who added them, as the backend stores it (the invite's relationship).

    The world's own "relation" is to the person cared for (a child is "son") or, for an elder, to their child; here both
    become relative to the primary caregiver: a second child is their brother or sister, a patient their spouse's spouse."""
    people = fam["people"]
    primary = next((p for p in people if p["role"] == "primary caregiver"), None)
    if primary is None:
        return {}
    out = {}
    for p in people:
        if p is primary:
            continue
        rel = p["relation"]
        if rel == "patient":
            rel = primary["relation"]  # the primary caregiver is the patient's spouse
        elif p["role"] == "caregiver" and rel in ("son", "daughter") and not primary["careRecipient"]:
            rel = "sister" if p["gender"] == "f" else "brother"  # both are children of the person cared for
        out[p["id"]] = (rel, primary["id"])
    return out


def roster_of(fam: dict) -> tuple[dict, list[dict]]:
    rel = relations(fam)

    def member(p: dict) -> dict:
        r = rel.get(p["id"])
        return {"id": p["id"], "name": p["name"], "role": BRAIN_ROLE[p["role"]], **({"relation": r[0], "relation_to": r[1]} if r else {})}

    members = [member(p) for p in fam["people"]]
    cared = next((p for p in fam["people"] if p["careRecipient"]), fam["people"][0])
    return member(cared), members


@dataclass
class Seeded:
    families: int = 0
    facts: dict[str, int] = field(default_factory=dict)
    events: int = 0
    turns: int = 0
    loops: int = 0
    notes: int = 0


async def prepare(engine) -> None:
    """Every table the brain and the dashboard routes read, plus the v2 migrations (as tests/conftest.py does)."""
    from app.care import baselines, memory_index, models, versions, why  # noqa: F401
    from app.db.session import Base
    from app.learn import models as learn_models  # noqa: F401
    from app.llm import spend  # noqa: F401
    from app.models import entities  # noqa: F401
    from app.specialists import channels  # noqa: F401
    from app.tasks import models as task_models, sandbox  # noqa: F401
    from eval.lab import queue as lab_queue  # noqa: F401

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
        from app.db.migrate import run_v2_migrations

        await run_v2_migrations(conn)


async def wipe(session: AsyncSession, family_id: str) -> None:
    """Remove an earlier seed of this family so a re-run starts clean (local databases only)."""
    for table in ("care_facts", "care_events", "open_loops", "turns", "memory_notes", "family_rosters"):
        await session.execute(text(f"DELETE FROM {table} WHERE family_id = :f"), {"f": family_id})  # noqa: S608 (fixed table names)


async def seed_family(session: AsyncSession, fam: dict, out: Seeded) -> None:
    fid = fam["familyId"]
    await wipe(session, fid)
    elder, members = roster_of(fam)
    await store.save_roster(session, fid, elder, members)
    eng = fam["engine"]
    # facts oldest first, so a later statement supersedes or waits beside an earlier one exactly as in life
    for f in sorted(eng["facts"], key=lambda x: -x["daysAgo"]):
        clock.set_now(ist((TODAY - timedelta(days=f["daysAgo"])).isoformat(), "10:00"))
        w = await store.write_fact(
            session, family_id=fid, subject_id=f["subject"], domain=f["domain"], key=f["key"], value=f["value"], text=f["text"],
            source_kind=f["source"], stated_by=f["statedBy"], source_ref=f"world:{fam['id']}",
        )
        out.facts[w.result] = out.facts.get(w.result, 0) + 1
    for n in eng["notes"]:
        clock.set_now(ist((TODAY - timedelta(days=3)).isoformat(), "02:15"))
        await store.upsert_note(session, family_id=fid, subject_id=n["subject"], slug=n["slug"], title=n["title"], body_md=n["body"],
                                actor_id="saheli", source="import", reason="world seed")
        out.notes += 1
    for t in eng["turns"]:
        clock.set_now(ist(t["day"], t["time"]))
        thread = t["who"] if t["role"] == "user" else t["thread"]
        await store.add_turn(session, family_id=fid, thread_id=thread, role=t["role"], text=t.get("text") or t["brief"],
                             speaker_id=t["who"] if t["role"] == "user" else None, meta={"world": True})
        out.turns += 1
    for i, e in enumerate(eng["events"]):
        payload = dict(e.get("payload") or {})
        if e["kind"].startswith("dose_"):
            payload.setdefault("day", e["day"])
            payload.setdefault("time", e["time"])
        await store.record_event(session, family_id=fid, subject_id=e["subject"], kind=e["kind"], summary=e["summary"], payload=payload,
                                 actor_id=e["subject"], at=ist(e["day"], e["time"]), ref=f"world:{fam['id']}:{i}")
        out.events += 1
    clock.set_now(ist(TODAY.isoformat(), "06:00"))
    for lp in eng["loops"]:
        d, tm = lp["wakeAt"].split("T")
        await store.open_loop(session, family_id=fid, subject_id=lp["subject"], kind=lp["kind"], title=lp["title"], detail=lp.get("detail") or {},
                              wake_at=ist(d, tm[:5]), dedupe_key=f"world:{fam['id']}:{lp['kind']}:{lp['title']}")
        out.loops += 1
    out.families += 1


async def seed(ids: list[str], url: str, *, fresh_schema: bool = True) -> Seeded:
    assert_local(url)
    engine = create_async_engine(url)
    if fresh_schema:
        await prepare(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    out = Seeded()
    fams = load(ids)
    try:
        for fam in fams:
            async with maker() as session:
                await seed_family(session, fam, out)
                await session.commit()
    finally:
        clock.set_now(None)
        await engine.dispose()
    return out


def main(argv: list[str]) -> int:
    url = os.getenv("DATABASE_URL") or os.getenv("TEST_DATABASE_URL") or ""
    if not url:
        print("set DATABASE_URL to a local Postgres")
        return 2
    ids = argv if argv and argv[0].startswith("w") else slice_ids(argv[0] if argv else "ci")
    t0 = time.monotonic()
    out = asyncio.run(seed(ids, url))
    print(f"seeded {out.families} families in {time.monotonic() - t0:.1f}s: facts {out.facts}, notes {out.notes}, turns {out.turns}, "
          f"events {out.events}, loops {out.loops}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
