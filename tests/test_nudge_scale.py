"""A tick at 5,000 families reads only people whose time has come, and two tasks cannot both send."""

import asyncio
import time
from datetime import datetime, timedelta

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.care import nudges, store
from app.care.models import CareEvent, FamilyRoster, NudgeDue, NudgeLedger, Turn
from app.core import clock
from app.db.session import Base
from tests.conftest import TEST_DB

pytestmark = pytest.mark.asyncio


class Host:
    def __init__(self, inbound: str):
        self.sent: list[str] = []
        self.inbound = inbound

    async def call(self, name, payload, **_kw):
        if name == "send_whatsapp":
            self.sent.append(payload.get("to") or "")
            return {"delivered": True, "wamid": "w"}
        if name == "nudge_gate":
            return {"window_open": True, "mode": "on", "last_inbound_at": self.inbound}
        return {}


async def test_a_tick_with_5000_families_reads_only_who_is_due(db):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    now = clock.now()
    later = now + timedelta(hours=6)
    rosters, dues, ids = [], [], []
    for i in range(5000):
        fid, pid = f"zzscale-{i:04d}", f"zp{i:04d}"
        ids.append(fid)
        rosters.append({"family_id": fid, "elder": {"id": pid, "name": "A", "role": "elder"}, "members": [], "updated_at": now})
        dues.append({"family_id": fid, "person_id": pid, "next_due": later, "claimed_until": None})
    await db.execute(insert(FamilyRoster), rosters)
    await db.execute(insert(NudgeDue), dues)
    await db.commit()
    host = Host(now.isoformat())

    started = time.perf_counter()
    out = await nudges.run(sessions, lambda _f: host, mode="on", now=now, families=set(ids))
    elapsed = time.perf_counter() - started

    assert out["people"] == 0 and out["due"] == 0 and host.sent == []
    assert elapsed < 60


async def test_two_tasks_send_one_nudge(monkeypatch):
    engine = create_async_engine(TEST_DB, pool_size=4)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    fid, pid = "zzscale-due", "zzscale-person"
    now = datetime(2026, 10, 16, 11, 0, tzinfo=clock.IST)
    past = now - timedelta(hours=2)
    inbound = past.isoformat()
    token_past = clock.set_now(past)
    token_now = None

    async def wipe():
        async with maker() as session:
            for model, col in (
                (NudgeDue, NudgeDue.family_id), (NudgeLedger, NudgeLedger.family_id),
                (Turn, Turn.family_id), (CareEvent, CareEvent.family_id), (FamilyRoster, FamilyRoster.family_id),
            ):
                await session.execute(delete(model).where(col == fid))
            await session.commit()

    await wipe()
    try:
        async with maker() as session:
            await store.save_roster(session, fid, {"id": pid, "name": "A", "role": "elder"}, [])
            await store.add_turn(session, family_id=fid, thread_id=pid, role="user", text="I went to the temple today", speaker_id=pid)
            await session.commit()
        clock.reset_now(token_past)
        token_now = clock.set_now(now)
        host = Host(inbound)

        async def fake_write(ctx):
            await asyncio.sleep(0.2)
            opt = ctx.options[0].id if ctx.options else "o1"
            return {"option_id": opt, "text": "How was the temple visit?"}, []

        monkeypatch.setattr(nudges, "write_and_check", fake_write)
        real_load = nudges.load_work
        door = asyncio.Event()
        arrived = {"n": 0}

        async def load_together(*args, **kwargs):
            work = await real_load(*args, **kwargs)
            arrived["n"] += 1
            if arrived["n"] >= 2:
                door.set()
            else:
                await door.wait()
            return work

        monkeypatch.setattr(nudges, "load_work", load_together)
        await asyncio.gather(
            nudges.run(maker, lambda _f: host, mode="on", now=now, families={fid}),
            nudges.run(maker, lambda _f: host, mode="on", now=now, families={fid}),
        )
        assert host.sent == [pid]
    finally:
        if token_now is not None:
            clock.reset_now(token_now)
        else:
            clock.reset_now(token_past)
        await wipe()
        await engine.dispose()
