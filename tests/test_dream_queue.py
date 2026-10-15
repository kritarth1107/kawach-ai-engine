"""The night is a queue: several families at once, a second task skips one already claimed, leftovers go first."""

import asyncio
import os
import time

import pytest
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.care import dream, store
from app.care.models import CareEvent, FamilyRoster
from app.core import clock
from tests.conftest import TEST_DB

pytestmark = pytest.mark.asyncio


async def test_someone_left_over_is_first_the_next_night(db):
    day = "2026-10-15"
    for fid, name in (("fam-q-none", "None"), ("fam-q-old", "Old"), ("fam-q-new", "New"), ("fam-q-done", "Done")):
        await store.save_roster(db, fid, {"id": fid + "-e", "name": name, "role": "elder"}, [{"id": fid + "-e", "name": name, "role": "elder"}])
    await store.record_event(db, family_id="fam-q-old", subject_id="fam-q-old-e", kind="dream", summary="old", ref="dream:2026-10-01")
    await store.record_event(db, family_id="fam-q-new", subject_id="fam-q-new-e", kind="dream", summary="new", ref="dream:2026-10-14")
    await store.record_event(db, family_id="fam-q-done", subject_id="fam-q-done-e", kind="dream", summary="done", ref=f"dream:{day}")
    await db.commit()
    pending, _ = await dream._pending(db, day)
    ids = [r.family_id for r in pending if r.family_id.startswith("fam-q-")]
    assert ids == ["fam-q-none", "fam-q-old", "fam-q-new"]
    assert dream.vertex_batch_fits() is False


async def test_a_second_connection_cannot_claim_the_same_family():
    engine = create_async_engine(TEST_DB)
    key = "dream:test:claim-once"
    one = await engine.connect()
    two = await engine.connect()
    try:
        assert await dream._try_claim(one, key)
        assert await dream._try_claim(two, key) is False
        await dream._release_claim(one, key)
        assert await dream._try_claim(two, key) is True
        await dream._release_claim(two, key)
    finally:
        await one.close()
        await two.close()
        await engine.dispose()


def _ist(y, m, d, hh, mm):
    from datetime import datetime
    return datetime(y, m, d, hh, mm, tzinfo=clock.IST)


def _server_url(database: str) -> str:
    """The CI Postgres is TEST_DATABASE_URL (port 5432). A hardcoded 5433 never connects there."""
    from sqlalchemy.engine import make_url

    url = make_url(os.getenv("TEST_DATABASE_URL") or TEST_DB)
    return url.set(database=database).render_as_string(hide_password=False)


async def test_the_scale_database_is_on_the_test_server(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/kawach_test")
    assert _server_url("postgres") == "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    assert _server_url("kawach_test_dream") == "postgresql+asyncpg://postgres:postgres@localhost:5432/kawach_test_dream"


async def _scale_engine():
    admin = create_async_engine(_server_url("postgres"), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        exists = (await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = 'kawach_test_dream'"))).first()
        if not exists:
            await conn.execute(text("CREATE DATABASE kawach_test_dream"))
    await admin.dispose()
    engine = create_async_engine(_server_url("kawach_test_dream"))
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        from app.db.session import Base
        from app.db.migrate import run_v2_migrations
        import app.care.models  # noqa: F401
        import app.learn.models  # noqa: F401
        import app.llm.spend  # noqa: F401
        import app.tasks.models  # noqa: F401

        await conn.run_sync(Base.metadata.create_all)
        await run_v2_migrations(conn)
    return engine


async def test_families_run_together_and_a_second_task_does_not_repeat_one(monkeypatch):
    monkeypatch.setenv("DREAM_PARALLEL", "4")
    monkeypatch.setenv("DAILY_SCORE", "off")
    engine = await _scale_engine()
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    seen: list[str] = []
    current = 0
    peak = 0

    async def fake(sessions, roster, day, *, with_models=True):
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.2)
        current -= 1
        seen.append(roster.family_id)
        async with sessions() as session:
            await store.record_event(session, family_id=roster.family_id, subject_id=(roster.elder or {}).get("id") or "family",
                                     kind="dream", summary="reviewed", ref=f"dream:{day}")
            await session.commit()
        return {"patterns": 0, "facts": 0, "errors": []}

    async def quiet(*a, **k):
        return {}

    monkeypatch.setattr(dream, "dream_family", fake)
    monkeypatch.setattr(dream, "upkeep", quiet)
    monkeypatch.setattr("app.learn.jobs.nightly", quiet)
    try:
        async with sessions() as session:
            await session.execute(delete(CareEvent))
            await session.execute(delete(FamilyRoster))
            for i in range(6):
                fid = f"scale-{i}"
                await store.save_roster(session, fid, {"id": fid + "-e", "name": "A", "role": "elder"},
                                        [{"id": fid + "-e", "name": "A", "role": "elder"}])
            await session.commit()
        clock.set_now(_ist(2026, 10, 16, 2, 0))
        started = time.monotonic()
        one, two = await asyncio.gather(dream.dream_all(sessions, budget_s=30), dream.dream_all(sessions, budget_s=30))
        elapsed = time.monotonic() - started
        assert one["done_now"] + two["done_now"] == 6
        assert sorted(seen) == [f"scale-{i}" for i in range(6)]
        assert peak > 1
        assert elapsed < 1.2  # six families at 0.2s, four at a time, not one after another
        again = await dream.dream_all(sessions, budget_s=30)
        assert again["done_now"] == 0
    finally:
        async with sessions() as session:
            await session.execute(delete(CareEvent))
            await session.execute(delete(FamilyRoster))
            await session.commit()
        await engine.dispose()


@pytest.mark.skipif(not os.getenv("DREAM_SCALE"), reason="set DREAM_SCALE=1 to measure 5,000 families")
async def test_five_thousand_families_finish_inside_two_hours(monkeypatch):
    monkeypatch.setenv("DREAM_PARALLEL", "8")
    monkeypatch.setenv("DAILY_SCORE", "off")
    monkeypatch.setenv("MEMORY_EMBEDDINGS", "off")
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    from app.llm import router, spend
    from app.llm.router import LLMReply

    class Model:
        async def complete(self, route, **kw):
            return LLMReply(text="Quiet day, nothing logged.", tool_calls=[], model="fake")

    router.register_provider("fake", Model())
    router.reset_breakers()
    spend.reset()
    engine = await _scale_engine()
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def quiet(*a, **k):
        return {}

    monkeypatch.setattr(dream, "upkeep", quiet)
    monkeypatch.setattr("app.learn.jobs.nightly", quiet)
    try:
        async with sessions() as session:
            await session.execute(delete(CareEvent))
            await session.execute(delete(FamilyRoster))
            await session.commit()
            for batch in range(50):
                for i in range(100):
                    n = batch * 100 + i
                    fid = f"syn-{n}"
                    await store.save_roster(session, fid, {"id": fid + "-e", "name": "A", "role": "elder"},
                                            [{"id": fid + "-e", "name": "A", "role": "elder"}])
                await session.commit()
        clock.set_now(_ist(2026, 10, 16, 2, 0))
        started = time.monotonic()
        stats = await dream.dream_all(sessions, budget_s=7200, with_models=True)
        elapsed = time.monotonic() - started
        assert stats["done_now"] == 5000 and stats["left"] == 0
        assert elapsed < 7200
        print(f"DREAM_SCALE {stats['done_now']} families in {elapsed:.1f}s workers={stats['workers']}")
    finally:
        router._providers.pop("fake", None)
        await engine.dispose()
