"""The start-up schema step (D-13 review): it never waits on live queries for a change that is already made, a missing
column waits only briefly and then stops the start, and a half-built index is built again."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.care import models, why  # noqa: F401  (tables a start creates on a fresh database)
from app.db import migrate
from app.models import entities  # noqa: F401
from app.tasks import models as task_models  # noqa: F401
from tests.conftest import TEST_DB


@pytest.fixture
async def engine():
    e = create_async_engine(TEST_DB)
    yield e
    await e.dispose()


async def catalog(engine):
    async with engine.connect() as c:
        return await migrate.read_catalog(c)


async def test_nothing_waits_on_live_queries_when_the_schema_is_there(engine):
    await migrate.ensure_schema(engine)  # the first run adds what this test database lacks
    async with engine.connect() as live:  # live traffic: open transactions reading the tables a start used to ALTER
        await live.execute(text("SELECT 1 FROM care_events LIMIT 1"))
        await live.execute(text("SELECT 1 FROM family_memories LIMIT 1"))
        await asyncio.wait_for(migrate.ensure_schema(engine), timeout=15)
        await live.rollback()


async def test_a_missing_column_waits_briefly_then_stops_the_start(engine, monkeypatch):
    monkeypatch.setattr(migrate, "LOCK_TIMEOUT", "200ms")
    monkeypatch.setattr(migrate, "TRIES", 2)
    spec = [("zz_schema_probe", "trace_id", "VARCHAR(160)")]
    async with engine.begin() as c:
        await c.execute(text("DROP TABLE IF EXISTS zz_schema_probe"))
        await c.execute(text("CREATE TABLE zz_schema_probe (id int)"))
    try:
        async with engine.connect() as live:
            await live.execute(text("SELECT 1 FROM zz_schema_probe"))  # a long query holds the table
            with pytest.raises(RuntimeError, match="zz_schema_probe.trace_id"):
                await asyncio.wait_for(migrate.ensure_columns(engine, spec, await catalog(engine)), timeout=15)
            await live.rollback()
        await migrate.ensure_columns(engine, spec, await catalog(engine))
        assert ("zz_schema_probe", "trace_id") in (await catalog(engine)).columns
    finally:
        async with engine.begin() as c:
            await c.execute(text("DROP TABLE IF EXISTS zz_schema_probe"))


async def test_a_half_built_index_is_built_again(engine):
    spec = [("zz_index_probe_uq", "zz_index_probe", "UNIQUE", "(v)")]
    async with engine.begin() as c:
        await c.execute(text("DROP TABLE IF EXISTS zz_index_probe"))
        await c.execute(text("CREATE TABLE zz_index_probe (v int)"))
        await c.execute(text("INSERT INTO zz_index_probe VALUES (1), (1)"))
    try:
        with pytest.raises(Exception):  # duplicates: the concurrent build fails half way and leaves an invalid index
            await migrate.build_index(engine, *spec[0])
        assert (await catalog(engine)).indexes["zz_index_probe_uq"] is False
        async with engine.begin() as c:
            await c.execute(text("DELETE FROM zz_index_probe"))
            await c.execute(text("INSERT INTO zz_index_probe VALUES (1)"))
        await migrate.ensure_indexes(engine, spec)
        assert (await catalog(engine)).indexes["zz_index_probe_uq"] is True
    finally:
        async with engine.begin() as c:
            await c.execute(text("DROP TABLE IF EXISTS zz_index_probe"))
