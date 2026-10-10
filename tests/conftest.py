import os

os.environ.setdefault("TASK_BROWSE_FIRST", "off")  # older task tests start at the cart; the browse tests turn it on
os.environ.setdefault("MEMORY_EMBEDDINGS", "off")  # tests never call Vertex; memory tests inject a fake embedder
os.environ.setdefault("TASK_PARALLEL", "1")  # all test sessions share one connection: no parallel steps
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core import clock

TEST_DB = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_test")


@pytest.fixture
async def db():
    """A session inside a transaction that is rolled back after the test."""
    from app.care import models  # noqa: F401
    from app.tasks import models as task_models  # noqa: F401
    from app.specialists import channels  # noqa: F401
    from app.llm import spend  # noqa: F401
    from app.care import baselines  # noqa: F401
    from app.learn import models as learn_models  # noqa: F401
    from eval.lab import queue as lab_queue  # noqa: F401
    from app.care import memory_index  # noqa: F401
    from app.db.session import Base
    from app.models import entities  # noqa: F401

    engine = create_async_engine(TEST_DB)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
        from app.db.migrate import run_v2_migrations

        await run_v2_migrations(conn)
    conn = await engine.connect()
    trans = await conn.begin()
    session = AsyncSession(bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        await session.close()
        await trans.rollback()
        await conn.close()
        await engine.dispose()


@pytest.fixture
def at():
    """Pin the clock: at('2026-10-02 08:00') in IST."""

    def _set(ist_text: str):
        when = datetime.fromisoformat(ist_text).replace(tzinfo=clock.IST).astimezone(timezone.utc)
        clock.set_now(when)
        return when

    yield _set
    # pytest-asyncio tears down in another context, so a token reset would fail.
    clock.set_now(None)


class FakeMatcher:
    """Stands in for the AI product matcher in unit tests (the real one is checked by eval/order_lang_eval.py). A test
    can set .script: {item name: {"exact": [listing names, best first], "closest": name}}; otherwise a listing is exact
    when it contains every word of the item name (tests are in English; production never matches words)."""

    def __init__(self):
        self.script: dict = {}
        self.calls: list = []

    async def choose(self, items, listings, *, medicine=False, family=None):
        self.calls.append(([i.get("name") for i in items], [[x.get("name") for x in ls] for ls in listings]))
        self.family = family
        out = []
        for it, ls in zip(items, listings):
            want = self.script.get(it.get("name"))
            names = [str(x.get("name") or "") for x in ls]
            if want is not None:
                exact = [names.index(n) for n in want.get("exact", []) if n in names]
                closest = names.index(want["closest"]) if want.get("closest") in names else None
            else:
                words = [w for w in str(it.get("name") or "").lower().split() if len(w) > 2]
                exact = [j for j, x in enumerate(ls) if x.get("available") is not False
                         and all(w in f"{x.get('name')} {x.get('pack') or ''}".lower() for w in words)]
                closest = None
            out.append({"exact": exact, "closest": closest if not exact else None, "why": "test"})
        return out


@pytest.fixture(autouse=True)
def fake_matcher(monkeypatch):
    from app.tasks import matcher

    fake = FakeMatcher()
    monkeypatch.setattr(matcher, "choose", fake.choose)
    return fake
