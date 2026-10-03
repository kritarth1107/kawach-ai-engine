import os
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
    from app.db.session import Base
    from app.models import entities  # noqa: F401

    engine = create_async_engine(TEST_DB)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
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
