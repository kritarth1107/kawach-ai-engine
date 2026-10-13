"""Plan B1.3 on every push (a shorter run; the nightly run takes the whole nudge slice for a week): world families, every
job that can message someone, an eager model that tries to message everyone it sees. Code must still hold the budget:
nobody in their night, nothing while paused, no alert to the person it is about, no near-duplicate bursts, a daily cap."""

import os
import uuid

import asyncpg
import pytest

TEST_DB = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_test")


@pytest.fixture
async def budget_db():
    base, _, _ = TEST_DB.rpartition("/")
    name = f"kawach_budget_{uuid.uuid4().hex[:8]}"
    admin = base.replace("+asyncpg", "") + "/postgres"
    conn = await asyncpg.connect(admin)
    await conn.execute(f'CREATE DATABASE "{name}"')
    await conn.close()
    try:
        yield f"{base}/{name}"
    finally:
        conn = await asyncpg.connect(admin)
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await conn.close()


async def test_a_simulated_week_holds_every_message_rule(budget_db, monkeypatch, tmp_path):
    monkeypatch.setenv("BUDGET_OUT", str(tmp_path))
    from eval.world import budget

    monkeypatch.setattr(budget, "OUT", tmp_path)
    rep = await budget.run(n_families=int(os.getenv("BUDGET_FAMILIES", "30")), days=int(os.getenv("BUDGET_DAYS", "2")), tick_min=10, url=budget_db)
    assert rep["brain_attempts"] > 100, "the eager model never ran: the sim is not exercising the send rules"
    assert not rep["failures"], f"{len(rep['failures'])} message rules broken:\n" + "\n".join(rep["failures"][:25])
    assert rep["worst_day"] <= budget.DAILY_CAP
