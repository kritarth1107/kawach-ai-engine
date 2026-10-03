"""Run the agent bench and write a scorecard (markdown). Free: no model calls.

    TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_test \
    .venv/bin/python eval/agent_bench.py [--out path.md]

Exit code 1 when the release gate fails. The same scenarios run in CI (tests/test_agent_bench.py).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core import clock
from app.specialists.bench import SCENARIOS, gate, run_scenario

DB = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_test")


async def main(out: str | None) -> int:
    from app.care import models  # noqa: F401
    from app.db.session import Base
    from app.models import entities  # noqa: F401
    from app.specialists import channels  # noqa: F401
    from app.tasks import models as task_models  # noqa: F401

    engine = create_async_engine(DB)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
    clock.set_now(datetime(2026, 10, 2, 4, 30, tzinfo=timezone.utc))
    results = []
    for sc in SCENARIOS:
        clock.set_now(datetime(2026, 10, 2, 4, 30, tzinfo=timezone.utc))  # scenarios may move the clock
        conn = await engine.connect()
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint")
        try:
            results.append(await run_scenario(session, sc))
        except Exception as exc:  # noqa: BLE001
            results.append({"name": sc.name, "about": sc.about, "safety": sc.safety, "passed": False, "problems": [f"crashed: {exc!r}"], "status": "crash"})
        finally:
            await session.close()
            await trans.rollback()
            await conn.close()
    await engine.dispose()

    ok, why = gate(results)
    lines = [
        f"# Agent bench scorecard — {datetime.now(clock.IST):%Y-%m-%d %H:%M} IST", "",
        f"**Release gate: {'PASS' if ok else 'FAIL'}** — {why}.", "",
        f"{sum(r['passed'] for r in results)}/{len(results)} scenarios passed; "
        f"{sum(r['passed'] for r in results if r['safety'])}/{sum(r['safety'] for r in results)} safety scenarios passed.", "",
        "| Scenario | Safety | Result | End status | Channel | Browser runs | Place attempts | Est. cost ₹ | Problems |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| `{r['name']}` — {r['about']} | {'✓' if r['safety'] else '—'} | {'pass' if r['passed'] else '**FAIL**'} | {r.get('status')} | "
            f"{r.get('channel') or '—'} | {r.get('browser_runs', '—')} | {r.get('place_attempts', '—')} | {r.get('cost_inr', 0)} | "
            f"{'; '.join(r['problems']) or '—'} |"
        )
    report = "\n".join(lines) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(report)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    sys.exit(asyncio.run(main(ap.parse_args().out)))
