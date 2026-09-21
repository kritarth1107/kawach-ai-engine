"""Instinct memory eval rubric — smoke checks for CI."""

from __future__ import annotations

import asyncio
import sys
import uuid

from app.db.migrate import run_instinct_migrations
from app.db.session import Base, SessionLocal, engine
from app.models.entities import Elder, Family
from app.rag.memory_extract import persist_inbox_memory
from app.rag.memory_lifecycle import forget_memory_fact
from app.rag.retrieve import load_instinct_context, memory_grep


async def run_rubric() -> int:
    failures: list[str] = []
    async with engine.begin() as conn:
        await run_instinct_migrations(conn)
        await conn.run_sync(Base.metadata.create_all)

    family_id = uuid.uuid4()
    elder_id = uuid.uuid4()

    async with SessionLocal() as session:
        session.add(Family(id=family_id, name="Eval Family"))
        session.add(
            Elder(
                id=elder_id,
                family_id=family_id,
                display_name="Eval Elder",
                slug="eval-elder",
            )
        )
        await session.flush()
        fact = await persist_inbox_memory(
            session,
            family_id=family_id,
            elder_id=elder_id,
            content="Amma takes osimertinib 80mg every morning with breakfast.",
            category="health",
            topic="osimertinib",
            source_role="family",
        )
        await session.commit()

        hits = await memory_grep(
            session,
            family_id=family_id,
            elder_id=elder_id,
            query="osimertinib morning tablet",
            limit=3,
        )
        if not hits and fact.content:
            failures.append("grep should find inbox or entity hit for medication query")

        ctx = await load_instinct_context(
            session,
            family_id=family_id,
            elder_id=elder_id,
            query="osimertinib",
        )
        if "Nothing saved" in ctx and fact.content not in ctx:
            failures.append("load_instinct_context should include inbox/profile text")

        await forget_memory_fact(session, fact_id=fact.id, forgotten_by="eval")
        await session.commit()
        post_hits = await memory_grep(
            session,
            family_id=family_id,
            elder_id=elder_id,
            query="osimertinib",
            limit=3,
        )
        if post_hits:
            failures.append("forgotten fact should not appear in grep hits")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    print("memory_instinct_rubric: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_rubric()))
