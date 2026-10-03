"""Recall benchmark with real embeddings (needs Vertex). Free-ish: ~20 embedding calls.

    DATABASE_URL=<test db> GCP_PROJECT_ID=<project> .venv/bin/python eval/memory_bench.py

Seeds the BENCH memories from tests/test_memory_upgrades.py plus noise into a scratch family, indexes them with the real
embedder, asks the Hinglish/Hindi paraphrases, and reports how many land in the top 3 (bar: 8/8).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def main() -> int:
    from sqlalchemy import delete

    from app.care import memory_index, store
    from app.care.models import CareEvent
    from app.core import clock
    from app.db.session import Base, SessionLocal, engine
    from tests.test_memory_upgrades import BENCH

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    fam, who = "bench-memory", "bench-elder"
    async with SessionLocal() as s:
        await s.execute(delete(CareEvent).where(CareEvent.family_id == fam))
        await s.execute(delete(memory_index.MemoryVector).where(memory_index.MemoryVector.family_id == fam))
        for i, (kind, text, _) in enumerate(BENCH):
            clock.set_now(datetime(2026, 10, 1 + i, 10, tzinfo=clock.IST))
            await store.record_event(s, family_id=fam, subject_id=who, kind=kind, summary=text)
        for d in range(10, 28):
            clock.set_now(datetime(2026, 10, d, 13, tzinfo=clock.IST))
            await store.record_event(s, family_id=fam, subject_id=who, kind="meal", summary="dal chawal for lunch")
        clock.set_now(datetime(2026, 10, 28, 10, tzinfo=clock.IST))
        n = await memory_index.index_subject(s, fam, who)
        found = []
        for _, text, q in BENCH:
            hits = await memory_index.search(s, fam, [who], q, limit=3)
            ok = any(text[:12] in h.text for h in hits)
            found.append(ok)
            print(f"{'✓' if ok else '✗'} {q!r} -> {[h.text[:50] for h in hits]}")
        await s.rollback()
    print(f"indexed {n}; found {sum(found)}/{len(found)} (bar 8/8)")
    return 0 if all(found) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
