"""Regular nudges become tuning examples (self-learning, 2026-10-13): every nudge that was really sent, passed the hourly
review (no made-up fact, no advice, personal, their language) and that the person answered without being bothered, as
one supervised example: the context the writer saw → the JSON it wrote. Format: Vertex AI Gemini supervised tuning JSONL.

The output holds real people's care data: write it only to a private local path (never git, never a shared folder), and
it goes through the lab's anonymising and the founder's yes before any upload (see the Tuning Engineer's guide).

    DATABASE_URL=... PYTHONPATH=. python -m eval.export_nudge_tuning --out /tmp/nudge_tuning.jsonl [--days 30] [--min-reaction neutral]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.care.models import NudgeLedger
from app.care.nudges import COMPOSE
from app.core import clock

GOOD = {"welcome": ("welcome",), "neutral": ("welcome", "neutral")}


def option_id(context: str, kind: str, field: str | None) -> str | None:
    """The option the writer picked, found again in the OPTIONS block of its context (kind, and field when it had one)."""
    lines = [m for m in re.finditer(r"^\s*-\s*(o\d+)\s*\[([a-z_]+)\](.*)$", context or "", re.M)]
    for m in lines:
        if m.group(2) == kind and (not field or f"(field: {field})" in m.group(3)):
            return m.group(1)
    return next((m.group(1) for m in lines if m.group(2) == kind), None)


def example(row) -> dict | None:
    """One Vertex tuning example from a ledger row, or None when the row cannot teach (no context or no option found)."""
    oid = option_id(row.context or "", row.kind, row.field)
    if not row.context or not row.text or oid is None:
        return None
    answer = {"option_id": oid, "text": row.text, "field": row.field, "expects_reply": row.kind != "early_bird"}
    return {"systemInstruction": {"role": "system", "parts": [{"text": COMPOSE}]},
            "contents": [{"role": "user", "parts": [{"text": row.context}]},
                         {"role": "model", "parts": [{"text": json.dumps(answer, ensure_ascii=False)}]}]}


async def rows_to_learn_from(sessions, days: int, min_reaction: str) -> list:
    since = clock.now() - timedelta(days=days)
    async with sessions() as s:
        got = list((await s.execute(
            select(NudgeLedger).where(NudgeLedger.status == "sent", NudgeLedger.sent_at >= since, NudgeLedger.replied_at.is_not(None),
                                      NudgeLedger.reaction.in_(GOOD[min_reaction]), NudgeLedger.context.is_not(None))
            .order_by(NudgeLedger.sent_at)
        )).scalars())
    return [r for r in got if (r.grade or {}).get("pass") is True]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-reaction", choices=sorted(GOOD), default="neutral")
    a = ap.parse_args()
    engine = create_async_engine(os.environ["DATABASE_URL"])
    rows = await rows_to_learn_from(async_sessionmaker(engine, expire_on_commit=False), a.days, a.min_reaction)
    examples = [e for e in (example(r) for r in rows) if e]
    old = os.umask(0o077)
    try:
        with open(a.out, "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    finally:
        os.umask(old)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    print(f"{len(examples)} examples from {len(rows)} good nudges ({a.days} days) → {a.out}; by kind {kinds}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
