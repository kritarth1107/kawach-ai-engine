"""The nightly Cloud Run Job (02:00 IST): Saheli's night for every family (app/care/dream.py).

The old v1 dream that consolidated the Instinct inbox into memory entities and rendered memory profiles is gone with the
old brain (trust plan A1; founder D9): what was useful in the inbox moved into the care record (app/care/memory_unify.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from app.db.session import SessionLocal, engine


async def _ensure_v2_schema() -> None:
    from app.care import baselines, models  # noqa: F401
    from app.learn import models as learn_models  # noqa: F401
    from app.llm import spend  # noqa: F401
    from app.specialists import channels  # noqa: F401
    from app.tasks import models as task_models  # noqa: F401

    from app.db.migrate import ensure_indexes, ensure_schema

    await ensure_schema(engine)
    await ensure_indexes(engine)


async def run_all() -> dict:
    from app.care import dream as v2_dream
    from app.llm import spend

    spend.configure(SessionLocal)
    await _ensure_v2_schema()
    return {"v2": await v2_dream.dream_all(SessionLocal, budget_s=3000)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Kavach nightly dream job (Saheli v2)")
    parser.add_argument("--all", action="store_true", help="Every family (the nightly job)")
    args = parser.parse_args()
    if not args.all:
        parser.print_help()
        return
    print(json.dumps(asyncio.run(run_all()), indent=2, default=str))


if __name__ == "__main__":
    main()
