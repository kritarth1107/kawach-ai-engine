"""Run the daily score for one IST day from the command line (a Cloud Run Job execution in production, with the dream
job's image and settings): every family's judge, checks, memory score and replay cases, then check 1 and the totals.

    python -m app.quality.run --day 2026-10-13 [--family <id prefix>] [--email]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging


async def main_async(day: str, family: str, email: bool) -> dict:
    from sqlalchemy import select

    from app.care.models import FamilyRoster
    from app.db.session import SessionLocal
    from app.jobs.dream import _ensure_v2_schema
    from app.llm import spend
    from app.quality import nightly

    spend.configure(SessionLocal)
    await _ensure_v2_schema()
    async with SessionLocal() as s:
        rosters = [r for r in (await s.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:"),
                                                                         ~FamilyRoster.family_id.startswith("sim-")))).scalars()
                   if r.family_id.startswith(family)]
    out: dict = {"day": day, "families": {}}
    for r in rosters:
        out["families"][r.family_id[:8]] = await nightly.family(SessionLocal, r, day)
    out["finish"] = await nightly.finish(SessionLocal, day)
    if email:
        from app.brain.host import LiveHost
        from app.quality import report

        out["email"] = await report.send(SessionLocal, LiveHost(), day, force=True)
    await spend.flush()
    out["spent_inr"] = round(spend.process_spent(), 2)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--family", default="")
    ap.add_argument("--email", action="store_true")
    a = ap.parse_args()
    print(json.dumps(asyncio.run(main_async(a.day, a.family, a.email)), indent=1, default=str))


if __name__ == "__main__":
    main()
