"""One learning cycle on a training database (the simulator's), for pre-launch training.

    DATABASE_URL=… .venv/bin/python eval/learn_cycle.py            # score, corpus, draft + gate (cheap model, capped)
    DATABASE_URL=… .venv/bin/python eval/learn_cycle.py --no-model # score and corpus only (₹0)

Simulated families count as consented. Prints the draft playbook and its gate report; compare it with
`eval/train.sh regress` run twice: LEARN_FORCE_VERSION=0 (no playbook) and LEARN_FORCE_VERSION=<draft>.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db.session import Base, SessionLocal, engine  # noqa: E402


async def main(argv: list[str]) -> int:
    from app.care import models, outcomes  # noqa: F401
    from app.care.models import FamilyRoster
    from app.learn import lessons, models as lm, scoring  # noqa: F401
    from app.llm import spend

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    spend.configure(SessionLocal)
    async with SessionLocal() as s:
        for r in (await s.execute(select(FamilyRoster).where(FamilyRoster.family_id.startswith(("month-", "sim-"))))).scalars():
            if (r.elder or {}).get("id") and not (await outcomes.consent(s, r.family_id, r.elder["id"]))["granted"]:
                await outcomes.set_consent(s, r.family_id, r.elder["id"], granted=True, by="training")
        await s.commit()
    out = {"scored": await scoring.score_pending(SessionLocal, limit=100000), "corpus": await scoring.build_corpus(SessionLocal, days=365, limit=100000)}
    if "--no-model" not in argv:
        out["weekly"] = await lessons.weekly(SessionLocal)
        if out["weekly"].get("draft"):
            async with SessionLocal() as s:
                pb = await s.get(lm.PlaybookVersion, out["weekly"]["draft"])
                out["lessons"] = pb.lessons
    await spend.flush()
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
