"""Replay past failures through today's Saheli and grade the new replies (paid: brain + judge calls).

    SIM_PROJECT=… SIM_MAX_INR=150 .venv/bin/python eval/replay_failures.py [eval/fixtures/failures.jsonl] --yes

Each case is sent to a fresh simulated family (the brain_scenarios one, with its care record) as the same kind of
person; the grader then checks the new reply with the old failure's note in mind. Prints fixed / still failing.
Runs in the simulator database and the training project, never production.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RECHECK = """Earlier a care companion's reply to this message failed review for: "{note}".
Here is a new reply to the same message. Does the new reply still have that problem, or any other real problem
(wrong facts, unsafe advice, wrong language/script, cold, too long for an elder, ignores the question)?
Return only JSON {{"pass": true|false, "note": "..."}}"""


def guard(argv: list[str]) -> None:
    project = os.getenv("SIM_PROJECT") or ""
    if "--yes" not in argv or not os.getenv("SIM_MAX_INR"):
        sys.exit("Paid: set SIM_MAX_INR and pass --yes (the founder's OK).")
    if not project or project == "kavach-care":
        sys.exit("Set SIM_PROJECT to the training project (never kavach-care).")
    os.environ.setdefault("GCP_PROJECT_ID", project)
    os.environ.setdefault("DATABASE_URL", os.getenv("SIM_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_sim"))


async def main(argv: list[str]) -> int:
    guard(argv)
    from eval import brain_scenarios as bs
    from app.learn import grader
    from app.llm import router, spend
    from app.sim.world import SimHost

    path = next((a for a in argv if a.endswith(".jsonl")), "eval/fixtures/failures.jsonl")
    cases = [json.loads(line) for line in open(path)]
    from sqlalchemy import text

    from app.care import models  # noqa: F401
    from app.db.migrate import run_v2_migrations
    from app.db.session import Base, engine
    from app.learn import models as learn_models  # noqa: F401
    from app.models import entities  # noqa: F401
    from app.tasks import models as task_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await run_v2_migrations(conn)
    fixed, still = 0, []
    for c in cases:
        if spend.process_spent() >= float(os.environ["SIM_MAX_INR"]):
            print(f"budget ₹{os.environ['SIM_MAX_INR']} reached; stopping")
            break
        run = bs.Run(family_id=f"replay-{c['id']}-{uuid.uuid4().hex[:4]}", host=SimHost())
        await bs.setup(run)
        who = bs.SUNITA if c["speaker"] == "caregiver" else bs.ELDER
        reply = await bs.say(run, who, c["said"], "2026-10-02 09:30")
        try:
            out = await router.complete("judge", system_stable=RECHECK.format(note=c["note"][:300]),
                                        messages=[{"role": "user", "content": [{"type": "text", "text": f"They: {c['said']}\nNew reply: {reply}"}]}],
                                        max_tokens=400, effort="low", essential=False)
            ok = bool(grader.parse(out.text).get("pass", False))
        except router.AllModelsFailed as exc:
            print(f"  judge unavailable: {exc}")
            break
        if ok:
            fixed += 1
        else:
            still.append(c["id"])
        print(f"  {'✓' if ok else '✗'} {c['id']} [{c['situation']}]")
    print(f"\n{fixed}/{fixed + len(still)} fixed; still failing: {', '.join(still) or 'none'}")
    return 0 if not still else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
