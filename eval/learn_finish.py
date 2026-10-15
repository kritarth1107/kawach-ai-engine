"""Finish a learning-and-memory month run that stopped early (budget): the final quiz and readout on the day each
family reached, then the scorecard. Uses the run's own database and its saved files.

    (same env as the run) PYTHONPATH=. python eval/learn_finish.py <run dir> [family ...]
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import zlib
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from app.core import clock
from app.db.session import SessionLocal
from eval import learn_month as lm
from eval import month_sim as ms
from eval.month_families import FAMILIES


def _msgs(lines: list[dict]) -> list[ms.Msg]:
    out = []
    for x in lines:
        for m in x["messages"]:
            at = datetime.strptime(f"{m['at']} 2026", "%d %b %H:%M %Y").replace(tzinfo=clock.IST)
            out.append(ms.Msg(at, m["who"], m["to"], m["text"], m["kind"], m.get("trigger", ""), tools=m.get("tools") or [],
                              alerts=m.get("alerts") or [], ms=m.get("ms", 0)))
    return out


async def load(outdir: Path, key: str) -> ms.Fam | None:
    saved = outdir / f"{key}.learn.json"
    log = outdir / f"{key}.jsonl"
    if not saved.exists() or not log.exists():
        return None
    spec = next(s for s in FAMILIES if s["key"] == key)
    spec = {**spec, "events": lm.safe_events(spec["events"])}
    lines = [json.loads(ln) for ln in log.open()]
    async with SessionLocal() as session:
        fid = (await session.execute(text("SELECT family_id FROM turns WHERE family_id LIKE :p GROUP BY family_id ORDER BY max(at) DESC LIMIT 1"),
                                     {"p": f"month-{key}-%"})).scalar()
    f = ms.Fam(spec=spec, family_id=fid, rng=random.Random(zlib.crc32(key.encode()) & 0xFFFF))
    last = max(x["day"] for x in lines)
    for subject, meds in spec["meds"].items():
        f.truth_meds[subject] = {ms.med_slug(n): [n, d, t] for n, d, t in meds}
    for e in spec["events"]:
        if e[0] <= last and e[4].get("med_change"):
            ms.apply_med_change(f, e[4]["med_change"])
    f.learn = lm.setup(f)
    st = json.loads(saved.read_text())
    L = f.learn
    L.quiz, L.reads, L.learned_on, L.nudges = st["quiz"], st["reads"], st["learned_on"], st["nudges"]
    L.stats.update(st["stats"])
    L.days = {int(k): v for k, v in st["days"].items()}
    L.today = L.days.get(last, {})
    f.msgs = _msgs(lines)
    f.verdicts = [v for x in lines for v in x["verdicts"]]
    f.checks = [c for x in lines for c in x["checks"]]
    f.audits = [x["audit"] for x in lines]
    f.spec["_last_day"] = last
    return f


async def main(argv: list[str]) -> int:
    dry = "--no-quiz" in argv  # a check of the loader and the readout, no model calls
    argv = [a for a in argv if a != "--no-quiz"]
    outdir = Path(argv[0])
    keys = argv[1:] or sorted(p.name.split(".")[0] for p in outdir.glob("*.learn.json"))
    from app.brain.tools import set_task_agent
    from app.sim.agent import FakeAgent

    set_task_agent(FakeAgent())
    fams = []
    for key in keys:
        f = await load(outdir, key)
        if not f:
            print(f"{key}: no saved run")
            continue
        last = f.spec["_last_day"]
        if dry:
            await lm.read(f, ms.day_dt(last, "23:55"), last)
        elif not any(q.get("final") for q in f.learn.quiz):
            await lm.quiz(f, ms.day_dt(last, "22:40"), last, final=True)
            await lm.read(f, ms.day_dt(last, "23:55"), last)
            lm.dump(f, outdir)
        print(f"{key}: final quiz on day {last}: {sum(q['score'] for q in f.learn.quiz if q.get('final'))}/{2 * sum(1 for q in f.learn.quiz if q.get('final'))}")
        fams.append(f)
    clock.set_now(None)
    rep = lm.report(fams)
    (outdir / "learn_report.md").write_text(rep)
    print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
