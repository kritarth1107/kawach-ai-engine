"""Seed the learning corpus from recorded simulation runs (free: no model calls).

    DATABASE_URL=<training db> .venv/bin/python eval/bootstrap_corpus.py [run_dir …]

Each judged Saheli message in the transcripts becomes an anonymised, scored example: the judge's verdict, whether
the person answered and how (the next message from them), and whether they corrected her. This gives the first
weekly lessons something to learn from before real families exist. Simulated people only; never real data.
"""

from __future__ import annotations

import asyncio
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import clock  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402

DEFAULT_RUNS = ["/home/m4dm4x/OpenBot/Shared/kavach-sim/round3"]


def _at(stamp: str) -> datetime:
    return datetime.strptime(f"2026 {stamp}", "%Y %d %b %H:%M").replace(tzinfo=clock.IST)


async def main(argv: list[str]) -> int:
    from eval.month_families import FAMILIES
    from app.brain import guards
    from app.learn import models as lm, scoring, situations
    from app.learn.anonymise import anonymise, leaks
    from app.learn.models import LearningExample

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    names = {f["key"]: [p["name"] for p in f["people"]] for f in FAMILIES}
    meds = {f["key"]: [m[0] if isinstance(m, (list, tuple)) else str(m) for ms in (f.get("meds") or {}).values() for m in (ms if isinstance(ms, list) else [ms])]
            for f in FAMILIES}
    roles = {p["id"]: ("elder" if p["recipient"] else ("self" if p["relation"] == "self" else "caregiver")) for f in FAMILIES for p in f["people"]}
    added = skipped = 0
    rid = 9_000_000
    async with SessionLocal() as s:
        for run in argv or DEFAULT_RUNS:
            for path in sorted(glob.glob(f"{run}/*.jsonl")):
                fam = Path(path).stem
                writes: dict[str, list[str]] = {}
                for line in open(path):
                    d = json.loads(line)
                    verdicts = {v["n"]: v for v in d["verdicts"]}
                    msgs = d["messages"]
                    for i, m in enumerate(msgs):
                        if m["who"] != "saheli":
                            writes.setdefault(m["who"], []).append(m["text"])
                            continue
                        v = verdicts.get(m["n"])
                        if not v or not m["text"].strip():
                            continue
                        to = m["to"]
                        prev = next((x for x in reversed(msgs[:i]) if x["who"] == to and x["kind"] == "human"), None) if m["kind"] == "reply" else None
                        nxt = next((x for x in msgs[i + 1:] if x["who"] == to and x["kind"] == "human"), None)
                        delay = int((_at(nxt["at"]) - _at(m["at"])).total_seconds()) if nxt else None
                        replied = nxt is not None and delay is not None and delay <= 6 * 3600
                        row = lm.ReplyLog(kind="proactive" if m["kind"] == "proactive" else "reply", situation="", replied=replied,
                                          reply_delay_s=delay if replied else None, tone=scoring.tone_of(nxt["text"]) if replied else None,
                                          corrected=bool(scoring.CORRECTED.search(nxt["text"])) if replied else None, dose_followed=None,
                                          judge_pass=bool(v.get("pass")))
                        role = roles.get(to, "caregiver")
                        sit = situations.tag(text=prev["text"] if prev else "", role="system" if m["kind"] == "proactive" else ("elder" if role == "elder" else "caregiver"),
                                             tools=[t["tool"] for t in m.get("tools") or [] if t.get("ok")], prompt=m.get("trigger") or "")
                        row.situation = sit
                        score = scoring.score_of(row)
                        ctx = anonymise(prev["text"] if prev else "", names=names.get(fam, []), medicines=meds.get(fam, []))
                        reply = anonymise(m["text"], names=names.get(fam, []), medicines=meds.get(fam, []))
                        if leaks(ctx, names=names.get(fam, [])) or leaks(reply, names=names.get(fam, [])):
                            skipped += 1
                            continue
                        rid += 1
                        s.add(LearningExample(reply_id=rid, situation=sit, kind=row.kind, speaker_role=role,
                                              lang=situations.lang_of(guards.profile(writes.get(to, [])[-8:])), context=ctx[:1000], reply=reply[:2000],
                                              score=score, at=clock.now(),
                                              signals={"replied": replied, "delay_s": row.reply_delay_s, "tone": row.tone, "corrected": row.corrected,
                                                       "judge_pass": row.judge_pass, "judge_issues": v.get("issues"), "source": f"sim:{Path(run).name}:{fam}"}))
                        added += 1
        await s.commit()
    print(json.dumps({"added": added, "skipped_leak": skipped}))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
