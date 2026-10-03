"""Turn the grader's failed replies into anonymised replay cases (free, no model calls).

    .venv/bin/python eval/export_failures.py [--days 14] [--out eval/fixtures/failures.jsonl]

Each case: the situation, who spoke, what they said and Saheli's failed reply (both anonymised, consenting families
only), and the grader's note. eval/replay_failures.py sends the same messages to today's Saheli and checks that the
problem is gone, so a fixed failure stays fixed.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def collect(session, *, days: int = 14, limit: int = 300) -> list[dict]:
    from sqlalchemy import select

    from app.care import outcomes, store
    from app.core import clock
    from app.learn.anonymise import anonymise, leaks
    from app.learn.models import ReplyLog
    from app.learn.scoring import _identity_words

    rows = list((await session.execute(
        select(ReplyLog).where(ReplyLog.judge_pass.is_(False), ReplyLog.at >= clock.now() - timedelta(days=days),
                               ~ReplyLog.family_id.startswith("shadow:")).order_by(ReplyLog.id.desc()).limit(limit)
    )).scalars())
    consent: dict[str, bool] = {}
    seen, out = set(), []
    for r in rows:
        if r.family_id not in consent:
            roster = await store.roster(session, r.family_id)
            elder = (roster.elder or {}).get("id") if roster else None
            consent[r.family_id] = bool(elder and (await outcomes.consent(session, r.family_id, elder))["granted"])
        if not consent[r.family_id] or not r.user_text:
            continue
        names, meds = await _identity_words(session, r.family_id)
        said, reply = anonymise(r.user_text, names=names, medicines=meds), anonymise(r.text, names=names, medicines=meds)
        if leaks(said + " " + reply, names=names) or said in seen:
            continue
        seen.add(said)
        out.append({"id": f"f{r.id}", "situation": r.situation, "speaker": r.speaker_role or "elder", "lang": r.lang,
                    "said": said, "bad_reply": reply, "note": r.judge_note or ""})
    return out


async def main(argv: list[str]) -> int:
    from app.db.session import SessionLocal

    days = int(argv[argv.index("--days") + 1]) if "--days" in argv else 14
    out = Path(argv[argv.index("--out") + 1] if "--out" in argv else "eval/fixtures/failures.jsonl")
    async with SessionLocal() as s:
        cases = await collect(s, days=days)
    out.parent.mkdir(parents=True, exist_ok=True)
    known = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    with out.open("a") as f:
        for c in cases:
            if c["id"] not in known:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(cases)} failures, {len([c for c in cases if c['id'] not in known])} new → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
