"""Daily: an AI judge grades a small sample of yesterday's real replies (capped by the spend guard).

A cheap Flash judge, ~3% of replies (at least 20, at most LEARN_GRADE_MAX), in batches. Failures lower the reply's
score and are kept with the judge's note; eval/export_failures.py turns them into permanent test cases.
Runs as a non-essential model role: it stops by itself at the daily soft cap.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core import clock
from app.learn.models import ReplyLog
from app.learn.scoring import score_of

logger = logging.getLogger(__name__)

PROMPT = """You check a care companion's WhatsApp replies to elderly people and their families in India.
For each numbered item you get what the person said (may be empty for messages Saheli started) and Saheli's message.
Fail a message only for a real problem: wrong or made-up facts, unsafe health advice, ignoring what was asked, wrong
language or script for the person, cold or preachy tone, needlessly long for an elder, repeating itself, or a promise
it cannot keep. Otherwise pass. Return only JSON: {"verdicts": [{"n": 1, "pass": true, "note": "short reason if failed"}]}"""
BATCH = 20


def share() -> float:
    return float(os.getenv("LEARN_GRADE_SHARE", "0.03"))


def max_per_day() -> int:
    return int(os.getenv("LEARN_GRADE_MAX", "200"))


def parse(text: str) -> dict:
    t = (text or "").strip()
    if "{" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}


async def grade_sample(sessions: async_sessionmaker, *, day_offset: int = 1) -> dict:
    from app.llm import router

    start = clock.ist().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=day_offset)
    async with sessions() as session:
        rows = list((await session.execute(
            select(ReplyLog).where(ReplyLog.at >= start, ReplyLog.at < start + timedelta(days=1), ReplyLog.judge_pass.is_(None))
        )).scalars())
        if not rows:
            return {"graded": 0, "failed": 0}
        rng = random.Random(start.isoformat())
        n = min(max_per_day(), max(min(20, len(rows)), int(len(rows) * share())))
        pick = rng.sample(rows, n)
        graded = failed = 0
        for i in range(0, len(pick), BATCH):
            chunk = pick[i:i + BATCH]
            items = "\n\n".join(
                f"#{j + 1} ({r.situation}; to {r.speaker_role}; writes {r.lang or 'unknown'})\n  They: {r.user_text[:400] or '(Saheli started)'}\n  Saheli: {r.text[:800]}"
                for j, r in enumerate(chunk)
            )
            try:
                out = await router.complete("judge_fast", system_stable=PROMPT, messages=[{"role": "user", "content": [{"type": "text", "text": items}]}],
                                            max_tokens=3000, effort="low", essential=False)
            except router.AllModelsFailed as exc:
                logger.warning("grader stopped: %s", exc)
                break
            got = {v.get("n"): v for v in parse(out.text).get("verdicts") or []}
            for j, r in enumerate(chunk):
                v = got.get(j + 1)
                if v is None:
                    continue
                r.judge_pass = bool(v.get("pass", True))
                r.judge_note = str(v.get("note") or "")[:300] or None
                if r.scored_at is not None:
                    r.score = score_of(r)
                graded += 1
                failed += 0 if r.judge_pass else 1
        await session.commit()
    return {"graded": graded, "failed": failed, "from": len(rows)}
