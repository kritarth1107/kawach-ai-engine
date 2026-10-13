"""Is the full-memory judge right? (trust plan C1.4)

Monthly: on the first Sunday of each month the Sunday learning job (app.learn.review) puts 30 of the judge's recent
verdicts on real replies in front of the Kavach team (admin "Daily score" page): for each, the reply, what they had
said, and the judge's verdict; a person answers yes (the reply is good) or no. Agreement = the share where the person
and the judge agree. It is saved as the "calibration" learning report; below 85% the judge's grades are marked not
trusted, and the learning loop does not promote a playbook on them (app.learn.lessons).

Before go-live, the same prompt is run on a fixed set of 30 hand-labelled invented replies (eval/judge_calibration.py,
eval/fixtures/judge_calibration.jsonl) and must agree on at least 85%.
"""

from __future__ import annotations

import random
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core import clock
from app.learn.models import ReplyLog
from app.quality.models import JudgeLabel, ReplyVerdict

BATCH_N = 30
AGREEMENT_BAR = 0.85
MIN_LABELS = 10


def month() -> str:
    return clock.ist().strftime("%Y-%m")


def first_sunday() -> bool:
    d = clock.ist()
    return d.weekday() == 6 and d.day <= 7


async def make_batch(sessions: async_sessionmaker, *, n: int = BATCH_N, batch: str | None = None) -> dict:
    """30 recent verdicts on real families' replies (failed ones up to half), once per month."""
    batch = batch or month()
    async with sessions() as s:
        have = (await s.execute(select(func.count()).select_from(JudgeLabel).where(JudgeLabel.batch == batch))).scalar_one()
        if have:
            return {"batch": batch, "created": 0, "size": int(have)}
        rows = list((await s.execute(
            select(ReplyVerdict).where(ReplyVerdict.judged_at >= clock.now() - timedelta(days=31), ~ReplyVerdict.family_id.startswith("shadow:"),
                                       ~ReplyVerdict.family_id.startswith("sim-")).order_by(ReplyVerdict.id.desc()).limit(2000)
        )).scalars())
        rng = random.Random(batch)
        failed = [r for r in rows if not r.passed]
        passed = [r for r in rows if r.passed]
        pick = rng.sample(failed, min(len(failed), n // 2))
        pick += rng.sample(passed, min(len(passed), n - len(pick)))
        if len(pick) < n:
            rest = [r for r in failed if r not in pick]
            pick += rng.sample(rest, min(len(rest), n - len(pick)))
        rng.shuffle(pick)
        for v in pick:
            await s.execute(insert(JudgeLabel).values(batch=batch, verdict_id=v.id, judge_pass=v.passed, created_at=clock.now())
                            .on_conflict_do_nothing(index_elements=[JudgeLabel.batch, JudgeLabel.verdict_id]))
        await s.commit()
    return {"batch": batch, "created": len(pick), "size": len(pick)}


async def label(sessions: async_sessionmaker, label_id: int, *, good: bool, by: str, note: str = "") -> dict | None:
    async with sessions() as s:
        row = await s.get(JudgeLabel, label_id)
        if row is None:
            return None
        row.label, row.labeled_by, row.labeled_at, row.note = bool(good), by[:120], clock.now(), (note or "")[:500]
        await s.commit()
        batch = row.batch
    return await agreement(sessions, batch)


async def agreement(sessions: async_sessionmaker, batch: str | None = None) -> dict:
    async with sessions() as s:
        if batch is None:
            batch = (await s.execute(select(func.max(JudgeLabel.batch)))).scalar_one_or_none()
        if not batch:
            return {"batch": None, "size": 0, "labelled": 0, "agreement": None, "trusted": None}
        rows = list((await s.execute(select(JudgeLabel).where(JudgeLabel.batch == batch))).scalars())
    done = [r for r in rows if r.label is not None]
    agree = sum(1 for r in done if r.label == r.judge_pass) / len(done) if done else None
    return {"batch": batch, "size": len(rows), "labelled": len(done), "agreement": round(agree, 3) if agree is not None else None,
            "trusted": (agree >= AGREEMENT_BAR) if agree is not None and len(done) >= MIN_LABELS else None, "bar": AGREEMENT_BAR,
            "judge_fail_rate": round(sum(1 for r in done if not r.judge_pass) / len(done), 3) if done else None,
            "people_fail_rate": round(sum(1 for r in done if not r.label) / len(done), 3) if done else None}


async def calibrate(sessions: async_sessionmaker) -> dict:
    """The Sunday job: a new batch on the first Sunday of the month; the agreement on the latest labelled batch."""
    from app.learn.review import _save

    made = await make_batch(sessions) if first_sunday() else None
    a = await agreement(sessions)
    if a["labelled"] < MIN_LABELS:
        return {"skipped": f"{a['labelled']} of {a['size']} labelled in batch {a['batch']}", "batch": made}
    return await _save(sessions, "calibration", {"n": a["labelled"], "agreement": a["agreement"], "trusted": a["trusted"], "batch": a["batch"],
                                                 "judge_fail_rate": a["judge_fail_rate"], "people_fail_rate": a["people_fail_rate"], "bar": AGREEMENT_BAR})


async def view(session: AsyncSession, batch: str | None = None) -> dict:
    """The batch to label: each reply, what they said, and the judge's verdict."""
    if batch is None:
        batch = (await session.execute(select(func.max(JudgeLabel.batch)))).scalar_one_or_none()
    if not batch:
        return {"batch": None, "items": []}
    rows = list((await session.execute(
        select(JudgeLabel, ReplyVerdict, ReplyLog).join(ReplyVerdict, ReplyVerdict.id == JudgeLabel.verdict_id)
        .join(ReplyLog, ReplyLog.id == ReplyVerdict.reply_id).where(JudgeLabel.batch == batch).order_by(JudgeLabel.id)
    )).all())
    items = []
    for lab, v, r in rows:
        items.append({"id": lab.id, "family": v.family_id[:8], "at": v.at.isoformat(), "kind": v.kind, "said": (r.user_text or "")[:600],
                      "reply": (r.text or "")[:1200], "judgePass": v.passed, "issues": v.issues or [],
                      "summary": (v.verdict or {}).get("summary") or "", "label": lab.label, "labeledBy": lab.labeled_by})
    return {"batch": batch, "items": items}
