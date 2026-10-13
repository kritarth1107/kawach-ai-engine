"""Watching the learning itself.

- calibration (Sunday): people's yes/no on 30 of the full-memory judge's verdicts each month (app.quality.calibration);
  below 85% agreement the judge's grades are not trusted for promotions until fixed.
- drift (nightly): last 7 days vs the 21 before, per situation: average score, reply rate, annoyed rate, judge
  failures. A clear drop raises an alarm on the learning page (and in the logs, for a Cloud Monitoring alert).
- rule proposals (weekly): from the week's judge failures, the strong model may propose a change to Saheli's rules.
  Proposals are only shown to the founder; approved ones enter the next playbook and go through its safety trial.
- gap specs (weekly): the top "can't do yet" requests become short feature drafts (what, why, examples, likely tool).
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from statistics import mean

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core import clock
from app.learn import gaps
from app.quality.judge import parse_json
from app.learn.models import LearnReport, ReplyLog, RuleProposal

logger = logging.getLogger(__name__)
DRIFT_MIN_N = 40


async def _save(sessions, kind: str, data: dict) -> dict:
    async with sessions() as s:
        s.add(LearnReport(kind=kind, at=clock.now(), data=data))
        await s.commit()
    return data


async def latest(sessions, kind: str) -> dict | None:
    async with sessions() as s:
        r = (await s.execute(select(LearnReport).where(LearnReport.kind == kind).order_by(LearnReport.at.desc()).limit(1))).scalars().first()
        return {**r.data, "at": r.at.isoformat()} if r else None


# ── judge calibration ──────────────────────────────────────────────────────────


async def calibrate(sessions: async_sessionmaker) -> dict:
    """People's yes/no on the full-memory judge's verdicts (trust plan C1.4; app.quality.calibration)."""
    from app.quality import calibration

    return await calibration.calibrate(sessions)


# ── drift ──────────────────────────────────────────────────────────────────────


def _rates(rows: list) -> dict:
    graded = [r for r in rows if r.judge_pass is not None]
    asked = [r for r in rows if r.replied is not None and r.kind != "reply"]
    return {"n": len(rows), "score": round(mean(r.score for r in rows), 3),
            "annoyed": round(sum(1 for r in rows if r.tone == "annoyed") / len(rows), 3),
            "reply_rate": round(sum(1 for r in asked if r.replied) / len(asked), 3) if asked else None,
            "judge_fail": round(sum(1 for r in graded if r.judge_pass is False) / len(graded), 3) if graded else None}


async def drift(sessions: async_sessionmaker) -> dict:
    now = clock.now()
    async with sessions() as s:
        rows = list((await s.execute(select(ReplyLog).where(ReplyLog.at >= now - timedelta(days=28), ReplyLog.score.is_not(None)))).scalars())
    recent = [r for r in rows if r.at >= now - timedelta(days=7)]
    before = [r for r in rows if r.at < now - timedelta(days=7)]
    alarms, by = [], {}
    groups = {"all": (recent, before)}
    for sit in {r.situation for r in rows}:
        groups[sit] = ([r for r in recent if r.situation == sit], [r for r in before if r.situation == sit])
    for name, (a, b) in groups.items():
        if len(a) < DRIFT_MIN_N or len(b) < DRIFT_MIN_N:
            continue
        ra, rb = _rates(a), _rates(b)
        by[name] = {"recent": ra, "before": rb}
        if ra["score"] < rb["score"] - 0.1:
            alarms.append(f"{name}: average score fell {rb['score']} → {ra['score']}")
        if ra["annoyed"] > rb["annoyed"] + 0.05:
            alarms.append(f"{name}: more annoyed replies {rb['annoyed']:.0%} → {ra['annoyed']:.0%}")
        if ra["judge_fail"] is not None and rb["judge_fail"] is not None and ra["judge_fail"] > rb["judge_fail"] + 0.05:
            alarms.append(f"{name}: judge failures up {rb['judge_fail']:.0%} → {ra['judge_fail']:.0%}")
        if ra["reply_rate"] is not None and rb["reply_rate"] is not None and ra["reply_rate"] < rb["reply_rate"] - 0.1:
            alarms.append(f"{name}: fewer people answer her messages {rb['reply_rate']:.0%} → {ra['reply_rate']:.0%}")
    for a in alarms:
        logger.error("LEARN ALERT drift %s", a)
    return await _save(sessions, "drift", {"alarms": alarms, "by": by})


# ── rule proposals ─────────────────────────────────────────────────────────────

PROPOSE = """You review a WhatsApp care companion (Saheli) for elderly people in India. Below are replies a quality
judge FAILED this week, with the judge's note. If a clear, repeated problem would be fixed by a short new rule in her
instructions, propose it. At most 3 proposals; none if the failures are one-offs. Each rule: one or two sentences,
general (no names), and never weaker on safety (alerts, doses, privacy). Return only JSON:
{"proposals": [{"situation": "...", "rule": "...", "why": "...", "examples": [n, n]}]}"""


async def propose_rules(sessions: async_sessionmaker) -> list[dict]:
    from app.learn.situations import SITUATIONS
    from app.llm import router

    async with sessions() as s:
        fails = list((await s.execute(
            select(ReplyLog).where(ReplyLog.judge_pass.is_(False), ReplyLog.at >= clock.now() - timedelta(days=7)).order_by(ReplyLog.id.desc()).limit(60)
        )).scalars())
    from app.learn import corrections
    from app.learn.anonymise import anonymise

    async with sessions() as s:
        fixed = corrections.for_proposals(await corrections.cases(s, days=7, limit=40))
    lines = [f"[{r.situation}] They: {anonymise(r.user_text)[:200]} | Saheli: {anonymise(r.text)[:300]} | judge: {r.judge_note}" for r in fails]
    lines += fixed  # people's own corrections count most when there are few families
    if len(lines) < 3:
        return []
    items = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    fails = lines
    try:
        out = await router.complete("learn", system_stable=PROPOSE, messages=[{"role": "user", "content": [{"type": "text", "text": items}]}],
                                    max_tokens=1500, effort="low", essential=False)
    except router.AllModelsFailed:
        return []
    got = []
    async with sessions() as s:
        for p in (parse_json(out.text).get("proposals") or [])[:3]:
            rule = str(p.get("rule") or "").strip()
            if not rule or len(rule) > 400:
                continue
            ev = [items.splitlines()[i - 1][:300] for i in p.get("examples") or [] if isinstance(i, int) and 1 <= i <= len(fails)]
            s.add(RuleProposal(created_at=clock.now(), situation=p.get("situation") if p.get("situation") in SITUATIONS else "general",
                               rule=rule, why=str(p.get("why") or "")[:500], evidence=ev, status="proposed"))
            got.append(p)
        await s.commit()
    return got


async def decide_rule(sessions: async_sessionmaker, rule_id: int, *, approve: bool, by: str) -> dict:
    async with sessions() as s:
        r = await s.get(RuleProposal, rule_id)
        if not r or r.status != "proposed":
            return {"ok": False}
        r.status, r.decided_by = ("approved" if approve else "rejected"), by
        await s.commit()
    return {"ok": True}


AUTO_RULES_PER_WEEK = 3  # at most 3 new lessons a week go into a trial, so every change stays attributable (§10)


async def auto_approve(session) -> dict:
    """Founder decision Q10 (13 Oct): self-learning goes live automatically behind safety gates. A proposed rule that
    passes the safety gate (no medical content, alerts, privacy or fixed rules: lessons.lesson_problems) is approved for
    the next trial (the replay gate, then the canary, decide whether it stays); one that fails is dropped and listed for
    the founder to read."""
    from app.learn import lessons

    week = clock.now() - timedelta(days=7)
    used = (await session.execute(select(func.count()).select_from(RuleProposal).where(
        RuleProposal.status.in_(("approved", "in_playbook")), RuleProposal.decided_by == "auto", RuleProposal.created_at >= week))).scalar_one()
    out = {"approved": 0, "dropped": 0}
    for r in (await session.execute(select(RuleProposal).where(RuleProposal.status == "proposed").order_by(RuleProposal.id))).scalars():
        # a proposed rule may be a little longer than a style lesson; everything else is the lessons' safety gate
        problems = [p for p in lessons.lesson_problems(r.rule) if p != "too long"] + (["too long"] if len(r.rule) > 400 else [])
        if problems:
            r.status, r.decided_by = "dropped", "auto"
            r.why = (r.why + f" | dropped by the safety gate: {', '.join(problems)}")[:500]
            out["dropped"] += 1
        elif used < AUTO_RULES_PER_WEEK:
            r.status, r.decided_by = "approved", "auto"
            used += 1
            out["approved"] += 1
    return out


async def approved_rules(session) -> dict[str, list[str]]:
    """Approved rules not yet in a playbook, by situation (the next draft takes them in)."""
    rows = (await session.execute(select(RuleProposal).where(RuleProposal.status == "approved"))).scalars()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r.situation, []).append(r.rule)
    return out


async def mark_in_playbook(session) -> None:
    for r in (await session.execute(select(RuleProposal).where(RuleProposal.status == "approved"))).scalars():
        r.status = "in_playbook"


# ── gap specs ──────────────────────────────────────────────────────────────────

SPEC = """For each requested capability below (things people asked a care companion on WhatsApp that it cannot do yet),
write a short feature draft: what it is, why families want it (from the examples), a likely tool or integration, risks,
and a size (S/M/L). Return only JSON: {"specs": [{"category": "...", "what": "...", "why": "...", "tool": "...", "risks": "...", "size": "S"}]}"""


async def gap_specs(sessions: async_sessionmaker) -> dict:
    from app.llm import router

    async with sessions() as s:
        top = [g for g in await gaps.ranking(s) if g["category"] != "other"][:5]
    if not top:
        return {"specs": []}
    try:
        out = await router.complete("learn", system_stable=SPEC, messages=[{"role": "user", "content": [{"type": "text", "text": json.dumps(top, ensure_ascii=False)}]}],
                                    max_tokens=2000, effort="low", essential=False)
    except router.AllModelsFailed:
        return {"specs": []}
    data = {"specs": (parse_json(out.text).get("specs") or [])[:5], "from": top}
    return await _save(sessions, "gap_specs", data)
