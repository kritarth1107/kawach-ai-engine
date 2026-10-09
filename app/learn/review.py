"""Watching the learning itself.

- calibration (weekly): a sample the cheap grader already graded is graded again by the strong judge; how often they
  agree. Below 80% the cheap grades are not trusted for promotions until fixed.
- drift (nightly): last 7 days vs the 21 before, per situation: average score, reply rate, annoyed rate, grader
  failures. A clear drop raises an alarm on the learning page (and in the logs, for a Cloud Monitoring alert).
- rule proposals (weekly): from the week's grader failures, the strong model may propose a change to Saheli's rules.
  Proposals are only shown to the founder; approved ones enter the next playbook and go through its safety trial.
- gap specs (weekly): the top "can't do yet" requests become short feature drafts (what, why, examples, likely tool).
"""

from __future__ import annotations

import json
import logging
import random
from datetime import timedelta
from statistics import mean

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core import clock
from app.learn import gaps, grader
from app.learn.models import LearnReport, ReplyLog, RuleProposal

logger = logging.getLogger(__name__)
CALIBRATION_N = 30
AGREEMENT_FLOOR = 0.8
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


# ── grader calibration ─────────────────────────────────────────────────────────


async def calibrate(sessions: async_sessionmaker) -> dict:
    from app.llm import router

    async with sessions() as s:
        rows = list((await s.execute(
            select(ReplyLog).where(ReplyLog.judge_pass.is_not(None), ReplyLog.at >= clock.now() - timedelta(days=14)).order_by(ReplyLog.id.desc()).limit(500)
        )).scalars())
    if len(rows) < 10:
        return {"skipped": "not enough graded replies"}
    pick = random.Random(clock.ist_day()).sample(rows, min(CALIBRATION_N, len(rows)))
    items = "\n\n".join(f"#{i + 1} ({r.situation}; to {r.speaker_role})\n  They: {r.user_text[:400] or '(Saheli started)'}\n  Saheli: {r.text[:800]}"
                        for i, r in enumerate(pick))
    try:
        out = await router.complete("judge", system_stable=grader.PROMPT, messages=[{"role": "user", "content": [{"type": "text", "text": items}]}],
                                    max_tokens=4000, effort="low", essential=False)
    except router.AllModelsFailed as exc:
        return {"skipped": str(exc)[:120]}
    strong = {v.get("n"): bool(v.get("pass", True)) for v in grader.parse(out.text).get("verdicts") or []}
    pairs = [(r.judge_pass, strong[i + 1]) for i, r in enumerate(pick) if i + 1 in strong]
    agree = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None
    data = {"n": len(pairs), "agreement": round(agree, 3) if agree is not None else None, "trusted": agree is None or agree >= AGREEMENT_FLOOR,
            "cheap_fail_rate": round(sum(1 for a, _ in pairs if not a) / len(pairs), 3) if pairs else None,
            "strong_fail_rate": round(sum(1 for _, b in pairs if not b) / len(pairs), 3) if pairs else None}
    if agree is not None and agree < AGREEMENT_FLOOR:
        logger.error("LEARN ALERT grader agreement %.0f%% < %.0f%%", agree * 100, AGREEMENT_FLOOR * 100)
    return await _save(sessions, "calibration", data)


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
            alarms.append(f"{name}: grader failures up {rb['judge_fail']:.0%} → {ra['judge_fail']:.0%}")
        if ra["reply_rate"] is not None and rb["reply_rate"] is not None and ra["reply_rate"] < rb["reply_rate"] - 0.1:
            alarms.append(f"{name}: fewer people answer her messages {rb['reply_rate']:.0%} → {ra['reply_rate']:.0%}")
    for a in alarms:
        logger.error("LEARN ALERT drift %s", a)
    return await _save(sessions, "drift", {"alarms": alarms, "by": by})


# ── rule proposals ─────────────────────────────────────────────────────────────

PROPOSE = """You review a WhatsApp care companion (Saheli) for elderly people in India. Below are replies a quality
grader FAILED this week, with the grader's note. If a clear, repeated problem would be fixed by a short new rule in her
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
    lines = [f"[{r.situation}] They: {anonymise(r.user_text)[:200]} | Saheli: {anonymise(r.text)[:300]} | grader: {r.judge_note}" for r in fails]
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
        for p in (grader.parse(out.text).get("proposals") or [])[:3]:
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
    data = {"specs": (grader.parse(out.text).get("specs") or [])[:5], "from": top}
    return await _save(sessions, "gap_specs", data)
