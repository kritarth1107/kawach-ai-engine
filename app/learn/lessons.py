"""Weekly: learn lessons from the anonymised corpus, check them, and try them on 10% of families.

1. For each situation with enough examples, the best and worst replies of the last four weeks go to a model that
   writes at most five short lessons about *how* to reply (length, tone, structure, language, when to ask) and
   picks the best example replies. It never sees a family's identity (the corpus is anonymised).
2. The draft passes a safety gate: no lesson may touch medical content, alerts, privacy or the fixed rules, and
   examples must be clean. A failed draft is rejected.
3. A passing draft becomes the canary (LEARN_CANARY_PERCENT of families, default 20%).
4. Nightly, the canary's scores are compared with live. Worse: rolled back automatically. Better: promoted when
   approved (or automatically with LEARN_AUTO_PROMOTE=on). No decision in 21 days: retired.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import timedelta
from statistics import mean, pvariance

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core import clock
from app.learn import playbook
from app.learn.anonymise import EMAIL, PHONE, PINCODE
from app.learn.models import LearningExample, PlaybookVersion, ReplyLog
from app.learn.situations import SITUATIONS

logger = logging.getLogger(__name__)

WINDOW_DAYS = 28
MAX_LESSONS = 5
LESSON_MAX_CHARS = 200
CANARY_MAX_DAYS = 21
CANARY_MIN_DAYS = 2  # a lesson runs on its share of families at least 2 days before it is promoted (§10)
ROLLBACK_POINTS = 0.02  # the canary's judge failures 2 points above the rest: rolled back at once
Z = 1.64  # one-sided ~95%

PROMPT = """You help a WhatsApp care companion called Saheli learn *how to reply* from real, anonymised conversations
with elderly people in India and their families. You get GOOD replies (people replied warmly, followed the advice,
did not complain) and BAD replies (ignored, annoyed, corrected her, or failed a quality check) for one situation.

Write at most 5 short lessons (each under 25 words) about style that separates good from bad: length, tone, warmth,
structure, asking vs telling, one question at a time, language and script, timing words. Lessons must be general
(no names, places, medicines or numbers) and must NOT be about medical decisions, doses, alerts, privacy, or
breaking any rule. If the difference is unclear, write fewer lessons.
Also pick up to 3 GOOD replies (by number) that best show these lessons.
Return only JSON: {"lessons": ["..."], "examples": [1, 4]}"""

FORBIDDEN = re.compile(
    r"\b(dose|dosage|mg|tablet count|double|extra (tablet|dose)|skip (the )?(alert|reminder|dose)|don'?t alert|no alert|"
    r"never tell|hide|keep (it )?secret|stop (the )?reminders|diagnos|instead of (a |the )?doctor|no need (to see|for) (a |the )?doctor|"
    r"always agree|phone number|home address|their address|override|system prompt|ignore (the|your) rules?)\b",
    re.I,
)


def min_examples() -> int:
    return int(os.getenv("LEARN_MIN_EXAMPLES", "30"))


def min_n() -> int:
    return int(os.getenv("LEARN_CANARY_MIN_N", "20"))


def parse(text: str) -> dict:
    t = (text or "").strip()
    if "{" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}


# ── safety gate ────────────────────────────────────────────────────────────────


def lesson_problems(lesson: str) -> list[str]:
    out = []
    if len(lesson) > LESSON_MAX_CHARS:
        out.append("too long")
    if hit := FORBIDDEN.search(lesson):
        out.append(f"touches a fixed rule or medical content ({hit.group(0)})")
    if re.search(r"\d", lesson):
        out.append("contains a number")
    if re.search(r"\[(PERSON|MED|PHONE|ADDRESS|PIN|ID)", lesson):
        out.append("contains a placeholder")
    return out


def example_problems(ex: dict) -> list[str]:
    from app.brain import guards

    out = []
    text = f"{ex.get('context', '')} {ex.get('reply', '')}"
    if PHONE.search(text) or EMAIL.search(text) or PINCODE.search(text):
        out.append("identifying detail")
    if guards.leaked_reasoning(ex.get("reply", "")):
        out.append("leaked reasoning")
    if len(ex.get("reply", "")) > 600:
        out.append("too long to be a model reply")
    return out


def gate(pb: PlaybookVersion) -> dict:
    report: dict = {"lessons": {}, "examples": {}, "passed": True}
    approved = {r for rs in ((pb.gate or {}).get("approved_rules") or {}).values() for r in rs}
    for situation, items in (pb.lessons or {}).items():
        if situation not in SITUATIONS and situation != "general":
            report["passed"] = False
            report["lessons"][situation] = ["unknown situation"]
            continue
        if len([x for x in items if x not in approved]) > MAX_LESSONS + 1:
            report["passed"] = False
            report["lessons"][situation] = ["too many lessons"]
        for lesson in items:
            if lesson in approved:  # the founder approved this rule; it still goes through the trial
                continue
            p = lesson_problems(lesson)
            if p:
                report["passed"] = False
                report["lessons"].setdefault(situation, []).append(f"{lesson[:80]}: {', '.join(p)}")
    for situation, items in (pb.examples or {}).items():
        for ex in items:
            p = example_problems(ex)
            if p:
                report["passed"] = False
                report["examples"].setdefault(situation, []).append(f"{ex.get('reply', '')[:60]}: {', '.join(p)}")
    return report


# ── the weekly job ─────────────────────────────────────────────────────────────


async def _next_version(session: AsyncSession) -> int:
    return int((await session.execute(select(func.coalesce(func.max(PlaybookVersion.version), 0)))).scalar_one()) + 1


async def draft(sessions: async_sessionmaker) -> PlaybookVersion | None:
    """Build next week's draft from the corpus. Situations without enough examples keep their live lessons."""
    from app.llm import router

    since = clock.now() - timedelta(days=WINDOW_DAYS)
    async with sessions() as session:
        rows = list((await session.execute(select(LearningExample).where(LearningExample.at >= since))).scalars())
        live, _ = await playbook.versions(session)
        lessons = dict((live.lessons if live else {}) or {})
        examples = dict((live.examples if live else {}) or {})
        by: dict[str, list[LearningExample]] = defaultdict(list)
        for r in rows:
            by[r.situation].append(r)
        learned = []
        for situation, items in by.items():
            if len(items) < min_examples() or situation not in SITUATIONS:
                continue
            ranked = sorted(items, key=lambda r: r.score, reverse=True)
            good = [r for r in ranked if r.score >= 0.2][:40]
            bad = [r for r in reversed(ranked) if r.score <= -0.1][:40]
            if len(good) < 5:
                continue
            listing = ("GOOD:\n" + "\n".join(f"{i + 1}. [{r.lang}] They: {r.context[:200]} | Saheli: {r.reply[:300]}" for i, r in enumerate(good))
                       + "\n\nBAD:\n" + "\n".join(f"- [{r.lang}] They: {r.context[:200]} | Saheli: {r.reply[:300]} (signals {r.signals})" for r in bad))
            try:
                out = await router.complete("learn", system_stable=PROMPT,
                                            messages=[{"role": "user", "content": [{"type": "text", "text": f"SITUATION: {situation} — {SITUATIONS[situation]}\n\n{listing}"}]}],
                                            max_tokens=6000, effort="low", essential=False)
                got = parse(out.text)
            except router.AllModelsFailed as exc:
                logger.warning("lessons stopped at %s: %s", situation, exc)
                break
            new = [str(x).strip() for x in got.get("lessons") or [] if str(x).strip()][:MAX_LESSONS]
            picks = [good[i - 1] for i in got.get("examples") or [] if isinstance(i, int) and 1 <= i <= len(good)][:3]
            if not picks:  # the model did not pick: the best replies in different scripts
                seen = set()
                picks = [r for r in good if not (r.lang in seen or seen.add(r.lang))][:3]
            if new:
                lessons[situation] = new
            examples[situation] = [{"context": r.context[:200], "reply": r.reply[:400], "lang": r.lang, "score": r.score} for r in picks]
            learned.append(situation)
        from app.learn import review

        rules = await review.approved_rules(session)
        for situation, items in rules.items():
            lessons[situation] = (list(lessons.get(situation) or []) + items)[: MAX_LESSONS + len(items)]
            learned.append(situation)
        if not learned:
            return None
        pb = PlaybookVersion(version=await _next_version(session), status="draft", lessons=lessons, examples=examples,
                             created_at=clock.now(), scope=[] if "general" in learned else sorted(set(learned)),  # a general rule touches every situation
                             note=f"learned: {', '.join(sorted(set(learned)))}; {len(rows)} examples" + (f"; {sum(map(len, rules.values()))} approved rules" if rules else ""),
                             gate={"approved_rules": rules})
        await review.mark_in_playbook(session)
        session.add(pb)
        await session.commit()
        return pb


async def gate_and_canary(sessions: async_sessionmaker, version: int, replay: dict | None = None) -> dict:
    """replay: the replay gate's result when it ran elsewhere (the learn-gate workflow in the test project, which posts it to
    /v2/admin/learn/playbooks/{version}/replay); without it the gate is tried here (production: unavailable, so it waits)."""
    async with sessions() as session:
        pb = await session.get(PlaybookVersion, version)
        if not pb or pb.status != "draft":
            return {"skipped": True}
        from app.learn import review

        rules = await review.approved_rules(session)
        lessons_now = dict(pb.lessons or {})
        for situation, items in rules.items():
            have = list(lessons_now.get(situation) or [])
            extra = [item for item in items if item not in have]
            if extra:
                lessons_now[situation] = have + extra
        if lessons_now != (pb.lessons or {}):
            pb.lessons = lessons_now
            if "general" in rules:
                pb.scope = []
            pb.gate = {**(pb.gate or {}), "approved_rules": {**((pb.gate or {}).get("approved_rules") or {}), **rules}}
        pb.gate = {**(pb.gate or {}), **gate(pb)}
        if pb.gate["passed"]:
            # §10: a lesson must also pass the replay cases on the real model before it reaches anyone: every fixed case
            # still passes, and the failures it targets now pass. No replay runner yet means it waits as a draft.
            if replay is None:
                replay = await replay_gate(sessions, pb)
            pb.gate = {**pb.gate, "replay": replay}
            if replay.get("status") == "failed":
                pb.gate["passed"] = False
            elif replay.get("status") != "passed":
                await session.commit()
                return {"version": version, "status": pb.status, "gate": pb.gate}
        if not pb.gate["passed"]:
            pb.status = "rejected"
        elif os.getenv("LEARN_AUTO_CANARY", "on") == "on":
            current = (await session.execute(select(PlaybookVersion).where(PlaybookVersion.status == "canary"))).scalars().all()
            for c in current:
                c.status, c.note = "retired", (c.note + " | replaced by a newer canary").strip(" |")
            pb.status, pb.canary_since = "canary", clock.now()
            await review.mark_in_playbook(session)
        await session.commit()
        playbook.reset_cache()
        return {"version": version, "status": pb.status, "gate": pb.gate}


async def replay_gate(sessions: async_sessionmaker, pb: PlaybookVersion) -> dict:
    """Run the replay cases (trust plan C3) with this draft's playbook. {"status": passed|failed|unavailable, ...}."""
    import importlib

    try:
        replay = importlib.import_module("app.learn.replay")
    except ImportError:
        return {"status": "unavailable", "why": "replay runner not built yet"}
    run = getattr(replay, "run_for_playbook", None)
    if run is None:
        return {"status": "unavailable", "why": "replay runner has no run_for_playbook"}
    try:
        from app.learn.replay import deploy_case_ids

        res = await run(sessions, pb, budget_inr=float(os.getenv("LEARN_REPLAY_MAX_INR", "30")), max_cases=8,
                        only_ids=deploy_case_ids() or None)
    except Exception as exc:  # noqa: BLE001 — a broken runner never lets a lesson through
        logger.exception("replay gate failed version=%s", pb.version)
        return {"status": "unavailable", "why": str(exc)[:200]}
    regressions = list(res.get("regressions") or [])
    return {"status": "failed" if regressions else "passed", **res}


# ── canary decisions ───────────────────────────────────────────────────────────


def _stats(xs: list[float]) -> tuple[float, float, int]:
    return (mean(xs), pvariance(xs), len(xs)) if xs else (0.0, 0.0, 0)


async def evaluate(sessions: async_sessionmaker) -> dict:
    """Compare the canary with live since the canary started. Safety signals veto a promotion."""
    async with sessions() as session:
        canary = (await session.execute(select(PlaybookVersion).where(PlaybookVersion.status == "canary"))).scalars().first()
        if not canary or not canary.canary_since:
            return {"canary": None}
        rows = list((await session.execute(
            select(ReplyLog).where(ReplyLog.at >= canary.canary_since, ReplyLog.score.is_not(None))
        )).scalars())
        scope = set(canary.scope or [])
        c = [r for r in rows if r.arm == "canary" and r.playbook_version == canary.version and (not scope or r.situation in scope)]
        live = [r for r in rows if r.arm == "live" and (not scope or r.situation in scope)]
        (mc, vc, nc), (ml, vl, nl) = _stats([r.score for r in c]), _stats([r.score for r in live])

        def rate(rs, f):
            return sum(1 for r in rs if f(r)) / len(rs) if rs else 0.0

        res = {"scope": sorted(scope), "n_canary": nc, "n_live": nl, "mean_canary": round(mc, 4), "mean_live": round(ml, 4),
               "annoyed_canary": round(rate(c, lambda r: r.tone == "annoyed"), 4), "annoyed_live": round(rate(live, lambda r: r.tone == "annoyed"), 4),
               "judge_fail_canary": round(rate([r for r in c if r.judge_pass is not None], lambda r: r.judge_pass is False), 4),
               "judge_fail_live": round(rate([r for r in live if r.judge_pass is not None], lambda r: r.judge_pass is False), 4),
               "checked_at": clock.now().isoformat()}
        decision = "waiting"
        # Safety first, at any sample size: a canary that fails the judge 2 points more often than the rest, or annoys
        # people more, is rolled back at once (§10).
        judged_c = [r for r in c if r.judge_pass is not None]
        worse_safety = (len(judged_c) >= 5 and res["judge_fail_canary"] > res["judge_fail_live"] + ROLLBACK_POINTS) or \
            (nc >= 5 and res["annoyed_canary"] > res["annoyed_live"] + 0.03)
        if worse_safety:
            decision = "rolled_back"
            canary.status = "rejected"
        elif nc >= min_n() and nl >= min_n() and clock.now() - canary.canary_since >= timedelta(days=CANARY_MIN_DAYS):
            z = (mc - ml) / math.sqrt(vc / nc + vl / nl) if (vc + vl) > 0 else 0.0
            res["z"] = round(z, 3)
            if z <= -Z:
                decision = "rolled_back"
                canary.status = "rejected"
            else:
                # not worse than the rest after two days: founder decision Q10 promotes it automatically
                decision = "not_worse"
                from app.learn import review

                cal = await review.latest(sessions, "calibration")
                if cal and cal.get("trusted") is False:
                    decision = "not_worse_but_grader_untrusted"
                elif os.getenv("LEARN_AUTO_PROMOTE", "on") == "on":
                    await _promote(session, canary, by="auto")
                    decision = "promoted"
        if decision == "waiting" and clock.now() - canary.canary_since > timedelta(days=CANARY_MAX_DAYS):
            decision = "no_difference"
            canary.status = "retired"
        res["decision"] = decision
        canary.canary = res
        await session.commit()
        playbook.reset_cache()
        return {"canary": canary.version, **res}


async def _promote(session: AsyncSession, pb: PlaybookVersion, *, by: str) -> None:
    for old in (await session.execute(select(PlaybookVersion).where(PlaybookVersion.status == "live"))).scalars():
        old.status = "retired"
    pb.status, pb.live_since, pb.approved_by = "live", clock.now(), by


async def approve(sessions: async_sessionmaker, version: int, by: str) -> dict:
    async with sessions() as session:
        pb = await session.get(PlaybookVersion, version)
        if not pb or pb.status not in ("canary", "draft"):
            return {"ok": False, "why": "only a draft or canary can be approved"}
        if pb.status == "draft" and not (pb.gate or {}).get("passed", False):
            pb.gate = gate(pb)
            if not pb.gate["passed"]:
                await session.commit()
                return {"ok": False, "why": "failed the safety gate", "gate": pb.gate}
        await _promote(session, pb, by=by)
        await session.commit()
    playbook.reset_cache()
    return {"ok": True, "live": version}


async def block(sessions: async_sessionmaker, version: int, by: str) -> dict:
    async with sessions() as session:
        pb = await session.get(PlaybookVersion, version)
        if not pb:
            return {"ok": False}
        was_live = pb.status == "live"
        pb.status, pb.note = "blocked", (pb.note + f" | blocked by {by}").strip(" |")
        if was_live:  # fall back to the previous good version
            prev = (await session.execute(select(PlaybookVersion).where(PlaybookVersion.status == "retired", PlaybookVersion.live_since.is_not(None))
                                          .order_by(PlaybookVersion.live_since.desc()))).scalars().first()
            if prev:
                prev.status = "live"
        await session.commit()
    playbook.reset_cache()
    return {"ok": True}


async def weekly(sessions: async_sessionmaker) -> dict:
    pb = await draft(sessions)
    if not pb:
        return {"draft": None, "why": "not enough new examples"}
    return {"draft": pb.version, **await gate_and_canary(sessions, pb.version)}
