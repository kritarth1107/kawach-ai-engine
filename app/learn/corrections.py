"""Learning that works with a handful of families: every correction becomes a case, a lesson candidate and a test.

The corpus-based weekly lessons need volume (many consented families). Until then, the strongest signal is when someone
corrects Saheli. Nightly this collects, from the last day:
  - chat corrections: the next message after Saheli's reply says she got it wrong ("galat", "maine aisa nahi kaha", …);
  - memory corrections: a caregiver changed on the dashboard a fact Saheli had saved in the last 3 days;
  - reply-guard rewrites: a draft the code checks stopped (made-up fact, wrong script, false claim, repeat);
  - thumbs-down votes on the dashboard.
Each becomes an anonymised case (family "learn", kind correction_case): the weekly rule proposals read them next to the
grader's failures, the admin console lists them, and GET /v2/learn/corrections?format=jsonl exports them as regression
fixtures for eval (so a correction is never needed twice). Also proposes per-family "usual orders" as skills a caregiver
approves.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, FamilyRoster, Turn
from app.core import clock

logger = logging.getLogger(__name__)

LEARN = "learn"


def learn_family(family_id: str) -> str:
    return f"{LEARN}:{family_id}"


async def note_guard(session: AsyncSession, *, family_id: str, thread_id: str, draft: str, problems: list[str], user_text: str) -> None:
    """A draft the reply guard sent back (kept out of the family's own ledger)."""
    await store.record_event(session, family_id=learn_family(family_id), subject_id=thread_id, kind="guard_rewrite",
                             summary="; ".join(problems)[:300], payload={"draft": draft[:600], "they": user_text[:300], "problems": problems[:6]})


async def _case(session, *, family_id: str, kind: str, ref: str, they: str, saheli: str, problem: str, situation: str = "general") -> bool:
    from app.learn.anonymise import anonymise

    return await store.record_event(
        session, family_id=LEARN, subject_id=LEARN, kind="correction_case", summary=f"{kind}: {problem[:200]}", ref=ref,
        payload={"kind": kind, "situation": situation, "they": anonymise(they or "")[:400], "saheli": anonymise(saheli or "")[:600],
                 "problem": anonymise(problem or "")[:400], "family": family_id[:8]},
    ) is not None


async def collect(sessions: async_sessionmaker, *, days: int = 1) -> dict:
    from app.learn.models import ReplyLog

    since = clock.now() - timedelta(days=days)
    out = Counter()
    async with sessions() as s:
        # chat corrections (scored nightly from what they said next)
        for r in (await s.execute(select(ReplyLog).where(ReplyLog.corrected.is_(True), ReplyLog.at >= since))).scalars():
            nxt = (await s.execute(select(Turn).where(Turn.family_id == r.family_id, Turn.thread_id == r.thread_id, Turn.role == "user",
                                                      Turn.at > r.at).order_by(Turn.at).limit(1))).scalar_one_or_none()
            if await _case(s, family_id=r.family_id, kind="chat_correction", ref=f"cc:reply:{r.id}", they=r.user_text, saheli=r.text,
                           problem=f"they corrected her: {nxt.text if nxt else ''}", situation=r.situation):
                out["chat"] += 1
        # guard rewrites
        for e in (await s.execute(select(CareEvent).where(CareEvent.kind == "guard_rewrite", CareEvent.at >= since))).scalars():
            p = e.payload or {}
            if await _case(s, family_id=e.family_id.removeprefix(f"{LEARN}:"), kind="guard_rewrite", ref=f"cc:guard:{e.id}",
                           they=p.get("they", ""), saheli=p.get("draft", ""), problem="; ".join(p.get("problems") or [])):
                out["guard"] += 1
        # memory corrections: a dashboard edit of something Saheli saved shortly before
        for e in (await s.execute(select(CareEvent).where(CareEvent.kind == "dashboard_edit", CareEvent.at >= since))).scalars():
            key = (e.payload or {}).get("key")
            name = e.summary.split(":")[0].strip().lower()
            saved = (await s.execute(select(CareEvent).where(
                CareEvent.family_id == e.family_id, CareEvent.subject_id == e.subject_id, CareEvent.kind.in_(("fact_created", "fact_superseded")),
                CareEvent.at >= e.at - timedelta(days=3), CareEvent.at < e.at).order_by(CareEvent.at.desc()))).scalars()
            hit = next((x for x in saved if (key and (x.payload or {}).get("key") == key) or (name and name in (x.summary or "").lower())), None)
            if hit and hit.actor_id != e.actor_id and await _case(
                    s, family_id=e.family_id, kind="memory_correction", ref=f"cc:mem:{e.id}", they=hit.summary,
                    saheli=f"saved: {hit.summary}", problem=f"a caregiver changed it on the dashboard to: {e.summary}"):
                out["memory"] += 1
        # thumbs down
        for e in (await s.execute(select(CareEvent).where(CareEvent.kind == "feedback", CareEvent.at >= since))).scalars():
            if (e.payload or {}).get("vote") == "down" and await _case(
                    s, family_id=e.family_id, kind="thumbs_down", ref=f"cc:fb:{e.id}", they="", saheli=str((e.payload or {}).get("target")),
                    problem="a caregiver voted it down on the dashboard"):
                out["thumbs_down"] += 1
        await s.commit()
    return dict(out)


async def cases(session: AsyncSession, *, days: int = 30, limit: int = 200) -> list[dict]:
    rows = (await session.execute(select(CareEvent).where(CareEvent.family_id == LEARN, CareEvent.kind == "correction_case",
                                                          CareEvent.at >= clock.now() - timedelta(days=days))
                                  .order_by(CareEvent.at.desc()).limit(limit))).scalars()
    return [{"id": e.id, "at": e.at.isoformat(), **(e.payload or {})} for e in rows]


def as_fixture(c: dict) -> str:
    """One JSONL line for eval: the situation, what they said, the reply that went wrong and why (expect: not that)."""
    return json.dumps({"source": c["kind"], "situation": c.get("situation") or "general", "message": c.get("they", ""),
                       "bad_reply": c.get("saheli", ""), "problem": c.get("problem", ""), "at": c.get("at")}, ensure_ascii=False)


def for_proposals(rows: list[dict]) -> list[str]:
    """Lines for the weekly rule proposals, next to the grader's failures."""
    return [f"[{c.get('situation') or 'general'}] They: {c.get('they', '')[:200]} | Saheli: {c.get('saheli', '')[:300]} | {c['kind']}: {c.get('problem', '')[:200]}"
            for c in rows]


async def propose_usuals(session: AsyncSession, family_id: str, subject_id: str) -> dict | None:
    """Something ordered 3+ times in 60 days becomes a suggested skill ('usually orders Amul milk from Blinkit') that a
    caregiver approves; then Saheli offers it first next time."""
    from app.care import skillbook
    from app.tasks.models import Task
    from app.tasks.skills import SKILLS

    rows = (await session.execute(select(Task).where(Task.family_id == family_id, Task.subject_id == subject_id, Task.status == "done",
                                                     Task.created_at >= clock.now() - timedelta(days=60)))).scalars()
    seen = Counter()
    for t in rows:
        if not (t.result or {}).get("placed"):
            continue
        for i in (t.result or {}).get("items") or []:
            # skills hold no numbers (amounts live in the care record): "Amul Taaza Milk 500 ml" → "Amul Taaza Milk"
            name = re.sub(r"\s{2,}", " ", re.sub(r"\b\d+(?:\.\d+)?\s*(?:ml|l|ltr|g|gm|kg|pcs?|x|n)?\b|[\d()]", " ", str(i.get("name") or ""), flags=re.I)).strip(" -,")
            if name:
                seen[(t.service, name)] += 1
    top = [(svc, name, n) for (svc, name), n in seen.most_common(3) if n >= 3]
    if not top:
        return None
    text = "Usually orders " + "; ".join(f"{name} from {SKILLS.get(svc, {}).get('label', svc)}" for svc, name, _ in top) + \
           ". Offer these first when they ask for these things."
    out = await skillbook.save_family(session, family_id, subject_id, text, title="Usual orders", source="dream", by="dream",
                                      evidence=[{"orders": [{"store": s, "item": n, "times": c} for s, n, c in top]}])
    return out if out.get("saved") else None


async def nightly(sessions: async_sessionmaker) -> dict:
    stats = await collect(sessions)
    proposed = 0
    async with sessions() as s:
        for r in (await s.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")))).scalars():
            elder = (r.elder or {}).get("id")
            if elder:
                try:
                    if await propose_usuals(s, r.family_id, elder):
                        proposed += 1
                except Exception:  # noqa: BLE001
                    logger.exception("usual orders failed family=%s", r.family_id)
        await s.commit()
    return {**stats, "usuals_proposed": proposed}
