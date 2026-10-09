"""Is the flywheel turning? The numbers from the founder's architecture slide: completion, time, reopen rate and cost per
task, plus adherence, whether people answer Saheli, approvals and how much the families hand over week by week.

GET /v2/learn/flywheel?days=14 (admin console). Read-only; every number is computed from the ledger, tasks and turns.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from statistics import median

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import doses, store
from app.care.models import CareEvent, CareFact, FamilyRoster, OpenLoop, Turn
from app.core import clock
from app.tasks.models import Task

DELEGATED_LOOPS = ("family_task", "appointment", "refill", "delivery")
REPLY_WINDOW = timedelta(hours=3)


def _placed(t: Task) -> bool:
    r = t.result or {}
    return bool(r.get("placed") or r.get("booked"))


def _declined(t: Task) -> bool:
    return any("declined by" in str(h.get("note", "")) or "picked" in str(h.get("note", "")) for h in t.history or [])


def _secs(a, b) -> float | None:
    return (b - a).total_seconds() if a and b else None


async def _adherence(session: AsyncSession, family_id: str, subject: str, days: list[str]) -> float | None:
    meds = await store.facts(session, family_id, subject, domains=["medicine"], statuses=("active",))
    if not meds:
        return None
    evs = await store.events(session, family_id, subject, since=clock.now() - timedelta(days=len(days) + 1),
                             kinds=["dose_taken", "dose_empty_strip"], limit=5000)
    by_day = defaultdict(list)
    for e in evs:
        by_day[e.day].append(e)
    due_n = taken_n = 0
    for day in days:
        for f in meds:
            if not doses.due_on(f.value, day):
                continue
            slots = len(f.value.get("times") or []) or 1
            name = doses._first(str(f.value.get("name") or f.key.split(":", 1)[-1]))
            hits = sum(1 for e in by_day.get(day, []) if name and name == doses._first(str((e.payload or {}).get("medicine") or e.summary.split(":")[0])))
            due_n += slots
            taken_n += min(slots, hits)
    return round(taken_n / due_n, 3) if due_n else None


async def compute(session: AsyncSession, *, days: int = 14) -> dict:
    now = clock.now()
    since = now - timedelta(days=days)
    families = [r for r in (await session.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")))).scalars()]
    fids = [f.family_id for f in families]

    # ── delegated work: orders, rides, bookings + chores, appointments, refills, deliveries
    tasks = list((await session.execute(select(Task).where(Task.family_id.in_(fids), Task.created_at >= since))).scalars())
    real = [t for t in tasks if not _declined(t)]
    finished = [t for t in real if t.status in ("done", "failed", "cancelled")]
    completed = [t for t in finished if t.status == "done" and _placed(t)]
    to_done = [s for s in (_secs(t.created_at, t.updated_at) for t in completed) if s]
    failed = [t for t in finished if t.status == "failed"]
    # reopened: the same family asked the same store again within a day of a failure (the first try did not do the job)
    reopened = sum(1 for f in failed if any(o.family_id == f.family_id and o.service == f.service and o.id != f.id
                                            and timedelta(0) < o.created_at - f.created_at < timedelta(days=1) for o in tasks))
    loops = list((await session.execute(select(OpenLoop).where(OpenLoop.family_id.in_(fids), OpenLoop.kind.in_(DELEGATED_LOOPS),
                                                               OpenLoop.created_at >= since))).scalars())
    loops_done = [l for l in loops if l.status == "done"]
    browser_inr = sum(float(((t.details or {}).get("metrics") or {}).get("cost_inr", 0) or 0) for t in tasks)

    # ── conversations: messages, model cost per message, do people answer Saheli's own messages?
    turns = list((await session.execute(select(Turn).where(Turn.family_id.in_(fids), Turn.at >= since).order_by(Turn.at))).scalars())
    user_turns = [t for t in turns if t.role == "user" and not t.thread_id.startswith("saheli")]
    by_thread = defaultdict(list)
    for t in user_turns:
        by_thread[(t.family_id, t.thread_id)].append(t.at)
    proactive = [t for t in turns if t.role == "assistant" and (t.meta or {}).get("proactive")
                 and not str((t.meta or {}).get("ref") or (t.meta or {}).get("fallback_delivery") or "").startswith("task:")]
    answered = sum(1 for p in proactive if any(timedelta(0) < u - p.at <= REPLY_WINDOW for u in by_thread[(p.family_id, p.thread_id)]))
    from app.llm.spend import LlmSpend

    spend_rows = (await session.execute(select(LlmSpend).where(LlmSpend.day >= since.date()))).scalars()
    llm_inr = sum(float(r.cost_inr or 0) for r in spend_rows)

    # ── check-ins and approvals
    checkins = list((await session.execute(select(CareEvent).where(CareEvent.family_id.in_(fids), CareEvent.kind == "checkin",
                                                                   CareEvent.at >= since))).scalars())
    checkin_answered = sum(1 for c in checkins if any(timedelta(0) < u - c.at <= REPLY_WINDOW for u in by_thread[(c.family_id, c.subject_id)]))
    approvals = [t for t in tasks if (t.details or {}).get("approved_by") or any("waiting for approval" in str(h.get("note", "")) for h in t.history or [])]

    # ── per family: adherence this week vs last, delegation per week, memory health
    week = [clock.ist_day(now - timedelta(days=d)) for d in range(1, 8)]
    prev = [clock.ist_day(now - timedelta(days=d)) for d in range(8, 15)]
    per_family = []
    for r in families:
        elder = (r.elder or {}).get("id")
        if not elder:
            continue
        from app.care import freshness

        facts = list((await session.execute(select(CareFact).where(CareFact.family_id == r.family_id,
                                                                   CareFact.status.in_(("active", "pending"))))).scalars())
        stale = await freshness.stale_facts(session, r.family_id, elder, [f for f in facts if f.subject_id == elder])
        fam_tasks = [t for t in tasks if t.family_id == r.family_id and not _declined(t)]
        per_family.append({
            "family": r.family_id, "elder": (r.elder or {}).get("name"),
            "adherence7d": await _adherence(session, r.family_id, elder, week),
            "adherencePrev7d": await _adherence(session, r.family_id, elder, prev),
            "delegated7d": sum(1 for t in fam_tasks if t.created_at >= now - timedelta(days=7))
                           + sum(1 for l in loops if l.family_id == r.family_id and l.created_at >= now - timedelta(days=7)),
            "delegatedPrev7d": sum(1 for t in fam_tasks if now - timedelta(days=14) <= t.created_at < now - timedelta(days=7)),
            "messages7d": sum(1 for t in user_turns if t.family_id == r.family_id and t.at >= now - timedelta(days=7)),
            "facts": sum(1 for f in facts if f.status == "active"), "pending": sum(1 for f in facts if f.status == "pending"),
            "staleFacts": len(stale),
        })

    def rate(a: int, b: int):
        return round(a / b, 3) if b else None

    return {
        "days": days, "families": len(fids),
        "tasks": {
            "started": len(real), "finished": len(finished), "completed": len(completed), "failed": len(failed),
            "completionRate": rate(len(completed), len(finished)), "reopenRate": rate(reopened, len(failed)) if failed else None,
            "medianMinutesToComplete": round(median(to_done) / 60, 1) if to_done else None,
            "chores": {"opened": len(loops), "done": len(loops_done), "completionRate": rate(len(loops_done), len(loops))},
            "browserInr": round(browser_inr, 1), "browserInrPerCompleted": round(browser_inr / len(completed), 1) if completed else None,
        },
        "conversations": {
            "userMessages": len(user_turns), "modelInr": round(llm_inr, 1),
            "modelInrPerMessage": round(llm_inr / len(user_turns), 2) if user_turns else None,
            "saheliOwnMessages": len(proactive), "answeredWithin3h": answered, "replyRate": rate(answered, len(proactive)),
        },
        "checkins": {"sent": len(checkins), "answered": checkin_answered, "answerRate": rate(checkin_answered, len(checkins))},
        "approvals": {"asked": len(approvals), "approved": sum(1 for t in approvals if (t.details or {}).get("approved_by"))},
        "perFamily": per_family,
    }
