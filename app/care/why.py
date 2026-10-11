"""The why-trail, read back (D-13): for a task, a care-record change, a reminder change or an alert, what led to it.

Every turn (a message, a scheduled wake-up, a task update, a dashboard action) leaves one trace_turns row: the trace it
belongs to, the message and reply (scrubbed: no login codes or secrets, app.core.trace.scrub), which model and role
answered and what it cost, and each tool call (name, a short scrubbed summary of its arguments, ok / refused / error).
Ledger events, memory versions, open loops, tasks and model spend carry the same trace id, so `chain()` puts the whole
story together for the admin console's "Why?" view: the message, model, role and cost, the tool calls, the confirms
(who, when, total: the confirm / place / cancel audit), the outcome, and anything learned (a skill, how the reply was
scored). Rows are kept 180 days, like the rest of the conversation-level history.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, String, Text, delete, false, or_, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core import clock, trace
from app.db.session import Base

logger = logging.getLogger(__name__)

KEEP_FOR = timedelta(days=180)
TEXT_CHARS = 800
MAX_TOOLS = 40
# what a confirm / place / cancel looks like in the ledger (the F4 audit)
AUDIT_KINDS = ("task_confirmed", "task_approved", "task_cancelled", "task_cancel_requested", "task_done", "task_failed", "task_placed")


class TraceTurn(Base):
    __tablename__ = "trace_turns"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(160))
    family_id: Mapped[str] = mapped_column(String(80))
    thread_id: Mapped[str] = mapped_column(String(64))  # who it was with (the speaker), or "dashboard"
    kind: Mapped[str] = mapped_column(String(16))  # message | system | dashboard
    channel: Mapped[str] = mapped_column(String(16), default="")
    message_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # the task this turn was about (its trace is the task's)
    turn_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # Saheli's reply turn (reply_log.turn_id)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ms: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")
    reply: Mapped[str] = mapped_column(Text, default="")
    models: Mapped[list] = mapped_column(JSONB, default=list)  # [{role, model, calls, costInr}]
    cost_inr: Mapped[float] = mapped_column(Float, default=0.0)
    tools: Mapped[list] = mapped_column(JSONB, default=list)  # [{name, args, status, detail}]
    tasks: Mapped[list] = mapped_column(JSONB, default=list)  # task ids its tool calls touched

    __table_args__ = (
        Index("ix_trace_turns_trace", "trace_id", "id"),
        Index("ix_trace_turns_family_at", "family_id", "at"),
    )


def _models(calls: list) -> tuple[list[dict], float]:
    by: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0.0, 0, 0])
    for role, model, cost, tin, tout in calls or []:
        b = by[(role, model)]
        b[0] += 1
        b[1] += cost
        b[2] += tin
        b[3] += tout
    rows = [{"role": r, "model": m, "calls": v[0], "costInr": round(v[1], 4), "tokensIn": v[2], "tokensOut": v[3]} for (r, m), v in by.items()]
    return rows, round(sum(v[1] for v in by.values()), 4)


def tool_rows(actions: list[dict]) -> tuple[list[dict], set[str]]:
    """The turn's tool calls, scrubbed, and every secret value they carried (to mask in the message and reply too)."""
    secrets: set[str] = set()
    for a in actions or []:
        secrets |= trace.secrets_in(a.get("args"))
    rows = []
    for a in (actions or [])[:MAX_TOOLS]:
        status = "ok" if a.get("ok") else ("refused" if a.get("refused") else "error")
        detail = trace.scrub(str(a.get("refused") or a.get("error") or ""), secrets)[:200]
        rows.append({"name": a.get("tool"), "args": trace.summarize(a.get("args"), secrets), "status": status, **({"detail": detail} if detail else {})})
    return rows, secrets


async def record_turn(session, *, trace_id: str, family_id: str, thread_id: str, kind: str, channel: str = "", message_ref: str | None = None,
                      task_id: str | None = None, turn_id: int | None = None, at: datetime | None = None, ms: int = 0, text: str = "",
                      reply: str = "", actions: list[dict] | None = None, calls: list | None = None) -> TraceTurn:
    tools, secrets = tool_rows(actions or [])
    models, cost = _models(calls or [])
    touched = sorted({str((a.get("args") or {}).get("task_id")) for a in actions or [] if (a.get("args") or {}).get("task_id")})
    row = TraceTurn(
        trace_id=trace.clip(trace_id), family_id=family_id[:80], thread_id=(thread_id or "")[:64], kind=kind[:16], channel=(channel or "")[:16],
        message_ref=trace.clip(message_ref), task_id=(str(task_id)[:64] if task_id else None), turn_id=turn_id, at=at or clock.now(), ms=ms,
        text=trace.scrub(text, secrets)[:TEXT_CHARS], reply=trace.scrub(reply, secrets)[:TEXT_CHARS], models=models, cost_inr=cost,
        tools=tools, tasks=touched,
    )
    session.add(row)
    await session.flush()
    return row


async def record_turn_safely(session, **kw) -> None:
    """Never costs the person their reply: a failure is rolled back to its own savepoint and logged."""
    try:
        async with session.begin_nested():
            await record_turn(session, **kw)
    except Exception:  # noqa: BLE001
        logger.warning("why-trail record failed trace=%s", kw.get("trace_id"), exc_info=True)


async def prune(session, *, keep_for: timedelta = KEEP_FOR) -> int:
    from app.llm.spend import LlmTraceSpend

    cutoff = clock.now() - keep_for
    n = (await session.execute(delete(TraceTurn).where(TraceTurn.at < cutoff))).rowcount or 0
    n += (await session.execute(delete(LlmTraceSpend).where(LlmTraceSpend.last_at < cutoff))).rowcount or 0
    return n


# ── reading a chain back ─────────────────────────────────────────────────────


def _iso(d) -> str | None:
    return d.isoformat() if d else None


def _turn_json(t: TraceTurn) -> dict:
    return {"trace": t.trace_id, "at": _iso(t.at), "kind": t.kind, "by": t.thread_id, "channel": t.channel, "taskId": t.task_id,
            "text": t.text, "reply": t.reply, "models": t.models, "costInr": t.cost_inr, "tools": t.tools, "ms": t.ms,
            "messageRef": t.message_ref if t.kind == "message" else None}


def _task_json(t) -> dict:
    d = t.details or {}
    r = t.result or {}
    return {"id": str(t.id), "trace": trace.of_task(t), "service": t.service, "kind": t.kind, "goal": t.goal, "status": t.status,
            "phase": t.phase, "requestedBy": t.requested_by, "subjectId": t.subject_id, "createdAt": _iso(t.created_at),
            "updatedAt": _iso(t.updated_at), "confirmedBy": d.get("confirmed_by"), "confirmedTotal": d.get("confirmed_total"),
            "approvedBy": d.get("approved_by"), "placed": bool(r.get("placed") or r.get("booked")), "orderId": r.get("order_id"),
            "total": r.get("total"), "agentCostInr": (d.get("metrics") or {}).get("cost_inr"),
            "history": [{"at": h.get("at"), "status": h.get("status"), "phase": h.get("phase"), "note": trace.scrub(h.get("note") or "")}
                        for h in (t.history or [])[-30:]]}


def _event_json(e) -> dict:
    p = e.payload or {}
    keep = {k: p[k] for k in ("task_id", "by", "total", "via", "key", "version", "result", "mode") if k in p and k != "result"}
    if isinstance(p.get("result"), dict):
        keep["result"] = {k: p["result"][k] for k in ("placed", "booked", "order_id", "total") if k in p["result"]}
    return {"id": e.id, "at": _iso(e.at), "kind": e.kind, "summary": trace.scrub(e.summary or "")[:300], "actor": e.actor_id,
            "subjectId": e.subject_id, "trace": e.trace_id, "payload": keep}


async def _resolve(session, *, trace_id=None, task_id=None, version_id=None, loop_id=None, event_id=None, family_id=None) -> dict:
    """What was asked about, its family, and the trace(s) to follow."""
    from app.care import versions
    from app.care.models import CareEvent, OpenLoop
    from app.tasks.models import Task

    focus: dict = {}
    traces: list[str] = []
    fam = family_id
    if task_id:
        t = await session.get(Task, uuid.UUID(str(task_id)))
        if not t:
            return {}
        fam, traces, focus = t.family_id, [trace.of_task(t)], {"type": "task", "id": str(t.id)}
    elif version_id:
        v = await session.get(versions.MemoryVersion, int(version_id))
        if not v:
            return {}
        fam, focus = v.family_id, {"type": "change", "id": v.id, **(await versions.view(session, v))}
        focus.pop("body", None)
        traces = [v.trace_id] if v.trace_id else []
    elif loop_id:
        lp = await session.get(OpenLoop, uuid.UUID(str(loop_id)))
        if not lp:
            return {}
        fam, traces = lp.family_id, [lp.trace_id] if lp.trace_id else []
        focus = {"type": "loop", "id": str(lp.id), "kind": lp.kind, "title": lp.title, "status": lp.status, "alert": lp.alert_rule,
                 "createdAt": _iso(lp.created_at)}
    elif event_id:
        e = await session.get(CareEvent, int(event_id))
        if not e:
            return {}
        fam, traces, focus = e.family_id, [e.trace_id] if e.trace_id else [], {"type": "event", **_event_json(e)}
    elif trace_id:
        traces, focus = [trace_id], {"type": "trace", "id": trace_id}
    if not fam or (family_id and fam != family_id):
        return {}  # asked within one family: never show another's
    return {"family": fam, "traces": [t for t in traces if t], "focus": focus}


async def chain(session, **ask) -> dict | None:
    """The why-trail for a task (task_id), a memory change (version_id), an open loop or alert (loop_id), a ledger
    event (event_id) or a trace (trace_id). None when there is no such thing."""
    from app.care import versions
    from app.care.models import CareEvent, OpenLoop
    from app.care.skillbook import Skill
    from app.learn.models import ReplyLog
    from app.llm import spend
    from app.tasks.models import Task

    got = await _resolve(session, **ask)
    if not got:
        return None
    fam = got["family"]
    traces: set[str] = set(got["traces"])
    out: dict = {"family": got["family"], "focus": got["focus"], "traces": [], "start": None, "turns": [], "tasks": [], "confirms": [],
                 "changes": [], "events": [], "loops": [], "spend": {"totalInr": 0.0, "rows": []}, "learned": [], "outcome": None}
    if not traces:
        out["note"] = "This was written before the why-trail existed (or by a path that keeps no trace)."
        return out

    # the tasks of these traces, and the turns that moved them (a "yes" that confirmed, filed with via=its own trace)
    tasks = list((await session.execute(select(Task).where(Task.trace_id.in_(traces), Task.family_id == fam))).scalars()) if traces else []
    if got["focus"].get("type") == "task":
        t = await session.get(Task, uuid.UUID(got["focus"]["id"]))
        tasks = [t] + [x for x in tasks if x.id != t.id]
    task_ids = [str(t.id) for t in tasks]
    if task_ids:
        via = (await session.execute(select(CareEvent.payload["via"].astext).where(
            CareEvent.family_id == fam, CareEvent.payload["task_id"].astext.in_(task_ids), CareEvent.payload["via"].astext.is_not(None)))).scalars().all()
        traces |= {v for v in via if v}
    # only traces this family's own rows carry: a trace id typed under another family shows nothing of it
    mine: set[str] = {trace.of_task(t) for t in tasks}
    for col, model in ((TraceTurn.trace_id, TraceTurn), (CareEvent.trace_id, CareEvent), (versions.MemoryVersion.trace_id, versions.MemoryVersion),
                       (OpenLoop.trace_id, OpenLoop)):
        mine |= set((await session.execute(select(col).where(col.in_(traces), model.family_id == fam).distinct())).scalars().all())
    tl = sorted(traces & mine)
    if not tl:
        out["note"] = "Nothing in this family carries that trace."
        return out
    task_cond = TraceTurn.task_id.in_(task_ids) if task_ids else false()

    turns = list((await session.execute(select(TraceTurn).where(
        TraceTurn.family_id == fam, or_(TraceTurn.trace_id.in_(tl), task_cond)).order_by(TraceTurn.at, TraceTurn.id).limit(200))).scalars())
    out["turns"] = [_turn_json(t) for t in turns]
    first = next((t for t in turns if t.trace_id in got["traces"]), turns[0] if turns else None)
    if first:
        out["start"] = {"kind": first.kind, "at": _iso(first.at), "by": first.thread_id, "channel": first.channel, "text": first.text}
    out["tasks"] = [_task_json(t) for t in tasks]

    ev_q = select(CareEvent).where(CareEvent.family_id == fam, or_(CareEvent.trace_id.in_(tl),
                                                                  CareEvent.payload["task_id"].astext.in_(task_ids) if task_ids else false()))
    events = list((await session.execute(ev_q.order_by(CareEvent.at, CareEvent.id).limit(300))).scalars())
    out["events"] = [_event_json(e) for e in events]
    out["confirms"] = [{"at": _iso(e.at), "kind": e.kind.removeprefix("task_"), "by": (e.payload or {}).get("by") or e.actor_id,
                        "total": (e.payload or {}).get("total"), "taskId": (e.payload or {}).get("task_id"), "via": (e.payload or {}).get("via"),
                        "summary": trace.scrub(e.summary or "")[:200]}
                       for e in events if e.kind in AUDIT_KINDS]

    vs = list((await session.execute(select(versions.MemoryVersion).where(versions.MemoryVersion.trace_id.in_(tl), versions.MemoryVersion.family_id == fam)
                                     .order_by(versions.MemoryVersion.id).limit(100))).scalars())
    for v in vs:
        d = await versions.view(session, v)
        out["changes"].append({"id": v.id, "kind": v.kind, "target": v.target, "op": v.op, "summary": d.get("summary"), "by": d.get("by"),
                               "where": d.get("where"), "at": _iso(v.at), "status": v.status})
    loops = list((await session.execute(select(OpenLoop).where(OpenLoop.trace_id.in_(tl), OpenLoop.family_id == fam)
                                        .order_by(OpenLoop.created_at).limit(50))).scalars())
    out["loops"] = [{"id": str(lp.id), "kind": lp.kind, "title": trace.scrub(lp.title)[:300], "status": lp.status, "alert": lp.alert_rule,
                     "createdAt": _iso(lp.created_at)} for lp in loops]

    rows = await spend.for_traces(session, tl)
    out["spend"] = {"totalInr": round(sum(r["costInr"] for r in rows), 4), "rows": rows,
                    "byTask": {tid: round(sum(r["costInr"] for r in rows if r["taskId"] == tid), 4) for tid in task_ids}}

    for tid in tl:  # skills this taught (their evidence names the trace)
        for s in (await session.execute(select(Skill).where(Skill.evidence.contains([{"trace": tid}]),
                                                            or_(Skill.family_id == fam, Skill.family_id.is_(None))).limit(10))).scalars():
            out["learned"].append({"type": "skill", "id": s.id, "scope": s.scope, "title": s.title, "status": s.status, "body": s.body[:200]})
    turn_ids = [t.turn_id for t in turns if t.turn_id]
    if turn_ids:
        for r in (await session.execute(select(ReplyLog).where(ReplyLog.turn_id.in_(turn_ids), ReplyLog.family_id == fam))).scalars():
            if r.scored_at:
                out["learned"].append({"type": "reply_score", "turnId": r.turn_id, "replied": r.replied, "tone": r.tone, "corrected": r.corrected,
                                       "judgePass": r.judge_pass, "score": r.score, "inCorpus": r.in_corpus})

    if tasks:
        t = tasks[0]
        out["outcome"] = {"type": "task", "status": t.status, "placed": bool((t.result or {}).get("placed") or (t.result or {}).get("booked")),
                          "orderId": (t.result or {}).get("order_id"), "at": _iso(t.updated_at)}
    elif turns:
        last = turns[-1]
        out["outcome"] = {"type": "reply", "reply": last.reply, "at": _iso(last.at)}
    out["traces"] = tl
    return out


async def recent(session, family_id: str, *, days: int = 14, limit: int = 40) -> dict:
    """What can be asked "why?" about in one family: its tasks, care-record and link changes, and alerts / questions."""
    from app.care import versions
    from app.care.models import OpenLoop
    from app.tasks.models import Task

    since = clock.now() - timedelta(days=max(1, min(days, 90)))
    tasks = (await session.execute(select(Task).where(Task.family_id == family_id, Task.created_at >= since)
                                   .order_by(Task.created_at.desc()).limit(limit))).scalars()
    vs = (await session.execute(select(versions.MemoryVersion).where(
        versions.MemoryVersion.family_id == family_id, versions.MemoryVersion.at >= since, versions.MemoryVersion.kind.in_(("fact", "link")),
        versions.MemoryVersion.op != "baseline").order_by(versions.MemoryVersion.id.desc()).limit(limit))).scalars()
    loops = (await session.execute(select(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.created_at >= since,
                                                          OpenLoop.alert_rule.is_not(None)).order_by(OpenLoop.created_at.desc()).limit(limit))).scalars()
    return {
        "tasks": [{"id": str(t.id), "service": t.service, "kind": t.kind, "status": t.status, "goal": t.goal[:120], "at": _iso(t.created_at),
                   "traced": bool(t.trace_id)} for t in tasks],
        "changes": [{"id": v.id, "kind": v.kind, "target": v.target, "op": v.op, "text": v.body[:140], "at": _iso(v.at), "traced": bool(v.trace_id)}
                    for v in vs],
        "alerts": [{"id": str(lp.id), "kind": lp.kind, "title": trace.scrub(lp.title)[:140], "status": lp.status, "at": _iso(lp.created_at),
                    "traced": bool(lp.trace_id)} for lp in loops],
    }
