"""Per-task metrics and cost, and per-agent rollups with alerts.

Cost is an estimate from browser steps (BROWSER_COST_PER_STEP_INR) plus a flat per-call cost for the
connector (CONNECTOR_COST_PER_CALL_INR); set them from the real invoices.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.tasks.models import Task

ALERT_MIN_TASKS = 5
ALERT_SUCCESS_BELOW = 0.7


def _per_step() -> float:
    # Measured 2026-10-11 against the Browser Use bill: two Swiggy orders estimated at ~200 steps cost $0.81 (~₹68), so
    # about ₹0.3 a step (₹1 paused orders at "₹200" after ~₹60 of real spend).
    return float(os.getenv("BROWSER_COST_PER_STEP_INR", "0.3"))


def _per_call() -> float:
    return float(os.getenv("CONNECTOR_COST_PER_CALL_INR", "0.05"))


def _m(task: Task) -> dict:
    return dict((task.details or {}).get("metrics") or {})


def _save(task: Task, m: dict) -> None:
    task.details = {**(task.details or {}), "metrics": m}


def on_create(task: Task, agent: str) -> None:
    _save(task, {"agent": agent, "created_at": clock.now().isoformat(), "browser_steps": 0, "browser_runs": 0, "connector_calls": 0, "cost_inr": 0.0})


def on_channel(task: Task, channel: str, why: str) -> None:
    m = _m(task)
    m.setdefault("channels", []).append({"channel": channel, "why": why, "phase": task.phase, "at": clock.now().isoformat()})
    _save(task, m)


def on_browser_run(task: Task, steps: int) -> None:
    m = _m(task)
    m["browser_runs"] = int(m.get("browser_runs", 0)) + 1
    m["browser_steps"] = int(m.get("browser_steps", 0)) + max(0, steps)
    m["cost_inr"] = round(float(m.get("cost_inr", 0)) + max(0, steps) * _per_step(), 2)
    _save(task, m)


def on_connector_call(task: Task) -> None:
    m = _m(task)
    m["connector_calls"] = int(m.get("connector_calls", 0)) + 1
    m["cost_inr"] = round(float(m.get("cost_inr", 0)) + _per_call(), 2)
    _save(task, m)


def on_milestone(task: Task, name: str) -> None:
    """ready (cart/fares), placed, failed, cancelled — first time only."""
    m = _m(task)
    m.setdefault(f"{name}_at", clock.now().isoformat())
    if name == "failed":
        m["failure"] = (task.history or [{}])[-1].get("note", "")[:200] if task.history else ""
    _save(task, m)


def _secs(a: str | None, b: str | None) -> float | None:
    if not (a and b):
        return None
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()


async def rollup(session: AsyncSession, days: int = 7) -> dict:
    since = clock.now() - timedelta(days=days)
    tasks = list((await session.execute(select(Task).where(Task.created_at >= since))).scalars())
    groups: dict[tuple, list[Task]] = defaultdict(list)
    for t in tasks:
        m = _m(t)
        groups[(m.get("agent") or "unknown", t.service)].append(t)
    rows, alerts = [], []
    for (agent, service), ts in sorted(groups.items()):
        finished = [t for t in ts if t.status in ("done", "failed", "cancelled")]
        ok = [t for t in finished if t.status == "done" and ((t.result or {}).get("placed") or (t.result or {}).get("booked") or (t.result or {}).get("cancelled"))]
        failed = [t for t in finished if t.status == "failed"]
        to_ready = [s for s in (_secs(_m(t).get("created_at"), _m(t).get("ready_at")) for t in ts) if s is not None]
        to_placed = [s for s in (_secs(_m(t).get("created_at"), _m(t).get("placed_at")) for t in ts) if s is not None]
        channels = defaultdict(int)
        for t in ts:
            for c in _m(t).get("channels") or []:
                channels[c["channel"]] += 1
        reasons = defaultdict(int)
        for t in failed:
            reasons[(_m(t).get("failure") or "unknown")[:80]] += 1
        decided = len(ok) + len(failed)
        rate = (len(ok) / decided) if decided else None
        row = {
            "agent": agent, "service": service, "tasks": len(ts), "placed": len(ok), "failed": len(failed),
            "cancelled": sum(1 for t in finished if t.status == "cancelled"), "live": len(ts) - len(finished),
            "successRate": round(rate, 3) if rate is not None else None,
            "medianSecondsToReady": round(median(to_ready)) if to_ready else None,
            "medianSecondsToPlaced": round(median(to_placed)) if to_placed else None,
            "costInr": round(sum(float(_m(t).get("cost_inr", 0)) for t in ts), 2),
            "browserSteps": sum(int(_m(t).get("browser_steps", 0)) for t in ts),
            "channels": dict(channels), "topFailures": sorted(reasons.items(), key=lambda x: -x[1])[:5],
        }
        rows.append(row)
        if decided >= ALERT_MIN_TASKS and rate is not None and rate < ALERT_SUCCESS_BELOW:
            alerts.append(f"{agent}/{service}: success {rate:.0%} over {decided} tasks (below {ALERT_SUCCESS_BELOW:.0%})")
    return {
        "days": days, "rows": rows, "alerts": alerts,
        "totals": {"tasks": len(tasks), "costInr": round(sum(r["costInr"] for r in rows), 2)},
    }
