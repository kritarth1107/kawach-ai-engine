"""One view of everything the family has handed over: every open job has a state, an owner and a next action.

The jobs live in two stores built for their work: care_tasks (orders, rides, bookings run by the browser agents and fast
paths) and open_loops (follow-ups, questions, family chores, appointment and refill reminders). This module reads both
into one shape, so the brain, WhatsApp ("what's pending?"), the dashboard Tasks page and the metrics see the same list:

    {id, kind, title, state, owner, waiting_on, next_action, due_at, updated_at, source}

state: requested | working | waiting | done | failed | cancelled. owner: "saheli" or a member id (who must act next).
A job with no next action or past its deadline is "stuck" and is flagged, never silently dropped.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import OpenLoop
from app.core import clock
from app.tasks.models import Task

SAHELI = "saheli"
SYSTEM_IDS = {"saheli-scheduler", "saheli", "system"}
STUCK_AFTER = timedelta(hours=24)
INPUT_ACTION = {
    "go": "say which option to get (or no)",
    "otp": "send the login code the store just sent",
    "confirm": "confirm the cart (or no)",
    "choice": "pick a ride option",
    "fee": "approve or refuse the cancellation fee",
    "swap": "pick an alternative for an item out of stock",
    "approve": "approve or decline (outside the family's limits)",
}
LOOP_KIND = {"delivery": "delivery", "family_task": "family_task", "appointment": "appointment", "refill": "refill", "question": "question",
             "followup": "follow_up", "checkin": "check_in", "confirm_fact": "confirm", "memory_check": "confirm",
             "confirm_skill": "confirm", "task": "follow_up", "watch": "watch"}


def _iso(at: datetime | None) -> str | None:
    return at.isoformat() if at else None


def from_task(t: Task) -> dict:
    from app.tasks.skills import SKILLS

    label = SKILLS.get(t.service, {}).get("label", t.service)
    total = (t.result or {}).get("total")
    title = f"{label}: {t.goal}" + (f" ({total})" if total else "")
    d = t.details or {}
    if t.status in ("queued", "running"):
        state, owner, action = "working", SAHELI, "looking up and getting it ready" if t.phase in ("browse", "prepare") else f"{t.phase} in progress"
    elif t.status in ("needs_input", "awaiting_confirm"):
        state = "waiting"
        ap = d.get("approval_needed") or {}
        owner = (ap.get("approvers") or [t.requested_by])[0] if t.input_needed == "approve" else t.requested_by
        action = INPUT_ACTION.get(t.input_needed or "", "answer Saheli")
    else:
        state, owner, action = t.status, None, None
    if t.status == "done" and (t.result or {}).get("placed"):
        action = None
    return {
        "id": f"task:{t.id}", "kind": "ride" if t.kind == "ride" else ("booking" if t.kind == "booking" else "order"), "title": title,
        "state": state, "owner": owner, "waiting_on": owner if state == "waiting" else None, "next_action": action,
        "due_at": _iso(t.deadline_at) if state in ("working", "waiting") else None, "updated_at": _iso(t.updated_at),
        "requested_by": t.requested_by, "subject": t.subject_id, "source": "task",
    }


def from_loop(l: OpenLoop) -> dict:
    d = l.detail or {}
    owner = l.owner_id if l.owner_id and l.owner_id not in SYSTEM_IDS else l.subject_id
    kind = LOOP_KIND.get(l.kind, "follow_up")
    action = d.get("next_action") or {
        "family_task": f"do it: {l.title}",
        "appointment": "go to the appointment" if l.wake_at and l.wake_at <= clock.now() else "be reminded before the appointment",
        "refill": "decide whether to reorder",
        "question": "answer Saheli's question",
        "follow_up": "checks back",
        "check_in": "answer the weekly check-in",
        "confirm": "confirm or correct what Saheli saved",
        "watch": "keeps an eye on it",
        "delivery": "confirm it arrived",
    }.get(kind, "checks back")
    saheli_acts = kind in ("follow_up", "watch")
    state = "done" if l.status == "done" else ("cancelled" if l.status in ("cancelled", "expired") else ("working" if saheli_acts else "waiting"))
    return {
        "id": f"loop:{l.id}", "kind": kind, "title": l.title, "state": state, "owner": SAHELI if saheli_acts else owner,
        "waiting_on": None if saheli_acts else owner, "next_action": action if state in ("working", "waiting") else None,
        "due_at": _iso(l.wake_at) if state in ("working", "waiting") else None, "updated_at": _iso(l.updated_at),
        "requested_by": None, "subject": l.subject_id, "source": "loop",
    }


def stuck(item: dict, now: datetime | None = None) -> str | None:
    """Why an open job is stuck (no next action, or nothing has moved past its deadline), else None."""
    if item["state"] not in ("working", "waiting"):
        return None
    now = now or clock.now()
    if not item.get("next_action"):
        return "no next action"
    due = datetime.fromisoformat(item["due_at"]) if item.get("due_at") else None
    if due and now - due > timedelta(hours=2):
        return "past its deadline"
    upd = datetime.fromisoformat(item["updated_at"]) if item.get("updated_at") else None
    if not due and upd and now - upd > STUCK_AFTER:
        return "no movement for a day"
    return None


async def items(session: AsyncSession, family_id: str, *, open_only: bool = True, days: int = 7, limit: int = 60) -> list[dict]:
    since = clock.now() - timedelta(days=days)
    tq = select(Task).where(Task.family_id == family_id)
    lq = select(OpenLoop).where(OpenLoop.family_id == family_id)
    if open_only:
        tq = tq.where(Task.status.in_(("queued", "running", "needs_input", "awaiting_confirm")))
        lq = lq.where(OpenLoop.status == "open")
    else:
        tq = tq.where(or_(Task.created_at >= since, Task.status.in_(("queued", "running", "needs_input", "awaiting_confirm"))))
        lq = lq.where(or_(OpenLoop.created_at >= since, OpenLoop.status == "open"))
    rows = [from_task(t) for t in (await session.execute(tq.order_by(Task.updated_at.desc()).limit(limit))).scalars()]
    rows += [from_loop(l) for l in (await session.execute(lq.order_by(OpenLoop.updated_at.desc()).limit(limit))).scalars()]
    now = clock.now()
    for r in rows:
        r["stuck"] = stuck(r, now)
    order = {"waiting": 0, "working": 1, "requested": 2, "failed": 3, "done": 4, "cancelled": 5}
    rows.sort(key=lambda r: (order.get(r["state"], 9), r.get("due_at") or "9999"))
    return rows[:limit]


def brief(rows: list[dict], names: dict[str, str], *, max_lines: int = 8) -> str:
    """OPEN WORK block for the brain: who must do what next."""
    live = [r for r in rows if r["state"] in ("working", "waiting")]
    if not live:
        return "OPEN WORK: nothing pending."
    out = ["OPEN WORK (every job: who acts next, and what):"]
    for r in live[:max_lines]:
        who = "Saheli" if r["owner"] == SAHELI else names.get(r["owner"] or "", r["owner"] or "someone")
        due = f", by {clock.ist(datetime.fromisoformat(r['due_at'])).strftime('%d %b %H:%M')}" if r.get("due_at") else ""
        flag = f" [STUCK: {r['stuck']}]" if r.get("stuck") else ""
        out.append(f"  - {r['title'][:90]} → {who}: {r['next_action']}{due}{flag}")
    if len(live) > max_lines:
        out.append(f"  …and {len(live) - max_lines} more (whats_pending lists all)")
    return "\n".join(out)
