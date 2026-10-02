"""Task runtime: carries each order or ride through prepare → confirm → place → done, durably.

Every minute a tick advances running tasks: it polls the browser agent, reads its structured report,
and moves the task on. Anything the family must know or decide (an OTP, the cart to confirm, the
order placed, a failure) becomes a system turn for the brain, which tells the person in its own words.
Nothing is placed without an explicit confirm, payment is cash on delivery only, and a cancel is
honoured at every step, including after placing (a cancel run on the service).
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.core import clock
from app.tasks.browser_use import AgentRun, BrowserAgent
from app.tasks.models import SkillNote, Task
from app.tasks.skills import COMMON, SKILLS, schema_for

logger = logging.getLogger(__name__)

LIVE = ("queued", "running", "needs_input", "awaiting_confirm")
STEPS = {"prepare": 60, "otp": 40, "place": 30, "cancel": 30}
TASK_LIFETIME = timedelta(minutes=45)
RUN_TIMEOUT = timedelta(minutes=12)
DEFAULT_ORDER_LIMIT = 1500  # ₹: above this, an order the elder asked for needs a caregiver's OK

Notify = Callable[[str, str, str], Awaitable[None]]  # (family_id, requested_by, prompt)


def rupees(text: str | None) -> float | None:
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)", (text or "").replace("₹", ""))
    return float(m.group(1).replace(",", "")) if m else None


def note(task: Task, text: str) -> None:
    task.history = [*(task.history or []), {"at": clock.now().isoformat(), "phase": task.phase, "status": task.status, "note": text[:300]}]
    task.updated_at = clock.now()


async def create(
    session: AsyncSession, *, family_id: str, subject_id: str, requested_by: str, service: str, kind: str, goal: str, details: dict
) -> Task:
    if service not in SKILLS:
        raise ValueError(f"unknown service {service}")
    now = clock.now()
    task = Task(
        id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, requested_by=requested_by, service=service, kind=kind,
        goal=goal, details=details, status="queued", phase="prepare", created_at=now, updated_at=now,
        deadline_at=now + TASK_LIFETIME, history=[], result={},
    )
    session.add(task)
    await session.flush()
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="task_started", summary=f"{SKILLS[service]['label']}: {goal}",
        payload={"task_id": str(task.id)}, actor_id=requested_by,
    )
    return task


async def live_tasks(session: AsyncSession, family_id: str) -> list[Task]:
    q = select(Task).where(Task.family_id == family_id, Task.status.in_(LIVE)).order_by(Task.created_at)
    return list((await session.execute(q)).scalars())


def describe(task: Task) -> str:
    r = task.result or {}
    bits = [f"[{task.id}] {SKILLS[task.service]['label']} {task.kind}: {task.goal}", f"status {task.status}/{task.phase}"]
    if task.input_needed:
        bits.append(f"waiting for {task.input_needed}")
    if r.get("items"):
        bits.append("cart: " + "; ".join(f"{i.get('qty', 1)} x {i.get('name')} {i.get('price', '')}".strip() for i in r["items"][:12]))
    if r.get("alternatives"):
        bits.append("alternatives: " + "; ".join(map(str, r["alternatives"][:5])))
    if r.get("options"):
        bits.append("ride options: " + "; ".join(f"{o.get('type')} {o.get('fare')} (pickup in {o.get('eta', '?')})" for o in r["options"][:6]))
    if r.get("surge"):
        bits.append("SURGE pricing")
    if r.get("total"):
        bits.append(f"total {r['total']}" + (f" incl. fees {r['fees']}" if r.get("fees") else ""))
    if r.get("eta") and task.status == "done":
        bits.append(f"eta {r['eta']}")
    if r.get("order_id") or r.get("ride_id"):
        bits.append(f"id {r.get('order_id') or r.get('ride_id')}")
    if task.cancel_requested:
        bits.append("cancel requested")
    return ", ".join(bits)


async def _hints(session: AsyncSession, service: str) -> str:
    learned = (
        await session.execute(select(SkillNote.note).where(SkillNote.service == service).order_by(SkillNote.id.desc()).limit(12))
    ).scalars()
    learned_txt = "\n".join(f"- {n}" for n in learned)
    return f"{COMMON}\n\nABOUT {SKILLS[service]['label'].upper()}:\n{SKILLS[service]['hints']}" + (
        f"\n\nLEARNED FROM EARLIER RUNS:\n{learned_txt}" if learned_txt else ""
    )


def _goal(task: Task) -> str:
    d = task.details or {}
    label = SKILLS[task.service]["label"]
    if task.kind == "ride":
        what = f"from {d.get('pickup')} to {d.get('drop')}" + (f", vehicle {d['vehicle']}" if d.get("vehicle") else "")
        if task.phase == "prepare":
            return f"On {label}, find the ride options {what} with cash payment. Do NOT book. Report the options with fares, surge, and whether you are logged in."
        if task.phase == "place":
            choice = d.get("choice") or "the option the family confirmed"
            return f"On {label}, book the ride {what}: {choice}, payment Cash. Report booked, ride_id, driver and status."
        if task.phase == "cancel":
            return f"On {label}, cancel the current ride {(task.result or {}).get('ride_id', '')}. If a cancellation fee is shown, do NOT confirm: report cancel_fee. Otherwise cancel and report cancelled."
    items = "; ".join(f"{i.get('qty', 1)} x {i.get('name')}" for i in d.get("items") or []) or task.goal
    where = f" Deliver to the saved address{(' near ' + d['area']) if d.get('area') else ''}." if task.phase in ("prepare", "place") else ""
    if task.phase == "prepare":
        return (f"On {label}, put exactly these in the cart: {items}.{where} Go to checkout up to the payment step and check that "
                f"Cash/Pay on Delivery is offered. Do NOT place the order. Report items with prices, total with fees, cod_available, eta.")
    if task.phase == "place":
        return (f"On {label}, the cart for: {items} is ready. Place the order now with Cash on Delivery only.{where} "
                f"Report placed, order_id, payment_method, total and eta.")
    if task.phase == "cancel":
        return (f"On {label}, open the order {(task.result or {}).get('order_id', 'just placed')} and cancel it. If a cancellation "
                f"fee is shown, do NOT confirm: report cancel_fee. Otherwise cancel and report cancelled.")
    return task.goal


async def _start_run(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for: Callable[[Task], Awaitable[str | None]], extra: str = "") -> None:
    goal = _goal(task) + (f"\n{extra}" if extra else "")
    run = await agent.run(
        goal=goal,
        hints=await _hints(session, task.service),
        schema=schema_for("ride" if task.kind == "ride" else "order", task.phase),
        session_id=task.agent_session,
        profile_id=None if task.agent_session else await profile_for(task),
        start_url=None if task.agent_session else SKILLS[task.service]["start_url"],
        max_steps=STEPS.get(task.phase, 40),
        metadata={"app": "kavach", "task": str(task.id), "phase": task.phase, "service": task.service},
    )
    task.agent_session, task.agent_task = run.session_id, run.task_id
    task.status, task.input_needed, task.runs = "running", None, task.runs + 1
    task.details = {**(task.details or {}), "run_started": clock.now().isoformat()}
    note(task, f"started {task.phase} run")


async def provide_input(session: AsyncSession, task: Task, *, kind: str, value: str, by: str, by_is_elder: bool) -> str:
    """The person answered: an OTP, a confirm, a fee approval or a choice. Returns what happens next."""
    if task.status not in ("needs_input", "awaiting_confirm"):
        return f"not waiting for input (status {task.status})"
    if kind == "confirm":
        if value.lower() not in ("yes", "confirm", "true"):
            task.status, task.cancel_requested = "cancelled", True
            note(task, f"declined by {by}")
            return "declined; nothing was placed"
        limit = (task.details or {}).get("order_limit", DEFAULT_ORDER_LIMIT)
        total = rupees((task.result or {}).get("total"))
        if task.kind == "order" and by_is_elder and total and total > limit:
            return f"total ₹{total:.0f} is above the ₹{limit} limit: a caregiver must confirm (ask with alert_caregiver reason approval)"
        task.phase, task.status = "place", "queued"
        task.details = {**(task.details or {}), "confirmed_by": by, "confirmed_total": (task.result or {}).get("total")}
        note(task, f"confirmed by {by}")
        return "confirmed; placing it now (NOT placed yet; you will get a task update when it is placed)"
    if kind == "otp":
        if not re.fullmatch(r"\d{4,8}", value.strip()):
            return "that is not a code; ask for the digits only"
        task.phase, task.status = "otp", "queued"
        task.details = {**(task.details or {}), "otp": value.strip()}
        note(task, "otp received")
        return "code received; continuing"
    if kind == "fee":
        if value.lower() in ("yes", "approve", "true"):
            task.details = {**(task.details or {}), "fee_ok": True}
            task.phase, task.status = "cancel", "queued"
            note(task, f"cancel fee approved by {by}")
            return "cancelling with the fee"
        task.status = "done"
        note(task, f"cancel fee declined by {by}; order stays")
        return "kept the order; not cancelled"
    if kind == "choice":
        task.details = {**(task.details or {}), "choice": value}
        task.phase, task.status = "place", "queued"
        note(task, f"choice {value} by {by}")
        return "booking that option now (NOT booked yet; you will get a task update)"
    return f"unknown input {kind}"


async def request_cancel(session: AsyncSession, agent: BrowserAgent, task: Task, *, by: str, reason: str) -> str:
    if task.status in ("failed", "cancelled"):
        return f"already {task.status}"
    task.cancel_requested = True
    note(task, f"cancel requested by {by}: {reason}")
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    if task.status == "done" and placed:
        task.phase, task.status = "cancel", "queued"
        return "already placed; cancelling it on the service now"
    if task.status == "running" and task.phase == "place":
        return "placing is in progress; it will be cancelled right after if it went through"
    if task.status == "running" and task.agent_task:
        try:
            await agent.stop(task.agent_task, end_session=True)
        except Exception:  # noqa: BLE001
            logger.warning("stop failed task=%s", task.id)
    task.status = "cancelled"
    note(task, "stopped before placing")
    return "stopped; nothing was placed"


def _outcome(task: Task, out: dict) -> tuple[str, str]:
    """(new status, what the family should hear) from an agent report for the current phase."""
    if out.get("needs_otp"):
        task.input_needed = "otp"
        return "needs_input", f"The service sent a login code to {out.get('otp_sent_to') or 'the family phone'}; ask the person who has that phone for it."
    if out.get("blocked"):
        return "failed", f"{SKILLS[task.service]['label']} blocked the request ({out.get('problem') or 'site wall'}). Offer another service."
    if task.phase in ("prepare", "otp"):
        if out.get("needs_prescription"):
            return "failed", "This medicine needs a prescription upload; ask the family to upload the prescription on the dashboard, then try again."
        if task.kind == "ride":
            if not out.get("options"):
                return "failed", f"No ride options came up ({out.get('problem') or 'unknown'})."
            task.input_needed = "choice"
            return "awaiting_confirm", "Ride options are ready; tell the person the fares (and any surge) and ask which one to book."
        if out.get("cod_available") is False:
            return "failed", "Cash on delivery is not available for this cart; nothing was placed."
        if not out.get("items"):
            return "failed", f"Could not build the cart ({out.get('problem') or 'unknown'})."
        task.input_needed = "confirm"
        return "awaiting_confirm", "The cart is ready; read the items and the total to the person and ask them to confirm."
    if task.phase == "place":
        if out.get("placed") or out.get("booked"):
            if task.kind == "order" and "cash" not in (out.get("payment_method") or "cash").lower() and "deliver" not in (out.get("payment_method") or "").lower():
                return "done", f"Placed, but the payment method shows '{out.get('payment_method')}', not cash on delivery: tell the caregiver now."
            return "done", "Placed. Tell the person the order/ride id and the ETA."
        return "failed", f"Placing did not go through ({out.get('problem') or 'unknown'}). Nothing was charged."
    if task.phase == "cancel":
        if out.get("cancel_fee") and not (task.details or {}).get("fee_ok"):
            task.input_needed = "fee"
            return "needs_input", f"Cancelling now costs {out['cancel_fee']}; ask whether to cancel anyway."
        if out.get("cancelled"):
            return "cancelled", "Cancelled on the service. Tell the person."
        return "failed", f"Could not cancel ({out.get('problem') or 'unknown'}); tell the caregiver to cancel from the app."
    return "failed", "Unexpected state."


async def tick(
    sessions: async_sessionmaker,
    agent: BrowserAgent,
    *,
    profile_for: Callable[[Task], Awaitable[str | None]],
    notify: Notify,
    limit: int = 20,
) -> dict:
    stats = {"started": 0, "polled": 0, "finished": 0, "expired": 0}
    async with sessions() as session:
        ids = list(
            (
                await session.execute(
                    select(Task.id).where(or_(Task.status.in_(("queued", "running")), (Task.status.in_(("needs_input", "awaiting_confirm"))) & (Task.deadline_at < clock.now())))
                    .order_by(Task.updated_at).limit(limit)
                )
            ).scalars()
        )
    for task_id in ids:
        message = None
        async with sessions() as session:
            task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update(skip_locked=True))).scalar_one_or_none()
            if not task:
                continue
            try:
                if task.deadline_at < clock.now() and task.status != "running":
                    task.status = "failed"
                    note(task, "timed out waiting")
                    message = "Nobody answered in time, so the task was stopped; nothing was placed." if task.phase != "cancel" else "Cancel was not finished; tell the caregiver."
                    stats["expired"] += 1
                elif task.status == "queued":
                    extra = f"Enter this code where the login asks for it: {task.details.get('otp')}. Then continue with the task." if task.phase == "otp" else ""
                    if task.phase == "otp":
                        task.phase = "prepare"
                    await _start_run(session, agent, task, profile_for, extra)
                    stats["started"] += 1
                elif task.status == "running":
                    run: AgentRun = await agent.poll(task.agent_task)
                    stats["polled"] += 1
                    if run.status in ("finished", "failed", "stopped"):
                        out = run.output or {}
                        if run.status != "finished" and not out:
                            out = {"blocked": False, "problem": run.error or run.status}
                        task.result = {**(task.result or {}), **{k: v for k, v in out.items() if v not in (None, "", [])}}
                        status, message = _outcome(task, out)
                        task.status = status
                        note(task, f"{task.phase} → {status}: {message}")
                        stats["finished"] += 1
                        if status == "done" and task.cancel_requested:
                            task.phase, task.status = "cancel", "queued"
                            message = "It was placed just before the cancel; cancelling it on the service now."
                        if status == "failed" and out.get("problem"):
                            # The next agent on this service reads what stopped this one.
                            session.add(SkillNote(service=task.service, note=f"{task.phase} failed once: {str(out['problem'])[:240]}", created_at=clock.now()))
                    elif clock.now() - datetime.fromisoformat(task.details["run_started"]) > RUN_TIMEOUT:
                        await agent.stop(task.agent_task)
                        task.status = "failed"
                        message = "The service took too long and the task was stopped; nothing was placed." if task.phase != "cancel" else "Cancel took too long; tell the caregiver."
                        note(task, "run timed out")
                if message:
                    await store.record_event(
                        session, family_id=task.family_id, subject_id=task.subject_id, kind=f"task_{task.status}",
                        summary=f"{SKILLS[task.service]['label']}: {message}", payload={"task_id": str(task.id), "result": task.result},
                    )
                await session.commit()
            except Exception:  # noqa: BLE001 — one task's failure must not stop the others
                logger.exception("task tick failed task=%s", task_id)
                await session.rollback()
                continue
        if message:
            await notify(task.family_id, task.requested_by, f"[Task update] {describe(task)}. {message}")
    return stats
