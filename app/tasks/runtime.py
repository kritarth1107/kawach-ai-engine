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
from app.specialists import channels, guard, metrics
from app.specialists.agents import specialist_for
from app.specialists.contract import CONTRACT_VERSION, Limits
from app.tasks.browser_use import AgentRun, BrowserAgent
from app.tasks.models import SkillNote, Task
from app.tasks.skills import SKILLS

logger = logging.getLogger(__name__)

LIVE = ("queued", "running", "needs_input", "awaiting_confirm")
TASK_LIFETIME = timedelta(minutes=45)
RUN_TIMEOUT = timedelta(minutes=12)

Notify = Callable[[str, str, str], Awaitable[None]]  # (family_id, requested_by, prompt)


rupees = guard.rupees
MAX_RETRIES = 1  # automatic retries of a browser run that crashed without a report (prepare only, never place)
MAX_SKILL_NOTES = 12


def note(task: Task, text: str) -> None:
    task.history = [*(task.history or []), {"at": clock.now().isoformat(), "phase": task.phase, "status": task.status, "note": text[:300]}]
    task.updated_at = clock.now()


class TaskRefused(ValueError):
    """The request breaks a rule (guard.check_request); nothing was started."""


async def create(
    session: AsyncSession, *, family_id: str, subject_id: str, requested_by: str, service: str, kind: str, goal: str, details: dict,
    limits: Limits | None = None,
) -> Task:
    if service not in SKILLS:
        raise ValueError(f"unknown service {service}")
    agent = specialist_for(service)
    # One live task per service per family: a retry or a repeated ask never starts a second order.
    for t in await live_tasks(session, family_id):
        if t.service == service and t.kind == kind:
            return t
    limits = limits or Limits()
    verdict = guard.check_request(kind, agent.name, details.get("items") or [], limits, pickup=details.get("pickup"), drop=details.get("drop"))
    if not verdict.ok:
        raise TaskRefused("; ".join(verdict.block))
    now = clock.now()
    task = Task(
        id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, requested_by=requested_by, service=service, kind=kind,
        goal=goal, details={**details, "contract": CONTRACT_VERSION, "agent": agent.name, "limits": limits.to_dict()},
        status="queued", phase="prepare", created_at=now, updated_at=now, deadline_at=now + TASK_LIFETIME, history=[], result={},
    )
    metrics.on_create(task, agent.name)
    session.add(task)
    await session.flush()
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="task_started", summary=f"{SKILLS[service]['label']}: {goal}",
        payload={"task_id": str(task.id), "agent": agent.name}, actor_id=requested_by,
    )
    return task


async def live_tasks(session: AsyncSession, family_id: str) -> list[Task]:
    q = select(Task).where(Task.family_id == family_id, Task.status.in_(LIVE)).order_by(Task.created_at)
    return list((await session.execute(q)).scalars())


def describe(task: Task) -> str:
    r = task.result or {}
    d = task.details or {}
    bits = [f"[{task.id}] {SKILLS[task.service]['label']} {task.kind}: {task.goal}", f"status {task.status}/{task.phase}"]
    if d.get("channel"):
        bits.append(f"via {d['channel']}")
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


async def _learned(session: AsyncSession, service: str) -> list[str]:
    return list((await session.execute(select(SkillNote.note).where(SkillNote.service == service).order_by(SkillNote.id.desc()).limit(MAX_SKILL_NOTES))).scalars())


async def _hints(session: AsyncSession, service: str) -> str:
    return specialist_for(service).hints(service, await _learned(session, service))


def _goal(task: Task) -> str:
    return specialist_for(task.service).goal(task)


async def _start_run(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for: Callable[[Task], Awaitable[str | None]], extra: str = "") -> None:
    spec = specialist_for(task.service)
    goal = _goal(task) + (f"\n{extra}" if extra else "")
    run = await agent.run(
        goal=goal,
        hints=spec.hints(task.service, await _learned(session, task.service)),
        schema=spec.schema(task),
        session_id=task.agent_session,
        profile_id=None if task.agent_session else await profile_for(task),
        start_url=None if task.agent_session else SKILLS[task.service]["start_url"],
        max_steps=spec.steps.get(task.phase, 40),
        metadata={"app": "kavach", "task": str(task.id), "phase": task.phase, "service": task.service, "agent": spec.name},
        llm=spec.model,
    )
    task.agent_session, task.agent_task = run.session_id, run.task_id
    task.status, task.input_needed, task.runs = "running", None, task.runs + 1
    task.details = {**(task.details or {}), "run_started": clock.now().isoformat(), "channel": "browser"}
    note(task, f"started {task.phase} run ({spec.name} agent, browser)")


async def provide_input(session: AsyncSession, task: Task, *, kind: str, value: str, by: str, by_is_elder: bool) -> str:
    """The person answered: an OTP, a confirm, a fee approval or a choice. Returns what happens next."""
    if task.status not in ("needs_input", "awaiting_confirm"):
        return f"not waiting for input (status {task.status})"
    if kind == "confirm":
        if value.lower() not in ("yes", "confirm", "true"):
            task.status, task.cancel_requested = "cancelled", True
            note(task, f"declined by {by}")
            metrics.on_milestone(task, "cancelled")
            return "declined; nothing was placed"
        d = task.details or {}
        # Who confirms matters: an elder may not confirm past the budget, whoever asked for the order.
        limit = d.get("order_limit") or Limits.from_dict(d.get("limits")).budget
        total = rupees((task.result or {}).get("total"))
        if task.kind == "order" and by_is_elder and total and total > limit:
            return f"total ₹{total:.0f} is above the ₹{limit} limit: a caregiver must confirm (ask with alert_caregiver reason approval)"
        if d.get("approval") and by_is_elder:
            return f"{'; '.join(d['approval'])} (ask with alert_caregiver reason approval)"
        fp = d.get("cart_fp") or guard.cart_fingerprint(task.result or {})
        task.details = {**d, "confirmed_by": by, "confirmed_total": (task.result or {}).get("total"), "confirmed_fp": fp,
                        "confirm_token": guard.confirm_token(str(task.id), fp, by)}
        task.phase, task.status = "place", "queued"
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
    if kind == "swap":
        d = task.details or {}
        alts = d.get("alternatives") or []
        if value.lower() in ("no", "none", "drop", "cancel"):
            task.status, task.cancel_requested = "cancelled", True
            note(task, f"no alternative wanted ({by})")
            return "dropped; nothing was ordered"
        pick = next((a for a in alts if value.lower() in a["name"].lower() or a["name"].lower() in value.lower()), None)
        if not pick:
            return "that is not one of the alternatives; read them again: " + "; ".join(a["name"] for a in alts)
        items = d.get("items") or []
        new_items = [{"name": pick["name"], "qty": items[0].get("qty", 1) if items else 1}] if len(items) <= 1 else (
            [i for i in items if not guard.missing_items([i], [])] + [{"name": pick["name"], "qty": 1}])
        v = guard.check_request("order", d.get("agent") or "shopping", new_items, Limits.from_dict(d.get("limits")))
        if not v.ok:
            return "cannot get that: " + "; ".join(v.block)
        task.details = {**d, "items": new_items, "alternatives": None, "channel": None, "connector_card": None}
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        note(task, f"swapped to {pick['name']} by {by}")
        return f"getting {pick['name']} instead; the new cart comes back for a confirm"
    if kind == "choice":
        d = task.details or {}
        if d.get("approval") and by_is_elder:
            return f"{'; '.join(d['approval'])} (ask with alert_caregiver reason approval)"
        fp = guard.cart_fingerprint({"items": [{"name": value}], "total": None}) + (d.get("cart_fp") or "")
        task.details = {**d, "choice": value, "confirmed_by": by, "confirmed_fp": fp, "confirm_token": guard.confirm_token(str(task.id), fp, by)}
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
    """(new status, what the family should hear) from an agent report for the current phase. Every cart and
    every placement passes the guard: a rule broken in the report stops it here, whatever the agent said."""
    d = task.details or {}
    limits = Limits.from_dict(d.get("limits"))
    spec = specialist_for(task.service)
    if out.get("needs_otp"):
        task.input_needed = "otp"
        return "needs_input", f"The service sent a login code to {out.get('otp_sent_to') or 'the family phone'}; ask the person who has that phone for it."
    if out.get("blocked"):
        return "failed", f"{SKILLS[task.service]['label']} blocked the request ({out.get('problem') or 'site wall'}). Offer another service."
    if task.phase in ("prepare", "otp"):
        if task.kind == "ride":
            if not out.get("options"):
                return "failed", f"No ride options came up ({out.get('problem') or 'unknown'})."
            v = guard.check_cart("ride", spec.name, [], out, limits)
            task.details = {**d, "approval": v.approval, "cart_fp": guard.cart_fingerprint({"items": [{"name": o.get("type")} for o in out["options"]], "total": None})}
            task.input_needed = "choice"
            metrics.on_milestone(task, "ready")
            extra = " ".join(v.warn + v.approval)
            return "awaiting_confirm", "Ride options are ready; tell the person the fares (and any surge) and ask which one to book." + (f" {extra}" if extra else "")
        if out.get("needs_prescription") and not guard.rx_covers(d.get("items") or [], limits.rx_on_file):
            return "failed", "This medicine needs a prescription upload; ask the family to upload the prescription on the dashboard, then try again."
        if not out.get("items"):
            alts = [a for a in out.get("alternatives") or [] if a.get("name")][:4]
            if alts and out.get("cod_available") is not False:
                # Out of stock: offer what the store has instead; nothing is swapped without the person choosing.
                task.details = {**d, "alternatives": alts}
                task.input_needed = "swap"
                listed = "; ".join(f"{a['name']}{(' ' + str(a['price'])) if a.get('price') else ''}" for a in alts)
                return "needs_input", f"Not available ({out.get('problem') or 'out of stock'}). The store has: {listed}. Ask which one to get instead, or whether to drop it."
            return "failed", f"Could not build the cart ({out.get('problem') or 'unknown'})."
        v = guard.check_cart("order", spec.name, d.get("items") or [], out, limits)
        if not v.ok:
            return "failed", "Stopped before placing: " + "; ".join(v.block) + ". Nothing was placed."
        task.details = {**d, "approval": v.approval, "cart_fp": guard.cart_fingerprint(out), "confirm_token": None}
        task.input_needed = "confirm"
        metrics.on_milestone(task, "ready")
        extra = " ".join(v.warn + v.approval)
        return "awaiting_confirm", "The cart is ready; read the items and the total to the person and ask them to confirm." + (f" {extra}" if extra else "")
    if task.phase == "place":
        if out.get("price_changed"):
            task.result = {**(task.result or {}), "total": out.get("new_total") or (task.result or {}).get("total")}
            task.details = {**d, "cart_fp": guard.cart_fingerprint(task.result), "confirm_token": None}
            task.input_needed = "confirm"
            return "awaiting_confirm", f"The total changed to {out.get('new_total')} since it was confirmed; nothing was placed. Read the new total and ask again."
        if out.get("placed") or out.get("booked"):
            v = guard.check_placed(task.kind, out, guard.rupees(d.get("confirmed_total")))
            metrics.on_milestone(task, "placed")
            if v.warn:
                return "done", "Placed, but: " + "; ".join(v.warn)
            return "done", "Placed. Tell the person the order/ride id and the ETA."
        if out.get("unclear"):
            return "failed", f"It is not clear whether the order went through ({out.get('problem')}). Do not order again; tell the caregiver to check the app."
        return "failed", f"Placing did not go through ({out.get('problem') or 'unknown'}). Nothing was charged."
    if task.phase == "cancel":
        if out.get("cancel_fee") and not d.get("fee_ok"):
            task.input_needed = "fee"
            return "needs_input", f"Cancelling now costs {out['cancel_fee']}; ask whether to cancel anyway."
        if out.get("cancelled"):
            metrics.on_milestone(task, "cancelled")
            return "cancelled", "Cancelled on the service. Tell the person."
        return "failed", f"Could not cancel ({out.get('problem') or 'unknown'}); tell the caregiver to cancel from the app."
    return "failed", "Unexpected state."


def _confirm_still_valid(task: Task) -> bool:
    d = task.details or {}
    return d.get("confirmed_fp") == (d.get("cart_fp") if task.kind == "order" else d.get("confirmed_fp")) and guard.token_valid(
        str(task.id), d.get("confirmed_fp") or "", d.get("confirmed_by") or "", d.get("confirm_token"))


async def _connector_step(session: AsyncSession, host, task: Task) -> dict | None:
    """Run prepare/place on the store's connector. Returns an agent-style report, or None to fall back to the browser."""
    metrics.on_connector_call(task)
    d = task.details or {}
    try:
        if task.phase == "place":
            r = await channels.connector_place(host, task)
            if r.get("shadow"):
                return {"placed": False, "problem": "shadow mode: nothing is placed"}
            st = r.get("status")
            if st == "placed":
                await channels.record(session, task.service, "connector", True)
                return {"placed": True, "order_id": r.get("orderId"), "total": r.get("total"), "payment_method": "Cash on Delivery", "eta": r.get("eta")}
            if st == "refused" and r.get("newCard"):
                task.details = {**d, "connector_card": r["newCard"]}
                return {"price_changed": True, "new_total": r.get("newTotal")}
            if st in ("unknown", "duplicate"):
                # The store may have taken it: never retry on another channel (no double order).
                return {"placed": False, "unclear": True, "problem": "the store did not answer after the order was sent; check the app before trying again"}
            if st == "expired":
                task.phase = "prepare"  # build a fresh card on the connector; the family confirms it again
                return {"requeue": True}
            await channels.record(session, task.service, "connector", False, r.get("detail"))
            return None
        r = await channels.connector_prepare(host, task)
        if r.get("ok"):
            await channels.record(session, task.service, "connector", True)
            task.details = {**(task.details or {}), "connector_card": r.get("card")}
            return {"items": r.get("items") or [], "total": r.get("total"), "fees": r.get("fees"), "cod_available": r.get("cod_available", True),
                    "address_used": r.get("address_used"), "eta": r.get("eta"), "logged_in": True}
        if r.get("kind") in ("cod_unavailable", "not_found"):
            await channels.record(session, task.service, "connector", True)
            return {"items": [], "cod_available": r.get("kind") != "cod_unavailable", "problem": r.get("detail"), "alternatives": r.get("alternatives") or []}
        await channels.record(session, task.service, "connector", False, r.get("detail"))
        return None
    except Exception as exc:  # noqa: BLE001 — a broken connector falls back to the browser
        await channels.record(session, task.service, "connector", False, str(exc)[:200])
        return None


async def tick(
    sessions: async_sessionmaker,
    agent: BrowserAgent,
    *,
    profile_for: Callable[[Task], Awaitable[str | None]],
    notify: Notify,
    limit: int = 20,
    host_for: Callable[[str], object] | None = None,
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
                    metrics.on_milestone(task, "failed")
                    stats["expired"] += 1
                elif task.status == "queued" and task.phase == "place" and not _confirm_still_valid(task):
                    task.status, task.input_needed = "awaiting_confirm", "confirm" if task.kind == "order" else "choice"
                    task.details = {**(task.details or {}), "confirm_token": None}
                    message = "The cart changed after it was confirmed, so nothing was placed. Read it again and ask the person to confirm."
                    note(task, "confirm no longer matches the cart")
                elif task.status == "queued":
                    report = None
                    if task.phase in ("prepare", "place") and (task.details or {}).get("channel") != "browser":
                        host = host_for(task.family_id) if host_for else None
                        if task.phase == "prepare":
                            channel, why = await channels.pick_channel(session, host, task)
                            metrics.on_channel(task, channel, why)
                            task.details = {**(task.details or {}), "channel": channel}
                        if (task.details or {}).get("channel") == "connector":
                            report = await _connector_step(session, host, task)
                            if report is None:
                                metrics.on_channel(task, "browser", "connector failed; falling back")
                                task.details = {**(task.details or {}), "channel": "browser"}
                                if task.phase == "place":
                                    task.phase = "prepare"  # the browser builds its own cart; the family confirms it again
                                note(task, "connector failed, falling back to the browser")
                    if report is not None and report.get("requeue"):
                        note(task, "connector card expired; building it again")
                    elif report is not None:
                        task.result = {**(task.result or {}), **{k: v for k, v in report.items() if v not in (None, "", [])}}
                        status, message = _outcome(task, report)
                        task.status = status
                        note(task, f"{task.phase} → {status} (connector): {message}")
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                    else:
                        extra = f"Enter this code where the login asks for it: {task.details.get('otp')}. Then continue with the task." if task.phase == "otp" else ""
                        if task.phase == "otp":
                            task.phase = "prepare"
                        await _start_run(session, agent, task, profile_for, extra)
                        stats["started"] += 1
                elif task.status == "running":
                    run: AgentRun = await agent.poll(task.agent_task)
                    stats["polled"] += 1
                    if run.status in ("finished", "failed", "stopped"):
                        metrics.on_browser_run(task, run.steps)
                        await channels.record(session, task.service, "browser", run.status == "finished" and not (run.output or {}).get("blocked"), run.error)
                        out = run.output or {}
                        retries = int((task.details or {}).get("retries", 0))
                        if run.status == "failed" and not out and retries < MAX_RETRIES and task.phase in ("prepare", "otp"):
                            # A crash or timeout of the browser run, not an answer from the store: try once more.
                            task.details = {**(task.details or {}), "retries": retries + 1}
                            task.status = "queued"
                            note(task, f"run failed ({run.error or 'no report'}); retrying once")
                            await session.commit()
                            continue
                        if run.status != "finished" and not out:
                            out = {"blocked": False, "problem": run.error or run.status}
                        task.result = {**(task.result or {}), **{k: v for k, v in out.items() if v not in (None, "", [])}}
                        status, message = _outcome(task, out)
                        task.status = status
                        note(task, f"{task.phase} → {status}: {message}")
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                        if status == "done" and task.cancel_requested:
                            task.phase, task.status = "cancel", "queued"
                            message = "It was placed just before the cancel; cancelling it on the service now."
                        if status == "failed" and out.get("problem"):
                            # The next agent on this service reads what stopped this one (once per distinct problem).
                            text_ = f"{task.phase} failed: {str(out['problem'])[:240]}"
                            if text_ not in await _learned(session, task.service):
                                session.add(SkillNote(service=task.service, note=text_, created_at=clock.now()))
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
