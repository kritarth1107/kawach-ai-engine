"""Task runtime: carries each order or ride through prepare → confirm → place → done, durably.

Every minute a tick advances running tasks: it polls the browser agent, reads its structured report,
and moves the task on. Anything the family must know or decide (an OTP, the cart to confirm, the
order placed, a failure) becomes a system turn for the brain, which tells the person in its own words.
Nothing is placed without an explicit confirm, payment is cash on delivery only, and a cancel is
honoured at every step, including after placing (a cancel run on the service).
"""

from __future__ import annotations

import logging
import os
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
from app.tasks import sandbox
from app.tasks.models import SkillNote, Task
from app.tasks.skills import SKILLS

logger = logging.getLogger(__name__)

LIVE = ("queued", "running", "needs_input", "awaiting_confirm")
TASK_LIFETIME = timedelta(minutes=45)
RUN_TIMEOUT = timedelta(minutes=12)

Notify = Callable[[str, str, str], Awaitable[None]]  # (family_id, requested_by, prompt)


rupees = guard.rupees
IDLE_CLOSE = timedelta(minutes=5)  # a browser waiting on a person longer than this is stopped (login stays in the profile)


def task_max_cost() -> float:
    return float(os.getenv("TASK_MAX_COST_INR", "80"))


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
    """What stopped earlier agents on this service (newest first)."""
    return list((await session.execute(select(SkillNote.note).where(SkillNote.service == service).order_by(SkillNote.id.desc()).limit(MAX_SKILL_NOTES))).scalars())


async def _skills_and_notes(session: AsyncSession, service: str) -> tuple[list[str], list[int]]:
    """Store skills that worked (best first), then problems earlier runs hit. Returns the lines and the skill ids used."""
    from app.care import skillbook

    skills = await skillbook.store_hints(session, service)
    lines = [f"path that worked ({s.title}, {s.successes} of {s.uses} runs): {s.body}" for s in skills]
    return lines + [n for n in await _learned(session, service) if not n.startswith("worked (")], [s.id for s in skills]


async def _hints(session: AsyncSession, service: str) -> str:
    return specialist_for(service).hints(service, (await _skills_and_notes(session, service))[0])


def _goal(task: Task) -> str:
    return specialist_for(task.service).goal(task)


async def _start_run(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for: Callable[[Task], Awaitable[str | None]], extra: str = "") -> None:
    spec = specialist_for(task.service)
    goal = _goal(task) + (f"\n{extra}" if extra else "")
    learned, skill_ids = await _skills_and_notes(session, task.service)
    task.details = {**(task.details or {}), "skills_used": skill_ids}
    profile_id = None
    if not task.agent_session:
        profile_id = await profile_for(task)
        await sandbox.bind_profile(session, task.family_id, task.service, profile_id)  # raises if it belongs to someone else
    run = await agent.run(
        goal=goal,
        hints=spec.hints(task.service, learned),
        schema=spec.schema(task),
        session_id=task.agent_session,
        profile_id=profile_id,
        start_url=None if task.agent_session else SKILLS[task.service]["start_url"],
        max_steps=spec.steps.get(task.phase, 40),
        metadata={"app": "kavach", "task": str(task.id), "phase": task.phase, "service": task.service, "agent": spec.name},
        llm=spec.model,
    )
    await sandbox.started(session, session_id=run.session_id, task_id=str(task.id), family_id=task.family_id, service=task.service,
                          profile_id=profile_id)
    task.agent_session, task.agent_task = run.session_id, run.task_id
    task.status, task.input_needed, task.runs = "running", None, task.runs + 1
    task.details = {**(task.details or {}), "run_started": clock.now().isoformat(), "channel": "browser"}
    note(task, f"started {task.phase} run ({spec.name} agent, browser)")


NO_WORDS = ("no", "nahi", "nahin", "na", "cancel", "drop", "none", "stop", "mat")


def _ride_fp(cart_fp: str | None, choice: str) -> str:
    return f"{cart_fp or ''}:{re.sub(r'[^a-z0-9]+', '', (choice or '').lower())}"


def _match_option(options: list[dict], value: str) -> dict | str | None:
    """The option the person named: an exact name first; a partial name only if it fits exactly one option.
    Returns "ambiguous" when it could be more than one."""
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x or "").lower()).strip()  # noqa: E731
    v = norm(value)
    if len(v) < 2:
        return None
    exact = [o for o in options if norm(o.get("type")) == v or norm(f"{o.get('type')} {o.get('fare') or ''}") == v]
    if len(exact) == 1:
        return exact[0]
    partial = [o for o in options if norm(o.get("type")) and (norm(o.get("type")) in v or v in norm(o.get("type")))]
    if len(partial) == 1:
        return partial[0]
    return "ambiguous" if partial or len(exact) > 1 else None


async def provide_input(session: AsyncSession, task: Task, *, kind: str, value: str, by: str, by_is_elder: bool) -> str:
    """The person answered: an OTP, a confirm, a fee approval, a ride choice or a swap. Returns what happens next.

    The answer must be the one the task is waiting for (task.input_needed); anything else is refused, so a
    stray "yes" or a ride name can never place something nobody was shown."""
    if task.status not in ("needs_input", "awaiting_confirm"):
        return f"not waiting for input (status {task.status})"
    value = (value or "").strip()
    d = task.details or {}
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    if kind in ("confirm", "choice", "swap") and value.lower() in NO_WORDS and not placed:
        task.status, task.cancel_requested = "cancelled", True
        note(task, f"declined by {by}")
        metrics.on_milestone(task, "cancelled")
        return "declined; nothing was placed"
    if task.input_needed and kind != task.input_needed:
        return f"the task is waiting for {task.input_needed}, not {kind}; ask for that"
    if kind == "confirm":
        if task.kind != "order" or task.status != "awaiting_confirm":
            return "nothing to confirm right now"
        if value.lower() not in ("yes", "confirm", "true", "haan", "ha", "ok", "okay", "theek hai"):
            return "say yes to place it, or no to drop it"
        fp = d.get("cart_fp")
        if not fp:
            return "there is no cart to confirm yet; wait for the task update"
        # Who confirms matters: an elder may not confirm past the budget, whoever asked for the order.
        limit = d.get("order_limit") or Limits.from_dict(d.get("limits")).budget
        total = rupees((task.result or {}).get("total"))
        if by_is_elder and total and total > limit:
            return f"total ₹{total:.0f} is above the ₹{limit} limit: a caregiver must confirm (ask with alert_caregiver reason approval)"
        if d.get("approval") and by_is_elder:
            return f"{'; '.join(d['approval'])} (ask with alert_caregiver reason approval)"
        task.details = {**d, "confirmed_by": by, "confirmed_total": (task.result or {}).get("total"), "confirmed_fp": fp,
                        "confirm_token": guard.confirm_token(str(task.id), fp, by), "place_started": None}
        task.phase, task.status = "place", "queued"
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"confirmed by {by}")
        return "confirmed; placing it now (NOT placed yet; you will get a task update when it is placed)"
    if kind == "otp":
        if not re.fullmatch(r"\d{4,8}", value):
            return "that is not a code; ask for the digits only"
        task.phase, task.status = "otp", "queued"
        task.details = {**d, "otp": value}
        note(task, "otp received")
        return "code received; continuing"
    if kind == "fee":
        if value.lower() in ("yes", "approve", "true", "haan", "ok"):
            task.details = {**d, "fee_ok": True}
            task.phase, task.status = "cancel", "queued"
            task.deadline_at = clock.now() + TASK_LIFETIME
            note(task, f"cancel fee approved by {by}")
            return "cancelling with the fee"
        task.status = "done"
        note(task, f"cancel fee declined by {by}; order stays")
        return "kept the order; not cancelled"
    if kind == "swap":
        if d.get("agent") == "pharmacy":
            return "medicines are never swapped for another; ask the caregiver or doctor which medicine to order"
        alts = guard.alternatives(d.get("alternatives"))
        low = value.lower()
        pick = next((a for a in alts if len(low) >= 3 and (low in a["name"].lower() or a["name"].lower() in low)), None)
        if not pick:
            return "that is not one of the alternatives; read them again: " + "; ".join(a["name"] for a in alts)
        items = list(d.get("items") or [])
        # Replace the requested item the alternative is closest to (by its main words); keep the rest.
        words = set(re.findall(r"[a-z]{4,}", pick["name"].lower()))
        best = max(range(len(items)), key=lambda i: len(words & set(re.findall(r"[a-z]{4,}", str(items[i].get("name", "")).lower()))), default=None)
        if best is not None and (len(items) == 1 or words & set(re.findall(r"[a-z]{4,}", str(items[best].get("name", "")).lower()))):
            new_items = items[:best] + [{"name": pick["name"], "qty": guard.qty(items[best].get("qty"))}] + items[best + 1:]
        else:
            new_items = items + [{"name": pick["name"], "qty": 1}]
        v = guard.check_request("order", d.get("agent") or "shopping", new_items, Limits.from_dict(d.get("limits")))
        if not v.ok:
            return "cannot get that: " + "; ".join(v.block)
        task.details = {**d, "items": new_items, "alternatives": None, "channel": None, "connector_card": None, "cart_fp": None}
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"swapped to {pick['name']} by {by}")
        return f"getting {pick['name']} instead; the new cart comes back for a confirm"
    if kind == "choice":
        if task.kind != "ride" or task.status != "awaiting_confirm":
            return "nothing to choose right now"
        options = (task.result or {}).get("options") or []
        opt = _match_option(options, value)
        if opt == "ambiguous":
            return "that could be more than one option; ask which exactly: " + "; ".join(f"{o.get('type')} {o.get('fare', '')}".strip() for o in options)
        if not opt:
            return "that is not one of the options; read them again: " + "; ".join(f"{o.get('type')} {o.get('fare', '')}".strip() for o in options)
        limits = Limits.from_dict(d.get("limits"))
        fare = rupees(opt.get("fare"))
        if by_is_elder and fare and fare > limits.budget:
            return f"the {opt.get('type')} fare ₹{fare:.0f} is above the ₹{limits.budget} limit: a caregiver must choose it (ask with alert_caregiver reason approval)"
        if d.get("approval") and by_is_elder:
            return f"{'; '.join(d['approval'])} (ask with alert_caregiver reason approval)"
        choice = f"{opt.get('type')} {opt.get('fare') or ''}".strip()
        fp = _ride_fp(d.get("cart_fp"), choice)
        task.details = {**d, "choice": choice, "confirmed_by": by, "confirmed_total": opt.get("fare"), "confirmed_fp": fp,
                        "confirm_token": guard.confirm_token(str(task.id), fp, by), "place_started": None}
        task.phase, task.status = "place", "queued"
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"choice {choice} by {by}")
        return f"booking {choice} now (NOT booked yet; you will get a task update)"
    return f"unknown input {kind}"


async def request_cancel(session: AsyncSession, agent: BrowserAgent, task: Task, *, by: str, reason: str) -> str:
    if task.status in ("failed", "cancelled"):
        return f"already {task.status}"
    if task.phase == "cancel" and task.status in LIVE:
        return "cancelling is already in progress; you will get a task update"
    task.cancel_requested = True
    note(task, f"cancel requested by {by}: {reason}")
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    if task.status == "done" and placed:
        task.phase, task.status = "cancel", "queued"
        task.deadline_at = clock.now() + TASK_LIFETIME
        return "already placed; cancelling it on the service now"
    if task.phase == "place" and task.status == "queued" and (task.details or {}).get("place_started"):
        return "placing has started; it will be cancelled right after if it went through"
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
            task.details = {**d, "approval": v.approval, "confirm_token": None,
                            "cart_fp": guard.cart_fingerprint({"items": [{"name": f"{o.get('type')} {o.get('fare')}"} for o in out["options"]], "total": None})}
            task.input_needed = "choice"
            metrics.on_milestone(task, "ready")
            extra = " ".join(v.warn + v.approval)
            return "awaiting_confirm", "Ride options are ready; tell the person the fares (and any surge) and ask which one to book." + (f" {extra}" if extra else "")
        if out.get("needs_prescription") and not guard.rx_covers(d.get("items") or [], limits.rx_on_file, cart=out.get("items")):
            return "failed", "This medicine needs a prescription upload; ask the family to upload the prescription on the dashboard, then try again."
        if not out.get("items"):
            alts = guard.alternatives(out.get("alternatives"))[:4]
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
        if out.get("price_changed") and task.kind == "ride":
            # Fares moved at booking: fetch fresh fares and let the person choose again.
            task.phase, task.input_needed = "prepare", None
            task.details = {**d, "confirm_token": None, "place_started": None, "choice": None, "cart_fp": None}
            return "queued", f"The fare changed at booking ({out.get('new_total') or 'new fare'}); nothing was booked. Getting fresh fares to choose again."
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
    """The confirm still matches what is in the cart (or the fares shown) and its token is genuine."""
    d = task.details or {}
    expected = d.get("cart_fp") if task.kind == "order" else _ride_fp(d.get("cart_fp"), d.get("choice") or "")
    return bool(expected) and d.get("confirmed_fp") == expected and guard.token_valid(
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
                task.details = {**(task.details or {}), "cart_fp": None, "confirm_token": None, "place_started": None}
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


def route(urls: list[str], limit: int = 8) -> str:
    """'https://x.com/search?q=atta', … -> '/ → /search → /cart → /checkout' (paths only, repeats removed)."""
    from urllib.parse import urlparse

    out: list[str] = []
    for u in urls:
        p = urlparse(u).path or "/"
        p = re.sub(r"/\d{3,}|/[0-9a-f]{8,}(-[0-9a-f]{4,})*", "/…", p)[:40]
        if not out or out[-1] != p:
            out.append(p)
    return " → ".join(out[:limit]) + (" → …" if len(out) > limit else "")


async def release_sessions(sessions: async_sessionmaker, agent: BrowserAgent, limit: int = 50) -> int:
    """Stop cloud browsers nobody is using: tasks that ended, or that wait on a person for more than a few minutes.
    (A finished Browser Use run keeps its browser running and billing; the family's profile keeps the login.)"""
    idle_before = clock.now() - IDLE_CLOSE
    async with sessions() as session:
        rows = list((await session.execute(
            select(Task).where(Task.agent_session.is_not(None), or_(
                Task.status.in_(("done", "failed", "cancelled")),
                (Task.status.in_(("awaiting_confirm", "needs_input"))) & (Task.updated_at < idle_before),
            )).limit(limit).with_for_update(skip_locked=True)
        )).scalars())
        released = 0
        for t in rows:
            if t.status == "needs_input" and t.input_needed == "otp":
                continue  # the login page is waiting for the code in this browser
            try:
                await agent.stop_session(t.agent_session)
                await sandbox.stopped(session, t.agent_session, f"task {t.status}")
            except Exception:  # noqa: BLE001 — already gone, or the service is down: the sweeper retries from the ledger
                logger.warning("stop session failed task=%s", t.id)
            t.agent_session = None
            released += 1
        await session.commit()
    return released


MAX_TICK_ERRORS = 3


async def _count_tick_error(session: AsyncSession, task_id) -> tuple[Task, str] | None:
    """A task whose tick keeps crashing (a report the code cannot read) is stopped after a few tries and the
    family is told, instead of being stuck 'running' and blocking that service for good."""
    try:
        task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update())).scalar_one_or_none()
        if not task or task.status in ("done", "failed", "cancelled"):
            return None
        n = int((task.details or {}).get("tick_errors", 0)) + 1
        task.details = {**(task.details or {}), "tick_errors": n}
        message = None
        if n >= MAX_TICK_ERRORS:
            task.status = "failed"
            message = ("Something went wrong while placing it. It is not clear whether it went through: do not order again; "
                       "tell the caregiver to check the app.") if task.phase in ("place", "cancel") else (
                "Something went wrong while working on it, so it was stopped; nothing was placed.")
            note(task, f"stopped after {n} errors")
            metrics.on_milestone(task, "failed")
            await store.record_event(
                session, family_id=task.family_id, subject_id=task.subject_id, kind="task_failed",
                summary=f"{SKILLS[task.service]['label']}: {message}", payload={"task_id": str(task.id)},
            )
        await session.commit()
        return (task, message) if message else None
    except Exception:  # noqa: BLE001
        logger.exception("could not count tick error task=%s", task_id)
        await session.rollback()
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
    try:
        stats["released"] = await release_sessions(sessions, agent)
    except Exception:  # noqa: BLE001 — cleanup must never stop the tick
        logger.exception("release sessions failed")
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
                elif task.status == "queued" and task.phase == "place" and (task.details or {}).get("place_started"):
                    # A place step started before and never recorded its result (a crash after the store call).
                    # It may have gone through: never place again.
                    task.status = "failed"
                    message = ("It is not clear whether it was placed (the last attempt was interrupted). Do not order again; "
                               "tell the caregiver to check the app.")
                    note(task, "place interrupted; not retried")
                    metrics.on_milestone(task, "failed")
                elif task.status == "queued":
                    if task.phase == "place":
                        # Mark before calling the store, and commit, so a crash after the call cannot place twice.
                        marker = clock.now().isoformat()
                        task.details = {**(task.details or {}), "place_started": marker}
                        await session.commit()
                        # Re-read from the database (not the identity map) under the lock: another tick or a cancel
                        # may have changed it between the commit and here.
                        task = (await session.execute(
                            select(Task).where(Task.id == task_id).with_for_update().execution_options(populate_existing=True)
                        )).scalar_one()
                        if task.status != "queued" or task.cancel_requested or (task.details or {}).get("place_started") != marker:
                            continue
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
                        if status == "done" and task.cancel_requested:
                            task.phase, task.status = "cancel", "queued"
                            task.deadline_at = clock.now() + TASK_LIFETIME
                            message = "It was placed just before the cancel; cancelling it on the service now."
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                    elif float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0)) >= task_max_cost():
                        # Too many browser steps already: stop rather than keep paying for a task that is not working.
                        task.status = "failed"
                        message = ("This took too many tries on the website, so it was stopped to keep costs down; nothing was placed. "
                                   "Offer to try another store, or the family can order in the app.") if task.phase != "place" else (
                            "This took too many tries and was stopped. It is not clear whether it was placed: do not order again; check the app.")
                        note(task, f"cost ceiling ₹{task_max_cost():.0f} reached")
                        metrics.on_milestone(task, "failed")
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
                        before = float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0))
                        metrics.on_browser_run(task, run.steps)
                        sandbox.audit(task, phase=task.phase, status=run.status, steps=run.steps, path=run.path,
                                      cost=float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0)) - before, error=run.error)
                        await sandbox.note_login(session, task.family_id, task.service, run.output or {})
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
                            if task.phase == "place":
                                out["unclear"] = True  # the run may have clicked place before it died
                        task.result = {**(task.result or {}), **{k: v for k, v in out.items() if v not in (None, "", [])}}
                        status, message = _outcome(task, out)
                        task.status = status
                        note(task, f"{task.phase} → {status}: {message}")
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                        if status == "done" and task.cancel_requested:
                            task.phase, task.status = "cancel", "queued"
                            task.deadline_at = clock.now() + TASK_LIFETIME
                            message = "It was placed just before the cancel; cancelling it on the service now."
                        from app.care import skillbook

                        used = list((task.details or {}).get("skills_used") or [])
                        if status in ("awaiting_confirm", "done") and task.phase != "cancel":
                            # The path that worked becomes (or reinforces) a store skill the next agent starts with.
                            await skillbook.record_store_success(
                                session, task.service, task.phase, route(run.path or [], limit=10).split(" → ") if run.path else [],
                                task_id=str(task.id), n_steps=run.steps, used=used,
                            )
                        elif status == "failed" and not out.get("blocked") and not out.get("needs_prescription"):
                            await skillbook.record_store_use(session, used, ok=False)
                        if status == "failed" and out.get("problem"):
                            # The next agent on this service reads what stopped this one (once per distinct problem).
                            text_ = f"{task.phase} failed: {str(out['problem'])[:240]}"
                            if text_ not in await _learned(session, task.service):
                                session.add(SkillNote(service=task.service, note=text_, created_at=clock.now()))
                    elif clock.now() - datetime.fromisoformat(task.details["run_started"]) > RUN_TIMEOUT:
                        await agent.stop(task.agent_task)
                        sandbox.audit(task, phase=task.phase, status="timeout", steps=run.steps, path=run.path, cost=0)
                        task.status = "failed"
                        message = {
                            "cancel": "Cancel took too long; tell the caregiver.",
                            "place": "Placing took too long and was stopped. It is not clear whether it went through: do not order again; tell the caregiver to check the app.",
                        }.get(task.phase, "The service took too long and the task was stopped; nothing was placed.")
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
                got = await _count_tick_error(session, task_id)
                if not got:
                    continue
                task, message = got
        if message:
            await notify(task.family_id, task.requested_by, f"[Task update] {describe(task)}. {message}")
    return stats
