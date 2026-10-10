"""Delivery tracking after an order is placed: the store's own status for the order, read by a fast model (stores word it
in their own way, in any language), told to the person once when it is on the way and once when it was delivered.
Delivered moves the order's follow-up on to "how was it?" (app.tasks.aftercare); a store cancellation is told plainly.

Where the status comes from:
- Instamart, Swiggy: the linked account's order list on the store's connector (free), every few minutes.
- Blinkit: its order list in the family's logged-in browser, a few short looks around the delivery time (each one
  opens the cloud browser for seconds).
- Other stores: none yet; the follow-up asks the person after the delivery time, as before.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import store
from app.core import clock
from app.llm import router
from app.tasks import aftercare, fastpath
from app.tasks.models import Task

logger = logging.getLogger(__name__)

CONNECTOR = ("instamart", "swiggy")
BROWSER = ("blinkit",)
TRACK_FOR = timedelta(hours=3)
CONNECTOR_EVERY = timedelta(minutes=4)
BROWSER_LOOKS = 3  # Blinkit: each look is a short logged-in browser session
NOT_FOUND_LIMIT = 4  # the order is not in the list (another account, or no connector link): stop asking

PROMPT = """You read an Indian delivery store's status for one order: the store's own words and fields, in any language.
Return JSON only: {"state": "preparing" | "on_the_way" | "delivered" | "cancelled" | "unknown", "minutes_left": <number or null>}
- delivered: the store says it was delivered or handed over.
- cancelled: the store cancelled it, it failed, or it was refunded instead of delivered.
- on_the_way: a delivery partner has picked it up or is heading to the address.
- preparing: placed and being packed or prepared; nobody is on the way yet.
- unknown: the status does not say.
- minutes_left: whole minutes until it arrives, only when the store gives a time still to come ("arriving in 8 mins",
  or a promised clock time compared with NOW); otherwise null. Never guess one."""


def trackable(task: Task) -> bool:
    r = task.result or {}
    return task.kind == "order" and bool(r.get("placed") and r.get("order_id")) and task.service in CONNECTOR + BROWSER


def _eta_minutes(text) -> int | None:
    """The store's delivery time as given at placing ("12 min", "10-15 min"): the later number, for when to look first."""
    nums = [int(n) for n in re.findall(r"\d{1,3}", str(text or ""))]
    return max(nums) if nums and "min" in str(text).lower() else None


def start(task: Task) -> None:
    """Called when the order is placed: when to look first."""
    if not trackable(task) or task.family_id.startswith("shadow:"):
        return
    now = clock.now()
    if task.service in BROWSER:
        first = timedelta(minutes=max(5, (_eta_minutes((task.result or {}).get("eta")) or 12) - 3))
    else:
        first = CONNECTOR_EVERY
    task.details = {**(task.details or {}), "tracking": {"placed_at": now.isoformat(), "next_at": (now + first).isoformat(),
                                                         "looks": 0, "missing": 0, "told": [], "state": None}}


async def judge(status: dict, now: datetime) -> dict:
    """The store's status → {"state", "minutes_left"}; unknown when the model could not answer."""
    text = f"NOW: {clock.ist(now):%d %b %Y %I:%M %p} IST\nSTATUS: {json.dumps(status, ensure_ascii=False)[:1500]}"
    try:
        reply = await router.complete("classify", system_stable=PROMPT, effort="low", max_tokens=600, timeout_s=25,
                                      messages=[{"role": "user", "content": [{"type": "text", "text": text}]}])
        from app.care.extract import parse_json

        data = parse_json(reply.text)
    except Exception as exc:  # noqa: BLE001 — looked again next time
        logger.warning("tracking judge failed: %s", exc)
        return {"state": "unknown", "minutes_left": None}
    state = data.get("state") if data.get("state") in ("preparing", "on_the_way", "delivered", "cancelled") else "unknown"
    mins = data.get("minutes_left")
    return {"state": state, "minutes_left": int(mins) if isinstance(mins, (int, float)) and 0 <= mins <= 600 else None}


async def _status(task: Task, host, agent, profile_for) -> dict | None:
    """The store's own status for this order, or None when it could not be read (or the order is not in the list)."""
    oid = str((task.result or {}).get("order_id"))
    if task.service in CONNECTOR:
        if host is None:
            return None
        got = await host.call("connector_order_status", {"store": task.service, "order_id": oid},
                              family_id=task.family_id, subject_id=task.subject_id, actor_id=task.requested_by)
        return {k: v for k, v in got.items() if k != "found"} if isinstance(got, dict) and got.get("found") else None
    if not hasattr(agent, "open_session"):
        return None
    got = await profile_for(task)
    profile_id = got.get("profileId") if isinstance(got, dict) else got
    sid = await agent.open_session(profile_id)
    try:
        return await fastpath.blinkit_order_status(await agent.cdp_url(sid), oid)
    finally:
        try:
            await agent.stop_session(sid)
        except Exception:  # noqa: BLE001 — the sweeper stops it
            logger.warning("tracking: stop session failed")


def _say(task: Task, label: str, state: str, minutes: int | None, words: str, ask_arrival: bool = False) -> str:
    what = f"{label} order ({task.goal[:80]})"
    if state == "on_the_way":
        when = f", arriving in about {minutes} min" if minutes else ""
        return f"[Task update] {what}: the store says it is on the way{when}. Tell them in one short line; ask nothing."
    if state == "delivered":
        if ask_arrival:
            return (f"[Task update] {what}: the store says it was delivered. Tell them in one short message and ask whether they got "
                    f"it and everything is all right (one question). Their answer goes into order_feedback with task_id {task.id}.")
        return f"[Task update] {what}: the store says it was delivered. Tell them in one short line; ask nothing."
    return (f"[Task update] {what}: the store cancelled it ({words[:120]}). Nothing will come and nothing is to be paid. Tell "
            "them plainly in one short line and offer to order it again.")


async def sweep(sessions: async_sessionmaker, agent, notify, *, profile_for=None, host_for=None, limit: int = 10) -> int:
    """Look at placed orders whose next look is due; tell the person what changed. Returns how many were looked at."""
    from app.tasks.runtime import note
    from app.tasks.skills import SKILLS

    now = clock.now()
    async with sessions() as session:
        rows = list((await session.execute(
            select(Task).where(Task.kind == "order", Task.status == "done", Task.service.in_(CONNECTOR + BROWSER),
                               Task.updated_at >= now - TRACK_FOR - timedelta(hours=1))
            .order_by(Task.updated_at).limit(200)
        )).scalars())
        due = [t.id for t in rows if (t.details or {}).get("tracking") and not t.details["tracking"].get("done")
               and datetime.fromisoformat(t.details["tracking"]["next_at"]) <= now][:limit]
    looked = 0
    for task_id in due:
        sends: list[str] = []
        async with sessions() as session:
            task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update(skip_locked=True))).scalar_one_or_none()
            if not task or not (task.details or {}).get("tracking"):
                continue
            tr = dict(task.details["tracking"])
            looked += 1
            tr["looks"] = int(tr.get("looks") or 0) + 1
            try:
                status = await _status(task, host_for(task.family_id) if host_for else None, agent, profile_for)
            except Exception as exc:  # noqa: BLE001 — looked again next time
                logger.warning("tracking look failed task=%s: %s", task.id, exc)
                status = None
            seen = await judge(status, now) if status else {"state": "unknown", "minutes_left": None}
            if status is None:
                tr["missing"] = int(tr.get("missing") or 0) + 1
            label = SKILLS[task.service]["label"]
            state, told = seen["state"], list(tr.get("told") or [])
            # delivered: the follow-up moves on to "how was it?"; the person it is for is asked whether they got it
            ask_arrival = await aftercare.store_delivered(session, task, now) if state == "delivered" else False
            if state in ("delivered", "cancelled") or (state == "on_the_way" and "on_the_way" not in told):
                words = str((status or {}).get("currentStatus") or (status or {}).get("status") or (status or {}).get("text") or "")
                sends.append(_say(task, label, state, seen["minutes_left"], words, ask_arrival=ask_arrival))
                told.append(state)
            tr.update(state=state if state != "unknown" else tr.get("state"), told=told, last_at=now.isoformat())
            placed_at = datetime.fromisoformat(tr["placed_at"])
            finished = state in ("delivered", "cancelled")
            gave_up = (now - placed_at >= TRACK_FOR or tr["missing"] >= NOT_FOUND_LIMIT
                       or (task.service in BROWSER and tr["looks"] >= BROWSER_LOOKS))
            if finished or gave_up:
                tr["done"] = True
            else:
                step = CONNECTOR_EVERY if task.service in CONNECTOR else timedelta(minutes=10 if state != "on_the_way" else 8)
                if seen["minutes_left"] and task.service in BROWSER:
                    step = timedelta(minutes=max(3, seen["minutes_left"] + 2))
                tr["next_at"] = (now + step).isoformat()
            result = dict(task.result or {})
            if finished:
                result.update({"delivered": state == "delivered", "store_cancelled": state == "cancelled", "tracked_at": now.isoformat()})
                loop = await aftercare.loop_for(session, task) if state == "cancelled" else None
                if loop:
                    await store.close_loop(session, loop.id, note="the store cancelled it")
                await store.record_event(session, family_id=task.family_id, subject_id=task.subject_id,
                                         kind="task_delivered" if state == "delivered" else "task_store_cancelled",
                                         summary=f"{label}: {'delivered' if state == 'delivered' else 'cancelled by the store'}",
                                         payload={"task_id": str(task.id), "order_id": result.get("order_id")})
                note(task, f"store says {state}")
                if state == "cancelled":
                    task.status = "cancelled"
            elif seen["minutes_left"]:
                result["eta_now"] = f"{seen['minutes_left']} min"
            task.result = result
            task.details = {**task.details, "tracking": tr}
            await session.commit()
        for text in sends:
            try:
                await notify(task.family_id, task.requested_by, text)
            except Exception:  # noqa: BLE001
                logger.exception("tracking notify failed task=%s", task_id)
    return looked
