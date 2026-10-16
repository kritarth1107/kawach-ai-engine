"""How a food log enters the agent, and what the sweep does after the turn.

The meal rules stay in agent.py. These names are re-exported there, so callers keep using app.nutrition.agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.core import clock
from app.nutrition import entries, jobs, lookup, nudge_hooks, photos
from app.nutrition import nutrients as N
from app.nutrition.models import NutritionEntry, NutritionFood, NutritionJob

if TYPE_CHECKING:
    from app.nutrition.agent import Who

logger = logging.getLogger("app.nutrition.agent")


def _attach(job: NutritionJob) -> dict:
    """A repeat log_food for a meal already pending or saved: no new job and no new entries."""
    if job.state in ("queued", "resolving"):
        return {"result": "looking_up", "already": True, "job_id": str(job.id), "say": "",
                "how": "This meal is already being saved. Say nothing. Do not say you are checking, and do not call log_food again for it."}
    if job.state == "needs_input":
        n = job.needs or {}
        return {"result": "needs_input", "already": True, "job_id": str(job.id), "needs": [n], "say": n.get("question") or "",
                "how": "This question is already waiting. Ask it only if you have not asked it yet. Pass the answer with answer_to."}
    return {"result": "saved", "already": True, "job_id": str(job.id), "say": "",
            "how": "Already in the food log. Say nothing about it."}


async def handle(session: AsyncSession, who: Who, args: dict, *, subject_id: str) -> dict:
    """log_food: a new meal, or an answer to the agent's question (answer_to)."""
    from app.nutrition.agent import NutritionAgent, clean_ask, merge

    if args.get("make_usual") and not any(args.get(k) for k in ("items", "from_photo", "label_photo", "skipped", "from_order_task")):
        return await nudge_hooks.note_stated_usual(session, who.family_id, subject_id)
    if args.get("answer_to"):
        job = await jobs.get(session, who.family_id, str(args["answer_to"]))
        if job is None or job.state not in jobs.LIVE:
            return {"result": "closed", "why": "That question is closed (saved, dropped or over 2 hours old). If they are telling you what "
                                               "they ate, call log_food again without answer_to."}
        merge(job, args)
        if args.get("for_whom"):
            job.subject_id = subject_id
            ask = dict(job.ask)
            ask.pop("for_whom", None)
            job.ask = ask
    else:
        ask = clean_ask(args)
        if who.message_ref:
            prior = await jobs.by_source(session, who.family_id, who.message_ref)
            if prior is not None:
                return _attach(prior)
        job = None
        if ask.get("items") and not any(ask.get(k) for k in ("from_photo", "label_photo", "skipped", "from_order_task")):
            existing = await jobs.same_recent(session, who.family_id, subject_id, who.speaker_id, ask)
            if existing is not None and existing.state == "needs_input":
                # An answer that arrived as a new call still closes the open question. It is not asked again.
                merge(existing, args)
                if args.get("for_whom"):
                    existing.subject_id = subject_id
                    kept = dict(existing.ask)
                    kept.pop("for_whom", None)
                    existing.ask = kept
                job = existing
            elif existing is not None:
                return _attach(existing)
        if job is None:
            job = await jobs.create(session, family_id=who.family_id, subject_id=subject_id, requested_by=who.speaker_id, ask=ask,
                                    source_ref=who.message_ref, channel=who.channel)
    out = await NutritionAgent(session, who).run(job, deadline=time.monotonic() + jobs.inline_seconds())
    await session.flush()
    if out.get("result") == "looking_up":
        jobs.kick_nutrition()
    elif out.get("refine"):
        jobs.kick_refine(job.id)
    if any(s.get("confidence") == "pending" for s in out.get("saved") or []):
        jobs.kick_fill()
    return out


async def run_background(session: AsyncSession, job: NutritionJob) -> dict:
    """A slow look-up finishing after the turn (no photo bytes here: a photo was read in the turn)."""
    from app.nutrition.agent import NutritionAgent, Who

    roster = await store.roster(session, job.family_id)
    names = {m.get("id"): m.get("name") for m in ([roster.elder, *roster.members] if roster else []) if m.get("id")}
    who = Who(family_id=job.family_id, speaker_id=job.requested_by, channel=job.channel or "whatsapp", essential=False, names=names)
    return await NutritionAgent(session, who).run(job, deadline=time.monotonic() + 20, background=True)


async def expire(session: AsyncSession, job: NutritionJob) -> dict:
    """2 hours without an answer (plan §N1.3): food they said they ate is kept as unquantified; an order's food, a photo
    nobody confirmed, or "who ate it?" saves nothing."""
    from app.nutrition.agent import eaten_time

    ask = job.ask or {}
    reason = (job.needs or {}).get("reason")
    items = [i for i in ask.get("items") or [] if i.get("name")]
    if reason in ("consumed", "who_ate", "confirm") or not items:
        if (job.draft or {}).get("photo_ref"):
            await photos.delete((job.draft or {})["photo_ref"])
        jobs.finish(job, "dropped", {"why": f"no answer ({reason})"})
        return {"result": "dropped"}
    at = eaten_time(ask, now=job.created_at)
    slot = await entries.assign_slot(session, job.family_id, job.subject_id, at, named=ask.get("meal") if isinstance(ask.get("meal"), str) else None)
    meal = slot.meal
    rows = await entries.save_meal(
        session, family_id=job.family_id, subject_id=job.subject_id, logged_by=job.requested_by,
        items=[entries.Item(name=i["name"], food=None, quantity=i.get("quantity"), unit=i.get("unit"), name_said=i.get("said") or i["name"],
                            confidence="unquantified") for i in items],
        meal=meal, eaten_at=at, day=slot.day, source="dashboard" if job.channel == "dashboard" else ("caregiver" if job.requested_by != job.subject_id else "chat"),
        source_ref=job.source_ref, job_id=job.id, reason="no answer in 2 hours: kept without numbers")
    jobs.finish(job, "saved", {"entries": [str(r.id) for r in rows], "unquantified": True})
    return {"result": "saved", "unquantified": True}


async def _mark_notice(sessions, job_id, notice: str) -> None:
    async with sessions() as session:
        row = await session.get(NutritionJob, job_id)
        if row is None:
            return
        row.result = {**(row.result or {}), "notice": notice}
        row.updated_at = clock.now()
        await session.commit()


async def tell(sessions, host_for, job: NutritionJob, out: dict) -> None:
    """Tell them what a background look-up found. A save goes out as that one line, once. A question still uses a system turn."""
    from app.brain.host import LiveHost, ShadowHost
    from app.brain.wake import _send, in_their_words, system_turn
    from app.care import store

    if out.get("result") != "saved" or out.get("notice") != "pending":
        if out.get("result") != "needs_input":
            return
        host = host_for(job.family_id) if host_for else (ShadowHost() if str(job.family_id).startswith("shadow:") else LiveHost())
        n = (out.get("needs") or [{}])[0]
        prompt = (f"[Nutrition update] To save the food {job.requested_by} told you about, ask them only this, with send_message (to "
                  f"{job.requested_by}), in their language: \"{n.get('question')}\". Do not read out options or ids. When they answer, "
                  f"call log_food with answer_to=\"{job.id}\". Then reply none.")
        try:
            await system_turn(sessions, host, job.family_id, prompt, f"nutrition:{job.id}:needs_input", deliver_to=job.requested_by,
                              fallback=str(n.get("question") or ""))
        except Exception:  # noqa: BLE001
            logger.exception("nutrition question not told family=%s", job.family_id)
        return
    line = f"{out.get('say') or ''} {out.get('today') or ''}".strip()
    if not line:
        return
    async with sessions() as session:
        row = await session.get(NutritionJob, job.id, with_for_update=True)
        notice = (row.result or {}).get("notice") if row else None
        if row is None or notice not in ("pending", "sending"):
            return
        if notice == "sending" and row.updated_at > clock.now() - timedelta(seconds=60):
            return
        family_id, person = row.family_id, row.requested_by
        row.result = {**(row.result or {}), "notice": "sending"}
        row.updated_at = clock.now()
        await session.commit()
    host = host_for(family_id) if host_for else (ShadowHost() if str(family_id).startswith("shadow:") else LiveHost())
    try:
        words = await in_their_words(sessions, family_id, person, line)
        async with sessions() as session:
            roster = await store.roster(session, family_id)
        known = {m.get("id") for m in ([roster.elder, *roster.members] if roster else [])}
        ok = bool(roster) and person in known and await _send(
            sessions, host, roster, family_id, person, words, f"nutrition:{job.id}:saved")
    except Exception:  # noqa: BLE001 — the next sweep sends the same line
        logger.exception("nutrition update not told family=%s", family_id)
        ok = False
    await _mark_notice(sessions, job.id, "sent" if ok else "pending")


async def refine_saved(sessions, job_id) -> None:
    """After they have been told, replace an estimate with a web look-up when one comes back. No second message."""
    async with sessions() as session:
        rows = list((await session.execute(select(NutritionEntry).where(
            NutritionEntry.job_id == job_id, NutritionEntry.status == "active"))).scalars())
        pending = []
        for e in rows:
            if not (e.flags or {}).get("refine") or e.food_id is None:
                continue
            food = await session.get(NutritionFood, e.food_id)
            if food is not None:
                pending.append((e, food))

        async def _look(entry, food):
            try:
                return await lookup.web({"name": entry.name_said or entry.name}, food.name, timeout_s=lookup.WEB_S)
            except lookup.Slow:
                return None

        looked = await asyncio.gather(*(_look(e, food) for e, food in pending))
        changed = False
        for (e, food), got in zip(pending, looked):
            if not got or not N.check_per_100g(got.get("per_100g") or {}).ok:
                continue
            food.per_100g = N.clean(got["per_100g"])
            food.source = str(got.get("source") or "web")[:12]
            food.source_urls = list(got.get("source_urls") or [])[:4]
            if food.source in ("usda", "ifct", "web", "off") and food.kind != "recipe":
                food.family_id = None  # a published value is shared; the estimate was only this family's until then
            chk = N.check_per_100g(food.per_100g)
            food.checked = {**chk.to_dict(), "rough": False}
            food.confidence = 0.8
            food.updated_at = clock.now()
            grams, vals, _est = entries.compute(food, e.quantity, e.unit)
            if not vals:
                continue
            e.nutrients, e.grams = vals, grams
            flags = dict(e.flags or {})
            flags.pop("refine", None)
            flags.pop("rough", None)
            facts = N.fact_tags(food.per_100g)
            if facts:
                flags["facts"] = facts
            else:
                flags.pop("facts", None)
            e.flags = flags
            e.version = (e.version or 1) + 1
            e.updated_at = clock.now()
            await entries.record(session, e, "write", reason="web look-up replaced the first estimate")
            changed = True
        if changed:
            await session.commit()


async def fill_missing(session: AsyncSession) -> dict:
    """Numbers for food saved without them, since 14 Oct. Catalogue, then one estimate, then the web. Never writes 0 kcal."""
    from sqlalchemy import and_, or_

    rows = list((await session.execute(select(NutritionEntry).where(
        NutritionEntry.status == "active", NutritionEntry.kind == "food",
        or_(NutritionEntry.confidence == "pending",
            and_(NutritionEntry.confidence == "unquantified", NutritionEntry.day >= "2026-10-14")),
    ).order_by(NutritionEntry.day.desc(), NutritionEntry.eaten_at.desc()).limit(12))).scalars())
    filled = still = 0
    for e in rows:
        if not entries.numbers_pending(e):
            continue
        flags = dict(e.flags or {})
        if int(flags.get("fill_tries") or 0) >= 4:
            still += 1
            continue
        item: dict = {"name": e.name_said or e.name}
        if e.quantity:
            item["quantity"] = float(e.quantity)
        if e.unit:
            item["unit"] = e.unit
        found = await lookup.resolve(session, e.family_id, [item], cache_only=True)
        food = found[0].food if found else None
        if food is None:
            try:
                raw = await lookup.web(item, None, timeout_s=lookup.WEB_S)
            except lookup.Slow:
                raw = None
            if raw and ("energy_kcal" in (raw.get("per_100g") or {}) or "water" in (raw.get("tags") or [])):
                food = await lookup.keep(session, e.family_id, item, raw)
        if food is None:
            found = await lookup.reply_lookup(session, e.family_id, [item], essential=False)
            food = found[0].food if found else None
        per = (food.per_100g or {}) if food is not None else {}
        if food is None or ("energy_kcal" not in per and "water" not in (food.tags or [])):
            flags["fill_tries"] = int(flags.get("fill_tries") or 0) + 1
            flags["numbers"] = "pending"
            e.flags, e.confidence, e.updated_at = flags, "pending", clock.now()
            still += 1
            continue
        grams, vals, est = entries.compute(food, e.quantity, e.unit)
        if not vals and "water" not in (food.tags or []):
            flags["fill_tries"] = int(flags.get("fill_tries") or 0) + 1
            flags["numbers"] = "pending"
            e.flags, e.confidence, e.updated_at = flags, "pending", clock.now()
            still += 1
            continue
        e.food_id = food.id
        e.name = (food.name or e.name)[:200]
        e.nutrients, e.grams = vals or {}, grams
        e.confidence = "estimate" if est or e.quantity is None else "said"
        flags.pop("numbers", None)
        flags.pop("fill_tries", None)
        if food.tags:
            flags["tags"] = list(food.tags)
        if food.source == "model" or (food.checked or {}).get("rough"):
            flags["refine"] = True
        e.flags = flags
        e.version = (e.version or 1) + 1
        e.updated_at = clock.now()
        await entries.record(session, e, "write", reason="numbers added")
        await entries.sync_meal_event(session, e.family_id, e.meal_id)
        filled += 1
    return {"filled": filled, "still": still, "seen": len(rows)}
