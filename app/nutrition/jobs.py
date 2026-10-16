"""The nutrition agent's work queue (plan §N1.3): one job per meal being logged.

    queued → resolving → needs_input → resolving → saved | dropped

- The reply waits for one catalogue hit or one estimate (about 12 s: ESTIMATE_BUDGET). A miss is saved as pending, never
  as 0 kcal, and the sweep fills the numbers after.
- A question waits 2 hours. Then food they said they ate is saved as "unquantified" (counted, no nutrients); a question
  about an order's food (did you eat it?) saves nothing.
- The sweep runs from /v2/jobs/wake (every 5 minutes) and /v2/jobs/nutrition.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock, trace
from app.nutrition.models import NutritionJob

logger = logging.getLogger(__name__)

ANSWER_WAIT = timedelta(hours=2)
SAME_MEAL = timedelta(hours=4)  # a repeat log_food for the same meal attaches to the job already pending or saved
LIVE = ("queued", "resolving", "needs_input")
COUNTED = ("quantity", "which", "label", "photo")  # at most 2 of these per meal; consent and identity never count
MAX_QUESTIONS = 2


def inline_seconds() -> float:
    """How long log_food may spend before the reply. The brain has already used part of the turn; past this the meal
    finishes in the background and that save is the one line they hear."""
    return float(os.getenv("NUTRITION_INLINE_S", "16"))


async def by_source(session: AsyncSession, family_id: str, source_ref: str) -> NutritionJob | None:
    """The job this WhatsApp message already started. A retry must not log the meal again."""
    return (await session.execute(
        select(NutritionJob).where(NutritionJob.family_id == family_id, NutritionJob.source_ref == source_ref)
        .order_by(NutritionJob.created_at).limit(1)
    )).scalars().first()


async def create(session: AsyncSession, *, family_id: str, subject_id: str, requested_by: str, ask: dict, source_ref: str | None,
                 channel: str) -> NutritionJob:
    now = clock.now()
    job = NutritionJob(id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, requested_by=requested_by[:64], state="queued", ask=ask,
                       needs=None, draft={}, answers=[], questions=0, result=None, source_ref=source_ref, channel=channel[:12],
                       deadline_at=now + ANSWER_WAIT, trace_id=trace.current(), created_at=now, updated_at=now)
    session.add(job)
    await session.flush()
    return job


async def get(session: AsyncSession, family_id: str, job_id: str) -> NutritionJob | None:
    try:
        job = await session.get(NutritionJob, uuid.UUID(str(job_id)), with_for_update=True, populate_existing=True)
    except ValueError:
        return None
    return job if job and job.family_id == family_id else None


def _item_key(item: dict) -> tuple | None:
    """The structured item the brain passed (name, quantity, unit). Never the person's words."""
    name = " ".join(str(item.get("name") or "").lower().split())
    if not name:
        return None
    qty = item.get("quantity")
    unit = str(item.get("unit") or "").strip().lower() or None
    return (name, float(qty) if isinstance(qty, (int, float)) else None, unit)


def items_key(ask: dict) -> tuple:
    return tuple(sorted(k for k in (_item_key(i) for i in ask.get("items") or [] if isinstance(i, dict)) if k))


def _minute(ask: dict) -> str:
    return str(ask.get("eaten_at") or "")[:16]


async def same_recent(session: AsyncSession, family_id: str, subject_id: str, requested_by: str, ask: dict) -> NutritionJob | None:
    """A log_food for the same person, meal, time and items as a job pending or saved in the last few hours.

    The brain sometimes calls log_food again for a meal it already sent (a 'that's good' while the look-up is still
    running). Attaching to that job saves nothing a second time. A different speaker, or a time they named that is not
    the same minute, is a different meal.
    """
    want = items_key(ask)
    if not want:
        return None
    meal = str(ask.get("meal") or "").strip().lower()
    since = clock.now() - SAME_MEAL
    rows = list((await session.execute(select(NutritionJob).where(
        NutritionJob.family_id == family_id, NutritionJob.subject_id == subject_id,
        NutritionJob.requested_by == requested_by[:64],
        NutritionJob.state.in_(("queued", "resolving", "needs_input", "saved")),
        NutritionJob.created_at >= since).order_by(NutritionJob.created_at.desc()).limit(12))).scalars())
    for job in rows:
        got = job.ask or {}
        other = str(got.get("meal") or "").strip().lower()
        if meal and other and meal != other:
            continue
        a, b = _minute(ask), _minute(got)
        if a and b and a != b:
            continue
        if items_key(got) == want:
            return job
    return None


async def open_for(session: AsyncSession, family_id: str, subject_ids: list[str]) -> list[NutritionJob]:
    """Questions still waiting for these people (newest first)."""
    return list((await session.execute(select(NutritionJob).where(
        NutritionJob.family_id == family_id, NutritionJob.subject_id.in_(subject_ids), NutritionJob.state.in_(("needs_input", "resolving")),
        NutritionJob.deadline_at > clock.now()).order_by(NutritionJob.created_at.desc()).limit(6))).scalars())


def ask_needs(job: NutritionJob, reason: str, question: str, options: list[str] | None = None, item: str | None = None) -> dict:
    job.state = "needs_input"
    needs = {"reason": reason, "question": question, "options": list(options or [])[:6], **({"item": item} if item else {}),
             "at": clock.now().isoformat()}
    job.needs = needs
    if reason in COUNTED:
        job.questions = (job.questions or 0) + 1
    job.deadline_at = clock.now() + ANSWER_WAIT
    job.updated_at = clock.now()
    return needs


def can_ask(job: NutritionJob) -> bool:
    return (job.questions or 0) < MAX_QUESTIONS


def finish(job: NutritionJob, state: str, result: dict) -> None:
    job.state, job.result, job.needs, job.updated_at = state, result, None, clock.now()


async def claim(session: AsyncSession, job: NutritionJob) -> bool:
    """One saver for this meal. The loser does not save again and does not send a second line."""
    if job.state == "saved":
        return False
    res = await session.execute(update(NutritionJob).where(
        NutritionJob.id == job.id, NutritionJob.state.in_(("queued", "resolving", "needs_input"))
    ).values(state="saving", updated_at=clock.now()))
    if not getattr(res, "rowcount", 0):
        return False
    job.state = "saving"
    return True


def context_block(jobs: list[NutritionJob], names: dict) -> str:
    """For the brain's turn context: food questions waiting for an answer."""
    rows = [j for j in jobs if j.state == "needs_input" and j.needs]
    if not rows:
        return ""
    lines = ["FOOD LOG QUESTIONS WAITING (the nutrition agent asked; when they answer, call log_food with answer_to = the job id "
             "and the answer as fields; if they say never mind or don't know what they ate, answer_to with cancel=true):"]
    for j in rows[:4]:
        n = j.needs or {}
        opts = f" (options: {', '.join(n.get('options') or [])})" if n.get("options") else ""
        lines.append(f"  - job {j.id} about {names.get(j.subject_id, 'them')}: {n.get('reason')}: \"{n.get('question')}\"{opts}")
    return "\n".join(lines)


# ── background ───────────────────────────────────────────────────────────────

_kicked: asyncio.Task | None = None
_refine: asyncio.Task | None = None
_fill: asyncio.Task | None = None


def kick_nutrition() -> None:
    """Finish slow look-ups now in the background (a turn just left one). One run at a time."""
    global _kicked
    if _kicked is not None and not _kicked.done():
        return
    import contextvars

    from app.db.session import SessionLocal

    ctx = contextvars.copy_context()
    ctx.run(trace.detach)
    try:
        _kicked = asyncio.get_running_loop().create_task(sweep(SessionLocal, None), context=ctx)
    except RuntimeError:
        logger.warning("no running loop for the nutrition sweep; the wake job picks it up")


def kick_refine(job_id) -> None:
    """A web look-up replaces an estimate after the person has already been told the first numbers. No second message."""
    global _refine
    if _refine is not None and not _refine.done():
        return

    from app.db.session import SessionLocal

    async def _run() -> None:
        from app.nutrition import agent

        try:
            await agent.refine_saved(SessionLocal, job_id)
        except Exception:  # noqa: BLE001 — the estimate they were told stays until the next sweep
            logger.exception("nutrition refine failed id=%s", job_id)

    try:
        _refine = asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        logger.warning("no running loop for the nutrition refine")


def kick_fill() -> None:
    """Fill numbers for food saved without them. One run at a time; the sweep retries whatever is left."""
    global _fill
    if _fill is not None and not _fill.done():
        return

    from app.db.session import SessionLocal

    async def _run() -> None:
        from app.nutrition import agent

        try:
            async with SessionLocal() as session:
                await agent.fill_missing(session)
                await session.commit()
        except Exception:  # noqa: BLE001 — the next sweep fills the same rows
            logger.exception("nutrition fill failed")

    try:
        _fill = asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        logger.warning("no running loop for the nutrition fill")


async def sweep(sessions, host_for=None) -> dict:
    """Slow look-ups still resolving, and questions past their 2 hours."""
    from app.nutrition import agent

    stats = {"resolved": 0, "expired": 0, "failed": 0}
    async with sessions() as session:
        ids = list((await session.execute(select(NutritionJob.id).where(or_(
            NutritionJob.state.in_(LIVE), NutritionJob.state == "saving")).limit(100))).scalars())
    for jid in ids:
        try:
            async with sessions() as session:
                job = await session.get(NutritionJob, jid, with_for_update=True, populate_existing=True)
                if job is None or job.state not in (*LIVE, "saving"):
                    continue
                if job.state == "saving":
                    if job.updated_at > clock.now() - timedelta(seconds=60):
                        continue  # a turn is in the middle of saving this meal
                    job.state = "resolving"
                if job.state == "needs_input":
                    if job.deadline_at > clock.now():
                        continue
                    out = await agent.expire(session, job)
                    await session.commit()
                    stats["expired"] += 1
                    continue
                if job.created_at < clock.now() - timedelta(hours=2):
                    # a look-up that never finished: kept the way an unanswered question is (never retried forever)
                    out = await agent.expire(session, job)
                    await session.commit()
                    stats["expired"] += 1
                    continue
                with trace.use(job.trace_id or trace.new("nutrition"), family_id=job.family_id):
                    out = await agent.run_background(session, job)
                await session.commit()
            stats["resolved"] += 1
            # A save made in the person's turn already has its line in that reply. Only a background save, or a question, is told.
            if out and (out.get("notice") == "pending" or out.get("result") == "needs_input"):
                await agent.tell(sessions, host_for, job, out)
            if out and out.get("refine"):
                await agent.refine_saved(sessions, job.id)
        except Exception:  # noqa: BLE001 — one job must not stop the others
            logger.exception("nutrition job failed id=%s", jid)
            stats["failed"] += 1
    await _tell_pending(sessions, host_for)
    try:
        from app.nutrition import entries

        async with sessions() as session:
            moved = await entries.reslot_nights(session)
            await session.commit()
        stats["reslot"] = moved["changed"]
        if moved["changed"]:
            logger.info("nutrition night slots moved=%s kept=%s", moved["changed"], moved["kept"])
    except Exception:  # noqa: BLE001 — a re-slot must not stop the rest of the sweep
        logger.exception("nutrition night reslot failed")
    try:
        async with sessions() as session:
            filled = await agent.fill_missing(session)
            await session.commit()
        stats["filled"] = filled["filled"]
        stats["still"] = filled["still"]
    except Exception:  # noqa: BLE001 — a fill must not stop the rest of the sweep
        logger.exception("nutrition fill failed")
    return stats


async def _tell_pending(sessions, host_for) -> None:
    """A background save whose line did not go out. One more try; a sent line is not sent again."""
    from app.nutrition import agent

    since = clock.now() - timedelta(hours=2)
    async with sessions() as session:
        ids = list((await session.execute(select(NutritionJob.id).where(
            NutritionJob.state == "saved", NutritionJob.updated_at > since,
            or_(NutritionJob.result["notice"].astext == "pending",
                NutritionJob.result["notice"].astext == "sending")).limit(50))).scalars())
    for jid in ids:
        try:
            async with sessions() as session:
                job = await session.get(NutritionJob, jid)
                if job is None:
                    continue
                result = job.result or {}
                if result.get("notice") == "sending" and job.updated_at > clock.now() - timedelta(seconds=60):
                    continue
                out = {"result": "saved", "notice": "pending", "say": result.get("say") or "", "today": result.get("today") or ""}
            if out["say"]:
                await agent.tell(sessions, host_for, job, out)
        except Exception:  # noqa: BLE001
            logger.exception("nutrition notice failed id=%s", jid)
