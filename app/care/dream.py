"""Saheli's night: she goes over each family's day and learns, while nobody is writing.

For every family, once a night (Cloud Scheduler calls /v2/jobs/dream every 15 minutes between 01:00 and 05:00;
each call works through the families not done yet tonight within a time budget). Families run DREAM_PARALLEL at a
time. Each family is claimed with its own lock, so a second dream task skips a family already being reviewed.
Anyone left over is first the next night. Model calls still go through the Gemini concurrency gate.

1. catch facts the day's chats left unsaved (the extractor)
2. write a short diary line per person from what was logged and said (recall finds it later)
3. relearn each person's normal: readings, rhythm, real dose times, whether reminders work (baselines)
4. look for patterns against the last two weeks and their own normal (patterns), skipping what the family
   voted down
5. tidy long memory notes and expire unanswered confirmations (consolidate)
6. record what was learned, so the dashboard can show "Saheli reviewed yesterday"

Steps 1, 2 and 5 call a cheap model and stop by themselves at the spend cap; 3 and 4 are plain counting.
Every step is isolated: one failure is logged and the rest still run. Global upkeep (old agent skill notes)
runs once a night.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import timedelta

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.care import baselines, patterns, store
from app.care.models import FamilyRoster, Turn
from app.core import clock

logger = logging.getLogger(__name__)

DIARY_SLUG = "diary"
DIARY_KEEP_LINES = 60  # older days stay findable in events and messages; the note stays small
SKILL_NOTES_KEEP = 20
DIARY_PROMPT = (
    "You write Saheli's private diary line about one person's day, for her own memory. Use ONLY the logged events and "
    "messages given. 1 to 3 short plain sentences: how they were, what happened (doses, readings, meals, mood, "
    "visits, worries), anything to follow up. No advice, no diagnosis, nothing invented. If nothing happened, write "
    "'Quiet day, nothing logged.' Plain text, English."
)


def night_of(at=None) -> str:
    """The day being reviewed: before 12:00 IST it is yesterday."""
    local = clock.ist(at)
    return (local - timedelta(days=1 if local.hour < 12 else 0)).strftime("%Y-%m-%d")


async def _done(session, family_id: str, day: str) -> bool:
    from app.care.models import CareEvent

    return bool((await session.execute(
        select(func.count()).select_from(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == "dream", CareEvent.ref == f"dream:{day}")
    )).scalar_one())


async def diary(session, family_id: str, person: dict, day: str) -> str | None:
    """One diary line for this person's day, from the ledger and their messages (cheap model, capped)."""
    from app.llm import router

    start = clock.ist().replace(year=int(day[:4]), month=int(day[5:7]), day=int(day[8:10]), hour=0, minute=0, second=0, microsecond=0)
    evs = await store.events(session, family_id, person["id"], day=day, limit=300)
    turns = list((await session.execute(
        select(Turn).where(Turn.family_id == family_id, Turn.thread_id == person["id"], Turn.at >= start, Turn.at < start + timedelta(days=1))
        .order_by(Turn.id).limit(80)
    )).scalars())
    if not evs and not turns:
        return None
    lines = [f"{clock.ist(e.at).strftime('%H:%M')} {e.kind}: {e.summary}" for e in evs if e.kind not in ("dream", "pattern", "feedback")]
    lines += [f"{clock.ist(t.at).strftime('%H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text[:200]}" for t in turns]
    try:
        out = await router.complete(
            "extract", system_stable=DIARY_PROMPT,
            messages=[{"role": "user", "content": [{"type": "text", "text": f"{person.get('name') or 'They'}, {day}:\n" + "\n".join(lines[-120:])}]}],
            max_tokens=400, effort="low",
        )
    except router.AllModelsFailed:
        return None
    text = " ".join((out.text or "").split())[:400]
    if not text:
        return None
    existing = {n.slug: n for n in await store.notes(session, family_id, [person["id"]])}
    old = existing[DIARY_SLUG].body_md.splitlines() if DIARY_SLUG in existing else []
    body = "\n".join([ln for ln in old if not ln.startswith(f"- {day}:")][-(DIARY_KEEP_LINES - 1):] + [f"- {day}: {text}"])
    await store.upsert_note(session, family_id=family_id, subject_id=person["id"], slug=DIARY_SLUG, title="Diary", body_md=body)
    return text


async def dream_family(sessions: async_sessionmaker, roster: FamilyRoster, day: str, *, with_models: bool = True) -> dict:
    fid = roster.family_id
    people = [p for p in [roster.elder or {}] + list(roster.members or []) if p.get("id")]
    seen, uniq = set(), []
    for p in people:
        if p["id"] not in seen:
            seen.add(p["id"])
            uniq.append(p)
    report: dict = {"family": fid, "day": day, "facts": 0, "diary": 0, "baselines": 0, "patterns": 0, "notes": 0, "errors": []}

    from app.core import trace

    night = f"job:night:{day}:{fid}"  # one trace per family per night: what it changed, asked and spent (D-13)

    async def step(name, fn):
        from app.care import versions

        try:
            with versions.attribution(actor_id="saheli", source="nightly", reason=f"nightly review of {day}"), trace.use(night, family_id=fid):
                async with sessions() as session:
                    result = await fn(session)
                    await session.commit()
                    return result
        except Exception as exc:  # noqa: BLE001 — one step's failure must not stop the night
            logger.exception("dream step %s failed family=%s", name, fid)
            report["errors"].append(f"{name}: {str(exc)[:120]}")
            return None

    if with_models:
        from app.care.extract import consolidate_family, extract_family

        report["facts"] = 0
        for _ in range(6):  # the day's backlog in a few batches
            got = await step("extract", lambda s: extract_family(s, fid))
            report["facts"] += (got or {}).get("facts", 0) if isinstance(got, dict) else 0
            if not (isinstance(got, dict) and got.get("more")):
                break
        for p in uniq:
            if await step(f"diary:{p['id']}", lambda s, p=p: diary(s, fid, p, day)):
                report["diary"] += 1
    from app.care import memory_index, memory_upkeep

    owner = next((m.get("id") for m in roster.members or [] if "primary" in str(m.get("role", "")).lower()), None)
    from app.care import outcomes_learn

    # what happened becomes visible, undoable learnings (best times to write, habits) before anything plans tomorrow (A5)
    learned = await step("outcomes", lambda s: outcomes_learn.run_family(s, fid, [p["id"] for p in uniq]))
    if learned:
        report["learned"] = learned.get("learned", 0)
    for p in uniq:
        base = await step(f"baseline:{p['id']}", lambda s, p=p: baselines.save(s, fid, p["id"]))
        if base is not None:
            report["baselines"] += 1
            from app.care import skillbook

            ask_to = owner if owner != p["id"] else None

            async def suggest(s, p=p, base=base, ask_to=ask_to):
                n = 0
                for out in (await skillbook.propose_from_style(s, fid, p["id"], base.get("style") or {}),
                            await skillbook.propose_from_timing(s, fid, p["id"])):
                    if out:
                        n += 1
                        await skillbook.ask_caregiver(s, fid, p["id"], out, ask_to)
                return n

            report["skills_proposed"] = report.get("skills_proposed", 0) + (await step(f"skill:{p['id']}", suggest) or 0)
        report["indexed"] = report.get("indexed", 0) + (await step(f"index:{p['id']}", lambda s, p=p: memory_index.nightly(s, fid, p["id"])) or 0)
        roll = await step(f"rollups:{p['id']}", lambda s, p=p: memory_upkeep.rollups(s, fid, p["id"], with_models=with_models)) or {}
        report["rollups"] = report.get("rollups", 0) + sum(1 for v in roll.values() if v)
        report["memory_checks"] = report.get("memory_checks", 0) + (
            await step(f"health:{p['id']}", lambda s, p=p: memory_upkeep.ask_about_health(s, fid, p["id"], owner if owner != p["id"] else None)) or 0)
        await step(f"card:{p['id']}", lambda s, p=p: memory_upkeep.save_profile_card(s, fid, p))
        new = await step(f"patterns:{p['id']}", lambda s, p=p: patterns.record_new(s, fid, p["id"]))
        report["patterns"] += len(new or [])
    if with_models:
        report["notes"] = await step("consolidate", lambda s: consolidate_family(s, fid)) or 0
        # after the rewrite, on the notes as they now stand: contradictions with the record or between notes become one
        # question each to a caregiver (D-12); nothing is changed
        for p in uniq:
            report["contradictions"] = report.get("contradictions", 0) + (await step(
                f"contradictions:{p['id']}",
                lambda s, p=p: memory_upkeep.ask_about_contradictions(s, fid, p["id"], owner if owner != p["id"] else None)) or 0)

    if with_models:
        from app.quality import nightly as quality

        if quality.enabled():
            # the daily score (trust plan C1, A2): the duplicate merge, the full-memory judge on every message of the day,
            # checks 2 and 3, the memory score and replay cases; after the night's own steps, before any brief
            try:
                with trace.use(night, family_id=fid):
                    report["quality"] = await quality.family(sessions, roster, day)
            except Exception as exc:  # noqa: BLE001 — the score must never cost the family its night
                logger.exception("daily score failed family=%s", fid)
                report["errors"].append(f"quality: {str(exc)[:120]}")

    async def mark(session):
        await store.record_event(session, family_id=fid, subject_id=(roster.elder or {}).get("id") or "family", kind="dream",
                                 summary=f"Reviewed {day}: {report['facts']} facts, {report['diary']} diary lines, {report['patterns']} new patterns",
                                 payload=report, ref=f"dream:{day}")

    await step("mark", mark)
    return report


async def upkeep(sessions: async_sessionmaker) -> dict:
    """Once a night for everyone: keep only recent skill notes per service, expire unanswered confirmations."""
    from app.care.extract import expire_stale_confirmations
    from app.tasks.models import SkillNote

    out: dict = {"skill_notes_removed": 0, "confirmations_expired": 0}
    async with sessions() as session:
        services = [r[0] for r in await session.execute(select(SkillNote.service).distinct())]
        for svc in services:
            keep = select(SkillNote.id).where(SkillNote.service == svc).order_by(SkillNote.id.desc()).limit(SKILL_NOTES_KEEP)
            res = await session.execute(delete(SkillNote).where(SkillNote.service == svc, SkillNote.id.not_in(keep)))
            out["skill_notes_removed"] += res.rowcount or 0
        out["confirmations_expired"] = await expire_stale_confirmations(session)
        from app.care import skillbook, versions

        out["skills"] = await skillbook.curate(session)
        out["versions_pruned"] = await versions.prune(session)
        from app.care import why

        out["trail_pruned"] = await why.prune(session)
        await session.commit()
    return out


def dream_parallel() -> int:
    """How many families are reviewed at once. Tests share one connection, so they set this to 1."""
    return max(1, int(os.getenv("DREAM_PARALLEL", "4")))


def vertex_batch_fits() -> bool:
    """The Vertex batch API is about half price and answers later, not in the call.

    A night step reads what the previous step just wrote (extract, then the diary, then the tidy-up), and a tick has
    to move on inside its budget. A batch that is still running when the tick ends does not fit, so the night stays
    on ordinary calls behind the Gemini concurrency gate.
    """
    return False


def _engine(sessions: async_sessionmaker):
    return sessions.kw.get("bind")


async def _try_claim(conn, key: str) -> bool:
    return bool((await conn.execute(text("SELECT pg_try_advisory_lock(hashtext(:k))"), {"k": key})).scalar())


async def _release_claim(conn, key: str) -> None:
    await conn.execute(text("SELECT pg_advisory_unlock(hashtext(:k))"), {"k": key})


async def _with_claim(sessions: async_sessionmaker, key: str, fn):
    """Hold this family's night lock on its own connection so a commit inside the night does not drop it.

    A test sessionmaker is bound to one connection. A second connection would not see the test's rows, and the same
    connection cannot run the night while a lock checkout blocks it. Tests set DREAM_PARALLEL=1 and skip the lock.
    """
    bind = _engine(sessions)
    if not isinstance(bind, AsyncEngine):
        return await fn()
    async with bind.connect() as conn:
        if not await _try_claim(conn, key):
            return None
        try:
            return await fn()
        finally:
            await _release_claim(conn, key)


async def _pending(session, day: str) -> tuple[list[FamilyRoster], int]:
    """Families not reviewed tonight, oldest previous night first (someone left over goes first)."""
    from app.care.models import CareEvent

    last = (
        select(CareEvent.family_id, func.max(CareEvent.ref).label("last_dream"))
        .where(CareEvent.kind == "dream")
        .group_by(CareEvent.family_id)
        .subquery()
    )
    done = select(CareEvent.family_id).where(CareEvent.kind == "dream", CareEvent.ref == f"dream:{day}")
    real = ~FamilyRoster.family_id.startswith("shadow:")
    rows = list((await session.execute(
        select(FamilyRoster)
        .outerjoin(last, last.c.family_id == FamilyRoster.family_id)
        .where(real, FamilyRoster.family_id.not_in(done))
        .order_by(last.c.last_dream.asc().nulls_first(), FamilyRoster.family_id)
    )).scalars())
    total = int((await session.execute(select(func.count()).select_from(FamilyRoster).where(real))).scalar_one())
    return rows, total


async def dream_all(sessions: async_sessionmaker, *, budget_s: float = 240.0, with_models: bool = True, host=None) -> dict:
    day = night_of()
    started = time.monotonic()
    async with sessions() as session:
        todo, families = await _pending(session, day)
    stats = {"day": day, "families": families, "done_now": 0, "left": 0, "patterns": 0, "facts": 0, "errors": 0,
             "workers": min(dream_parallel(), max(1, len(todo)))}
    queue: asyncio.Queue = asyncio.Queue()
    for roster in todo:
        queue.put_nowait(roster)

    async def one(roster: FamilyRoster):
        async def run():
            async with sessions() as session:
                if await _done(session, roster.family_id, day):
                    return None
            return await dream_family(sessions, roster, day, with_models=with_models)

        return await _with_claim(sessions, f"dream:{day}:{roster.family_id}", run)

    async def worker() -> None:
        while time.monotonic() - started <= budget_s:
            try:
                roster = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                rep = await one(roster)
            except Exception:  # noqa: BLE001 — one family must not stop the night
                logger.exception("dream family failed family=%s", roster.family_id)
                stats["errors"] += 1
                continue
            if not rep:
                continue
            stats["done_now"] += 1
            stats["patterns"] += rep["patterns"]
            stats["facts"] += rep["facts"]
            stats["errors"] += len(rep["errors"])

    await asyncio.gather(*(worker() for _ in range(stats["workers"])))
    async with sessions() as session:
        still, _ = await _pending(session, day)
    stats["left"] = len(still)
    if not still:
        async def finish():
            stats["upkeep"] = await upkeep(sessions)
            from app.learn import jobs as learn_jobs

            stats["learn"] = await learn_jobs.nightly(sessions, with_models=with_models)
            if with_models:
                from app.quality import nightly as quality

                if quality.enabled():
                    try:  # check 1 from the delivery ledger, each family's overall and everyone's (trust plan C1)
                        stats["quality"] = await quality.finish(sessions, day, host)
                    except Exception:  # noqa: BLE001
                        logger.exception("daily score finish failed")
            return True

        await _with_claim(sessions, f"dream-upkeep:{day}", finish)
    logger.info("dream %s", stats)
    return stats
