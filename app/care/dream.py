"""Saheli's night: she goes over each family's day and learns, while nobody is writing.

For every family, once a night (Cloud Scheduler calls /v2/jobs/dream every 15 minutes between 01:00 and 05:00;
each call works through the families not done yet tonight within a time budget):

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

import logging
import time
from datetime import timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

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

    async def step(name, fn):
        try:
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

        got = await step("extract", lambda s: extract_family(s, fid))
        report["facts"] = (got or {}).get("facts", 0) if isinstance(got, dict) else 0
        for p in uniq:
            if await step(f"diary:{p['id']}", lambda s, p=p: diary(s, fid, p, day)):
                report["diary"] += 1
    from app.care import memory_index, memory_upkeep

    owner = next((m.get("id") for m in roster.members or [] if "primary" in str(m.get("role", "")).lower()), None)
    for p in uniq:
        base = await step(f"baseline:{p['id']}", lambda s, p=p: baselines.save(s, fid, p["id"]))
        if base is not None:
            report["baselines"] += 1
            from app.care import skillbook

            if await step(f"skill:{p['id']}", lambda s, p=p, base=base: skillbook.propose_from_style(s, fid, p["id"], base.get("style") or {})):
                report["skills_proposed"] = report.get("skills_proposed", 0) + 1
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

    out = {"skill_notes_removed": 0, "confirmations_expired": 0}
    async with sessions() as session:
        services = [r[0] for r in await session.execute(select(SkillNote.service).distinct())]
        for svc in services:
            keep = select(SkillNote.id).where(SkillNote.service == svc).order_by(SkillNote.id.desc()).limit(SKILL_NOTES_KEEP)
            res = await session.execute(delete(SkillNote).where(SkillNote.service == svc, SkillNote.id.not_in(keep)))
            out["skill_notes_removed"] += res.rowcount or 0
        out["confirmations_expired"] = await expire_stale_confirmations(session)
        from app.care import skillbook

        out["skills"] = await skillbook.curate(session)
        await session.commit()
    return out


async def dream_all(sessions: async_sessionmaker, *, budget_s: float = 240.0, with_models: bool = True) -> dict:
    day = night_of()
    started = time.monotonic()
    async with sessions() as session:
        rosters = list((await session.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")).order_by(FamilyRoster.family_id))).scalars())
        todo = [r for r in rosters if not await _done(session, r.family_id, day)]
    stats = {"day": day, "families": len(rosters), "done_now": 0, "left": 0, "patterns": 0, "facts": 0, "errors": 0}
    for i, r in enumerate(todo):
        if time.monotonic() - started > budget_s:
            stats["left"] = len(todo) - i
            break
        rep = await dream_family(sessions, r, day, with_models=with_models)
        stats["done_now"] += 1
        stats["patterns"] += rep["patterns"]
        stats["facts"] += rep["facts"]
        stats["errors"] += len(rep["errors"])
    if not stats["left"]:
        stats["upkeep"] = await upkeep(sessions)
        from app.learn import jobs as learn_jobs

        stats["learn"] = await learn_jobs.nightly(sessions, with_models=with_models)
    logger.info("dream %s", stats)
    return stats
