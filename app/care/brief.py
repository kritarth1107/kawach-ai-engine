"""The nightly brief: each caregiver gets the care recipient's whole day on WhatsApp (founder 2026-10-12: "a daily brief
of the elder to the caregiver every night, with all the details: what they did, medicine, health, orders, reminders.
Caregiver can change the time; by default after the elder is asleep").

- When: the caregiver's own time if they set one (daily_brief tool), else 30 minutes after the care recipient's saved
  bedtime (routine sleep), else 22:00. A time before noon is a morning brief about the day before. Once a day each.
- What: a digest built from data (today's schedule with each dose's status, the day's ledger: reminders, doses, vitals,
  symptoms, mood, alerts, check-ins; their own messages; orders and rides for them; care record changes; what is waiting
  for a caregiver; tomorrow's medicine times). Saheli writes it in the caregiver's language from that digest only.
- WhatsApp lets a business start a chat only within 24 hours of the person's last message (outside it, only an approved
  template). A caregiver who has not written in that time gets the brief the next time they write (it waits as an open
  loop for them).
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, FamilyRoster, OpenLoop, Turn
from app.core import clock

logger = logging.getLogger(__name__)

DEFAULT_TIME = time(22, 0)
AFTER_BEDTIME = timedelta(minutes=30)
LATEST_DEFAULT = time(23, 45)
LATE_OK = timedelta(hours=2)  # a brief whose time passed (the service was down) still goes this long after it
WINDOW = timedelta(hours=23, minutes=30)  # WhatsApp's 24-hour window, with a margin
SETTING_KEY = "routine:daily_brief"
CAREGIVER_ROLES = ("caregiver",)


def _hm(v: object) -> time | None:
    try:
        h, m = str(v).strip()[:5].split(":")
        return time(int(h), int(m))
    except (ValueError, AttributeError):
        return None


def _times(fact) -> list[time]:
    v = (fact.value or {}) if fact else {}
    raw = ([v.get("time")] if v.get("time") else []) + list(v.get("times") or [])
    return [t for t in (_hm(x) for x in raw) if t]


async def setting(session: AsyncSession, family_id: str, caregiver: str) -> dict:
    """The caregiver's own choice: {"time": "HH:MM"} or {"off": True}, or {} for the default."""
    row = await store.active_fact(session, family_id, caregiver, SETTING_KEY)
    return dict(row.value or {}) if row else {}


async def default_time(session: AsyncSession, family_id: str, elder: str) -> time:
    """30 minutes after the care recipient's bedtime, when an evening bedtime is saved; else 22:00."""
    sleep = await store.active_fact(session, family_id, elder, "routine:sleep")
    evening = [t for t in _times(sleep) if t.hour >= 12]
    if not evening:
        return DEFAULT_TIME
    at = datetime.combine(datetime(2000, 1, 1), min(evening)) + AFTER_BEDTIME
    return min(at.time(), LATEST_DEFAULT) if at.day == 1 else LATEST_DEFAULT


async def brief_time(session: AsyncSession, family_id: str, caregiver: str, elder: str) -> time | None:
    s = await setting(session, family_id, caregiver)
    if s.get("off"):
        return None
    return _hm(s.get("time")) or await default_time(session, family_id, elder)


def due_day(at: time, now: datetime) -> str | None:
    """The IST day whose brief is due now, or None: on its time, up to LATE_OK after. A morning time is about the day before."""
    local = clock.ist(now)
    send_at = local.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    if not (send_at <= local <= send_at + LATE_OK):
        return None
    day = local.date() if at.hour >= 12 else local.date() - timedelta(days=1)
    return day.isoformat()


async def sent_already(session: AsyncSession, family_id: str, caregiver: str, day: str) -> bool:
    return bool(await session.scalar(select(func.count()).select_from(CareEvent).where(
        CareEvent.family_id == family_id, CareEvent.kind == "daily_brief", CareEvent.payload["to"].astext == caregiver,
        CareEvent.payload["day"].astext == day)))


async def in_window(session: AsyncSession, family_id: str, person: str, now: datetime) -> bool:
    """Did they write to Saheli within WhatsApp's 24-hour window?"""
    last = await session.scalar(select(func.max(Turn.at)).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user"))
    return bool(last and now - last < WINDOW)


async def digest(session: AsyncSession, host, family_id: str, elder: dict, day: str, *, now: datetime | None = None) -> str:
    """The day as data, for Saheli to write the brief from (and the fallback if she sends nothing)."""
    from app.tasks.models import Task

    now = now or clock.now()
    eid, name = elder["id"], (elder.get("name") or "them")
    start = datetime.fromisoformat(f"{day}T00:00:00+05:30")
    end = start + timedelta(days=1)
    parts: list[str] = [f"DAY: {day} (IST), about {name}"]

    sched: dict = {}
    if day == clock.ist_day(now):
        try:
            sched = await host.call("get_today_schedule", {}, family_id=family_id.removeprefix("shadow:"), subject_id=eid, actor_id=eid) or {}
        except Exception:  # noqa: BLE001 — the rest of the brief still goes
            logger.warning("brief: schedule unavailable family=%s", family_id)
    items = [i for i in (sched.get("items") or []) if i.get("time")]
    if items:
        lines = []
        for i in sorted(items, key=lambda x: str(x.get("time"))):
            st = {"completed": "taken", "missed": "missed", "due": "not marked", "upcoming": "later today"}.get(str(i.get("status")), str(i.get("status")))
            who = " (marked on the dashboard)" if i.get("markedBy") and i.get("markedBy") != eid and st == "taken" else ""
            dose = f" {i['dosage']}" if i.get("dosage") else ""
            lines.append(f"  {i.get('time')} {i.get('title')}{dose}: {st}{who}")
        parts.append("SCHEDULE (each dose or item and what happened):\n" + "\n".join(lines))

    events = [e for e in await store.events(session, family_id, eid, day=day) if e.kind not in ("dose_corrected", "checkin")]
    if events:
        parts.append("LEDGER (reminders sent, doses, readings, symptoms, mood, alerts, orders; IST):\n"
                     + "\n".join(f"  {clock.ist(e.at).strftime('%H:%M')} {e.kind}: {e.summary[:200]}" for e in events[-40:]))

    said = (await session.execute(select(Turn.at, Turn.text).where(
        Turn.family_id == family_id, Turn.thread_id == eid, Turn.role == "user", Turn.at >= start, Turn.at < end).order_by(Turn.at))).all()
    said = [(at, t) for at, t in said if (t or "").strip() and not str(t).startswith(("v2:", "[Task update]"))]
    if said:
        parts.append(f"WHAT {name.upper()} SAID TO SAHELI ({len(said)} messages; summarise what they talked about, do not quote all):\n"
                     + "\n".join(f"  {clock.ist(at).strftime('%H:%M')} {str(t)[:160]}" for at, t in said[-25:]))
    else:
        parts.append(f"{name} did not write to Saheli today.")

    from app.tasks import runtime

    tasks = (await session.execute(select(Task).where(
        Task.family_id == family_id, Task.subject_id == eid, or_(Task.created_at >= start, Task.updated_at >= start), Task.created_at < end)
        .order_by(Task.created_at))).scalars().all()
    if tasks:
        lines = []
        for t in tasks:
            r = t.result or {}
            outcome = ("placed" if r.get("placed") else "booked" if r.get("booked") else t.status)
            total = f", {r.get('total') or (t.details or {}).get('confirmed_total')}" if (r.get("total") or (t.details or {}).get("confirmed_total")) else ""
            lines.append(f"  {t.service} {t.kind}: {runtime.bought(t)}: {outcome}{total}")
        parts.append("ORDERS AND RIDES FOR THEM:\n" + "\n".join(lines))

    from app.care import versions

    changed = [v for v in await versions.changes(session, family_id, [eid], limit=20) if v.at and start <= v.at < end]
    if changed:
        parts.append("CARE RECORD CHANGES:\n" + "\n".join(f"  {v.op} {v.kind}: {(v.body or v.title or '')[:140]}" for v in changed[:10]))

    loops = [lp for lp in await store.live_loops(session, family_id, [eid]) if lp.kind not in ("delivery", "brief")]
    if loops:
        parts.append("WAITING FOR A CAREGIVER OR OPEN:\n" + "\n".join(f"  {lp.kind}: {lp.title[:140]}" for lp in loops[:8]))

    meds = [f for f in await store.facts(session, family_id, eid, domains=["medicine", "appointment"]) if f.status == "active"]
    tomorrow = (start + timedelta(days=1)).date().isoformat()
    upcoming = [f"{(f.value or {}).get('name') or f.key.split(':', 1)[-1]} at {', '.join((f.value or {}).get('times') or [])}"
                for f in meds if f.domain == "medicine" and (f.value or {}).get("times")]
    upcoming += [f"appointment: {f.text[:120]}" for f in meds if f.domain == "appointment" and str((f.value or {}).get("when", "")).startswith(tomorrow)]
    if upcoming:
        parts.append("TOMORROW: " + "; ".join(upcoming[:8]))
    return "\n\n".join(parts)


def prompt_for(caregiver: dict, elder: dict, day: str, text: str) -> str:
    cg, name = caregiver["id"], (elder.get("name") or "them")
    return (f"[Daily brief] Tonight's brief about {name}'s day ({day}) for person {cg} ({caregiver.get('name') or 'the caregiver'}). "
            f"Write it for them with send_message: one short opening line, then short sections with bold titles (medicines; health "
            f"and mood; what {name} did and talked about; orders and rides; reminders; anything that needs them), skipping a "
            "section that has nothing, and one closing line about tomorrow. Use only the data below: times, names, doses and "
            "amounts exactly as given; a dose that is not marked is 'not marked', never 'taken'. Warm and plain, readable in a "
            f"minute. Then reply none.\n\n{text}")


async def send(sessions: async_sessionmaker, host, roster, caregiver: dict, day: str, *, now: datetime | None = None, on_request: bool = False) -> str:
    """Build and deliver one caregiver's brief. Returns 'sent', 'waiting' (outside WhatsApp's 24-hour window: it goes when
    they next write) or 'empty'."""
    from app.brain.wake import system_turn
    from app.care import checkins

    now = now or clock.now()
    fid, cg = roster.family_id, caregiver["id"]
    async with sessions() as session:
        text = await digest(session, host, fid, roster.elder, day, now=now)
        open_now = on_request or await in_window(session, fid, cg, now)
        if not open_now:
            name = roster.elder.get("name") or "them"
            await store.open_loop(session, family_id=fid, subject_id=cg, kind="brief", owner_id=cg,
                                  title=f"Daily brief about {name} for {day}: goes to {caregiver.get('name') or 'the caregiver'} when they next write on WhatsApp",
                                  detail={"day": day, "text": text[:6000], "max_wakes": 0}, alert_rule="dashboard",
                                  dedupe_key=f"brief:{cg}:{day}")
            await store.record_event(session, family_id=fid, subject_id=roster.elder["id"], kind="daily_brief",
                                     summary=f"Daily brief for {day} waiting for {caregiver.get('name') or cg}", payload={"to": cg, "day": day, "waiting": True})
            await session.commit()
            return "waiting"
        await store.record_event(session, family_id=fid, subject_id=roster.elder["id"], kind="daily_brief",
                                 summary=f"Daily brief for {day} to {caregiver.get('name') or cg}", payload={"to": cg, "day": day})
        await session.commit()
    ref = f"brief:{cg}:{day}" + (f":{now.strftime('%H%M')}" if on_request else "")
    started = clock.now()
    await system_turn(sessions, host, fid, prompt_for(caregiver, roster.elder, day, text), ref, deliver_to=cg)
    async with sessions() as session:
        if await checkins.went_out(session, fid, cg, ref, started - timedelta(seconds=5)):
            return "sent"
    # she sent nothing: the digest itself goes, so a brief is never lost
    res = await host.call("send_whatsapp", {"to": cg, "text": f"Daily brief: {roster.elder.get('name') or ''}\n\n{text}"[:3900]},
                          family_id=fid.removeprefix("shadow:"), subject_id=roster.elder["id"], actor_id=roster.elder["id"])
    async with sessions() as session:
        await store.add_turn(session, family_id=fid, thread_id=cg, role="assistant", text=f"Daily brief ({day})",
                             meta={"proactive": True, "delivered": (res or {}).get("delivered"), "ref": f"{ref}:plain"})
        await session.commit()
    return "sent"


def caregivers(roster) -> list[dict]:
    elder = (roster.elder or {}).get("id")
    return [m for m in roster.members or [] if m.get("id") and m.get("id") != elder
            and any(r in str(m.get("role") or "").lower() for r in CAREGIVER_ROLES)]


async def run(sessions: async_sessionmaker, host_for, *, now: datetime | None = None) -> dict:
    """Every few minutes: send each caregiver's brief when its time comes (once a day each)."""
    now = now or clock.now()
    stats = {"due": 0, "sent": 0, "waiting": 0}
    async with sessions() as session:
        rosters = list((await session.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")))).scalars())
    for r in rosters:
        if not (r.elder or {}).get("id"):
            continue
        for cg in caregivers(r):
            try:
                async with sessions() as session:
                    at = await brief_time(session, r.family_id, cg["id"], r.elder["id"])
                    day = due_day(at, now) if at else None
                    if not day or await sent_already(session, r.family_id, cg["id"], day):
                        continue
                stats["due"] += 1
                out = await send(sessions, host_for(r.family_id), r, cg, day, now=now)
                stats["sent" if out == "sent" else "waiting"] += 1
            except Exception:  # noqa: BLE001 — one caregiver's brief must not stop the others
                logger.exception("daily brief failed family=%s", r.family_id)
    return stats


async def waiting_for(session: AsyncSession, family_id: str, person: str) -> list[OpenLoop]:
    return [lp for lp in await store.live_loops(session, family_id, [person]) if lp.kind == "brief" and lp.owner_id == person]
