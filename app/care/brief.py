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


class BackendDown(Exception):
    """The backend did not answer (a restart, a deploy): the brief waits for the next run instead of going without its
    schedule (live 14 Oct 20:30: two briefs were built while the backend restarted and held with no doses in them)."""


def backend_down(exc: BaseException) -> bool:
    import httpx

    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (502, 503, 504)
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
    """30 minutes after the care recipient's bedtime, when an evening bedtime is saved; else 22:00.

    Bedtimes are activities since 2026-10-13 (activity:sleep); an older record may still hold routine:sleep. Reading only
    the old key left every brief at 22:00 (world message-budget sim, 2026-10-13)."""
    from app.brain.wake import own_night

    night = await own_night(session, family_id, elder)
    evening = [time(night[0] // 60, night[0] % 60)] if night and night[0] >= 12 * 60 else []
    if not evening:
        sleep = await store.active_fact(session, family_id, elder, "routine:sleep")
        evening = [t for t in _times(sleep) if t.hour >= 12]
    if not evening:
        return DEFAULT_TIME
    at = datetime.combine(datetime(2000, 1, 1), min(evening)) + AFTER_BEDTIME
    return min(at.time(), LATEST_DEFAULT) if at.day == 1 else LATEST_DEFAULT


async def brief_time(session: AsyncSession, family_id: str, caregiver: str, elder: str) -> time | None:
    """The caregiver's own time when they chose one; else the default, moved out of the caregiver's own night when Saheli
    knows it (a caregiver who sleeps at 21:30 got the 22:00 brief in their sleep): 30 minutes before their bedtime when
    that is still evening, else their morning (wake-up plus 30 minutes, a brief about the day before)."""
    s = await setting(session, family_id, caregiver)
    if s.get("off"):
        return None
    chosen = _hm(s.get("time"))
    if chosen:
        return chosen
    at = await default_time(session, family_id, elder)
    from app.brain.wake import QUIET_END, QUIET_START, _inside, own_night

    # Their own night, or the default quiet hours when Saheli doesn't know it: the same night send_message refuses to write
    # in (live 2026-10-13: a caregiver with no sleep time got the 22:00 default, her brief was refused as night, and the
    # raw data went out instead).
    night = await own_night(session, family_id, caregiver) or (QUIET_START * 60, QUIET_END * 60)
    if not _inside(at.hour * 60 + at.minute, night):
        return at
    before = night[0] - 30
    if 18 * 60 <= before < 24 * 60:
        return time(before // 60, before % 60)
    morning = (night[1] + 30) % (24 * 60)
    return time(morning // 60, morning % 60) if morning < 12 * 60 else at


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


async def chat_open(host, session: AsyncSession, family_id: str, person: str, now: datetime) -> bool:
    """Is their WhatsApp chat open? The backend keeps every inbound message's time (also a button tap that never reached
    Saheli); without it, whether they wrote to Saheli in the last 24 hours."""
    try:
        res = await host.call("whatsapp_window", {"to": person}, family_id=family_id.removeprefix("shadow:"), subject_id=person, actor_id=person)
        if isinstance(res, dict) and isinstance(res.get("open"), bool):
            return res["open"]
    except Exception:  # noqa: BLE001
        pass
    return await in_window(session, family_id, person, now)


async def language_setting(session: AsyncSession, family_id: str, person: str) -> dict | None:
    from app.care import language

    row = next((f for f in await store.facts(session, family_id, person, domains=["language"]) if f.status == "active"), None)
    return language.normalise(row.value) if row and isinstance(row.value, dict) else None


async def digest(session: AsyncSession, host, family_id: str, elder: dict, day: str, *, now: datetime | None = None,
                 to: str | None = None, summary_defaults: list[str] | None = None) -> str:
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
        except Exception as exc:  # noqa: BLE001 — the rest of the brief still goes, unless the backend itself is down
            if backend_down(exc):
                raise BackendDown(str(exc)[:200]) from exc
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

    all_events = [e for e in await store.events(session, family_id, eid, day=day) if e.kind != "dose_corrected"]
    # A check-in that went out, and whether they wrote back, is part of the day. One that never went out is not.
    checkins = [e for e in all_events if e.kind == "checkin" and (e.payload or {}).get("sent") is not False]
    events = [e for e in all_events if e.kind != "checkin"]
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
    if checkins:
        lines = []
        said_at = [at for at, _t in said]
        for i, e in enumerate(checkins):
            nxt = checkins[i + 1].at if i + 1 < len(checkins) else end
            replied = any(e.at < at < nxt for at in said_at)
            topic = str((e.payload or {}).get("topic") or e.summary or "check-in").split(":")[0][:80]
            lines.append(f"  {clock.ist(e.at).strftime('%H:%M')} check-in ({topic}): {'they wrote back' if replied else 'no reply'}")
        parts.append("CHECK-INS (whether they wrote back after Saheli checked in):\n" + "\n".join(lines))

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

    try:  # the food log (NUTRITION_PLAN §N9): meals, totals against targets, water, not logged, waiting, strict rules
        from app.nutrition import report as food_report

        async with session.begin_nested():
            food = await food_report.brief_block(session, family_id, eid, day, now=now, to=to, summary_defaults=summary_defaults)
        if food:
            parts.append(food)
    except Exception:  # noqa: BLE001 — the rest of the brief still goes
        logger.warning("brief: food block failed family=%s", family_id, exc_info=True)

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
    from app.care import outcomes_learn

    noticed = await outcomes_learn.new_since(session, family_id, eid, now=end)
    if noticed:  # one line each in the brief: "I noticed … so I'll …" (trust plan A5)
        parts.append("WHAT SAHELI LEARNED TODAY (say each in one line as \"I noticed … so I'll …\"; they can undo it on the dashboard):\n"
                     + "\n".join(f"  {n}" for n in noticed[:3]))
    return "\n\n".join(parts)


def prompt_for(caregiver: dict, elder: dict, day: str, text: str, *, send: bool = True) -> str:
    cg, name = caregiver["id"], (elder.get("name") or "them")
    how = "Write it for them with send_message: " if send else "Write the WhatsApp message: "
    end = "Then reply none." if send else "Write only that message."
    return (f"[Daily brief] Tonight's brief about {name}'s day ({day}) for person {cg} ({caregiver.get('name') or 'the caregiver'}). "
            f"{how}one short opening line, then short sections with bold titles (medicines; health "
            f"and mood; food and nutrition; what {name} did and talked about; orders and rides; reminders; anything that needs "
            "them), skipping a section that has nothing, and one closing line about tomorrow. Use only the data below: times, "
            "names, doses and amounts exactly as given; a dose that is not marked is 'not marked', never 'taken'. Do not say "
            "how they feel, or that they are fine, stable, peaceful, or free of symptoms or complaints, unless the data says "
            "that. If the data has no mood or health, say there is no mood or health update. If they wrote and a later "
            "check-in has no reply, say that plainly. Food and nutrition (from FOOD): the meals with their main items, the "
            "day's kcal and the nutrients named in FOOD against any targets (who set them), water, meals not logged (never call them skipped "
            "unless FOOD says skipped), delivered food not yet confirmed eaten, and any strict rule or target gone past, "
            f"plainly and without judging the food; say which numbers are only estimates. Warm and plain, readable in a minute. {end}\n\n{text}")


WRITER = (
    "You write one WhatsApp message from the data in the user message. Use only that data. "
    "Do not say how they feel, or that they are fine, stable, peaceful, or free of symptoms or complaints, unless the data says that. "
    "If the data has no mood or health, say there is no mood or health update. If a check-in has no reply, say that plainly. "
    "Times, names, doses and amounts exactly as given. A dose that is not marked is 'not marked', never 'taken'. "
    "Skip a section that has nothing. Write only the message, no preamble."
)
FOOD_WRITER = (
    "You write one short WhatsApp note of this person's own food, from the FOOD block only. "
    "Numbers only as written. No advice, no judgement, no mood and no health. Write only the message."
)


async def write_text(prompt: str, *, food: bool = False) -> str:
    """The brief writer's message. Empty when no model answers: the raw data is never sent instead."""
    from app.llm import router

    try:
        out = await router.complete(
            "worker", system_stable=FOOD_WRITER if food else WRITER,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt[:8000]}]}],
            max_tokens=2000, effort="low", timeout_s=40,
        )
    except Exception:
        logger.warning("brief writer failed", exc_info=True)
        return ""
    text = (out.text or "").strip()
    return "" if text.lower() == "none" else text


def keep(text: str, known: str) -> str:
    """Drop brief sentences the day's data does not support (a mood, a reading, a time, a condition)."""
    from app.brain import guards

    kept = []
    for s in guards.sentences(text):
        if guards.brief_mood_claims(s, known) or guards.ungrounded(s, known=known):
            continue
        kept.append(s.strip())
    return " ".join(kept).strip()


def _lines_under(digest: str, header: str) -> list[str]:
    lines, on = [], False
    for line in digest.splitlines():
        if line.startswith("  "):
            if on:
                lines.append(line.strip())
        else:
            on = line.startswith(header)
    return lines


def factual_fallback(digest: str, name: str) -> str:
    """What the data itself says, when the writer invents a mood or sends nothing. No health claim the data lacks."""
    parts: list[str] = []
    for line in digest.splitlines():
        if "did not write to Saheli today" in line:
            parts.append(line.strip())
    said = _lines_under(digest, "WHAT ")
    if said and not parts:
        parts.append("What they wrote: " + "; ".join(said[:8]))
    checks = _lines_under(digest, "CHECK-INS")
    if checks:
        parts.append("Check-ins: " + "; ".join(checks[:8]))
    if not any(k in digest.lower() for k in ("vital:", "symptom:", "mood:")):
        parts.append("No mood or health update today.")
    text = " ".join(parts).strip()
    return text or f"{name} did not write to Saheli today. No mood or health update today."


async def _checked(text: str, known: str) -> str:
    """The same no-invention check a reply gets, then the brief's own mood check."""
    from app.brain import grounding

    text = keep(text, known)
    if not text:
        return ""
    claims = await grounding.check(text, memory=known, their_messages=[], tool_results=[])
    if claims:
        text = keep(grounding.strip(text, claims), known)
    return text


async def deliver_waiting(session: AsyncSession, family_id: str, speaker_id: str, elder_name: str) -> list[str]:
    """Waiting briefs for this speaker, as their own messages: the writer's text, not a rewrite by this conversation.
    A brief that invents a mood or a fact is not sent; the day's data is said plainly instead. The loop closes."""
    from sqlalchemy.orm.attributes import flag_modified

    out: list[str] = []
    for lp in await waiting_for(session, family_id, speaker_id):
        detail = dict(lp.detail or {})
        known = str(detail.get("text") or "")
        food = known.startswith("OWN FOOD RECAP")
        written = str(detail.get("written") or "").strip()
        if not written:
            written = await write_text(known, food=food)
        cleaned = await _checked(written, known)
        if not cleaned and not food:
            cleaned = await _checked(factual_fallback(known, elder_name), known)
        if cleaned and guards_mood_left(cleaned, known):
            cleaned = keep(cleaned, known)
        if not cleaned:
            tries = int(detail.get("tries") or 0) + 1
            detail["tries"] = tries
            lp.detail = detail
            flag_modified(lp, "detail")
            if tries >= 2:
                await store.close_loop(session, lp.id, note="not written")
            continue
        await store.add_turn(
            session, family_id=family_id, thread_id=speaker_id, role="assistant", text=cleaned,
            meta={"proactive": True, "via": "brief", "ref": f"brief-waiting:{lp.id}"},
        )
        await store.close_loop(session, lp.id, note="sent with their reply")
        out.append(cleaned)
    return out


def guards_mood_left(text: str, known: str) -> bool:
    from app.brain import guards

    return bool(guards.brief_mood_claims(text, known))


async def send(sessions: async_sessionmaker, host, roster, caregiver: dict, day: str, *, now: datetime | None = None, on_request: bool = False) -> str:
    """Build and deliver one caregiver's brief. Returns 'sent', 'waiting' (outside WhatsApp's 24-hour window: it goes when
    they next write) or 'empty'."""
    from app.brain.wake import system_turn
    from app.care import checkins

    now = now or clock.now()
    fid, cg = roster.family_id, caregiver["id"]
    if cg == (roster.elder or {}).get("id") or cg not in {m["id"] for m in caregivers(roster)}:
        # The brief is about the person cared for and goes only to their caregivers, never to them.
        logger.error("brief refused: not a caregiver family=%s to=%s", fid, cg)
        return "empty"
    name = roster.elder.get("name") or "them"

    async def wait_for_them(session: AsyncSession, text: str, why: str) -> None:
        written = await _checked(await write_text(prompt_for(caregiver, roster.elder, day, text, send=False)), text)
        await store.open_loop(session, family_id=fid, subject_id=cg, kind="brief", owner_id=cg,
                              title=f"Daily brief about {name} for {day}: goes to {caregiver.get('name') or 'the caregiver'} when they next write on WhatsApp",
                              detail={"day": day, "text": text[:6000], "written": written[:4000], "max_wakes": 0}, alert_rule="dashboard",
                              dedupe_key=f"brief:{cg}:{day}")
        await store.record_event(session, family_id=fid, subject_id=roster.elder["id"], kind="daily_brief",
                                 summary=f"Daily brief for {day} waiting for {caregiver.get('name') or cg} ({why})",
                                 payload={"to": cg, "day": day, "waiting": True, "why": why})

    async with sessions() as session:
        text = await digest(session, host, fid, roster.elder, day, now=now, to=cg,
                            summary_defaults=[m["id"] for m in caregivers(roster)])
        open_now = on_request or await chat_open(host, session, fid, cg, now)
        if not open_now:
            await wait_for_them(session, text, "chat closed")
            await session.commit()
            # the approved "daily summary is ready" template tells them now; their tap opens the chat and the brief goes
            saved = (await language_setting(session, fid, cg)) or {}
            try:
                await host.call("send_template", {"to": cg, "template": {
                    "purpose": "summary", "params": [(str(name).split() or ["-"])[0]], **({"lang": saved["language"]} if saved.get("language") else {})}},
                    family_id=fid.removeprefix("shadow:"), subject_id=roster.elder["id"], actor_id=roster.elder["id"])
            except Exception:  # noqa: BLE001 — the brief still waits for their next message
                logger.warning("brief template failed family=%s", fid)
            return "waiting"
    ref = f"brief:{cg}:{day}" + (f":{now.strftime('%H%M')}" if on_request else "")
    from app.brain.wake import quiet_for

    night = False  # set in the loop (it always runs); the type gate cannot see that
    for attempt in range(2):
        started = clock.now()
        prompt = prompt_for(caregiver, roster.elder, day, text)
        if attempt:
            prompt = "(Your last try sent nothing: write the brief now with send_message, in plain words.)\n" + prompt
        await system_turn(sessions, host, fid, prompt, ref, deliver_to=cg)
        async with sessions() as session:
            if await checkins.went_out(session, fid, cg, ref, started - timedelta(seconds=5)):
                await store.record_event(session, family_id=fid, subject_id=roster.elder["id"], kind="daily_brief",
                                         summary=f"Daily brief for {day} to {caregiver.get('name') or cg}", payload={"to": cg, "day": day})
                await session.commit()
                return "sent"
            night, _ = await quiet_for(session, fid, cg)
        if night and not on_request:
            break  # it is their night: the brief waits for their next message rather than waking them
    # She sent nothing (their night, or the writer failed twice). The brief waits for their next message and stays on the
    # dashboard. The raw data block is never sent: it is written for the writer, not for a person (live 2026-10-13).
    async with sessions() as session:
        await wait_for_them(session, text, "their night" if night else "not written")
        await session.commit()
    if not night:
        logger.error("daily brief not written family=%s to=%s day=%s", fid, cg, day)
    return "waiting"


def caregivers(roster) -> list[dict]:
    elder = (roster.elder or {}).get("id")
    return [m for m in roster.members or [] if m.get("id") and m.get("id") != elder
            and any(r in str(m.get("role") or "").lower() for r in CAREGIVER_ROLES)]


def self_care(roster) -> bool:
    """The person cared for manages their own care (their role is a caregiver's). They never get the caregiver brief."""
    elder = roster.elder or {}
    if any(r in str(elder.get("role") or "").lower() for r in CAREGIVER_ROLES):
        return True
    eid = elder.get("id")
    return any(m.get("id") == eid and any(r in str(m.get("role") or "").lower() for r in CAREGIVER_ROLES)
               for m in (roster.members or []))


async def send_own_food(sessions: async_sessionmaker, host, roster, day: str, *, now: datetime | None = None) -> str:
    """A self-care user's own food recap at brief time. Not the caregiver brief, which never goes to the person it is about.
    Nothing goes out during their night. Returns 'sent', 'waiting' or 'empty'."""
    from app.brain.wake import quiet_for, system_turn
    from app.care import checkins
    from app.nutrition import report as food_report

    now = now or clock.now()
    elder = roster.elder or {}
    eid = elder.get("id")
    food = ""
    night = False
    if not eid or not self_care(roster):
        return "empty"
    fid = roster.family_id
    name = elder.get("name") or "them"

    async def hold(session: AsyncSession, food: str, why: str) -> None:
        await store.open_loop(session, family_id=fid, subject_id=eid, kind="brief", owner_id=eid,
                              title=f"Food recap for {day}: goes to {name} when they next write on WhatsApp",
                              detail={"day": day, "text": food_report.recap_waiting_text(elder, day, food)[:6000], "max_wakes": 0},
                              alert_rule="dashboard", dedupe_key=f"foodrecap:{eid}:{day}")
        await store.record_event(session, family_id=fid, subject_id=eid, kind="daily_brief",
                                 summary=f"Food recap for {day} waiting for {name} ({why})",
                                 payload={"to": eid, "day": day, "waiting": True, "why": why, "food": True})

    async with sessions() as session:
        async with session.begin_nested():
            food = await food_report.brief_block(session, fid, eid, day, now=now, to=eid, summary_defaults=[eid])
        if not food:
            return "empty"
        night, _ = await quiet_for(session, fid, eid, now)
        open_now = (not night) and await chat_open(host, session, fid, eid, now)
        if night or not open_now:
            await hold(session, food, "their night" if night else "chat closed")
            await session.commit()
            if not night:
                saved = (await language_setting(session, fid, eid)) or {}
                try:
                    await host.call("send_template", {"to": eid, "template": {
                        "purpose": "summary", "params": [(str(name).split() or ["-"])[0]],
                        **({"lang": saved["language"]} if saved.get("language") else {})}},
                        family_id=fid.removeprefix("shadow:"), subject_id=eid, actor_id=eid)
                except Exception:  # noqa: BLE001 — the recap still waits for their next message
                    logger.warning("food recap template failed family=%s", fid)
            return "waiting"
    ref = f"foodrecap:{eid}:{day}"
    prompt = food_report.recap_prompt(elder, day, food)
    started = clock.now()

    async def went(session: AsyncSession) -> bool:
        return await checkins.went_out(session, fid, eid, ref, started - timedelta(seconds=5))

    async def mark_sent(session: AsyncSession) -> None:
        await store.record_event(session, family_id=fid, subject_id=eid, kind="daily_brief",
                                 summary=f"Food recap for {day} to {name}",
                                 payload={"to": eid, "day": day, "food": True})
        await session.commit()

    await system_turn(sessions, host, fid, prompt, ref, deliver_to=eid)
    async with sessions() as session:
        if await went(session):
            await mark_sent(session)
            return "sent"
        night, _ = await quiet_for(session, fid, eid, now)
    if not night:
        prompt = "(Your last try sent nothing: write the food recap now with send_message, in plain words, no advice.)\n" + prompt
        await system_turn(sessions, host, fid, prompt, ref, deliver_to=eid)
        async with sessions() as session:
            if await went(session):
                await mark_sent(session)
                return "sent"
            night, _ = await quiet_for(session, fid, eid)
    async with sessions() as session:
        await hold(session, food, "their night" if night else "not written")
        await session.commit()
    return "waiting"


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
            except BackendDown as exc:
                # Nothing was recorded, so the next run (minutes later, up to LATE_OK) builds it again with the schedule.
                logger.warning("daily brief deferred, backend down family=%s: %s", r.family_id, exc)
                stats["deferred"] = stats.get("deferred", 0) + 1
                stats["due"] -= 1
                return stats
            except Exception:  # noqa: BLE001 — one caregiver's brief must not stop the others
                logger.exception("daily brief failed family=%s", r.family_id)
        eid = (r.elder or {}).get("id")
        if not eid or not self_care(r):
            continue
        try:
            async with sessions() as session:
                at = await brief_time(session, r.family_id, eid, eid)
                day = due_day(at, now) if at else None
                if not day or await sent_already(session, r.family_id, eid, day):
                    continue
            out = await send_own_food(sessions, host_for(r.family_id), r, day, now=now)
            if out == "empty":
                continue
            stats["due"] += 1
            stats["sent" if out == "sent" else "waiting"] += 1
        except Exception:  # noqa: BLE001 — one person's recap must not stop the others
            logger.exception("food recap failed family=%s", r.family_id)
    return stats


async def waiting_for(session: AsyncSession, family_id: str, person: str) -> list[OpenLoop]:
    return [lp for lp in await store.live_loops(session, family_id, [person]) if lp.kind == "brief" and lp.owner_id == person]
