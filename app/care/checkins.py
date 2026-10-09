"""Saheli's own check-ins: when to look in on someone, and about what.

Founder's rules (2026-10-09):
- Only at decent hours (DAY_START–DAY_END IST). Never at night.
- When a conversation has just ended, ask once about today's schedule items that are past their time and not marked.
- Ask about how they feel, problems, new reports and readings now and then, built from what Saheli knows of them.
  Weight only about every 20 days.
- The same for self care: the subject is whoever the care record is about (an elder, or a person caring for themselves).

The plan is deterministic (what is due, and is now a good moment); the words are Saheli's own, on a system turn with
the person's whole context. At most one check-in per person per run, a few per day, never while an order is moving
for them or right after another message.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, FamilyRoster, Turn
from app.core import clock

logger = logging.getLogger(__name__)

DAY_START, DAY_END = time(9, 0), time(20, 30)  # no check-in outside these hours (IST)
HEALTH_WINDOWS = ((time(10, 30), time(12, 30)), (time(16, 30), time(19, 30)))  # the gentler questions
AFTER_CHAT = (timedelta(minutes=5), timedelta(minutes=45))  # "the conversation has ended": last message this long ago
MIN_GAP = timedelta(hours=3)  # since Saheli last wrote to them on her own
MAX_PER_DAY = 3
EVERY = {"feelings": timedelta(days=3), "readings": timedelta(days=4), "weight": timedelta(days=20), "reports": timedelta(days=7)}
READING_FOR = {"bp": ("hypertension", "blood pressure", "bp", "high bp", "heart"), "sugar": ("diabetes", "sugar", "diabetic", "hba1c")}


def in_day(at: datetime) -> bool:
    t = clock.ist(at).time()
    return DAY_START <= t <= DAY_END


def in_health_window(at: datetime) -> bool:
    t = clock.ist(at).time()
    return any(a <= t <= b for a, b in HEALTH_WINDOWS)


async def _last_user_message(session: AsyncSession, family_id: str, person: str) -> datetime | None:
    return (await session.execute(
        select(Turn.at).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user").order_by(Turn.id.desc()).limit(1)
    )).scalar_one_or_none()


async def _proactive_today(session: AsyncSession, family_id: str, person: str) -> list[datetime]:
    start = clock.now() - timedelta(hours=16)
    rows = (await session.execute(
        select(Turn.at, Turn.meta).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "assistant", Turn.at >= start)
    )).all()
    return [at for at, meta in rows if (meta or {}).get("proactive")]


async def _asked(session: AsyncSession, family_id: str, subject: str) -> dict[str, datetime]:
    """When each check-in topic was last asked (care events kind=checkin)."""
    rows = (await session.execute(
        select(CareEvent.at, CareEvent.payload).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject,
                                                      CareEvent.kind == "checkin", CareEvent.at >= clock.now() - timedelta(days=40))
    )).all()
    out: dict[str, datetime] = {}
    for at, payload in rows:
        topic = (payload or {}).get("topic")
        if topic and (topic not in out or at > out[topic]):
            out[topic] = at
    return out


async def _last_vital(session: AsyncSession, family_id: str, subject: str, kind: str) -> CareEvent | None:
    rows = (await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject, CareEvent.kind == "vital")
        .order_by(CareEvent.at.desc()).limit(200)
    )).scalars()
    return next((e for e in rows if (e.payload or {}).get("kind") == kind), None)


async def _readings_needed(session: AsyncSession, family_id: str, subject: str) -> list[str]:
    conditions = " ".join(f.text.lower() for f in await store.facts(session, family_id, subject, domains=["condition"]))
    return [k for k, words in READING_FOR.items() if any(w in conditions for w in words)]


def unmarked_due(schedule: dict, now: datetime) -> list[dict]:
    """Today's items past their time that nobody marked (a dose added after its time today was never due)."""
    out = []
    for i in (schedule or {}).get("items") or []:
        if i.get("status") not in ("due", "missed") or i.get("markedBy"):
            continue
        created = i.get("createdAt")
        if created:
            try:
                if datetime.fromisoformat(created.replace("Z", "+00:00")) > now:
                    continue
            except ValueError:
                pass
        out.append(i)
    return out


async def plan(session: AsyncSession, host, family_id: str, subject: str, *, now: datetime | None = None,
               busy: bool = False) -> tuple[str, str] | None:
    """(topic, prompt) for the one check-in to send to this person now, or None."""
    now = now or clock.now()
    if busy or not in_day(now):
        return None
    sent = await _proactive_today(session, family_id, subject)
    if len(sent) >= MAX_PER_DAY or (sent and now - max(sent) < MIN_GAP / 3):
        return None
    last_msg = await _last_user_message(session, family_id, subject)
    asked = await _asked(session, family_id, subject)
    day = clock.ist_day(now)

    # 1. A conversation just ended: today's unmarked schedule items, once a day per set.
    after_chat = last_msg is not None and AFTER_CHAT[0] <= now - last_msg <= AFTER_CHAT[1]
    talking = last_msg is not None and now - last_msg < AFTER_CHAT[0]
    evening = clock.ist(now).time() >= time(19, 0)  # the day's last look, if nothing was said about them
    if not talking and (after_chat or evening):
        try:
            schedule = await host.call("get_today_schedule", {}, family_id=family_id.removeprefix("shadow:"), subject_id=subject, actor_id=subject)
        except Exception:  # noqa: BLE001 — no schedule today: nothing to ask
            schedule = {}
        due = unmarked_due(schedule, now)
        key = "doses:" + ",".join(sorted(str(i.get("scheduleId")) for i in due))
        if due and (asked.get(key) is None or clock.ist_day(asked[key]) != day):
            listed = "; ".join(f"{i.get('title')}{(' ' + i['dosage']) if i.get('dosage') else ''} at {i.get('time')}" for i in due[:8])
            return key, (
                f"[Check-in: today's schedule] These items of person {subject} were due today and nobody has marked them yet: "
                f"{listed}. Ask them once, short and warm, in their language and script, whether they did them (one message; they "
                f"can answer for all at once). When they answer later, mark each one (log_dose taken or missed). send_message to "
                f"{subject}, then reply none.")

    # 2. A gentler health question, at most one a day, in a good window, not mid-conversation.
    if not in_health_window(now) or (last_msg and now - last_msg < AFTER_CHAT[1]) or (sent and now - max(sent) < MIN_GAP):
        return None
    if any(clock.ist_day(at) == day for t, at in asked.items() if t in EVERY):
        return None
    due_topics: list[tuple[timedelta, str, str]] = []
    weight = await _last_vital(session, family_id, subject, "weight")
    if now - (weight.at if weight else datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY["weight"] and \
            now - asked.get("weight", datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY["weight"]:
        last = f"Their last weight on record is {weight.summary} from {clock.ist(weight.at).strftime('%d %b')}." if weight else "No weight is on record yet."
        due_topics.append((now - asked.get("weight", now - EVERY["weight"]) - EVERY["weight"], "weight",
                           f"[Check-in: weight] {last} Ask person {subject} once, lightly, whether they could check their weight "
                           f"today or tomorrow and tell you (no pressure; skip it if they have no scale). log_vital weight when "
                           f"they answer. send_message, then reply none."))
    for kind in await _readings_needed(session, family_id, subject):
        reading = await _last_vital(session, family_id, subject, kind)
        if (reading is None or now - reading.at >= EVERY["readings"]) and now - asked.get(f"reading:{kind}", datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY["readings"]:
            name = {"bp": "blood pressure", "sugar": "sugar"}[kind]
            last = f" Their last {name} on record: {reading.summary} ({clock.ist(reading.at).strftime('%d %b')})." if reading else ""
            due_topics.append((now - asked.get(f"reading:{kind}", now - EVERY["readings"]) - EVERY["readings"], f"reading:{kind}",
                               f"[Check-in: {name}] Person {subject} has a condition where {name} matters.{last} Ask once, short, "
                               f"whether they checked their {name} today and what it was; if they have no machine, do not push. "
                               f"log_vital {kind} when they answer, and follow your red-flag rules. send_message, then reply none."))
    for topic, text in (
        ("feelings", f"[Check-in: how they are] Ask person {subject} one short, warm question in their language and script about how "
                     f"they are feeling today: any pain, trouble, sleep or worry. Fit it to what you know of them (their conditions, "
                     f"the last things they told you, how they felt last time) and do not repeat a question from the last days. If "
                     f"they mention a problem later, log it (log_event symptom) and follow your red-flag rules. send_message, then reply none."),
        ("reports", f"[Check-in: reports] Ask person {subject} once, short, whether they had any doctor visit, test or new report "
                    f"this week. If yes, ask them to send a photo of it here (you read it and show what you found before anything "
                    f"is saved). send_message, then reply none."),
    ):
        if now - asked.get(topic, datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY[topic]:
            due_topics.append((now - asked.get(topic, now - EVERY[topic]) - EVERY[topic], topic, text))
    if not due_topics:
        return None
    due_topics.sort(key=lambda x: x[0], reverse=True)  # the most overdue first
    _, topic, prompt = due_topics[0]
    return topic, prompt


STYLE = (" Do not start with the same greeting as your last message to them; vary your words. If the conversation shows they are "
         "unwell or upset, follow up on that instead.")


async def run(sessions: async_sessionmaker, host_for, *, now: datetime | None = None) -> dict:
    """Every few minutes: for each person cared for (elders and people caring for themselves), send the check-in that is due."""
    from app.brain.wake import system_turn
    from app.tasks import runtime

    now = now or clock.now()
    stats = {"people": 0, "sent": 0}
    if not in_day(now):
        return stats
    async with sessions() as session:
        rosters = list((await session.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")))).scalars())
    for r in rosters:
        async with sessions() as session:
            subjects = await cared_for(session, r)
        for subject in subjects:
            stats["people"] += 1
            try:
                async with sessions() as session:
                    live = await runtime.live_tasks(session, r.family_id)
                    got = await plan(session, host_for(r.family_id), r.family_id, subject, now=now, busy=bool(live))
                    if not got:
                        continue
                    topic, prompt = got
                    await store.record_event(session, family_id=r.family_id, subject_id=subject, kind="checkin",
                                             summary=f"Saheli checked in: {topic.split(':')[0]}", payload={"topic": topic})
                    await session.commit()
                await system_turn(sessions, host_for(r.family_id), r.family_id, prompt + STYLE,
                                  f"checkin:{topic}:{subject}:{clock.ist_day(now)}", deliver_to=subject)
                stats["sent"] += 1
            except Exception:  # noqa: BLE001 — one person's problem must not stop the others
                logger.exception("check-in failed family=%s", r.family_id)
    return stats


async def cared_for(session: AsyncSession, roster: FamilyRoster) -> list[str]:
    """The care recipient, plus any member caring for themselves (they have their own medicines or conditions on record)."""
    from app.care.domains import HEALTH_DOMAINS

    elder = (roster.elder or {}).get("id")
    out = [elder] if elder else []
    for m in roster.members or []:
        mid = m.get("id")
        if mid and mid != elder and await store.facts(session, roster.family_id, mid, domains=sorted(HEALTH_DOMAINS)):
            out.append(mid)
    return out
