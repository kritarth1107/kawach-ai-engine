"""Saheli's own check-ins (nudges): when to look in on someone, and about what.

Founder's rules (2026-10-09):
- Only at decent hours (DAY_START–DAY_END IST). Never at night.
- When a conversation has just ended, ask once about today's schedule items that are past their time and not marked.
- Ask about how they feel, problems, new reports and readings now and then, built from what Saheli knows of them.
  Weight only about every 20 days.
- The same for self care: the subject is whoever the care record is about (an elder, or a person caring for themselves).

Founder 2026-10-12 ("nudges should be there for all, personal and with memory, and for collecting more info about them
for better nudges in future"):
- Everyone who talks to Saheli gets them: the care recipient, people caring for themselves, and caregivers.
- Personal nudges from memory: a follow-up on something they told Saheli (a plan, a goal, a worry, a person), and one
  light getting-to-know question at a time (their day, what they enjoy, the people and dates that matter...), saved to
  their care record so later nudges can be about their life.
- An order moving for one person no longer holds back everyone's nudges (live: any open order in the family stopped
  all check-ins for the whole day), and an order left waiting for an answer stops holding them back after a while.

The plan is deterministic (what is due, and is now a good moment); the words are Saheli's own, on a system turn with
the person's whole context. At most one check-in per person per run, one per gentle window and a few per day, never
while an order is moving for them or right after another message.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from sqlalchemy import func, select
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
GENTLE_PER_DAY = {True: 2, False: 1}  # gentle nudges a day: a person cared for (one per window) / a caregiver
EVERY = {"feelings": timedelta(days=3), "readings": timedelta(days=4), "weight": timedelta(days=20), "reports": timedelta(days=7),
         "confirm": timedelta(days=30), "follow_up": timedelta(days=2), "know": timedelta(days=2)}
KNOW_AGAIN = timedelta(days=21)  # a getting-to-know question they did not answer is not asked again before this
TOLD_WITHIN = timedelta(days=4)  # a follow-up needs something they told Saheli this recently
MOVING_WAIT = timedelta(minutes=30)  # an order waiting this long for their answer no longer holds their nudges back
NUDGED_ROLES = ("caregiver", "family member")  # members (besides people cared for) who get personal nudges, never a doctor

# What Saheli learns about each person, one light question at a time (item, what to learn, how to save it, the care
# record domains that already answer it). Answers fill the care record; later nudges and replies use them.
KNOW = (
    ("naming", "what they like to be called", "naming (address_as)", ("naming",)),
    ("routine", "their usual day: when they wake, eat, walk or exercise, rest and sleep", "remember routine", ("routine",)),
    ("interests", "what they enjoy: hobbies, music, shows, prayer, games, reading", "remember preference", ("preference",)),
    ("food", "food they love and food they avoid", "remember preference (or diet for a rule)", ("diet", "dish")),
    ("people", "the people they are close to and talk to often", "remember family or contact", ("family", "contact")),
    ("dates", "dates that matter to them: birthday, anniversary, festivals they keep", "remember occasion", ("occasion",)),
    ("goals", "something they want to do more of or keep up (a habit, health, learning)", "a note (or remember routine)", ()),
    ("help", "what they would most like Saheli's help with", "a note", ()),
)
READING_FOR = {"bp": ("hypertension", "blood pressure", "bp", "high bp", "heart"), "sugar": ("diabetes", "sugar", "diabetic", "hba1c")}


def in_day(at: datetime) -> bool:
    t = clock.ist(at).time()
    return DAY_START <= t <= DAY_END


def in_health_window(at: datetime) -> bool:
    return window_of(at) is not None


def window_of(at: datetime) -> int | None:
    t = clock.ist(at).time()
    return next((i for i, (a, b) in enumerate(HEALTH_WINDOWS) if a <= t <= b), None)


def gentle(topic: str) -> bool:
    return topic in EVERY or topic.startswith(("reading:", "confirm:", "know:"))


def moving_for(tasks: list, person: str, now: datetime) -> bool:
    """An order or ride is moving for this person (theirs, or for them) and has not been left waiting for long."""
    for t in tasks:
        if person not in (t.requested_by, t.subject_id):
            continue
        if t.status in ("queued", "running"):
            return True
        changed = t.updated_at or t.created_at
        if changed and now - changed < MOVING_WAIT:
            return True
    return False


def _hm(v: object) -> int | None:
    try:
        h, m = str(v).strip()[:5].split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


async def awake(session: AsyncSession, family_id: str, person: str, now: datetime) -> bool:
    """Is this a good time for them by their own day (their routine's wake and sleep times, when known)? Someone who wakes
    at 14:00 gets no morning nudge; someone who sleeps at 20:00 gets none late in the evening."""
    facts = {f.key: f.value or {} for f in await store.facts(session, family_id, person, domains=["routine"]) if f.status == "active"}

    def times(key: str) -> list[int]:
        v = facts.get(key) or {}
        return [m for m in (_hm(t) for t in ([v.get("time")] if v.get("time") else []) + list(v.get("times") or [])) if m is not None]

    t = clock.ist(now)
    mins = t.hour * 60 + t.minute
    wake, sleep = times("routine:wake"), times("routine:sleep")
    if wake and mins < max(wake) + 60:
        return False
    if sleep and min(sleep) > 12 * 60 and mins > min(sleep) - 60:  # (a bedtime after midnight is past every window)
        return False
    return True


async def _told_lately(session: AsyncSession, family_id: str, person: str, now: datetime) -> bool:
    """Did they tell Saheli something (a few words, not just 'ok') lately, before the current conversation?"""
    rows = (await session.execute(
        select(Turn.text).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user",
                                Turn.at >= now - TOLD_WITHIN, Turn.at <= now - AFTER_CHAT[1])
        .order_by(Turn.id.desc()).limit(30)
    )).scalars()
    return any(len((t or "").split()) >= 3 for t in rows)


async def about_them(session: AsyncSession, family_id: str, person: str, now: datetime) -> str:
    """What a personal nudge is built from: what they said lately (with Saheli's answers) and what is known of them."""
    turns = [t for t in await store.recent_turns(session, family_id, person, limit=40)
             if t.at and t.at >= now - timedelta(days=7) and (t.text or "").strip().lower() not in ("", "none")
             and not (t.text or "").startswith(("[Task update]", "v2:"))][-12:]
    said = "\n".join(f"  {clock.ist(t.at).strftime('%d %b %H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text[:180]}" for t in turns)
    facts = [f for f in await store.facts(session, family_id, person) if f.status == "active"][:14]
    known = "\n".join(f"  - {f.text[:160]}" for f in facts)
    notes = "\n".join(f"  - {n.body_md.strip()[:300]}" for n in (await store.notes(session, family_id, [person]))[:2] if (n.body_md or "").strip())
    parts = [f"What they said lately:\n{said}" if said else "", f"What you know about them:\n{known}" if known else "",
             f"Notes about them:\n{notes}" if notes else ""]
    return "\n".join(p for p in parts if p)


async def next_to_know(session: AsyncSession, family_id: str, person: str, asked: dict[str, datetime], now: datetime) -> tuple | None:
    """The first getting-to-know item the care record does not answer yet and that was not asked lately."""
    have = {f.domain for f in await store.facts(session, family_id, person) if f.status == "active" and f.key != "routine:daily_brief"}
    for item in KNOW:
        key, _, _, domains = item
        if domains and have.intersection(domains):
            continue
        if f"know:{key}" in asked and now - asked[f"know:{key}"] < KNOW_AGAIN:
            continue
        return item
    return None


async def _last_user_message(session: AsyncSession, family_id: str, person: str) -> datetime | None:
    return (await session.execute(
        select(Turn.at).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user").order_by(Turn.id.desc()).limit(1)
    )).scalar_one_or_none()


async def _proactive_today(session: AsyncSession, family_id: str, person: str) -> list[datetime]:
    start = clock.now() - timedelta(hours=16)
    rows = (await session.execute(
        select(Turn.at, Turn.meta).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "assistant", Turn.at >= start)
    )).all()
    # Order and ride updates answer something they asked for: they are not nudges and do not use up the day's check-ins.
    return [at for at, meta in rows if (meta or {}).get("proactive")
            and not str((meta or {}).get("ref") or (meta or {}).get("fallback_delivery") or "").startswith("task:")]


RETRY_UNSENT = timedelta(hours=6)  # a check-in that did not go out (nothing fitted, or it was held back) is tried again after this


async def _asked_rows(session: AsyncSession, family_id: str, subject: str) -> list[tuple[str, datetime, bool]]:
    rows = (await session.execute(
        select(CareEvent.at, CareEvent.payload).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject,
                                                      CareEvent.kind == "checkin", CareEvent.at >= clock.now() - timedelta(days=40))
    )).all()
    return [((p or {}).get("topic"), at, (p or {}).get("sent") is not False) for at, p in rows if (p or {}).get("topic")]


async def _asked(session: AsyncSession, family_id: str, subject: str, now: datetime | None = None) -> dict[str, datetime]:
    """When each check-in topic was last asked (care events kind=checkin). One that did not go out counts only for a few
    hours, so it is tried again (live 2026-10-12: two check-ins were held back and then counted as asked for days)."""
    now = now or clock.now()
    out: dict[str, datetime] = {}
    for topic, at, sent in await _asked_rows(session, family_id, subject):
        if not sent and now - at >= RETRY_UNSENT:
            continue
        if topic not in out or at > out[topic]:
            out[topic] = at
    return out


async def went_out(session: AsyncSession, family_id: str, person: str, ref: str, since: datetime) -> bool:
    """Did a message for this check-in reach the person (sent by the brain, or by the fallback)?"""
    rows = (await session.execute(
        select(Turn.meta).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "assistant", Turn.at >= since)
    )).scalars()
    return any(str((m or {}).get("ref") or (m or {}).get("fallback_delivery") or "").startswith(ref) for m in rows)


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
               busy: bool = False, cared: bool = True, name: str = "") -> tuple[str, str] | None:
    """(topic, prompt) for the one check-in to send to this person now, or None. cared: the care record is about them
    (health check-ins too); a caregiver gets the personal ones only."""
    now = now or clock.now()
    if busy or not in_day(now):
        return None
    sent = await _proactive_today(session, family_id, subject)
    if len(sent) >= MAX_PER_DAY or (sent and now - max(sent) < MIN_GAP / 3):
        return None
    from app.learn import timing

    if await timing.quiet_reason(session, family_id, subject, "checkin"):
        return None  # sending would be held back: plan nothing rather than count a check-in that cannot go
    last_msg = await _last_user_message(session, family_id, subject)
    asked = await _asked(session, family_id, subject, now)
    day = clock.ist_day(now)

    # 1. A conversation just ended: today's unmarked schedule items, once a day per set.
    after_chat = last_msg is not None and AFTER_CHAT[0] <= now - last_msg <= AFTER_CHAT[1]
    talking = last_msg is not None and now - last_msg < AFTER_CHAT[0]
    evening = clock.ist(now).time() >= time(19, 0)  # the day's last look, if nothing was said about them
    if cared and not talking and (after_chat or (evening and await awake(session, family_id, subject, now))):
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

    # 2. A gentler question (health, or personal), one per window and a few a day, in a good window, not mid-conversation.
    win = window_of(now)
    if win is None or (last_msg and now - last_msg < AFTER_CHAT[1]) or (sent and now - max(sent) < MIN_GAP):
        return None
    if not await awake(session, family_id, subject, now):
        return None
    today = [at for t, at, went in await _asked_rows(session, family_id, subject) if went and gentle(t) and clock.ist_day(at) == day]
    if len(today) >= GENTLE_PER_DAY[cared] or any(window_of(at) == win for at in today):
        return None
    who = f"person {subject}" + (f" ({name})" if name else "")
    due_topics: list[tuple[timedelta, int, str, str]] = []  # (how overdue, rank on a tie: feelings before weight, topic, prompt)
    if not cared:
        return await _personal(session, family_id, subject, who, asked, now, due_topics)
    weight = await _last_vital(session, family_id, subject, "weight")
    if now - (weight.at if weight else datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY["weight"] and \
            now - asked.get("weight", datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY["weight"]:
        last = f"Their last weight on record is {weight.summary} from {clock.ist(weight.at).strftime('%d %b')}." if weight else "No weight is on record yet."
        overdue = now - asked.get("weight", now - EVERY["weight"]) - EVERY["weight"]
        if weight:  # a weight on record that is older than 20 days is overdue by that much
            overdue = max(overdue, now - max(weight.at, asked.get("weight", weight.at)) - EVERY["weight"])
        due_topics.append((overdue, 3, "weight",
                           f"[Check-in: weight] {last} Ask person {subject} once, lightly, whether they could check their weight "
                           f"today or tomorrow and tell you (no pressure; skip it if they have no scale). log_vital weight when "
                           f"they answer. send_message, then reply none."))
    for kind in await _readings_needed(session, family_id, subject):
        reading = await _last_vital(session, family_id, subject, kind)
        if (reading is None or now - reading.at >= EVERY["readings"]) and now - asked.get(f"reading:{kind}", datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY["readings"]:
            name = {"bp": "blood pressure", "sugar": "sugar"}[kind]
            last = f" Their last {name} on record: {reading.summary} ({clock.ist(reading.at).strftime('%d %b')})." if reading else ""
            due_topics.append((now - asked.get(f"reading:{kind}", now - EVERY["readings"]) - EVERY["readings"], 1, f"reading:{kind}",
                               f"[Check-in: {name}] Person {subject} has a condition where {name} matters.{last} Ask once, short, "
                               f"whether they checked their {name} today and what it was; if they have no machine, do not push. "
                               f"log_vital {kind} when they answer, and follow your red-flag rules. send_message, then reply none."))
    # A problem they told Saheli about (dizziness, pain) since the last time she asked how they are: ask about it first.
    told = await _symptoms_since(session, family_id, subject, asked.get("feelings"), now)
    said = "; ".join(clock.ist(e.at).strftime("%d %b %H:%M") + " " + e.summary for e in told[-3:])
    about = f" Since you last asked, they told you: {said}. Ask how that is now, first." if told else ""
    for topic, rank, text in (
        ("feelings", 0, f"[Check-in: how they are] Ask person {subject} one short, warm question in their language and script about how "
                        f"they are feeling today: any pain, trouble, sleep or worry.{about} Fit it to what you know of them (their "
                        f"conditions, the last things they told you, how they felt last time) and do not repeat a question from the last "
                        f"days. If they mention a problem later, log it (log_event symptom) and follow your red-flag rules; when they answer, "
                        f"note how they feel too (log_event mood with mood_score and mood_word). send_message, then reply none."),
        ("reports", 2, f"[Check-in: reports] Ask person {subject} once, short, whether they had any doctor visit, test or new report "
                       f"this week. If yes, ask them to send a photo of it here (you read it and show what you found before anything "
                       f"is saved). send_message, then reply none."),
    ):
        if told and topic == "feelings":
            due_topics.append((timedelta(days=365), rank, topic, text))
        elif now - asked.get(topic, datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY[topic]:
            due_topics.append((now - asked.get(topic, now - EVERY[topic]) - EVERY[topic], rank, topic, text))
    # An old fact (a medicine nobody has mentioned in 3 months, last year's doctor): check it is still right.
    from app.care import freshness

    facts = await store.facts(session, family_id, subject)
    old = await freshness.stalest(session, family_id, subject, facts, now=now)
    if old and now - asked.get(f"confirm:{old.key}", datetime.min.replace(tzinfo=now.tzinfo)) >= EVERY["confirm"]:
        due_topics.append((timedelta(days=1), 2, f"confirm:{old.key}",
                           f"[Check-in: still right?] The care record says about person {subject}: \"{old.text}\" [{old.key}], and "
                           f"nobody has confirmed it for a long time. Ask them once, short and natural, whether it is still the "
                           f"same. When they answer later: still right → fact_still_true {old.key}; changed → remember the new "
                           f"details; stopped → stop. send_message, then reply none."))
    return await _personal(session, family_id, subject, who, asked, now, due_topics)


async def _personal(session: AsyncSession, family_id: str, subject: str, who: str, asked: dict[str, datetime], now: datetime,
                    due_topics: list) -> tuple[str, str] | None:
    """Add the personal nudges that are due (a follow-up from memory, a getting-to-know question) and pick the most overdue."""
    never = datetime.min.replace(tzinfo=now.tzinfo)
    about: str | None = None
    if now - asked.get("follow_up", never) >= EVERY["follow_up"] and await _told_lately(session, family_id, subject, now):
        about = await about_them(session, family_id, subject, now)
        due_topics.append((now - asked.get("follow_up", now - EVERY["follow_up"]) - EVERY["follow_up"], 1, "follow_up",
                           f"[Check-in: following up] Something personal for {who}, from what they told you.\n{about}\nPick ONE thing "
                           "that deserves a follow-up now: a plan or event they mentioned, a goal they are keeping, something they "
                           "worried about or enjoyed, a person they talked about. Ask about it in one short, warm message with "
                           "send_message, the way a friend who remembers would. Not an order or a reminder (those have their own "
                           "follow-ups), nothing they already answered, nothing medical you are unsure of. When they answer, save "
                           "anything new worth keeping (remember, or a note). If nothing deserves a follow-up, reply none and send "
                           "nothing."))
    last_know = max((at for t, at in asked.items() if t.startswith("know:")), default=never)
    item = await next_to_know(session, family_id, subject, asked, now) if now - last_know >= EVERY["know"] else None
    if item:
        key, what, how, _ = item
        about = about if about is not None else await about_them(session, family_id, subject, now)
        due_topics.append((now - (last_know if last_know != never else now - EVERY["know"]) - EVERY["know"], 2, f"know:{key}",
                           f"[Check-in: getting to know them] You learn about each person a little at a time, so your later messages "
                           f"can be about their life. Ask {who} ONE light, natural question to learn {what}." + (f"\n{about}\n" if about else " ")
                           + "Tie it to something you know of them if you can. One question only, no list, no interview tone; simple "
                           f"words for an elder. When they answer, save it ({how}). If they would rather not say, accept it kindly. "
                           "send_message, then reply none."))
    if not due_topics:
        return None
    due_topics.sort(key=lambda x: (x[0], -x[1]), reverse=True)  # the most overdue first; on a tie, feelings before weight
    _, _, topic, prompt = due_topics[0]
    return topic, prompt


async def _symptoms_since(session: AsyncSession, family_id: str, subject: str, since: datetime | None, now: datetime) -> list[CareEvent]:
    """Problems they reported in the last two days that Saheli has not asked about since."""
    start = max(since or now - timedelta(days=2), now - timedelta(days=2))
    return list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject, CareEvent.kind == "symptom",
                                CareEvent.at > start).order_by(CareEvent.at)
    )).scalars())


STYLE = (" Do not start with the same greeting as your last message to them; vary your words. If the conversation shows they are "
         "unwell or upset, follow up on that instead.")


async def chat_open(host, family_id: str, person: str) -> bool:
    """WhatsApp lets Saheli start a chat only within 24 hours of the person's last message, and a check-in has no
    template: while their chat is closed none is planned (no brain call for a message that cannot go; their reminders
    still reach them as templates). Unknown (backend unreachable): try, the send gate decides."""
    try:
        res = await host.call("whatsapp_window", {"to": person}, family_id=family_id.removeprefix("shadow:"), subject_id=person, actor_id=person)
    except Exception:  # noqa: BLE001
        return True
    return bool((res or {}).get("open", True))


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
            people = await nudge_people(session, r)
        names = {m.get("id"): (m.get("name") or "").split(" ")[0] for m in [r.elder or {}, *(r.members or [])] if m.get("id")}
        for subject, cared in people:
            stats["people"] += 1
            try:
                async with sessions() as session:
                    live = await runtime.live_tasks(session, r.family_id)
                    got = await plan(session, host_for(r.family_id), r.family_id, subject, now=now, busy=moving_for(live, subject, now),
                                     cared=cared, name=names.get(subject, ""))
                    if not got or not await chat_open(host_for(r.family_id), r.family_id, subject):
                        continue
                    topic, prompt = got
                    # counted before the brain runs, so a crash cannot make it repeat; marked "not sent" below if nothing went
                    ev_id = await store.record_event(session, family_id=r.family_id, subject_id=subject, kind="checkin",
                                                     summary=f"Saheli checked in: {topic.split(':')[0]}", payload={"topic": topic})
                    await session.commit()
                ref = f"checkin:{topic}:{subject}:{clock.ist_day(now)}"
                started = clock.now()
                await system_turn(sessions, host_for(r.family_id), r.family_id, prompt + STYLE, ref, deliver_to=subject)
                async with sessions() as session:
                    if await went_out(session, r.family_id, subject, ref, started - timedelta(seconds=5)):
                        stats["sent"] += 1
                    elif ev_id is not None and (row := await session.get(CareEvent, ev_id)) is not None:
                        row.payload = {**(row.payload or {}), "sent": False}
                        await session.commit()
            except Exception:  # noqa: BLE001 — one person's problem must not stop the others
                logger.exception("check-in failed family=%s", r.family_id)
    return stats


async def nudge_people(session: AsyncSession, roster: FamilyRoster) -> list[tuple[str, bool]]:
    """Everyone who gets nudges, with whether the care record is about them: the people cared for, then each caregiver or
    family member who has written to Saheli in the last 30 days."""
    cared = await cared_for(session, roster)
    out = [(p, True) for p in cared]
    since = clock.now() - timedelta(days=30)
    for m in roster.members or []:
        mid, role = m.get("id"), str(m.get("role") or "").lower()
        if not mid or mid in cared or not any(r in role for r in NUDGED_ROLES):
            continue
        wrote = await session.scalar(select(func.count()).select_from(Turn).where(
            Turn.family_id == roster.family_id, Turn.thread_id == mid, Turn.role == "user", Turn.at >= since))
        if wrote:
            out.append((mid, False))
    return out


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
