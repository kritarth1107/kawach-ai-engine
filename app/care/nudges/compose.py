"""What a nudge may say, the one worker call that writes it, and the checks before it is sent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import NudgeLedger, Turn
from app.core import clock

logger = logging.getLogger(__name__)

from app.care.nudges.ledger import (
    COLLECT_EVERY, DOSES_AFTER_MIN, HABIT_SHADOW, MAX_LEN, PRIORITY,
    SENT, SLOTS, _OPTIONAL_CHAT, _day_start, _iso, _log, _when, asked_open,
    learned, sent_lines, sent_today,
)
from app.care.nudges.planner import Clock, Person, _LIGHT_WHY

# ── what to say: context and options (structured, no reading of people's words) ───────────────────────────────────


@dataclass
class Option:
    id: str
    kind: str
    why: str
    about_ref: str | None = None
    field: str | None = None
    score: float = 0.0


@dataclass
class Context:
    person: Person
    text: str  # what the composer sees
    known: str  # what any number, name or time in the message must come from
    recent: list[str]  # the last messages either way (newest last)
    saheli_recent: list[str]  # Saheli's own recent lines, newest last (for the opening and repeat checks)
    profile: dict | None
    options: list[Option]
    max_len: int
    expects_reply: bool = True
    questions: bool = True  # false after two unanswered: a warm line, and a question mark is rejected
    ladder: int = 0
    cost: float = 0.0  # INR spent writing and checking it (every try), logged on its ledger row
    avoid: list[str] = field(default_factory=list)  # names they asked never to be called (trust plan A3)
    never: list[str] = field(default_factory=list)  # what the family said is wrong (rejected facts), never used or asked
    memory: str = ""  # what the no-invention check counts as known (not Saheli's own earlier lines)
    theirs: list[str] = field(default_factory=list)  # what they wrote lately
    sent_texts: list[str] = field(default_factory=list)  # lines sent to them in the last few days: do not send one again
    medicines: list[str] = field(default_factory=list)  # names on their record: a regular nudge may not ask about them
    today_asks: list[str] = field(default_factory=list)  # questions we already asked today and they have not answered


def _medicine_reminder(reminder: dict) -> bool:
    return str(reminder.get("aboutKey") or "").startswith("medicine:")


def medicine_names(facts, reminders: list[dict] | None = None) -> list[str]:
    """Names and doses saved on the care record, and the name of a reminder linked to a medicine. Not a list of drugs:
    only what this person's record already holds."""
    names: list[str] = []
    for f in facts:
        if getattr(f, "domain", None) != "medicine":
            continue
        value = f.value or {}
        for key in ("name", "dose"):
            got = str(value.get(key) or "").strip()
            if len(got) >= 3:
                names.append(got)
        slug = str(getattr(f, "key", "") or "").split(":", 1)[-1].replace("_", " ").strip()
        if len(slug) >= 4:
            names.append(slug)
    for r in reminders or []:
        if not _medicine_reminder(r):
            continue
        slug = str(r.get("aboutKey") or "").split(":", 1)[-1].replace("_", " ").strip()
        if len(slug) >= 4:
            names.append(slug)
    return list(dict.fromkeys(names))


def _mentions(text: str, name: str) -> bool:
    hay, needle = text.casefold(), name.casefold()
    if len(needle) < 3 or needle not in hay:
        return False
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return False
        before = hay[i - 1] if i else " "
        after = hay[i + len(needle)] if i + len(needle) < len(hay) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        start = i + 1


def near_dose(times, now: datetime, *, before: int = 45, after: int = 30) -> bool:
    """True when now is close to one of this medicine's dose times (45 minutes before, 30 after)."""
    m = clock.ist(now).hour * 60 + clock.ist(now).minute
    for t in times or []:
        hm = _hm(t)
        if hm is None:
            continue
        delta = (m - hm + 12 * 60) % (24 * 60) - 12 * 60
        if -before <= delta <= after:
            return True
    return False


async def stock_ask(session: AsyncSession, person: Person, facts: list, now: datetime) -> tuple[str, str, str, str] | None:
    """One medicine with no count, at a quiet hour that is not its dose time. One a day. Two sent asks with no reply
    and that medicine is left alone. The answer is a count, never a dose."""
    from app.care import features

    today = clock.ist_day(now)
    sent = (await session.execute(select(NudgeLedger.field, NudgeLedger.replied_at, NudgeLedger.sent_at).where(
        NudgeLedger.family_id == person.family_id, NudgeLedger.person_id == person.id, NudgeLedger.status == "sent",
        NudgeLedger.field.like("stock:%")))).all()
    if any(at and clock.ist_day(at) == today for _field, _replied, at in sent):
        return None
    silent: dict[str, int] = {}
    for asked, replied, _at in sent:
        if asked and replied is None:
            silent[asked] = silent.get(asked, 0) + 1
    known = {r["key"] for r in await features.stock(session, person.family_id, person.id) if r.get("stock") is not None}
    meds = [f for f in facts if getattr(f, "domain", None) == "medicine" and getattr(f, "status", "active") == "active"
            and not str(f.key).endswith(":none") and f.key not in known]
    for f in sorted(meds, key=lambda x: x.key):
        fld = f"stock:{f.key}"
        if silent.get(fld, 0) >= 2:
            continue
        times = [t for t in ((f.value or {}).get("times") or []) if isinstance(t, str)]
        if near_dose(times, now):
            continue
        name = (f.value or {}).get("name") or str(f.key).split(":", 1)[-1].replace("_", " ")
        why = (f"how many tablets of {name} are left in the strip: one question, only the count; "
               "do not mention the dose, the time, or whether they took it")
        return "stock", why, fld, fld
    return None


def _hm(v) -> int | None:
    try:
        h, m = str(v).strip()[:5].split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


def _said(t: str) -> str:
    return _norm(re.sub(r"(?i)^reminder:\s*", "", t or ""))


def _repeated(text: str, earlier: list[str]) -> str | None:
    """The earlier line this repeats, word for word or nearly, whatever they answered."""
    from app.brain import guards

    n = _said(text)
    if len(n) < 8:
        return None
    for old in earlier:
        o = _said(old)
        if not o:
            continue
        if n == o or guards.similar(text, old) >= 0.85:
            return old
    return None


async def last_messages(session: AsyncSession, family_id: str, person: str, gate: dict, n: int = 5) -> list[tuple[datetime, str, str]]:
    """The last n messages either way in their own thread: engine turns plus what the backend sent them itself (a reminder,
    the morning schedule), oldest first, a message both sides recorded only once."""
    turns = [t for t in await store.recent_turns(session, family_id, person, limit=20) if (t.text or "").strip() and t.text.strip().lower() != "none"]
    rows = [(t.at, "them" if t.role == "user" else "Saheli", t.text.strip()) for t in turns]
    for o in gate.get("last_outbound_preview") or []:
        at = _iso(o.get("at"))
        text = str(o.get("text") or "").strip()
        if not at or not text:
            continue
        key = _norm(text)[:60]
        if any(who == "Saheli" and _norm(t)[:60] == key and abs((at - when).total_seconds()) < 180 for when, who, t in rows):
            continue
        rows.append((at, "Saheli", text))
    rows.sort(key=lambda r: r[0])
    return rows[-n:]


async def build_context(session: AsyncSession, host, p: Person, ck: Clock, now: datetime, *, early_bird: bool = False,
                        only: list[Option] | None = None, extra: str = "", light: str = "",
                        hold_chat: bool = False, quiet_why: str = "") -> Context:
    from app.brain.loop import writing_profiles
    from app.care import activities, checkins, language, naming
    from app.care import mood as care_mood
    from app.care import rejected as rejected_facts

    fid = p.family_id
    facts = [f for f in await store.facts(session, fid, p.id) if f.status == "active"]
    by_domain: dict[str, list] = {}
    for f in facts:
        by_domain.setdefault(f.domain, []).append(f)
    speech = next((language.normalise(f.value) for f in by_domain.get("language", []) if isinstance(f.value, dict)), None)
    names = await naming.for_person(session, fid, p.id)
    wrong = await rejected_facts.rejected(session, fid, p.id, now=now)
    local = clock.ist(now)
    recent = await last_messages(session, fid, p.id, ck.gate)
    nudges = [r for r in ck.rows if r.status in SENT][:10]

    acts = await activities.status(session, fid, p.id)
    schedule: dict = {}
    if p.cared:
        try:
            schedule = await host.call("get_today_schedule", {}, family_id=fid.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id) or {}
        except Exception:  # noqa: BLE001
            schedule = {}
    try:
        rems = (await host.call("list_reminders", {}, family_id=fid.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id) or {}).get("reminders") or []
    except Exception:  # noqa: BLE001
        rems = []

    open_fields, today_asks = await asked_open(session, p.id, now)
    options = only if only is not None else await candidates(session, p, ck, now, facts=facts, acts=acts, schedule=schedule, reminders=rems,
                                                              early_bird=early_bird, open_fields=open_fields, light=light,
                                                              hold_chat=hold_chat, quiet_why=quiet_why)
    options = sorted(options, key=lambda o: -o.score)[:5]

    m_now = local.hour * 60 + local.minute
    coming = []
    for i in (schedule.get("items") or []):
        t = _hm(i.get("time"))
        if t is not None and t > m_now and i.get("status") in ("upcoming", "due", None):
            coming.append(f"{i.get('time')} {i.get('title')} (its reminder goes on its own: do not remind or advise about it)")
    for r in rems:
        if _medicine_reminder(r):
            continue  # a dose reminder goes at its own time, never as "coming up" for a regular nudge
        for t in r.get("times") or []:
            if (_hm(t) or -1) > m_now:
                coming.append(f"{t} reminder: {r.get('text')}")
    for a in acts:
        if a["today"] == "upcoming" and a.get("start"):
            coming.append(f"{a['start']} {a['label']}" + (f" (streak {a['streak']})" if a["streak"] > 1 else ""))
    for f in by_domain.get("appointment", []):
        when = _iso((f.value or {}).get("when"))
        if when and now <= when <= now + timedelta(hours=24):
            coming.append(f"appointment: {f.text}")
    mood = await care_mood.last(session, fid, p.id, since=now - timedelta(hours=24))
    symptoms = await checkins._symptoms_since(session, fid, p.id, ck.last_out, now)
    known_facts = [f.text for f in facts if f.domain not in ("language", "learned", "medicine")][:24]
    notes = [n.body_md.strip()[:400] for n in (await store.notes(session, fid, [p.id]))[:2] if (n.body_md or "").strip()]
    max_len = MAX_LEN[p.elder]
    lines = [
        f"PERSON: {p.name or 'them'}" + (f" (call them {names['name']})" if names["name"] else "")
        + f"; {'the person cared for' if p.elder else ('caring for themselves' if p.cared else p.role or 'family member')}",
        *([f"NEVER CALL THEM: {', '.join(names['avoid'])}"] if names["avoid"] else []),
        *([rejected_facts.block(wrong)] if wrong else []),
        f"WRITE IN: {language.write_in(speech) if speech else 'the language and script they write to you in (their messages below); English if none'}",
        f"NOW: {local.strftime('%A %d %B, %H:%M')} IST",
        "LAST MESSAGES (oldest first):\n" + ("\n".join(f"  {clock.ist(at).strftime('%d %b %H:%M')} {who}: {t[:220]}" for at, who, t in recent) or "  (none yet)"),
        "WHAT YOU KNOW ABOUT THEM:\n" + ("\n".join(f"  - {t[:180]}" for t in known_facts) or "  (little yet)"),
        *(["NOTES:\n" + "\n".join(f"  - {n}" for n in notes)] if notes else []),
        "COMING UP (next 24 h):\n" + ("\n".join(f"  - {c}" for c in coming[:8]) or "  (nothing on record)"),
        *([f"MOOD TODAY: {mood.summary}"] if mood else []),
        *(["SINCE YOU LAST WROTE THEY MENTIONED: " + "; ".join(e.summary for e in symptoms[-3:])] if symptoms else []),
        "YOUR LAST NUDGES (do not repeat their kind or opening):\n"
        + ("\n".join(f"  - {r.kind}: {r.text[:120]}{' (answered)' if r.replied_at else ''}" for r in nudges) or "  (none)"),
        "OPTIONS (pick one id):\n" + "\n".join(f"  - {o.id} [{o.kind}] {o.why}" + (f" (field: {o.field})" if o.field else "") for o in options),
        f"LIMITS: at most {max_len} characters; at most one question"
        + ("; no question at all" if light == "warm" else "")
        + ("; expects_reply false: no question they must answer, a soft 'whenever you're up' tone" if early_bird else "")
        + ("; one easy question, not a dose, and not a question already asked today" if light and light != "warm" else ""),
    ]
    if extra:
        lines.append(extra)
    profiles = await writing_profiles(session, fid, [{"id": p.id}])
    earlier = list(dict.fromkeys([*await sent_lines(session, fid, p.id, now - timedelta(days=3)),
                                  *[r.text for r in await sent_today(session, p.id, now - timedelta(days=3)) if r.text]]))
    saheli = [t for _, who, t in recent if who == "Saheli"] + [r.text for r in reversed(nudges) if r.text] + earlier[:8]
    known = "\n".join([*known_facts, *notes, *coming, *(t for _, _, t in recent), *(o.why for o in options), local.strftime("%A %d %B %H:%M")])
    memory = "\n".join([*known_facts, *notes, *coming, *(o.why for o in options), *([mood.summary] if mood else []),
                        *(e.summary for e in symptoms[-3:]), f"Now: {local.strftime('%A %d %B %H:%M')} IST"])
    return Context(person=p, text="\n".join(lines), known=known, recent=[t for _, _, t in recent], saheli_recent=saheli[-15:],
                   profile=profiles.get(p.id), options=options, max_len=max_len,
                   expects_reply=not early_bird and light != "warm", questions=light != "warm", ladder=ck.run,
                   avoid=names["avoid"], never=rejected_facts.terms(wrong), memory=memory,
                   theirs=[t for _, who, t in recent if who == "them"], sent_texts=earlier,
                   medicines=medicine_names(facts, rems), today_asks=today_asks)


async def candidates(session: AsyncSession, p: Person, ck: Clock, now: datetime, *, facts: list, acts: list[dict], schedule: dict,
                     reminders: list[dict], early_bird: bool = False, open_fields: set[str] | None = None, light: str = "",
                     hold_chat: bool = False, quiet_why: str = "") -> list[Option]:
    """What is worth writing about now (the plan's §3.4), each scored by how due it is and how much it helps, minus a
    penalty for the same kind among the last two nudges and for a field asked within its cooldown."""
    from app.care import checkins

    fid = p.family_id
    local = clock.ist(now)
    m_now = local.hour * 60 + local.minute
    rows = [r for r in ck.rows if r.status in SENT]
    last_kinds = [r.kind for r in rows[:2]]
    asked_field = {}
    for r in rows:
        if r.field and r.field not in asked_field:
            asked_field[r.field] = _when(r)
    for topic, at in (await checkins._asked(session, fid, p.id, now)).items():  # what the old check-ins asked counts too
        fld = {"follow_up": "follow_up"}.get(topic, topic.split(":")[0] if topic.startswith("doses:") else topic)
        if fld not in asked_field or at > asked_field[fld]:
            asked_field[fld] = at
    today = clock.ist_day(now)
    out: list[Option] = []
    lr = await learned(session, fid, p.id, now)

    def add(kind: str, why: str, *, about: str | None = None, fld: str | None = None, boost: float = 0.0, cooldown: timedelta | None = None):
        about, fld = short_ref(about), short_ref(fld)
        if fld and cooldown and fld in asked_field and now - asked_field[fld] < cooldown:
            return
        if about and any(r.about_ref == about and clock.ist_day(_when(r)) == today for r in rows):
            return
        if (fld and fld in lr.unwelcome) or kind in lr.unwelcome:
            return  # they did not want this asked: it rests for 30 days
        if fld and open_fields and fld in open_fields and not light:
            return  # asked today, in this family or another, and they have not answered
        score = PRIORITY.get(kind, 3) + boost - (3 if kind in last_kinds else 0) + lr.weight(kind)
        out.append(Option(id=f"o{len(out) + 1}", kind=kind, why=why, about_ref=about, field=fld, score=score))

    if light == "warm":
        day = clock.ist(now).strftime("%Y-%m-%d")
        add("warm", _LIGHT_WHY["warm"], fld="warm", about=f"light:warm:{day}")
        return out
    if light:
        day = clock.ist(now).strftime("%Y-%m-%d")
        add("follow_up", _LIGHT_WHY[light], fld="hello" if light == "opener" else f"gentle:{light}", about=f"light:{light}:{day}")
        return out
    if early_bird:
        plan = [f"{a['start']} {a['label']}" for a in acts if a["today"] == "upcoming" and a.get("start")][:3]
        add("early_bird", "a soft no-rush good-morning for when they wake" + (f"; today: {', '.join(plan)}" if plan else ""), boost=2)
        return out

    if p.cared:
        # Only items at least 2 hours past their time (the backend's own missed-dose follow-up has asked by then), once a
        # day, naming each item and its time (live 2026-10-13: an evening "your morning medications?" fit anyone).
        due_items = [i for i in checkins.unmarked_due(schedule, now) if (_hm(i.get("time")) or 24 * 60) <= m_now - DOSES_AFTER_MIN]
        if due_items:
            ids = ",".join(sorted(str(i.get("scheduleId")) for i in due_items))
            listed = "; ".join(f"{i.get('title')} at {i.get('time')}" for i in due_items[:6])
            add("doses", f"today's items past their time that nobody marked: {listed} (ask once whether they were done, naming "
                "each item and its time as listed here, never 'your medicines' in general)",
                about=f"doses:{ids}", fld="doses", boost=2, cooldown=timedelta(hours=20))
    # A medicine reminder is not an option: it goes at its own time (a nudge once asked if a dose had been taken).
    for r in reminders:
        if _medicine_reminder(r):
            continue
        for t in r.get("times") or []:
            mins = _hm(t)
            if mins is not None and 0 < mins - m_now <= 240:
                add("upcoming", f"their reminder at {t}: {r.get('text')}", about=f"reminder:{r.get('reminderId')}:{t}")
    for a in acts:
        if a["kind"] in ("habit", "quit"):
            if a["today"] == "upcoming" and a.get("start") and 0 < (_hm(a["start"]) or 0) - m_now <= 240:
                add("upcoming", f"{a['label']} at {a['start']} today" + (f", streak {a['streak']}" if a["streak"] > 1 else ""),
                    about=a["key"], fld=f"activity:{a['type']}:plan", cooldown=timedelta(hours=20))
            elif a["today"] in ("not_heard",) or a["kind"] == "quit":
                add("habit", f"{a['text']}: today {a['today'].replace('_', ' ')}" + (f", streak {a['streak']}" if a["streak"] > 1 else ""),
                    about=a["key"], fld=f"activity:{a['type']}:done", cooldown=timedelta(hours=20), boost=1 if a["streak"] > 1 else 0)
    for f in facts:
        if f.domain == "appointment":
            when = _iso((f.value or {}).get("when"))
            if when and timedelta(0) < when - now <= timedelta(hours=30):
                add("upcoming", f"appointment: {f.text}", about=f.key, boost=1)
            elif when and timedelta(0) < now - when <= timedelta(days=2):
                add("follow_up", f"how the appointment went: {f.text}", about=f.key, boost=1)
    if await checkins._told_lately(session, fid, p.id, now):
        add("follow_up", "something they told you in the last few days (a plan, a worry, a person, an event) that deserves a follow-up now; "
            "see LAST MESSAGES and WHAT YOU KNOW", fld="follow_up", cooldown=timedelta(hours=10))
    for lp in (await store.live_loops(session, fid, [p.id]))[:3]:
        if lp.wake_at is None or lp.wake_at - now > timedelta(hours=6):
            add("follow_up", f"an open thread: {lp.title}", about=f"loop:{lp.id}")
    if p.cared:
        symptoms = await checkins._symptoms_since(session, fid, p.id, None, now)
        if symptoms:
            add("collect", f"how {symptoms[-1].summary} is now", about=f"symptom:{symptoms[-1].id}", fld="feelings", boost=2)
        # A data question only with something of theirs to tie it to (a bare "how did you sleep?" fits anyone)
        anchor = personal_anchor(facts, acts, now)
        if anchor:
            add("collect", f"how they are feeling today, tied to {anchor}", fld="feelings", cooldown=COLLECT_EVERY["feelings"])
        # (no standalone sleep question: written from a bedtime it still came out as "how did you sleep?", which fits anyone)
        for kind in await checkins._readings_needed(session, fid, p.id):
            add("collect", f"today's {'blood pressure' if kind == 'bp' else 'sugar'} reading, asked as ONE question (\"what did it show "
                f"today?\"), if they checked it", fld=f"reading:{kind}", cooldown=COLLECT_EVERY["readings"])
        weight = await checkins._last_vital(session, fid, p.id, "weight")
        if weight is None or now - weight.at >= COLLECT_EVERY["weight"]:
            add("collect", "their weight, whenever they next check it (no pressure)", fld="weight", cooldown=COLLECT_EVERY["weight"])
        add("collect", "any doctor visit, test or new report this week", fld="reports", cooldown=COLLECT_EVERY["reports"])
    item = await checkins.next_to_know(session, fid, p.id, {f: at for f, at in asked_field.items() if f.startswith("know:")}, now)
    wrote_today = await session.scalar(select(Turn.id).where(
        Turn.thread_id == p.id, Turn.role == "user", Turn.at >= _day_start(now), ~Turn.family_id.startswith("shadow:")).limit(1))
    if item and wrote_today:  # never a cold opener: only after they have written today, tied to what they said
        add("know", f"one light question to learn {item[1]}, tied to something they said today (save the answer: {item[2]})",
            fld=f"know:{item[0]}", cooldown=checkins.KNOW_AGAIN)
    for name, start, end in SLOTS:
        if start <= m_now < end:
            done = any(r.kind == "reflect" and r.field == f"reflect:{name}" and clock.ist_day(_when(r)) == today for r in rows)
            if not done:
                what = {"morning": "their plan for the morning", "afternoon": "how their energy is this afternoon",
                        "evening": "the best part of their day"}[name]
                add("reflect", what, fld=f"reflect:{name}")
    likes = [f.text for f in facts if f.domain in ("preference", "dish", "occasion")][:4]
    if likes:
        add("delight", "something light from what they enjoy: " + "; ".join(likes), fld="delight", cooldown=timedelta(hours=8))
    from app.care import meal_nudges

    roster = await store.roster(session, fid)
    elder = (roster.elder or {}) if roster else None
    meals_mode = str(ck.gate.get("meals_mode") or os.getenv("NUDGE_MEALS") or "off").lower()
    for m in await meal_nudges.options(session, p, ck.rows, now, acts=acts, elder=elder, meals_mode=meals_mode):
        add(m["kind"], m["why"], about=m["about"], fld=m["fld"])
    g = await meal_nudges.goal_option(session, p, ck.rows, now, meals_mode=meals_mode)
    if g:
        add(g["kind"], g["why"], about=g["about"], fld=g["fld"])
    ask = await stock_ask(session, p, facts, now)
    if ask and not light and not early_bird:
        add(ask[0], ask[1], about=ask[2], fld=ask[3])
    if hold_chat:
        out = [o for o in out if not (o.kind in _OPTIONAL_CHAT and not (o.kind == "follow_up" and "appointment" in (o.about_ref or "")))]
        already = any(r.kind == "checkin" for r in await sent_today(session, p.id, now))
        if not quiet_why and not already and not light:
            add("checkin", "one easy check-in, not a stack of questions: they can answer when they are free", fld="checkin")
    return out


def short_ref(ref: str | None, limit: int = 160) -> str | None:
    """A reference that fits its ledger column (live 2026-10-13: "doses:<every unmarked schedule id>" overflowed 200
    characters and that nudge was lost). A long one keeps its start and a hash, so the same set compares equal."""
    if ref is None or len(ref) <= limit:
        return ref
    return f"{ref[:limit - 17]}#{hashlib.sha1(ref.encode()).hexdigest()[:16]}"


def personal_anchor(facts: list, acts: list[dict], now: datetime) -> str:
    """Something of theirs to tie a data question to: what they did in the last two days, else a condition they live with."""
    recent = [a for a in acts if a.get("last") and a["last"].get("done") and (_iso(a["last"].get("at")) or now) >= now - timedelta(days=2)]
    if recent:
        return f"their {recent[0]['label'].lower()} ({recent[0]['last']['summary'][:60]})"
    cond = next((f.text for f in facts if f.domain == "condition"), None)
    return f"what they live with: {cond[:80]}" if cond else ""


# ── write, check, send ────────────────────────────────────────────────────────────────────────────────────────────

COMPOSE = """You write ONE short WhatsApp message from Saheli, a warm care companion, to one person after a quiet spell. You keep
the conversation alive the way a caring friend who remembers things would.

You get who they are, the time, the last messages between you, what you know about them, what is coming up, your last
nudges, and OPTIONS (things worth writing about now). Pick ONE option and write the message for it.

Rules:
- Personal: it names something real from what you were given (their plan, habit, something they said, something coming
  up, the people or things they love). A message that would fit anyone ("How are you today?", "How did you sleep?") is not
  allowed: even a data question about feelings or a reading carries something of theirs, and the personal detail named in
  the option you pick goes into the message itself (yesterday's walk, their knee, what they told you).
- Short: one or two lines within the character limit. One question at most, and one means one (not two joined, not a
  question plus "tell me"). No lists, no lecture.
- The name: exactly the one in PERSON ("call them …"), written in their script, or no name at all; never a different
  word for them.
- Medicines: never remind them of a dose, a time or how to take it, and never ask whether they took one, even if they
  just set that reminder or it is in the last messages. Their reminders do that. Only a "doses" option may ask whether
  today's past items were done. A "stock" option asks only how many of that one medicine are left. Name it. Do not
  mention the dose, the time, or whether they took it.
- Fresh: a different opening and angle from the last messages and your last nudges. Never repeat a line they already got
  in the last few days, never ask a question already asked today, and never contradict the last messages; if the last
  chat ended on an open thread, pick it up.
- Their language, script and dialect exactly as WRITE IN says. English stays English. Never switch language. The name
  they like; emoji only if their messages use them. Everyday words, the way a person texts now: no "Reminder:" label,
  no poster line, no slogan.
- Safe: only what you were given (never invent a number, name, time, event or reading, and never put a day or time on
  something unless the context gives it: not "yesterday's gym" when it does not say yesterday); no medical advice; nothing
  alarming; nothing about another person that they would not want shared. Warm, never abrupt.
- Do not greet by name at the start, and no "Reminder:" label.
Return only JSON: {"option_id": "...", "text": "...", "field": "<the data your question asks for, from the option, or null>",
"expects_reply": true}

Good (generic examples; write in their language): "Gym at 7 today — legs or cardio? 💪" · "How did the call with the bank go?" ·
"Day 4 without sweets! What helped most today?"
Bad: "Hello! How are you today?" (not personal) · "Take your medicines, drink water and walk." (a list, a lecture) ·
"Your BP was 150 yesterday, that is dangerous." (invented, alarming)"""


def parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def compose(ctx: Context, problems: list[str] | None = None) -> dict:
    from app.llm import router

    text = ctx.text + (("\n\nYour last try had these problems; write it again without them:\n- " + "\n- ".join(problems)) if problems else "")
    reply = await router.complete("worker", system_stable=COMPOSE, effort="low", max_tokens=1500, timeout_s=40,
                                  messages=[{"role": "user", "content": [{"type": "text", "text": text}]}])
    ctx.cost += _cost(reply)
    return parse(reply.text or "")


def _cost(reply) -> float:
    from app.llm import spend

    return spend.cost_inr(reply.model or "", reply.usage or {})


def check(draft: dict, ctx: Context) -> list[str]:
    """What is wrong with a written nudge before it goes (the plan's §3.6); empty means it may go."""
    from app.brain import guards

    text = str(draft.get("text") or "").strip()
    if not text:
        return ["no message was written"]
    problems = []
    if draft.get("option_id") not in {o.id for o in ctx.options}:
        problems.append("option_id must be one of the OPTIONS ids")
    if len(text) > int(ctx.max_len * 1.15):
        problems.append(f"too long: {len(text)} characters, the limit is {ctx.max_len}")
    marks = text.count("?") + text.count("？")
    if not ctx.questions and marks:
        problems.append("no question until they write")
    elif marks > 1:
        problems.append("more than one question")
    if _said(text) == "reminder" or text.lower().lstrip().startswith("reminder:"):
        problems.append('no "Reminder:" label; write it the way a person would text')
    again = _repeated(text, ctx.sent_texts)
    if again:
        problems.append(f"this line already went to them in the last few days: \"{again[:80]}\"")
    dup = guards.duplicate_message(text, [(t, False) for t in ctx.saheli_recent], near=0.6)
    if dup:
        problems.append(f"too close to an earlier message: \"{dup[:80]}\"")
    asked = guards.question_again(text, ctx.today_asks)
    if asked:
        problems.append(f"this question was already asked today: \"{asked[:80]}\"")
    first = " ".join(_norm(text).split()[:3])
    if first and any(" ".join(_norm(t).split()[:3]) == first for t in ctx.saheli_recent[-3:]):
        problems.append("opens the same way as one of your last messages")
    problems += [x for x in guards.language_problems(text, ctx.profile, who="they") if not x.startswith("do not mix scripts")]
    local = clock.ist()
    problems += guards.ungrounded(text, known=ctx.known, now_minutes=local.hour * 60 + local.minute)
    from app.care import naming

    for name in naming.uses_avoided(text, ctx.avoid):
        problems.append(f"they asked never to be called \"{name}\": use the name in PERSON or none")
    chosen = next((o for o in ctx.options if o.id == draft.get("option_id")), None)
    if (chosen is None or chosen.kind not in ("doses", "stock")) and any(_mentions(text, name) for name in ctx.medicines):
        problems.append("a regular nudge does not ask about a medicine or its dose; their reminders do that")
    low = text.lower()
    for term in ctx.never:
        if term.lower() in low:
            problems.append(f"\"{term}\" is something the family said is wrong: never use or ask about it")
    return problems


async def grounded(ctx: Context, text: str) -> list[str]:
    """The shared no-invention check (trust plan C4): every specific in the nudge must come from what Saheli knows or what
    they wrote, never from her own earlier lines; her own health advice is a problem too."""
    from app.brain import grounding

    return grounding.problems(await grounding.check(text, memory=ctx.memory, their_messages=ctx.theirs, tool_results=[]))



VERIFY = """You check one WhatsApp message Saheli is about to send, against the context she had. Reply with problems only for:
- invented: it states anything not in the context, including saying something already happened when the context does not
  say so (a walk "done" that is still ahead, "yesterday's" anything without a date), or a number, name or time not given;
- medical: it tells them what to take, stop or change, or how to take a medicine, or it asks whether they took a dose
  or reminds them of a dose or its time (their own reminders do that; only a doses option may ask about an unmarked item;
  a stock option asking only how many are left is not a dose reminder);
- two_questions: it asks more than one thing (two questions joined, or a question plus "tell me");
- not_personal: nothing in it is specific to this person (it would fit anyone).
Return only JSON: {"ok": true|false, "problems": ["invented: ...", ...]}"""


async def verify(ctx: Context, text: str) -> list[str]:
    """A fast second look by a model at what code cannot see (an event said to have happened, advice); fails open."""
    from app.llm import router

    try:
        rep = await router.complete("judge_fast", system_stable=VERIFY, effort="low", max_tokens=800, timeout_s=30, messages=[
            {"role": "user", "content": [{"type": "text", "text": f"CONTEXT:\n{ctx.text}\n\nMESSAGE:\n{text}"}]}])
    except Exception:  # noqa: BLE001 — no second look available: the code checks stand
        return []
    ctx.cost += _cost(rep)
    got = parse(rep.text or "")
    return [] if got.get("ok", True) else [str(x)[:160] for x in (got.get("problems") or ["the reviewer found a problem"])][:4]


async def write_and_check(ctx: Context) -> tuple[dict | None, list[str]]:
    problems: list[str] = []
    for _ in range(3):  # the first try and up to 2 more (the plan's §3.5)
        try:
            draft = await compose(ctx, problems or None)
        except Exception as exc:  # noqa: BLE001
            return None, [f"compose_failed:{type(exc).__name__}"]
        text = str(draft.get("text") or "")
        problems = check(draft, ctx) or await grounded(ctx, text) or await verify(ctx, text)
        if not problems:
            return draft, []
    return None, problems


async def send(sessions: async_sessionmaker, host, p: Person, ctx: Context, draft: dict, *, kind_default: str, mode: str,
               now: datetime, run: int, about: str | None = None, recheck=None, meals_mode: str | None = None) -> str:
    """Send (or, in shadow mode, only log) a checked nudge. The person's clock is read again under the family's lock right
    before it goes: if they wrote or anything else went out meanwhile, it is dropped (never mid-chat)."""
    from app.brain import guards
    from app.brain.loop import _lock_family
    from app.learn import scoring

    opt = next((o for o in ctx.options if o.id == draft.get("option_id")), ctx.options[0] if ctx.options else None)
    kind = opt.kind if opt else kind_default
    # The option's own field, never the writer's words for it (live 2026-10-13: "whether he ended up ordering pizza",
    # "cigarettes_smoked_today"): cooldowns, what was unwelcome and which tool saves the answer all key on it.
    fld = opt.field if opt else None
    about = about or (opt.about_ref if opt else None)
    text = str(draft.get("text") or "").strip()
    if kind in HABIT_SHADOW and meals_mode != "on":
        mode = "shadow"  # meal, water, sleep and goal asks share nudge.meals: in shadow they are written and logged, never sent
    async with sessions() as session:
        await _lock_family(session, p.family_id)
        if recheck is not None and not await recheck(session):
            await _log(session, p, now, kind=kind, status="skipped", reason="talking", text=text, fld=fld, about=about, run=run, cost=ctx.cost,
                       context=ctx.text)
            await session.commit()
            return "talking"
        greeted, mid = guards.opening_state(await store.recent_turns(session, p.family_id, p.id, limit=12), clock.now(), day_of=clock.ist_day)
        text = guards.tidy_opening(text, greeted_today=greeted, mid_conversation=mid)
        if mode != "on":
            await _log(session, p, now, kind=kind, status="shadow", text=text, fld=fld, about=about, run=run, cost=ctx.cost,
                       context=ctx.text)
            await session.commit()
            return "shadow"
        row = await _log(session, p, now, kind=kind, status="planned", text=text, fld=fld, about=about, run=run, cost=ctx.cost,
                       context=ctx.text)
        if row is None:
            await session.commit()
            return "duplicate"
        saved = ((ctx.profile or {}).get("saved") or {})
        opener = fld == "hello"
        payload: dict = {"to": p.id, "text": text, "kind": "nudge", "ref": row.ref}
        if saved.get("voice") == "always":
            payload["voice"] = True  # their setting: a voice note, not a typed line alone
        if opener:
            first = ((p.name or "there").split() or ["there"])[0] or "there"
            template: dict = {"purpose": "update", "params": [first]}
            if saved.get("language"):
                template["lang"] = saved["language"]
            payload["template"] = template  # kavach_saheli_update only; the text waits until they tap or reply
        try:
            # labelled for the backend's delivery ledger: a nudge counts toward "3 unanswered → the caregiver hears" (B2)
            res = await host.call("send_whatsapp", payload, family_id=p.family_id.removeprefix("shadow:"), subject_id=p.id, actor_id=p.id)
        except Exception as exc:  # noqa: BLE001
            row.status, row.reason = "failed", f"send:{type(exc).__name__}"
            await session.commit()
            return "failed"
        row.wamid = str((res or {}).get("wamid") or "")[:120] or None
        if (res or {}).get("via") == "template" and opener:
            row.status, row.sent_at, row.reason = "sent", clock.now(), "waiting:update"
            await session.commit()
            return "waiting"  # the template went; the greeting is queued, not said in the chat yet
        if not (res or {}).get("delivered", True) or (res or {}).get("via") == "template":
            # not said yet: the chat was closed (only the opener template went, or it waits for their next message)
            row.status = "failed"
            row.reason = ("waiting:opener" if (res or {}).get("via") == "template" else str((res or {}).get("reason") or "not_delivered"))[:160]
            await session.commit()
            return "failed"
        turn_id = await store.add_turn(session, family_id=p.family_id, thread_id=p.id, role="assistant", text=text, message_ref=row.ref,
                                       meta={"proactive": True, "ref": row.ref, "nudge": {"id": str(row.id), "kind": kind, "field": fld, "about_ref": about}})
        await scoring.record(session, family_id=p.family_id, thread_id=p.id, turn_id=turn_id, kind="proactive", situation="regular_nudge",
                             speaker_role="system", lang="", user_text="", text=text, tools=[])
        row.status, row.sent_at = "sent", clock.now()
        await session.commit()
    return "sent"
