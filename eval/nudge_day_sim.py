"""Regular nudges, a simulated day per persona (Kavach NUdge's plan §5): the nudge job runs every tick on a fake clock and a
fake backend, the people answer (or not) the way each persona does, and the timeline is checked against the rules.

Personas (all invented): an early-rising elder who writes Hindi; a late riser caring for themselves with an hourly
quit-smoking reminder; a caregiver abroad; an elder who never replies; an elder with a ride being booked mid-morning.

Offline by default (a fake writer, no model calls, safe in CI). --real writes with the production models in the sunny-ship
test project (never kavach-care), with a spend guard.

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_nudgesim \
    PYTHONPATH=. python -m eval.nudge_day_sim [--days 2] [--tick 1] [--real --max-inr 50]
    PYTHONPATH=. python -m eval.nudge_day_sim --learn   # 14-day routine: learned times within 30 min by day 7
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import clock
from app.llm import router
from app.llm.router import LLMReply
from app.sim.world import SimHost, World

OUT = Path(os.getenv("NUDGE_SIM_OUT", "runs/nudge_day_sim"))
DAY = "2026-10-20"
LINES = {
    "en": ["How was the walk this morning?", "Did you manage a proper lunch today?", "What are you reading these days?",
           "Any plans for the evening?", "How is the knee feeling now?", "Which song is stuck in your head today?",
           "Did the call with your friend happen?", "What made you smile today?", "How did you sleep last night, one to five?",
           "Is the new plant doing well?", "Any good news today?", "What did you cook today?", "How is work treating you this week?",
           "Did you step out for some fresh air?", "What are you looking forward to tomorrow?", "How is the energy this afternoon?"],
    "hi": ["आज सुबह की सैर कैसी रही?", "दोपहर में थोड़ा आराम हुआ?", "आज खाने में क्या बना?", "घुटने का दर्द आज कैसा है?",
           "आज कोई अच्छा भजन सुना?", "शाम की चाय हो गई?", "आज किसी अपने से बात हुई?", "कल रात नींद ठीक से आई?",
           "आज बगीचे में फूल खिले?", "आज मंदिर जाना हुआ?", "आज मन कैसा है?", "पोते-पोतियों से बात हुई आज?"],
}


class FakeWriter:
    """Writes a different plain line each time in the language the context asks for (no model)."""

    def __init__(self) -> None:
        self.n = 0

    async def complete(self, route, *, messages, **kw):
        ctx = messages[-1]["content"][0]["text"]
        lang = "hi" if "Devanagari" in ctx.split("NOW:")[0] or "Hindi" in ctx.split("NOW:")[0] else "en"
        self.n += 1
        if "no question at all" in ctx:
            line = ("सब ठीक है।", "ठीक हूँ, यही काफी है।")[self.n % 2] if lang == "hi" else ("Glad all is good here.", "All good here, and that is enough.")[self.n % 2]
        else:
            line = LINES[lang][self.n % len(LINES[lang])]
        return LLMReply(text=json.dumps({"option_id": "o1", "text": line, "field": None, "expects_reply": "no question at all" not in ctx}), tool_calls=[], model="fake")


@dataclass
class Persona:
    key: str
    person: dict
    elder: bool  # the care recipient of their own family
    sleep: str
    wake: str
    lang: str
    replies: str  # always | half | never | daily
    flows: list[tuple[str, str]] = field(default_factory=list)  # (from, to) HH:MM: a ride or order being booked
    quit_reminder: bool = False  # an hourly custom reminder 14:00–23:00 linked to a quit activity
    meals: bool = False  # nudge.meals on: meal asks use the real nutrition hooks against an empty food log

    @property
    def family(self) -> str:
        return f"sim-nudge-{self.key}"


PERSONAS = [
    Persona("early_elder", {"id": "p-early", "name": "Savitri", "role": "elder"}, True, "21:00", "05:30", "hi", "half"),
    Persona("late_self", {"id": "p-late", "name": "Arjun", "role": "primary caregiver"}, False, "03:00", "13:30", "en", "half", quit_reminder=True),
    Persona("caregiver_abroad", {"id": "p-abroad", "name": "Neha", "role": "primary caregiver"}, False, "11:30", "19:00", "en", "daily"),
    Persona("never_replies", {"id": "p-never", "name": "Gopal", "role": "elder"}, True, "22:00", "06:30", "en", "never"),
    Persona("mid_order", {"id": "p-order", "name": "Leela", "role": "elder"}, True, "22:00", "06:00", "en", "half", flows=[("10:00", "12:00")]),
    Persona("meal_self", {"id": "p-meals", "name": "Meera", "role": "elder"}, True, "22:00", "07:00", "en", "half", meals=True),
    Persona("meal_late", {"id": "p-meal-late", "name": "Ravi", "role": "elder"}, True, "03:00", "13:30", "en", "half", meals=True),
]
CARED_FOR = {"caregiver_abroad": {"id": "p-abroad-mother", "name": "Usha", "role": "elder"}}


def ist(day: str, hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=clock.IST)


async def seed(sessions, p: Persona) -> None:
    from app.care import store

    async with sessions() as s:
        person = p.person
        if p.key == "caregiver_abroad":
            elder = CARED_FOR[p.key]
            await store.save_roster(s, p.family, elder, [elder, person])
        else:
            await store.save_roster(s, p.family, person, [person])
        for type_, at in (("sleep", p.sleep), ("wake_up", p.wake)):
            await store.write_fact(s, family_id=p.family, subject_id=person["id"], domain="activity", key=f"activity:{type_}",
                                   value={"type": type_, "kind": "routine", "start": at}, text=f"{type_} {at}", source_kind="caregiver_said")
        await store.write_fact(s, family_id=p.family, subject_id=person["id"], domain="language", key="language:preferred",
                               value={"language": p.lang}, text=f"{person['name']} writes {p.lang}", source_kind="caregiver_said")
        if not p.elder and p.key == "late_self":
            await store.write_fact(s, family_id=p.family, subject_id=person["id"], domain="condition", key="condition:anxiety",
                                   value={"name": "Anxiety"}, text="Arjun has anxiety", source_kind="caregiver_said")
        if p.quit_reminder:
            await store.write_fact(s, family_id=p.family, subject_id=person["id"], domain="activity", key="activity:quit_smoking",
                                   value={"type": "quit_smoking", "kind": "quit", "label": "No cigarettes"}, text="Quitting cigarettes",
                                   source_kind="caregiver_said")
        clock.set_now(ist(DAY, "00:00") - timedelta(hours=3))  # they wrote the evening before (the chat is open)
        await store.add_turn(s, family_id=p.family, thread_id=person["id"], role="user", speaker_id=person["id"],
                             text="नमस्ते, मैं ठीक हूँ" if p.lang == "hi" else "Hi, all good here")
        await s.commit()


async def answer(sessions, p: Persona, now: datetime, n_seen: int) -> bool:
    from app.care import nudges, store

    if p.replies == "never" or (p.replies == "half" and n_seen % 2) or (p.replies == "daily" and clock.ist(now).strftime("%H:%M") != "20:00"):
        return False
    async with sessions() as s:
        await store.add_turn(s, family_id=p.family, thread_id=p.person["id"], role="user", speaker_id=p.person["id"],
                             text="हाँ, सब ठीक है" if p.lang == "hi" else "Yes, all fine, thanks")
        await nudges.note_reply(s, p.family, p.person["id"])
        await s.commit()
    return True


async def run(days: int, tick: int, url: str, real: bool, max_inr: float) -> dict:
    from app.api.brain import EventIn, push_event
    from app.care import nudges
    from app.care.models import NudgeLedger
    from eval.world.seed_engine import assert_local, prepare

    assert_local(url)
    engine = create_async_engine(url, pool_size=10)
    await prepare(engine)
    async with engine.begin() as c:
        await c.execute(text("DELETE FROM nudge_ledger WHERE family_id LIKE 'sim-nudge-%'"))
        await c.execute(text("DELETE FROM nudge_due WHERE family_id LIKE 'sim-nudge-%'"))
        for t in ("turns", "care_events", "care_facts", "family_rosters"):
            await c.execute(text(f"DELETE FROM {t} WHERE family_id LIKE 'sim-nudge-%'"))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    old_routes = os.environ.get("MODEL_ROUTES")
    if not real:
        router.register_provider("fakewriter", FakeWriter())
        # judge_fast too: the no-invention check and the reviewer otherwise call a real model (the sim is not a paid run)
        os.environ["MODEL_ROUTES"] = json.dumps({"worker": ["fakewriter:m"], "judge_fast": ["fakewriter:m"]})
    router.reset_breakers()
    hosts = {p.family: SimHost(World()) for p in PERSONAS}
    spare = SimHost(World())  # for another job's families in the same database
    for p in PERSONAS:
        if p.meals:
            hosts[p.family].world.meals_mode = "on"
    for p in PERSONAS:
        await seed(sessions, p)
    timeline: dict[str, list] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    pending: dict[str, datetime] = {}
    start = ist(DAY, "00:00")
    try:
        for minute in range(0, days * 24 * 60, tick):
            now = start + timedelta(minutes=minute)
            clock.set_now(now)
            hhmm = clock.ist(now).strftime("%H:%M")
            for p in PERSONAS:
                h = hosts[p.family]
                h.world.flows.pop(p.person["id"], None)
                if any(a <= hhmm < b for a, b in p.flows):
                    h.world.flows[p.person["id"]] = "ride:await_choice"
                if p.quit_reminder and hhmm.endswith(":00") and "14:00" <= hhmm <= "23:00":
                    line = f"Hour {hhmm[:2]}: still smoke-free? 🚭"
                    async with sessions() as s:
                        await push_event(EventIn(family_id=p.family, subject_id=p.person["id"], kind="reminder_sent", summary=f"reminder: no cigarette at {hhmm}",
                                                 payload={"custom": True, "reminderId": "rem-quit", "slot": hhmm, "dateKey": clock.ist_day(now), "preview": line,
                                                          "aboutKey": "activity:quit_smoking"}, ref=f"nudge:quit:{clock.ist_day(now)}:{hhmm}:sent"), s)
                    h.world.outbound.append({"to": p.person["id"], "text": line, "at": now.isoformat()})
                    timeline[p.key].append((now, "reminder", line))
                    pending[p.key] = now + timedelta(minutes=2)
                    seen[p.key] += 1
            for p in PERSONAS:  # the person answers 2 minutes after a message they see (if they answer)
                due = pending.get(p.key)
                if (due and now >= due) or (p.replies == "daily" and hhmm == "20:00"):
                    pending.pop(p.key, None)
                    if await answer(sessions, p, now, seen[p.key]):
                        timeline[p.key].append((now, "they wrote", "(reply)"))
            # only this sim's families: another job's world can share the database (the nightly's message-week runs the
            # 200-family budget world first); a follow-up for one of its reminders goes to a throwaway host
            out = await nudges.run(sessions, lambda f: hosts[f], mode="on", families=set(hosts))
            await nudges.reminder_followups(sessions, lambda f: hosts.get(f) or spare, mode="on")
            for p in PERSONAS:
                sent = [m for m in hosts[p.family].world.sent if m.get("to") == p.person["id"]]
                for m in sent[len([x for x in timeline[p.key] if x[1] == "nudge"]):]:
                    timeline[p.key].append((now, "nudge", m["text"]))
                    pending[p.key] = now + timedelta(minutes=2)
                    seen[p.key] += 1
            if real and out and sum(float(r or 0) for r in []) > max_inr:
                break
        async with sessions() as s:
            rows = list((await s.execute(select(NudgeLedger).where(NudgeLedger.family_id.like("sim-nudge-%")).order_by(NudgeLedger.planned_at))).scalars())
        report = await check(sessions, timeline, rows, days)
    finally:
        clock.set_now(None)
        if old_routes is None:
            os.environ.pop("MODEL_ROUTES", None)
        else:
            os.environ["MODEL_ROUTES"] = old_routes
        router.reset_breakers()
        await engine.dispose()
    return report


async def check(sessions, timeline: dict[str, list], rows: list, days: int) -> dict:
    """The rules, per persona: none in their night (late risers: at most 2 soft morning messages), an hour of quiet before
    each nudge (3 hours for a caregiver), the unanswered ladder, nothing while a ride is booked, a follow-up 5 minutes
    after an unanswered reminder and none after a reply."""
    from app.brain.wake import quiet_for
    from app.care import nudges

    fails: list[str] = []
    by = {p.key: p for p in PERSONAS}
    kinds = {(r.person_id, r.sent_at.replace(second=0, microsecond=0) if r.sent_at else None): r.kind for r in rows if r.status == "sent"}
    async with sessions() as s:
        for key, events in timeline.items():
            p = by[key]
            events.sort(key=lambda e: e[0])
            gap = timedelta(hours=1 if p.elder or p.key == "late_self" else 3)
            last_any = ist(DAY, "00:00") - timedelta(hours=3)
            per_day: dict[str, int] = defaultdict(int)
            birds: dict[str, int] = defaultdict(int)
            for at, what, line in events:
                if what != "nudge":
                    last_any = at
                    continue
                kind = kinds.get((p.person["id"], at.replace(second=0, microsecond=0)), "?")
                night, _ = await quiet_for(s, p.family, p.person["id"], at)
                wrote = [e[0] for e in events if e[1] == "they wrote" and e[0] < at]
                if night and await nudges.woke_early(s, p.family, p.person["id"], at, max(wrote) if wrote else None):
                    night = False  # a late riser who answered this morning is awake (§1.5)
                local = clock.ist(at)
                if kind == "early_bird":
                    birds[local.strftime("%Y-%m-%d")] += 1
                elif night and kind != "reminder_followup":
                    fails.append(f"{key}: a {kind} nudge in their night at {local:%a %H:%M}")
                if kind not in ("reminder_followup", "early_bird") and at - last_any < gap:
                    fails.append(f"{key}: a {kind} nudge {int((at - last_any).total_seconds() // 60)} min after the last message at {local:%a %H:%M}")
                if any(a <= local.strftime("%H:%M") < b for a, b in p.flows):
                    fails.append(f"{key}: a nudge while a ride was being booked at {local:%H:%M}")
                if kind != "reminder_followup":
                    per_day[local.strftime("%Y-%m-%d")] += 1
                last_any = at
            if any(n > 2 for n in birds.values()):
                fails.append(f"{key}: more than 2 early-bird messages in a morning")
            if p.replies == "never" and any(n > 3 for n in per_day.values()):
                fails.append(f"{key}: {dict(per_day)} nudges a day to someone who never answers (2 questions, then one warm line)")
            if p.meals:
                meal_rows = [r for r in rows if r.person_id == p.person["id"] and r.kind == "meal" and r.status == "sent" and r.sent_at]
                per_day_meals: dict[str, int] = defaultdict(int)
                wh, wm = (int(x) for x in p.wake.split(":"))
                wake_m = wh * 60 + wm
                for r in meal_rows:
                    local_m = clock.ist(r.sent_at)
                    per_day_meals[local_m.strftime("%Y-%m-%d")] += 1
                    if local_m.hour * 60 + local_m.minute < wake_m:
                        fails.append(f"{key}: a meal ask before wake at {local_m:%a %H:%M}")
                if any(n > 4 for n in per_day_meals.values()):
                    fails.append(f"{key}: {dict(per_day_meals)} meal asks in a day (at most 4)")
                habit_rows = [r for r in rows if r.person_id == p.person["id"] and r.kind in ("water", "sleep") and r.status == "sent" and r.sent_at]
                per_day_sleep: dict[str, int] = defaultdict(int)
                for r in habit_rows:
                    local_m = clock.ist(r.sent_at)
                    if local_m.hour * 60 + local_m.minute < wake_m:
                        fails.append(f"{key}: a {r.kind} ask before wake at {local_m:%a %H:%M}")
                    if r.kind == "sleep":
                        per_day_sleep[local_m.strftime("%Y-%m-%d")] += 1
                if any(n > 1 for n in per_day_sleep.values()):
                    fails.append(f"{key}: {dict(per_day_sleep)} sleep asks in a day")
                doses = [r.sent_at for r in rows if r.person_id == p.person["id"] and r.kind == "doses" and r.status == "sent" and r.sent_at]
                for r in meal_rows:
                    if any(abs((r.sent_at - d).total_seconds()) < 30 * 60 for d in doses):
                        fails.append(f"{key}: a meal ask within 30 min of a medicine question")
            if p.quit_reminder:
                rems = [e for e in events if e[1] == "reminder"]
                for at, _, _ in rems:
                    replied = any(e[1] == "they wrote" and at < e[0] <= at + timedelta(minutes=5) for e in events)
                    fup = [e for e in events if e[1] == "nudge" and at + timedelta(minutes=5) <= e[0] <= at + timedelta(minutes=10)
                           and kinds.get((p.person["id"], e[0].replace(second=0, microsecond=0))) == "reminder_followup"]
                    if replied and fup:
                        fails.append(f"{key}: a follow-up after they answered the {clock.ist(at):%H:%M} reminder")
    sent = defaultdict(lambda: defaultdict(int))
    for r in rows:
        sent[r.person_id][f"{r.status}:{r.reason or r.kind}"] += 1
    return {"fails": fails, "ledger": {k: dict(v) for k, v in sent.items()},
            "timeline": {k: [(clock.ist(at).strftime("%a %H:%M"), what, line) for at, what, line in sorted(v, key=lambda e: e[0])] for k, v in timeline.items()},
            "days": days}


def main(argv: list[str]) -> int:
    if "--learn" in argv:
        from eval.nudge_learn_sim import main as learn_main

        return learn_main(argv)
    days = int(argv[argv.index("--days") + 1]) if "--days" in argv else 2
    tick = int(argv[argv.index("--tick") + 1]) if "--tick" in argv else 1
    real = "--real" in argv
    max_inr = float(argv[argv.index("--max-inr") + 1]) if "--max-inr" in argv else 50.0
    if real and os.getenv("GCP_PROJECT_ID") == "kavach-care":
        print("Refusing: paid runs go to the sunny-ship test project, never kavach-care.")
        return 2
    url = os.getenv("DATABASE_URL") or ""
    report = asyncio.run(run(days, tick, url, real, max_inr))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    lines = [f"# Regular nudges: {days}-day simulation, tick {tick} min", ""]
    for key, ev in report["timeline"].items():
        lines += [f"## {key}", ""] + [f"- {t} {what}: {line}" for t, what, line in ev] + [""]
    lines += ["## Ledger", "", "```", json.dumps(report["ledger"], indent=1), "```", "", "## Rule failures", ""]
    lines += [f"- {f}" for f in report["fails"]] or ["none"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{len(report['fails'])} rule failures; timeline in {OUT}/report.md")
    for f in report["fails"][:30]:
        print("  -", f)
    return 1 if report["fails"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
