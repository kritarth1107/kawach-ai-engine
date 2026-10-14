"""14 days of one invented person with a real routine, so learned meal, water and sleep times can be scored.

The routine has a little day-to-day noise, one odd lunch, and a later weekend. Each day they log about 70
minutes after the true time. An ask is wasted when it goes out before that time, or after the log is already in.
Target: the learned weekday and weekend times are within 30 minutes of the truth by day 7, and at most 10% of
asks are wasted. Also proves that five silent hours are dropped, a fast hour receives the asks, and a time
they state (or a "don't ask") applies at once and undoes.

No model calls. Never a real family.

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_nudgesim \\
    PYTHONPATH=. python -m eval.nudge_learn_sim
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import clock
from app.llm import router
from app.llm.router import LLMReply

OUT = Path(os.getenv("NUDGE_LEARN_OUT", "runs/nudge_learn_sim"))
FAM = "sim-nudge-learn"
PID = "p-learn"
START = "2026-10-05"  # a Monday
DAYS = 14
TICK = 15
WAKE = 7 * 60
WEEKDAY = {"breakfast": 8 * 60, "lunch": 13 * 60, "snack": 16 * 60 + 30, "dinner": 20 * 60 + 30, "water": 11 * 60, "sleep": 7 * 60 + 15}
WEEKEND = {"breakfast": 9 * 60, "lunch": 14 * 60 + 30}
NOISE = [-8, 5, 12, -11, 3, 9, -4, 7, -12, 6, 2, -9, 11, -3]
LOG_AFTER = 70


_LINES = [
    "How was the walk this morning?", "Did you manage a proper lunch today?", "What are you reading these days?",
    "Any plans for the evening?", "Which song is stuck in your head today?", "Did the call with your friend happen?",
    "What made you smile today?", "How did you sleep last night, one to five?", "Is the new plant doing well?",
    "Any good news today?", "What did you cook today?", "How is work treating you this week?",
    "Did you step out for some fresh air?", "What are you looking forward to tomorrow?", "How is the energy this afternoon?",
    "Did the parcel arrive?", "Was the market crowded today?", "Any plans with the neighbours?",
]


class _Writer:
    def __init__(self) -> None:
        self.n = 0
        self.habit = 0

    async def complete(self, route, *, messages, **kw):
        self.n += 1
        text = messages[-1]["content"][0]["text"]
        chosen = "o1"
        for line in text.splitlines():
            if any(f"[{k}]" in line for k in ("meal", "water", "sleep")):
                chosen = line.strip().lstrip("-").strip().split()[0]
                self.habit += 1
                break
        line = _LINES[self.n % len(_LINES)]
        return LLMReply(text=json.dumps({"option_id": chosen, "text": line, "field": None, "expects_reply": True}),
                        tool_calls=[], model="fake")


def ist(day: str, minute: int) -> datetime:
    return datetime.fromisoformat(f"{day}T00:00:00").replace(tzinfo=clock.IST) + timedelta(minutes=minute)


def truth(day_i: int) -> dict[str, int]:
    weekend = date_of(day_i).weekday() >= 5
    out = dict(WEEKDAY)
    if weekend:
        out.update(WEEKEND)
    for slot in out:
        extra = 150 if slot == "lunch" and day_i == 2 else NOISE[day_i]
        out[slot] = out[slot] + extra
    return out


def date_of(day_i: int):
    return datetime.fromisoformat(START).date() + timedelta(days=day_i)


def _hm(value) -> int | None:
    from app.care.meal_nudges import _hm as parse

    return parse(value)


async def _seed(sessions) -> None:
    from app.care import store

    async with sessions() as s:
        person = {"id": PID, "name": "Meera", "role": "elder"}
        await store.save_roster(s, FAM, person, [person])
        for type_, at in (("sleep", "22:30"), ("wake_up", "07:00")):
            await store.write_fact(s, family_id=FAM, subject_id=PID, domain="activity", key=f"activity:{type_}",
                                   value={"type": type_, "kind": "routine", "start": at}, text=f"{type_} {at}", source_kind="caregiver_said")
        await store.write_fact(s, family_id=FAM, subject_id=PID, domain="language", key="language:preferred",
                               value={"language": "en"}, text="Meera writes English", source_kind="caregiver_said")
        clock.set_now(ist("2026-10-04", 22 * 60))
        await store.add_turn(s, family_id=FAM, thread_id=PID, role="user", speaker_id=PID, text="All good, talk tomorrow")
        await s.commit()


async def _log(sessions, day: str, slot: str, eat: int, now: datetime) -> None:
    from app.care import store
    from app.nutrition.models import NutritionEntry

    when = ist(day, eat)
    async with sessions() as s:
        if slot == "sleep":
            await store.record_event(s, family_id=FAM, subject_id=PID, kind="sleep", summary="up", payload={"type": "sleep"}, at=when)
        else:
            s.add(NutritionEntry(
                family_id=FAM, subject_id=PID, logged_by=PID, meal_id=uuid.uuid4(),
                meal="other" if slot == "water" else slot, kind="water" if slot == "water" else "food",
                eaten_at=when, day=day, name="water" if slot == "water" else slot, name_said="",
                nutrients={"water_ml": 250.0} if slot == "water" else {},
                flags={"tags": ["water"]} if slot == "water" else {}, created_at=now, updated_at=now))
        await s.commit()


async def _prove_hours_and_statements(sessions) -> list[str]:
    """A second invented person: five silent 2pm asks drop that hour, 7pm answers draw the ask there,
    and a stated lunch time or a 'don't ask about water' applies at once and undoes."""
    from app.care import meal_nudges, nudges, store
    from app.care.models import NudgeLedger

    fails = []
    fam, pid = "sim-nudge-hours", "p-hours"
    async with sessions() as s:
        person = {"id": pid, "name": "Arun", "role": "elder"}
        await store.save_roster(s, fam, person, [person])
        await store.write_fact(s, family_id=fam, subject_id=pid, domain="activity", key="activity:wake_up",
                               value={"type": "wake_up", "kind": "routine", "start": "07:00"}, text="wake 07:00", source_kind="caregiver_said")
        for i in range(5):
            sent = ist("2026-10-08", 14 * 60) + timedelta(days=i)
            s.add(NudgeLedger(family_id=fam, person_id=pid, kind="meal", status="sent", field="meal:lunch",
                              slot_key=f"silent-{i}", ref=f"nudge:hours-silent-{i}", text="lunch?", planned_at=sent, sent_at=sent))
        for i in range(3):
            sent = ist("2026-10-09", 19 * 60) + timedelta(days=i)
            s.add(NudgeLedger(family_id=fam, person_id=pid, kind="water", status="sent", field="water:count",
                              slot_key=f"fast-{i}", ref=f"nudge:hours-fast-{i}", text="water?", planned_at=sent, sent_at=sent,
                              replied_at=sent + timedelta(minutes=2)))
        clock.set_now(ist("2026-10-14", 12 * 60))
        got = await meal_nudges.learn_reply_hours(s, fam, pid)
        await meal_nudges.state_habit(s, fam, pid, what="lunch", at="14:00")
        await s.commit()
    if got not in ("created", "superseded"):
        fails.append(f"hour learning did not save ({got})")
    async with sessions() as s:
        fact = await store.active_fact(s, fam, pid, meal_nudges.ASK_HOURS_KEY)
        avoid = set((fact.value or {}).get("avoid_hours") or []) if fact else set()
        prefer = (fact.value or {}).get("prefer_hour") if fact else None
        if 14 not in avoid or prefer != 19:
            fails.append(f"hour learning is avoid={sorted(avoid)} prefer={prefer}")
        clock.set_now(ist("2026-10-14", 19 * 60 + 50))
        moved = await meal_nudges.times(s, fam, pid, [])
        if moved.get("lunch") != 19 * 60:
            fails.append(f"lunch did not move to 19:00 ({moved.get('lunch')})")
        said = "sim-nudge-said"
        who = {"id": "p-said", "name": "Lata", "role": "elder"}
        await store.save_roster(s, said, who, [who])
        await store.write_fact(s, family_id=said, subject_id=who["id"], domain="activity", key="activity:wake_up",
                               value={"type": "wake_up", "kind": "routine", "start": "07:00"}, text="wake 07:00", source_kind="caregiver_said")
        await meal_nudges.state_habit(s, said, who["id"], what="lunch", at="15:00")
        await meal_nudges.state_habit(s, said, who["id"], what="water", skip=True)
        await s.commit()
        clock.set_now(ist("2026-10-14", 15 * 60 + 50))
        if (await meal_nudges.times(s, said, who["id"], [])).get("lunch") != 15 * 60:
            fails.append("a stated lunch at 15:00 was not used at once")
        p = nudges.Person(said, who["id"], who["name"], "elder", True, True)
        rows = await nudges.ledger_rows(s, said, who["id"])
        opted = await meal_nudges.options(s, p, rows, clock.now(), acts=[], meals_mode="on")
        if any(o["kind"] == "water" for o in opted):
            fails.append("water was still offered after they asked not to be asked")
        if not any(o["fld"] == "meal:lunch" for o in opted):
            fails.append(f"lunch at the stated time was not offered ({opted})")
        await store.stop_fact(s, family_id=said, subject_id=who["id"], key=meal_nudges.TIMES_KEY, reason="undo", source_kind="caregiver_said")
        await store.stop_fact(s, family_id=said, subject_id=who["id"], key=meal_nudges.SKIP_KEY, reason="undo", source_kind="caregiver_said")
        await s.commit()
        clock.set_now(ist("2026-10-14", 14 * 60 + 30))
        if (await meal_nudges.times(s, said, who["id"], [])).get("lunch") != 13 * 60 + 30:
            fails.append("undoing the stated lunch did not restore the usual time")
        opted = await meal_nudges.options(s, p, rows, ist("2026-10-14", 12 * 60 + 10), acts=[], meals_mode="on")
        if not any(o["kind"] == "water" for o in opted):
            fails.append("undoing 'don't ask about water' did not bring the water ask back")
    return fails


async def run(url: str) -> dict:
    from app.care import meal_nudges, nudges
    from app.sim.world import SimHost, World
    from eval.world.seed_engine import assert_local, prepare

    assert_local(url)
    engine = create_async_engine(url, pool_size=5)
    await prepare(engine)
    async with engine.begin() as c:
        await c.execute(text("DELETE FROM nudge_ledger WHERE family_id LIKE 'sim-nudge-%'"))
        for table in ("turns", "care_events", "care_facts", "family_rosters"):
            await c.execute(text(f"DELETE FROM {table} WHERE family_id LIKE 'sim-nudge-%'"))
        await c.execute(text("DELETE FROM nutrition_entries WHERE family_id LIKE 'sim-nudge-%'"))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    old = os.environ.get("MODEL_ROUTES")
    writer = _Writer()
    router.register_provider("fakewriter", writer)
    os.environ["MODEL_ROUTES"] = json.dumps({"worker": ["fakewriter:m"], "judge_fast": ["fakewriter:m"]})
    router.reset_breakers()
    host = SimHost(World())
    host.world.meals_mode = "on"
    fails: list[str] = []
    try:
        await _seed(sessions)
        logged: set[tuple[str, str]] = set()
        offered: set[tuple[str, str]] = set()
        asks: list[dict] = []
        settled = None
        learned_on: dict[int, dict] = {}
        start = ist(START, 0)
        for minute in range(0, DAYS * 24 * 60, TICK):
            now = start + timedelta(minutes=minute)
            clock.set_now(now)
            day_i = minute // (24 * 60)
            if minute % (24 * 60) == 0:
                print(f"day {day_i + 1}", flush=True)
            day = date_of(day_i).isoformat()
            local = clock.ist(now).hour * 60 + clock.ist(now).minute
            plan = truth(day_i)
            for slot, eat in plan.items():
                if (day, slot) not in logged and local >= eat + LOG_AFTER:
                    await _log(sessions, day, slot, eat, now)
                    logged.add((day, slot))
            if local == 22 * 60:
                async with sessions() as s:
                    from app.care import store
                    await store.add_turn(s, family_id=FAM, thread_id=PID, role="user", speaker_id=PID, text="Still here")
                    await s.commit()
            await nudges.run(sessions, lambda f: host, mode="on")
            async with sessions() as s:
                await nudges.note_reply(s, FAM, PID)
                await s.commit()
                person = nudges.Person(FAM, PID, "Meera", "elder", True, True)
                rows = await nudges.ledger_rows(s, FAM, PID)
                opts = await meal_nudges.options(s, person, rows, now, acts=[], meals_mode="on")
            for o in opts:
                slot = "sleep" if o["kind"] == "sleep" else "water" if o["kind"] == "water" else str(o["fld"]).split(":")[-1]
                if (day, slot) in offered:
                    continue
                offered.add((day, slot))
                eat = plan.get(slot)
                wasted = "before" if eat is not None and local < eat else "after" if (day, slot) in logged else None
                asks.append({"day": day, "kind": o["kind"], "slot": slot, "at": f"{local // 60:02d}:{local % 60:02d}", "wasted": wasted})
            if local >= 23 * 60 + 30:
                async with sessions() as s:
                    await meal_nudges.learn_rhythms(s, FAM, PID)
                    await s.commit()
                    meals = await store_fact(s, meal_nudges.CLOCKS_KEY)
                    water = await store_fact(s, meal_nudges.WATER_KEY)
                    sleep = await store_fact(s, meal_nudges.SLEEP_KEY)
                learned_on[day_i + 1] = {"meals": meals, "water": water, "sleep": sleep}
                if settled is None and _settled(meals, water, sleep):
                    settled = day_i + 1
        wasted = [a for a in asks if a["wasted"]]
        pct = (100.0 * len(wasted) / len(asks)) if asks else 100.0
        if settled is None or settled > 7:
            fails.append(f"learned times were not within 30 min by day 7 (settled={settled})")
        if not asks:
            fails.append("no habit ask was sent")
        elif pct > 10:
            fails.append(f"{len(wasted)} of {len(asks)} asks wasted ({pct:.1f}%)")
        print(f"writer calls {writer.n}, habit options seen {writer.habit}", flush=True)
        fails += await _prove_hours_and_statements(sessions)
        report = {"fails": fails, "days_to_within_30": settled, "asks": len(asks), "wasted": len(wasted),
                  "wasted_pct": round(pct, 1), "wasted_detail": wasted[:20], "learned": learned_on.get(settled or 7),
                  "days": DAYS}
    finally:
        clock.set_now(None)
        if old is None:
            os.environ.pop("MODEL_ROUTES", None)
        else:
            os.environ["MODEL_ROUTES"] = old
        router.reset_breakers()
        await engine.dispose()
    return report


async def store_fact(session, key: str) -> dict:
    from app.care import store

    fact = await store.active_fact(session, FAM, PID, key)
    return dict(fact.value or {}) if fact else {}


def _close(value, truth_min: int) -> bool:
    got = _hm(value)
    return got is not None and abs(got - truth_min) <= 30


def _settled(meals: dict, water: dict, sleep: dict) -> bool:
    weekday = meals.get("weekday") if isinstance(meals.get("weekday"), dict) else {}
    weekend = meals.get("weekend") if isinstance(meals.get("weekend"), dict) else {}
    meals_ok = all(_close(weekday.get(slot) or meals.get(slot), WEEKDAY[slot]) for slot in ("breakfast", "lunch", "dinner"))
    weekend_ok = _close(weekend.get("lunch"), WEEKEND["lunch"]) and _close(weekend.get("breakfast"), WEEKEND["breakfast"])
    waters = water.get("weekday") or water.get("times") or []
    water_ok = any(_close(t, WEEKDAY["water"]) for t in waters)
    sleep_ok = _close(sleep.get("weekday") or sleep.get("at"), WEEKDAY["sleep"])
    return bool(meals_ok and weekend_ok and water_ok and sleep_ok)


def main(argv: list[str]) -> int:
    url = os.getenv("DATABASE_URL") or ""
    report = asyncio.run(run(url))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    lines = ["# Learned timing, 14 days", "",
             f"Within 30 min on day {report['days_to_within_30']}.",
             f"Asks {report['asks']}, wasted {report['wasted']} ({report['wasted_pct']}%).", "",
             "## Failures", ""]
    lines += [f"- {f}" for f in report["fails"]] or ["none"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{len(report['fails'])} failures; within 30 min on day {report['days_to_within_30']}; "
          f"wasted {report['wasted']}/{report['asks']} ({report['wasted_pct']}%)")
    for f in report["fails"]:
        print("  -", f)
    return 1 if report["fails"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
