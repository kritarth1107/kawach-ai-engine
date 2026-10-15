"""Plan B1.3: a simulated week of every job that can message someone, with no AI, for the world's nudge slice.

The model is replaced by an eager one: on every scheduler turn it tries to message every person it can see. Whatever
reaches people must then be held by code: their own night, a paused family, an alert never to the person it is about,
no two messages within 2 minutes (a medicine reminder excepted), no repeat of a text in a day, and a daily cap.

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_budget \
    PYTHONPATH=. python -m eval.world.budget [--families 200] [--days 7] [--tick 5]
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import clock
from app.llm import router
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost, World
from eval.world import make
from eval.world.seed_engine import assert_local, ist, prepare, seed_family, Seeded
from eval.world.slice import slice_ids

UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
DAILY_CAP = int(os.getenv("BUDGET_DAILY_CAP", "6"))  # proactive messages a person may get in a day (medicine reminders apart)
OUT = Path(os.getenv("BUDGET_OUT", "runs/budget"))


class EagerBrain:
    """Every scheduler turn tries to message every person named in it; everything else answers 'none'."""

    def __init__(self) -> None:
        self.n = 0

    async def complete(self, route, *, messages, tools=None, **kw):
        last = messages[-1]
        can_send = any((t.get("name") if isinstance(t, dict) else getattr(t, "name", None)) == "send_message" for t in (tools or []))
        if can_send and last["role"] == "user":
            text = "".join(c.get("text", "") for c in last["content"] if isinstance(c, dict))
            ids = list(dict.fromkeys(UUID.findall(text)))
            if ids:
                calls = []
                for i, pid in enumerate(ids[:4]):
                    self.n += 1
                    calls.append(ToolCall(f"c{self.n}", "send_message", {"to": pid, "text": f"Namaste! Ek chhota sa sawaal #{self.n}: aaj kaisa raha?"}))
                return LLMReply(text="", tool_calls=calls, model="eager")
        return LLMReply(text="none", tool_calls=[], model="eager")


@dataclass
class Fam:
    spec: dict
    host: SimHost
    sends: list[dict] = field(default_factory=list)


def world_host(fam: dict) -> SimHost:
    w = World()
    for s in fam["backend"]["schedules"]:
        w.schedules[s["scheduleId"]] = {"scheduleId": s["scheduleId"], "sourceKey": s.get("sourceKey"), "title": s["title"], "dose": s["dosage"],
                                        "time": s["time"], "active": True, "person": s["recipientUserId"]}
    closed = {x["phone"] for x in fam["backend"]["windows"] if x["hoursSinceInbound"] >= 24}
    w.closed = {p["id"] for p in fam["people"] if p["phone"] in closed}
    w.other_caregivers = sum(1 for p in fam["people"] if p["role"] in ("primary caregiver", "caregiver") and not p["selfCare"])
    w.paused = "paused" in (fam.get("shapes") or [])  # the admin console's pause, as the backend applies it
    return SimHost(w)


async def run(n_families: int, days: int, tick_min: int, url: str) -> dict:
    assert_local(url)
    engine = create_async_engine(url, pool_size=10)
    await prepare(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    brain = EagerBrain()
    router.register_provider("eager", brain)
    old_routes = os.environ.get("MODEL_ROUTES")
    os.environ["MODEL_ROUTES"] = json.dumps({r: ["eager:m"] for r in ("brain", "extract", "worker", "classify", "sim", "judge", "judge_fast", "speech", "compose")})
    router.reset_breakers()
    from app.api import brain as brain_api

    old_kick = brain_api.kick_tasks
    brain_api.kick_tasks = lambda: None
    try:
        return await _run(engine, sessions, brain, n_families, days, tick_min)
    finally:
        brain_api.kick_tasks = old_kick
        if old_routes is None:
            os.environ.pop("MODEL_ROUTES", None)
        else:
            os.environ["MODEL_ROUTES"] = old_routes
        router.reset_breakers()
        clock.set_now(None)
        await engine.dispose()


async def _run(engine, sessions: async_sessionmaker, brain: EagerBrain, n_families: int, days: int, tick_min: int) -> dict:
    world = make.build()
    ids = slice_ids("nudge", world)[:n_families]
    fams = {f["familyId"]: Fam(f, world_host(f)) for f in world if f["id"] in set(ids)}
    seeded = Seeded()
    async with sessions() as s:
        for f in fams.values():
            await seed_family(s, f.spec, seeded)
        await s.commit()

    def host_for(fid: str) -> SimHost:
        return fams[fid].host

    from app.brain.wake import wake_due
    from app.care import activities, brief, careteam, checkins

    start = ist(make.TODAY.isoformat(), "00:00")
    t0 = time.monotonic()
    ticks = 0
    for minute in range(0, days * 24 * 60, tick_min):
        now = start + timedelta(minutes=minute)
        clock.set_now(now)
        ticks += 1
        for f in fams.values():  # the backend's medicine reminders at this minute
            for r in f.host.fire_due_reminders():
                f.sends.append({"at": now, "to": next((s["person"] for s in f.host.world.schedules.values() if s["scheduleId"] == r["scheduleId"]), None),
                                "kind": "medicine_reminder", "text": r["item"]})
        try:
            await activities.plan_all(sessions)
        except Exception as exc:  # noqa: BLE001
            print("plan_all failed", exc)
        for _ in range(10):  # every due loop runs this tick (50 per call)
            st = await wake_due(sessions, host_for)
            if st["due"] < 50:
                break
        await careteam.flush(sessions, host_for)
        await checkins.run(sessions, host_for)
        await brief.run(sessions, host_for)
    clock.set_now(None)
    # what reached people (SimHost world.sent / alerts / templates)
    for fid, f in fams.items():
        for m in f.host.world.sent:
            text = " ".join((m.get("text") or "").split())
            # the nightly brief goes at the caregiver's chosen time (default: after the elder sleeps), so only their own known night counts
            f.sends.append({"at": datetime.fromisoformat(m["at"]), "to": m.get("to"), "kind": "brief" if text.startswith("Daily brief:") else "message", "text": text})
        for m in f.host.world.templates:
            f.sends.append({"at": datetime.fromisoformat(m["at"]), "to": m.get("to"), "kind": "template", "text": f"[template {m.get('purpose')}]"})
        for a in f.host.world.alerts:
            f.sends.append({"at": datetime.fromisoformat(a["at"]), "to": "caregivers", "kind": "alert", "text": a.get("message") or a.get("reason") or "",
                            "about": a.get("about") or a.get("subjectId")})
    report = await check(sessions, fams)
    report.update({"families": len(fams), "days": days, "tick_min": tick_min, "ticks": ticks, "seconds": round(time.monotonic() - t0),
                   "brain_attempts": brain.n})
    return report


async def check(sessions: async_sessionmaker, fams: dict[str, Fam]) -> dict:
    from app.brain.wake import _inside, own_night, quiet_for

    fails: list[str] = []
    per_day: Counter = Counter()
    worst = 0
    timelines: dict[str, list] = {}
    async with sessions() as s:
        for fid, f in fams.items():
            people = {p["id"]: p for p in f.spec["people"]}
            sends = sorted(f.sends, key=lambda x: x["at"])
            by_person: dict[str, list] = defaultdict(list)
            for m in sends:
                by_person[m["to"]].append(m)
            for pid, ms in by_person.items():
                if pid not in people:
                    continue
                days: Counter = Counter()
                texts: dict[str, set] = defaultdict(set)
                prev = None
                for m in ms:
                    d = clock.ist(m["at"]).strftime("%Y-%m-%d")
                    if m["kind"] != "medicine_reminder":
                        days[d] += 1
                        if m["kind"] == "brief":
                            night = await own_night(s, fid, pid)
                            local = clock.ist(m["at"])
                            quiet = bool(night) and _inside(local.hour * 60 + local.minute, night)
                        else:
                            quiet, _ = await quiet_for(s, fid, pid, m["at"])
                        if quiet:
                            fails.append(f"{f.spec['id']} {people[pid]['key']}: message in their night at {clock.ist(m['at']):%a %H:%M}: {m['text'][:60]}")
                        key = m["text"].strip().lower()
                        if key and key in texts[d]:
                            fails.append(f"{f.spec['id']} {people[pid]['key']}: same text twice on {d}: {m['text'][:60]}")
                        texts[d].add(key)
                        if prev and prev["kind"] != "medicine_reminder" and (m["at"] - prev["at"]) < timedelta(minutes=2):
                            fails.append(f"{f.spec['id']} {people[pid]['key']}: two messages within 2 minutes at {clock.ist(m['at']):%a %H:%M}")
                    if "paused" in f.spec["shapes"] and m["kind"] != "medicine_reminder":
                        fails.append(f"{f.spec['id']} {people[pid]['key']}: a message while Saheli is paused for the family")
                    prev = m
                for d, n in days.items():
                    per_day[n] += 1
                    worst = max(worst, n)
                    if n > DAILY_CAP:
                        fails.append(f"{f.spec['id']} {people[pid]['key']}: {n} messages on {d} (cap {DAILY_CAP})")
                timelines[f"{f.spec['id']}:{people[pid]['key']}"] = [f"{clock.ist(m['at']):%a %H:%M} {m['kind']}: {m['text'][:80]}" for m in ms]
            for m in sends:
                if m["kind"] == "alert" and m.get("about") and people.get(m["about"], {}).get("selfCare") and \
                        not any(p["role"] in ("primary caregiver", "caregiver") and p["id"] != m["about"] for p in people.values()):
                    fails.append(f"{f.spec['id']}: an alert about a self-care person with nobody else to tell")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timelines.json").write_text(json.dumps(timelines, ensure_ascii=False, indent=1))
    return {"failures": fails, "messages_per_person_day": dict(sorted(per_day.items())), "worst_day": worst}


def main(argv: list[str]) -> int:
    def opt(name: str, default: int) -> int:
        return int(argv[argv.index(name) + 1]) if name in argv else default

    url = os.getenv("DATABASE_URL") or ""
    rep = asyncio.run(run(opt("--families", 200), opt("--days", 7), opt("--tick", 5), url))
    fails = rep.pop("failures")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "budget_report.json").write_text(json.dumps({**rep, "failures": fails}, ensure_ascii=False, indent=1))  # read by the nightly gate
    print(json.dumps(rep, indent=1))
    kinds = Counter(re.sub(r"^\S+ \S+: ", "", f).split(" at ")[0].split(" on ")[0][:40] for f in fails)
    print(f"{len(fails)} problems; by kind: {dict(kinds.most_common(8))}")
    for x in fails[:40]:
        print("  ", x)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
