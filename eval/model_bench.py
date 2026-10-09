"""Which model should be Saheli's brain? The same scripted conversations on each candidate, scored in code and by a judge.

Paid (real model calls). Runs only in a separate GCP project, never kavach-care, with a rupee cap:

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_sim GCP_PROJECT_ID=<sim project> \
    PYTHONPATH=. .venv/bin/python eval/model_bench.py --max-inr 2000 --yes [--models gemini:gemini-3.5-flash@asia-south1,...] [--no-judge]

Per candidate (MODEL_ROUTES brain = that model only, so a fallback cannot hide a failure):
  - the 16 brain scenarios (eval/brain_scenarios.py): hard checks on what was saved, sent and ordered;
  - the founder's cases from 2026-10-09 (below): dose corrections and earlier days, a weekly medicine not due, a stale
    medicine re-checked before ordering, crisp replies, Marwari in Devanagari, red-flag handling mid-conversation;
  - latency per turn, rupees per message (from the spend ledger), and a judge's 1-5 for natural, warm, correct, brief.
Stops at --max-inr. Prints a table and writes runs/model_bench_<time>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODELS = ["gemini:gemini-3.5-flash@asia-south1", "gemini:gemini-3.8-flash", "gemini:gemini-3.1-pro-preview", "claude:claude-sonnet-5-5"]
JUDGE_ROUTE = "gemini:gemini-3.1-pro-preview"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

MAA = {"id": "elder-vasu", "name": "Vasundara", "role": "elder"}
SON = {"id": "cg-kritarth", "name": "Kritarth", "role": "primary caregiver"}
FAMILY = [MAA, SON]
MAA_SETUP = (
    "Medical record for my mother Vasundara, 68, Raipur. She speaks Marwari. Medicines: BP 1 tablet at 8 am, Folvite 5mg half "
    "tablet at 1 pm, Shelcal 500mg 1 tablet at 2 pm, Vitamin D3 60000 IU one capsule only on Sundays at 10 am. Call her "
    "Vasundara ji. Family doctor Dr Anil Sharma."
)


def words(text: str) -> int:
    return len(re.findall(r"[^\W\d_]+", text or ""))


async def founder_cases(say, run, store, SessionLocal) -> None:
    """Cases from the founder's live feedback on 2026-10-09 (run.check records failures)."""
    from app.care import doses

    fam = run.family_id
    await say(run, SON, MAA_SETUP, "2026-10-07 10:00", elder=MAA, members=FAMILY)
    # a weekly medicine not due: "I took everything today" on a Friday must not mark Vitamin D3
    r = await say(run, MAA, "आज मैंने सारी दवाई ले ली।", "2026-10-09 19:00", elder=MAA, members=FAMILY)
    async with SessionLocal() as s:
        ev = await store.events(s, fam, MAA["id"], day="2026-10-09", kinds=list(doses.KINDS))
    run.check(not any("vitamin" in (e.summary or "").lower() for e in ev), "weekly Vitamin D3 not marked on a Friday")
    run.check(sum(1 for e in ev if e.kind == "dose_taken") >= 3, f"today's three doses marked taken ({[e.summary for e in ev]})")
    run.check(words(r) <= 35, f"crisp reply ({words(r)} words)")
    run.check(bool(DEVANAGARI.search(r)), "Devanagari for a Marwari speaker")
    # a correction for today and an earlier day
    await say(run, MAA, "नहीं नहीं, गलती हो गई। आज कोई दवाई नहीं ली। कल फोलविट छोड़ के सारी ली थी।", "2026-10-09 19:02", elder=MAA, members=FAMILY)
    async with SessionLocal() as s:
        today = await store.events(s, fam, MAA["id"], day="2026-10-09", kinds=list(doses.KINDS))
        yday = await store.events(s, fam, MAA["id"], day="2026-10-08", kinds=list(doses.KINDS))
    run.check(today and all(e.kind != "dose_taken" for e in today), f"today corrected to not taken ({[e.summary for e in today]})")
    run.check(any("folvite" in (e.summary or "").lower() and e.kind != "dose_taken" for e in yday), "yesterday's Folvite marked not taken")
    run.check(sum(1 for e in yday if e.kind == "dose_taken") >= 2, f"yesterday's other doses marked taken ({[e.summary for e in yday]})")
    # mid-conversation worry: short, asks about red flags or tells the family, no re-greeting
    r = await say(run, MAA, "जी घबरा रह्यो है, चक्कर आवे से", "2026-10-09 19:04", elder=MAA, members=FAMILY)
    alerted = any(a.get("tool") in ("alert_caregiver", "send_message") for a in run.results[-1].actions)
    run.check(alerted or re.search(r"(छाती|सीने|सांस|साँस)", r), "dizziness: tells the family or asks about chest pain/breathing")
    run.check(words(r) <= 45, f"short under worry ({words(r)} words)")
    # caregiver asks for the day: factual and brief
    r = await say(run, SON, "Maa ka aaj kaisa raha?", "2026-10-09 20:00", elder=MAA, members=FAMILY)
    run.check(words(r) <= 90, f"caregiver summary brief ({words(r)} words)")
    run.check(bool(re.search(r"(dawai|medicine|दवा|nahi|नहीं|missed)", r, re.I)), "mentions today's missed medicines")


async def run_candidate(route: str, args, judge) -> dict:
    import brain_scenarios as bs
    from app.brain.tools import set_task_agent
    from app.care import store
    from app.core import clock
    from app.db.session import SessionLocal
    from app.llm import router, spend
    from app.sim.world import SimHost

    # the other roles as in production (BENCH_BASE_ROUTES), only the brain changes
    base = json.loads(os.getenv("BENCH_BASE_ROUTES") or "{}")
    os.environ["MODEL_ROUTES"] = json.dumps({**base, "brain": [route], "brain_hard": [route]})
    router.reset_breakers()
    cost = {"inr": 0.0}
    original = spend.record

    async def counted(role, model, usage):
        inr = await original(role, model, usage)
        cost["inr"] += inr or 0.0
        return inr

    spend.record = counted
    latencies, scen = [], []
    base_say = bs.say

    async def say(run, who, msg, when, elder=None, members=None):
        from app.brain.loop import TurnRequest, run_turn

        clock.set_now(bs.ist(when))
        async with SessionLocal() as session:
            res = await run_turn(session, run.host, TurnRequest(
                family_id=run.family_id, elder=elder or bs.ELDER, speaker=who, members=members or bs.MEMBERS, text=msg,
                message_ref=f"m-{uuid.uuid4().hex[:8]}"))
        run.replies.append(res.reply)
        run.results.append(res)
        latencies.append(res.ms)
        if judge is not None:
            run.judged.append((who.get("role"), msg, res.reply))
        return res.reply

    bs.say = say
    try:
        names = [n for n in bs.SCENARIOS if n not in args.skip and (not args.only or n in args.only)]
        for name in names + ["founder_cases"]:
            if cost["inr"] > args.max_inr_each:
                scen.append({"name": name, "passed": False, "failures": ["skipped: candidate budget reached"]})
                continue
            run = bs.Run(family_id=f"bench-{name}-{uuid.uuid4().hex[:6]}", host=SimHost())
            run.judged = []
            set_task_agent(run.agent)
            started = time.monotonic()
            try:
                if name == "founder_cases":
                    await founder_cases(say, run, store, SessionLocal)
                else:
                    await bs.SCENARIOS[name](run)
            except Exception as exc:  # noqa: BLE001
                run.failures.append(f"crashed: {type(exc).__name__}: {str(exc)[:160]}")
            finally:
                clock.set_now(None)
            scores = []
            if judge is not None:
                for role, msg, reply in run.judged[-6:]:
                    s = await judge(role, msg, reply)
                    if s:
                        scores.append(s)
            scen.append({"name": name, "passed": not run.failures, "failures": run.failures, "s": round(time.monotonic() - started, 1),
                         "judge": round(sum(scores) / len(scores), 2) if scores else None})
            print(f"  {'✓' if not run.failures else '✗'} {route} {name} {'; '.join(run.failures)[:200]}")
    finally:
        bs.say = base_say
        spend.record = original
    judged = [x["judge"] for x in scen if x.get("judge")]
    msgs = max(1, len(latencies))
    lat = sorted(latencies)
    return {"model": route, "scenarios": scen, "passed": sum(1 for x in scen if x["passed"]), "total": len(scen),
            "judge": round(sum(judged) / len(judged), 2) if judged else None, "inr": round(cost["inr"], 1),
            "inr_per_message": round(cost["inr"] / msgs, 2), "p50_ms": lat[len(lat) // 2] if lat else None,
            "p90_ms": lat[int(len(lat) * 0.9)] if lat else None}


def make_judge(total_cap):
    from app.llm import router

    route = router.parse_route(JUDGE_ROUTE)
    prompt = ("You grade one reply from Saheli, a WhatsApp care companion for elderly Indian parents and their adult children. "
              "Score 1-5 overall: sounds like a warm, real person (not a script), crisp and direct, in the person's language "
              "and its own script, correct and safe, does what was asked. Return only JSON {\"score\": n, \"why\": \"...\"}.")

    async def judge(role, msg, reply):
        try:
            out = await router.complete("judge", system_stable=prompt, routes=[route], max_tokens=300, effort="low",
                                        messages=[{"role": "user", "content": [{"type": "text", "text": f"Speaker ({role}): {msg}\nSaheli: {reply}"}]}])
            m = re.search(r'"score"\s*:\s*(\d)', out.text or "")
            return int(m.group(1)) if m else None
        except Exception:  # noqa: BLE001
            return None

    return judge


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--max-inr", type=float, required=True)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--skip", default="", help="comma-separated scenario names to skip")
    ap.add_argument("--only", default="", help="comma-separated scenario names to run (founder cases always run)")
    args = ap.parse_args()
    args.skip = {s for s in args.skip.split(",") if s}
    args.only = {s for s in args.only.split(",") if s}
    project = os.getenv("GCP_PROJECT_ID", "")
    if not project or project == "kavach-care":
        print("Refusing: set GCP_PROJECT_ID to a separate project (never kavach-care).")
        return 2
    if "5433" not in os.getenv("DATABASE_URL", "") and "kawach_sim" not in os.getenv("DATABASE_URL", "") and "kawach_bench" not in os.getenv("DATABASE_URL", ""):
        print("Refusing: DATABASE_URL must be the simulator database.")
        return 2
    models = [m for m in args.models.split(",") if m]
    if not args.yes:
        print(f"Paid run on {project}: {len(models)} models, cap ₹{args.max_inr}. Add --yes to start.")
        return 1
    args.max_inr_each = args.max_inr * 0.8 / len(models)

    from sqlalchemy import text

    import brain_scenarios as bs
    import importlib

    for mod in ("app.care.baselines", "app.care.memory_index", "app.care.models", "app.care.skillbook", "app.care.versions",
                "app.learn.models", "app.llm.spend", "app.models.entities", "app.specialists.channels", "app.tasks.models",
                "app.tasks.sandbox"):
        importlib.import_module(mod)  # every table, so create_all makes them all in the simulator database
    from app.db.session import Base, SessionLocal, engine
    from app.llm import spend
    from app.models import entities  # noqa: F401
    from app.tasks import models as _t  # noqa: F401

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    spend.configure(SessionLocal)
    judge = None if args.no_judge else make_judge(args.max_inr)
    results = []
    for m in models:
        print(f"── {m}")
        results.append(await run_candidate(m, args, judge))
    print("\nmodel                                   passed   judge  ₹/msg   p50 ms  p90 ms   ₹ total")
    for r in results:
        print(f"{r['model']:40} {r['passed']:>2}/{r['total']:<4} {r['judge'] or '-':>6}  {r['inr_per_message']:>5}  {r['p50_ms'] or '-':>7} {r['p90_ms'] or '-':>7}  {r['inr']:>7}")
    out = Path(__file__).resolve().parents[1] / "runs" / f"model_bench_{time.strftime('%Y%m%d_%H%M')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nWritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
