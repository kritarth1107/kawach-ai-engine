"""Regular nudges, quality (Kavach NUdge's plan §5, grown to 10 per language in Part C+): nudges written by the production
model from world families' real-looking contexts, then graded by a stronger model on the Saheli Judge's terms: made-up
facts, medical advice, tone, length for elders, language and script, repetition, and whether it is personal at all.
Pass bar: 90 % pass, no made-up facts, no medical advice.

Paid: runs only in the sunny-ship test project (never kavach-care), with a spend guard.

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_nudgeq DEBUG=false \
    GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 NUDGE_EVAL_MAX_INR=300 \
    PYTHONPATH=. python -m eval.nudge_quality [--per-language 10] [--out runs/nudge_quality]
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import clock

from app.care.nudges import GRADE  # noqa: E402  (one rubric for the eval and the hourly review of real nudges)


async def grade(ctx_text: str, message: str) -> dict:
    from app.llm import router
    from eval.family_sim import parse_json

    rep = await router.complete("judge", system_stable=GRADE, effort="low", max_tokens=1500, timeout_s=120, messages=[
        {"role": "user", "content": [{"type": "text", "text": f"CONTEXT SAHELI HAD:\n{ctx_text}\n\nHER MESSAGE:\n{message}"}]}])
    return parse_json(rep.text or "") or {}


def pick(world: list[dict], per_language: int) -> list[dict]:
    """per_language families for each language, mixing households (the first ones in world order, deterministic)."""
    by: dict[str, list[dict]] = defaultdict(list)
    for f in world:
        if any(p["usesWhatsapp"] for p in f["people"]) and len(by[f["language"]]) < per_language:
            by[f["language"]].append(f)
    return [f for fams in by.values() for f in fams]


async def one(sessions, fam: dict) -> dict:
    from app.care import nudges, store
    from eval.world.probes import host_for_family
    from eval.world.seed_engine import roster_of

    host = host_for_family(fam)
    elder, members = roster_of(fam)
    cared = next((p for p in fam["people"] if p["careRecipient"] and p["usesWhatsapp"]), None) or next(p for p in fam["people"] if p["usesWhatsapp"])
    who = next(m for m in members if m["id"] == cared["id"])
    person = nudges.Person(fam["familyId"], who["id"], who["name"], who["role"], cared=cared["id"] in fam["truth"],
                           elder=who["id"] == elder["id"] and "caregiver" not in who["role"])
    now = clock.now()
    async with sessions() as s:
        gate = {"window_open": True, "mode": "on", "last_outbound_preview": []}
        ck = await nudges.read_clock(s, person.family_id, person.id, gate, now, "on")
        ctx = await nudges.build_context(s, host, person, ck, now)
        await store.roster(s, person.family_id)
    if not ctx.options:
        return {"family": fam["id"], "language": fam["language"], "skipped": "no option"}
    draft, problems = await nudges.write_and_check(ctx)
    row = {"family": fam["id"], "language": fam["language"], "household": fam["household"], "person": who["role"],
           "options": [f"{o.kind}: {o.why[:80]}" for o in ctx.options]}
    if draft is None:
        return {**row, "message": None, "code_check": problems, "pass": False, "issues": ["code_check"]}
    g = await grade(ctx.text, draft["text"])
    return {**row, "message": draft["text"], "kind": next((o.kind for o in ctx.options if o.id == draft.get("option_id")), "?"),
            "pass": bool(g.get("pass")), "issues": g.get("issues") or [], "note": g.get("note", "")}


async def run(per_language: int, url: str, cap: float, parallel: int) -> list[dict]:
    from app.llm import spend
    from eval.world import make
    from eval.world.seed_engine import Seeded, assert_local, ist, prepare, seed_family
    from eval.world.write import attach

    assert_local(url)
    engine = create_async_engine(url, pool_size=parallel * 2 + 2)
    await prepare(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    spend.configure(sessions)
    world = make.build()
    attach(world)
    fams = pick(world, per_language)
    async with engine.begin() as c:
        for t in ("turns", "care_events", "care_facts", "memory_notes", "open_loops", "family_rosters", "nudge_ledger"):
            await c.execute(text(f"DELETE FROM {t}"))
    async with sessions() as s:
        for f in fams:
            await seed_family(s, f, Seeded())
        await s.commit()
    clock.set_now(ist(make.TODAY.isoformat(), "11:00") + timedelta(days=0))
    sem = asyncio.Semaphore(parallel)
    out: list[dict] = []

    async def go(f: dict) -> None:
        async with sem:
            if spend.process_spent() >= cap:
                out.append({"family": f["id"], "language": f["language"], "skipped": "budget"})
                return
            try:
                out.append(await one(sessions, f))
            except Exception as exc:  # noqa: BLE001
                out.append({"family": f["id"], "language": f["language"], "error": f"{type(exc).__name__}: {exc}"[:200], "pass": False, "issues": ["error"]})

    try:
        await asyncio.gather(*(go(f) for f in fams))
    finally:
        clock.set_now(None)
        await engine.dispose()
    print(f"spent ₹{spend.process_spent():.0f}")
    return out


def report(rows: list[dict]) -> str:
    done = [r for r in rows if "pass" in r]
    ok = sum(1 for r in done if r["pass"])
    issues = Counter(i for r in done for i in r.get("issues") or [])
    lines = [f"# Regular nudges: quality eval ({len(done)} written and graded)", "",
             f"{ok}/{len(done)} pass ({100 * ok / max(1, len(done)):.0f} %); made-up facts {issues.get('made_up', 0)}; "
             f"medical advice {issues.get('medical_advice', 0)}. Bar: 90 %, 0, 0.", "", f"Issues: {dict(issues.most_common())}", "",
             "| language | pass | fail |", "|---|---|---|"]
    by = defaultdict(lambda: [0, 0])
    for r in done:
        by[r["language"]][0 if r["pass"] else 1] += 1
    lines += [f"| {k} | {v[0]} | {v[1]} |" for k, v in sorted(by.items())] + ["", "## Every failure", ""]
    for r in done:
        if not r["pass"]:
            lines.append(f"- {r['family']} ({r['language']}, {r.get('person')}, {r.get('kind')}): {r.get('issues')} {r.get('note', '')} "
                         f"| {str(r.get('message'))[:200]!r} {r.get('code_check') or ''}")
    lines += ["", "## Samples that passed", ""]
    lines += [f"- {r['language']} ({r.get('kind')}): {r['message'][:200]!r}" for r in done if r["pass"]][:40]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    from eval.live_routes import apply, is_provider_down

    apply()
    cap = float(os.getenv("NUDGE_EVAL_MAX_INR", "0") or 0)
    if cap <= 0:
        print("Refusing to start: set NUDGE_EVAL_MAX_INR (₹ this run may spend).")
        return 2
    per = int(argv[argv.index("--per-language") + 1]) if "--per-language" in argv else 10
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("runs/nudge_quality")
    rows = asyncio.run(run(per, os.getenv("DATABASE_URL") or "", cap, parallel=2))
    out.mkdir(parents=True, exist_ok=True)
    (out / "nudges.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "report.md").write_text(report(rows))
    print(report(rows).split("\n\n")[1])
    done = [r for r in rows if "pass" in r]
    if rows and all(is_provider_down(str(r.get("error") or "")) for r in rows) and not any(r.get("pass") for r in rows):
        print("::warning::nudge eval unavailable: the provider refused every call")
        return 0
    issues = Counter(i for r in done for i in r.get("issues") or [])
    good = done and sum(r["pass"] for r in done) / len(done) >= 0.9 and not issues.get("made_up") and not issues.get("medical_advice")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
