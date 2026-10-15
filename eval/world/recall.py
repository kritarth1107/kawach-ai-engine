"""Recall on the test world (trust plan C2.2, A2): the story families' memory quiz, scored at last, and their memory score.

Each story family (eval/world/stories.py) has 45 days of WhatsApp history in which facts are told and changed, and a
10-question quiz at the end ("What dose of X does she take now?", "What is the cat's name?", ...). Tonight's story
families (a rotation of RECALL_FAMILIES a night, all 100 every few nights):
1. are seeded as they are today (record, notes) with their history as real chat turns at the times it was said;
2. go through Saheli's nightly re-read of those chats (the production extractor), as every family does at 02:00;
3. ask the quiz, one message per question, through the real brain (run_turn with SimHost), as the family member who asks;
4. each answer is scored: by code where it can confirm the fact (every word or number of the expected answer is in her
   reply), else by the judge (another script, other words); code never marks an answer wrong on its own;
5. get the production memory score (app.quality.memory_score: stored, recalled, used) on the same seeded memory.
Results: recall.jsonl (one row per question) and recall_meta_0.json beside the probes, read by eval/world/report.py.

Paid runs call the live Bedrock routes in ap-south-1, never Vertex.

    DATABASE_URL=... WEB_SEARCH_PROVIDER=off RECALL_MAX_INR=600 PYTHONPATH=. python -m eval.world.recall full [--families 25]
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import json
import re
import sys
import time
import uuid
from datetime import date, timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import clock
from eval.world import make
from eval.world.probes import OUT, host_for_family, night_index
from eval.world.seed_engine import BRAIN_ROLE, TODAY, Seeded, assert_local, ist, prepare, roster_of, seed_family
from eval.world.slice import slice_ids

RECALL_FAMILIES = int(os.getenv("RECALL_FAMILIES", "25"))
KINDS = [("What dose of", "medicine_dose"), ("Is ", "allergy"), ("Who is", "doctor"), ("How many regular medicines", "medicine_count"),
         ("What changed", "arc"), ("What did the tests", "arc"), ("Why was", "arc"), ("Who else started", "arc"), ("Where did", "arc"),
         ("What sad news", "arc")]

SCORE_PROMPT = """You check a care companion's answers to a memory quiz. For each item you get the QUESTION, the EXPECTED
answer (the truth) and Saheli's REPLY (in the family's own language and script). Decide whether the reply gives the
expected fact correctly: other words, language or script are fine; a wrong, missing or invented value is not; "I don't
know" is not correct. Return ONLY JSON: {"items": [{"n": 1, "correct": true, "why": "short"}]}"""


def kind_of(q: str) -> str:
    for prefix, kind in KINDS:
        if q.startswith(prefix):
            return kind
    return "life_detail"


def tonight_families(fams: list[dict], n: int, day: date | None = None) -> list[dict]:
    story = sorted((f for f in fams if f["tier"] == "story" and (f.get("story") or {}).get("quiz")), key=lambda f: f["id"])
    if n <= 0 or len(story) <= n:
        return story
    chunks = (len(story) + n - 1) // n
    k = night_index(day) % chunks
    return story[k * n:(k + 1) * n]


def code_confirms(expected: str, reply: str) -> bool:
    """Every word (3+ letters) and number of the expected answer is in her reply. Only ever confirms."""
    from app.brain import guards

    exp = guards.ascii_digits(expected or "").lower()
    rep = guards.ascii_digits(reply or "").lower()
    try:
        parsed = json.loads(expected)
        if isinstance(parsed, dict):
            exp = " ".join(str(v) for v in parsed.values() if v).lower()
    except (json.JSONDecodeError, TypeError):
        pass
    nums = re.findall(r"\d+(?:\.\d+)?", exp)
    words = [w for w in re.findall(r"[a-z]{3,}", exp) if w not in ("and", "the", "with", "from", "mg", "at")]
    if not nums and not words:
        return False
    return all(n in re.findall(r"\d+(?:\.\d+)?", rep) for n in nums) and all(w in rep for w in words)


async def judge_answers(items: list[dict]) -> dict[int, bool | None]:
    from app.llm import router
    from eval.family_sim import parse_json

    if not items:
        return {}
    text = json.dumps([{"n": i["n"], "QUESTION": i["q"], "EXPECTED": i["a"], "REPLY": i["reply"]} for i in items], ensure_ascii=False)
    try:
        rep = await router.complete("judge", system_stable=SCORE_PROMPT, messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                                    max_tokens=6000, effort="low", timeout_s=150)
        got = {int(x["n"]): bool(x.get("correct")) for x in parse_json(rep.text).get("items") or [] if str(x.get("n", "")).isdigit()}
    except Exception:  # noqa: BLE001 — not graded: counted as an error, never as correct
        got = {}
    return {i["n"]: got.get(i["n"]) for i in items}


async def seed_history(session, fam: dict) -> int:
    from app.care import store

    people = {p["key"]: p for p in fam["people"]}
    n = 0
    for x in fam["story"]["history"]:
        who = people.get(x["who"])
        if not who or not x.get("text"):
            continue
        clock.set_now(ist(x["day"], x["time"]))
        await store.add_turn(session, family_id=fam["familyId"], thread_id=who["id"], role="user", text=x["text"], speaker_id=who["id"],
                             message_ref=f"story-{fam['id']}-{x['id']}")
        if x.get("saheli"):
            clock.set_now(clock.now() + timedelta(minutes=1))
            await store.add_turn(session, family_id=fam["familyId"], thread_id=who["id"], role="assistant", text=x["saheli"])
        n += 1
    return n


async def run_family(sessions: async_sessionmaker, fam: dict, cap: float) -> tuple[list[dict], dict]:
    from app.brain.loop import TurnRequest, run_turn
    from app.care import store
    from app.care.extract import extract_family
    from app.llm import spend
    from app.quality import memory_score

    fid = fam["familyId"]
    rows: list[dict] = []
    if spend.process_spent() >= cap:
        return [{"family": fam["id"], "quiz": "extract", "skipped": "budget"}], {"family": fam["id"], "skipped": "budget"}
    async with sessions() as s:
        await seed_family(s, fam, Seeded())
        told = await seed_history(s, fam)
        await s.commit()
    clock.set_now(ist(TODAY.isoformat(), "02:00"))
    for _ in range(12):  # the nightly re-read over the chats, in batches
        if spend.process_spent() >= cap:
            print(f"::error::recall spend cap ₹{cap:.0f} reached during the re-read (spent ₹{spend.process_spent():.0f})", flush=True)
            break
        async with sessions() as s:
            got = await extract_family(s, fid)
            await s.commit()
        if not isinstance(got, dict) or got.get("skipped"):
            break
    elder, members = roster_of(fam)
    people = {p["key"]: p for p in fam["people"]}
    host = host_for_family(fam)
    asked = []
    for i, z in enumerate(fam["story"]["quiz"]):
        if spend.process_spent() >= cap:
            rows.append({"family": fam["id"], "quiz": z["id"], "skipped": "budget"})
            continue
        who = people.get(z["by"]) or people[next(iter(people))]
        text = z.get("text") or z["q"]
        clock.set_now(ist(TODAY.isoformat(), "20:30") + timedelta(minutes=2 * i))
        try:
            async with sessions() as s:
                res = await run_turn(s, host, TurnRequest(family_id=fid, elder=elder, speaker={"id": who["id"], "name": who["name"], "role": BRAIN_ROLE[who["role"]]},
                                                          members=members, text=text, message_ref=f"quiz-{fam['id']}-{z['id']}-{uuid.uuid4().hex[:6]}"))
            reply = res.reply or ""
        except Exception as exc:  # noqa: BLE001
            reply = ""
            rows.append({"family": fam["id"], "quiz": z["id"], "error": f"{type(exc).__name__}: {exc}"[:200], "correct": None})
            continue
        asked.append({"n": i + 1, "id": z["id"], "q": z["q"], "a": z["a"], "text": text, "reply": reply, "kind": kind_of(z["q"])})
    by_code = {x["n"]: True for x in asked if code_confirms(x["a"], x["reply"])}
    by_judge = await judge_answers([x for x in asked if x["n"] not in by_code])
    for x in asked:
        correct = True if x["n"] in by_code else by_judge.get(x["n"])
        rows.append({"family": fam["id"], "quiz": x["id"], "kind": x["kind"], "language": fam["language"], "q": x["q"], "expected": x["a"],
                     "text": x["text"], "reply": x["reply"][:600], "correct": correct, "by": "code" if x["n"] in by_code else "judge"})
    mem: dict = {}
    try:
        async with sessions() as s:
            roster = await store.roster(s, fid)
            if roster is not None:
                got = await memory_score.score_family(s, roster, (TODAY - timedelta(days=1)).isoformat(), ask=False, cases=False)
                mem = {k: got.get(k) for k in ("people", "score", "stored", "recalled", "used", "unknowns", "failures")}
            await s.rollback()
    except Exception as exc:  # noqa: BLE001
        mem = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return rows, {"family": fam["id"], "told": told, "memory": mem}


async def main_async(ids: list[str], n: int, parallel: int, url: str, cap: float) -> int:
    from app.api import brain as brain_api
    from app.brain.tools import set_task_agent
    from app.llm import spend
    from app.sim.agent import FakeAgent
    from eval.world.write import attach

    assert_local(url)
    engine = create_async_engine(url, pool_size=max(5, parallel * 2))
    await prepare(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    spend.configure(sessions)
    brain_api.kick_tasks = lambda: None
    set_task_agent(FakeAgent())
    world = make.build()
    attach(world)
    from eval.world.stories import attach_stories

    attach_stories(world)
    est = float(os.getenv("EST_INR_PER_RECALL_FAMILY", "8"))
    afford = max(1, int(cap // est)) if cap > 0 and est > 0 else n
    n = min(n, afford)
    fams = tonight_families([f for f in world if f["id"] in set(ids)], n)
    print(f"recall: {len(fams)} story families tonight, cap ₹{cap:.0f}", flush=True)
    rows: list[dict] = []
    fam_meta: list[dict] = []
    sem = asyncio.Semaphore(parallel)
    t0 = time.monotonic()

    async def one(f: dict) -> None:
        async with sem:
            try:
                r, m = await run_family(sessions, f, cap)
            except Exception as exc:  # noqa: BLE001 — one family must not stop the night
                r, m = [{"family": f["id"], "error": f"{type(exc).__name__}: {exc}"[:200], "correct": None}], {"family": f["id"], "error": str(exc)[:200]}
            rows.extend(r)
            fam_meta.append(m)

    await asyncio.gather(*(one(f) for f in fams))
    clock.set_now(None)
    await engine.dispose()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recall.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    scored = [m["memory"] for m in fam_meta if isinstance(m.get("memory"), dict) and m["memory"].get("score") is not None]
    answered = [r for r in rows if r.get("correct") is not None]
    meta = {"families": len(fams), "questions": len(answered), "correct": sum(1 for r in answered if r["correct"]),
            "errors": sum(1 for r in rows if r.get("correct") is None), "spent": round(spend.process_spent(), 2), "cap": cap,
            "minutes": round((time.monotonic() - t0) / 60, 1),
            "memory_score": {"families": len(scored), "min": min((m["score"] for m in scored), default=None),
                             "mean": round(sum(m["score"] for m in scored) / len(scored), 1) if scored else None,
                             "stored_mean": round(sum((m.get("stored") or 0) for m in scored) / len(scored), 1) if scored else None,
                             "recalled_mean": round(sum((m.get("recalled") or 0) for m in scored) / len(scored), 1) if scored else None},
            "per_family": fam_meta}
    (OUT / "recall_meta_0.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    pct = 100 * meta["correct"] / max(1, meta["questions"])
    print(f"recall {meta['correct']}/{meta['questions']} ({pct:.1f} %), {meta['errors']} not scored; memory score mean {meta['memory_score']['mean']}; "
          f"spent ₹{meta['spent']:.0f} of ₹{cap:.0f} in {meta['minutes']} min")
    skipped = sum(1 for r in rows if r.get("skipped") == "budget")
    if skipped or meta["spent"] > cap + 1:
        print(f"::error::recall spend cap ₹{cap:.0f} reached; stopped with {skipped} questions not asked (spent ₹{meta['spent']:.0f})", flush=True)
    return 0


def main(argv: list[str]) -> int:
    from eval.live_routes import apply

    apply()
    cap = float(os.getenv("RECALL_MAX_INR", "0") or 0)
    if cap <= 0:
        print("Refusing to start: set RECALL_MAX_INR (₹ this run may spend).")
        return 2
    n = RECALL_FAMILIES
    if "--families" in argv:
        i = argv.index("--families")
        n = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    par = 5
    if "--parallel" in argv:
        i = argv.index("--parallel")
        par = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    ids = argv if argv and argv[0].startswith("w") else slice_ids(argv[0] if argv else "ci")
    return asyncio.run(main_async(ids, n, par, os.getenv("DATABASE_URL") or "", cap))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
