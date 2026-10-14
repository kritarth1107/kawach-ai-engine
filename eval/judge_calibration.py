"""Is the full-memory judge good enough to go live? (trust plan C1.4) Paid; live Bedrock routes, never Vertex.

30 hand-labelled replies on an invented family (eval/fixtures/judge_calibration.jsonl): 15 good (including hard goods: a
fact from the notes, tool-sourced information with a check-with-your-doctor, "I don't know", red-flag safety steps, a
correction respected) and 15 bad (made-up facts, her own medical advice, an avoided name, wrong time/medicine/person/
language, an ignored question, a broken promise, a correction ignored, a lecture, a missed alert). Each goes to the
production judge exactly as a real reply would (app.quality.judge: JUDGE_PROMPT, the household and memory block, the same
payload, route `judge` from deploy-gcp.yml, effort medium). Agreement = the judge's pass/fail equals the label; the bar is
85 %. It also reports whether a bad reply was failed for the labelled issue.

    GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 CALIBRATION_MAX_INR=60 \
    PYTHONPATH=. python -m eval.judge_calibration [--repeat 1] [--out runs/judge_calibration.json]
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")

import asyncio
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "judge_calibration.jsonl"
BAR = 0.85


def inputs(c: dict) -> tuple[list[str], str]:
    from app.quality.judge import JUDGE_PROMPT

    system = [JUDGE_PROMPT, f"{c['household']}\n\nMEMORY AS IT STOOD AT 13 Oct 2026 {c.get('at', '09:30')} IST:\n{c['record']}\n\n{c['notes']}"]
    payload = {"message_to": c["message_to"], "kind": "reply" if c["said"] else "proactive", "situation": "", "language_setting": c["language"],
               "writes_in": "latin-indic", "today_events_so_far": "TODAY SO FAR (IST, from the ledger; this is what actually happened):\n" + "\n".join(f"  {e}" for e in c["events"]),
               "conversation_before (oldest first, last 3 days)": c["conversation"],
               "what they said just now": c["said"] or "(Saheli started this message; nobody had just written)",
               "tool calls and results in this turn": c["tools"], "SAHELI'S MESSAGE TO GRADE": c["reply"]}
    return system, json.dumps(payload, ensure_ascii=False)


async def grade(c: dict) -> dict:
    from app.llm import router
    from app.quality import judge

    system, user = inputs(c)
    for _ in range(2):
        rep = await router.complete("judge", system_stable=system, messages=[{"role": "user", "content": [{"type": "text", "text": user}]}],
                                    max_tokens=4000, effort=judge.effort(), timeout_s=180)
        v = judge.normalise(judge.parse_json(rep.text))
        if v["parsed"]:
            return {"id": c["id"], "label": c["label"], "issue": c["issue"], "passed": v["passed"], "found": v["found"], "summary": v["summary"], "model": rep.model}
    return {"id": c["id"], "label": c["label"], "issue": c["issue"], "passed": None, "found": [], "summary": "not parseable", "model": rep.model}


async def run(repeat: int) -> dict:
    from app.llm import spend

    cases = [json.loads(line) for line in FIXTURES.read_text().splitlines() if line.strip()]
    sem = asyncio.Semaphore(8)

    async def one(c: dict) -> dict:
        async with sem:
            return await grade(c)

    rows = []
    for _ in range(repeat):
        rows += await asyncio.gather(*(one(c) for c in cases))
    judged = [r for r in rows if r["passed"] is not None]
    agree = [r for r in judged if r["passed"] == (r["label"] == "good")]
    bad = [r for r in judged if r["label"] == "bad"]
    return {"cases": len(cases), "runs": len(rows), "judged": len(judged), "agreement": round(len(agree) / len(judged), 3) if judged else None,
            "bar": BAR, "passes_bar": bool(judged) and len(agree) / len(judged) >= BAR,
            "false_fail": [r["id"] for r in judged if r["label"] == "good" and not r["passed"]],
            "false_pass": [r["id"] for r in judged if r["label"] == "bad" and r["passed"]],
            "issue_matched": f"{sum(1 for r in bad if r['issue'] in r['found'])}/{len(bad)}",
            "route": (json.loads(os.environ.get("MODEL_ROUTES") or "{}").get("judge") or ["default"])[0],
            "spent_inr": round(spend.process_spent(), 2), "rows": rows}


def main(argv: list[str]) -> int:
    from eval.world.probes import production_routes

    if os.getenv("GCP_PROJECT_ID") == "kavach-care":
        print("Refusing: paid runs go to the sunny-ship test project, never kavach-care.")
        return 2
    if not os.getenv("MODEL_ROUTES") and production_routes():
        os.environ["MODEL_ROUTES"] = production_routes() or ""  # the production judge, exactly
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 1
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("runs/judge_calibration.json")
    res = asyncio.run(run(repeat))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"judge {res['route']}: agreement {res['agreement']} on {res['judged']} (bar {BAR}); false fails {res['false_fail']}; "
          f"false passes {res['false_pass']}; issue matched {res['issue_matched']}; ₹{res['spent_inr']}")
    return 0 if res["passes_bar"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
