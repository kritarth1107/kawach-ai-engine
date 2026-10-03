"""Tuned Saheli vs the base model on the holdout families (never trained on). Paid: needs --yes and a cap.

    .venv/bin/python eval/compare_models.py HOLDOUT.jsonl --tuned ENDPOINT --base gemini-3.5-flash \
        --project P [--region us-central1] [--n 120] --max-inr 300 --yes

For each decision point (where Saheli acted after someone spoke), both models get the same history and answer:
  - tool agreement: did each call the same tools as the real, high-scoring turn (or answer in text when it did)?
  - judge preference: for text replies, the strong judge picks the better one (order shuffled, ties allowed).
The tuned model passes only if tool agreement is not worse (by more than 3 points) and the judge prefers it at least
as often as the base. Only then: eval/train.sh compare, then the 10% trial — never straight to production.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INR_PER_CALL = 0.6  # rough: ~3k tokens in, ~300 out on Flash; the judge call ~₹1.5
JUDGE = """Two replies from a WhatsApp care companion for elderly people in India, to the same message. Which is
better: warm, brief, in the person's language and script, correct, safe, does what was asked? Return only JSON
{"better": "A" | "B" | "tie", "why": "..."}"""


def decision_points(example: dict) -> list[dict]:
    """Each model step that directly follows a user text, with the history before it and what Saheli really did."""
    out, contents = [], example["contents"]
    for i, c in enumerate(contents):
        if c["role"] != "model" or i == 0:
            continue
        prev = contents[i - 1]
        if prev["role"] == "user" and any("text" in p for p in prev["parts"]):
            ref_calls = sorted(p["functionCall"]["name"] for p in c["parts"] if "functionCall" in p)
            ref_text = "".join(p.get("text", "") for p in c["parts"])
            out.append({"history": contents[:i], "calls": ref_calls, "text": ref_text, "system": example.get("systemInstruction"),
                        "tools": example.get("tools")})
    return out


def agrees(ref_calls: list[str], got_calls: list[str]) -> bool:
    """Same set of tools (order and repeats ignored); both empty means both answered in text."""
    return set(ref_calls) == set(got_calls)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "pass": False}
    tool_base = sum(r["base_agrees"] for r in rows) / n
    tool_tuned = sum(r["tuned_agrees"] for r in rows) / n
    prefs = [r["prefer"] for r in rows if r.get("prefer")]
    tuned_wins, base_wins = prefs.count("tuned"), prefs.count("base")
    return {"n": n, "tool_agreement": {"base": round(tool_base, 3), "tuned": round(tool_tuned, 3)},
            "judge": {"tuned": tuned_wins, "base": base_wins, "tie": prefs.count("tie")},
            "pass": tool_tuned >= tool_base - 0.03 and tuned_wins >= base_wins}


async def _ask(client, model: str, point: dict) -> tuple[list[str], str]:
    from google.genai import types

    cfg = {"max_output_tokens": 800, "automatic_function_calling": {"disable": True}}
    if point.get("system"):
        cfg["system_instruction"] = point["system"]["parts"][0]["text"]
    if point.get("tools"):
        cfg["tools"] = point["tools"]
    resp = await client.aio.models.generate_content(model=model, contents=point["history"], config=types.GenerateContentConfig(**cfg))
    parts = (resp.candidates[0].content.parts if resp.candidates and resp.candidates[0].content else None) or []
    return sorted(p.function_call.name for p in parts if p.function_call), "".join(p.text for p in parts if p.text and not p.thought)


async def _judge(client, model: str, said: str, a: str, b: str) -> str:
    resp = await client.aio.models.generate_content(model=model, contents=f"{JUDGE}\n\nThey said: {said}\n\nA: {a}\n\nB: {b}")
    from app.learn.grader import parse

    return str(parse(resp.text or "").get("better") or "tie")


async def run(argv: list[str]) -> int:
    from google import genai

    def flag(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    if "--yes" not in argv or not flag("--max-inr"):
        sys.exit("Paid: pass --max-inr N and --yes (the founder's OK).")
    project = flag("--project")
    if project == "kavach-care":
        sys.exit("Use the training project, not kavach-care.")
    n, cap = int(flag("--n", "120")), float(flag("--max-inr"))
    points = [p for line in open(argv[0]) for p in decision_points(json.loads(line))]
    random.Random(7).shuffle(points)
    points = points[: min(n, int(cap / (INR_PER_CALL * 2 + 1.5)))]
    print(f"{len(points)} decision points, est. ₹{len(points) * (INR_PER_CALL * 2 + 1.5):.0f} (cap ₹{cap:.0f})")
    client = genai.Client(vertexai=True, project=project, location=flag("--region", "us-central1"))
    base, tuned, judge = flag("--base", "gemini-3.5-flash"), flag("--tuned"), flag("--judge", "gemini-3.1-pro-preview")
    rows, rnd = [], random.Random(11)
    for i, p in enumerate(points):
        try:
            (bc, bt), (tc, tt) = await asyncio.gather(_ask(client, base, p), _ask(client, tuned, p))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {i}: {exc}")
            continue
        row = {"ref": p["calls"], "base": bc, "tuned": tc, "base_agrees": agrees(p["calls"], bc), "tuned_agrees": agrees(p["calls"], tc)}
        if not p["calls"] and bt and tt:
            said = "".join(x.get("text", "") for x in p["history"][-1]["parts"])
            flip = rnd.random() < 0.5
            pick = await _judge(client, judge, said, tt if flip else bt, bt if flip else tt)
            row["prefer"] = "tie" if pick not in ("A", "B") else ("tuned" if (pick == "A") == flip else "base")
            row["texts"] = {"base": bt[:400], "tuned": tt[:400]}
        rows.append(row)
    report = summarise(rows)
    out = Path(flag("--out", "eval/compare_report.json"))
    out.write_text(json.dumps({"summary": report, "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run(sys.argv[1:])))
