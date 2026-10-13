"""The nightly report (plan B3.6): merge every shard's probe results, compare with the previous night, write report.md.

    PYTHONPATH=. python -m eval.world.report <results dir> [--previous <previous results dir>] [--out report.md]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.world.probes import summary


def load(d: Path) -> list[dict]:
    rows = []
    for p in sorted(d.rglob("probes.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def main(argv: list[str]) -> int:
    d = Path(argv[0])
    prev = Path(argv[argv.index("--previous") + 1]) if "--previous" in argv else None
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else d / "report.md"
    rows = load(d)
    text = summary(rows, 0).replace(", 0 min", "")
    if prev and prev.exists():
        before = {r["probe"]: r for r in load(prev) if "pass" in r}
        now = {r["probe"]: r for r in rows if "pass" in r}
        new_fail = [p for p, r in now.items() if not r["pass"] and before.get(p, {}).get("pass")]
        fixed = [p for p, r in now.items() if r["pass"] and before.get(p, {}).get("pass") is False]
        text += f"\n## Compared with the previous night\n\n{len(new_fail)} newly failing, {len(fixed)} fixed.\n\n"
        for p in new_fail[:40]:
            r = now[p]
            text += f"- NEW FAIL {p} ({r['language']}, {r['topic']}): {r['fails']} | {r['text'][:100]!r} → {str(r.get('reply'))[:140]!r}\n"
    judged = [json.loads(line) for p in sorted(d.rglob("judged.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    unjudged = sum(1 for j in judged if j.get("skipped"))
    judged = [j for j in judged if not j.get("skipped")]
    if unjudged:
        text += f"\n{unjudged} picked replies were not judged: the judge's spend guard was reached.\n"
    if judged:
        from collections import Counter

        ok = sum(1 for j in judged if (j.get("judge") or {}).get("pass"))
        issues = Counter(i for j in judged for i in (j.get("judge") or {}).get("issues") or [])
        text += f"\n## Judge (stronger model, safety + failures + 30 % sample)\n\n{ok}/{len(judged)} judged good. Issues: {dict(issues.most_common(10))}\n\n"
        notes = {r["probe"]: r for r in rows}
        for j in judged:
            g = j.get("judge") or {}
            if not g.get("pass"):
                r = notes.get(j["probe"], {})
                text += f"- {j['probe']} ({r.get('language')}, {r.get('topic')}) score {g.get('score')}: {g.get('issues')} {g.get('note', '')}\n"
    fails = [r for r in rows if r.get("pass") is False]
    text += "\n## Every failure\n\n"
    for r in fails[:300]:
        text += (f"- {r['probe']} ({r['household']}, {r['language']}, {r['topic']}, {r.get('who')}): {r['fails']}\n"
                 f"  - said: {r['text'][:160]!r}\n  - Saheli: {str(r.get('reply'))[:240]!r}\n  - tools: {r.get('tools')}\n")
    out.write_text(text)
    passed = sum(1 for r in rows if r.get("pass"))
    total = sum(1 for r in rows if "pass" in r)
    print(f"{passed}/{total} probes passed; report at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
