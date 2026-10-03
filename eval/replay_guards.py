"""Replay recorded simulation transcripts through the reply guards (free: no model calls).

    .venv/bin/python eval/replay_guards.py [run_dir] [--freeze tests/fixtures/guard_replay.json]

For every Saheli message the judge graded, runs grounding, language and repetition checks with what Saheli
could know at that moment (the conversation so far), and reports how often each check fires on messages the
judge passed (false alarms: each costs a rewrite call) and on messages it failed. --freeze writes a small
fixed sample that tests/test_guard_replay.py re-checks in CI, so the guards cannot quietly get noisier.
"""

from __future__ import annotations

import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.brain import guards as g  # noqa: E402

DEFAULT_RUN = "/home/m4dm4x/OpenBot/Shared/kavach-sim/round3"
KNOWN_CHARS = 10**7  # all of it: in production Saheli also has the care record, notes and recall


def cases(run_dir: str):
    """Yield one dict per graded Saheli message with the context the guards need."""
    for f in sorted(glob.glob(f"{run_dir}/*.jsonl")):
        hist: list[str] = []
        by_person: dict[str, list[str]] = defaultdict(list)
        to_person: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for line in open(f):
            d = json.loads(line)
            verdicts = {v["n"]: v for v in d["verdicts"]}
            for m in d["messages"]:
                stamp = f"[{m['at']}] "
                if m["who"] != "saheli":
                    by_person[m["who"]].append(m["text"])
                    hist.append(stamp + m["text"])
                    continue
                v = verdicts.get(m["n"])
                if v:
                    day = m["at"][:6]
                    yield {
                        "family": Path(f).stem, "day": d["day"], "passed": bool(v["pass"]), "issues": v.get("issues") or [],
                        "text": m["text"], "known": "\n".join(hist)[-KNOWN_CHARS:],
                        "writes": by_person[m["to"]][-8:], "earlier": [t for a, t in to_person[m["to"]] if a == day][-8:],
                    }
                hist.append(stamp + m["text"])
                to_person[m["to"]].append((m["at"][:6], m["text"]))


def relevant(text: str, known: str, tail: int = 1500) -> str:
    """The lines of `known` that can ground `text` (sharing a number, time or condition word), plus the latest tail."""
    t = g.ascii_digits(text).lower()
    keys = set(g.numbers_in(t)) | {c for c in g.CONDITIONS if c in t}
    keep = [ln for ln in known.splitlines() if keys and any(k in g.ascii_digits(ln).lower() for k in keys)]
    return ("\n".join(keep)[-6000:] + "\n" + known[-tail:]).strip()


def check(c: dict) -> dict[str, list[str]]:
    return {
        "ground": g.ungrounded(c["text"], known=c["known"]),
        "lang": g.language_problems(c["text"], g.profile(c["writes"])),
        "repeat": g.repeats(c["text"], c["earlier"]),
        "leak": ["leaked reasoning"] if g.leaked_reasoning(c["text"]) else [],
    }


def rates(items) -> dict[str, dict[str, float]]:
    n = Counter()
    hit = Counter()
    for c in items:
        res = check(c)
        side = "passed" if c["passed"] else "failed"
        n[side] += 1
        for k, p in res.items():
            if p:
                hit[(k, side)] += 1
    return {k: {side: round(hit[(k, side)] / n[side], 4) if n[side] else 0.0 for side in ("passed", "failed")}
            for k in ("ground", "lang", "repeat", "leak")}


def main(argv: list[str]) -> int:
    freeze = None
    if "--freeze" in argv:
        i = argv.index("--freeze")
        freeze = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    run = argv[0] if argv else DEFAULT_RUN
    items = list(cases(run))
    r = rates(items)
    print(f"{len(items)} graded messages from {run}")
    print("check   | fires on passed (false alarm) | fires on failed")
    for k, v in r.items():
        print(f"{k:7} | {v['passed']:.1%} | {v['failed']:.1%}")
    if freeze:
        rnd = random.Random(7)
        passed = [c for c in items if c["passed"]]
        failed = [c for c in items if not c["passed"]]
        sample = rnd.sample(passed, min(150, len(passed))) + rnd.sample(failed, min(60, len(failed)))
        for c in sample:
            c["known"] = relevant(c["text"], c["known"])
            c["writes"] = [w[:300] for w in c["writes"][-4:]]
            c["earlier"] = [e[:400] for e in c["earlier"][-5:]]
        Path(freeze).write_text(json.dumps({"source": run, "rates_at_freeze": rates(sample), "cases": sample}, ensure_ascii=False))
        print(f"froze {len(sample)} cases to {freeze}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
