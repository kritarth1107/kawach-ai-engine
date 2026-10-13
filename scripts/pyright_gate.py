"""Type-check gate: pyright errors may only go down.

The baseline (scripts/pyright_baseline.json) holds today's error count per file. CI fails when any
file has more errors than its baseline or a new file has any. Run with --update after fixing errors
to lower the baseline (never to raise it for new code).

    python scripts/pyright_gate.py            # check
    python scripts/pyright_gate.py --update   # rewrite the baseline from the current counts
"""

from __future__ import annotations

import collections
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "pyright_baseline.json"


def counts() -> tuple[dict[str, int], dict[str, list[str]]]:
    cmd = ["pyright", "--outputjson", "--pythonpath", sys.executable, "app"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False).stdout
    data = json.loads(out)
    n: dict[str, int] = collections.Counter()
    msgs: dict[str, list[str]] = collections.defaultdict(list)
    for d in data["generalDiagnostics"]:
        if d["severity"] != "error":
            continue
        f = str(pathlib.Path(d["file"]).resolve().relative_to(ROOT))
        n[f] += 1
        line = d["range"]["start"]["line"] + 1
        msgs[f].append(f"{f}:{line}: {d['message'].splitlines()[0]} ({d.get('rule', '')})")
    return dict(n), msgs


def main() -> int:
    now, msgs = counts()
    if "--update" in sys.argv:
        BASELINE.write_text(json.dumps(dict(sorted(now.items())), indent=1) + "\n")
        print(f"baseline written: {sum(now.values())} errors in {len(now)} files")
        return 0
    base = json.loads(BASELINE.read_text())
    worse = {f: c for f, c in now.items() if c > base.get(f, 0)}
    for f, c in sorted(worse.items()):
        print(f"::error file={f}::{c} type errors (baseline {base.get(f, 0)})")
        for m in msgs[f]:
            print("  " + m)
    fixed = sum(max(0, c - now.get(f, 0)) for f, c in base.items())
    print(f"pyright: {sum(now.values())} errors (baseline {sum(base.values())}); {fixed} fixed since baseline")
    if fixed:
        print("lower the baseline: python scripts/pyright_gate.py --update")
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
