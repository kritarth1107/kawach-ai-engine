"""Lightweight Saheli caregiver eval harness."""

from __future__ import annotations

import json
from pathlib import Path


def load_cases(path: Path | None = None) -> list[dict]:
    fixture = path or Path(__file__).parent / "fixtures" / "caregiver_cases.jsonl"
    cases: list[dict] = []
    for line in fixture.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def score_reply(case: dict, reply: str, tool_trace: list[dict] | None = None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    lower = reply.lower()
    tool_names = {row.get("tool", "") for row in (tool_trace or [])}

    for token in case.get("expect", []):
        if token.startswith("tool:"):
            tool = token.split(":", 1)[1]
            if tool not in tool_names:
                reasons.append(f"missing tool {tool}")
        elif token not in lower:
            reasons.append(f"missing '{token}'")

    for token in case.get("forbid", []):
        if token in lower:
            reasons.append(f"forbidden '{token}'")

    return (len(reasons) == 0, reasons)


def main() -> int:
    cases = load_cases()
    passed = 0
    for case in cases:
        ok, reasons = score_reply(case, case.get("sample_reply", ""), case.get("sample_tools"))
        if ok:
            passed += 1
        else:
            print(f"FAIL: {case['prompt']} -> {reasons}")
    print(f"{passed}/{len(cases)} fixture rules validated")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
