"""Mutation check: break each safety rule on purpose and prove a test (or the ordering bench) catches it. Free.

    PYTHONPATH=. .venv/bin/python eval/mutation_check.py          # all
    PYTHONPATH=. .venv/bin/python eval/mutation_check.py skill    # only mutations whose name contains "skill"

Each mutation swaps one exact line of code, runs the tests named for it, and restores the file. A mutation that the
tests do NOT catch means that safety rule has no real test: exit code 1. Never run two of these (or a test run) at the
same time: they share the test database and the working tree.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


@dataclass
class Mutation:
    name: str
    file: str
    old: str
    new: str
    tests: list[str]  # pytest node ids / -k filters, or ["BENCH"] for the ordering bench


MUTATIONS = [
    # memory undo
    Mutation("undo-ends-medicine-without-ok", "app/care/versions.py",
             'plan.needs_ok = not caregiver or (plan.action in ("stop", "restart") and not confirmed)', "plan.needs_ok = not caregiver",
             ["tests/test_memory_versions.py"]),
    Mutation("undo-approval-merges-old-fields", "app/care/store.py",
             'if not (row.source_ref or "").startswith("undo:"):', "if True:", ["tests/test_memory_versions.py"]),
    Mutation("undo-other-family-version", "app/care/versions.py",
             "return row if row and row.family_id == family_id else None", "return row", ["tests/test_memory_versions.py"]),
    Mutation("reject-turns-reminders-off", "app/brain/tools.py",
             'if row and row.domain == "medicine" and a["approve"] and row.status in ("active", "stopped"):',
             'if row and row.domain == "medicine":', ["tests/test_memory_versions.py"]),
    Mutation("viewer-confirms-change", "app/brain/tools.py",
             'raise ToolRefused("Only a caregiver (primary or co-caregiver) can confirm this change.")', "pass",
             ["tests/test_memory_versions.py"]),
    Mutation("forget-restore-crosses-people", "app/care/memory_upkeep.py",
             'if item.get("subject") in mine:', "if True:", ["tests/test_memory_versions.py"]),
    Mutation("prune-drops-compare-base", "app/care/versions.py",
             "MemoryVersion.id.not_in(keep)", "MemoryVersion.id > 0", ["tests/test_memory_versions.py"]),
    # family skills
    Mutation("skill-ai-instructions", "app/care/skillbook.py",
             'checks.append((AI_INSTRUCTIONS, "reads like an instruction to Saheli, not how they like things"))', "pass",
             ["tests/test_skill_safety.py"]),
    Mutation("skill-safety-steering", "app/care/skillbook.py",
             "checks.append((SAFETY_DENY, \"touches who gets told or how a symptom is treated; Saheli's safety rules decide that\"))", "pass",
             ["tests/test_skill_safety.py"]),
    Mutation("skill-care-record-conflicts", "app/care/skillbook.py",
             "return problems(text, family=True) + await record_conflicts(session, family_id, subject_id, text)",
             "return problems(text, family=True)", ["tests/test_skill_safety.py"]),
    Mutation("skill-render-recheck", "app/care/skillbook.py",
             "if not problems(s.body, family=True) and not conflicts_with(facts, s.body)]", "]", ["tests/test_skill_safety.py"]),
    Mutation("skills-in-emergencies", "app/brain/loop.py",
             '(to.get("name") or "").split(" ")[0], safety=safety)', '(to.get("name") or "").split(" ")[0])',
             ["tests/test_skill_safety.py"]),
    Mutation("viewer-sets-skills", "app/brain/tools.py",
             'return not (self.speaker_is_elder or self.is_system) and "caregiver" in str(self.speaker.get("role", "")).lower()',
             "return not (self.speaker_is_elder or self.is_system)", ["tests/test_skill_safety.py"]),
    # store skills and the browser sandbox
    Mutation("store-hints-unfiltered", "app/care/skillbook.py",
             "rows = [s for s in rows if store_skill_ok(s) and (s.uses < 4 or _rate(s) >= MIN_STORE_RATE)]", "rows = rows", ["BENCH"]),
    Mutation("placed-before-yes", "app/tasks/runtime.py",
             'if task.phase in ("prepare", "otp") and (out.get("placed") or out.get("booked") or out.get("order_id") or out.get("ride_id")):',
             "if False:", ["BENCH"]),
    Mutation("profile-reuse-history", "app/tasks/sandbox.py",
             "owners = {(o.family_id, o.service) for o in other} | {(f, s) for f, s in seen}",
             "owners = {(o.family_id, o.service) for o in other}", ["tests/test_sandbox.py"]),
    Mutation("sweeper-kills-waiting-order", "app/tasks/sandbox.py",
             'IN_USE = ("queued", "running", "needs_input", "awaiting_confirm")', 'IN_USE = ("queued", "running", "needs_input")',
             ["tests/test_sandbox.py"]),
    # red flags
    Mutation("hinglish-chest-pain", "app/brain/guards.py", r'r"\b(chest|seen[ae]|chh?aa?ti|heart|breath\w*|saans|',
             r'r"\b(chest|breath\w*|saans|', ["tests/test_review_fixes.py"]),
    # voice notes
    Mutation("unsure-voice-changes-dose", "app/brain/tools.py",
             "force_confirm=ctx.voice_unsure and domain in HEALTH_DOMAINS,", "force_confirm=False,", ["tests/test_voice.py"]),
    Mutation("voice-note-not-flagged", "app/brain/loop.py", "        voice_block(req),\n", "", ["tests/test_voice.py"]),
    # backups
    Mutation("restore-over-data", "backup/backup.py",
             "        if not overwrite and has_data(name, target):", "        if False:", ["tests/test_backup.py"]),
    Mutation("failed-rerun-replaces-good", "backup/backup.py",
             'if manifest["ok"] or not (old and old.get("ok")):', "if True:", ["tests/test_backup.py"]),
]


def run(tests: list[str]) -> bool:
    """True if the tests pass."""
    if tests == ["BENCH"]:
        out = subprocess.run([PY, "eval/agent_bench.py"], cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": ".", **_env()})
        return "Release gate: PASS" in out.stdout
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", *tests], cwd=ROOT, capture_output=True, text=True, env=_env()).returncode == 0


def _env() -> dict:
    import os

    return dict(os.environ)


def main(argv: list[str]) -> int:
    pick = [m for m in MUTATIONS if not argv or any(a in m.name for a in argv)]
    missed, broken = [], []
    for m in pick:
        path = ROOT / m.file
        src = path.read_text()
        if src.count(m.old) != 1:
            broken.append(m.name)
            print(f"  ?     {m.name}: the line to mutate is gone or not unique ({m.file}); update this check")
            continue
        try:
            path.write_text(src.replace(m.old, m.new))
            caught = not run(m.tests)
        finally:
            path.write_text(src)
        print(f"  {'ok' if caught else 'MISS'}  {m.name}")
        if not caught:
            missed.append(m.name)
    print(f"\n{len(pick) - len(missed) - len(broken)}/{len(pick)} mutations caught" + (f"; missed: {', '.join(missed)}" if missed else "")
          + (f"; stale: {', '.join(broken)}" if broken else ""))
    return 1 if missed or broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
