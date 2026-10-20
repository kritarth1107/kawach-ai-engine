"""Fail a push whose commits touch another owner's files without Reviewed-by.

Reads the ```owners block in OWNERS.md. Longest path prefix wins. A file with no row is unowned.
Every commit needs Author-agent. A commit needs Reviewed-by for each owner of its files other than
the author.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def rules(text: str) -> list[tuple[str, str]]:
    start = text.find("```owners")
    if start < 0:
        raise SystemExit("OWNERS.md has no owners block")
    body = text[start + len("```owners"):]
    end = body.find("```")
    rows = []
    for line in body[:end].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, owner = line.split("\t", 1)
        rows.append((path.strip(), owner.strip()))
    rows.sort(key=lambda row: len(row[0]), reverse=True)
    return rows


def owner_of(path: str, rows: list[tuple[str, str]]) -> str | None:
    for prefix, owner in rows:
        if path == prefix or path.startswith(prefix):
            return owner
    return None


# An author whose changes to these paths are reviewed on the pull request by the founder, not by the owner: the nightly
# trainer (eval/trainer.py) may only edit Saheli's instruction text, and its PR is merged by the founder.
DELEGATES = {"Saheli Trainer": ("app/brain/persona/",)}


def needed(files: list[str], message: str, rows: list[tuple[str, str]]) -> list[str]:
    author = ""
    for line in message.splitlines():
        if line.startswith("Author-agent:"):
            author = line.split(":", 1)[1].strip()
    problems = []
    if not author:
        problems.append("missing Author-agent")
    owners = []
    for path in files:
        if path.startswith(DELEGATES.get(author, ())):
            continue
        who = owner_of(path, rows)
        if who and who != author and who not in owners:
            owners.append(who)
    reviewed = [line.split(":", 1)[1].strip() for line in message.splitlines() if line.startswith("Reviewed-by:")]
    for who in owners:
        if who not in reviewed:
            problems.append(f"missing Reviewed-by: {who}")
    return problems


def commits() -> list[str]:
    before = os.environ.get("BEFORE", "")
    sha = os.environ.get("SHA", "HEAD")
    if not before or set(before) <= {"0"}:
        return [sha]
    out = subprocess.check_output(["git", "rev-list", f"{before}..{sha}"], text=True)
    found = [line for line in out.split() if line]
    return found or [sha]


def main() -> int:
    rows = rules(Path("OWNERS.md").read_text())
    failed = False
    for sha in commits():
        parents = subprocess.check_output(["git", "log", "-1", "--format=%P", sha], text=True).split()
        if len(parents) > 1:  # a merge commit: its parents' commits are checked, it adds no change of its own
            continue
        message = subprocess.check_output(["git", "log", "-1", "--format=%B", sha], text=True)
        names = subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha], text=True)
        files = [line for line in names.splitlines() if line]
        bad = needed(files, message, rows)
        if bad:
            failed = True
            print(f"{sha[:12]}: {', '.join(bad)}")
            for path in files:
                who = owner_of(path, rows)
                if who:
                    print(f"  {path}  {who}")
    if failed:
        return 1
    print("owners ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
