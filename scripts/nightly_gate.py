"""The engine deploy gate on the nightly world score (trust plan C2.3, founder decision D8).

An engine deploy is refused while the latest full nightly run on master failed. Exceptions:
- the commit is the fix: its message contains [fixes-nightly];
- the founder's override: the label `nightly-override` on the pull request of this commit, added by the founder
  (FOUNDER_LOGIN, the repository owner by default), or a manual deploy run by the founder with override_nightly=true.
Only full nights count (run name "Nightly full ..." on master, scheduled or started by hand); fixture runs, slices and
branches never gate deploys; a cancelled run is skipped; no full night yet means nothing to gate on.

    GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo GITHUB_SHA=... python scripts/nightly_gate.py [--dry-run]
    (--dry-run prints the decision and exits 0 either way; DRY_LATEST=failure|success fakes the latest night)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

WORKFLOW = "nightly-saheli.yml"
LABEL = "nightly-override"
FIX_TAG = "[fixes-nightly]"


def counts(run: dict) -> bool:
    title = str(run.get("display_title") or run.get("name") or "")
    return (run.get("head_branch") in ("master", "main") and run.get("event") in ("schedule", "workflow_dispatch")
            and title.startswith("Nightly full") and "fixture" not in title.lower() and run.get("conclusion") in ("success", "failure"))


def latest_night(runs: list[dict]) -> dict | None:
    full = [r for r in runs if counts(r)]
    full.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return full[0] if full else None


def founder_labelled(prs: list[dict], events_of, founder: str) -> bool:
    """A PR of this commit carries the override label, and the founder is the one who added it."""
    for pr in prs:
        if LABEL not in {lb.get("name") for lb in pr.get("labels") or []}:
            continue
        added = [e for e in events_of(pr["number"]) if e.get("event") == "labeled" and (e.get("label") or {}).get("name") == LABEL]
        if added and (added[-1].get("actor") or {}).get("login", "").lower() == founder.lower():
            return True
    return False


def decide(runs: list[dict], *, message: str, prs: list[dict], events_of, founder: str, actor: str = "", override_input: bool = False) -> tuple[bool, str]:
    night = latest_night(runs)
    if night is None:
        return True, "no full nightly on master yet: nothing to gate on"
    where = f"run {night.get('id')} ({night.get('created_at')}, {night.get('html_url', '')})"
    if night["conclusion"] == "success":
        return True, f"the latest nightly passed: {where}"
    if FIX_TAG in (message or ""):
        return True, f"the latest nightly failed ({where}), and this commit is marked {FIX_TAG}"
    if override_input and actor.lower() == founder.lower():
        return True, f"the latest nightly failed ({where}); the founder ({actor}) overrode it for this deploy"
    if founder_labelled(prs, events_of, founder):
        return True, f"the latest nightly failed ({where}); the founder added the '{LABEL}' label"
    return False, (f"the latest nightly FAILED ({where}). Engine deploys wait until a commit with {FIX_TAG} in its message, or the founder's "
                   f"'{LABEL}' label on its pull request, or a manual deploy by the founder with override_nightly=true.")


def _get(url: str, token: str) -> object:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                               "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (api.github.com)
        return json.loads(r.read())


def main(argv: list[str]) -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "kritarth1107/kawach-ai-engine")
    token = os.environ.get("GITHUB_TOKEN", "")
    sha = os.environ.get("GITHUB_SHA", "")
    founder = os.environ.get("FOUNDER_LOGIN") or repo.split("/")[0]
    api = f"https://api.github.com/repos/{repo}"
    if os.environ.get("DRY_LATEST"):
        runs = [{"id": 0, "head_branch": "master", "event": "schedule", "display_title": "Nightly full (dry run)", "conclusion": os.environ["DRY_LATEST"],
                 "created_at": "now"}]
    else:
        runs = list(_get(f"{api}/actions/workflows/{WORKFLOW}/runs?branch=master&status=completed&per_page=30", token).get("workflow_runs") or [])  # type: ignore[union-attr]
    message = os.environ.get("COMMIT_MESSAGE", "")
    if not message and sha and token:
        message = str((_get(f"{api}/commits/{sha}", token) or {}).get("commit", {}).get("message") or "")  # type: ignore[union-attr]
    prs: list[dict] = []
    if sha and token and not os.environ.get("DRY_LATEST"):
        try:
            prs = list(_get(f"{api}/commits/{sha}/pulls", token) or [])  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — a direct push has no pull request
            prs = []

    def events_of(n: int) -> list[dict]:
        return list(_get(f"{api}/issues/{n}/events?per_page=100", token) or [])  # type: ignore[arg-type]

    ok, why = decide(runs, message=message, prs=prs, events_of=events_of, founder=founder, actor=os.environ.get("GITHUB_ACTOR", ""),
                     override_input=os.environ.get("OVERRIDE_NIGHTLY", "").lower() == "true")
    print(("DEPLOY ALLOWED: " if ok else "DEPLOY BLOCKED: ") + why)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Nightly gate\n\n{'Allowed' if ok else '**Blocked**'}: {why}\n")
    if "--dry-run" in argv:
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
