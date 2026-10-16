"""Post a night's scores.json to the engine (/v2/admin/eval-scores), and email the team when it cannot be stored.

The engine may be between deploys (13 Oct 00:34: a 404 because no deploy carrying the route had landed, and the score was
lost without a word). So: retry while the engine answers 404, 5xx or not at all (up to POST_TRIES, every POST_WAIT_S);
stop at once on 401/403/422 (a setup problem); on a final failure email the team through Resend with the night's verdict
and the broken bars (the engine never got to email a failing night), and how to post it again
(gh workflow run nightly-saheli.yml -f repost_run=<run id>). Never fails the job: the night's gate decides that.

    ID_TOKEN=... ENGINE_URL=... RESEND_API_KEY=... python3 scripts/post_eval_scores.py scores.json --kind nightly \
        --run-id 123 --commit <sha> --branch master
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request


def post(url: str, token: str, body: bytes) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    secret = os.environ.get("ENGINE_ADMIN_SECRET", "")
    if secret:
        headers["x-admin-secret"] = secret
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (the engine's own URL)
            return r.status, r.read().decode()[:600]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:600]
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, str(e)[:300]


def email(subject: str, text: str) -> bool:
    key = os.environ.get("RESEND_API_KEY", "")
    if not key:
        print("::warning::no RESEND_API_KEY: the team could not be emailed")
        return False
    to = [x.strip() for x in os.environ.get("TEAM_REPORT_TO", "kritarth@kavach.care").split(",") if x.strip()]
    body = json.dumps({"from": "Kavach nightly <alerts@emails.kavach.care>", "to": to, "subject": subject[:160], "text": text[:20000]}).encode()
    req = urllib.request.Request("https://api.resend.com/emails", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            return 200 <= r.status < 300
    except urllib.error.HTTPError as exc:
        # 403 is this key not being allowed to send as `from` (the domain is not on the key, or the key can only
        # write to its owner). The body says which. The key itself is not printed.
        detail = exc.read().decode(errors="replace")[:300]
        print(f"::warning::the team email failed: HTTP {exc.code} from {body_from(body)}: {detail}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::the team email failed: {exc}")
        return False


def body_from(body: bytes) -> str:
    try:
        return str(json.loads(body).get("from") or "")
    except json.JSONDecodeError:
        return ""


def main(argv: list[str]) -> int:
    path = argv[0]

    def opt(name: str, default: str = "") -> str:
        return argv[argv.index(name) + 1] if name in argv else default

    kind, run_id, commit, branch = opt("--kind", "nightly"), opt("--run-id", os.environ.get("GITHUB_RUN_ID", "")), opt("--commit", os.environ.get("GITHUB_SHA", "")), \
        opt("--branch", os.environ.get("GITHUB_REF_NAME", ""))
    try:
        sc = json.load(open(path))
    except (OSError, json.JSONDecodeError) as exc:
        sc = {"passed": False, "failures": [f"no readable scores.json ({exc})"]}
    body = json.dumps({"run_id": run_id, "kind": kind, "day": opt("--day", datetime.date.today().isoformat()), "passed": bool(sc.get("passed")),
                       "commit": commit, "branch": branch, "scores": sc, "failures": sc.get("failures") or []}).encode()
    url = os.environ.get("ENGINE_URL", "https://kawach-ai-engine-bcvowb5kfq-el.a.run.app").rstrip("/") + "/v2/admin/eval-scores"
    token = os.environ.get("ID_TOKEN", "")
    tries, wait = int(os.environ.get("POST_TRIES", "12")), float(os.environ.get("POST_WAIT_S", "150"))
    status, text = 0, "no token"
    secret = os.environ.get("ENGINE_ADMIN_SECRET", "")
    if not token and not secret:
        if sc.get("passed") and not sc.get("unavailable"):
            print("the night passed; the engine is not reachable from GitHub, so the score stays in the artifact")
            return 0
        status, text = 0, "the engine is on a private network; GitHub cannot post the score"
    for attempt in range(1, tries + 1):
        if not token and not secret:
            break
        status, text = post(url, token, body)
        print(f"post {attempt}/{tries}: {status} {text[:200]}")
        if 200 <= status < 300:
            return 0
        if status in (400, 401, 403, 422):
            break  # a setup problem: waiting will not fix it
        if attempt < tries:
            time.sleep(wait)
    print(f"::warning::the night's score could not be stored on the engine ({status}): emailing the team")
    if sc.get("unavailable"):
        verdict = "UNAVAILABLE (not a quality failure; deploys go on if the tests passed)"
    elif sc.get("passed"):
        verdict = "PASSED"
    else:
        verdict = "FAILED"
    lines = [f"The nightly world score of run {run_id} ({kind}, {branch}, commit {commit[:12]}) could not be stored on the engine.",
             f"Last answer: {status} {text[:300]}", "",
             f"The night itself: {verdict}."]
    if sc.get("failures"):
        lines += ["Broken bars:", *[f"  - {f}" for f in sc["failures"][:30]]]
    j = (sc.get("judge") or {}).get("overall") or {}
    lines += ["", f"Judge pass {j.get('pct')}% of {j.get('n')}; probes run {(sc.get('probes') or {}).get('executed')} of {(sc.get('probes') or {}).get('selected')}.",
              "", f"Post it again once the engine answers: gh workflow run nightly-saheli.yml -f repost_run={run_id}",
              f"Run: https://github.com/kritarth1107/kawach-ai-engine/actions/runs/{run_id}"]
    email(f"Nightly score NOT STORED (run {run_id}): the night {verdict.split(' (')[0]}", "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
