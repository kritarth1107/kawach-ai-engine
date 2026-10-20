"""The nightly trainer (founder 20 Oct, the Harness² method in our own code): it fixes the replay cases Saheli still fails by
editing her instruction text, and opens a pull request for the founder. It never changes live Saheli.

One night, inside AWS (kavach-trainer task, 04:00 IST, Gemini in kavach-eval-1015, a throwaway database in the task):
1. cases: the open replay cases (the judge's failures and corrections from real chats, anonymised; the repo's invented
   families), grouped by kind. Health kinds and sensitive cases are left out.
2. baseline: each chosen kind's cases run on today's instructions (3 runs, 2 must pass) to see which still fail.
3. editors: two model editors read the failure and Saheli's instruction files. One proposes a change to how she works
   (procedure), the other to what she knows (knowledge). Each answers one exact text edit.
4. probe: each edit that passes the locks runs on the same cases; it is kept only when it fixes at least one and
   breaks none of them.
5. regression: the kept edits together run on every fixed case. A fixed case that passed before and fails now drops
   the edit that caused it.
6. pull request: one per kind, from the kavach-trainer GitHub App, with the edit, before and after, results and cost.
   The App cannot merge; the founder merges or closes. A closed PR's edit is not proposed again.

Locks: only ALLOWED files; no edit that removes or adds text about emergencies, alerts or medicine doses; at most
MAX_ADDED characters per edit; the fixed prompt stays under its budget; ₹ cap TRAINER_MAX_INR (default 300).

    REPLAY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/kawach_replay GCP_PROJECT_ID=kavach-eval-1015 \
    TRAINER_MAX_INR=300 PYTHONPATH=. python -m eval.trainer [--dry-run] [--kinds a,b] [--cases file.jsonl]
"""

from __future__ import annotations

import os

os.environ.setdefault("MEMORY_EMBEDDINGS", "off")
os.environ.setdefault("PERSON_SYNC", "off")
os.environ.setdefault("WEB_SEARCH_PROVIDER", "off")

import asyncio
import base64
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PERSONA_DIR = "app/brain/persona"
ALLOWED = ("core.md", "reply.md", "memory.md", "nutrition.md", "order.md", "photos.md", "setup.md")
# Text about these stays as the people who own it wrote it: the trainer neither removes nor adds it.
PROTECTED = re.compile(r"emergenc|crisis|suicid|self.?harm|ambulance|\b11[02]\b|\b108\b|alert|dose|overdose|insulin|dosage|mg\b|"
                       r"red.?flag|chest pain|stroke|faint|unconscious|bleed", re.I)
MAX_ADDED = 600
PREFIX_TOKENS = 10000  # tests/test_gemini_prefix.py
SKIP_KINDS = ("health", "safety", "crisis", "emergency", "medicine", "alert")  # kinds starting with these
LABEL = "saheli-trainer"
REPO = os.getenv("TRAINER_REPO", "kritarth1107/kawach-ai-engine")
AUTHOR = "Saheli Trainer"

EDITOR = """You improve the written instructions of Saheli, a WhatsApp care companion for elders and for people caring for
themselves. Below are her instruction FILES, then FAILURES: cases where her reply broke what MUST HOLD. You may propose
ONE edit to ONE file.

Your lane: {lane}

Rules:
- Fix the kind of mistake, not the one example: write the general rule a careful person would follow. Never name the
  case's people, food, places or words.
- Prefer sharpening or replacing an existing sentence over adding a new one. Added text must be short and plain.
- Never touch, add or remove text about emergencies, alerts, red flags, medicines or doses.
- "find" must be copied exactly from one file and appear in it once; "replace" is the new text for it (it may keep "find"
  and add after it).
- If no instruction change can fix this (it is a code bug or the case itself is wrong), answer {{"skip": "<why>"}}.
- Room: {room}
{avoid}
Answer only JSON: {{"file": "<one of {files}>", "find": "<exact text>", "replace": "<new text>", "why": "<one sentence>"}}"""

LANES = {
    "procedure": "HOW SHE WORKS: the steps she takes, what she checks first, which tool and when, what she does when unsure.",
    "knowledge": "WHAT SHE KNOWS: what words and situations mean, what people in this situation expect, rules of the domain.",
}


@dataclass
class Edit:
    file: str
    find: str
    replace: str
    why: str
    lane: str
    kind: str = ""
    fixed: list[str] = field(default_factory=list)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)


# ── the instruction texts ────────────────────────────────────────────────────


def originals() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1] / PERSONA_DIR
    return {f: (root / f).read_text() for f in ALLOWED}


def use_texts(texts: dict[str, str]) -> None:
    """Every reader of the instructions in this process sees `texts` (the persona is read once at import)."""
    from app.brain import loop
    from app.brain import persona

    persona.PERSONA_CORE = texts["core.md"]
    for name in ("order", "nutrition", "setup", "photos", "memory"):
        persona._BLOCKS[name] = texts[f"{name}.md"]
    persona.REPLY_FORMAT = texts["reply.md"]
    loop.REPLY_FORMAT = texts["reply.md"]


def apply(texts: dict[str, str], edits: list[Edit]) -> dict[str, str]:
    out = dict(texts)
    for e in edits:
        out[e.file] = out[e.file].replace(e.find, e.replace, 1)
    return out


def prefix_tokens(texts: dict[str, str]) -> int:
    """The fixed prefix, counted as tests/test_gemini_prefix.py counts it."""
    from app.brain import persona, tools

    use_texts(texts)
    text = persona.for_groups(tools.groups_for()) + "\n\n" + texts["reply.md"]
    tool_tok = len(json.dumps([{"n": t.name, "d": t.description, "s": t.schema} for t in tools.specs_for()])) // 4
    return len(text) // 4 + tool_tok


def lock(e: Edit, texts: dict[str, str], rejected: list[str]) -> str | None:
    """Why this edit may not be tried, or None."""
    if e.file not in ALLOWED:
        return f"{e.file} is not an instruction file the trainer may edit"
    if not e.find.strip() or texts[e.file].count(e.find) != 1:
        return "the text to replace is not in the file exactly once"
    if e.find == e.replace:
        return "no change"
    if PROTECTED.search(e.find) or PROTECTED.search(e.replace):
        return "touches emergency, alert or medicine text"
    if len(e.replace) - len(e.find) > MAX_ADDED:
        return f"adds more than {MAX_ADDED} characters"
    norm = re.sub(r"\W+", " ", e.replace.lower()).strip()
    if any(norm and r and (norm in r or r in norm) for r in rejected):
        return "the founder closed this edit before"
    after = apply(texts, [e])
    try:
        n = prefix_tokens(after)
    finally:
        use_texts(texts)
    if n >= PREFIX_TOKENS:
        return f"the fixed prompt would be {n} tokens (limit {PREFIX_TOKENS})"
    return None


# ── cases ────────────────────────────────────────────────────────────────────


async def load(case_files: list[str]) -> list[dict]:
    """The repo's open cases plus production's anonymised open and fixed ones (read only)."""
    from eval import replay_cases as runner
    from eval import sensitive

    cases = runner.load_cases(case_files)
    if os.getenv("TRAINER_PROD_CASES", "off") == "on":  # the AWS task; a laptop uses the repo's cases
        from sqlalchemy import select

        from app.db.session import SessionLocal
        from app.quality.cases import as_row
        from app.quality.models import ReplayCase

        async with SessionLocal() as s:
            rows = list((await s.execute(select(ReplayCase).where(ReplayCase.status.in_(("open", "fixed")))
                                         .order_by(ReplayCase.created_at.desc()).limit(400))).scalars())
        seen = {c["id"] for c in cases}
        cases += [c for c in (as_row(r) for r in rows if r.anonymised or not r.family_id) if c["id"] not in seen]
    return [c for c in cases if not sensitive.held_back(c)]


def pick(cases: list[dict], *, groups: int, per: int, kinds: set[str] | None, busy: set[str]) -> dict[str, list[dict]]:
    """Open cases by kind, the kinds with the most cases first; health and kinds with an open trainer PR left out."""
    by: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        k = str(c.get("kind") or "other")
        if c.get("status") == "open" and not k.startswith(SKIP_KINDS) and k not in busy and (not kinds or k in kinds):
            by[k].append(c)
    order = sorted(by, key=lambda k: (-len(by[k]), k))[:groups]
    return {k: sorted(by[k], key=lambda c: str(c.get("created_at") or ""), reverse=True)[:per] for k in order}


def brief(case: dict, result: dict | None) -> str:
    ctx = case.get("context") or {}
    turns = [f"  {t.get('role')}: {str(t.get('text') or '')[:200]}" for t in (ctx.get("turns") or [])[-6:]]
    last = ((result or {}).get("runs") or [{}])[-1]
    return "\n".join([
        f"CASE {case['id']} (kind {case.get('kind')})",
        f"Speaker: {(ctx.get('speaker') or {}).get('role') or 'unknown'}; time {ctx.get('time') or ''}",
        "Conversation before:", *turns,
        f"Their message: {str(case.get('message') or ctx.get('message') or '')[:400]}",
        f"Her bad reply (at the time): {str(case.get('bad_reply') or '')[:400]}",
        f"MUST HOLD: {json.dumps(case.get('assertion') or [], ensure_ascii=False)[:500]} {case.get('rubric') or ''}",
        f"Today's reply: {str(last.get('reply') or '')[:400]}",
        f"Why it still fails: {str(last.get('why') or '')[:300]}"])


def room(texts: dict[str, str]) -> str:
    """How much the always-sent instructions may grow (core.md and reply.md are on every call; the others only with their tools)."""
    left = PREFIX_TOKENS - prefix_tokens(texts)
    use_texts(texts)
    always = f"core.md and reply.md are sent on every call and have about {max(left, 0) * 4} characters of room left"
    if left < 60:
        always += ", so an edit there must not make the file longer (replace wording; put a new rule in the topic file it belongs to)"
    return always + f"; every edit adds at most {MAX_ADDED} characters."


async def propose(lane: str, kind: str, failing: list[tuple[dict, dict]], texts: dict[str, str], rejected: list[str]) -> Edit | str:
    """One edit from this lane's editor; when it is unusable or locked, the editor hears why and gets one more try."""
    from app.llm import router
    from eval.family_sim import parse_json

    avoid = ("The founder already rejected these changes; propose something different:\n" + "\n".join(f"- {r[:200]}" for r in rejected[:10])) if rejected else ""
    system = EDITOR.format(lane=LANES[lane], avoid=avoid, files=", ".join(ALLOWED), room=room(texts))
    files = "\n\n".join(f"=== FILE {f} ===\n{t}" for f, t in texts.items())
    fails = "\n\n".join(brief(c, r) for c, r in failing)
    messages = [{"role": "user", "content": [{"type": "text", "text": f"FILES:\n{files}\n\nFAILURES:\n{fails}"}]}]
    why = ""
    for _ in range(2):
        try:
            rep = await router.complete("learn", system_stable=system, messages=messages, max_tokens=4000, effort="medium", timeout_s=120)
            g = parse_json(rep.text) or {}
        except Exception as exc:  # noqa: BLE001
            return f"editor error: {str(exc)[:120]}"
        if g.get("skip"):
            return f"editor: {str(g['skip'])[:200]}"
        if not all(isinstance(g.get(k), str) for k in ("file", "find", "replace")):
            why = f"answer not usable: {(rep.text or '')[:160]!r}"
        else:
            e = Edit(file=g["file"].strip(), find=g["find"], replace=g["replace"], why=str(g.get("why") or "")[:300], lane=lane, kind=kind)
            why = lock(e, texts, rejected) or ""
            if not why:
                return e
        messages += [{"role": "assistant", "content": [{"type": "text", "text": rep.text or ""}]},
                     {"role": "user", "content": [{"type": "text", "text": f"That edit cannot be used: {why}. Answer one corrected edit as JSON only."}]}]
    return f"editor: {why}"


# ── GitHub (the kavach-trainer App) ──────────────────────────────────────────


class GitHub:
    API = "https://api.github.com"

    def __init__(self) -> None:
        self.token = ""

    @staticmethod
    def _b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    def app_jwt(self) -> str:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        app_id = os.environ["TRAINER_APP_ID"]
        pem = os.environ["TRAINER_APP_KEY"].encode()
        key = serialization.load_pem_private_key(pem, password=None)
        now = int(time.time())
        head = self._b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
        body = self._b64(json.dumps({"iat": now - 60, "exp": now + 540, "iss": app_id}).encode())
        sig = key.sign(f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())  # type: ignore[union-attr,call-arg]
        return f"{head}.{body}.{self._b64(sig)}"

    async def call(self, method: str, path: str, body: dict | None = None, *, bearer: str | None = None) -> dict | list:
        import httpx

        headers = {"Authorization": f"Bearer {bearer or self.token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.request(method, self.API + path, headers=headers, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"GitHub {method} {path}: {r.status_code} {r.text[:200]}")
        return r.json() if r.content else {}

    async def login(self) -> None:
        jwt = self.app_jwt()
        inst = await self.call("GET", f"/repos/{REPO}/installation", bearer=jwt)
        tok = await self.call("POST", f"/app/installations/{inst['id']}/access_tokens", bearer=jwt)  # type: ignore[index]
        self.token = tok["token"]  # type: ignore[index]

    async def trainer_prs(self) -> tuple[set[str], list[str]]:
        """(kinds with an open trainer PR, the replacement text of every PR the founder closed without merging)."""
        busy: set[str] = set()
        rejected: list[str] = []
        prs = await self.call("GET", f"/repos/{REPO}/pulls?state=all&per_page=100")
        for p in prs if isinstance(prs, list) else []:
            if LABEL not in {lb["name"] for lb in p.get("labels") or []}:
                continue
            body = p.get("body") or ""
            m = re.search(r"<!-- trainer (\{.*?\}) -->", body, re.S)
            meta = json.loads(m.group(1)) if m else {}
            if p["state"] == "open":
                busy.add(str(meta.get("kind") or ""))
            elif not p.get("merged_at") and meta.get("replace"):
                rejected.append(re.sub(r"\W+", " ", str(meta["replace"]).lower()).strip())
        return busy, rejected

    async def open_pr(self, e: Edit, *, day: str, results: dict, spent: float) -> str:
        """One commit on a new branch from master, then the pull request. The text is edited on master's copy."""
        ref = await self.call("GET", f"/repos/{REPO}/git/ref/heads/master")
        base = ref["object"]["sha"]  # type: ignore[index]
        path = f"{PERSONA_DIR}/{e.file}"
        cur = await self.call("GET", f"/repos/{REPO}/contents/{path}?ref={base}")
        text = base64.b64decode(cur["content"]).decode()  # type: ignore[index]
        if text.count(e.find) != 1:
            raise RuntimeError(f"{path} changed on master; the edit no longer applies")
        new = text.replace(e.find, e.replace, 1)
        commit = await self.call("GET", f"/repos/{REPO}/git/commits/{base}")
        tree = await self.call("POST", f"/repos/{REPO}/git/trees", {"base_tree": commit["tree"]["sha"],  # type: ignore[index]
                                                                     "tree": [{"path": path, "mode": "100644", "type": "blob", "content": new}]})
        title = f"Saheli: {e.why[:70].rstrip('.')}" if e.why else f"Saheli: fix {e.kind}"
        msg = (f"{title}\n\nFixes {len(e.fixed)} open replay case(s) of kind {e.kind} ({', '.join(e.fixed)}).\n"
               f"Regression: {results.get('fixed_pass', 0)}/{results.get('fixed_total', 0)} fixed cases pass.\n\n"
               f"Author-agent: {AUTHOR}\n")
        made = await self.call("POST", f"/repos/{REPO}/git/commits", {"message": msg, "tree": tree["sha"], "parents": [base]})  # type: ignore[index]
        branch = f"trainer/{day}-{re.sub(r'[^a-z0-9]+', '-', e.kind.lower()).strip('-')[:40]}"
        await self.call("POST", f"/repos/{REPO}/git/refs", {"ref": f"refs/heads/{branch}", "sha": made["sha"]})  # type: ignore[index]
        meta = json.dumps({"kind": e.kind, "file": e.file, "replace": e.replace[:400]}, ensure_ascii=False)
        body = "\n".join([
            f"**What:** {e.why}", "",
            f"**File:** `{path}` ({e.lane} edit)", "",
            "**Before → after on the failing cases:**",
            *(f"- `{cid}`: before: {str(e.before.get(cid) or '')[:240]!r}\n  after: {str(e.after.get(cid) or '')[:240]!r}" for cid in e.fixed), "",
            f"**Regression:** {results.get('fixed_pass', 0)}/{results.get('fixed_total', 0)} fixed cases pass; "
            f"{results.get('known', 0)} already failed before this change (model baseline).",
            f"**Cost of tonight's run:** ₹{spent:.0f}", "",
            "Merge to put it live (the normal deploy runs the tests first). Close it to reject; the trainer will not propose it again.",
            f"<!-- trainer {meta} -->", "", "🤖 Opened by the kavach-trainer GitHub App (eval/trainer.py)"])
        pr = await self.call("POST", f"/repos/{REPO}/pulls", {"title": title, "head": branch, "base": "master", "body": body})
        await self.call("POST", f"/repos/{REPO}/issues/{pr['number']}/labels", {"labels": [LABEL]})  # type: ignore[index]
        return str(pr["html_url"])  # type: ignore[index]


# ── the night ────────────────────────────────────────────────────────────────


def _reply(r: dict | None) -> str:
    return str((((r or {}).get("runs") or [{}])[-1]).get("reply") or "")


async def night(argv: list[str]) -> dict:
    from app.learn.replay import EVAL_PROJECT, REFUSED_PROJECTS
    from app.llm import spend
    from eval import replay_cases as runner
    from eval.live_routes import gemini_routes

    project = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or ""
    if project in REFUSED_PROJECTS or project != EVAL_PROJECT:
        raise SystemExit(f"the trainer runs on Gemini in {EVAL_PROJECT} only (got {project or 'none'})")
    os.environ["MODEL_ROUTES"] = json.dumps(gemini_routes())
    url = os.getenv("REPLAY_DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("set REPLAY_DATABASE_URL to the throwaway database")
    dry = "--dry-run" in argv
    cap = float(os.getenv("TRAINER_MAX_INR", "300"))
    kinds = set(argv[argv.index("--kinds") + 1].split(",")) if "--kinds" in argv else None
    files = [argv[argv.index("--cases") + 1]] if "--cases" in argv else []
    day = time.strftime("%Y-%m-%d")
    gh = GitHub()
    busy: set[str] = set()
    rejected: list[str] = []
    if not dry and not (os.getenv("TRAINER_APP_ID") and os.getenv("TRAINER_APP_KEY")):
        raise SystemExit("TRAINER_APP_ID or TRAINER_APP_KEY is not set: nothing ran, nothing spent")
    if not dry:
        await gh.login()
        busy, rejected = await gh.trainer_prs()
    report: dict = {"day": day, "dry_run": dry, "kinds": {}, "prs": [], "notes": []}
    if await runner.preflight():
        report["notes"].append("the model did not answer; nothing ran")
        return report
    cases = await load(files)
    groups = pick(cases, groups=int(os.getenv("TRAINER_GROUPS", "3")), per=int(os.getenv("TRAINER_PER_KIND", "3")), kinds=kinds, busy=busy)
    texts = originals()
    use_texts(texts)

    def left() -> float:
        return cap - spend.process_spent()

    kept: list[Edit] = []
    for kind, group in groups.items():
        if left() < cap * 0.35:
            report["notes"].append(f"budget: stopped before {kind}")
            break
        base = {r["id"]: r for r in await runner.run_cases(group, url, min(left(), cap * 0.1), parallel=3, deadline_s=900)}
        failing = [(c, base[c["id"]]) for c in group if base.get(c["id"], {}).get("passed") is False]
        info: dict = {"cases": [c["id"] for c in group], "failing": [c["id"] for c, _ in failing], "tried": []}
        report["kinds"][kind] = info
        if not failing:
            continue
        best: Edit | None = None
        for lane in LANES:
            e = await propose(lane, kind, failing, texts, rejected)
            if isinstance(e, str):
                info["tried"].append({"lane": lane, "result": e})
                continue
            use_texts(apply(texts, [e]))
            try:
                probe = {r["id"]: r for r in await runner.run_cases(group, url, min(left(), cap * 0.1), parallel=3, deadline_s=900)}
            finally:
                use_texts(texts)
            e.fixed = [c["id"] for c, _ in failing if probe.get(c["id"], {}).get("passed") is True]
            broke = [cid for cid, r in base.items() if r.get("passed") is True and probe.get(cid, {}).get("passed") is False]
            e.before = {c["id"]: _reply(base.get(c["id"])) for c, _ in failing}
            e.after = {cid: _reply(probe.get(cid)) for cid in e.fixed}
            info["tried"].append({"lane": lane, "file": e.file, "why": e.why, "fixed": e.fixed, "broke": broke})
            if e.fixed and not broke and (best is None or (len(e.fixed), -len(e.replace)) > (len(best.fixed), -len(best.replace))):
                best = e
        if best:
            kept.append(best)

    results: dict = {}
    if kept:
        fixed_cases = runner.choose([c for c in cases if c.get("status") == "fixed"], int(os.getenv("TRAINER_REGRESSION_MAX", "40")))
        known = runner.model_baseline()
        use_texts(apply(texts, kept))
        try:
            after = {r["id"]: r for r in await runner.run_cases(fixed_cases, url, max(left(), 0), parallel=3, deadline_s=1500)}
        finally:
            use_texts(texts)
        bad = [cid for cid, r in after.items() if r.get("passed") is False and cid not in known]
        if bad:  # which edit broke them: each kept edit alone on those cases
            again = [c for c in fixed_cases if c["id"] in bad]
            base_bad = {r["id"]: r for r in await runner.run_cases(again, url, max(left(), 0), parallel=3, deadline_s=600)}
            real = [cid for cid in bad if base_bad.get(cid, {}).get("passed") is True]
            for e in list(kept):
                if not real:
                    break
                use_texts(apply(texts, [e]))
                try:
                    solo = {r["id"]: r for r in await runner.run_cases([c for c in again if c["id"] in real], url, max(left(), 0), parallel=3, deadline_s=600)}
                finally:
                    use_texts(texts)
                if any(r.get("passed") is False for r in solo.values()):
                    kept.remove(e)
                    report["notes"].append(f"dropped the {e.kind} edit: it broke {', '.join(k for k, r in solo.items() if r.get('passed') is False)}")
        ran = [r for r in after.values() if r.get("passed") is not None]
        results = {"fixed_total": len(ran), "fixed_pass": sum(1 for r in ran if r["passed"]), "known": sum(1 for r in ran if not r["passed"] and r["id"] in known)}
        report["regression"] = results
    report["spent_inr"] = round(spend.process_spent(), 2)
    report["kept"] = [{"kind": e.kind, "file": e.file, "lane": e.lane, "why": e.why, "fixed": e.fixed, "find": e.find, "replace": e.replace} for e in kept]
    if not dry:
        for e in kept:
            try:
                report["prs"].append(await gh.open_pr(e, day=day, results=results, spent=report["spent_inr"]))
            except Exception as exc:  # noqa: BLE001 — the next kind still gets its PR
                report["notes"].append(f"PR for {e.kind} failed: {str(exc)[:200]}")
    return report


async def main_async(argv: list[str]) -> int:
    from app.jobs.learn_gate import wait_replay_db

    await wait_replay_db()
    report = await night(argv)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str), flush=True)
    out = Path(os.getenv("TRAINER_OUT", "runs/trainer"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report['day']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(sys.argv[1:])))
