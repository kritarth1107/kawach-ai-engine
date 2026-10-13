"""Replay cases: every source, the nightly export, and the fix loop (trust plan C3).

Sources, every night (all into replay_cases, app.quality.replay):
- the full-memory judge's failures and the memory score's failures (app.quality.nightly / memory_score)
- guard rewrites (including the no-invention check), people's corrections right after a reply, a caregiver's dashboard
  edit of something Saheli had just saved, and thumbs-down: the correction cases app.learn.corrections collects, each
  with the snapshot of the moment it happened (here)
- the nightly probes' failures, from the world score the nightly posts (invented families: rebuilt from the world)
- founder-reported bugs: invented families in the repo (eval/replay_cases.py builds them from eval/founder_points_eval.py)
Export: every open and fixed case, as JSONL, to a private bucket in the test project (REPLAY_EXPORT_BUCKET), where the
pre-deploy runner reads them. Fix loop: a run on master posts its results; an open case that passed becomes fixed (it then
guards every deploy); a fixed case that failed stays fixed and blocks the deploy (a regression).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care.models import CareEvent, FamilyRoster
from app.core import clock
from app.quality import replay, snapshot
from app.quality.models import ReplayCase

logger = logging.getLogger(__name__)

EXPORT_PREFIX = "cases"


def bucket() -> str:
    return os.getenv("REPLAY_EXPORT_BUCKET", "").strip()


def rules_from_problems(problems: list[str], avoid: list[str]) -> list[dict]:
    """The guard's own findings as assertions (the guard checks Saheli's draft, so these are about her output)."""
    rules: list[dict] = []
    text = " ".join(problems)
    if avoid and ("never" in text.lower() or "avoid" in text.lower() or "name" in text.lower()):
        rules.append({"type": "not_contains", "words": avoid})
    if any(k in text for k in ("not in memory", "nobody reported", "not in their record", "made up", "invent", "ground")):
        rules.append({"type": "grounded"})
    if "advice" in text:
        rules.append({"type": "no_medical_advice"})
    if any(k in text for k in ("claims", "did not", "promise", "message was sent", "ordered", "booked")):
        rules.append({"type": "no_false_claim"})
    return rules or [{"type": "grounded"}]


async def from_corrections(sessions: async_sessionmaker, *, since: datetime | None = None) -> dict:
    """Correction cases of the last day (app.learn.corrections) as replay cases with their snapshot."""
    from app.learn.corrections import LEARN
    from app.learn.models import ReplyLog

    since = since or clock.now() - timedelta(hours=26)
    out = {"chat_correction": 0, "guard_rewrite": 0, "memory_correction": 0, "thumbs_down": 0, "skipped": 0}
    async with sessions() as s:
        evs = list((await s.execute(select(CareEvent).where(CareEvent.family_id == LEARN, CareEvent.kind == "correction_case",
                                                            CareEvent.at >= since).order_by(CareEvent.at))).scalars())
        for e in evs:
            ref = e.ref or ""
            p = e.payload or {}
            kind = str(p.get("kind") or "correction")
            try:
                src_id = int(ref.rsplit(":", 1)[-1])
            except ValueError:
                out["skipped"] += 1
                continue
            made = False
            if ref.startswith("cc:reply:"):
                r = await s.get(ReplyLog, src_id)
                roster = await s.get(FamilyRoster, r.family_id) if r else None
                if r is not None and roster is not None:
                    snap = await snapshot.take(s, roster, r.thread_id, r.at, message=r.user_text, speaker_role=r.speaker_role, trace=r.trace, kind=r.kind)
                    made = await replay.add_snapshot(s, source="correction", source_ref=f"correction:reply:{r.id}", kind=kind, day=clock.ist(r.at).strftime("%Y-%m-%d"),
                                                     snap=snap, message=r.user_text, bad_reply=r.text, assertion=[{"type": "grounded"}],
                                                     rubric=f"The person corrected this reply ({str(p.get('problem') or '')[:300]}); the new reply must not repeat the mistake.",
                                                     family_id=r.family_id, subject_id=r.thread_id, thread_id=r.thread_id)
            elif ref.startswith("cc:guard:"):
                g = await s.get(CareEvent, src_id)
                fid = (g.family_id.removeprefix(f"{LEARN}:") if g else "")
                roster = await s.get(FamilyRoster, fid) if fid else None
                if g is not None and roster is not None:
                    gp = g.payload or {}
                    snap = await snapshot.take(s, roster, g.subject_id, g.at, message=str(gp.get("they") or ""), kind="reply")
                    naming = next((f for f in snap["facts"] if f["key"] == "naming:address_as" and f["status"] == "active"), None)
                    avoid = ((naming or {}).get("value") or {}).get("avoid") or []
                    problems = [str(x) for x in gp.get("problems") or []]
                    made = await replay.add_snapshot(s, source="guard", source_ref=f"guard:{g.id}", kind="guard_rewrite", day=clock.ist(g.at).strftime("%Y-%m-%d"),
                                                     snap=snap, message=str(gp.get("they") or ""), bad_reply=str(gp.get("draft") or ""),
                                                     assertion=rules_from_problems(problems, avoid),
                                                     rubric="The reply guard stopped this draft: " + "; ".join(problems)[:600],
                                                     family_id=fid, subject_id=g.subject_id, thread_id=g.subject_id)
            elif ref.startswith(("cc:mem:", "cc:fb:")):
                ev = await s.get(CareEvent, src_id)
                roster = await s.get(FamilyRoster, ev.family_id) if ev else None
                if ev is not None and roster is not None:
                    snap = await snapshot.take(s, roster, ev.subject_id, ev.at, message="", kind="memory" if ref.startswith("cc:mem:") else "feedback")
                    made = await replay.add_snapshot(s, source="correction" if ref.startswith("cc:mem:") else "thumbs_down", source_ref=ref.replace("cc:", "correction:"),
                                                     kind=kind, day=clock.ist(ev.at).strftime("%Y-%m-%d"), snap=snap, message=str(p.get("they") or ""),
                                                     bad_reply=str(p.get("saheli") or ""), assertion=[{"type": "grounded"}],
                                                     rubric=f"{kind}: {str(p.get('problem') or '')[:400]}", family_id=ev.family_id, subject_id=ev.subject_id)
            if made:
                out[kind] = out.get(kind, 0) + 1
        await s.commit()
    return out


def probe_rules(fails: list[str], issues: list[str] | None = None) -> list[dict]:
    rules: list[dict] = []
    for f in fails or []:
        if f.startswith("used the word '"):
            rules.append({"type": "not_contains", "words": [f.split("'")[1]]})
        elif f.startswith("expected one of"):
            try:
                tools = json.loads(f.split("expected one of ", 1)[1].split("]")[0].replace("'", '"') + "]")
            except (ValueError, IndexError):
                tools = []
            if tools:
                rules.append({"type": "must_tool", "tools": tools})
        elif f.startswith("should not have used"):
            try:
                rules.append({"type": "no_tool", "tools": json.loads(f.split("used ", 1)[1].replace("'", '"'))})
            except (ValueError, IndexError):
                pass
        elif f.startswith("started an order"):
            rules.append({"type": "no_tool", "tools": ["start_task"]})
        elif f.startswith("too long"):
            rules.append({"type": "max_sentences", "n": 4})
    if issues:
        rules.append({"type": "grounded"})
    return rules or [{"type": "grounded"}]


async def from_probes(session: AsyncSession, scores: dict, day: str) -> int:
    """The night's probe failures (invented families: the runner rebuilds the family from the world)."""
    n = 0
    judged = {j["probe"]: j for j in scores.get("judge_failures") or []}
    for ex in scores.get("examples") or []:
        j = judged.pop(ex["probe"], None)
        n += await replay.add(session, source="probe", source_ref=f"probe:{ex['probe']}", kind=(ex.get("topic") or "probe")[:40], day=day,
                              context={"v": 1, "world": {"family": ex.get("family"), "probe": ex["probe"]}, "language": ex.get("language")},
                              message=ex.get("text") or "", bad_reply=ex.get("reply") or "", assertion=probe_rules(ex.get("fails") or [], (j or {}).get("issues")),
                              rubric="; ".join([*(ex.get("fails") or []), *([j["note"]] if j and j.get("note") else [])])[:1000])
    for pid, j in judged.items():
        n += await replay.add(session, source="probe", source_ref=f"probe:{pid}", kind="probe_judge", day=day,
                              context={"v": 1, "world": {"family": j.get("family"), "probe": pid}}, assertion=[{"type": "grounded"}],
                              rubric=f"The nightly judge failed this reply: {j.get('issues')} {j.get('note') or ''}"[:1000])
    return n


def as_row(c: ReplayCase) -> dict:
    return {"id": str(c.id), "source": c.source, "kind": c.kind, "status": c.status, "day": c.day, "context": c.context, "message": c.message,
            "bad_reply": c.bad_reply, "assertion": c.assertion, "rubric": c.rubric, "created_at": c.created_at.isoformat(), "anonymised": c.anonymised}


async def export(sessions: async_sessionmaker) -> dict:
    """Every open and fixed case to the private test bucket (gs://<REPLAY_EXPORT_BUCKET>/cases/latest.jsonl)."""
    name = bucket()
    async with sessions() as s:
        rows = list((await s.execute(select(ReplayCase).where(ReplayCase.status.in_(("open", "fixed"))).order_by(ReplayCase.created_at))).scalars())
    real = [r for r in rows if r.family_id]
    if any(not r.anonymised for r in real):
        logger.error("replay export: %d real-family cases are not anonymised; they stay out of the export", sum(1 for r in real if not r.anonymised))
    body = "".join(json.dumps(as_row(r), ensure_ascii=False, default=str) + "\n" for r in rows if r.anonymised or not r.family_id)
    if not name:
        return {"exported": False, "why": "REPLAY_EXPORT_BUCKET is not set", "cases": len(rows)}
    try:
        from google.cloud import storage

        b = storage.Client().bucket(name)
        for obj in (f"{EXPORT_PREFIX}/latest.jsonl", f"{EXPORT_PREFIX}/{clock.ist_day()}.jsonl"):
            b.blob(obj).upload_from_string(body, content_type="application/x-ndjson")
    except Exception as exc:  # noqa: BLE001 — the runner then uses yesterday's export
        logger.exception("replay export failed")
        return {"exported": False, "why": str(exc)[:200], "cases": len(rows)}
    return {"exported": True, "bucket": name, "cases": body.count("\n"), "fixed": sum(1 for r in rows if r.status == "fixed")}


async def apply_results(session: AsyncSession, results: list[dict], *, commit: str, branch: str) -> dict:
    """A runner's results: kept on each case; on master an open case that passed becomes fixed (the fix loop)."""
    import uuid as _uuid

    flipped = updated = 0
    for r in results:
        try:
            cid = _uuid.UUID(str(r.get("id")))
        except ValueError:
            continue
        c = await session.get(ReplayCase, cid)
        if c is None:
            continue
        c.last_run_at = clock.now()
        c.last_result = {"passed": bool(r.get("passed")), "runs": (r.get("runs") or [])[:3], "commit": commit[:40], "branch": branch[:80]}
        updated += 1
        if branch in ("master", "main") and r.get("passed") and c.status == "open":
            c.status, c.fixed_at, c.fixed_commit = "fixed", clock.now(), commit[:64]
            flipped += 1
    await session.commit()
    return {"updated": updated, "fixed_now": flipped}


async def counts(session: AsyncSession) -> dict:
    from sqlalchemy import func

    rows = (await session.execute(select(ReplayCase.status, ReplayCase.source, func.count()).group_by(ReplayCase.status, ReplayCase.source))).all()
    out: dict = {"open": 0, "fixed": 0, "dropped": 0, "by_source": {}}
    for status, source, n in rows:
        out[status] = out.get(status, 0) + n
        out["by_source"].setdefault(source, {})[status] = n
    return out
