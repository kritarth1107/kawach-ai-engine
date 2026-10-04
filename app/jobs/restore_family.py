"""Bring one family's memory back from a nightly snapshot (decrypted with `python backup.py family FAMILY_ID DATE --out snap.json`).

    python -m app.jobs.restore_family snap.json            # what would change (nothing is written)
    python -m app.jobs.restore_family snap.json --apply    # restore it

Nothing is overwritten without a trace:
- memory notes and family skills come back as new versions, so each can be undone from WhatsApp or the dashboard;
- care-record facts that differ come back only as pending changes a caregiver confirms, so a medicine, an allergy or a
  dose never changes silently (and reminders change only once confirmed);
- notes and facts that exist now but not in the snapshot are only reported, never removed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import skillbook, store, versions
from app.care.models import CareFact, MemoryNote
from app.core import clock


def _clean(v: dict | None) -> dict:
    return {k: x for k, x in (v or {}).items() if x not in (None, "", []) and k != "stopped"}


async def plan(session: AsyncSession, snap: dict) -> dict:
    """What a restore would change: {notes: [...], facts: [...], skills: [...], only_now: {...}}."""
    fid = snap["family_id"]
    out: dict = {"family": fid, "taken_at": snap.get("taken_at"), "notes": [], "facts": [], "skills": [], "only_now": {"notes": [], "facts": []}}
    now_notes = {(n.subject_id, n.slug): n for n in (await session.execute(select(MemoryNote).where(MemoryNote.family_id == fid))).scalars()}
    snap_notes = {(n["subject_id"], n["slug"]): n for n in snap.get("notes") or []}
    for k, n in snap_notes.items():
        cur = now_notes.get(k)
        if not cur or cur.body_md != n["body_md"]:
            out["notes"].append({"subject": k[0], "slug": k[1], "title": n["title"], "change": "missing now" if not cur else "differs"})
    out["only_now"]["notes"] = [f"{s}/{slug}" for (s, slug) in now_notes if (s, slug) not in snap_notes]

    now_facts = {(f.subject_id, f.key): f for f in (await session.execute(select(CareFact).where(
        CareFact.family_id == fid, CareFact.status == "active"))).scalars()}
    snap_facts = {(f["subject_id"], f["key"]): f for f in snap.get("facts") or [] if f.get("status") == "active"}
    for k, f in snap_facts.items():
        cur = now_facts.get(k)
        if not cur or _clean(cur.value) != _clean(f.get("value")):
            out["facts"].append({"subject": k[0], "key": k[1], "domain": f["domain"], "text": f.get("text") or "",
                                 "now": cur.text if cur else None, "change": "missing now" if not cur else "differs"})
    out["only_now"]["facts"] = [f"{s}/{key}" for (s, key) in now_facts if (s, key) not in snap_facts]

    for sk in snap.get("skills") or []:
        cur = await session.get(skillbook.Skill, sk["id"])
        if not cur or cur.family_id != fid or cur.scope != "family":
            out["skills"].append({"id": sk["id"], "body": sk["body"], "change": "missing now"})
        elif cur.body != sk["body"] or cur.status != sk["status"]:
            out["skills"].append({"id": sk["id"], "body": sk["body"], "status": sk["status"], "change": "differs"})
    return out


async def apply(session: AsyncSession, snap: dict, *, by: str = "restore") -> dict:
    p = await plan(session, snap)
    fid = snap["family_id"]
    taken = str(snap.get("taken_at") or "")[:10]
    reason = f"restored from the {taken} snapshot"
    done = {"notes": 0, "facts_pending": 0, "skills": 0, "skipped": []}
    with versions.attribution(actor_id=by, source="snapshot", reason=reason):
        snap_notes = {(n["subject_id"], n["slug"]): n for n in snap.get("notes") or []}
        for item in p["notes"]:
            n = snap_notes[(item["subject"], item["slug"])]
            await store.upsert_note(session, family_id=fid, subject_id=n["subject_id"], slug=n["slug"], title=n["title"], body_md=n["body_md"],
                                    op="snapshot")
            done["notes"] += 1
        snap_facts = {(f["subject_id"], f["key"]): f for f in snap.get("facts") or [] if f.get("status") == "active"}
        for item in p["facts"]:
            f = snap_facts[(item["subject"], item["key"])]
            w = await store.write_fact(session, family_id=fid, subject_id=f["subject_id"], domain=f["domain"], key=f["key"],
                                       value=_clean(f.get("value")), text=f.get("text") or f["key"], source_kind="import",
                                       source_ref=f"undo:snapshot:{taken}", stated_by=by, replace=True, force_confirm=True)
            done["facts_pending"] += 1 if w.result == "pending" else 0
        for item in p["skills"]:
            sk = next(s for s in snap.get("skills") or [] if s["id"] == item["id"])
            if sk["status"] in ("active", "proposed") and skillbook.problems(sk["body"]):
                done["skipped"].append(f"skill {sk['id']}: wording no longer allowed")
                continue
            cur = await session.get(skillbook.Skill, sk["id"])
            if cur and cur.family_id == fid and cur.scope == "family":
                prior = versions.skill_state(cur)
                cur.body, cur.status, cur.title = sk["body"], sk["status"], sk["title"]
                cur.updated_at, cur.updated_by, cur.version = clock.now(), by, cur.version + 1
                await session.flush()
                await versions.record_skill(session, cur, "snapshot", prior=prior)
            else:
                now = clock.now()
                s = skillbook.Skill(scope="family", family_id=fid, subject_id=sk.get("subject_id"), service="", title=sk["title"], body=sk["body"],
                                    steps=[], source=sk.get("source") or "caregiver", evidence=[], status=sk["status"], created_at=now,
                                    updated_at=now, updated_by=by)
                session.add(s)
                await session.flush()
                await versions.record_skill(session, s, "snapshot")
            done["skills"] += 1
    return {"plan": p, "done": done}


async def _main(path: str, do_apply: bool, by: str) -> dict:
    from app.db.session import SessionLocal
    from app.jobs.dream import _ensure_v2_schema

    await _ensure_v2_schema()
    snap = json.loads(open(path, encoding="utf-8").read())
    async with SessionLocal() as session:
        if not do_apply:
            return await plan(session, snap)
        out = await apply(session, snap, by=by)
        await session.commit()
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--by", default="restore", help="who is restoring (shown in memory history)")
    a = ap.parse_args(argv)
    print(json.dumps(asyncio.run(_main(a.snapshot, a.apply, a.by)), indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
