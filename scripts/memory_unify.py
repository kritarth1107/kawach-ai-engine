"""One memory is the truth (trust plan A1): the inventory of the old memory stores, and the move job's report.

Counts only, never what was said: family ids are shortened to 8 characters, no text, names or numbers leave the stores.

    # 1. Inventory, read only, straight from production (Mongo and the engine's Postgres through the Cloud SQL proxy):
    python scripts/memory_unify.py inventory --mongo-uri-file /tmp/kv_mongo_uri --pg-env /tmp/kv_engine_db.env --out report.md

    # 2. The move job on the deployed engine (/v2/jobs/memory-unify): dry run first, then --apply; --undo takes it back.
    python scripts/memory_unify.py job --engine-url-file /tmp/kv_engine_url --secret-file /tmp/kv_api_secret [--apply] [--undo] --out report.md

The inventory says what each old store holds per family; the job (the engine's own code, with the duplicate check, a model
judgement) says what of it the care record already has and what moves. Secrets are read from the files given, never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict


def short(fid: object) -> str:
    s = str(fid)
    return "i:" + s.split(":", 1)[1][:8] if s.startswith("instinct-only:") else s[:8]


# ── inventory (read only) ────────────────────────────────────────────────────


def mongo_inventory(uri: str) -> tuple[dict[str, Counter], dict[str, str]]:
    """Per backend family: counts in each old Mongo store. Also the Instinct family id → backend family id map."""
    import pymongo

    db = pymongo.MongoClient(uri, serverSelectionTimeoutMS=15000).get_default_database()
    out: dict[str, Counter] = defaultdict(Counter)
    for p in db.elderprofiles.find({}, {"familyId": 1, "facts": 1, "tuning": 1, "nameToUse": 1, "allergies": 1, "dietRules": 1, "medicines": 1}):
        c = out[str(p.get("familyId"))]
        c["profiles"] += 1
        for f in p.get("facts") or []:
            c[f"profile_fact:{f.get('status') or 'learned'}"] += 1
            c["profile_fact:pinned"] += bool(f.get("pinned"))
            c["profile_fact:blocked"] += bool(f.get("blocked"))
        t = p.get("tuning") or {}
        c["tuning:addressAs"] += bool(t.get("addressAs"))
        c["tuning:avoidMaa"] += bool(t.get("avoidMaa"))
        c["nameToUse"] += bool(p.get("nameToUse"))
        c["allergies"] += len(p.get("allergies") or [])
        c["dietRules"] += len(p.get("dietRules") or [])
        c["medicines"] += len(p.get("medicines") or [])
    for u in db.elderusuals.find({}, {"familyId": 1, "items": 1, "brandPrefs": 1}):
        c = out[str(u.get("familyId"))]
        c["usual_items"] += len(u.get("items") or [])
        c["usual_brands"] += len(u.get("brandPrefs") or [])
    for s in db.sahelicompanions.find({}, {"familyId": 1, "birthday": 1, "importantDates": 1}):
        c = out[str(s.get("familyId"))]
        c["companion_birthday"] += bool(s.get("birthday"))
        c["companion_dates"] += len(s.get("importantDates") or [])
    for v in db.voice_preferences.find({}, {"familyId": 1}):
        out[str(v.get("familyId"))]["voice_preferences"] += 1
    for v in db.sahelipermissions.find({}, {"familyId": 1}):
        out[str(v.get("familyId"))]["permissions"] += 1
    for coll, key in (("elderwellbeingdays", "wellbeing_days"), ("dailysnapshots", "daily_snapshots")):
        for r in db[coll].aggregate([{"$group": {"_id": "$familyId", "n": {"$sum": 1}}}]):
            out[str(r["_id"])][key] += r["n"]
    tenants = {str(t.get("aiFamilyId")): str(t.get("familyId")) for t in db.aitenants.find({}, {"familyId": 1, "aiFamilyId": 1})}
    return out, tenants


def dsn_from(path: str) -> str:
    for line in open(path):
        v = line.strip()
        if "=" in v and not v.startswith("postgres"):
            v = v.split("=", 1)[1].strip().strip("'\"")
        if v.startswith("postgres"):
            return re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql://", v)
    raise SystemExit("no DSN in the env file")


async def pg_inventory(dsn: str, tenants: dict[str, str]) -> dict[str, Counter]:
    import asyncpg

    out: dict[str, Counter] = defaultdict(Counter)
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        for r in await conn.fetch("""SELECT family_id::text f, count(*) n,
                                            count(*) FILTER (WHERE entity_id IS NULL AND forgotten_at IS NULL AND superseded_by IS NULL) inbox
                                     FROM family_memories GROUP BY 1"""):
            fam = tenants.get(r["f"], "instinct-only:" + r["f"])
            out[fam]["instinct_rows"] += r["n"]
            out[fam]["instinct_inbox_live"] += r["inbox"]
        for r in await conn.fetch("SELECT family_id::text f, count(*) n FROM memory_entities GROUP BY 1"):
            out[tenants.get(r["f"], "instinct-only:" + r["f"])]["instinct_entities"] += r["n"]
        for r in await conn.fetch("SELECT family_id::text f, count(*) n FROM memory_profiles GROUP BY 1"):
            out[tenants.get(r["f"], "instinct-only:" + r["f"])]["instinct_profiles"] += r["n"]
        for r in await conn.fetch("""SELECT family_id f, count(*) FILTER (WHERE status = 'active') active,
                                            count(*) FILTER (WHERE status = 'pending') pending,
                                            count(*) FILTER (WHERE source_ref LIKE 'import:%') imported
                                     FROM care_facts WHERE family_id NOT LIKE 'shadow:%' AND family_id NOT LIKE 'sim-%' GROUP BY 1"""):
            c = out[r["f"]]
            c["engine_facts_active"] += r["active"]
            c["engine_facts_pending"] += r["pending"]
            c["engine_facts_imported"] += r["imported"]
        for r in await conn.fetch("""SELECT family_id f, count(*) n FROM memory_notes
                                     WHERE family_id NOT LIKE 'shadow:%' AND family_id NOT LIKE 'sim-%' GROUP BY 1"""):
            out[r["f"]]["engine_notes"] += r["n"]
    finally:
        await conn.close()
    return out


COLUMNS = [
    ("profiles", "ElderProfile docs"), ("profile_fact:learned", "facts learned"), ("profile_fact:caregiver_confirmed", "confirmed"),
    ("profile_fact:caregiver_edited", "edited"), ("profile_fact:rejected", "rejected"), ("profile_fact:faded", "faded"),
    ("profile_fact:pinned", "pinned"), ("profile_fact:blocked", "blocked"), ("tuning:addressAs", "addressAs"), ("tuning:avoidMaa", "avoidMaa"),
    ("nameToUse", "nameToUse"), ("allergies", "allergies"), ("dietRules", "diet"), ("medicines", "medicines"),
    ("usual_items", "usual items"), ("usual_brands", "brand prefs"), ("companion_birthday", "birthday"), ("companion_dates", "dates"),
    ("voice_preferences", "voice prefs"), ("permissions", "permissions"), ("wellbeing_days", "wellbeing days*"),
    ("daily_snapshots", "snapshots*"), ("instinct_rows", "Instinct rows"), ("instinct_inbox_live", "inbox (live)"),
    ("instinct_entities", "entities"), ("instinct_profiles", "profiles"), ("engine_facts_active", "engine facts"),
    ("engine_facts_pending", "engine pending"), ("engine_facts_imported", "engine imported"), ("engine_notes", "engine notes"),
]


def inventory_md(rows: dict[str, Counter]) -> str:
    fams = sorted(f for f in rows if any(rows[f].values()))
    head = "| store | " + " | ".join(short(f) for f in fams) + " | total |"
    lines = [head, "|" + "---|" * (len(fams) + 2)]
    for key, label in COLUMNS:
        vals = [rows[f][key] for f in fams]
        if not any(vals):
            continue
        lines.append(f"| {label} | " + " | ".join(str(v) for v in vals) + f" | {sum(vals)} |")
    lines.append("")
    lines.append("i:… = an Instinct family with no backend family linked (no AiTenant row): nothing to move it to.")
    lines.append("\\* not moved: rebuilt by the engine's mood, baselines and nightly brief (kept read only until the backups age out).")
    return "\n".join(lines)


def inventory(args) -> str:
    rows, tenants = mongo_inventory(open(args.mongo_uri_file).read().strip())
    for fam, c in asyncio.run(pg_inventory(dsn_from(args.pg_env), tenants)).items():
        rows[fam].update(c)
    return inventory_md(rows)


# ── the move job on the deployed engine ──────────────────────────────────────


def call_job(args) -> dict:
    base = open(args.engine_url_file).read().strip().rstrip("/")
    secret = open(args.secret_file).read().strip()
    token = subprocess.run([args.gcloud, "auth", "print-identity-token"], capture_output=True, text=True, check=True).stdout.strip()
    q = f"apply={'true' if args.apply else 'false'}&undo={'true' if args.undo else 'false'}&family={args.family}&store={args.store}"
    req = urllib.request.Request(f"{base}/v2/jobs/memory-unify?{q}", method="POST", data=b"",
                                 headers={"Authorization": f"Bearer {token}", "X-Kavach-Secret": secret})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def job_md(out: dict) -> str:
    fams = out.get("families") or {}
    lines = [f"applied: {out.get('applied')} · undo: {out.get('undo')}", ""]
    if out.get("undo"):
        lines += ["| family | " + " | ".join(["undone", "facts removed", "facts restored", "pending removed", "never-use removed", "note lines", "changed since"]) + " |",
                  "|---|---|---|---|---|---|---|---|"]
        for f, r in sorted(fams.items()):
            lines.append(f"| {f} | " + " | ".join(str(r.get(k, 0)) for k in ("undone", "fact_removed", "fact_restored", "pending_removed",
                                                                                    "rejected_removed", "note_lines", "changed_since")) + " |")
        return "\n".join(lines)
    outcomes = sorted({k for r in fams.values() for st in (r.get("by_store") or {}).values() for k in st})
    for f, r in sorted(fams.items()):
        if r.get("error"):
            lines.append(f"### {f}: error ({r['error']})")
            continue
        lines.append(f"### {f}: {r.get('items', 0)} items, {r.get('people', 0)} people, {r.get('model_calls', 0)} model calls, permissions {r.get('permissions')}")
        stores = r.get("by_store") or {}
        if stores:
            lines.append("| store | " + " | ".join(outcomes) + " |")
            lines.append("|" + "---|" * (len(outcomes) + 1))
            for st, c in sorted(stores.items()):
                lines.append(f"| {st} | " + " | ".join(str(c.get(o, 0)) for o in outcomes) + " |")
        if r.get("profile_facts_by_status"):
            lines.append("profile facts by old status: " + ", ".join(f"{k.split(':', 1)[1]} {v}" for k, v in sorted(r["profile_facts_by_status"].items())))
        lines.append("")
    lines.append("totals: " + ", ".join(f"{k} {v}" for k, v in sorted((out.get("totals") or {}).items())))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--mongo-uri-file", required=True)
    inv.add_argument("--pg-env", required=True)
    inv.add_argument("--out", default="")
    job = sub.add_parser("job")
    job.add_argument("--engine-url-file", required=True)
    job.add_argument("--secret-file", required=True)
    job.add_argument("--apply", action="store_true")
    job.add_argument("--undo", action="store_true")
    job.add_argument("--family", default="")
    job.add_argument("--store", default="")
    job.add_argument("--gcloud", default="gcloud")
    job.add_argument("--json", default="", help="also save the job's raw counts (JSON) here")
    job.add_argument("--out", default="")
    args = ap.parse_args()
    if args.cmd == "inventory":
        text = inventory(args)
    else:
        out = call_job(args)
        if args.json:
            json.dump(out, open(args.json, "w"), indent=2)
        text = job_md(out)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
