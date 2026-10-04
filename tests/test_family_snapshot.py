"""Per-family memory snapshots: the backup job's SQL reads the real schema, snapshots are encrypted per family and kept
two years, and restoring one brings notes and skills back as undoable versions and facts only as pending changes."""

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import select, text

from app.care import skillbook, store, versions
from app.care.models import CareFact, OpenLoop
from app.jobs import restore_family

spec = importlib.util.spec_from_file_location("backup_mod", Path(__file__).parent.parent / "backup" / "backup.py")
backup = importlib.util.module_from_spec(spec)
sys.modules["backup_mod"] = backup
spec.loader.exec_module(backup)

FAM, ELDER, CG = "fam-snap", "e-snap", "c-snap"


async def seed(db):
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}, text="Metformin 500 mg at 08:00",
                           source_kind="caregiver_said", stated_by=CG)
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="allergy", key="allergy:milk", value={"allergen": "milk"},
                           text="Allergic to milk", source_kind="caregiver_said", stated_by=CG)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="family", title="Family", body_md="- son Ankit in Pune\n- loves bhajans")
    await store.upsert_note(db, family_id="shadow:x", subject_id=ELDER, slug="family", title="Family", body_md="- shadow")
    return await skillbook.save_family(db, FAM, ELDER, "Remind her after puja, not before", by=CG)


async def snapshot(db) -> dict:
    sql = backup.family_sql(set(backup.FAMILY_TABLES))
    raw = [r[0] for r in await db.execute(text(sql))]
    assert all("\n" not in r for r in raw)  # the backup job reads one family per line
    rows = [json.loads(r) for r in raw]
    assert all(not r["family_id"].startswith("shadow:") for r in rows)
    return next(r for r in rows if r["family_id"] == FAM)


async def test_snapshot_sql_reads_the_real_schema(db, at):
    at("2026-10-02 09:00")
    await seed(db)
    snap = await snapshot(db)
    assert {f["key"] for f in snap["facts"]} == {"medicine:metformin", "allergy:milk"}
    assert snap["notes"][0]["body_md"].startswith("- son Ankit") and snap["skills"][0]["body"].startswith("Remind her")
    # a fresh database without some tables still produces one snapshot per family
    only_facts = [json.loads(r[0]) for r in await db.execute(text(backup.family_sql({"care_facts"})))]
    assert [r["family_id"] for r in only_facts].count(FAM) == 1 and "skills" not in only_facts[0]
    assert backup.family_sql(set()) == ""


async def test_restore_dry_run_then_apply(db, at):
    at("2026-10-02 09:00")
    sk = await seed(db)
    snap = json.loads(json.dumps(await snapshot(db), default=str))
    # things go wrong after the snapshot
    at("2026-10-03 09:00")
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "1000 mg", "times": ["08:00", "20:00"]}, text="Metformin 1000 mg",
                           source_kind="caregiver_said", stated_by=CG, replace=True)
    await store.stop_fact(db, family_id=FAM, subject_id=ELDER, key="allergy:milk", reason="mistake", source_kind="caregiver_said", stated_by=CG)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="family", title="Family", body_md="- (wiped)")
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="new", title="New", body_md="- new since")
    await skillbook.decide(db, FAM, sk["id"], action="edit", by=CG, body="Short Hinglish replies")

    p = await restore_family.plan(db, snap)
    assert {f["key"] for f in p["facts"]} == {"medicine:metformin", "allergy:milk"}
    assert [n["slug"] for n in p["notes"]] == ["family"] and p["only_now"]["notes"] == [f"{ELDER}/new"]
    assert len(p["skills"]) == 1

    out = await restore_family.apply(db, snap, by=CG)
    assert out["done"] == {"notes": 1, "facts_pending": 2, "skills": 1, "skipped": []}
    # notes and skills are back, each as an undoable version
    n = (await store.notes(db, FAM, [ELDER]))
    assert next(x for x in n if x.slug == "family").body_md == "- son Ankit in Pune\n- loves bhajans"
    last = (await versions.changes(db, FAM, [ELDER], kinds=("note",), target="family"))[0]
    assert last.op == "snapshot" and last.source == "snapshot" and "snapshot" in last.reason
    assert (await db.get(skillbook.Skill, sk["id"])).body == "Remind her after puja, not before"
    # facts did NOT change: they wait for a caregiver
    act = {f.key: f for f in (await db.execute(select(CareFact).where(CareFact.family_id == FAM, CareFact.status == "active"))).scalars()}
    assert act["medicine:metformin"].value["dose"] == "1000 mg" and "allergy:milk" not in act
    loops = list((await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "confirm_fact",
                                                          OpenLoop.status == "open"))).scalars())
    assert len(loops) == 2
    # approving brings back the exact old value
    pend = next(f for f in await store.facts(db, FAM, ELDER, statuses=("pending",)) if f.key == "medicine:metformin")
    await store.resolve_pending(db, fact_id=pend.id, approve=True, by=CG)
    f = await store.active_fact(db, FAM, ELDER, "medicine:metformin")
    assert f.value == {"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}


def test_snapshot_upload_and_two_year_retention(tmp_path):
    st = backup.DirStore(str(tmp_path))
    rows = [json.dumps({"family_id": "fam/1", "facts": []}), json.dumps({"family_id": "fam-2"}), "not json"]
    out = backup.snapshot_families(st, rows, "age1x", date(2026, 10, 4), encrypt=lambda b: b[::-1])
    assert out["count"] == 2 and out["failed"] == 1
    k = backup.family_key("fam/1", date(2026, 10, 4))
    assert k == "families/fam_1/2026/10/04.json.age" and json.loads(st.get_bytes(k)[::-1])["family_id"] == "fam/1"
    # 800 days of snapshots: 14 daily + 8 Sundays + 24 firsts kept
    for i in range(800):
        st.put_bytes(backup.family_key("fam-2", date(2026, 10, 4) - timedelta(days=i)), b"x")
    backup.prune_families(st, date(2026, 10, 4))
    kept = st.keys("families/fam-2/")
    firsts = [k for k in kept if k.endswith("/01.json.age")]
    assert len(firsts) == 24 and len(kept) <= 14 + 8 + 24
    # the full-backup prune ignores family snapshots
    assert backup.parse_prefix(kept[0]) is None
