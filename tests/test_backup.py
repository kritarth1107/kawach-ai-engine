"""Backup retention and store logic (the dump/encrypt/restore round trip is backup/roundtrip.sh)."""

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("backup", Path(__file__).parent.parent / "backup" / "backup.py")
backup = importlib.util.module_from_spec(spec)
sys.modules["backup"] = backup
spec.loader.exec_module(backup)


def days(n, end=date(2026, 10, 4)):
    return [end - timedelta(days=i) for i in range(n)]


def test_keep_dates_daily_weekly_monthly():
    today = date(2026, 10, 4)
    keep = backup.keep_dates(days(400), today)
    assert set(days(14)) <= keep
    sundays = [d for d in days(400) if d.weekday() == 6][:8]
    firsts = [d for d in days(400) if d.day == 1][:12]
    assert set(sundays) <= keep and set(firsts) <= keep
    assert len(keep) <= 14 + 8 + 12
    assert date(2026, 9, 2) not in keep  # a plain weekday older than 14 days


def test_keep_dates_never_drops_future():
    today = date(2026, 10, 4)
    assert date(2026, 10, 5) in backup.keep_dates([date(2026, 10, 5)], today)


def test_parse_prefix():
    assert backup.parse_prefix("2026/10/04/postgres.dump.age") == date(2026, 10, 4)
    assert backup.parse_prefix("latest.json") is None
    assert backup.parse_prefix("2026/13/04/x") is None


def test_libpq_url():
    url = "postgresql+asyncpg://kawach:pw@/kawach_ai?host=/cloudsql/p:r:i"
    assert backup.libpq_url(url) == "postgresql://kawach:pw@/kawach_ai?host=/cloudsql/p:r:i"


def _seed(store, d, ok=True):
    p = backup.date_prefix(d)
    store.put_bytes(f"{p}/postgres.dump.age", b"x")
    store.put_bytes(f"{p}/manifest.json", json.dumps({"ok": ok}).encode())


def test_prune_keeps_policy_and_ignores_failed_days(tmp_path):
    store = backup.DirStore(str(tmp_path))
    today = date(2026, 10, 4)
    for d in days(60):
        _seed(store, d)
    # a run of failed days must not push good backups out of the 14-day window
    for d in days(5):
        _seed(store, d, ok=False)
    store.put_bytes("latest.json", b"{}")
    gone = backup.prune(store, today)
    kept_days = {backup.parse_prefix(k) for k in store.keys() if backup.parse_prefix(k)}
    good = [d for d in days(60) if d not in days(5)]
    assert set(good[:14]) <= kept_days
    assert set(days(5)) <= kept_days  # recent failures stay for inspection
    assert store.get_bytes("latest.json") == b"{}"
    assert gone and all(backup.parse_prefix(k) not in kept_days for k in gone)


def test_restore_refuses_prod_and_missing(tmp_path):
    store = backup.DirStore(str(tmp_path))
    with pytest.raises(SystemExit, match="no backup"):
        backup.restore(store, "2026-10-04", "id", pg_url="postgresql://x/scratch", mongo_uri=None,
                       prod=backup.Sources("postgresql://x/prod", None))
    _seed(store, date(2026, 10, 4))
    with pytest.raises(SystemExit, match="production"):
        backup.restore(store, "2026-10-04", "id", pg_url="postgresql://x/prod", mongo_uri=None,
                       prod=backup.Sources("postgresql://x/prod", None))
    _seed(store, date(2026, 10, 3), ok=False)
    with pytest.raises(SystemExit, match="did not finish"):
        backup.restore(store, "2026-10-03", "id", pg_url="postgresql://x/scratch", mongo_uri=None,
                       prod=backup.Sources("postgresql://x/prod", None))


async def test_overview_shows_last_backup(db):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.learn import jobs

    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    await db.execute(text("DROP TABLE IF EXISTS backup_runs"))
    assert await jobs.last_backup(sessions) is None
    await db.execute(text(
        "CREATE TABLE backup_runs (id BIGSERIAL PRIMARY KEY, day DATE NOT NULL, ok BOOLEAN NOT NULL,"
        " started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, bytes BIGINT DEFAULT 0, detail JSONB)"))
    await db.execute(text("INSERT INTO backup_runs (day, ok, finished_at, bytes, detail) VALUES"
                          " ('2026-10-03', true, '2026-10-03T22:00:00Z', 1234, '{}'),"
                          " ('2026-10-04', false, '2026-10-04T22:00:00Z', 0, '{\"error\": \"pg_dump exit 1\"}')"))
    out = await jobs.last_backup(sessions)
    assert out["day"] == "2026-10-04" and out["ok"] is False and out["error"] == "pg_dump exit 1" and out["families"] is None
    await db.execute(text("INSERT INTO backup_runs (day, ok, finished_at, bytes, detail) VALUES"
                          " ('2026-10-05', true, '2026-10-05T22:00:00Z', 9, '{\"families\": {\"count\": 3, \"failed\": 0}}')"))
    assert (await jobs.last_backup(sessions))["families"] == {"count": 3, "failed": 0}
    assert out["lastOkAt"].startswith("2026-10-03")
