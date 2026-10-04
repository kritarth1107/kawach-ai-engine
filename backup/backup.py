"""Encrypted nightly backups of the engine Postgres and the backend MongoDB, kept outside Google.

    python backup.py run                       # dump, encrypt, upload, record, prune
    python backup.py list                      # dates in the store
    python backup.py restore DATE [--pg-url URL] [--mongo-uri URI] [--target-prod --yes]
    python backup.py latest                    # newest manifest as JSON (for verify.sh)

The job holds only the age *public* key (BACKUP_AGE_RECIPIENT), so a leaked job cannot read a backup.
Restores need the private key file (BACKUP_AGE_IDENTITY_FILE) and go to a scratch database unless
--target-prod --yes is given.

Store: BACKUP_TARGET=r2 (default) uses BACKUP_R2_ENDPOINT / BACKUP_R2_ACCESS_KEY_ID /
BACKUP_R2_SECRET_ACCESS_KEY / BACKUP_R2_BUCKET. BACKUP_TARGET=dir:/path writes to a local folder (tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
KEEP_DAILY, KEEP_WEEKLY, KEEP_MONTHLY = 14, 8, 12


# ── retention ────────────────────────────────────────────────────────────────

def keep_dates(dates: list[date], today: date) -> set[date]:
    """14 newest daily, plus the 8 newest Sundays and the 12 newest 1st-of-months (both counted among all dates)."""
    ds = sorted({d for d in dates if d <= today}, reverse=True)
    keep = set(ds[:KEEP_DAILY])
    keep |= set([d for d in ds if d.weekday() == 6][:KEEP_WEEKLY])
    keep |= set([d for d in ds if d.day == 1][:KEEP_MONTHLY])
    keep |= {d for d in dates if d > today}  # clock skew: never delete the future
    return keep


def date_prefix(d: date) -> str:
    return f"{d:%Y/%m/%d}"


def parse_prefix(key: str) -> date | None:
    """'2026/10/04/postgres.dump.age' → date(2026, 10, 4)."""
    parts = key.split("/")
    if len(parts) < 4:
        return None
    try:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


# ── stores ───────────────────────────────────────────────────────────────────

class DirStore:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, src: Path) -> None:
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def put_bytes(self, key: str, data: bytes) -> None:
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    def get(self, key: str, dst: Path) -> None:
        shutil.copyfile(self.root / key, dst)

    def get_bytes(self, key: str) -> bytes | None:
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def keys(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)


class S3Store:
    def __init__(self) -> None:
        import boto3

        self.bucket = need("BACKUP_R2_BUCKET")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=need("BACKUP_R2_ENDPOINT"),
            aws_access_key_id=need("BACKUP_R2_ACCESS_KEY_ID"),
            aws_secret_access_key=need("BACKUP_R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )

    def put(self, key: str, src: Path) -> None:
        self.s3.upload_file(str(src), self.bucket, key)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType="application/json")

    def get(self, key: str, dst: Path) -> None:
        self.s3.download_file(self.bucket, key, str(dst))

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def keys(self) -> list[str]:
        out, token = [], None
        while True:
            kw = {"Bucket": self.bucket, **({"ContinuationToken": token} if token else {})}
            page = self.s3.list_objects_v2(**kw)
            out += [o["Key"] for o in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return sorted(out)
            token = page["NextContinuationToken"]

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)


def need(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"missing env {name}")
    return v


def store_from_env():
    target = os.environ.get("BACKUP_TARGET", "r2")
    if target.startswith("dir:"):
        return DirStore(target[4:])
    return S3Store()


# ── helpers ──────────────────────────────────────────────────────────────────

def libpq_url(url: str) -> str:
    """SQLAlchemy URL (postgresql+asyncpg://…) → libpq URL that pg_dump understands."""
    for drv in ("+asyncpg", "+psycopg2", "+psycopg"):
        url = url.replace(drv, "")
    return url


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "?"


def dump_encrypted(dump_cmd: list[str], out: Path, recipient: str) -> None:
    """dump | age -r recipient > out, failing if either side fails."""
    with out.open("wb") as f:
        dump = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE)
        enc = subprocess.Popen(["age", "-r", recipient], stdin=dump.stdout, stdout=f)
        assert dump.stdout is not None
        dump.stdout.close()
        enc_rc, dump_rc = enc.wait(), dump.wait()
    if dump_rc or enc_rc:
        raise RuntimeError(f"{dump_cmd[0]} exit {dump_rc}, age exit {enc_rc}")


def decrypt(src: Path, dst: Path, identity: str) -> None:
    subprocess.run(["age", "-d", "-i", identity, "-o", str(dst), str(src)], check=True)


def record_run(manifest: dict) -> None:
    """Best effort: one row in the engine's backup_runs so the founder page can show 'last backup OK'."""
    url = os.environ.get("BACKUP_PG_URL") or os.environ.get("DATABASE_URL")
    if not url:
        return
    sql = (
        "CREATE TABLE IF NOT EXISTS backup_runs (id BIGSERIAL PRIMARY KEY, day DATE NOT NULL, ok BOOLEAN NOT NULL,"
        " started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, bytes BIGINT DEFAULT 0, detail JSONB);"
        "INSERT INTO backup_runs (day, ok, started_at, finished_at, bytes, detail) VALUES"
        " (:'day', :'ok', :'started', :'finished', :'bytes', :'detail');"
    )
    total = sum(p.get("bytes", 0) for p in manifest.get("parts", {}).values())
    detail = json.dumps({k: v for k, v in manifest.items() if k != "parts"} | {"parts": list(manifest.get("parts", {}))})
    try:
        subprocess.run(
            ["psql", libpq_url(url), "-v", "ON_ERROR_STOP=1", "-q",
             "-v", f"day={manifest['day']}", "-v", f"ok={'true' if manifest['ok'] else 'false'}",
             "-v", f"started={manifest['startedAt']}", "-v", f"finished={manifest['finishedAt']}",
             "-v", f"bytes={total}", "-v", f"detail={detail}"],
            input=sql, text=True, check=True, capture_output=True, timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not record run: {exc}", file=sys.stderr)


# ── commands ─────────────────────────────────────────────────────────────────

@dataclass
class Sources:
    pg_url: str | None
    mongo_uri: str | None


def sources_from_env() -> Sources:
    pg = os.environ.get("BACKUP_PG_URL") or os.environ.get("DATABASE_URL")
    mongo = os.environ.get("BACKUP_MONGODB_URI") or os.environ.get("MONGODB_URI")
    return Sources(libpq_url(pg) if pg else None, mongo or None)


def run_backup(store, src: Sources, recipient: str, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(IST)
    day = now.astimezone(IST).date()
    prefix = date_prefix(day)
    manifest: dict = {
        "day": day.isoformat(), "startedAt": now.isoformat(), "ok": False, "parts": {},
        "tools": {"pg_dump": tool_version(["pg_dump", "--version"]), "mongodump": tool_version(["mongodump", "--version"]),
                  "age": tool_version(["age", "--version"])},
    }
    if not src.pg_url and not src.mongo_uri:
        raise SystemExit("nothing to back up: set BACKUP_PG_URL/DATABASE_URL and/or BACKUP_MONGODB_URI")
    with tempfile.TemporaryDirectory() as tmp:
        jobs = []
        if src.pg_url:
            jobs.append(("postgres", "postgres.dump.age", ["pg_dump", "-Fc", "--no-owner", "--no-acl", src.pg_url]))
        if src.mongo_uri:
            jobs.append(("mongo", "mongo.archive.age", ["mongodump", f"--uri={src.mongo_uri}", "--archive", "--gzip", "--quiet"]))
        try:
            for name, fname, cmd in jobs:
                out = Path(tmp) / fname
                dump_encrypted(cmd, out, recipient)
                key = f"{prefix}/{fname}"
                store.put(key, out)
                manifest["parts"][name] = {"key": key, "bytes": out.stat().st_size, "sha256": sha256(out)}
            manifest["ok"] = True
        except Exception as exc:  # noqa: BLE001
            manifest["error"] = str(exc)[:500]
        manifest["finishedAt"] = datetime.now(IST).isoformat()
        body = json.dumps(manifest, indent=2).encode()
        store.put_bytes(f"{prefix}/manifest.json", body)
        if manifest["ok"]:
            store.put_bytes("latest.json", body)
    record_run(manifest)
    if manifest["ok"]:
        manifest["pruned"] = prune(store, day)
    return manifest


def prune(store, today: date) -> list[str]:
    keys = store.keys()
    by_day: dict[date, list[str]] = {}
    for k in keys:
        d = parse_prefix(k)
        if d:
            by_day.setdefault(d, []).append(k)
    # only days with a successful manifest count as backups; a failed day never pushes a good one out
    good = []
    for d, ks in by_day.items():
        raw = store.get_bytes(f"{date_prefix(d)}/manifest.json")
        if raw and json.loads(raw).get("ok"):
            good.append(d)
    keep = keep_dates(good, today)
    gone = []
    for d, ks in by_day.items():
        if d in keep or (d not in good and (today - d).days < KEEP_DAILY):
            continue
        for k in ks:
            store.delete(k)
            gone.append(k)
    return gone


def list_days(store) -> list[str]:
    return sorted({k.rsplit("/", 1)[0] for k in store.keys() if parse_prefix(k)})


def restore(store, day: str, identity: str, *, pg_url: str | None, mongo_uri: str | None,
            prod: Sources, target_prod: bool = False, yes: bool = False) -> dict:
    d = date.fromisoformat(day)
    raw = store.get_bytes(f"{date_prefix(d)}/manifest.json")
    if not raw:
        raise SystemExit(f"no backup for {day}")
    manifest = json.loads(raw)
    if not manifest.get("ok"):
        raise SystemExit(f"backup for {day} did not finish: {manifest.get('error')}")
    for target, prod_url in ((pg_url, prod.pg_url), (mongo_uri, prod.mongo_uri)):
        if target and prod_url and target.rstrip("/") == prod_url.rstrip("/") and not (target_prod and yes):
            raise SystemExit("refusing to restore over production; pass --target-prod --yes to really do it")
    done = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, target in (("postgres", pg_url), ("mongo", mongo_uri)):
            part = manifest["parts"].get(name)
            if not part or not target:
                continue
            enc = Path(tmp) / f"{name}.age"
            store.get(part["key"], enc)
            if sha256(enc) != part["sha256"]:
                raise SystemExit(f"{name}: checksum mismatch, backup is damaged")
            plain = Path(tmp) / name
            decrypt(enc, plain, identity)
            if name == "postgres":
                subprocess.run(["pg_restore", "--clean", "--if-exists", "--no-owner", "--no-acl", "-d", libpq_url(target), str(plain)],
                               check=True)
            else:
                subprocess.run(["mongorestore", f"--uri={target}", "--archive=" + str(plain), "--gzip", "--drop", "--quiet"], check=True)
            done[name] = part["key"]
    return {"day": day, "restored": done}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("list")
    sub.add_parser("latest")
    r = sub.add_parser("restore")
    r.add_argument("day")
    r.add_argument("--pg-url", default=os.environ.get("RESTORE_PG_URL"))
    r.add_argument("--mongo-uri", default=os.environ.get("RESTORE_MONGODB_URI"))
    r.add_argument("--target-prod", action="store_true")
    r.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    store = store_from_env()
    if a.cmd == "run":
        m = run_backup(store, sources_from_env(), need("BACKUP_AGE_RECIPIENT"))
        print(json.dumps(m, indent=2))
        return 0 if m["ok"] else 1
    if a.cmd == "list":
        print("\n".join(list_days(store)))
        return 0
    if a.cmd == "latest":
        raw = store.get_bytes("latest.json")
        print(raw.decode() if raw else "{}")
        return 0 if raw else 1
    if not a.pg_url and not a.mongo_uri:
        raise SystemExit("give --pg-url and/or --mongo-uri (a scratch database)")
    out = restore(store, a.day, need("BACKUP_AGE_IDENTITY_FILE"), pg_url=a.pg_url and libpq_url(a.pg_url), mongo_uri=a.mongo_uri,
                  prod=sources_from_env(), target_prod=a.target_prod, yes=a.yes)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
