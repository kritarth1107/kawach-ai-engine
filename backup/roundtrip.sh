#!/usr/bin/env bash
# Local proof that a backup can be restored: seed Postgres + Mongo, back up to a folder,
# restore into scratch databases, and compare. Uses throwaway containers only.
#   backup/roundtrip.sh
set -euo pipefail
cd "$(dirname "$0")"
IMG=kavach-backup:test
NET=kavach-bk-$$
WORK=$(mktemp -d)
cleanup() { docker rm -f "$NET-pg" "$NET-mongo" >/dev/null 2>&1 || true; docker network rm "$NET" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

docker build -q -t "$IMG" . >/dev/null
docker network create "$NET" >/dev/null
docker run -d --name "$NET-pg" --network "$NET" -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg16 >/dev/null
docker run -d --name "$NET-mongo" --network "$NET" mongo:7 >/dev/null
for _ in $(seq 60); do docker exec "$NET-pg" pg_isready -U postgres >/dev/null 2>&1 && break; sleep 1; done
for _ in $(seq 60); do docker exec "$NET-mongo" mongosh --quiet --eval 'db.runCommand({ping:1}).ok' >/dev/null 2>&1 && break; sleep 1; done
sleep 2

psql() { docker exec -i "$NET-pg" psql -U postgres -v ON_ERROR_STOP=1 -qtA "$@"; }
psql -c "CREATE DATABASE prod" >/dev/null; psql -c "CREATE DATABASE scratch" >/dev/null
psql -d prod <<'SQL' >/dev/null
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE care_facts (id serial primary key, family_id text, body text, embedding vector(3));
INSERT INTO care_facts (family_id, body, embedding)
  SELECT 'fam-' || (g % 7), 'Metformin 500 mg 08:00 — दवाई ' || g, '[1,2,3]' FROM generate_series(1, 500) g;
SQL
docker exec "$NET-mongo" mongosh --quiet kavach --eval '
  db.users.insertMany(Array.from({length: 300}, (_, i) => ({phone: "+9190000" + String(i).padStart(5, "0"), name: "User " + i})));
  db.families.insertOne({name: "Kritarth’s Family", members: 3});' >/dev/null

docker run --rm --user "$(id -u):$(id -g)" -v "$WORK:/keys" --entrypoint age-keygen "$IMG" -o /keys/id.txt 2>/dev/null
RECIPIENT=$(grep -o 'age1[0-9a-z]*' "$WORK/id.txt")

run() { docker run --rm --user "$(id -u):$(id -g)" --network "$NET" -v "$WORK:/work" -e BACKUP_TARGET=dir:/work/store \
  -e BACKUP_PG_URL="postgresql://postgres:pw@$NET-pg:5432/prod" \
  -e BACKUP_MONGODB_URI="mongodb://$NET-mongo:27017/kavach" \
  -e BACKUP_AGE_RECIPIENT="$RECIPIENT" -e BACKUP_AGE_IDENTITY_FILE=/work/id.txt "$IMG" "$@"; }

run run >"$WORK/run.json"
grep -q '"ok": true' "$WORK/run.json" || { cat "$WORK/run.json"; echo "FAIL: backup not ok"; exit 1; }
# the stored files must not contain plain text
if grep -rqa "Metformin" "$WORK/store"; then echo "FAIL: plaintext in backup"; exit 1; fi
DAY=$(run list | tail -1 | tr / -)

# one encrypted memory snapshot per family, readable with the private key
grep -q '"count": 7' "$WORK/run.json" || { cat "$WORK/run.json"; echo "FAIL: family snapshots"; exit 1; }
run family fam-3 "$DAY" --out /work/fam3.json >/dev/null
F=$(python3 -c "import json;d=json.load(open('$WORK/fam3.json'));print(d['family_id'], len(d['facts']))")
[ "$F" = "fam-3 72" ] || { echo "FAIL: family snapshot content $F"; exit 1; }

# restoring over the source must be refused
if run restore "$DAY" --pg-url "postgresql://postgres:pw@$NET-pg:5432/prod" >/dev/null 2>&1; then echo "FAIL: restore over prod allowed"; exit 1; fi

run restore "$DAY" --pg-url "postgresql://postgres:pw@$NET-pg:5432/scratch" >/dev/null

A=$(psql -d prod -c "SELECT count(*), md5(string_agg(body || embedding::text, ',' ORDER BY id)) FROM care_facts")
B=$(psql -d scratch -c "SELECT count(*), md5(string_agg(body || embedding::text, ',' ORDER BY id)) FROM care_facts")
[ "$A" = "$B" ] || { echo "FAIL: postgres mismatch $A vs $B"; exit 1; }

# mongo: drop the source, restore, compare
docker exec "$NET-mongo" mongosh --quiet kavach --eval 'db.dropDatabase()' >/dev/null
run restore "$DAY" --mongo-uri "mongodb://$NET-mongo:27017" --target-prod --yes >/dev/null
M=$(docker exec "$NET-mongo" mongosh --quiet kavach --eval 'db.users.countDocuments() + "," + db.families.findOne().name')
[ "$M" = "300,Kritarth’s Family" ] || { echo "FAIL: mongo mismatch $M"; exit 1; }

# a damaged file must be caught by the checksum
f=$(find "$WORK/store" -name 'postgres.dump.age'); printf 'x' >>"$f"
if run restore "$DAY" --pg-url "postgresql://postgres:pw@$NET-pg:5432/scratch" >/dev/null 2>&1; then echo "FAIL: damaged backup restored"; exit 1; fi

echo "OK: postgres $A, mongo $M, 7 family snapshots ($F), encrypted, prod-guard and checksum checks pass"
