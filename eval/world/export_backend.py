"""Export what the backend's world tests need (plan B1.4): the ci slice's families (people, backend state, places,
orders) as one gzipped JSON, and one recorded engine response per dashboard route (from the world reads).

    PYTHONPATH=. python -m eval.world.export_backend ../kavach-backend/scripts/fixtures
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from eval.world import make
from eval.world.slice import slice_ids

KEEP = ("id", "familyId", "tier", "household", "language", "people", "shapes", "places", "orders", "backend", "truth")


def main(argv: list[str]) -> int:
    out = Path(argv[0] if argv else "../kavach-backend/scripts/fixtures")
    out.mkdir(parents=True, exist_ok=True)
    world = make.build()
    ids = set(slice_ids("ci", world))
    fams = [{k: f[k] for k in KEEP} for f in world if f["id"] in ids]
    lock = json.loads((make.DATA / "world.lock").read_text())
    payload = {"world": lock["sha256"], "slice": "ci", "families": fams}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    with gzip.GzipFile(out / "world-ci.json.gz", "wb", mtime=0) as fh:
        fh.write(data)
    print(f"wrote {len(fams)} families ({len(data) // 1024} KB raw) to {out / 'world-ci.json.gz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
