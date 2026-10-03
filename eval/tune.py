"""Fine-tune Saheli's brain on Vertex AI (Gemini supervised tuning). Every paid or uploading step needs --yes.

    .venv/bin/python eval/tune.py export  out_dir [--min-score 0.4] [--no-tools]   # free: build train/validation/holdout JSONL
    .venv/bin/python eval/tune.py upload  out_dir gs://bucket/path --project P --yes
    .venv/bin/python eval/tune.py start   gs://bucket/path --project P --region us-central1 --base gemini-3.5-flash --name saheli-v1 --yes
    .venv/bin/python eval/tune.py status  JOB_NAME --project P --region us-central1
    .venv/bin/python eval/tune.py route   ENDPOINT_RESOURCE            # prints the MODEL_ROUTES entry to try the tuned model

Training data leaves India (tuning runs in us-central1 / europe-west4): only consented, anonymised families.
Before going live a tuned model goes through eval/compare_models.py, eval/train.sh compare and the 10% trial.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GCLOUD = str(Path.home() / "google-cloud-sdk/bin/gcloud")


def _flag(argv: list[str], name: str, default: str | None = None) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else default


def _need_yes(argv: list[str], what: str) -> None:
    if "--yes" not in argv:
        sys.exit(f"Refusing to {what} without --yes (the founder's OK).")


async def export(argv: list[str]) -> int:
    from app.db.session import SessionLocal
    from app.learn import tuning

    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    async with SessionLocal() as s:
        split, stats = await tuning.build(s, min_score=float(_flag(argv, "--min-score", "0.4")), with_tools="--no-tools" not in argv)
    for name, rows in (("train", split.train), ("validation", split.validation), ("holdout", split.holdout)):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.update({k: len(getattr(split, k)) for k in ("train", "validation", "holdout")})
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if stats["train"] < 100:
        print("Note: Google recommends at least 100–500 good examples; collect more before tuning.")
    return 0


def upload(argv: list[str]) -> int:
    _need_yes(argv, "upload training data")
    src, dst, project = argv[0], argv[1], _flag(argv, "--project")
    if project == "kavach-care" and "--allow-prod-project" not in argv:
        sys.exit("Use the training project (or pass --allow-prod-project deliberately).")
    for name in ("train", "validation"):
        subprocess.run([GCLOUD, "storage", "cp", f"{src}/{name}.jsonl", f"{dst}/{name}.jsonl", f"--project={project}"], check=True)
    print(f"uploaded to {dst}")
    return 0


def start(argv: list[str]) -> int:
    _need_yes(argv, "start a paid tuning job")
    import urllib.request

    dst, project = argv[0], _flag(argv, "--project")
    region, base, name = _flag(argv, "--region", "us-central1"), _flag(argv, "--base", "gemini-3.5-flash"), _flag(argv, "--name", "saheli-tuned")
    token = subprocess.run([GCLOUD, "auth", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip()
    body = {"baseModel": base, "supervisedTuningSpec": {"trainingDatasetUri": f"{dst}/train.jsonl", "validationDatasetUri": f"{dst}/validation.jsonl"},
            "tunedModelDisplayName": name}
    req = urllib.request.Request(f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/tuningJobs",
                                 data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        print(r.read().decode())
    return 0


def status(argv: list[str]) -> int:
    import urllib.request

    job, project, region = argv[0], _flag(argv, "--project"), _flag(argv, "--region", "us-central1")
    token = subprocess.run([GCLOUD, "auth", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip()
    path = job if job.startswith("projects/") else f"projects/{project}/locations/{region}/tuningJobs/{job}"
    req = urllib.request.Request(f"https://{region}-aiplatform.googleapis.com/v1/{path}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as r:
        d = json.loads(r.read().decode())
    print(json.dumps({k: d.get(k) for k in ("name", "state", "tunedModel", "error", "tuningDataStats")}, indent=2))
    return 0


def route(argv: list[str]) -> int:
    endpoint = argv[0]
    region = endpoint.split("/locations/")[1].split("/")[0] if "/locations/" in endpoint else "us-central1"
    print(json.dumps({"brain_tuned": [f"gemini:{endpoint}@{region}", "gemini:gemini-3.5-flash@asia-south1"]}))
    print("Add this to MODEL_ROUTES and set BRAIN_ROLE=brain_tuned for a compare run (never straight to production).")
    return 0


def main(argv: list[str]) -> int:
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "export":
        return asyncio.run(export(rest))
    return {"upload": upload, "start": start, "status": status, "route": route}.get(cmd, lambda _: (print(__doc__), 2)[1])(rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
