"""Export the anonymised corpus for fine-tuning Gemini Flash on Vertex AI (later, with founder OK).

    DATABASE_URL=… .venv/bin/python eval/export_tuning.py out_dir [--min-score 0.5]

Writes:
- sft.jsonl: good replies in Vertex's supervised-tuning format ({"contents": [user, model]}), the situation in
  the system instruction
- pairs.jsonl: {situation, context, chosen, rejected} for preference tuning (best vs worst reply in the same
  situation and language)
Only families that agreed are in the corpus, and it is already anonymised.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402


async def main(argv: list[str]) -> int:
    from app.learn.models import LearningExample
    from app.learn.situations import SITUATIONS

    out = Path(argv[0] if argv else "tuning_export")
    out.mkdir(parents=True, exist_ok=True)
    min_score = float(argv[argv.index("--min-score") + 1]) if "--min-score" in argv else 0.5
    async with SessionLocal() as s:
        rows = list((await s.execute(select(LearningExample))).scalars())
    sft = [r for r in rows if r.score >= min_score and r.context.strip()]
    with (out / "sft.jsonl").open("w") as f:
        for r in sft:
            f.write(json.dumps({
                "systemInstruction": {"role": "system", "parts": [{"text": f"You are Saheli, a WhatsApp care companion. Situation: {SITUATIONS.get(r.situation, r.situation)}."}]},
                "contents": [{"role": "user", "parts": [{"text": r.context}]}, {"role": "model", "parts": [{"text": r.reply}]}],
            }, ensure_ascii=False) + "\n")
    groups = defaultdict(list)
    for r in rows:
        groups[(r.situation, r.lang)].append(r)
    pairs = 0
    with (out / "pairs.jsonl").open("w") as f:
        for (situation, lang), rs in groups.items():
            good = sorted([r for r in rs if r.score >= min_score], key=lambda r: -r.score)
            bad = sorted([r for r in rs if r.score <= -0.2], key=lambda r: r.score)
            for g, b in zip(good, bad):
                f.write(json.dumps({"situation": situation, "lang": lang, "context": b.context, "chosen": g.reply, "rejected": b.reply}, ensure_ascii=False) + "\n")
                pairs += 1
    print(json.dumps({"examples": len(rows), "sft": len(sft), "pairs": pairs, "dir": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
