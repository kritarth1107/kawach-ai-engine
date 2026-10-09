"""When is fine-tuning worth it? A readiness report over the anonymised, consented corpus (nothing is trained here).

Tuning a fast model on Saheli's best real replies (dialects, elder tone) is the way to get top-model quality at a fast
model's cost, but only with enough good examples across situations and languages. Until then it would overfit a few
conversations. GET /v2/learn/tuning-readiness shows how far we are; eval/export_tuning.py + eval/train.sh tune run it
(paid, founder OK, separate project).
"""

from __future__ import annotations

from collections import Counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import FamilyRoster

MIN_SFT = 2000          # good replies in total
MIN_PER_SITUATION = 100  # in each of the main situations
MIN_PAIRS = 300          # best-vs-worst pairs for preference tuning
MAIN_SITUATIONS = ("dose", "chit_chat", "order", "health", "caregiver_update")
GOOD = 0.5


async def readiness(session: AsyncSession) -> dict:
    from app.care import outcomes
    from app.learn import corrections
    from app.learn.models import LearningExample

    rows = list((await session.execute(select(LearningExample))).scalars())
    good = [r for r in rows if r.score >= GOOD and (r.context or "").strip()]
    by_sit = Counter(r.situation for r in good)
    by_lang = Counter(r.lang for r in good)
    groups = Counter((r.situation, r.lang) for r in rows if r.score <= -0.2)
    pairs = sum(min(groups[k], Counter((r.situation, r.lang) for r in good)[k]) for k in groups)
    fams = [r for r in (await session.execute(select(FamilyRoster).where(~FamilyRoster.family_id.startswith("shadow:")))).scalars()]
    consented = 0
    for f in fams:
        elder = (f.elder or {}).get("id")
        if elder and (await outcomes.consent(session, f.family_id, elder)).get("granted"):
            consented += 1
    corr = await corrections.cases(session, days=180, limit=2000)
    missing = []
    if consented == 0:
        missing.append("no family has agreed to share anonymised replies for learning (dashboard → Wellbeing → "Help make Saheli better" checkbox)")
    if len(good) < MIN_SFT:
        missing.append(f"{MIN_SFT - len(good)} more good replies (have {len(good)})")
    thin = [s for s in MAIN_SITUATIONS if by_sit[s] < MIN_PER_SITUATION]
    if thin:
        missing.append("more examples in: " + ", ".join(f"{s} ({by_sit[s]}/{MIN_PER_SITUATION})" for s in thin))
    if pairs < MIN_PAIRS:
        missing.append(f"{MIN_PAIRS - pairs} more best-vs-worst pairs (have {pairs})")
    return {"ready": not missing, "missing": missing, "goodReplies": len(good), "corpus": len(rows), "pairs": pairs,
            "bySituation": dict(by_sit), "byLanguage": dict(by_lang), "families": len(fams), "consented": consented,
            "correctionCases": len(corr),
            "plan": "when ready: eval/export_tuning.py → eval/train.sh tune (Vertex supervised tuning of a Flash model in the "
                    "training project) → eval/compare_models.py on the holdout → 10% trial → live only if it wins"}
