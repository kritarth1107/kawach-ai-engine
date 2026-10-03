"""Fine-tuning data for Saheli's brain (Gemini on Vertex AI), with her tool calls kept so tuning teaches style
without breaking how she uses tools.

Each example is one turn as Saheli handled it: what the person said, every tool call with its (trimmed) result,
and the final reply. Only high-scoring turns that passed the grader, from families that agreed to share, all
anonymised. Format: Vertex/Gemini "contents" with functionCall / functionResponse parts; the tool declarations are
attached so the model learns to call the same tools. (Check the current Vertex dataset spec before uploading:
see eval/tune.py and the Tuning Engineer's plan.)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.learn.anonymise import anonymise, leaks, looks_like_junk
from app.learn.models import ReplyLog

SYSTEM = ("You are Saheli, a warm WhatsApp care companion for elderly people in India and their families. Reply in the "
          "person's own language and script, briefly and kindly, and use your tools to remember, log, remind and alert.")


@dataclass
class Split:
    train: list[dict]
    validation: list[dict]
    holdout: list[dict]


def _anon_obj(x, names, meds):
    if isinstance(x, str):
        return anonymise(x, names=names, medicines=meds)
    if isinstance(x, list):
        return [_anon_obj(v, names, meds) for v in x]
    if isinstance(x, dict):
        return {k: _anon_obj(v, names, meds) for k, v in x.items()}
    return x


def to_example(trace: list[dict], *, situation: str, names: list[str], meds: list[str]) -> dict | None:
    """One Vertex SFT example from a turn trace, anonymised; None if anything identifying survives."""
    contents: list[dict] = []
    if any(looks_like_junk(s.get("text") or "") for s in trace):
        return None
    for step in trace:
        if step["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": anonymise(step["text"], names=names, medicines=meds)}]})
        elif step["role"] == "model" and step.get("calls"):
            parts = [{"functionCall": {"name": c["name"], "args": _anon_obj(c.get("args") or {}, names, meds)}} for c in step["calls"]]
            contents.append({"role": "model", "parts": parts})
        elif step["role"] == "tool":
            parts = []
            for r in step.get("results") or []:
                content = r.get("content") or ""
                try:
                    body = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    body = {"result": content}
                parts.append({"functionResponse": {"name": r.get("name"), "response": _anon_obj(body if isinstance(body, dict) else {"result": body}, names, meds)}})
            if parts:
                contents.append({"role": "user", "parts": parts})
        elif step["role"] == "model" and step.get("text"):
            contents.append({"role": "model", "parts": [{"text": anonymise(step["text"], names=names, medicines=meds)}]})
    if not contents or contents[-1]["role"] != "model" or "text" not in contents[-1]["parts"][0]:
        return None
    flat = json.dumps(contents, ensure_ascii=False)
    if leaks(flat, names=names, medicines=meds):
        return None
    return {"systemInstruction": {"role": "system", "parts": [{"text": f"{SYSTEM} Situation: {situation}."}]}, "contents": contents}


def tool_declarations() -> list[dict]:
    from app.brain import tools

    return [{"functionDeclarations": [{"name": s.name, "description": s.description[:1000], "parameters": s.schema} for s in tools.specs()]}]


async def build(session: AsyncSession, *, min_score: float = 0.4, limit: int = 20000, seed: int = 7, with_tools: bool = True) -> tuple[Split, dict]:
    """Collect, anonymise and split. Holdout is by family (never trained on), validation by turn."""
    from app.care import outcomes, store
    from app.learn.scoring import _identity_words

    rows = list((await session.execute(
        select(ReplyLog).where(ReplyLog.trace.is_not(None), ReplyLog.score >= min_score, ReplyLog.judge_pass.is_not(False),
                               ~ReplyLog.family_id.startswith("shadow:")).order_by(ReplyLog.id.desc()).limit(limit)
    )).scalars())
    consent: dict[str, bool] = {}
    words: dict[str, tuple] = {}
    by_family: dict[str, list[dict]] = {}
    stats = {"rows": len(rows), "no_consent": 0, "dropped_leak": 0, "examples": 0}
    decl = tool_declarations() if with_tools else None
    for r in rows:
        if r.family_id not in consent:
            roster = await store.roster(session, r.family_id)
            elder = (roster.elder or {}).get("id") if roster else None
            consent[r.family_id] = bool(elder and (await outcomes.consent(session, r.family_id, elder))["granted"])
        if not consent[r.family_id]:
            stats["no_consent"] += 1
            continue
        if r.family_id not in words:
            words[r.family_id] = await _identity_words(session, r.family_id)
        names, meds = words[r.family_id]
        ex = to_example(r.trace, situation=r.situation, names=names, meds=meds)
        if not ex:
            stats["dropped_leak"] += 1
            continue
        if decl:
            ex["tools"] = decl
        by_family.setdefault(r.family_id, []).append(ex)
        stats["examples"] += 1
    fams = sorted(by_family)
    rnd = random.Random(seed)
    rnd.shuffle(fams)
    n_hold = max(1, len(fams) // 5) if len(fams) >= 5 else 0
    hold = [e for f in fams[:n_hold] for e in by_family[f]]
    rest = [e for f in fams[n_hold:] for e in by_family[f]]
    rnd.shuffle(rest)
    n_val = max(1, len(rest) // 10) if len(rest) >= 20 else 0
    stats.update({"families": len(fams), "holdout_families": n_hold})
    return Split(train=rest[n_val:], validation=rest[:n_val], holdout=hold), stats
