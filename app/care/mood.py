"""How the person seems, through the day (founder 2026-10-13: "moods in dashboard should be also changed regularly and
show properly"; Home showed "—" all day because its mood came only from the backend's nightly snapshot).

A mood is a care event of kind "mood" with a score (1 very low … 5 very good, the scale the backend's wellbeing days use)
and one word for the dashboard. Saheli logs it in the conversation (log_event); the 30-minute re-read fills in a
conversation where she did not. The same score again within MERGE_GAP is not logged twice; a change always is.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import CareEvent
from app.core import clock

SCALE = {1: "very low", 2: "low", 3: "okay", 4: "good", 5: "very good"}
MERGE_GAP = timedelta(minutes=45)
STALE = timedelta(hours=12)  # the re-read does not log a mood from a conversation older than this (a backlog read late)


def score_of(v) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n in SCALE else None


def _severity(score: int) -> str:
    return "concern" if score == 1 else "watch" if score == 2 else "info"


async def last(session: AsyncSession, family_id: str, subject_id: str, *, since=None) -> CareEvent | None:
    q = select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "mood")
    if since is not None:
        q = q.where(CareEvent.at >= since)
    return (await session.execute(q.order_by(CareEvent.at.desc(), CareEvent.id.desc()).limit(1))).scalar_one_or_none()


async def record(session: AsyncSession, *, family_id: str, subject_id: str, score: int | None, word: str, summary: str,
                 severity: str | None = None, actor_id: str | None = None, source: str = "chat", at=None) -> bool:
    """Log a mood (at: when they showed it, now by default). False when it repeats the last one (same score within MERGE_GAP)."""
    now = at or clock.now()
    prev = await last(session, family_id, subject_id, since=now - MERGE_GAP)
    if prev is not None and score is not None and score_of((prev.payload or {}).get("score")) == score:
        return False
    word = (word or "").strip().lower()[:24]
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="mood", summary=summary[:500],
        payload={"severity": severity or (_severity(score) if score else "info"), **({"score": score} if score else {}),
                 **({"word": word} if word else {}), "source": source},
        actor_id=actor_id, at=now,
    )
    return True


def as_json(e: CareEvent | None) -> dict | None:
    if e is None:
        return None
    p = e.payload or {}
    score = score_of(p.get("score"))
    return {"score": score, "word": p.get("word") or (SCALE[score] if score else None), "summary": e.summary,
            "at": e.at.isoformat(), "source": p.get("source") or "chat"}


async def now_for(session: AsyncSession, family_id: str, subject_id: str) -> dict | None:
    """The latest mood heard in the last 24 hours, for Home."""
    return as_json(await last(session, family_id, subject_id, since=clock.now() - timedelta(hours=24)))


# ── how they are now, from the whole picture (founder 2026-10-13) ──────────────────────────────────────────────────
# "how can I be relieved if I'm doing potty time to time and the stomach is unwell? Not getting fever doesn't mean
# relieved. Make the mood more accurate, understanding the feeling of the chat, not just guessing from the last message."
# Live: the re-read saw only the newest lines; "no fever" became "relieved" while loose motions went on, and "a little
# better" became "stomach issues resolved". Every mood is now written from the whole recent conversation and the health
# problems they reported.

ASSESS = """You decide how a person is feeling NOW, from their recent WhatsApp conversation with Saheli (a care companion)
and the health problems they reported. Read the whole picture, never just the last message.

Rules:
- An ongoing physical problem (upset stomach, loose motions, vomiting, pain, fever, dizziness, a cough, breathlessness, a
  wound, poor sleep) means they are not "relieved", "good" or "fine" — even when their last message denies a different
  symptom. "No fever" while the loose motions go on = still unwell (uncomfortable), not relieved.
- Better is not over: "a little better", "kam hua hai" = improving, still unwell (score 2-3); never "good" or "resolved".
- A problem is over only when they say so in their own words (it stopped, they feel fine now).
- Feelings count too (worried, irritated, lonely, bored, cheerful, calm), from how they write; never guess one they did
  not show. Testing or arguing with Saheli is irritation at Saheli, not their health.
- Nothing invented: every part of the summary comes from their words or the reported problems.

Return only JSON: {"score": 1-5 (1 very low, 2 low, 3 okay, 4 good, 5 very good), "word": "one English word for the
dashboard, e.g. unwell, uncomfortable, improving, tired, worried, irritated, lonely, calm, cheerful, relieved",
"summary": "one short line in English, third person, how they are and why, e.g. 'Still has loose motions since the
morning after street food; no fever.'", "confident": true or false}"""

LOOK_BACK = timedelta(hours=24)
PROBLEMS_BACK = timedelta(hours=48)


async def assess(session: AsyncSession, family_id: str, subject_id: str, *, name: str = "", hint: dict | None = None,
                 now=None) -> dict | None:
    """{score, word, summary} for how they are now, read by a model from their last day of conversation and the problems
    they reported in the last two days; None when nothing can be read (the caller keeps its own reading then)."""
    import json
    import re

    from app.llm import router

    now = now or clock.now()
    turns = [t for t in await store.recent_turns(session, family_id, subject_id, limit=40)
             if t.at and t.at >= now - LOOK_BACK and (t.text or "").strip() and (t.text or "").strip().lower() != "none"][-24:]
    problems = list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.subject_id == subject_id,
                                CareEvent.kind.in_(("symptom", "vital", "red_flag")), CareEvent.at >= now - PROBLEMS_BACK)
        .order_by(CareEvent.at)
    )).scalars())
    if not turns and not problems:
        return None
    lines = [f"PERSON: {name or 'them'}", f"NOW: {clock.ist(now).strftime('%a %d %b %H:%M')} IST",
             "CONVERSATION (oldest first):"]
    lines += [f"  {clock.ist(t.at).strftime('%d %b %H:%M')} {'them' if t.role == 'user' else 'Saheli'}: {t.text[:300]}" for t in turns] or ["  (none today)"]
    lines.append("HEALTH PROBLEMS THEY REPORTED (last 2 days, oldest first):")
    lines += [f"  {clock.ist(e.at).strftime('%d %b %H:%M')} {e.summary[:200]}" for e in problems] or ["  (none)"]
    if hint:
        lines.append(f"A FIRST READING (may be wrong; check it against everything above): {json.dumps(hint, ensure_ascii=False)[:300]}")
    try:
        reply = await router.complete("worker", system_stable=ASSESS, effort="low", max_tokens=800, timeout_s=30,
                                      messages=[{"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}]}])
    except Exception:  # noqa: BLE001 — the caller's own reading stands
        return None
    m = re.search(r"\{.*\}", reply.text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return None
    score = score_of(data.get("score"))
    word = str(data.get("word") or "").strip().lower()[:24]
    summary = str(data.get("summary") or "").strip()[:500]
    if not score or not word or not summary:
        return None
    return {"score": score, "word": word, "summary": summary, "confident": bool(data.get("confident", True))}
