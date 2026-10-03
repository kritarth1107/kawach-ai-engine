"""Log every message Saheli sends, then score it from what happened next (no model, no one asked).

Signals (weights are a first guess; the weekly report shows how each one moves):
- they replied (+), quickly (+), warmly (+); or were annoyed (−−) or corrected her (−−)
- a nudge about a medicine was followed by the dose within 90 minutes (+) or not (−)
- the daily grader passed (+) or failed (−−) it
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta

from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.models import CareEvent, Turn
from app.core import clock
from app.learn import situations as sit
from app.learn.anonymise import anonymise, leaks
from app.learn.models import LearningExample, ReplyLog

logger = logging.getLogger(__name__)

SCORE_AFTER = timedelta(hours=6)  # wait this long for a reaction
REPLY_WINDOW = timedelta(hours=6)
DOSE_WINDOW = timedelta(minutes=90)
KEEP_RAW_DAYS = 90  # raw reply_log rows (with the person's words) are deleted after this

WARM = re.compile(r"\b(thank|thanks|shukriya|dhanyavad|dhanyawad|bahut (accha|achha|badiya)|great|lovely|wah|accha laga|sweet|love you|good)\b|😊|🙏|❤|🥰|👍|धन्यवाद|शुक्रिया", re.I)
CORRECTED = re.compile(
    r"\b(galat|wrong|not right|incorrect|nahi kaha|i didn'?t say|i never said|already told|pehle (hi )?bata|kitni baar|that'?s not|not what i|"
    r"maine aisa nahi|aisa nahi|you said .{0,20} but|no,? (it'?s|its|the)|nahi,? (wo|woh|yeh|ye))\b",
    re.I,
)
WEIGHTS = {"replied_proactive": 0.3, "replied_reply": 0.15, "ignored_proactive": -0.2, "fast": 0.1, "warm": 0.25, "annoyed": -0.6,
           "corrected": -0.5, "dose_yes": 0.4, "dose_no": -0.2, "judge_pass": 0.2, "judge_fail": -0.8}
NO_REPLY_EXPECTED = {"thanks", "emergency", "dose_report", "vital_report"}


async def record(session: AsyncSession, *, family_id: str, thread_id: str, turn_id: int | None, kind: str, situation: str,
                 speaker_role: str, lang: str, user_text: str, text: str, tools: list[str], playbook_version: int = 0, arm: str = "live",
                 trace: list | None = None) -> None:
    """Called for every message Saheli sends (reply or proactive). Cheap: one insert, no reads."""
    if not text or text.strip().lower() == "none":
        return
    now = clock.now()
    session.add(ReplyLog(
        family_id=family_id, thread_id=thread_id, turn_id=turn_id, at=now, kind=kind, situation=situation, speaker_role=speaker_role,
        lang=lang, user_text=(user_text or "")[:1000], text=text[:2000], text_len=len(text), tools=list(tools or [])[:12],
        playbook_version=playbook_version, arm=arm, sent_hour=clock.ist(now).hour, trace=trace,
    ))


def tone_of(text: str) -> str:
    if sit.ANNOYED.search(text or ""):
        return "annoyed"
    if WARM.search(text or ""):
        return "warm"
    return "neutral"


def score_of(r: ReplyLog) -> float:
    s = 0.0
    expect = r.kind in ("proactive", "relay") or r.situation not in NO_REPLY_EXPECTED
    if r.replied:
        s += WEIGHTS["replied_proactive"] if r.kind != "reply" else WEIGHTS["replied_reply"]
        if r.reply_delay_s is not None and r.reply_delay_s <= 600:
            s += WEIGHTS["fast"]
    elif r.replied is False and r.kind != "reply" and expect:
        s += WEIGHTS["ignored_proactive"]
    if r.tone == "warm":
        s += WEIGHTS["warm"]
    elif r.tone == "annoyed":
        s += WEIGHTS["annoyed"]
    if r.corrected:
        s += WEIGHTS["corrected"]
    if r.dose_followed is True:
        s += WEIGHTS["dose_yes"]
    elif r.dose_followed is False:
        s += WEIGHTS["dose_no"]
    if r.judge_pass is True:
        s += WEIGHTS["judge_pass"]
    elif r.judge_pass is False:
        s += WEIGHTS["judge_fail"]
    return max(-1.0, min(1.0, round(s, 3)))


async def score_pending(sessions: async_sessionmaker, *, limit: int = 3000) -> int:
    """Score messages old enough to have had a reaction."""
    cutoff = clock.now() - SCORE_AFTER
    done = 0
    async with sessions() as session:
        rows = list((await session.execute(
            select(ReplyLog).where(ReplyLog.scored_at.is_(None), ReplyLog.at <= cutoff).order_by(ReplyLog.at).limit(limit)
        )).scalars())
        for r in rows:
            nxt = (await session.execute(
                select(Turn).where(Turn.family_id == r.family_id, Turn.thread_id == r.thread_id, Turn.role == "user",
                                   Turn.at > r.at, Turn.at <= r.at + REPLY_WINDOW).order_by(Turn.at).limit(1)
            )).scalar_one_or_none()
            r.replied = nxt is not None
            if nxt:
                r.reply_delay_s = int((nxt.at - r.at).total_seconds())
                r.tone = tone_of(nxt.text)
                r.corrected = bool(CORRECTED.search(nxt.text or ""))
            if r.kind != "reply" and sit.MED_WORDS.search(r.text or ""):
                took = (await session.execute(
                    select(CareEvent.id).where(and_(CareEvent.family_id == r.family_id, CareEvent.kind == "dose_taken",
                                                    CareEvent.at > r.at, CareEvent.at <= r.at + DOSE_WINDOW)).limit(1)
                )).first()
                r.dose_followed = took is not None
            r.score = score_of(r)
            r.scored_at = clock.now()
            done += 1
        await session.commit()
    return done


# ── the shared corpus ──────────────────────────────────────────────────────────


async def _identity_words(session: AsyncSession, family_id: str) -> tuple[list[str], list[str]]:
    """Names (household and anyone in the care record) and medicine names for this family, to scrub."""
    roster = await store.roster(session, family_id)
    names = []
    if roster:
        names += [(roster.elder or {}).get("name", "")] + [m.get("name", "") for m in roster.members or []]
    from app.care.models import CareFact

    facts = (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.status.in_(("active", "pending"))))).scalars()
    meds = []
    for f in facts:
        v = f.value or {}
        if f.domain in ("doctor", "contact", "home", "family") and v.get("name"):
            names.append(str(v["name"]))
        if f.domain == "medicine":
            meds.append(str(v.get("name") or f.key.split(":", 1)[1]))
    return [n for n in names if n], meds


async def build_corpus(sessions: async_sessionmaker, *, days: int = 3, limit: int = 3000) -> dict:
    """Copy scored messages from families that agreed into the shared, anonymised corpus."""
    from app.care import outcomes

    since = clock.now() - timedelta(days=days)
    out = {"added": 0, "skipped_no_consent": 0, "skipped_leak": 0}
    async with sessions() as session:
        rows = list((await session.execute(
            select(ReplyLog).where(ReplyLog.scored_at.is_not(None), ReplyLog.in_corpus.is_(False), ReplyLog.at >= since,
                                   ~ReplyLog.family_id.startswith("shadow:")).order_by(ReplyLog.id).limit(limit)
        )).scalars())
        consent: dict[str, bool] = {}
        words: dict[str, tuple] = {}
        for r in rows:
            if r.family_id not in consent:
                roster = await store.roster(session, r.family_id)
                elder = (roster.elder or {}).get("id") if roster else None
                consent[r.family_id] = bool(elder and (await outcomes.consent(session, r.family_id, elder))["granted"])
            if not consent[r.family_id]:
                out["skipped_no_consent"] += 1
                continue
            if r.family_id not in words:
                words[r.family_id] = await _identity_words(session, r.family_id)
            names, meds = words[r.family_id]
            ctx, reply = anonymise(r.user_text, names=names, medicines=meds), anonymise(r.text, names=names, medicines=meds)
            if leaks(ctx, names=names) or leaks(reply, names=names):
                out["skipped_leak"] += 1
                r.in_corpus = True  # never retried
                continue
            session.add(LearningExample(
                reply_id=r.id, situation=r.situation, kind=r.kind, speaker_role=r.speaker_role, lang=r.lang, context=ctx, reply=reply,
                score=r.score or 0.0, at=r.at,
                signals={"replied": r.replied, "delay_s": r.reply_delay_s, "tone": r.tone, "corrected": r.corrected,
                         "dose_followed": r.dose_followed, "judge_pass": r.judge_pass, "playbook": r.playbook_version, "arm": r.arm},
            ))
            r.in_corpus = True
            out["added"] += 1
        await session.commit()
    return out


async def prune(sessions: async_sessionmaker) -> int:
    """Raw reply_log rows hold the person's words; keep them only as long as they are useful."""
    async with sessions() as session:
        res = await session.execute(delete(ReplyLog).where(ReplyLog.at < clock.now() - timedelta(days=KEEP_RAW_DAYS)))
        await session.commit()
        return res.rowcount or 0
