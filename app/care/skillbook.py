"""Skills Saheli writes for herself (Hermes-style), kept outside the model.

Two kinds in one table:
- store skills: the path that worked on a store's website, shared by every family (the site is the same for all).
  Saved after a successful run, anonymised, and fed to the next agent on that store. A skill that keeps failing
  goes stale; unused ones fade (stale after 30 days, archived after 90).
- family skills: how a person likes things ("remind after puja, not before"; "short Hinglish, no emoji").
  A caregiver's skill is active at once; one the nightly dream proposes waits for a caregiver's yes.
  They shape tone and timing only: anything about medicines, doses, alerts, allergies, payment or privacy is refused,
  because those live in the care record and the fixed rules.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from sqlalchemy import DateTime, Index, Integer, String, Text, func, select
from sqlalchemy import text as sql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core import clock
from app.db.session import Base

MAX_BODY = 600
MAX_IN_CONTEXT = 5
CONTEXT_CHARS = 600
MAX_ACTIVE_PER_PERSON = 12
STORE_HINTS = 3
STALE_AFTER = timedelta(days=30)
ARCHIVE_AFTER = timedelta(days=90)
FAIL_STREAK_STALE = 2
STATUSES = ("proposed", "active", "stale", "archived", "blocked")


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(8))  # store | family
    family_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    service: Mapped[str] = mapped_column(String(24), default="")
    title: Mapped[str] = mapped_column(String(120))
    body: Mapped[str] = mapped_column(Text)
    steps: Mapped[list] = mapped_column(JSONB, default=list)
    source: Mapped[str] = mapped_column(String(12))  # auto | caregiver | dream
    evidence: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(10), default="active")
    uses: Mapped[int] = mapped_column(Integer, default=0)
    successes: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    fail_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[str] = mapped_column(String(64), default="")
    version: Mapped[int] = mapped_column(Integer, default=1)

    __table_args__ = (
        Index("ix_skills_family", "family_id", "subject_id", "status"),
        Index("ix_skills_store", "service", "status", postgresql_where=sql("scope = 'store'")),
    )


# ── what a family skill may say ──────────────────────────────────────────────

NOT_FOR_SKILLS = re.compile(
    r"\b(medicines?|medication|meds?|tablets?|pills?|insulin|dose|doses|dosage|dawa|dawai|goli|दवा|दवाई|गोली|"
    r"allerg\w*|emergenc\w*|alerts?|red ?flags?|ambulance|"
    r"pay(ment)?|cod|cash|upi|card|budget|price|order|"
    r"secret|hide|don'?t tell|never tell|private|privacy|password|otp|"
    r"reminder time|change (the )?reminder|stop (the )?reminders?|no reminders?|"
    r"ignore|override|rules?|instructions?|system prompt|pretend|act as)\b",
    re.I,
)


def problems(text: str) -> list[str]:
    from app.learn import lessons

    t = (text or "").strip()
    out = []
    if not t:
        return ["empty"]
    if len(t) > MAX_BODY:
        out.append(f"longer than {MAX_BODY} characters")
    for rx, why in ((lessons.FORBIDDEN, "touches a fixed rule or medical content"), (NOT_FOR_SKILLS, "belongs in the care record or the fixed rules")):
        m = rx.search(t)
        if m:
            out.append(f"{why} ({m.group(0)})")
    if re.search(r"\d", t):
        out.append("contains a number (times and amounts live in the care record)")
    if re.search(r"https?://|www\.", t, re.I):
        out.append("contains a link")
    return out


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def view(s: Skill) -> dict:
    rate = round(s.successes / s.uses, 2) if s.uses else None
    return {"id": s.id, "scope": s.scope, "subjectId": s.subject_id, "service": s.service, "title": s.title, "body": s.body,
            "steps": s.steps or [], "source": s.source, "status": s.status, "uses": s.uses, "successes": s.successes,
            "failures": s.failures, "successRate": rate, "lastUsedAt": s.last_used_at.isoformat() if s.last_used_at else None,
            "createdAt": s.created_at.isoformat(), "updatedAt": s.updated_at.isoformat(), "updatedBy": s.updated_by, "version": s.version}


# ── family skills ────────────────────────────────────────────────────────────

async def save_family(session: AsyncSession, family_id: str, subject_id: str, body: str, *, title: str = "", source: str = "caregiver",
                      by: str = "", evidence: list | None = None) -> dict:
    body = re.sub(r"\s+", " ", (body or "").strip())
    bad = problems(body)
    if bad:
        return {"saved": False, "problems": bad}
    now = clock.now()
    rows = list((await session.execute(select(Skill).where(
        Skill.scope == "family", Skill.family_id == family_id, Skill.subject_id == subject_id))).scalars())
    same = next((r for r in rows if _norm(r.body) == _norm(body)), None)
    if source == "dream":
        prior = next((r for r in rows if r.source == "dream" and r.title == title), None)
        if prior and prior.status in ("blocked", "active"):
            return {"saved": False, "id": prior.id, "status": prior.status, "problems": ["already decided"]}
        same = same or prior
    status = "active" if source == "caregiver" else "proposed"
    if same:
        if same.status == "blocked" and source != "caregiver":
            return {"saved": False, "id": same.id, "status": "blocked", "problems": ["a caregiver removed this before"]}
        same.body, same.title = body, title or same.title
        same.status = "active" if source == "caregiver" else (same.status if same.status == "active" else status)
        same.updated_at, same.updated_by, same.version = now, by, same.version + 1
        same.evidence = ((same.evidence or []) + (evidence or []))[-20:]
        s = same
    else:
        s = Skill(scope="family", family_id=family_id, subject_id=subject_id, service="", title=(title or body)[:120], body=body,
                  steps=[], source=source, evidence=(evidence or [])[-20:], status=status, created_at=now, updated_at=now, updated_by=by)
        session.add(s)
    await session.flush()
    await _cap_active(session, family_id, subject_id)
    return {"saved": True, "id": s.id, "status": s.status, "title": s.title}


async def _cap_active(session: AsyncSession, family_id: str, subject_id: str) -> None:
    rows = list((await session.execute(select(Skill).where(
        Skill.scope == "family", Skill.family_id == family_id, Skill.subject_id == subject_id, Skill.status == "active",
    ).order_by(Skill.updated_at.desc()))).scalars())
    for r in rows[MAX_ACTIVE_PER_PERSON:]:
        r.status = "archived"


async def family_skills(session: AsyncSession, family_id: str, subject_ids: list[str] | None = None,
                        statuses: tuple[str, ...] = ("proposed", "active", "stale")) -> list[Skill]:
    q = select(Skill).where(Skill.scope == "family", Skill.family_id == family_id, Skill.status.in_(statuses))
    if subject_ids:
        q = q.where(Skill.subject_id.in_(subject_ids))
    return list((await session.execute(q.order_by(Skill.status, Skill.updated_at.desc()))).scalars())


async def _family_row(session: AsyncSession, family_id: str, skill_id: int) -> Skill | None:
    s = await session.get(Skill, skill_id)
    return s if s and s.scope == "family" and s.family_id == family_id else None


async def decide(session: AsyncSession, family_id: str, skill_id: int, *, action: str, by: str, body: str | None = None) -> dict:
    """approve | edit | remove | restore, for one family skill."""
    s = await _family_row(session, family_id, skill_id)
    if not s:
        return {"ok": False, "error": "not found"}
    now = clock.now()
    if action == "edit":
        new = re.sub(r"\s+", " ", (body or "").strip())
        bad = problems(new)
        if bad:
            return {"ok": False, "problems": bad}
        s.body, s.status = new, "active"
    elif action == "approve":
        s.status = "active"
    elif action == "remove":
        s.status = "blocked"
    elif action == "restore":
        s.status = "active"
    else:
        return {"ok": False, "error": f"unknown action {action}"}
    s.updated_at, s.updated_by, s.version = now, by, s.version + 1
    if s.status == "active":
        await session.flush()
        await _cap_active(session, family_id, s.subject_id or "")
    return {"ok": True, "skill": view(s)}


async def forget_matching(session: AsyncSession, family_id: str, subject_ids: list[str], words: str, *, by: str) -> dict:
    ws = [w for w in re.findall(r"\w+", (words or "").lower()) if len(w) > 2]
    if not ws:
        return {"removed": []}
    gone = []
    for s in await family_skills(session, family_id, subject_ids):
        hay = f"{s.title} {s.body}".lower()
        if sum(w in hay for w in ws) >= max(1, (len(ws) + 1) // 2):
            s.status, s.updated_at, s.updated_by, s.version = "blocked", clock.now(), by, s.version + 1
            gone.append(s.title)
    return {"removed": gone}


async def context_block(session: AsyncSession, family_id: str, subject_id: str, name: str) -> str:
    """'HOW KAMLA LIKES THINGS' for the person in the conversation; placed after the fixed rules, never above them."""
    rows = [s for s in await family_skills(session, family_id, [subject_id], statuses=("active",)) if not problems(s.body)]
    lines, used = [], 0
    for s in rows[:MAX_IN_CONTEXT]:
        line = f"  - {s.body}"
        if used + len(line) > CONTEXT_CHARS:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return (f"HOW {(name or 'THEY').upper()} LIKES THINGS (from the family; tone and timing only — the fixed rules, the care record "
            "and safety always come first):\n" + "\n".join(lines))


async def propose_from_style(session: AsyncSession, family_id: str, subject_id: str, style: dict) -> dict | None:
    """The nightly dream suggests a reply-style skill once their reactions show what works; a caregiver approves it."""
    if not style or "good_reply_chars" not in style:
        return None
    words = max(8, round(style["good_reply_chars"] / 6))
    length = "Short replies work best with them" if words <= 15 else "They respond well to fuller, warmer replies" if words >= 35 else (
        "Medium-length replies work best with them")
    emoji = "a little emoji is fine" if style.get("emoji_rate", 0) >= 0.3 else "keep it plain, without emoji"
    out = await save_family(session, family_id, subject_id, f"{length}; {emoji}.", title="Reply style", source="dream", by="dream",
                            evidence=[{"style": {k: style.get(k) for k in ("their_words", "good_reply_chars", "bad_reply_chars", "emoji_rate")}}])
    return out if out.get("saved") else None


# ── store skills ─────────────────────────────────────────────────────────────

def _clean_steps(steps: list[str]) -> list[str]:
    from app.learn import anonymise

    out = []
    for st in steps[:10]:
        st = anonymise.anonymise(str(st)[:80])
        if anonymise.leaks(st) or re.search(r"\[(PERSON|NAME|PHONE|ADDRESS|PIN|ID|MED)", st):
            continue
        if re.search(r"\d{4,}", st):  # order ids, phones, OTPs, pincodes
            continue
        out.append(st)
    return out


async def record_store_success(session: AsyncSession, service: str, phase: str, steps: list[str], *, task_id: str, n_steps: int,
                               used: list[int] | None = None) -> Skill | None:
    """A run that worked: credit the skills it was given, and save its path as a skill (or reinforce the same path)."""
    await record_store_use(session, used or [], ok=True)
    clean = _clean_steps(steps)
    if len(clean) < 2:
        return None
    now = clock.now()
    body = " → ".join(clean)
    rows = list((await session.execute(select(Skill).where(Skill.scope == "store", Skill.service == service, Skill.title == f"{phase} path",
                                                           Skill.status.in_(("active", "stale", "proposed"))))).scalars())
    same = next((r for r in rows if r.body == body), None)
    ev = {"task": task_id, "steps": n_steps, "at": now.isoformat()}
    if same:
        same.evidence = ((same.evidence or []) + [ev])[-20:]
        if same.id not in (used or []):
            same.uses += 1
            same.successes += 1
        same.status, same.fail_streak, same.last_used_at, same.updated_at = "active", 0, now, now
        return same
    s = Skill(scope="store", family_id=None, subject_id=None, service=service, title=f"{phase} path", body=body, steps=clean,
              source="auto", evidence=[ev], status="active", uses=1, successes=1, last_used_at=now, created_at=now, updated_at=now, updated_by="auto")
    session.add(s)
    await session.flush()
    return s


async def record_store_use(session: AsyncSession, skill_ids: list[int], *, ok: bool) -> None:
    now = clock.now()
    for sid in dict.fromkeys(skill_ids or []):
        s = await session.get(Skill, int(sid))
        if not s or s.scope != "store":
            continue
        s.uses += 1
        s.last_used_at = now
        if ok:
            s.successes += 1
            s.fail_streak = 0
        else:
            s.failures += 1
            s.fail_streak += 1
            if s.fail_streak >= FAIL_STREAK_STALE and s.status == "active":
                s.status = "stale"
        s.updated_at = now


async def store_hints(session: AsyncSession, service: str, limit: int = STORE_HINTS) -> list[Skill]:
    """Top active store skills by (smoothed) success rate."""
    rows = list((await session.execute(select(Skill).where(Skill.scope == "store", Skill.service == service, Skill.status == "active"))).scalars())
    rows.sort(key=lambda s: ((s.successes + 1) / (s.uses + 2), s.successes, s.updated_at), reverse=True)
    return rows[:limit]


async def store_list(session: AsyncSession, service: str | None = None) -> list[dict]:
    q = select(Skill).where(Skill.scope == "store", Skill.status.in_(("active", "stale")))
    if service:
        q = q.where(Skill.service == service)
    return [view(s) for s in (await session.execute(q.order_by(Skill.service, Skill.successes.desc()))).scalars()]


# ── curator (nightly) ────────────────────────────────────────────────────────

async def curate(session: AsyncSession) -> dict:
    """Store skills unused 30 days → stale, stale 90 days → archived. Unanswered family proposals go after 30 days.
    Active family skills never fade: a caregiver set or approved them."""
    now = clock.now()
    last = func.coalesce(Skill.last_used_at, Skill.updated_at)
    staled = archived = 0
    for s in (await session.execute(select(Skill).where(Skill.scope == "store", Skill.status == "active", last < now - STALE_AFTER))).scalars():
        s.status, s.updated_at = "stale", now
        staled += 1
    for s in (await session.execute(select(Skill).where(Skill.scope == "store", Skill.status == "stale", last < now - ARCHIVE_AFTER))).scalars():
        s.status, s.updated_at = "archived", now
        archived += 1
    for s in (await session.execute(select(Skill).where(Skill.scope == "family", Skill.status == "proposed",
                                                        Skill.updated_at < now - STALE_AFTER))).scalars():
        s.status, s.updated_at = "archived", now
        archived += 1
    return {"stale": staled, "archived": archived}
