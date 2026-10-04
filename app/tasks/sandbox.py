"""Each Browser Use task as a clean, isolated, logged sandbox.

- Profiles: one per family per service (the backend makes them). The engine also refuses to use a profile it has
  seen bound to another family, so a login can never cross families even if a lookup goes wrong.
- Login state per family + service, learned from run reports only (no paid probing).
- A ledger of every cloud browser the engine started, so a session lost on any exit path is still stopped
  by the sweeper (a finished run keeps its browser running, and billing).
- A per-task audit: each run's phase, result, steps, domains visited and browser cost.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

from sqlalchemy import DateTime, Index, String, Text, select
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from app.core import clock
from app.db.session import Base

logger = logging.getLogger(__name__)

SWEEP_AFTER = timedelta(minutes=20)
MAX_AUDIT = 12


class BrowserSession(Base):
    __tablename__ = "browser_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64))
    family_id: Mapped[str] = mapped_column(String(80))
    service: Mapped[str] = mapped_column(String(24))
    profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stop_reason: Mapped[str] = mapped_column(String(40), default="")

    __table_args__ = (Index("ix_browser_sessions_open", "started_at", postgresql_where=sql("stopped_at IS NULL")),)


class ServiceLogin(Base):
    __tablename__ = "service_logins"

    family_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    service: Mapped[str] = mapped_column(String(24), primary_key=True)
    profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(10), default="unknown")  # ok | expired | unknown
    last_login_ok_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_problem: Mapped[str] = mapped_column(Text, default="")


class ProfileConflict(Exception):
    pass


def _fam(family_id: str) -> str:
    return family_id.removeprefix("shadow:")


async def bind_profile(session: AsyncSession, family_id: str, service: str, profile_id: str | None) -> None:
    """Remember which family a profile belongs to; refuse a profile already bound to another family or service."""
    if not profile_id:
        return
    fam = _fam(family_id)
    other = (await session.execute(select(ServiceLogin).where(ServiceLogin.profile_id == profile_id))).scalars().all()
    if any(o.family_id != fam or o.service != service for o in other):
        logger.error("profile %s offered for %s/%s but bound to %s", profile_id, fam, service, [(o.family_id, o.service) for o in other])
        raise ProfileConflict("this browser login belongs to another family or store")
    row = await session.get(ServiceLogin, (fam, service))
    if row is None:
        session.add(ServiceLogin(family_id=fam, service=service, profile_id=profile_id, state="unknown"))
    elif row.profile_id != profile_id:
        row.profile_id, row.state = profile_id, "unknown"  # a new profile: login unknown until a run reports


async def started(session: AsyncSession, *, session_id: str, task_id: str, family_id: str, service: str, profile_id: str | None) -> None:
    if not session_id or await session.get(BrowserSession, session_id):
        return
    session.add(BrowserSession(session_id=session_id, task_id=task_id, family_id=_fam(family_id), service=service, profile_id=profile_id,
                               started_at=clock.now()))


async def stopped(session: AsyncSession, session_id: str | None, reason: str) -> None:
    if not session_id:
        return
    row = await session.get(BrowserSession, session_id)
    if row and row.stopped_at is None:
        row.stopped_at, row.stop_reason = clock.now(), reason[:40]


async def note_login(session: AsyncSession, family_id: str, service: str, out: dict) -> None:
    """From a run's own report: logged in → ok; asked for an OTP / not logged in → expired."""
    if not out:
        return
    fam = _fam(family_id)
    row = await session.get(ServiceLogin, (fam, service))
    if row is None:
        row = ServiceLogin(family_id=fam, service=service, state="unknown")
        session.add(row)
    now = clock.now()
    row.last_seen_at = now
    if out.get("needs_otp") or out.get("logged_in") is False:
        row.state = "expired"
        row.last_problem = "asked for a login code" if out.get("needs_otp") else "not logged in"
    elif out.get("logged_in") is True or out.get("placed") or out.get("booked"):
        row.state, row.last_login_ok_at, row.last_problem = "ok", now, ""


def audit(task, *, phase: str, status: str, steps: int, path: list[str], cost: float, error: str | None = None) -> None:
    domains = list(dict.fromkeys(h for h in (urlparse(u).hostname for u in path or []) if h))[:8]
    entry = {"at": clock.now().isoformat(), "phase": phase, "status": status, "steps": int(steps), "domains": domains, "cost_inr": round(cost, 2)}
    if error:
        entry["error"] = str(error)[:160]
    task.details = {**(task.details or {}), "audit": ((task.details or {}).get("audit") or [])[-(MAX_AUDIT - 1):] + [entry]}


def audit_text(task) -> list[str]:
    """Plain words for the Live tasks page."""
    out = []
    for a in (task.details or {}).get("audit") or []:
        where = ", ".join(a.get("domains") or []) or "no pages"
        what = {"finished": "finished", "failed": "failed", "stopped": "was stopped", "timeout": "took too long"}.get(a.get("status"), a.get("status"))
        out.append(f"{a.get('phase', '').capitalize()} run {what} after {a.get('steps', 0)} steps on {where} (≈₹{a.get('cost_inr', 0):.0f})"
                   + (f": {a['error']}" if a.get("error") else ""))
    return out


async def sweep(sessions: async_sessionmaker, agent, *, older_than: timedelta = SWEEP_AFTER, limit: int = 50) -> dict:
    """Stop cloud browsers the engine started over 20 min ago that no live task is using any more."""
    from app.tasks.models import Task

    cutoff = clock.now() - older_than
    out = {"checked": 0, "stopped": 0, "failed": 0}
    async with sessions() as session:
        rows = list((await session.execute(select(BrowserSession).where(
            BrowserSession.stopped_at.is_(None), BrowserSession.started_at < cutoff).limit(limit))).scalars())
        for r in rows:
            out["checked"] += 1
            in_use = (await session.execute(select(Task.id).where(
                Task.agent_session == r.session_id, Task.status.in_(("queued", "running", "needs_input"))))).first()
            if in_use:
                continue
            try:
                await agent.stop_session(r.session_id)
            except Exception as exc:  # noqa: BLE001 — already gone counts as stopped; anything else retries next sweep
                if "404" not in str(exc):
                    out["failed"] += 1
                    continue
            r.stopped_at, r.stop_reason = clock.now(), "sweeper"
            out["stopped"] += 1
        await session.commit()
    if out["stopped"]:
        logger.warning("browser sweeper stopped %s orphaned sessions", out["stopped"])
    return out


async def logins(session: AsyncSession, family_id: str) -> list[dict]:
    rows = (await session.execute(select(ServiceLogin).where(ServiceLogin.family_id == _fam(family_id)))).scalars()
    return [{"service": r.service, "state": r.state, "lastLoginOkAt": r.last_login_ok_at.isoformat() if r.last_login_ok_at else None,
             "lastSeenAt": r.last_seen_at.isoformat() if r.last_seen_at else None, "problem": r.last_problem or None} for r in rows]
