"""Hand simulator roles (family members, the judge) to outside players: the Saheli Lab bots.

The simulator calls ask(); the request waits in lab_requests until a bot claims and answers it through the
bridge (eval/lab/bridge.py). Saheli herself is never played from here: her replies always come from her own brain.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, Integer, String, Text, select, update
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base, SessionLocal

CLAIM_TTL = timedelta(minutes=10)


class LabRequest(Base):
    __tablename__ = "lab_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16), index=True)  # sim | judge
    group: Mapped[str] = mapped_column(String(16), index=True)  # north | south_east | english | judge
    family: Mapped[str] = mapped_column(String(64), default="")
    person: Mapped[str] = mapped_column(String(64), default="")
    system: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(12), default="pending", index=True)  # pending | claimed | answered | expired
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def external_roles() -> set[str]:
    return {r.strip() for r in os.getenv("SIM_EXTERNAL_ROLES", "").split(",") if r.strip()}


def run_id() -> str:
    return os.getenv("SIM_RUN_ID", "default")


def now() -> datetime:
    return datetime.now(timezone.utc)  # real time: bots work in real time, not the simulator's clock


async def ask(role: str, system: str, prompt: str, *, group: str, family: str = "", person: str = "",
              timeout_s: float | None = None, poll_s: float = 2.0) -> str | None:
    """Queue a request and wait for a bot's answer. None on timeout (the caller decides what to do)."""
    timeout_s = timeout_s if timeout_s is not None else float(os.getenv("SIM_EXTERNAL_TIMEOUT", "1800"))
    async with SessionLocal() as s:
        r = LabRequest(run_id=run_id(), role=role, group=group, family=family, person=person, system=system, prompt=prompt,
                       status="pending", created_at=now())
        s.add(r)
        await s.commit()
        rid = r.id
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_s)
        async with SessionLocal() as s:
            row = await s.get(LabRequest, rid)
            if row and row.status == "answered":
                return row.answer or ""
    async with SessionLocal() as s:
        await s.execute(update(LabRequest).where(LabRequest.id == rid, LabRequest.status != "answered").values(status="expired"))
        await s.commit()
    return None


async def claim(group: str, player: str, *, limit: int = 10, role: str | None = None) -> list[LabRequest]:
    stale = now() - CLAIM_TTL
    async with SessionLocal() as s:
        q = select(LabRequest).where(LabRequest.group == group, (LabRequest.status == "pending") | (
            (LabRequest.status == "claimed") & (LabRequest.claimed_at < stale)))
        if role:
            q = q.where(LabRequest.role == role)
        rows = list((await s.execute(q.order_by(LabRequest.id).limit(limit).with_for_update(skip_locked=True))).scalars())
        for r in rows:
            r.status, r.claimed_by, r.claimed_at = "claimed", player, now()
        await s.commit()
        return rows


async def answer(answers: list[dict], player: str) -> dict:
    ok = skipped = 0
    async with SessionLocal() as s:
        for a in answers:
            row = await s.get(LabRequest, int(a["id"]))
            if not row or row.status not in ("claimed", "pending"):
                skipped += 1
                continue
            row.answer, row.status, row.answered_at, row.claimed_by = str(a.get("text", ""))[:8000], "answered", now(), player
            ok += 1
        await s.commit()
    return {"answered": ok, "skipped": skipped}
