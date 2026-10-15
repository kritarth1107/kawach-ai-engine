"""What Saheli started today, for one person: the nudge ledger, the WhatsApp window, and open follow-ups.

The window is the engine's own rule (closed when the gate says so, there is no last message, or that message is
older than 23 hours 50 minutes), except an answer today always leaves it open. A reopen template is mentioned
only when a ledger row says one was sent (reason waiting:update or waiting:opener).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import NudgeLedger, OpenLoop, Turn
from app.care.nudges import WINDOW, _iso
from app.core import clock

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def window_open(*, gate_open: bool | None, last_in: datetime | None, answered_today: bool, now: datetime) -> bool:
    """An answer today opens the window. Otherwise follow the nudge clock."""
    if answered_today:
        return True
    if last_in is None or now - last_in >= WINDOW:
        return False
    if gate_open is False:
        return False
    return True


def opener(rows: list[dict]) -> dict | None:
    """The reopen template, only when a row records that one was sent."""
    for row in rows:
        if str(row.get("reason") or "").startswith("waiting:"):
            return {"at": row.get("at"), "reason": row.get("reason") or "waiting:opener", "text": row.get("text") or ""}
    return None


def _bounds(day: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(f"{day}T00:00:00").replace(tzinfo=clock.IST)
    return start, start + timedelta(days=1)


def _aware(at: datetime | None) -> datetime | None:
    if at is None:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def _parsed(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return _aware(datetime.fromisoformat(raw))
    except ValueError:
        return None


def _on(at: datetime | None, start: datetime, end: datetime) -> bool:
    at = _aware(at)
    return at is not None and start <= at.astimezone(clock.IST) < end


def _clip(text: str | None) -> str:
    return " ".join((text or "").split())[:240]


async def _gate(host, family_id: str, person: str) -> dict:
    if host is None:
        return {}
    try:
        out = await asyncio.wait_for(
            host.call("nudge_gate", {"to": person}, family_id=family_id, subject_id=person, actor_id=person),
            timeout=4,
        )
        return out if isinstance(out, dict) else {}
    except Exception:  # noqa: BLE001 — the page still has the ledger and the turns
        return {}


async def view(session: AsyncSession, host, family_id: str, person: str, day: str | None = None) -> dict:
    now = clock.now()
    day = day if day and _DAY.fullmatch(day) else clock.ist_day(now)
    start, end = _bounds(day)
    gate = await _gate(host, family_id, person)
    last_user = await session.scalar(
        select(Turn.at).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user").order_by(Turn.at.desc()).limit(1)
    )
    turns = list((await session.execute(
        select(Turn).where(Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user", Turn.at >= start - timedelta(days=2))
        .order_by(Turn.at)
    )).scalars())
    ledger = list((await session.execute(
        select(NudgeLedger).where(
            NudgeLedger.family_id == family_id, NudgeLedger.person_id == person, NudgeLedger.status != "shadow",
            or_(
                and_(NudgeLedger.sent_at >= start, NudgeLedger.sent_at < end),
                and_(NudgeLedger.planned_at >= start, NudgeLedger.planned_at < end),
                and_(NudgeLedger.replied_at >= start, NudgeLedger.replied_at < end),
            ),
        ).order_by(NudgeLedger.planned_at)
    )).scalars())
    last_in = max([t for t in (_aware(last_user), _iso(gate.get("last_inbound_at"))) if t], default=None)
    started: list[dict] = []
    for row in ledger:
        if row.status == "shadow":
            continue
        at = row.sent_at or row.planned_at
        if not _on(at, start, end) and not _on(row.replied_at, start, end):
            continue
        started.append({
            "id": str(row.id),
            "kind": row.kind or "",
            "status": row.status or "",
            "reason": row.reason or "",
            "text": _clip(row.text),
            "at": at.isoformat() if at else "",
            "repliedAt": row.replied_at.isoformat() if row.replied_at else None,
            "reply": None,
            "_sent": row.sent_at,
        })
    for turn in turns:
        sent = _aware(turn.at)
        prior = [r for r in started if _aware(r["_sent"]) is not None and sent is not None and _aware(r["_sent"]) <= sent]
        if not prior or prior[-1]["reply"]:
            continue
        prior[-1]["reply"] = _clip(turn.text)
        if not prior[-1]["repliedAt"]:
            prior[-1]["repliedAt"] = turn.at.isoformat()
    answered = any(_on(t.at, start, end) for t in turns) or any(
        _on(_parsed(r.get("repliedAt")), start, end) for r in started
    )
    gate_open = gate.get("window_open") if "window_open" in gate else None
    open_ = window_open(gate_open=gate_open if isinstance(gate_open, bool) else None, last_in=last_in, answered_today=answered, now=now)
    for row in started:
        row.pop("_sent", None)
    loops = list((await session.execute(
        select(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.subject_id == person, OpenLoop.status == "open")
        .order_by(OpenLoop.created_at)
    )).scalars())
    return {
        "day": day,
        "personId": person,
        "window": {"open": open_, "lastInboundAt": last_in.isoformat() if last_in else None, "answeredToday": answered},
        "template": None if open_ else opener(started),
        "started": started,
        "loops": [
            {"id": str(o.id), "kind": o.kind or "", "title": _clip(o.title) or "Something still open",
             "wakeAt": o.wake_at.isoformat() if o.wake_at else None, "since": o.created_at.isoformat()}
            for o in loops
        ],
    }
