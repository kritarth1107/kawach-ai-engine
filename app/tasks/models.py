"""Durable tasks: an order or a ride the brain asked for, carried through to done, failed or cancelled."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy import text as sql
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Task(Base):
    __tablename__ = "care_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(80))
    subject_id: Mapped[str] = mapped_column(String(64))  # the care recipient it is for
    requested_by: Mapped[str] = mapped_column(String(64))  # who asked; updates go to them
    service: Mapped[str] = mapped_column(String(24))
    kind: Mapped[str] = mapped_column(String(16))  # order | ride
    goal: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSONB, default=dict)  # items, pickup, drop, notes, limits
    # queued → running → (needs_input | awaiting_confirm) → running → done | failed | cancelled
    status: Mapped[str] = mapped_column(String(20), default="queued")
    # prepare | otp | place | cancel
    phase: Mapped[str] = mapped_column(String(12), default="prepare")
    input_needed: Mapped[str | None] = mapped_column(String(24), nullable=True)  # otp | confirm | fee | choice
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    agent_session: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_task: Mapped[str | None] = mapped_column(String(64), nullable=True)
    live_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)  # latest agent report
    history: Mapped[list] = mapped_column(JSONB, default=list)  # [{at, phase, status, note}]
    runs: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_care_tasks_family_live", "family_id", "status"),
        Index("ix_care_tasks_active", "status", postgresql_where=sql("status IN ('queued', 'running')")),
    )


class SkillNote(Base):
    """What agents learned about a service, fed to the next run (newest first)."""

    __tablename__ = "task_skill_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    service: Mapped[str] = mapped_column(String(24), index=True)
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
