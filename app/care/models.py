"""Care Memory tables.

Keyed by Kavach ids (family_id, subject_id = the backend userId of the elder or caregiver),
so the backend and the brain share one identity without the ai-tenant mapping.

- care_facts: the care record. Structured, never compressed, never deleted. A change
  supersedes the old row, so "what was the dose before 3 Sep" always has an answer.
- care_events: append-only ledger of everything that happened (reminders sent, doses,
  vitals, alerts, messages). "Why was there no reminder" is answered from here.
- memory_notes: narrative markdown per person and per family (stories, dishes, people).
- open_loops: what is unfinished: questions waiting for an answer, follow-ups, tasks.
- turns / thread_summaries: working context of each conversation thread.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy import text as sql
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class CareFact(Base):
    __tablename__ = "care_facts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64), index=True)
    subject_id: Mapped[str] = mapped_column(String(64))
    domain: Mapped[str] = mapped_column(String(32))
    key: Mapped[str] = mapped_column(String(160))
    value: Mapped[dict] = mapped_column(JSONB, default=dict)
    text: Mapped[str] = mapped_column(Text, default="")
    # active | pending | superseded | stopped | retracted
    status: Mapped[str] = mapped_column(String(16), default="active")
    # elder_said | caregiver_said | prescription | lab | dashboard | import | inferred
    source_kind: Mapped[str] = mapped_column(String(24))
    source_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    stated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    supersedes: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_care_facts_subject", "family_id", "subject_id", "domain"),
        Index("ix_care_facts_key", "family_id", "subject_id", "key"),
        # One live value per key. Pending rows wait beside the active one for confirmation.
        Index(
            "uq_care_facts_active_key",
            "family_id",
            "subject_id",
            "key",
            unique=True,
            postgresql_where=sql("status = 'active'"),
        ),
    )


class CareEvent(Base):
    __tablename__ = "care_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))
    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    kind: Mapped[str] = mapped_column(String(40))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    day: Mapped[str] = mapped_column(String(10))  # IST yyyy-mm-dd
    summary: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    tsv = mapped_column(TSVECTOR, nullable=True)

    __table_args__ = (
        Index("ix_care_events_subject_at", "family_id", "subject_id", "at"),
        Index("ix_care_events_day", "family_id", "subject_id", "day", "kind"),
        Index("ix_care_events_tsv", "tsv", postgresql_using="gin"),
        # Idempotent writes from webhook retries and scheduler re-runs.
        Index("uq_care_events_ref", "family_id", "kind", "ref", unique=True, postgresql_where=sql("ref IS NOT NULL")),
    )


class MemoryNote(Base):
    __tablename__ = "memory_notes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))  # a person, or "family"
    slug: Mapped[str] = mapped_column(String(160))
    title: Mapped[str] = mapped_column(String(255))
    body_md: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    tsv = mapped_column(TSVECTOR, nullable=True)

    __table_args__ = (
        Index("uq_memory_notes_slug", "family_id", "subject_id", "slug", unique=True),
        Index("ix_memory_notes_tsv", "tsv", postgresql_using="gin"),
    )


class OpenLoop(Base):
    __tablename__ = "open_loops"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))
    # question | followup | confirm_fact | task | watch
    kind: Mapped[str] = mapped_column(String(24))
    title: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    # open | done | cancelled | expired
    status: Mapped[str] = mapped_column(String(16), default="open")
    owner_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # who must answer
    wake_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    alert_rule: Mapped[str | None] = mapped_column(Text, nullable=True)
    dedupe_key: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_open_loops_live", "family_id", "subject_id", "status"),
        Index("ix_open_loops_wake", "status", "wake_at"),
        Index(
            "uq_open_loops_dedupe",
            "family_id",
            "dedupe_key",
            unique=True,
            postgresql_where=sql("status = 'open' AND dedupe_key IS NOT NULL"),
        ),
    )


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    family_id: Mapped[str] = mapped_column(String(64))
    thread_id: Mapped[str] = mapped_column(String(64))  # one per person talking to Saheli
    speaker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | system
    text: Mapped[str] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    message_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    extracted: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_turns_thread_at", "family_id", "thread_id", "at"),
        Index("uq_turns_ref", "family_id", "message_ref", unique=True, postgresql_where=sql("message_ref IS NOT NULL")),
    )


class ThreadSummary(Base):
    __tablename__ = "thread_summaries"

    family_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    covers_until_turn: Mapped[int] = mapped_column(BigInteger, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FamilyRoster(Base):
    """Last known household for a family, so scheduled wake-ups can run a turn without the backend."""

    __tablename__ = "family_rosters"

    family_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    elder: Mapped[dict] = mapped_column(JSONB, default=dict)
    members: Mapped[list] = mapped_column(JSONB, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
