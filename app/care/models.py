"""Care Memory tables.

Keyed by Kavach ids (family_id, subject_id = the backend userId of the elder or caregiver),
so the backend and the brain share one identity without the ai-tenant mapping.

- care_facts: the care record. Structured, never compressed, never deleted. A change
  supersedes the old row, so "what was the dose before 3 Sep" always has an answer.
- care_events: append-only ledger of everything that happened (reminders sent, doses,
  vitals, alerts, messages). "Why was there no reminder" is answered from here.
- memory_notes: narrative markdown per person and per family (stories, dishes, people).
- care_links: how record items connect (a medicine is for a condition, prescribed by a doctor, from a health record).
- open_loops: what is unfinished: questions waiting for an answer, follow-ups, tasks.
- turns / thread_summaries: working context of each conversation thread.

Ledger events, memory versions, open loops and tasks carry the trace id of the work that wrote them (app.core.trace).
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
    trace_id: Mapped[str | None] = mapped_column(String(160), nullable=True)  # index: app.db.migrate (added after the table)

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
    trace_id: Mapped[str | None] = mapped_column(String(160), nullable=True)

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


class CareLink(Base):
    """How two things in one person's care record connect (D-13): a medicine is *for* a condition, *prescribed by* a
    doctor, *from* a health record; an appointment is *with* a doctor; two people are related.

    Links point at fact keys (medicine:metformin, condition:diabetes, doctor:dr_sharma), at a household member
    (person:<id>) or at a health record (record:<document id>), never at fact rows, so a dose change keeps them. Like
    facts they are never deleted: a link ends (valid_to, with why in note), a weaker source's health link waits as
    pending for a caregiver, and every change is a memory version (undo, restore)."""

    __tablename__ = "care_links"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))
    from_ref: Mapped[str] = mapped_column(String(200))
    relation: Mapped[str] = mapped_column(String(24))  # for | prescribed_by | from_record | with_doctor | related_to
    to_ref: Mapped[str] = mapped_column(String(200))
    label: Mapped[str] = mapped_column(String(200), default="")  # how the other end reads: "diabetes", "Dr Sharma"
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)  # e.g. {"as": "son"}, {"date": "2026-09-03"}
    # active | pending | ended | rejected
    status: Mapped[str] = mapped_column(String(16), default="active")
    source_kind: Mapped[str] = mapped_column(String(24))
    stated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(160), nullable=True)

    __table_args__ = (
        Index("ix_care_links_from", "family_id", "subject_id", "from_ref"),
        Index("ix_care_links_to", "family_id", "subject_id", "to_ref"),
        Index("uq_care_links_active", "family_id", "subject_id", "from_ref", "relation", "to_ref", unique=True,
              postgresql_where=sql("status = 'active'")),
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
        Index("ix_turns_thread_id", "family_id", "thread_id", "id"),  # recent_turns orders by id
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


from app.care import skillbook as _skillbook  # noqa: E402,F401  (registers the skills table)
from app.care import versions as _versions  # noqa: E402,F401  (registers memory_versions)


class NudgeLedger(Base):
    """Every regular nudge and reminder follow-up (Kavach NUdge's plan §3.1): planned, sent, skipped and why, shadow, the
    reply. One row per person per planned minute; the ref is never reused, so a retry never collides with a turn."""

    __tablename__ = "nudge_ledger"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64), index=True)
    person_id: Mapped[str] = mapped_column(String(64))
    # follow_up | upcoming | collect | habit | reflect | know | delight | early_bird | reminder_followup | doses
    kind: Mapped[str] = mapped_column(String(32))
    slot_key: Mapped[str] = mapped_column(String(120))  # person:YYYY-MM-DDTHH:MM (the minute it was planned)
    status: Mapped[str] = mapped_column(String(16))  # planned | sent | skipped | failed | shadow
    reason: Mapped[str] = mapped_column(String(160), default="")  # talking, busy, asleep, window_closed, ladder, guard:<x>…
    ref: Mapped[str] = mapped_column(String(160))  # nudge:<id>
    text: Mapped[str] = mapped_column(Text, default="")
    field: Mapped[str | None] = mapped_column(String(160), nullable=True)  # the data asked for (weight, activity:gym:plan…)
    about_ref: Mapped[str | None] = mapped_column(String(200), nullable=True)  # the reminder, activity, loop or fact it is about
    planned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    replied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    unanswered_run: Mapped[int] = mapped_column(Integer, default=0)  # the unanswered ladder's position when it went
    cost_inr: Mapped[float | None] = mapped_column(Float, nullable=True)
    # What Saheli learns from (self-learning, 2026-10-13): what the writer saw, how they took it, the review grade
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    reaction: Mapped[str | None] = mapped_column(String(16), nullable=True)  # welcome | neutral | unwelcome | fewer | stop
    grade: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # {"pass", "issues", "note", "at"} (the hourly review)

    __table_args__ = (
        Index("uq_nudge_ledger_slot", "family_id", "person_id", "slot_key", unique=True),
        Index("ix_nudge_ledger_person_sent", "family_id", "person_id", "sent_at"),
    )
