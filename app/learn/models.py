"""Tables for the learning loop."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ReplyLog(Base):
    """One row per message Saheli sent, and (later) how it went."""

    __tablename__ = "reply_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    family_id: Mapped[str] = mapped_column(String(64))
    thread_id: Mapped[str] = mapped_column(String(64))  # the person it went to
    turn_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    kind: Mapped[str] = mapped_column(String(16))  # reply | proactive | relay
    situation: Mapped[str] = mapped_column(String(32))
    speaker_role: Mapped[str] = mapped_column(String(16), default="")  # elder | caregiver | system
    lang: Mapped[str] = mapped_column(String(16), default="")  # latin-indic | latin-english | devanagari | …
    user_text: Mapped[str] = mapped_column(Text, default="")  # what they said (raw; anonymised only when it enters the corpus)
    text: Mapped[str] = mapped_column(Text)
    text_len: Mapped[int] = mapped_column(Integer, default=0)
    tools: Mapped[list] = mapped_column(JSONB, default=list)
    playbook_version: Mapped[int] = mapped_column(Integer, default=0)
    arm: Mapped[str] = mapped_column(String(8), default="live")  # live | canary
    sent_hour: Mapped[int] = mapped_column(Integer, default=0)  # IST hour
    # outcome, filled by the scorer
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    replied: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reply_delay_s: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tone: Mapped[str | None] = mapped_column(String(12), nullable=True)  # warm | neutral | annoyed
    corrected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    dose_followed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    judge_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    judge_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    in_corpus: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_reply_log_unscored", "scored_at", "at"),
        Index("ix_reply_log_thread", "family_id", "thread_id", "at"),
        Index("ix_reply_log_version", "playbook_version", "arm", "at"),
    )


class LearningExample(Base):
    """An anonymised, scored example from a family that agreed to share (no names, numbers or places)."""

    __tablename__ = "learning_corpus"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reply_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    situation: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    speaker_role: Mapped[str] = mapped_column(String(16))
    lang: Mapped[str] = mapped_column(String(16))
    context: Mapped[str] = mapped_column(Text)  # what they said, anonymised
    reply: Mapped[str] = mapped_column(Text)  # Saheli's message, anonymised
    score: Mapped[float] = mapped_column(Float)
    signals: Mapped[dict] = mapped_column(JSONB, default=dict)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PlaybookVersion(Base):
    """Learned style: short lessons and example replies per situation. Never above the safety rules."""

    __tablename__ = "playbook_versions"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(12))  # draft | rejected | canary | live | retired | blocked
    lessons: Mapped[dict] = mapped_column(JSONB, default=dict)  # {situation: [lesson, …]}
    examples: Mapped[dict] = mapped_column(JSONB, default=dict)  # {situation: [{"context", "reply", "lang"}, …]}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    canary_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    live_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    gate: Mapped[dict] = mapped_column(JSONB, default=dict)
    canary: Mapped[dict] = mapped_column(JSONB, default=dict)
    note: Mapped[str] = mapped_column(Text, default="")
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)


class CapabilityGap(Base):
    """Something a person asked that Saheli could not do (the list of what to build next)."""

    __tablename__ = "capability_gaps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    family_id: Mapped[str] = mapped_column(String(64))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    asked: Mapped[str] = mapped_column(Text)  # anonymised
    how: Mapped[str] = mapped_column(String(24))  # said_cannot | tool_refused | unknown_tool
