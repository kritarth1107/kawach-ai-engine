"""Tables of the daily score (trust plan C1, A2, C2, and the replay cases C3 runs before every deploy).

All new tables (created by create_all at start; nothing here changes an existing table):
- reply_verdicts: the full-memory judge's structured verdict on one real Saheli message (one row per reply_log row).
- daily_scores: the three checks of 3 Oct and their minimum, per family and for everyone ("all"), per IST day.
- memory_scores: the nightly memory score per person (stored, recalled, used) and per family (subject "family" = the
  minimum across its people).
- replay_cases: every failure as a case that can be run again: the context snapshot, the bad reply, what must hold.
- eval_scores: the nightly world score (scores.json from the 1,000-family probes) and later the replay runs.
- judge_labels: people's yes/no on the judge's verdicts (the monthly calibration), and the fixture-set runs.
- trainer_runs: one night of the trainer (eval/trainer.py): what it tried, what it kept, its pull requests and cost.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ReplyVerdict(Base):
    __tablename__ = "reply_verdicts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reply_id: Mapped[int] = mapped_column(BigInteger)  # reply_log.id
    family_id: Mapped[str] = mapped_column(String(64))
    thread_id: Mapped[str] = mapped_column(String(64))  # the person the message went to
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # when Saheli sent it (memory is rebuilt as of this moment)
    day: Mapped[str] = mapped_column(String(10))  # IST yyyy-mm-dd of the message
    kind: Mapped[str] = mapped_column(String(16), default="reply")  # reply_log.kind: reply | proactive | relay
    passed: Mapped[bool] = mapped_column(Boolean)
    # the issue names found (made_up_fact, own_medical_advice, wrong_person, wrong_name, wrong_medicine, wrong_time,
    # wrong_language, ignored_ask, broken_promise, tone, correction_ignored, missed_alert, unneeded_alert)
    issues: Mapped[list] = mapped_column(JSONB, default=list)
    verdict: Mapped[dict] = mapped_column(JSONB, default=dict)  # the judge's whole structured answer
    model: Mapped[str] = mapped_column(String(80), default="")
    cost_inr: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_version: Mapped[int] = mapped_column(Integer, default=1)
    judged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("uq_reply_verdicts_reply", "reply_id", unique=True),
        Index("ix_reply_verdicts_day", "day", "family_id"),
    )


class DailyScore(Base):
    __tablename__ = "daily_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    day: Mapped[date] = mapped_column(Date)
    family_id: Mapped[str] = mapped_column(String(64))  # "all" for every family together
    # reminders (check 1) | remembered (check 2) | corrections (check 3) | overall (the minimum) | replies (judge pass rate)
    check: Mapped[str] = mapped_column(String(16))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-100; None = no data that day (never shown as 100)
    n: Mapped[int] = mapped_column(Integer, default=0)  # the denominator
    failures: Mapped[list] = mapped_column(JSONB, default=list)  # [{family, person, what, ref, ...}] one line each
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)  # counts behind the number (held by rule, source, ...)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("uq_daily_scores", "day", "family_id", "check", unique=True),)


class MemoryScore(Base):
    __tablename__ = "memory_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))  # a person, or "family" (the minimum across its people)
    day: Mapped[date] = mapped_column(Date)
    stored: Mapped[float | None] = mapped_column(Float, nullable=True)
    recalled: Mapped[float | None] = mapped_column(Float, nullable=True)
    used: Mapped[float | None] = mapped_column(Float, nullable=True)  # None = no conversation that day ("no data", not 100)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # the minimum of the three parts that have data
    unknowns: Mapped[list] = mapped_column(JSONB, default=list)  # truths missing from the record (asked, never a failure)
    failures: Mapped[list] = mapped_column(JSONB, default=list)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("uq_memory_scores", "family_id", "subject_id", "day", unique=True),)


class ReplayCase(Base):
    """One bad reply, kept so it is fixed once and never comes back (trust plan C3 runs these before every deploy).

    Real families' cases stay in this production table (never in the public repo). context holds what is needed to run
    the turn again: {"time": ISO, "family_id", "speaker": {id, role}, "elder": {...}, "members": [...], "memory": {"record":
    text, "notes": text, "rejected": [...], "pending": [...]}, "turns": [{role, text, at}] (the last turns), "message":
    the person's message (or the scheduler prompt), "settings": {"language": {...}}, "events": [...]}.
    assertion is a list of code-checkable rules, each {"type": ..., ...}:
      not_contains {"words": [...]}            avoided names, rejected items, the old value of a correction
      language {"script": "devanagari", "language": "hi"}
      medicine_time {"medicine": "...", "times": ["08:00"]}
      no_tool {"tools": ["start_task"]}       a tool that must not be called
      must_tool {"tools": ["alert_caregiver"]}
      max_sentences {"n": 3}
      grounded {}                             every specific must be in memory or this chat (the judge decides)
    plus rubric: one line for the judge for what code cannot check."""

    __tablename__ = "replay_cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(24))  # judge | memory_score | correction | guard | thumbs_down | probe | founder
    source_ref: Mapped[str] = mapped_column(String(200))  # idempotency: the same failure never makes two cases
    kind: Mapped[str] = mapped_column(String(40))  # the failure kind (made_up_fact, avoided_name, recall_medicine, ...)
    family_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    day: Mapped[str] = mapped_column(String(10))
    context: Mapped[dict] = mapped_column(JSONB, default=dict)
    message: Mapped[str] = mapped_column(Text, default="")
    bad_reply: Mapped[str] = mapped_column(Text, default="")
    assertion: Mapped[list] = mapped_column(JSONB, default=list)
    rubric: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(12), default="open")  # open | fixed | dropped
    anonymised: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # {"passed": bool, "runs": [...], "commit": sha}
    fixed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fixed_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("uq_replay_cases_ref", "source_ref", unique=True),
        Index("ix_replay_cases_status", "status", "created_at"),
        Index("ix_replay_cases_family", "family_id", "created_at"),
    )


class EvalScore(Base):
    __tablename__ = "eval_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(80))  # the GitHub run id (or a fixture name)
    kind: Mapped[str] = mapped_column(String(16), default="nightly")  # nightly | fixture | replay
    day: Mapped[date] = mapped_column(Date)
    passed: Mapped[bool] = mapped_column(Boolean)
    commit: Mapped[str] = mapped_column(String(64), default="")
    branch: Mapped[str] = mapped_column(String(80), default="")
    scores: Mapped[dict] = mapped_column(JSONB, default=dict)  # scores.json as the night wrote it
    failures: Mapped[list] = mapped_column(JSONB, default=list)  # the broken bars, one line each
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("uq_eval_scores_run", "run_id", "kind", unique=True), Index("ix_eval_scores_day", "kind", "day"))


class JudgeLabel(Base):
    __tablename__ = "judge_labels"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch: Mapped[str] = mapped_column(String(16))  # the month, yyyy-mm
    verdict_id: Mapped[int] = mapped_column(BigInteger)  # reply_verdicts.id
    judge_pass: Mapped[bool] = mapped_column(Boolean)
    label: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # a person's yes (the reply is good) / no
    labeled_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("uq_judge_labels", "batch", "verdict_id", unique=True),)


class TrainerRun(Base):
    """One run of the nightly trainer: the full report (kinds, cases, every edit tried and why it was kept or not, the
    regression, pull requests, notes) for the admin's Trainer page. Anonymised case ids only, no message text of real
    families beyond the anonymised replies the report quotes."""

    __tablename__ = "trainer_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    day: Mapped[str] = mapped_column(String(10))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | done | failed
    dry_run: Mapped[bool] = mapped_column(Boolean, default=False)
    spent_inr: Mapped[float] = mapped_column(Float, default=0.0)
    cap_inr: Mapped[float] = mapped_column(Float, default=0.0)
    cases: Mapped[int] = mapped_column(Integer, default=0)  # open and fixed cases it could see
    tried: Mapped[int] = mapped_column(Integer, default=0)
    kept: Mapped[int] = mapped_column(Integer, default=0)
    prs: Mapped[list] = mapped_column(JSONB, default=list)
    error: Mapped[str] = mapped_column(Text, default="")
    report: Mapped[dict] = mapped_column(JSONB, default=dict)
    task_id: Mapped[str] = mapped_column(String(64), default="")

    __table_args__ = (Index("ix_trainer_runs_started", "started_at"),)
