"""Nutrition tables (plan §N2). Created by `ensure_schema` (create_all); their indexes are in app.db.migrate INDEXES.

- nutrition_foods: the catalogue and cache. family_id NULL is a generic food shared by every family (no personal data in
  it, ever); a family's home recipe or a pack they sent a label for keeps its family_id.
- nutrition_entries: one row per item eaten, with the nutrients computed when it was saved (a later catalogue fix never
  changes history silently; a recompute is a versioned change). Every change is a memory version (kind "food").
- nutrition_jobs: the agent's work on one log_food call (resolving, waiting for an answer, saved, dropped).
- nutrition_pantry: food at home from delivered grocery orders (or that someone said they bought), with counts left.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class NutritionFood(Base):
    __tablename__ = "nutrition_foods"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # NULL: generic, shared by all
    name: Mapped[str] = mapped_column(String(200))  # canonical English name, e.g. "Egg, whole, boiled"
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)  # names it was asked by (any language), lower case
    brand: Mapped[str | None] = mapped_column(String(120), nullable=True)
    restaurant: Mapped[str | None] = mapped_column(String(160), nullable=True)
    kind: Mapped[str] = mapped_column(String(16), default="generic")  # generic | packaged | restaurant | recipe
    per_100g: Mapped[dict] = mapped_column(JSONB, default=dict)
    per_serving: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    serving_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    units: Mapped[dict] = mapped_column(JSONB, default=dict)  # {"piece": 50, "katori": 150}: grams per unit
    default_unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    typical_qty: Mapped[float | None] = mapped_column(Float, nullable=True)  # a usual portion in default_unit
    countable: Mapped[bool] = mapped_column(Boolean, default=False)  # eaten in pieces (eggs, rotis): an unsaid count is asked
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)  # counters: egg, tea, coffee, water, fruit, sweet…
    ingredients: Mapped[str | None] = mapped_column(Text, nullable=True)
    allergens: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    veg: Mapped[str | None] = mapped_column(String(8), nullable=True)  # veg | egg | nonveg | vegan | jain
    source: Mapped[str] = mapped_column(String(12), default="model")  # label | off | ifct | usda | web | model | caregiver | seed
    source_urls: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    checked: Mapped[dict] = mapped_column(JSONB, default=dict)  # the data checks: {"ok", "atwater", "rough", "notes"}
    barcode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    uses: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class NutritionEntry(Base):
    __tablename__ = "nutrition_entries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))  # who ate
    logged_by: Mapped[str] = mapped_column(String(64))  # the speaker id, or "saheli"
    meal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))  # groups one meal (one log)
    meal: Mapped[str] = mapped_column(String(12))  # breakfast | lunch | snack | dinner | after_dinner | late_night | other
    kind: Mapped[str] = mapped_column(String(10), default="food")  # food | skipped (they said they skipped this meal)
    eaten_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    day: Mapped[str] = mapped_column(String(10))  # IST day of eaten_at
    food_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    name_said: Mapped[str] = mapped_column(String(200), default="")
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(24), nullable=True)
    grams: Mapped[float | None] = mapped_column(Float, nullable=True)
    nutrients: Mapped[dict] = mapped_column(JSONB, default=dict)
    confidence: Mapped[str] = mapped_column(String(14), default="said")  # label | said | photo | estimate | unquantified
    source: Mapped[str] = mapped_column(String(12), default="chat")  # chat | photo | label | order | dashboard | caregiver | nudge | backfill
    source_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)  # message ref, task id, care event id
    photo_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)
    consumed: Mapped[str] = mapped_column(String(4), default="yes")  # yes | part
    portion_eaten: Mapped[str | None] = mapped_column(String(12), nullable=True)  # all | most | half | little
    status: Mapped[str] = mapped_column(String(10), default="active")  # active | corrected | deleted
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    flags: Mapped[dict] = mapped_column(JSONB, default=dict)  # {"allergy": [...], "rules": [...], "deviation": [...], "rough": true}
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class NutritionJob(Base):
    __tablename__ = "nutrition_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str] = mapped_column(String(64))
    requested_by: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(14), default="queued")  # queued | resolving | needs_input | saved | dropped
    ask: Mapped[dict] = mapped_column(JSONB, default=dict)  # the structured log_food arguments (as answered so far)
    needs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # {"reason", "question", "options", "item"}
    draft: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # resolved items before save, photo reading
    answers: Mapped[list] = mapped_column(JSONB, default=list)
    questions: Mapped[int] = mapped_column(Integer, default=0)  # questions asked about this meal (at most 2)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    source_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    channel: Mapped[str] = mapped_column(String(12), default="whatsapp")
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    trace_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class NutritionPantry(Base):
    __tablename__ = "nutrition_pantry"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # NULL: the household's
    food_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    brand: Mapped[str | None] = mapped_column(String(120), nullable=True)
    count: Mapped[float] = mapped_column(Float, default=0)
    unit: Mapped[str] = mapped_column(String(24), default="piece")
    unit_g: Mapped[float | None] = mapped_column(Float, nullable=True)  # grams in one unit, when known
    per_unit_nutrients: Mapped[dict] = mapped_column(JSONB, default=dict)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(12), default="order")  # order | said | dashboard
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    best_before: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(10), default="in_stock")  # in_stock | finished | removed
