"""Contract v1 between the brain and the specialist agents.

A task carries its goal and the limits it must stay inside. Limits are built from the care record at the
moment the task starts, so an agent never needs to read memory and can never widen them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store

CONTRACT_VERSION = 1

# Statuses an agent report can lead to (the task row keeps the same names).
STATUSES = ("queued", "running", "needs_input", "awaiting_confirm", "done", "failed", "cancelled")
INPUTS = ("otp", "confirm", "fee", "choice")

DEFAULT_BUDGET = {"order": 1500, "ride": 800}  # ₹; above this an elder's request needs a caregiver's OK
MAX_QTY = {"shopping": 6, "pharmacy": 3}  # per line, unless the person asked for more


@dataclass
class Limits:
    """What no agent may go past. Checked in code on every report (app.specialists.guard)."""

    cod_only: bool = True
    allergies: list[str] = field(default_factory=list)
    never_order: list[str] = field(default_factory=list)
    diet_rules: list[str] = field(default_factory=list)
    budget: int = 1500
    requester_is_elder: bool = False
    max_qty: int = 6
    place: dict = field(default_factory=dict)  # saved delivery place {addressId, nickname, pincode, full}
    rx_on_file: list[str] = field(default_factory=list)  # medicines with a prescription on file

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Limits":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


async def build_limits(session: AsyncSession, *, family_id: str, subject_id: str, kind: str, agent: str,
                       requester_is_elder: bool, place: dict | None = None) -> Limits:
    rows = await store.facts(session, family_id, subject_id, domains=["allergy", "no_order", "diet", "medicine"], statuses=("active",))
    allergies = [r.value.get("allergen") or r.key.split(":", 1)[1] for r in rows if r.domain == "allergy"]
    never = [r.value.get("item") or r.key.split(":", 1)[1].replace("_", " ") for r in rows if r.domain == "no_order"]
    diet = [r.text for r in rows if r.domain == "diet"]
    rx = [r.value.get("name") or r.key.split(":", 1)[1] for r in rows if r.domain == "medicine" and r.value.get("prescription")]
    return Limits(
        allergies=allergies, never_order=never, diet_rules=diet, budget=DEFAULT_BUDGET.get(kind, 1500),
        requester_is_elder=requester_is_elder, max_qty=MAX_QTY.get(agent, 6), place=place or {}, rx_on_file=rx,
    )
