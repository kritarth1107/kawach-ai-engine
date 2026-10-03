"""What people ask that Saheli cannot do yet: the list of what to build next.

Found three ways: Saheli says she can't ("I can't do that", "mere paas ye suvidha nahi"), a tool refuses because
the thing is outside what it can do, or the model asks for a tool that does not exist.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.learn.anonymise import anonymise
from app.learn.models import CapabilityGap

CANNOT = re.compile(
    r"\b(i can'?t|i cannot|i am not able|i'?m not able|i don'?t have (a way|access)|not something i can|i'?m unable|"
    r"main (ye|yeh|woh|wo)? ?nahi kar sakti|nahi kar paungi|mere paas (ye|yeh|aisi)? ?(suvidha|tarika)|mujhse (ye|yeh) nahi ho)\b|"
    r"मैं (यह|ये)? ?नहीं कर सकती",
    re.I,
)
SAFETY_REFUSALS = re.compile(r"\b(dose|dawai|medicine|doctor (hi|will)|double|extra|secret|ignore)\b", re.I)  # deliberate refusals, not gaps
CATEGORIES = {
    "bills_payments": r"\b(bill|electricity|bijli|recharge|pay|payment|upi|gas cylinder|rent|emi)\b",
    "calls": r"\b(call|phone kar|video call|ring)\b",
    "doctor_booking": r"\b(appointment|book (a )?doctor|consult|teleconsult|clinic slot)\b",
    "lab_tests": r"\b(lab|blood test|sample|home collection|test book|report download)\b",
    "home_services": r"\b(plumber|electrician|maid|cook|cleaning|repair|carpenter|nurse|attendant|physio)\b",
    "medicine_info": r"\b(side effect|interaction|substitute|generic|what is this medicine)\b",
    "entertainment": r"\b(song|bhajan|music|play|news|cricket|movie|tv|serial|story)\b",
    "travel_tickets": r"\b(train|flight|ticket|bus|irctc)\b",
    "documents": r"\b(aadhaar|pension|bank|insurance claim|form|certificate)\b",
    "shopping_other": r"\b(amazon|flipkart|clothes|saree|gift)\b",
    "reminders_other": r"\b(alarm|wake me|remind me to call)\b",
}


def category(text: str) -> str:
    for name, pat in CATEGORIES.items():
        if re.search(pat, text or "", re.I):
            return name
    return "other"


def detect(*, user_text: str, reply: str, actions: list[dict]) -> str | None:
    """How this turn shows a gap, or None."""
    if any(a.get("unknown") for a in actions):
        return "unknown_tool"
    if CANNOT.search(reply or "") and not SAFETY_REFUSALS.search(reply or ""):
        return "said_cannot"
    return None


async def record(session: AsyncSession, *, family_id: str, user_text: str, how: str, names: list[str]) -> None:
    session.add(CapabilityGap(family_id=family_id, at=clock.now(), category=category(user_text),
                              asked=anonymise(user_text, names=names)[:400], how=how))


async def ranking(session: AsyncSession, *, days: int = 30) -> list[dict]:
    rows = list((await session.execute(select(CapabilityGap).where(CapabilityGap.at >= clock.now() - timedelta(days=days)))).scalars())
    by = Counter(r.category for r in rows)
    fams: dict[str, set] = {}
    samples: dict[str, list] = {}
    for r in rows:
        fams.setdefault(r.category, set()).add(r.family_id)
        samples.setdefault(r.category, [])
        if len(samples[r.category]) < 3:
            samples[r.category].append(r.asked)
    return [{"category": c, "asks": n, "families": len(fams[c]), "examples": samples[c]} for c, n in by.most_common()]
