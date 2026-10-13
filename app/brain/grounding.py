"""No invention: every specific in Saheli's reply must come from memory, this chat or a tool result (trust plan C4).

Founder 2026-10-13: "Any name, food, event or time in her reply has to match memory or this chat, or she rewrites it.
Round 2 still showed made-up life details and her own medical advice." Examples it catches: "Mala temple gayi hain" when
nothing says so, a helper's phone number nobody gave, "one rasgulla is fine", potassium thresholds, NSAID advice, walk
limits.

How:
- One fast model call reads only SAHELI'S OWN draft (never the person's words as rules) and lists every specific claim with
  its exact words: a person and what about them, a food, an event (also "you already took it"), a time or date, a place,
  a medicine or dose, a number, and advice. For each it says whether the SOURCES support it: memory (the care record,
  notes, events, household), what the people wrote in this chat, and this turn's tool results. Saheli's own earlier replies
  are NOT a source, so an invention never becomes "known" by being repeated.
- Code decides: an unsupported claim is a problem, and so is prescriptive health/food/medicine advice that does not repeat
  a tool result (web_search), a doctor's instruction in memory, or a safety step (call 112, tell the family, see a doctor).
- The loop then asks for a rewrite and checks the rewrite again; if it still invents, the sentences with those exact words
  are removed by code (app.brain.loop).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

KINDS = ("person", "food", "event", "time", "date", "place", "medicine", "number", "advice")

PROMPT = """You check a WhatsApp reply from Saheli, a care companion for elders and families, before it is sent.

1. List every specific claim in the REPLY: a person (a name or relation and what is said about them), a food or dish, an
   event (something that happened or will happen, including "you already took your tablet" or "Rohit called"), a time, a
   date, a place, a medicine or dose, a number (a reading, money, a count), and advice (anything telling them what to do
   or not do about their health, food, medicine or activity). Quote the exact words from the REPLY.
   NOT claims: greetings, questions to the person, offers ("shall I…"), what Saheli herself will do, feelings, general
   kindness, and restating what the person just said in this message.
2. For each claim say whether the SOURCES support it: the same fact in any language or wording. SOURCES are MEMORY (care
   record, notes, events, household), THEIR MESSAGES (what the people wrote) and TOOL RESULTS (this turn). Nothing else
   counts: not general knowledge about this family, not a guess.
3. For advice: prescriptive = true when it tells them to do something specific about their own health, food, medicine or
   activity (a dose, "X is fine to eat", a threshold, "you can skip it", how far to walk). source = "tool" when it repeats a
   TOOL RESULT (e.g. a web search) as general information, "doctor" when it repeats a doctor's instruction in MEMORY,
   "safety" when it is a safety step (call 112 or an ambulance, tell the family, see a doctor), else "none".

Return only JSON: {"claims": [{"kind": "...", "quote": "exact words", "supported": true|false, "prescriptive": true|false,
"source": "tool|doctor|safety|none"}]}. An empty list when there are no claims."""


@dataclass
class Claim:
    kind: str
    quote: str
    supported: bool
    prescriptive: bool = False
    source: str = "none"

    @property
    def problem(self) -> bool:
        if self.kind == "advice":
            return self.prescriptive and self.source not in ("tool", "doctor", "safety")
        return not self.supported


def needs_check(draft: str) -> bool:
    """Cheap pre-check on Saheli's own text: anything long enough to carry a specific."""
    return len((draft or "").strip()) >= 25


def _parse(raw: str) -> list[Claim]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return []
    out = []
    for c in (data.get("claims") or []) if isinstance(data, dict) else []:
        if not isinstance(c, dict) or c.get("kind") not in KINDS or not str(c.get("quote") or "").strip():
            continue
        out.append(Claim(kind=c["kind"], quote=str(c["quote"]).strip()[:300], supported=c.get("supported") is True,
                         prescriptive=c.get("prescriptive") is True, source=str(c.get("source") or "none")))
    return out


async def check(draft: str, *, memory: str, their_messages: list[str], tool_results: list[str]) -> list[Claim]:
    """The claims in the draft that memory, their messages and this turn's tools do not support (or own advice)."""
    if os.getenv("GROUNDING_CHECK", "on").strip().lower() == "off" or not needs_check(draft):
        return []
    sources = ("MEMORY\n" + (memory or "")[:24000] + "\n\nTHEIR MESSAGES\n" + "\n".join(f"- {m[:600]}" for m in their_messages[-30:] if m)
               + "\n\nTOOL RESULTS\n" + ("\n".join(t[:2000] for t in tool_results[-12:]) or "- none"))
    try:
        from app.llm import router

        reply = await router.complete("judge_fast", system_stable=PROMPT, effort=os.getenv("GROUNDING_EFFORT", "minimal"), max_tokens=1500, timeout_s=12,
                                      messages=[{"role": "user", "content": [{"type": "text", "text": f"{sources}\n\nREPLY\n{draft}"}]}])
    except Exception:  # noqa: BLE001 — the other guards still run; a missed check is logged, never blocks the reply
        logger.warning("grounding check failed")
        return []
    return [c for c in _parse(reply.text or "") if c.problem]


def problems(claims: list[Claim]) -> list[str]:
    out = []
    for c in claims:
        if c.kind == "advice":
            out.append(f"'{c.quote}' is your own health advice: give no doses, food rules or limits of your own; share only what a "
                       "search result or their doctor said (as general information, check with the doctor), or ask")
        else:
            out.append(f"'{c.quote}' is not in memory or this chat: remove it, or ask them instead of stating it")
    return out


_SENT = re.compile(r"(?<=[.!?।])\s+|\n+")


def strip(text: str, claims: list[Claim]) -> str:
    """Remove the sentences that carry an unsupported quote (the last resort after a rewrite still invents)."""
    quotes = [c.quote.lower() for c in claims if c.quote]
    kept = [s for s in _SENT.split(text or "") if s.strip() and not any(q in s.lower() for q in quotes)]
    return " ".join(kept).strip()
