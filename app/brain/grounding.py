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
- In the reply loop (actions=True) it also checks what Saheli says she did or will do: "I've saved / set / told …" must be
  a tool result of this turn or in memory, and "I will track / remind / do X from now on" must be something one of her
  tools can do (the tool list is given). On 13 Oct she promised to "track your proteins and calories from now on" with no
  tool that could; that promise is now rewritten into what she can do.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

KINDS = ("person", "food", "event", "time", "date", "place", "medicine", "number", "advice", "action")
DOCTOR_NOTE = "[health information with no doctor note]"

PROMPT = """You check a WhatsApp reply from Saheli, a care companion for elders and families, before it is sent.

1. List every specific claim in the REPLY: a person (a name or relation and what is said about them), a food or dish, an
   event (something that happened or will happen, including "you already took your tablet" or "Rohit called"), a time, a
   date, a place, a medicine or dose, a number (a reading, money, a count), and advice (anything telling them what to do
   or not do about their health, food, medicine or activity). Quote the exact words from the REPLY.
   NOT claims: greetings, offers ("shall I…"), what Saheli herself will do, feelings, general kindness, and restating what
   the person just said in this message. A question is not a claim, unless it names a dish, medicine, dose, place or
   relation the sources do not give: that name is still a claim.
2. For each claim say whether the SOURCES support it: the same fact in any language or wording. SOURCES are MEMORY (care
   record, notes, events, household), THEIR MESSAGES (what the people wrote) and TOOL RESULTS (this turn). Nothing else
   counts: not general knowledge about this family, not a guess.
3. For advice: prescriptive = true when it tells them to do something specific about their own health, food, medicine or
   activity (a dose, "X is fine to eat", a threshold, "you can skip it", how far to walk). Recommending a dish, snack or
   drink for their health is prescriptive even when hedged ("a good option", "usually", "try this"), and it is unsupported
   unless that dish is in THEIR MESSAGES, MEMORY or a TOOL RESULT. A place, or a kinship word used to address them, is a
   claim and is unsupported when that place or relation is not in the sources. Telling them to drink water or ORS, to
   rest, or to see a doctor for loose motions, weakness or dehydration is a safety step (source "safety"), not a dish
   recommendation. source = "tool" when it repeats a
   TOOL RESULT (e.g. a web search) as general information, "doctor" when it repeats a doctor's instruction in MEMORY,
   "safety" when it is a safety step (call 112 or an ambulance, tell the family, see a doctor), else "none".

Return only JSON: {"claims": [{"kind": "...", "quote": "exact words", "supported": true|false, "prescriptive": true|false,
"source": "tool|doctor|safety|none"}]}. An empty list when there are no claims."""

# Only when this turn looked health information up (web_search): health information always comes with when to see a doctor.
_DOCTOR_RULE = """N. doctor_note_missing = true when the REPLY shares general health or medicine information (a medicine's side
   effects or precautions, what a symptom or reading may mean, self-care for a problem) and says nothing about seeing or
   checking with a doctor or pharmacist (when to see one, or "check with your doctor"). False when it only repeats their own
   schedule, doses, readings or record, or already has such a line, or shares no health information. Add
   "doctor_note_missing": true|false to the JSON.

"""

_ACTIONS_RULE = """4. Also list what Saheli says SHE has done or WILL do (kind "action"): "I've saved / recorded / noted / set / ordered /
   booked / told X", "done", "I will track / remind / check / send / call / do X", "from now on I'll …". Quote the exact
   words. These are claims even when the person asked for exactly that: being asked is not doing.
   A place name, or addressing them by a relation the sources do not give, is a claim and is unsupported when that
   place or relation is not in the sources.
   supported = true only when:
   - a past action ("I've told Rohit", "saved", "done"): a TOOL RESULT of this turn shows that very action (each result
     starts with the tool's name, e.g. "send_message: …" for telling someone, "remember: …" for saving a fact), or MEMORY
     shows it was done earlier. No matching tool result means unsupported.
   - a promise ("I will …", "from now on …"): one of SAHELI'S ABILITIES below does that exact thing, or a tool result of
     this turn already set it up (e.g. a reminder). A nearby ability is not enough: noting a meal is not counting its
   calories, a reminder is not a phone call. Logging one meal does not cover a promise to track protein, calories or
   diet from now on. A tool result that says "one meal only" does not support "I set up tracking" or "I will track".
   A promise no ability covers is unsupported, however kind it sounds.
   Offers and questions ("shall I…?", "want me to…?") are not claims.

"""

PROMPT_ACTIONS = PROMPT.replace("what Saheli herself will do, ", "").replace("\nReturn only JSON", "\n" + _ACTIONS_RULE + "Return only JSON", 1)


# Health turns only, with the doctor-note rule: her own dose decisions (13 Oct live: "hold off on that Magnesium tablet
# tonight" passed as general information). Kept out of the other checks: every extra rule at effort "minimal" cost the
# action/promise claims accuracy (eval/grounding_eval.py 10-12 of 13; KP, 14 Oct), so health turns are checked at "low".
_DOSE_RULE = """M. For each advice claim also add "dose_change": true when it tells them to skip, hold off, delay, double, stop,
   start or change the amount or timing of a medicine or dose ("hold off on that tablet tonight", "take two", "skip the
   missed dose", "don't take it now"), whatever the reason given; false otherwise. Taking a due or late dose as prescribed,
   or keeping to the prescribed dose (not taking an extra or a double one), is not a dose change.

"""


def _system(actions: bool, doctor_note: bool) -> str:
    base = PROMPT_ACTIONS if actions else PROMPT
    if doctor_note:
        n = 5 if actions else 4
        rules = _DOCTOR_RULE.replace("N.", f"{n}.") + _DOSE_RULE.replace("M.", f"{n + 1}.")
        base = base.replace("\nReturn only JSON", "\n" + rules + "Return only JSON", 1)
    return base


def effort_for(doctor_note: bool) -> str:
    """Health turns carry two more rules: checked at "low" (13/13 in grounding_eval with them; minimal dropped to 10-12)."""
    return os.getenv("GROUNDING_EFFORT_HEALTH", "low") if doctor_note else os.getenv("GROUNDING_EFFORT", "minimal")


@lru_cache(maxsize=1)
def abilities() -> str:
    """One line per tool Saheli has: what she can actually do (for promise claims)."""
    from app.brain import tools

    lines = []
    for spec in sorted(tools.specs(), key=lambda s: s.name):
        desc = " ".join((spec.description or "").split())
        first = re.split(r"(?<=\.)\s", desc, maxsplit=1)[0]
        lines.append(f"- {spec.name}: {first[:160]}")
    return "\n".join(lines)


@dataclass
class Claim:
    kind: str
    quote: str
    supported: bool
    prescriptive: bool = False
    source: str = "none"
    dose_change: bool = False

    @property
    def problem(self) -> bool:
        if self.source == "doctor_note":
            return True
        if self.kind == "advice":
            if self.dose_change:
                # only the doctor changes a dose: a search result or a "safety" reason never makes "hold off on that tablet
                # tonight" hers to say (13 Oct live: a caregiver with an upset stomach was told to skip his magnesium)
                return self.source != "doctor"
            return self.prescriptive and self.source not in ("tool", "doctor", "safety")
        return not self.supported


def needs_check(draft: str) -> bool:
    """Cheap pre-check on Saheli's own text: anything long enough to carry a specific."""
    return len((draft or "").strip()) >= 25


def _parse(raw: str, doctor_note: bool = True) -> list[Claim]:
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
                         prescriptive=c.get("prescriptive") is True, source=str(c.get("source") or "none"), dose_change=c.get("dose_change") is True))
    if isinstance(data, dict) and data.get("doctor_note_missing") is True and doctor_note:
        # Health information always comes with when to see a doctor (persona; founder's health rule). No quote: nothing to
        # strip, only a rewrite that adds the line.
        out.append(Claim(kind="advice", quote=DOCTOR_NOTE, supported=False, prescriptive=True, source="doctor_note"))
    return out


async def check(draft: str, *, memory: str, their_messages: list[str], tool_results: list[str], actions: bool = False,
                doctor_note: bool = False) -> list[Claim]:
    """The claims in the draft that memory, their messages and this turn's tools do not support (or own advice).
    actions=True (the reply loop): also what Saheli says she did or will do, against this turn's tools and her abilities."""
    if os.getenv("GROUNDING_CHECK", "on").strip().lower() == "off" or not needs_check(draft):
        return []
    sources = ("MEMORY\n" + (memory or "")[:24000] + "\n\nTHEIR MESSAGES\n" + "\n".join(f"- {m[:600]}" for m in their_messages[-30:] if m)
               + "\n\nTOOL RESULTS\n" + ("\n".join(t[:2000] for t in tool_results[-12:]) or "- none"))
    try:
        from app.llm import router

        system = _system(actions, doctor_note) + ("\n\nSAHELI'S ABILITIES (her tools)\n" + abilities() if actions else "")
        reply = await router.complete("judge_fast", system_stable=system, effort=effort_for(doctor_note), max_tokens=4000, timeout_s=12,
                                      messages=[{"role": "user", "content": [{"type": "text", "text": f"{sources}\n\nREPLY\n{draft}"}]}])
    except Exception:  # noqa: BLE001 — the other guards still run; a missed check is logged, never blocks the reply
        logger.warning("grounding check failed")
        return []
    return [c for c in _parse(reply.text or "", doctor_note) if c.problem and (actions or c.kind != "action")]


def problems(claims: list[Claim]) -> list[str]:
    out = []
    for c in claims:
        if c.source == "doctor_note":
            out.append("you share health or medicine information but not when to see a doctor: add one short line (when to see "
                       "a doctor, or to check with their doctor or pharmacist)")
        elif c.kind == "advice" and c.dose_change:
            out.append(f"'{c.quote}' tells them to skip, delay, double or change a dose: only their doctor decides that. Do not say it; "
                       "ask them to check with their doctor or pharmacist (or that you are checking with the family), and offer to note it")
        elif c.kind == "advice":
            out.append(f"'{c.quote}' is your own health advice: give no doses, food rules or limits of your own; share only what a "
                       "search result or their doctor said (as general information, check with the doctor), or ask")
        elif c.kind == "action":
            out.append(f"'{c.quote}' says you did or will do something that no tool did this turn and none of your tools can do: "
                       "do it now with a tool if one fits, otherwise say plainly what you did and what you can do")
        else:
            out.append(f"'{c.quote}' is not in memory or this chat: remove it, or ask them instead of stating it")
    return out


_SENT = re.compile(r"(?<=[.!?।])\s+|\n+")


def strip(text: str, claims: list[Claim]) -> str:
    """Remove the sentences that carry an unsupported quote (the last resort after a rewrite still invents)."""
    quotes = [c.quote.lower() for c in claims if c.quote and c.source != "doctor_note"]
    kept = [s for s in _SENT.split(text or "") if s.strip() and not any(q in s.lower() for q in quotes)]
    return " ".join(kept).strip()
