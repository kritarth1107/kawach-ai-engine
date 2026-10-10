"""What happened afterwards, and whether Saheli's help was useful.

Outcomes (a fall, a hospital or doctor visit, a medicine change, recovery) are the answer key for learning:
they tell us which patterns and alerts mattered. Feedback (👍/👎) tells us which ones a family finds useful.

Both arrive three ways, none of them a form:
- said in chat ("Papa gir gaye", "doctor ne Telma 80 kar di"): the brain calls log_outcome
- one tap on a WhatsApp button Saheli attached (ids "v2:oc:…" and "v2:fb:…"), handled here without a model call
- the dashboard (Log an event, 👍/👎 on cards), or automatically (a caregiver changed a medicine)

Cross-family learning uses only families whose caregiver said yes (learning consent); a family's own
memory always learns.
"""

from __future__ import annotations

import re
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import CareEvent
from app.core import clock

OUTCOMES = {
    "fall": "Fall",
    "hospital_visit": "Hospital visit",
    "er_visit": "Emergency room",
    "doctor_visit": "Doctor visit",
    "medicine_changed": "Medicine changed",
    "recovered": "Better / recovered",
    "all_fine": "All fine",
    "other": "Other",
}
SOURCES = ("said", "button", "dashboard", "auto")
CONSENT_KEY = "family:learning_consent"


async def record_outcome(session: AsyncSession, *, family_id: str, subject_id: str, kind: str, summary: str, source: str,
                         actor_id: str | None, related: str | None = None, ref: str | None = None) -> bool:
    if kind not in OUTCOMES:
        raise ValueError(f"unknown outcome {kind}")
    return await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="outcome", summary=f"{OUTCOMES[kind]}: {summary}".strip(": "),
        payload={"outcome": kind, "source": source, "related": related}, actor_id=actor_id, ref=ref,
    )


async def record_feedback(session: AsyncSession, *, family_id: str, subject_id: str, target: str, vote: str, by: str | None,
                          source: str) -> bool:
    """One vote per person per target per day (a re-tap the same day is ignored; a change of mind the next day counts)."""
    if vote not in ("up", "down"):
        raise ValueError("vote must be up or down")
    return await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="feedback", summary=f"{'👍' if vote == 'up' else '👎'} {target}",
        payload={"target": target, "vote": vote, "source": source}, actor_id=by, ref=f"fb:{target}:{by}:{clock.ist_day()}",
    )


async def votes(session: AsyncSession, family_id: str, *, days: int = 90) -> dict[str, dict[str, int]]:
    """{target: {"up": n, "down": n}} using each person's latest vote."""
    rows = list((await session.execute(
        select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == "feedback",
                                CareEvent.at >= clock.now() - timedelta(days=days)).order_by(CareEvent.at)
    )).scalars())
    latest: dict[tuple, str] = {}
    for r in rows:
        p = r.payload or {}
        latest[(p.get("target"), r.actor_id)] = p.get("vote")
    out: dict[str, dict[str, int]] = {}
    for (target, _), v in latest.items():
        out.setdefault(target, {"up": 0, "down": 0})[v] += 1
    return out


async def recent(session: AsyncSession, family_id: str, subject_id: str, *, days: int = 60) -> list[dict]:
    rows = await store.events(session, family_id, subject_id, since=clock.now() - timedelta(days=days), kinds=["outcome"], limit=500)
    return [{"at": r.at.isoformat(), "kind": (r.payload or {}).get("outcome"), "label": OUTCOMES.get((r.payload or {}).get("outcome"), ""),
             "summary": r.summary, "source": (r.payload or {}).get("source")} for r in reversed(rows)]


# ── consent ────────────────────────────────────────────────────────────────────


async def consent(session: AsyncSession, family_id: str, elder_id: str) -> dict:
    f = await store.active_fact(session, family_id, elder_id, CONSENT_KEY)
    v = (f.value if f else None) or {}
    return {"granted": bool(v.get("granted")), "by": v.get("by"), "at": v.get("at")}


async def set_consent(session: AsyncSession, family_id: str, elder_id: str, *, granted: bool, by: str) -> dict:
    await store.write_fact(
        session, family_id=family_id, subject_id=elder_id, domain="family", key=CONSENT_KEY,
        value={"granted": granted, "by": by, "at": clock.now().isoformat()},
        text=("Family agreed" if granted else "Family did not agree") + " to share anonymised data to improve Saheli for everyone",
        source_kind="dashboard", stated_by=by,
    )
    return {"granted": granted}


# ── automatic outcomes ─────────────────────────────────────────────────────────


async def note_fact_change(session: AsyncSession, *, family_id: str, subject_id: str, domain: str, result: str, text: str,
                           source_kind: str, actor_id: str | None) -> None:
    """A caregiver (or a prescription) changed or stopped a medicine: that is an outcome worth learning from."""
    if domain != "medicine" or source_kind not in ("caregiver_said", "dashboard", "prescription") or result not in ("superseded", "stopped"):
        return
    await record_outcome(session, family_id=family_id, subject_id=subject_id, kind="medicine_changed", summary=text[:160],
                         source="auto", actor_id=actor_id)


# ── buttons ────────────────────────────────────────────────────────────────────

BUTTON_SETS = {
    # set: [(code, english, hinglish, hindi)]
    "outcome": [("all_fine", "All fine", "Sab theek", "सब ठीक"), ("doctor_visit", "Saw doctor", "Doctor dikhaya", "डॉक्टर को दिखाया"),
                ("hospital_visit", "Hospital", "Hospital", "अस्पताल")],
    "visit": [("nochange", "No change", "Koi badlav nahi", "कोई बदलाव नहीं"), ("medicine_changed", "Medicine changed", "Dawai badli", "दवा बदली"),
              ("later", "Tell later", "Baad mein", "बाद में")],
    "feedback": [("up", "👍 Useful", "👍 Kaam ka", "👍 काम का"), ("down", "👎 Not useful", "👎 Kaam ka nahi", "👎 काम का नहीं")],
    # the one confirm of an order (key = task id): a tap places, changes or cancels it without a model reading words
    "order": [("yes", "Yes, order it", "Haan, mangao", "हाँ, मंगाओ"), ("change", "Change", "Badlo", "बदलो"), ("cancel", "Cancel", "Cancel karo", "रद्द करो")],
}


def buttons(kind: str, key: str, profile: dict | None) -> list[dict]:
    """WhatsApp reply buttons for a question Saheli asks; ids carry the meaning so a tap needs no model."""
    from app.brain import guards

    k = guards.canned_key(profile)
    col = 3 if k in ("devanagari", "mr") else (2 if k == "indic" else 1)
    tag = {"feedback": "fb", "order": "od"}.get(kind, "oc")
    return [{"id": f"v2:{tag}:{row[0]}:{key}"[:250], "title": row[col][:20]} for row in BUTTON_SETS[kind]]


ORDER_TAP = {
    "english": {"yes": "Okay, placing the order now 🙏", "cancel": "Okay, cancelled. Nothing was ordered 🙏", "done": "That order is not waiting any more."},
    "indic": {"yes": "Theek hai, order kar rahi hoon 🙏", "cancel": "Theek hai, cancel kar diya. Kuch order nahi hua 🙏", "done": "Woh order ab ruka hua nahi hai."},
    "devanagari": {"yes": "ठीक है, ऑर्डर कर रही हूँ 🙏", "cancel": "ठीक है, रद्द कर दिया। कुछ ऑर्डर नहीं हुआ 🙏", "done": "वो ऑर्डर अब रुका हुआ नहीं है।"},
}


async def handle_order_button(session: AsyncSession, *, family_id: str, speaker_id: str, elder_id: str, code: str, task_id: str,
                              profile: dict | None) -> tuple[str | None, str | None]:
    """A tap on an order's confirm buttons (key "<task id>:<cart check>"): yes places it (the same checks as a spoken yes),
    cancel drops it. Returns (reply, None) when done, or (None, note for the brain) when the brain should answer: change,
    a yes that needs explaining, or a yes on an older confirm whose cart has changed since (never placed on an old tap)."""
    import uuid as _uuid

    from app.brain import guards
    from app.brain.tools import task_agent
    from app.tasks import runtime
    from app.tasks.models import Task

    tid, _, fp = task_id.partition(":")
    if code == "change":
        return None, f"(tapped the Change button on order {tid}: ask in one short line what they want to change)"
    t = guards.canned(profile, ORDER_TAP)
    try:
        task = await session.get(Task, _uuid.UUID(tid), with_for_update=True, populate_existing=True)
    except ValueError:
        task = None
    if not task or task.family_id != family_id or task.status not in runtime.LIVE:
        return t["done"], None
    if code == "yes":
        if fp and fp != str((task.details or {}).get("cart_fp") or "")[:8]:
            return None, (f"(tapped Yes on an older confirm of order {tid}; the cart or amount changed since, so nothing was placed: "
                          "tell them the current cart and amount from task_status in one short line and ask again)")
        out = await runtime.provide_input(session, task, kind="confirm", value="yes", by=speaker_id, by_is_elder=speaker_id == elder_id)
        if out.startswith("confirmed"):
            return t["yes"], None
        return None, f"(tapped Yes on order {tid}; it could not be placed yet: {out[:300]}. Tell them in one short line)"
    if code == "cancel":
        await runtime.request_cancel(session, task_agent(), task, by=speaker_id, reason="tapped cancel")
        return t["cancel"], None
    return None, None


THANKS = {
    "english": {"up": "Thank you, noted 🙏", "down": "Thank you, I'll mention fewer things like this 🙏", "all_fine": "So glad to hear that 🙏",
                "doctor_visit": "Noted 🙏 If the doctor changed anything, tell me and I'll update the reminders.",
                "hospital_visit": "Noted. Wishing a quick recovery 🙏 Tell me what the doctors say, or any medicine change.",
                "nochange": "Noted, nothing changes 🙏", "medicine_changed": "Please tell me the new medicine, dose and times, and I'll update the reminders.",
                "later": "Sure, whenever you're ready 🙏"},
    "indic": {"up": "Shukriya, note kar liya 🙏", "down": "Shukriya, aisi baatein kam bataungi 🙏", "all_fine": "Sunke bahut achha laga 🙏",
              "doctor_visit": "Note kar liya 🙏 Doctor ne kuch badla ho to bataiye, main reminder badal dungi.",
              "hospital_visit": "Note kar liya. Jaldi theek ho jayein 🙏 Doctor kya kehte hain ya dawai mein badlav ho to bataiye.",
              "nochange": "Theek hai, kuch nahi badla 🙏", "medicine_changed": "Nayi dawai, dose aur time bataiye, main reminder badal dungi.",
              "later": "Theek hai, jab aapko theek lage 🙏"},
    "devanagari": {"up": "शुक्रिया, नोट कर लिया 🙏", "down": "शुक्रिया, ऐसी बातें कम बताऊँगी 🙏", "all_fine": "सुनकर बहुत अच्छा लगा 🙏",
                   "doctor_visit": "नोट कर लिया 🙏 डॉक्टर ने कुछ बदला हो तो बताइए, मैं रिमाइंडर बदल दूँगी।",
                   "hospital_visit": "नोट कर लिया। जल्दी ठीक हो जाएँ 🙏 डॉक्टर क्या कहते हैं या दवा में बदलाव हो तो बताइए।",
                   "nochange": "ठीक है, कुछ नहीं बदला 🙏", "medicine_changed": "नई दवा, खुराक और समय बताइए, मैं रिमाइंडर बदल दूँगी।",
                   "later": "ठीक है, जब आपको ठीक लगे 🙏"},
}
BUTTON_ID = re.compile(r"^v2:(fb|oc|od):([a-z_]+):(.+)$")


def parse_button(text: str) -> tuple[str, str, str] | None:
    m = BUTTON_ID.match((text or "").strip())
    return (m.group(1), m.group(2), m.group(3)) if m else None


async def handle_button(session: AsyncSession, *, family_id: str, elder_id: str, speaker_id: str, text: str, profile: dict | None) -> str | None:
    """A tapped Saheli button: record it and return the reply (no model call). None if it is not a Saheli button."""
    parsed = parse_button(text)
    if not parsed:
        return None
    tag, code, key = parsed
    subject = elder_id
    m = re.search(r"(?:^|:)subj=([^:]+)", key)
    if m:
        subject = m.group(1)
    if tag == "fb" and code in ("up", "down"):
        await record_feedback(session, family_id=family_id, subject_id=subject, target=key, vote=code, by=speaker_id, source="button")
    elif tag == "oc" and (code in OUTCOMES or code in ("nochange", "later")):
        if code in OUTCOMES:
            await record_outcome(session, family_id=family_id, subject_id=subject, kind=code, summary=f"answer to follow-up {key}",
                                 source="button", actor_id=speaker_id, related=key, ref=f"oc:{key}:{speaker_id}:{code}")
        await close_followups(session, family_id, key)
    else:
        return None
    from app.brain import guards

    table = guards.canned(profile, THANKS)
    return table.get(code, table["up"])


async def close_followups(session: AsyncSession, family_id: str, key: str) -> None:
    from app.care.models import OpenLoop

    rows = (await session.execute(
        select(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.status == "open", OpenLoop.dedupe_key == f"followup:{key}")
    )).scalars()
    for loop in rows:
        await store.close_loop(session, loop.id, note="answered")
