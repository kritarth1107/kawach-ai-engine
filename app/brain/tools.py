"""Saheli's care tools. Memory tools act on Postgres directly; world tools go through the host."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.brain import policy
from app.brain.host import ToolHost
from app.care import store
from app.care.domains import DOMAINS, FOOD_TIMING, fact_key, slug
from app.core import clock
from app.llm.router import ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class TurnCtx:
    session: AsyncSession
    host: ToolHost
    family_id: str
    elder: dict  # {"id", "name"}
    speaker: dict  # {"id", "name", "role": "elder" | "caregiver" | ...}
    members: list[dict]
    message_ref: str | None = None
    alerts: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)

    @property
    def elder_id(self) -> str:
        return self.elder["id"]

    @property
    def speaker_is_elder(self) -> bool:
        return self.speaker.get("id") == self.elder_id

    @property
    def is_system(self) -> bool:
        return self.speaker.get("role") == "system"

    @property
    def source_kind(self) -> str:
        if self.is_system:
            return "inferred"
        return "elder_said" if self.speaker_is_elder else "caregiver_said"

    @property
    def actor_id(self) -> str:
        return self.elder_id if self.is_system else (self.speaker.get("id") or self.elder_id)

    def subject(self, about: str | None) -> str:
        ids = {m.get("id") for m in self.members} | {self.elder_id}
        return about if about in ids else self.elder_id


Handler = Callable[[TurnCtx, dict], Awaitable[Any]]
_TOOLS: dict[str, tuple[ToolSpec, Handler]] = {}


def tool(name: str, description: str, properties: dict, required: list[str]):
    schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    def wrap(fn: Handler) -> Handler:
        _TOOLS[name] = (ToolSpec(name, description, schema), fn)
        return fn

    return wrap


def specs() -> list[ToolSpec]:
    return [spec for spec, _ in _TOOLS.values()]


async def run(ctx: TurnCtx, name: str, args: dict) -> tuple[str, bool]:
    """Run a tool; returns (json result, is_error). Errors go back to the model, never to the user."""
    if name not in _TOOLS:
        return json.dumps({"error": f"unknown tool {name}"}), True
    try:
        async with ctx.session.begin_nested():
            out = await _TOOLS[name][1](ctx, args)
        ctx.actions.append({"tool": name, "args": args, "ok": True})
        return json.dumps(out, default=str, ensure_ascii=False), False
    except ToolRefused as exc:
        ctx.actions.append({"tool": name, "args": args, "ok": False, "refused": str(exc)})
        return json.dumps({"refused": str(exc)}), True
    except Exception as exc:  # noqa: BLE001 — the model decides what to tell the person
        logger.exception("tool %s failed", name)
        ctx.actions.append({"tool": name, "args": args, "ok": False, "error": str(exc)[:200]})
        return json.dumps({"error": "this did not work right now", "detail": str(exc)[:200]}), True


class ToolRefused(Exception):
    pass


ABOUT = {"type": "string", "description": "Person id this is about. Leave out for the care recipient."}


# ── memory ─────────────────────────────────────────────────────────────────────


@tool(
    "remember",
    "Save or update a fact in the care record so every later reply uses it. Use it before replying whenever "
    "someone tells you something that matters for care: a medicine or dose time, an allergy, a diet rule, a dish "
    "they cook, a routine, a naming preference or correction, family rules (who to call first, who pays, who must "
    "not be told), doctors, hospital, helpers. The same domain+name updates the existing fact. For medicine, give "
    "details {name, dose, times: ['HH:MM', ...], food_timing, days: [0-6, Monday=0] or omit for daily, "
    "instructions}; their reminders follow these times.",
    {
        "domain": {"type": "string", "enum": sorted(DOMAINS)},
        "name": {"type": "string", "description": "What it is, e.g. 'Metformin', 'milk', 'low_salt', 'address_as', 'call_first'"},
        "details": {"type": "object", "description": "Structured details"},
        "sentence": {"type": "string", "description": "One plain sentence for the record, e.g. 'Metformin 500 mg at 08:00 after breakfast'"},
        "about": ABOUT,
    },
    ["domain", "name", "details", "sentence"],
)
async def remember(ctx: TurnCtx, a: dict) -> dict:
    domain = a["domain"]
    subject = ctx.subject(a.get("about"))
    details = dict(a.get("details") or {})
    if domain == "medicine":
        details["name"] = details.get("name") or a["name"]
        times = details.get("times") or []
        details["times"] = sorted({t for t in times if isinstance(t, str) and len(t) <= 5 and ":" in t})
        if details.get("food_timing") not in FOOD_TIMING:
            details.pop("food_timing", None)
    w = await store.write_fact(
        ctx.session,
        family_id=ctx.family_id,
        subject_id=subject,
        domain=domain,
        key=fact_key(domain, a["name"]),
        value=details,
        text=a["sentence"],
        source_kind=ctx.source_kind,
        source_ref=ctx.message_ref,
        stated_by=ctx.speaker.get("id"),
        confidence=0.9 if ctx.speaker_is_elder else 1.0,
    )
    await store.record_event(
        ctx.session,
        family_id=ctx.family_id,
        subject_id=subject,
        kind="fact_" + w.result,
        summary=a["sentence"],
        payload={"key": w.fact.key},
        actor_id=ctx.speaker.get("id"),
    )
    out: dict = {"result": w.result, "key": w.fact.key}
    if w.result == "pending":
        out["note"] = "This changes a fact a caregiver or prescription set; it waits for a caregiver to confirm."
    if domain == "medicine" and w.result in ("created", "superseded") and subject == ctx.elder_id:
        out["reminders"] = await ctx.host.call(
            "sync_medicine_schedule",
            {
                "key": w.fact.key,
                "name": details["name"],
                "dose": details.get("dose"),
                "times": details["times"],
                "food_timing": details.get("food_timing"),
                "days": details.get("days"),
                "instructions": details.get("instructions"),
                "active": True,
            },
            family_id=ctx.family_id,
            subject_id=subject,
            actor_id=ctx.actor_id,
        )
    return out


@tool(
    "stop",
    "End a fact: a medicine that was stopped, a diet rule lifted, a routine that no longer happens, an allergy "
    "that was a mistake. A stop the elder reports for a health fact waits for a caregiver to confirm.",
    {"domain": {"type": "string", "enum": sorted(DOMAINS)}, "name": {"type": "string"}, "reason": {"type": "string"}, "about": ABOUT},
    ["domain", "name", "reason"],
)
async def stop(ctx: TurnCtx, a: dict) -> dict:
    subject = ctx.subject(a.get("about"))
    key = fact_key(a["domain"], a["name"])
    w = await store.stop_fact(
        ctx.session,
        family_id=ctx.family_id,
        subject_id=subject,
        key=key,
        reason=a["reason"],
        source_kind=ctx.source_kind,
        stated_by=ctx.speaker.get("id"),
        source_ref=ctx.message_ref,
    )
    if not w:
        return {"result": "not_found", "key": key}
    await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=subject, kind="fact_" + w.result,
        summary=f"{a['name']}: {a['reason']}", payload={"key": key}, actor_id=ctx.speaker.get("id"),
    )
    if w.result == "stopped" and a["domain"] == "medicine" and subject == ctx.elder_id:
        await ctx.host.call(
            "sync_medicine_schedule",
            {"key": key, "name": a["name"], "times": [], "active": False},
            family_id=ctx.family_id, subject_id=subject, actor_id=ctx.actor_id,
        )
    return {"result": w.result, "key": key}


@tool(
    "note",
    "Add to the memory notes about a person or the family: stories, people in their life, what they enjoy, "
    "how they like things done. Use remember instead for anything that drives care (medicine, diet, allergy, "
    "naming, routine).",
    {
        "about": {"type": "string", "description": "Person id, or 'family'"},
        "topic": {"type": "string", "description": "Short topic, e.g. 'food', 'grandchildren', 'health history'"},
        "text": {"type": "string"},
    },
    ["topic", "text"],
)
async def note(ctx: TurnCtx, a: dict) -> dict:
    subject = "family" if a.get("about") == "family" else ctx.subject(a.get("about"))
    existing = {n.slug: n for n in await store.notes(ctx.session, ctx.family_id, [subject])}
    s = slug(a["topic"])
    line = f"- {clock.ist_day()}: {a['text'].strip()}"
    body = (existing[s].body_md.rstrip() + "\n" + line) if s in existing else line
    await store.upsert_note(ctx.session, family_id=ctx.family_id, subject_id=subject, slug=s, title=a["topic"].strip().title(), body_md=body)
    return {"saved": True, "topic": s}


@tool(
    "recall",
    "Search memory: past messages and events, notes, and the care record including past values with dates. "
    "Use it before saying you don't know, and for questions about the past ('what did the doctor say', "
    "'what was the dose before').",
    {"query": {"type": "string"}, "about": ABOUT},
    ["query"],
)
async def recall(ctx: TurnCtx, a: dict) -> dict:
    subjects = [ctx.subject(a.get("about")), "family"]
    hits = await store.recall(ctx.session, ctx.family_id, subjects, a["query"], limit=15)
    return {"hits": [{"when": clock.ist(h.when).strftime("%d %b %Y %H:%M"), "source": h.source, "text": h.text} for h in hits]}


@tool(
    "confirm_change",
    "A caregiver approves or rejects a PENDING care-record change (only caregivers may do this).",
    {"key": {"type": "string", "description": "The fact key shown in the care record"}, "approve": {"type": "boolean"}},
    ["key", "approve"],
)
async def confirm_change(ctx: TurnCtx, a: dict) -> dict:
    if ctx.speaker_is_elder:
        raise ToolRefused("Only a caregiver can confirm this change.")
    pending = [
        f for f in await store.facts(ctx.session, ctx.family_id, ctx.elder_id, statuses=("pending",)) if f.key == a["key"]
    ]
    if not pending:
        return {"result": "nothing_pending", "key": a["key"]}
    row = await store.resolve_pending(ctx.session, fact_id=pending[-1].id, approve=bool(a["approve"]), by=ctx.speaker["id"])
    if row and row.domain == "medicine":
        v = row.value
        await ctx.host.call(
            "sync_medicine_schedule",
            {
                "key": row.key, "name": v.get("name", row.key), "dose": v.get("dose"), "times": v.get("times", []),
                "food_timing": v.get("food_timing"), "days": v.get("days"), "instructions": v.get("instructions"),
                "active": row.status == "active",
            },
            family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
        )
    return {"result": row.status if row else "not_found", "key": a["key"]}


# ── the day's care ─────────────────────────────────────────────────────────────


@tool(
    "log_dose",
    "Record what happened with a medicine dose: taken, skipped, refused, missed, or the strip is empty.",
    {
        "medicine": {"type": "string", "description": "Medicine name or key"},
        "outcome": {"type": "string", "enum": ["taken", "skipped", "refused", "missed", "empty_strip"]},
        "note": {"type": "string"},
    },
    ["medicine", "outcome"],
)
async def log_dose(ctx: TurnCtx, a: dict) -> dict:
    name = a["medicine"].split(":", 1)[-1]
    summary = f"{name}: {a['outcome']}" + (f" ({a['note']})" if a.get("note") else "")
    await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind=f"dose_{a['outcome']}",
        summary=summary, payload={"medicine": name}, actor_id=ctx.speaker.get("id"), ref=ctx.message_ref and f"{ctx.message_ref}:{slug(name)}",
    )
    tool_name = "mark_schedule_completed" if a["outcome"] == "taken" else "mark_schedule_missed"
    res = await ctx.host.call(
        tool_name, {"title": name, "note": a.get("note") or a["outcome"]},
        family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )
    out = {"logged": summary, "schedule": res}
    if a["outcome"] == "empty_strip":
        out["next"] = "Ask whether to reorder; ordering needs the medicine name and a confirmed quantity."
    return out


@tool(
    "log_vital",
    "Record a reading: bp ('140/90'), sugar (mg/dL, say fasting or after food in note), weight (kg), "
    "temperature (°F or °C), spo2 (%), pulse. Returns red_flag when the reading needs a caregiver now.",
    {
        "kind": {"type": "string", "enum": ["bp", "sugar", "weight", "temperature", "spo2", "pulse"]},
        "value": {"type": "string"},
        "unit": {"type": "string"},
        "note": {"type": "string"},
    },
    ["kind", "value"],
)
async def log_vital(ctx: TurnCtx, a: dict) -> dict:
    flag = policy.vital_red_flag(a["kind"], a["value"])
    summary = f"{a['kind']} {a['value']}{(' ' + a['unit']) if a.get('unit') else ''}" + (f" ({a['note']})" if a.get("note") else "")
    await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="vital", summary=summary,
        payload={**a, "red_flag": flag}, actor_id=ctx.speaker.get("id"),
    )
    await ctx.host.call(
        "log_vitals", {"kind": a["kind"], "value": a["value"], "unit": a.get("unit"), "note": a.get("note")},
        family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )
    return {"logged": summary, "red_flag": flag}


@tool(
    "log_event",
    "Record anything else about the day for the dashboard and daily snapshot: a meal or missed meal, water, "
    "mood, a symptom (pain, dizziness, swelling, constipation, wound, up at night, did not get out of bed), "
    "a routine done or missed, a home problem (helper did not come, power or water off, cylinder low, locked out), "
    "a call with family.",
    {
        "kind": {"type": "string", "enum": ["meal", "water", "mood", "symptom", "routine", "home", "sleep", "social", "other"]},
        "summary": {"type": "string"},
        "severity": {"type": "string", "enum": ["info", "watch", "concern"]},
    },
    ["kind", "summary"],
)
async def log_event(ctx: TurnCtx, a: dict) -> dict:
    await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind=a["kind"], summary=a["summary"],
        payload={"severity": a.get("severity", "info")}, actor_id=ctx.speaker.get("id"),
    )
    if a["kind"] == "symptom":
        await ctx.host.call(
            "log_symptom", {"symptom": a["summary"], "severity": a.get("severity", "info")},
            family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
        )
    return {"logged": a["summary"]}


@tool(
    "reminder_log",
    "What the reminder system actually did on a day: which reminders were sent, failed or never attempted, "
    "and the doses marked. Use it before answering any question about a reminder.",
    {"day": {"type": "string", "description": "YYYY-MM-DD, default today"}},
    [],
)
async def reminder_log(ctx: TurnCtx, a: dict) -> dict:
    day = a.get("day") or clock.ist_day()
    sent = await ctx.host.call(
        "get_reminder_log", {"dateKey": day}, family_id=ctx.family_id, subject_id=ctx.elder_id,
        actor_id=ctx.actor_id,
    )
    doses = await store.events(ctx.session, ctx.family_id, ctx.elder_id, day=day)
    return {
        "day": day,
        "reminder_attempts": sent.get("attempts", []),
        "medicine_schedule": sent.get("scheduled", []),
        "logged": [f"{clock.ist(e.at).strftime('%H:%M')} {e.kind}: {e.summary}" for e in doses if e.kind.startswith(("dose_", "reminder"))],
        "rule": "If a reminder is not in reminder_attempts with delivered=true, it was not sent. Say so plainly; never invent why.",
    }


@tool(
    "set_reminder",
    "A one-off or repeating reminder that is not a medicine (medicines get reminders through remember). "
    "times are 'HH:MM' in IST.",
    {"text": {"type": "string"}, "times": {"type": "array", "items": {"type": "string"}}},
    ["text", "times"],
)
async def set_reminder(ctx: TurnCtx, a: dict) -> dict:
    return await ctx.host.call(
        "create_reminder", {"text": a["text"], "times": a["times"], "kind": "multi_time"},
        family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )


@tool(
    "today_schedule",
    "Today's care schedule with each item's status (done, missed, due, upcoming).",
    {},
    [],
)
async def today_schedule(ctx: TurnCtx, a: dict) -> dict:
    return await ctx.host.call(
        "get_today_schedule", {}, family_id=ctx.family_id, subject_id=ctx.elder_id,
        actor_id=ctx.actor_id,
    )


# ── loops ──────────────────────────────────────────────────────────────────────


@tool(
    "open_loop",
    "Track something unfinished: a question you asked and need an answer to, a follow-up to do later, "
    "a task in progress. Set check_back_in_minutes to be woken up. if_no_answer decides what happens then: "
    "ask_again, alert_caregiver (only when the elder not answering is itself a concern, e.g. after a fall or a "
    "missed critical dose), or dashboard.",
    {
        "kind": {"type": "string", "enum": ["question", "followup", "task", "watch"]},
        "title": {"type": "string"},
        "check_back_in_minutes": {"type": "integer", "minimum": 5, "maximum": 10080},
        "if_no_answer": {"type": "string", "enum": ["ask_again", "alert_caregiver", "dashboard"]},
        "about": ABOUT,
    },
    ["kind", "title"],
)
async def open_loop_tool(ctx: TurnCtx, a: dict) -> dict:
    wake = clock.now() + timedelta(minutes=a["check_back_in_minutes"]) if a.get("check_back_in_minutes") else None
    loop = await store.open_loop(
        ctx.session, family_id=ctx.family_id, subject_id=ctx.subject(a.get("about")), kind=a["kind"], title=a["title"],
        owner_id=ctx.speaker.get("id"), wake_at=wake, alert_rule=a.get("if_no_answer"),
        dedupe_key=f"{a['kind']}:{slug(a['title'])[:80]}",
    )
    return {"loop_id": str(loop.id), "wake_at": clock.ist(wake).strftime("%H:%M") if wake else None}


@tool(
    "close_loop",
    "Close an open loop when it is answered, done, or no longer needed.",
    {"loop_id": {"type": "string"}, "outcome": {"type": "string"}},
    ["loop_id", "outcome"],
)
async def close_loop_tool(ctx: TurnCtx, a: dict) -> dict:
    try:
        loop_id = uuid.UUID(a["loop_id"])
    except ValueError as exc:
        raise ToolRefused("loop_id must be the id shown in OPEN LOOPS") from exc
    loop = await store.close_loop(ctx.session, loop_id, note=a["outcome"])
    return {"closed": bool(loop)}


@tool(
    "send_message",
    "Send a WhatsApp message to someone in the family other than the person you are replying to: pass a message "
    "on to the elder from a caregiver, or, on a scheduled wake-up, start the follow-up you promised. Your normal "
    "reply already goes to the speaker; do not use this for that.",
    {"to": {"type": "string", "description": "Person id from HOUSEHOLD"}, "text": {"type": "string"}},
    ["to", "text"],
)
async def send_message(ctx: TurnCtx, a: dict) -> dict:
    ids = {m.get("id") for m in ctx.members} | {ctx.elder_id}
    if a["to"] not in ids:
        raise ToolRefused("Send only to people in HOUSEHOLD, by id.")
    if a["to"] == ctx.speaker.get("id"):
        raise ToolRefused("Your reply already goes to the speaker.")
    res = await ctx.host.call(
        "send_whatsapp", {"to": a["to"], "text": a["text"]},
        family_id=ctx.family_id.removeprefix("shadow:"), subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )
    await store.add_turn(
        ctx.session, family_id=ctx.family_id, thread_id=a["to"], role="assistant", text=a["text"],
        meta={"proactive": True, "delivered": res.get("delivered")},
    )
    return res


# ── alerts ─────────────────────────────────────────────────────────────────────


@tool(
    "alert_caregiver",
    "Tell the caregivers. WhatsApp goes out only for: red_flag (medical emergency or red-flag reading), "
    "no_answer (the elder did not answer a check that matters), safety (high-confidence mood or safety concern: "
    "confusion, a scam, an unusual or bulk order), approval (an order or booking needs their OK). Anything else "
    "goes to the dashboard. issue names the problem so the same issue is sent once a day.",
    {
        "reason": {"type": "string", "enum": sorted(policy.ALERT_REASONS)},
        "issue": {"type": "string", "description": "Short stable name, e.g. 'fall-bathroom', 'bp-high', 'scam-call'"},
        "message": {"type": "string", "description": "What the caregiver reads: what happened, what you did, what they should do"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    ["reason", "issue", "message", "confidence"],
)
async def alert_caregiver(ctx: TurnCtx, a: dict) -> dict:
    decision = policy.alert_decision(a["reason"], float(a["confidence"]))
    ref = policy.issue_ref(a["reason"], a["issue"], clock.ist_day())
    fresh = await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id,
        kind="alert_whatsapp" if decision.whatsapp else "alert_dashboard",
        summary=a["message"], payload={**a, "why": decision.why}, actor_id=ctx.speaker.get("id"), ref=ref,
    )
    if not fresh:
        return {"sent": False, "why": "already sent for this issue today"}
    ctx.alerts.append({**a, "whatsapp": decision.whatsapp})
    kind = {"red_flag": "health_red_flag", "no_answer": "nudge_silence", "safety": "unusual_activity", "approval": "order_placed"}.get(
        a["reason"], "care_note"
    )
    if a["reason"] == "red_flag":
        await ctx.host.call(
            "trigger_emergency_escalation", {"message": a["message"]},
            family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
        )
    res = await ctx.host.call(
        "notify_caregivers",
        {"message": a["message"], "urgency": "high" if a["reason"] == "red_flag" else "medium", "kind": kind if decision.whatsapp else "care_note"},
        family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )
    return {"sent": True, "whatsapp": decision.whatsapp, "why": decision.why, "result": res}


# ── records ────────────────────────────────────────────────────────────────────


@tool(
    "search_records",
    "Search uploaded medical records and lab reports (prescriptions, blood tests, discharge summaries).",
    {"query": {"type": "string"}},
    ["query"],
)
async def search_records(ctx: TurnCtx, a: dict) -> dict:
    return await ctx.host.call(
        "search_lab_reports", {"query": a["query"]}, family_id=ctx.family_id, subject_id=ctx.elder_id,
        actor_id=ctx.actor_id,
    )


@tool(
    "lab_trend",
    "Values of one lab marker over time (HbA1c, creatinine, TSH, haemoglobin, …).",
    {"marker": {"type": "string"}},
    ["marker"],
)
async def lab_trend(ctx: TurnCtx, a: dict) -> dict:
    return await ctx.host.call(
        "get_lab_trends", {"marker": a["marker"]}, family_id=ctx.family_id, subject_id=ctx.elder_id,
        actor_id=ctx.actor_id,
    )


# ── orders and rides (until the task runtime replaces them) ────────────────────


async def _care_limits(ctx: TurnCtx) -> tuple[list[str], list[str]]:
    rows = await store.facts(ctx.session, ctx.family_id, ctx.elder_id, domains=["allergy", "no_order"], statuses=("active",))
    allergies = [r.value.get("allergen") or r.key.split(":", 1)[1] for r in rows if r.domain == "allergy"]
    never = [r.value.get("item") or r.key.split(":", 1)[1].replace("_", " ") for r in rows if r.domain == "no_order"]
    return allergies, never


@tool(
    "order",
    "Order groceries, medicines or food for the care recipient (cash on delivery, the family's own accounts). "
    "goal says exactly what to buy and from where if they said. Only call after the person confirmed the items. "
    "Allergies and the never-order list are checked first.",
    {
        "goal": {"type": "string", "description": "e.g. '2 packets Aashirvaad atta 5kg from Instamart'"},
        "confirmed": {"type": "boolean", "description": "The person confirmed items and quantity"},
    },
    ["goal", "confirmed"],
)
async def order(ctx: TurnCtx, a: dict) -> dict:
    allergies, never = await _care_limits(ctx)
    conflicts = policy.order_conflicts(a["goal"], allergies, never)
    if conflicts:
        raise ToolRefused(f"Blocked by the care record: {', '.join(conflicts)}. Tell them kindly and offer something safe.")
    return await ctx.host.call(
        "browser_order", {"goal": a["goal"], "message": a["goal"], "userConfirmed": bool(a["confirmed"])},
        family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )


@tool(
    "ride",
    "Book, check or cancel a cab or auto for the care recipient.",
    {
        "action": {"type": "string", "enum": ["book", "status", "cancel"]},
        "pickup": {"type": "string"},
        "drop": {"type": "string"},
        "confirmed": {"type": "boolean"},
    },
    ["action"],
)
async def ride(ctx: TurnCtx, a: dict) -> dict:
    tool_name = {"book": "book_ride", "status": "ride_status", "cancel": "cancel_ride"}[a["action"]]
    args = {"pickup": a.get("pickup"), "drop": a.get("drop"), "userConfirmed": bool(a.get("confirmed"))} if a["action"] == "book" else {}
    return await ctx.host.call(
        tool_name, args, family_id=ctx.family_id, subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )
