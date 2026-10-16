"""Care views: refills, emergency card, care team, reports, wellbeing, family tasks, spending."""

from __future__ import annotations

import os

from app.brain.tools.base import ABOUT, ABOUT_OWN, ToolRefused, TurnCtx, tool
from app.brain.tools.health import _not_theirs
from app.care import store
from app.care.domains import slug
from app.core import clock

# ── care views: refills, emergency card, care team, reports, wellbeing, family tasks, spending ──
# Each reads the same data the dashboard shows (app.care.features), so WhatsApp and the dashboard agree.

DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://app.kavach.care").rstrip("/")


def _name_of(ctx: TurnCtx, subject: str) -> str:
    if subject == ctx.elder_id:
        return ctx.elder.get("name") or "them"
    for m in ctx.members:
        if m.get("id") == subject:
            return m.get("name") or "them"
    return "them"


@tool(
    "set_stock",
    "Call this whenever someone says how many of a medicine are on hand, or that a new strip, bottle or pack has "
    "arrived ('10 left', 'naya patta aaya', a new bottle). count is tablets or doses on hand now. If they name a new "
    "strip or bottle but no number of tablets, ask how many are in it and then call this; do not guess a pack size. "
    "Saheli then works out days left and can offer a reorder before it runs out.",
    {"medicine": {"type": "string"}, "count": {"type": "integer", "minimum": 0, "maximum": 2000}, "about": ABOUT_OWN},
    ["medicine", "count"],
)
async def set_stock_tool(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    subject = ctx.own(a.get("about"))
    await _not_theirs(ctx, a, subject, a["medicine"])
    f = await features.find_medicine(ctx.session, ctx.family_id, subject, a["medicine"])
    if not f:
        raise ToolRefused("That medicine is not in the care record. Save it with remember first.")
    return await features.set_stock(
        ctx.session, family_id=ctx.family_id, subject_id=subject, key=f.key, name=features.med_name(f), count=int(a["count"]),
        actor_id=ctx.speaker.get("id"),
    )


@tool(
    "medicine_stock",
    "Tablets left and days left for each medicine, and which ones need a refill soon.",
    {"about": ABOUT},
    [],
)
async def medicine_stock(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    rows = await features.stock(ctx.session, ctx.family_id, ctx.subject(a.get("about")))
    return {"medicines": rows, "refill_within_days": features.REFILL_DAYS, "unknown": [r["name"] for r in rows if r["stock"] is None]}


@tool(
    "emergency_card",
    "The person's emergency card (blood group, allergies, conditions, medicines, doctors, hospital, who to call) "
    "and a link anyone can open in an emergency. Send it when someone asks for it or there is an emergency.",
    {"about": ABOUT},
    [],
)
async def emergency_card(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    subject = ctx.subject(a.get("about"))
    card = await features.emergency(ctx.session, ctx.family_id, subject)
    link = await ctx.host.call("emergency_link", {}, family_id=ctx.family_id.removeprefix("shadow:"), subject_id=subject, actor_id=ctx.actor_id)
    missing = [k for k in ("blood_group",) if k not in card["profile"]]
    return {"text": features.emergency_text(_name_of(ctx, subject), card), "link": link.get("url"), "missing": missing}


@tool(
    "care_team",
    "Doctors, hospital, contacts, helpers and appointments (upcoming and past, with the questions to ask the doctor).",
    {"about": ABOUT},
    [],
)
async def care_team_tool(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    return await features.care_team(ctx.session, ctx.family_id, ctx.subject(a.get("about")))


@tool(
    "add_doctor_question",
    "Add a question to ask the doctor at an appointment (saved with the appointment and shown on the report).",
    {"appointment": {"type": "string", "description": "Appointment name or key, e.g. the doctor's name"}, "question": {"type": "string"}, "about": ABOUT_OWN},
    ["appointment", "question"],
)
async def add_doctor_question(ctx: TurnCtx, a: dict) -> dict:
    subject = ctx.own(a.get("about"))
    appts = await store.facts(ctx.session, ctx.family_id, subject, domains=["appointment"], statuses=("active",))
    want = slug(a["appointment"].split(":", 1)[-1])
    hit = next((f for f in appts if f.key.split(":", 1)[1] == want or want in f.key or want in slug(str(f.value.get("doctor", "")))), None)
    if not hit:
        raise ToolRefused("No such appointment. Save the appointment with remember (domain appointment) first.")
    qs = [*(hit.value.get("questions") or []), a["question"].strip()]
    w = await store.write_fact(
        ctx.session, family_id=ctx.family_id, subject_id=subject, domain="appointment", key=hit.key, value={**hit.value, "questions": qs},
        text=hit.text, source_kind=ctx.source_kind, source_ref=ctx.message_ref, stated_by=ctx.speaker.get("id"),
    )
    return {"result": w.result, "questions": qs}


@tool(
    "weekly_report",
    "A care report for the last N days (default 7) to share with the doctor or family: adherence, readings, "
    "symptoms, mood, alerts and orders, with a link to the printable report. It also groups problems that came back in "
    "the last 3 months (three falls with each one's date and circumstances) with a question for the doctor: use it when "
    "they prepare for a doctor visit or ask what to tell the doctor.",
    {"days": {"type": "integer", "minimum": 1, "maximum": 31}, "about": ABOUT},
    [],
)
async def weekly_report(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features, symptom_patterns

    subject = ctx.subject(a.get("about"))
    r = await features.report(ctx.session, ctx.family_id, subject, int(a.get("days") or 7))
    r["patterns"] = await symptom_patterns.patterns(ctx.session, ctx.family_id, subject)
    return {
        "summary": features.report_facts_text(_name_of(ctx, subject), r),
        "link": f"{DASHBOARD_URL}/dashboard/report?recipient={subject}&days={r['days']}",
    }


@tool(
    "wellbeing",
    "How the person has been lately: when they were last heard from, how many days they talked, mood and concerns.",
    {"days": {"type": "integer", "minimum": 3, "maximum": 60}, "about": ABOUT},
    [],
)
async def wellbeing_tool(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    subject = ctx.subject(a.get("about"))
    w = await features.wellbeing(ctx.session, ctx.family_id, subject, int(a.get("days") or 14))
    return {"summary": features.wellbeing_text(_name_of(ctx, subject), w), "streak": w["streak"], "lastHeard": w["lastHeard"]}


@tool(
    "assign_family_task",
    "Give a family member a care task (call Maa tonight, pick up medicines, take Papa to the clinic). Saheli "
    "reminds them on WhatsApp when it is due. 'to' is a person id from HOUSEHOLD; due is 'YYYY-MM-DDTHH:MM' IST.",
    {"title": {"type": "string"}, "to": {"type": "string"}, "due": {"type": "string"}, "about": ABOUT},
    ["title", "to"],
)
async def assign_family_task(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    ids = {m.get("id") for m in ctx.members}
    if a["to"] not in ids:
        raise ToolRefused("Assign only to people in HOUSEHOLD, by id.")
    due = features.parse_when(a.get("due"))
    loop = await features.add_family_task(
        ctx.session, family_id=ctx.family_id, subject_id=ctx.subject(a.get("about")), title=a["title"], assignee=a["to"],
        due=due, by=ctx.speaker.get("id"),
    )
    return {"task_id": str(loop.id), "due": clock.ist(due).strftime("%d %b %H:%M") if due else None, "done_with": "close_loop"}


@tool(
    "family_tasks",
    "The family's care tasks: who has what, when it is due, and what was done recently. Close one with close_loop.",
    {},
    [],
)
async def family_tasks_tool(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    return {"tasks": await features.family_tasks(ctx.session, ctx.family_id)}


@tool(
    "spending",
    "What was spent on orders and rides Saheli placed in a month (cash on delivery), by service and by person.",
    {"month": {"type": "string", "description": "YYYY-MM, default this month"}},
    [],
)
async def spending_tool(ctx: TurnCtx, a: dict) -> dict:
    from app.care import features

    return await features.spending(ctx.session, ctx.family_id, a.get("month"))


@tool(
    "past_orders",
    "Orders and rides Saheli placed for this family before: the exact items, the service, the total and when. "
    "Use it for 'the usual', 'same as last time', or to answer what was ordered.",
    {"item": {"type": "string", "description": "Optional word to filter by, e.g. 'atta' or 'Telma'"}, "limit": {"type": "integer"}},
    [],
)
async def past_orders(ctx: TurnCtx, a: dict) -> dict:
    from sqlalchemy import select

    from app.tasks.models import Task

    rows = (await ctx.session.execute(
        select(Task).where(Task.family_id == ctx.family_id, Task.status == "done").order_by(Task.created_at.desc()).limit(60)
    )).scalars()
    word = (a.get("item") or "").lower()
    out = []
    for t in rows:
        r = t.result or {}
        if not (r.get("placed") or r.get("booked")) or t.cancel_requested:
            continue
        items = [f"{i.get('qty', 1)} x {i.get('name')}" for i in (r.get("items") or (t.details or {}).get("items") or [])]
        if word and word not in (" ".join(items) + " " + t.goal).lower():
            continue
        fb = r.get("feedback") or {}
        liked = {"yes": "liked", "no": "did not like", "mixed": "mixed"}
        said = [f"{liked[v['liked']]} item {n}" + (f" ({v['note']})" if v.get("note") else "")
                for n, v in (fb.get("items") or {}).items() if v.get("liked") in liked]
        out.append({"when": clock.ist(t.created_at).strftime("%d %b %Y"), "service": t.service, "kind": t.kind, "items": items,
                    "total": r.get("total") or r.get("fare"), "for": t.subject_id, **({"they_said": "; ".join(said)} if said else {})})
        if len(out) >= int(a.get("limit") or 10):
            break
    return {"orders": out} if out else {"orders": [], "note": "Nothing placed before; ask what they want, with brand and size."}


@tool(
    "patterns",
    "Patterns noticed in the last two weeks without anyone asking: dose times that keep slipping, readings drifting, "
    "a complaint that keeps coming back, low moods, skipped meals, going quiet, late nights. Use it when someone asks "
    "'how has she been?', 'anything I should know?', or before the weekly check-in. Facts from the logs, not diagnoses.",
    {"about": ABOUT},
    [],
)
async def patterns_tool(ctx: TurnCtx, a: dict) -> dict:
    from app.care import patterns

    found = await patterns.find(ctx.session, ctx.family_id, ctx.subject(a.get("about")))
    if not found:
        return {"patterns": [], "note": "Nothing stands out in the last two weeks."}
    return {"patterns": [{"title": p.title, "detail": p.detail, "suggestion": p.suggestion, "level": p.severity} for p in found]}
