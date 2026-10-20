"""Health tools: doses, readings and events of the day, alerts to the family and the care team, records."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta

from app.brain import guards, policy
from app.brain.tools.base import ABOUT_OWN, ToolRefused, TurnCtx, tool
from app.care import store
from app.care.domains import slug
from app.core import clock

# ── the day's care ─────────────────────────────────────────────────────────────


@tool(
    "fact_still_true",
    "The person confirmed that a saved fact is still right (for one marked 'check it is still right'): it is fresh again. "
    "If it changed, use remember or stop instead.",
    {"key": {"type": "string", "description": "The fact's key from the CARE RECORD, e.g. medicine:shelcal"}, "about": ABOUT_OWN},
    ["key"],
)
async def fact_still_true(ctx: TurnCtx, a: dict) -> dict:
    from app.care import freshness

    subject = ctx.own(a.get("about"))
    row = await store.active_fact(ctx.session, ctx.family_id, subject, a["key"])
    if not row:
        raise ToolRefused(f"No active fact {a['key']} for this person.")
    await freshness.confirm(ctx.session, family_id=ctx.family_id, subject_id=subject, key=row.key, by=ctx.speaker.get("id"),
                            summary=f"{row.text}: still right")
    return {"confirmed": row.key}


@tool(
    "note_habit",
    "The person stated when they usually eat, drink water, or wake, or asked not to be asked about one of these. "
    "Save it at once. This is not a meal they just had (that is log_food) and not a medicine.",
    {
        "what": {"type": "string", "enum": ["breakfast", "lunch", "snack", "dinner", "water", "sleep"]},
        "at": {"type": "string", "description": "Usual time HH:MM, 24-hour. Leave out when they asked not to be asked."},
        "skip": {"type": "boolean", "description": "True when they asked not to be asked about this."},
        "about": ABOUT_OWN,
    },
    ["what"],
)
async def note_habit(ctx: TurnCtx, a: dict) -> dict:
    from app.care import meal_nudges

    subject = ctx.own(a.get("about"))
    try:
        result = await meal_nudges.state_habit(ctx.session, ctx.family_id, subject, what=str(a.get("what") or ""),
                                                at=a.get("at"), skip=bool(a.get("skip")))
    except ValueError as e:
        raise ToolRefused(str(e)) from e
    return {"saved": result, "what": a.get("what"), "at": a.get("at"), "skip": bool(a.get("skip"))}


async def _not_theirs(ctx: TurnCtx, a: dict, subject: str, medicine: str) -> None:
    """A medicine saved with no about that the speaker does not take: say who takes it, and let the model read from the
    message whose dose or stock it is (saves default to the speaker's own, founder 2026-10-16)."""
    from app.care import features

    if str(a.get("about") or "").strip() or subject == ctx.elder_id or ctx.is_system:
        return
    if await features.find_medicine(ctx.session, ctx.family_id, subject, medicine):
        return
    owners = []
    for p in [ctx.elder, *ctx.members]:
        pid = p.get("id")
        if pid and pid != subject and await features.find_medicine(ctx.session, ctx.family_id, pid, medicine):
            owners.append(f"{p.get('name') or pid} (id {pid})")
    if owners:
        raise ToolRefused(f"Nothing saved: {ctx.member_name(subject)}, who is writing, has no {medicine} in their care record; "
                          f"{', '.join(owners)} has it. If the message is about that person's medicine, call again with about = "
                          "their id.")


@tool(
    "log_dose",
    "Record what happened with a medicine dose: taken, skipped, refused, missed, or the strip is empty. Works for today or "
    "an earlier day (day), and for one dose of a medicine taken more than once a day (time). Logging the same dose again "
    "replaces the earlier answer (a correction). not_taken unmarks it: they have not taken it yet (marked by mistake, or "
    "for the wrong day), so it shows as still to take.",
    {
        "medicine": {"type": "string", "description": "Medicine name or key"},
        "outcome": {"type": "string", "enum": ["taken", "skipped", "refused", "missed", "empty_strip", "not_taken"]},
        "day": {"type": "string", "description": "today (default), yesterday, or YYYY-MM-DD within the last 7 days"},
        "time": {"type": "string", "description": "The dose time (HH:MM) when the medicine is taken more than once a day"},
        "note": {"type": "string"},
        "about": ABOUT_OWN,
    },
    ["medicine", "outcome"],
)
async def log_dose(ctx: TurnCtx, a: dict) -> dict:
    from app.care import doses

    subject = ctx.own(a.get("about"))
    name = a["medicine"].split(":", 1)[-1].replace("_", " ")
    await _not_theirs(ctx, a, subject, a["medicine"])
    day = doses.resolve_day(a.get("day"))
    if not day:
        raise ToolRefused("day must be today, yesterday or a date in the last 7 days.")
    today = day == clock.ist_day()
    fact = await doses.medicine_fact(ctx.session, ctx.family_id, subject, name)
    scheduled = [str(t) for t in ((fact.value or {}).get("times") or []) if isinstance(t, str)] if fact else []
    not_due = bool(fact and not doses.due_on(fact.value, day))
    if not_due and a["outcome"] != "taken":
        # live 2026-10-09: "I took nothing today" logged the weekly Vitamin D3 as skipped on a day it was not due
        raise ToolRefused(f"{name} is taken only on {doses.day_label(fact.value.get('days'))}; it was not due on {day}, "
                          "so there is nothing to mark. Log only the medicines due that day.")
    first = slug(name).split("_")[0]
    earlier = []
    if a["outcome"] == "taken" and first and today:
        recent = await store.events(ctx.session, ctx.family_id, subject, since=clock.now() - timedelta(hours=3), kinds=["dose_taken"])
        earlier = [e for e in recent if first in (e.summary or "").lower() or first in json.dumps(e.payload or {}).lower()]
    summary = f"{name}: {a['outcome']}" + ("" if today else f" (for {day})") + (f" ({a['note']})" if a.get("note") else "")
    saved = await doses.record(
        ctx.session, family_id=ctx.family_id, subject_id=subject, kind=f"dose_{a['outcome']}", medicine=name, summary=summary,
        day=day, time=a.get("time"), actor_id=ctx.speaker.get("id"), ref=ctx.message_ref and f"{ctx.message_ref}:{slug(name)}:{day}",
    )
    if saved is None:
        return {"logged": summary, "already": True, "medicine": name, "scheduled": scheduled}
    if a["outcome"] == "not_taken":
        res = await ctx.host.call(
            "clear_schedule_completion", {"title": name, "dateKey": day, **({"time": a["time"]} if a.get("time") else {})},
            family_id=ctx.family_id, subject_id=subject, actor_id=ctx.actor_id,
        )
        return {"unmarked": f"{name} for {day}: not taken yet", "schedule": res, "medicine": name, "scheduled": scheduled}
    tool_name = "mark_schedule_completed" if a["outcome"] == "taken" else "mark_schedule_missed"
    res = await ctx.host.call(
        tool_name, {"title": name, "dateKey": day, "note": a.get("note") or a["outcome"], **({"time": a["time"]} if a.get("time") else {})},
        family_id=ctx.family_id, subject_id=subject, actor_id=ctx.actor_id,
    )
    out = {"logged": summary, "schedule": res, "medicine": name, "scheduled": scheduled}
    if not_due:
        out["not_due"] = (f"{name} is taken only on {doses.day_label(fact.value.get('days'))}, not on {day}. Ask gently whether "
                          "it was taken early by mistake; if it was, tell the caregivers (alert_caregiver safety) instead of advising.")
    if earlier:
        out["possible_double_dose"] = (
            f"{name} was already logged as taken at {clock.ist(earlier[-1].at).strftime('%H:%M')}. Ask gently whether this is a second "
            "tablet. If it really was taken twice, tell the caregivers (alert_caregiver: red_flag for blood thinners, insulin, sugar or BP "
            "medicines, otherwise safety) and do not advise on dosing yourself."
        )
    if a["outcome"] == "empty_strip":
        out["next"] = "Ask how many tablets are left or whether to reorder (set_stock, then start_task with a pharmacy)."
    if a["outcome"] in ("skipped", "refused"):
        out["next"] = skip_next(ctx, subject, name, a.get("note") or "")
    return out


def _cared_for(ctx: TurnCtx, subject: str) -> bool:
    """A care recipient's medicine (family is told of a refusal), not a caregiver's own (self-care: only recorded)."""
    if subject == ctx.elder_id:
        return True
    role = next((str(m.get("role") or "") for m in ctx.members if m.get("id") == subject), "")
    return "caregiver" not in role.lower()


def skip_next(ctx: TurnCtx, subject: str, name: str, said_reason: str) -> str:
    """A skip is a conversation, not only a record (founder 20 Oct): why, then the right next step, for any medicine."""
    family = ("If they still do not want to take it, tell the caregivers once (alert_caregiver reason safety, issue "
              f"'{name[:30]}-not-taken', with the reason they gave)." if _cared_for(ctx, subject) else
              "If they still do not want to take it, it stays recorded as skipped with their reason; nobody else is told "
              "(it is their own medicine).")
    known = f" They said: {said_reason[:120]}." if said_reason and said_reason not in ("skipped", "refused") else ""
    return (f"Recorded {name} as not taken.{known} Do not stop at recording it: unless their words already gave the reason, ask "
            "kindly why, in one short question (did the doctor say to stop or pause it, a side effect, run out, feeling unwell, "
            "or just not feeling like it). When they answer, log_dose again with the same outcome and their reason as note. "
            "A doctor's instruction or a side effect: accept it, never argue; a side effect is worth telling their doctor. "
            "Run out: offer to reorder. Just not feeling like it: encourage once, warmly and briefly (it was prescribed for "
            "them; you will remind them gently next time), never pressure, never medical advice. If they then take it, log "
            f"it as taken. {family}")


VITAL_KINDS = ["bp", "sugar", "weight", "height", "bmi", "body_fat", "muscle_mass", "waist", "temperature", "spo2", "pulse", "other"]


@tool(
    "log_vital",
    "Record a reading; every reading reaches their Health Records. bp ('140/90'), sugar (mg/dL, say fasting or after food "
    "in note), weight (kg), height (cm), bmi, body_fat (%), muscle_mass (kg), waist (cm), temperature (°F or °C), spo2 "
    "(%), pulse; anything else as other with its name. Typed, said, or seen in a photo (a scale, BP monitor, glucometer, "
    "an InBody or body-composition scan): one call per value, each of them, with taken_on when it was measured on another "
    "day. Returns red_flag when the reading needs a caregiver now.",
    {
        "kind": {"type": "string", "enum": VITAL_KINDS},
        "name": {"type": "string", "description": "For other: what was measured (e.g. visceral fat, HbA1c)"},
        "value": {"type": "string"},
        "unit": {"type": "string"},
        "taken_on": {"type": "string", "description": "YYYY-MM-DD when measured, if not today"},
        "note": {"type": "string"},
        "about": ABOUT_OWN,
    },
    ["kind", "value"],
)
async def log_vital(ctx: TurnCtx, a: dict) -> dict:
    subject = ctx.own(a.get("about"))
    a = {k: v for k, v in a.items() if k != "about" and v not in (None, "")}
    if a["kind"] == "other" and not str(a.get("name") or "").strip():
        raise ToolRefused("Say what was measured in name (e.g. visceral fat).")
    at = None
    if a.get("taken_on"):
        try:
            day = datetime.strptime(str(a["taken_on"])[:10], "%Y-%m-%d").date()
        except ValueError:
            raise ToolRefused("taken_on must be YYYY-MM-DD.")
        if day > clock.ist().date():
            raise ToolRefused("taken_on is in the future.")
        if day < clock.ist().date():
            at = datetime.combine(day, time(12, 0), tzinfo=clock.IST)
    flag = policy.vital_red_flag(a["kind"], a["value"]) if not at else None  # an old reading is history, not an emergency now
    label = a["name"].strip() if a["kind"] == "other" else a["kind"]
    summary = f"{label} {a['value']}{(' ' + a['unit']) if a.get('unit') else ''}" + (f" ({a['note']})" if a.get("note") else "")
    await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=subject, kind="vital", summary=summary,
        payload={**a, "red_flag": flag}, actor_id=ctx.speaker.get("id"), at=at,
    )
    await ctx.host.call(
        "log_vitals", {"kind": a["kind"], "name": a.get("name"), "value": a["value"], "unit": a.get("unit"), "note": a.get("note"),
                       "taken_on": at.date().isoformat() if at else None},
        family_id=ctx.family_id, subject_id=subject, actor_id=ctx.actor_id,
    )
    return {"logged": summary, "red_flag": flag, "health_records": "saved"}


@tool(
    "log_event",
    "Record anything else about the day for the dashboard and daily snapshot: a meal, missed meal or water only for someone "
    "whose food tracking is off (otherwise log_food; never to record that they agreed with a meal already saved), "
    "mood, a symptom (pain, dizziness, swelling, constipation, wound, up at night, did not get out of bed), "
    "a routine done or missed, a home problem (helper did not come, power or water off, cylinder low, locked out), "
    "a call with family. Mood: whenever how they feel shows (they say it, or it is clear from how they write), give "
    "mood_score and mood_word, with kind mood or with the event that shows it (a visit that made them happy: kind social "
    "plus the mood); once per conversation, and again when it changes.",
    {
        "kind": {"type": "string", "enum": ["meal", "water", "mood", "symptom", "routine", "home", "sleep", "social", "other"]},
        "summary": {"type": "string"},
        "severity": {"type": "string", "enum": ["info", "watch", "concern"]},
        "mood_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "How they feel: 1 very low, 2 low, 3 okay, 4 good, 5 very good"},
        "mood_word": {"type": "string", "description": "How they feel in one English word for the dashboard, e.g. cheerful, calm, tired, "
                                                       "worried, sad, irritated, lonely, unwell"},
        "meal_skipped": {"type": "boolean", "description": "kind meal: they skipped the meal or did not eat"},
        "about": ABOUT_OWN,
    },
    ["kind", "summary"],
)
async def log_event(ctx: TurnCtx, a: dict) -> dict:
    subject = ctx.own(a.get("about"))
    from app.care import mood

    score = mood.score_of(a.get("mood_score"))
    if a["kind"] == "mood" or score:
        word, summary = str(a.get("mood_word") or ""), a["summary"]
        # read again from the whole conversation and the problems they reported, not from this message alone
        name = next((m.get("name") for m in [ctx.elder, *ctx.members] if m.get("id") == subject), "") or ""
        seen = await mood.assess(ctx.session, ctx.family_id, subject, name=name, hint={"score": score, "word": word, "summary": summary})
        if seen:
            score, word, summary = seen["score"], seen["word"], seen["summary"]
        new = await mood.record(ctx.session, family_id=ctx.family_id, subject_id=subject, score=score, word=word,
                                summary=summary, severity=a.get("severity") if a["kind"] == "mood" else None, actor_id=ctx.speaker.get("id"))
        if a["kind"] == "mood":
            return {"logged": summary, "word": word} if new else {"logged": False, "why": "the same mood was noted a little while ago"}
    if a["kind"] in ("meal", "water"):
        from app.nutrition import settings as food_settings

        if food_settings.tracked(await food_settings.get(ctx.session, ctx.family_id, subject)):
            return {"logged": False, "why": "Food tracking is on. A meal or water is log_food, and only when they named "
                    "what they ate or drank. Agreeing with a meal already saved is not logged again. Say nothing about logging it."}
    await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=subject, kind=a["kind"], summary=a["summary"],
        payload={"severity": a.get("severity", "info"), **({"meal_skipped": True} if a["kind"] == "meal" and a.get("meal_skipped") else {})},
        actor_id=ctx.speaker.get("id"),
    )
    if a["kind"] == "symptom":
        await ctx.host.call(
            "log_symptom", {"symptom": a["summary"], "severity": a.get("severity", "info")},
            family_id=ctx.family_id, subject_id=subject, actor_id=ctx.actor_id,
        )
    return {"logged": a["summary"]}


# ── alerts ─────────────────────────────────────────────────────────────────────


async def _follow_up_alert(ctx: TurnCtx, a: dict) -> None:
    """Next morning, ask a caregiver how things turned out (one tap: all fine / saw doctor / hospital)."""
    # never the person the alert was about (caring for themselves, they are their own primary caregiver; founder
    # 2026-10-13: "he can not be caregiver for himself")
    others = [m for m in ctx.members if m.get("id") != ctx.elder_id and m.get("role") not in ("elder", "family doctor")]
    owner = next((m.get("id") for m in others if "primary" in str(m.get("role", "")).lower()), None) or next(
        (m.get("id") for m in others), None)
    if not owner:
        return
    key = f"alert:{policy.issue_ref(a['reason'], a['issue'], clock.ist_day()).split(':', 2)[2]}:subj={ctx.elder_id}"
    from app.learn import timing

    tomorrow = clock.ist().replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=1)
    wake = await timing.next_send_at(ctx.session, ctx.family_id, owner, "followup", default_hour=10, earliest=tomorrow)
    await store.open_loop(
        ctx.session, family_id=ctx.family_id, subject_id=ctx.elder_id, kind="followup",
        title=f"Yesterday's alert ({a['issue']}): {a['message'][:140]}. Ask how {ctx.elder.get('name', 'they')} is now.",
        detail={"key": key, "set": "outcome", "max_wakes": 1}, owner_id=owner, wake_at=wake, alert_rule="dashboard",
        dedupe_key=f"followup:{key}",
    )


@tool(
    "alert_caregiver",
    "Tell the caregivers. WhatsApp goes out only for: red_flag (medical emergency or red-flag reading), "
    "no_answer (the elder did not answer a check that matters), safety (high-confidence mood or safety concern: "
    "confusion, a scam, an unusual or bulk order), approval (an order or booking needs their OK), crisis (any talk of "
    "suicide, self-harm, wanting to die, or harming or killing someone, even if it may be a joke or a test: the Kavach "
    "care team is emailed at once too). Else the dashboard. issue names the problem so the same issue "
    "is sent once a day. Never sent to the person it is about; say only who the result says was told.",
    {
        "reason": {"type": "string", "enum": sorted(policy.ALERT_REASONS)},
        "issue": {"type": "string", "description": "Short stable name, e.g. 'fall-bathroom', 'bp-high', 'scam-call'"},
        "message": {"type": "string", "description": "What the caregiver reads: what happened, what you did, what they should do"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "about": {"type": "string", "description": "HOUSEHOLD id, blank: writer"},
    },
    ["reason", "issue", "message", "confidence"],
)
async def alert_caregiver(ctx: TurnCtx, a: dict) -> dict:
    # Whose emergency it is: the person named, else the one writing (live 19 Oct: a caregiver's own message was filed
    # and sent as "Emergency alert from care recipient" about his mother). A scheduled turn keeps the care recipient.
    who = ctx.own(a.get("about"))
    if ctx.told_by_tasks and a["reason"] != "red_flag":
        # cancel_task in this turn: the task system tells the caregivers about the order (one message each)
        raise ToolRefused("Not sent: the task system is telling the caregivers about this order right now, with what to do.")
    if a["reason"] == "red_flag" and ctx.user_text:
        why = guards.red_flag_unsupported(ctx.user_text)
        if why:
            # Not an emergency: it goes to the dashboard as a note, never as a WhatsApp emergency.
            await store.record_event(
                ctx.session, family_id=ctx.family_id, subject_id=who, kind="alert_dashboard", summary=a["message"],
                payload={**a, "why": why, "downgraded": True}, actor_id=ctx.speaker.get("id"),
                ref=policy.issue_ref("note", a["issue"], clock.ist_day()),
            )
            return {"sent": False, "whatsapp": False, "dashboard": True, "why": why}
    decision = policy.alert_decision(a["reason"], float(a["confidence"]))
    ref = policy.issue_ref(a["reason"], a["issue"], clock.ist_day())
    fresh = await store.record_event(
        ctx.session, family_id=ctx.family_id, subject_id=who,
        kind="alert_whatsapp" if decision.whatsapp else "alert_dashboard",
        summary=a["message"], payload={**a, "why": decision.why}, actor_id=ctx.speaker.get("id"), ref=ref,
    )
    if not fresh:
        if a["reason"] == "crisis":
            # every line of a crisis reaches the Kavach care team (batched into one update), even under an issue the
            # family was told about today
            from app.care import careteam

            team = await careteam.notify(ctx.session, ctx.host, family_id=ctx.family_id, subject_id=who, kind="crisis",
                                         summary=a["message"], said=ctx.user_text, actor_id=ctx.speaker.get("id"))
            return {"sent": False, "why": "the family was told about this today already",
                    "told": guards.who_was_told(ctx.profiles.get(ctx.speaker.get("id")), team=team, already_today=True)}
        return {"sent": False, "why": "already sent for this issue today"}
    ctx.alerts.append({**a, "whatsapp": decision.whatsapp})
    kind = {"red_flag": "health_red_flag", "no_answer": "nudge_silence", "safety": "unusual_activity", "approval": "order_placed",
            "crisis": "crisis"}.get(a["reason"], "care_note")
    crisis = a["reason"] == "crisis"
    if a["reason"] == "red_flag":
        # notify false: the caregivers get one WhatsApp, the notify_caregivers below (live 19 Oct: two for one red flag)
        await ctx.host.call(
            "trigger_emergency_escalation", {"message": a["message"], "notify": False},
            family_id=ctx.family_id, subject_id=who, actor_id=ctx.actor_id,
        )
    if decision.whatsapp and a["reason"] in ("red_flag", "safety", "crisis"):
        await _follow_up_alert(ctx, a)
    # A crisis over many messages is one WhatsApp to the family per half hour (the team gets every line by email): live
    # 2026-10-13 five alerts went out in six minutes, each under a new issue name.
    told_lately = crisis and await _crisis_told_lately(ctx)
    res = {} if told_lately else await ctx.host.call(
        "notify_caregivers",
        {"message": a["message"], "urgency": "high" if a["reason"] in ("red_flag", "crisis") else "medium",
         "kind": kind if decision.whatsapp else "care_note"},
        family_id=ctx.family_id, subject_id=who, actor_id=ctx.actor_id,
    )
    told = int((res or {}).get("notifiedCount") or 0)
    others = (res or {}).get("others")
    alone = others == 0  # no caregiver in the family but the person themselves
    team = None
    if crisis or (a["reason"] == "red_flag" and alone):
        from app.care import careteam

        team = await careteam.notify(ctx.session, ctx.host, family_id=ctx.family_id, subject_id=who,
                                     kind="crisis" if crisis else "emergency_alone", summary=a["message"], said=ctx.user_text,
                                     family_told=told, actor_id=ctx.speaker.get("id"))
    who = guards.who_was_told(
        ctx.profiles.get(ctx.speaker.get("id")), n=told, lately=told_lately, alone=alone, team=team)
    return {"sent": True, "whatsapp": decision.whatsapp, "why": decision.why, "told": who, "result": res}


async def _crisis_told_lately(ctx: TurnCtx) -> bool:
    since = clock.now() - timedelta(minutes=30)
    rows = await store.events(ctx.session, ctx.family_id, ctx.elder_id, since=since, kinds=["alert_whatsapp"])
    return sum(1 for e in rows if (e.payload or {}).get("reason") == "crisis") > 1  # this alert's own event is one


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


@tool(
    "health_record",
    "Act on a health record (a photo or PDF of a report or prescription) that Saheli read and is waiting for an answer "
    "(today's events say 'health_record id …'). Nothing from it is saved until they choose. choice: add_medicines (save it, "
    "remember it and add its new medicines to the reminders), keep_record (save and remember, no reminders), file_only (keep "
    "only the file), delete, mine (the name on it is theirs after all), or fix (correct what was read, e.g. 'Shelcal is 500' or "
    "'Hb is 9.8'; then read the corrected lines back and ask again). Use the id from the events.",
    {
        "id": {"type": "string"},
        "choice": {"type": "string", "enum": ["add_medicines", "keep_record", "file_only", "delete", "mine", "fix"]},
        "corrections": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string", "description": "The medicine or test as it was read"},
            "value": {"type": "string", "description": "The right value, strength or dose"},
            "times": {"type": "array", "items": {"type": "string"}, "description": "Dose times HH:MM, for a medicine"},
            "remove": {"type": "boolean", "description": "Drop this line (it was misread or is not on the paper)"},
        }, "required": ["name"]}},
    },
    ["id", "choice"],
)
async def health_record(ctx: TurnCtx, a: dict) -> dict:
    return await ctx.host.call(
        "record_review", {"document_id": a["id"], "choice": a["choice"], "corrections": a.get("corrections") or []},
        family_id=ctx.family_id.removeprefix("shadow:"), subject_id=ctx.elder_id, actor_id=ctx.actor_id,
    )
