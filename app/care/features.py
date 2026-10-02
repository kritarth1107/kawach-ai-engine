"""Care views built from the care record, the ledger and open loops.

Each one is used twice: by a brain tool on WhatsApp and by a /v2/dash endpoint on the dashboard,
so the two always show the same thing.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.domains import slug
from app.care.models import CareEvent, CareFact, OpenLoop, Turn
from app.core import clock

REFILL_DAYS = int(os.getenv("REFILL_DAYS", "5"))
TAKEN_KINDS = ("dose_taken", "dose_empty_strip")
MISSED_KINDS = ("dose_missed", "dose_skipped", "dose_refused")
CONCERN_KINDS = ("alert_whatsapp", "alert_dashboard")


def med_name(f: CareFact) -> str:
    return f.value.get("name") or f.key.split(":", 1)[1].replace("_", " ").title()


def mentions(text: str, name: str) -> bool:
    """A dose event belongs to a medicine when the first word of its name appears in it."""
    first = slug(name).split("_")[0]
    return bool(first) and first in (text or "").lower()


def doses_per_day(f: CareFact) -> float:
    times = [t for t in (f.value.get("times") or []) if isinstance(t, str)]
    days = f.value.get("days") or None
    per = float(len(times))
    if days:
        per *= len(days) / 7
    return per


def _dose_of(e: CareEvent, name: str) -> bool:
    return mentions(str((e.payload or {}).get("medicine", "")), name) or mentions(e.summary, name)


# ── refills ────────────────────────────────────────────────────────────────────


async def stock(session: AsyncSession, family_id: str, subject_id: str) -> list[dict]:
    """Tablets left per active medicine: the last count someone gave, minus doses taken since."""
    meds = await store.facts(session, family_id, subject_id, domains=["medicine"], statuses=("active",))
    if not meds:
        return []
    evs = await store.events(
        session, family_id, subject_id, since=clock.now() - timedelta(days=180),
        kinds=["stock_set", *TAKEN_KINDS], limit=20000,
    )
    out = []
    for f in meds:
        name = med_name(f)
        last = None
        for e in evs:
            if e.kind == "stock_set" and (e.payload or {}).get("key") == f.key:
                last = e
        per = doses_per_day(f)
        row = {"key": f.key, "name": name, "dose": f.value.get("dose"), "perDay": round(per, 2), "stock": None, "daysLeft": None, "asOf": None, "low": False}
        if last is not None:
            taken = sum(1 for e in evs if e.kind in TAKEN_KINDS and e.at > last.at and _dose_of(e, name))
            left = max(int((last.payload or {}).get("count", 0)) - taken, 0)
            row.update(stock=left, asOf=last.at.isoformat())
            if per > 0:
                row["daysLeft"] = math.floor(left / per)
                row["low"] = row["daysLeft"] <= REFILL_DAYS
        out.append(row)
    return out


async def set_stock(session: AsyncSession, *, family_id: str, subject_id: str, key: str, name: str, count: int, actor_id: str | None) -> dict:
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="stock_set",
        summary=f"{name}: {count} left", payload={"key": key, "name": name, "count": int(count)}, actor_id=actor_id,
    )
    rows = [r for r in await stock(session, family_id, subject_id) if r["key"] == key]
    row = rows[0] if rows else {"key": key, "name": name, "stock": count, "low": False}
    if not row.get("low"):
        # Restocked: the refill question is answered.
        live = (
            await session.execute(
                select(OpenLoop).where(OpenLoop.family_id == family_id, OpenLoop.dedupe_key == f"refill:{subject_id}:{key}", OpenLoop.status == "open")
            )
        ).scalars()
        for loop in live:
            await store.close_loop(session, loop.id, note=f"restocked: {count} left")
    return row


async def find_medicine(session: AsyncSession, family_id: str, subject_id: str, name_or_key: str) -> CareFact | None:
    meds = await store.facts(session, family_id, subject_id, domains=["medicine"], statuses=("active",))
    want = name_or_key.split(":", 1)[-1]
    for f in meds:
        if f.key == name_or_key or f.key.split(":", 1)[1] == slug(want):
            return f
    for f in meds:
        if mentions(med_name(f), want) or mentions(want, med_name(f)):
            return f
    return None


async def refill_candidates(session: AsyncSession) -> list[tuple[str, str, dict]]:
    """(family, subject, stock row) for every medicine running low, across all families."""
    pairs = (
        await session.execute(
            select(CareFact.family_id, CareFact.subject_id)
            .where(CareFact.domain == "medicine", CareFact.status == "active", ~CareFact.family_id.startswith("shadow:"))
            .distinct()
        )
    ).all()
    out = []
    for fid, sid in pairs:
        for row in await stock(session, fid, sid):
            if row["low"]:
                out.append((fid, sid, row))
    return out


# ── emergency card ─────────────────────────────────────────────────────────────


def _v(f: CareFact, *names: str) -> str | None:
    for n in names:
        if f.value.get(n):
            return str(f.value[n])
    return None


async def emergency(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    rows = await store.facts(session, family_id, subject_id, statuses=("active",))
    by: dict[str, list[CareFact]] = defaultdict(list)
    for f in rows:
        by[f.domain].append(f)
    profile = {f.key.split(":", 1)[1]: (_v(f, "value", "blood_group", "text") or f.text) for f in by["profile"]}
    contacts = [
        {"name": _v(f, "name") or f.key.split(":", 1)[1].replace("_", " ").title(), "phone": _v(f, "phone", "number"),
         "relation": _v(f, "relation", "role"), "emergency": bool(f.value.get("emergency")), "text": f.text}
        for f in by["contact"]
    ]
    contacts.sort(key=lambda c: not c["emergency"])
    return {
        "profile": profile,
        "conditions": [f.text for f in by["condition"]],
        "allergies": [{"allergen": _v(f, "allergen") or f.key.split(":", 1)[1], "reaction": _v(f, "reaction"), "text": f.text} for f in by["allergy"]],
        "medicines": [{"name": med_name(f), "dose": f.value.get("dose"), "times": f.value.get("times") or [], "text": f.text} for f in by["medicine"]],
        "diet": [f.text for f in by["diet"]],
        "doctors": [{"name": _v(f, "name") or f.key.split(":", 1)[1].replace("_", " ").title(), "phone": _v(f, "phone"),
                     "speciality": _v(f, "speciality", "specialty"), "text": f.text} for f in by["doctor"]],
        "hospital": [{"name": _v(f, "name") or f.text, "phone": _v(f, "phone"), "address": _v(f, "address"), "text": f.text} for f in by["hospital"]],
        "contacts": contacts,
        "familyRules": [f.text for f in by["family"] if f.key.split(":", 1)[1] in ("call_first", "payer", "neighbour")],
    }


def emergency_text(name: str, card: dict) -> str:
    lines = [f"*Emergency card: {name}*"]
    p = card["profile"]
    if p:
        lines.append(" · ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in p.items()))
    if card["allergies"]:
        lines.append("Allergies: " + ", ".join(a["allergen"] + (f" ({a['reaction']})" if a.get("reaction") else "") for a in card["allergies"]))
    else:
        lines.append("Allergies: none known")
    if card["conditions"]:
        lines.append("Conditions: " + "; ".join(card["conditions"]))
    if card["medicines"]:
        lines.append("Medicines: " + "; ".join(f"{m['name']}{(' ' + m['dose']) if m.get('dose') else ''}" for m in card["medicines"]))
    for d in card["doctors"][:2]:
        lines.append(f"Doctor: {d['name']}{(' ' + d['phone']) if d.get('phone') else ''}")
    for h in card["hospital"][:1]:
        lines.append(f"Hospital: {h['name']}{(' ' + h['phone']) if h.get('phone') else ''}")
    for c in card["contacts"][:3]:
        lines.append(f"Call: {c['name']}{(' (' + c['relation'] + ')') if c.get('relation') else ''}{(' ' + c['phone']) if c.get('phone') else ''}")
    return "\n".join(lines)


# ── care team and appointments ─────────────────────────────────────────────────


def parse_when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=clock.IST)


def appointment_json(f: CareFact) -> dict:
    when = parse_when(f.value.get("when"))
    return {
        "key": f.key, "doctor": f.value.get("doctor") or f.key.split(":", 1)[1].replace("_", " ").title(),
        "when": clock.ist(when).strftime("%Y-%m-%dT%H:%M") if when else f.value.get("when"),
        "place": f.value.get("place"), "purpose": f.value.get("purpose"), "questions": f.value.get("questions") or [],
        "status": f.value.get("status") or "planned", "upcoming": bool(when and when >= clock.now() - timedelta(hours=2)), "text": f.text,
    }


async def appointment_loops(session: AsyncSession, *, family_id: str, subject_id: str, f: CareFact, owner_id: str | None) -> list[str]:
    """Wake Saheli the evening before and two hours before, to remind and offer a cab."""
    when = parse_when(f.value.get("when"))
    if not when or when <= clock.now():
        return []
    doctor = f.value.get("doctor") or "the doctor"
    at = clock.ist(when).strftime("%d %b %H:%M")
    place = f" at {f.value['place']}" if f.value.get("place") else ""
    out = []
    eve = clock.ist(when).replace(hour=19, minute=0) - timedelta(days=1)
    for tag, wake in (("eve", eve), ("2h", when - timedelta(hours=2))):
        if wake <= clock.now():
            continue
        loop = await store.open_loop(
            session, family_id=family_id, subject_id=subject_id, kind="appointment",
            title=f"Appointment with {doctor} on {at}{place}: remind them, share the questions to ask, offer a cab",
            detail={"key": f.key, "when": when.isoformat(), "max_wakes": 1, "stage": tag},
            owner_id=owner_id, wake_at=wake, alert_rule="dashboard", dedupe_key=f"appt:{subject_id}:{f.key}:{tag}",
        )
        out.append(str(loop.id))
    return out


async def care_team(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    rows = await store.facts(session, family_id, subject_id, domains=["doctor", "hospital", "contact", "home", "appointment"], statuses=("active",))
    card = await emergency(session, family_id, subject_id)
    appts = sorted((appointment_json(f) for f in rows if f.domain == "appointment"), key=lambda a: a["when"] or "")
    return {
        "doctors": card["doctors"], "hospital": card["hospital"], "contacts": card["contacts"],
        "helpers": [{"name": f.value.get("name") or f.key.split(":", 1)[1].replace("_", " ").title(), "text": f.text, "phone": f.value.get("phone")} for f in rows if f.domain == "home"],
        "upcoming": [a for a in appts if a["upcoming"]],
        "past": [a for a in reversed(appts) if not a["upcoming"]][:10],
    }


# ── wellbeing ──────────────────────────────────────────────────────────────────


async def wellbeing(session: AsyncSession, family_id: str, subject_id: str, days: int = 14) -> dict:
    since = clock.now() - timedelta(days=days)
    turns = list(
        (
            await session.execute(
                select(Turn.at).where(Turn.family_id == family_id, Turn.thread_id == subject_id, Turn.role == "user").order_by(Turn.at)
            )
        ).scalars()
    )
    last = turns[-1] if turns else None
    talked = {clock.ist_day(t) for t in turns if t >= since}
    evs = await store.events(session, family_id, subject_id, since=since, kinds=["mood", "symptom", "sleep", "social", "meal", *CONCERN_KINDS], limit=5000)
    day_list = []
    for back in range(days - 1, -1, -1):
        d = clock.ist_day(clock.now() - timedelta(days=back))
        es = [e for e in evs if e.day == d]
        sev = "info"
        for e in es:
            s = (e.payload or {}).get("severity") or ("concern" if e.kind in CONCERN_KINDS else "info")
            if s == "concern" or (s == "watch" and sev == "info"):
                sev = s
        day_list.append({
            "day": d, "talked": d in talked, "level": sev,
            "mood": [e.summary for e in es if e.kind == "mood"], "symptoms": [e.summary for e in es if e.kind == "symptom"],
        })
    streak = 0
    for d in reversed(day_list):
        if not d["talked"]:
            break
        streak += 1
    return {
        "lastHeard": last.isoformat() if last else None,
        "daysTalked": sum(1 for d in day_list if d["talked"]),
        "streak": streak,
        "days": day_list,
        "recentMood": [{"at": e.at.isoformat(), "text": e.summary, "level": (e.payload or {}).get("severity", "info")} for e in evs if e.kind == "mood"][-8:],
        "concerns": [{"at": e.at.isoformat(), "text": e.summary} for e in evs if e.kind in CONCERN_KINDS or (e.payload or {}).get("severity") == "concern"][-8:],
    }


def wellbeing_text(name: str, w: dict) -> str:
    last = clock.ist(datetime.fromisoformat(w["lastHeard"])).strftime("%d %b %H:%M") if w["lastHeard"] else "not yet"
    parts = [f"{name}: last heard {last}; talked {w['daysTalked']} of the last {len(w['days'])} days (streak {w['streak']})."]
    if w["recentMood"]:
        parts.append("Mood lately: " + "; ".join(m["text"] for m in w["recentMood"][-4:]))
    if w["concerns"]:
        parts.append("Watch: " + "; ".join(c["text"] for c in w["concerns"][-3:]))
    return " ".join(parts)


# ── weekly report ──────────────────────────────────────────────────────────────


async def report(session: AsyncSession, family_id: str, subject_id: str, days: int = 7) -> dict:
    days = max(1, min(days, 31))
    end = clock.ist()
    start_day = clock.ist_day(clock.now() - timedelta(days=days - 1))
    since = clock.now() - timedelta(days=days)
    meds = await store.facts(session, family_id, subject_id, domains=["medicine"], statuses=("active",))
    evs = await store.events(session, family_id, subject_id, since=since, limit=20000)
    evs = [e for e in evs if e.day >= start_day]
    per_day_expected = sum(len(f.value.get("times") or []) for f in meds)

    adherence = []
    for back in range(days - 1, -1, -1):
        d = clock.ist_day(clock.now() - timedelta(days=back))
        taken = sum(1 for e in evs if e.day == d and e.kind in TAKEN_KINDS)
        missed = sum(1 for e in evs if e.day == d and e.kind in MISSED_KINDS)
        adherence.append({"day": d, "expected": per_day_expected, "taken": min(taken, per_day_expected) if per_day_expected else taken, "missed": missed})
    by_med = []
    for f in meds:
        name = med_name(f)
        exp = len(f.value.get("times") or []) * days
        tk = sum(1 for e in evs if e.kind in TAKEN_KINDS and _dose_of(e, name))
        ms = sum(1 for e in evs if e.kind in MISSED_KINDS and _dose_of(e, name))
        by_med.append({"name": name, "dose": f.value.get("dose"), "times": f.value.get("times") or [], "expected": exp, "taken": min(tk, exp) if exp else tk, "missed": ms})
    total_exp = sum(a["expected"] for a in adherence)
    total_taken = sum(a["taken"] for a in adherence)

    vitals = [
        {"at": e.at.isoformat(), "kind": (e.payload or {}).get("kind"), "value": (e.payload or {}).get("value"), "note": (e.payload or {}).get("note"), "redFlag": (e.payload or {}).get("red_flag")}
        for e in evs if e.kind == "vital"
    ]
    from app.tasks.models import Task
    from app.tasks.runtime import rupees

    tasks = list(
        (
            await session.execute(
                select(Task).where(Task.family_id == family_id, Task.subject_id == subject_id, Task.created_at >= since).order_by(Task.created_at)
            )
        ).scalars()
    )
    appts = [appointment_json(f) for f in await store.facts(session, family_id, subject_id, domains=["appointment"], statuses=("active",))]
    out = {
        "from": start_day, "to": clock.ist_day(), "days": days, "generatedAt": end.isoformat(),
        "adherence": {"percent": round(100 * total_taken / total_exp) if total_exp else None, "taken": total_taken, "expected": total_exp, "byDay": adherence, "byMedicine": by_med},
        "vitals": vitals,
        "symptoms": [{"at": e.at.isoformat(), "text": e.summary, "level": (e.payload or {}).get("severity", "info")} for e in evs if e.kind == "symptom"],
        "mood": [{"at": e.at.isoformat(), "text": e.summary} for e in evs if e.kind in ("mood", "sleep", "meal")],
        "alerts": [{"at": e.at.isoformat(), "text": e.summary, "whatsapp": e.kind == "alert_whatsapp"} for e in evs if e.kind in CONCERN_KINDS],
        "changes": [{"at": e.at.isoformat(), "text": e.summary, "kind": e.kind} for e in evs if e.kind.startswith("fact_")],
        "orders": [{"at": t.created_at.isoformat(), "service": t.service, "kind": t.kind, "goal": t.goal, "status": t.status, "total": rupees((t.result or {}).get("total"))} for t in tasks],
        "appointments": [a for a in appts if a["when"] and a["when"][:10] >= start_day],
    }
    return out


def report_facts_text(name: str, r: dict) -> str:
    a = r["adherence"]
    lines = [f"Report for {name}, {r['from']} to {r['to']}."]
    lines.append(f"Medicines: {a['taken']} of {a['expected']} doses marked taken" + (f" ({a['percent']}%)." if a["percent"] is not None else "."))
    for m in a["byMedicine"]:
        if m["missed"] or (m["expected"] and m["taken"] < m["expected"]):
            lines.append(f"- {m['name']}: {m['taken']}/{m['expected']} taken, {m['missed']} missed")
    vit = defaultdict(list)
    for v in r["vitals"]:
        vit[v["kind"]].append(v["value"])
    for k, vals in vit.items():
        lines.append(f"{(k or 'reading').upper()}: " + ", ".join(str(x) for x in vals[-7:]))
    if r["symptoms"]:
        lines.append("Symptoms: " + "; ".join(s["text"] for s in r["symptoms"][-6:]))
    if r["mood"]:
        lines.append("Mood/sleep/meals: " + "; ".join(m["text"] for m in r["mood"][-5:]))
    if r["alerts"]:
        lines.append("Alerts: " + "; ".join(x["text"] for x in r["alerts"][-4:]))
    if r["changes"]:
        lines.append("Care record changes: " + "; ".join(c["text"] for c in r["changes"][-5:]))
    return "\n".join(lines)


NARRATIVE_PROMPT = """You write the summary paragraph of a weekly care report that a family will show to the elder's doctor.
Write 3-5 plain sentences: medicine adherence (with the numbers), readings and any trend, symptoms, mood/sleep, anything the doctor should ask about.
Use only the facts given. No diagnosis, no advice, no greetings. English."""


async def narrative(session: AsyncSession, family_id: str, subject_id: str, name: str, r: dict) -> str:
    """LLM summary, written once per subject, window and day; falls back to the plain facts."""
    from app.llm import router

    ref = f"report:{subject_id}:{r['to']}:{r['days']}"
    cached = (
        await session.execute(
            select(CareEvent).where(CareEvent.family_id == family_id, CareEvent.kind == "report_narrative", CareEvent.ref == ref)
        )
    ).scalar_one_or_none()
    if cached:
        return cached.summary
    facts = report_facts_text(name, r)
    try:
        reply = await router.complete(
            "extract", system_stable=NARRATIVE_PROMPT,
            messages=[{"role": "user", "content": [{"type": "text", "text": facts}]}], max_tokens=600, effort="low", timeout_s=40,
        )
        text = (reply.text or "").strip() or facts
    except Exception:  # noqa: BLE001 — the numbers are enough without the paragraph
        return facts
    await store.record_event(session, family_id=family_id, subject_id=subject_id, kind="report_narrative", summary=text, ref=ref)
    return text


# ── family tasks ───────────────────────────────────────────────────────────────


def family_task_json(o: OpenLoop) -> dict:
    d = o.detail or {}
    return {
        "id": str(o.id), "title": o.title, "assignee": o.owner_id, "subjectId": o.subject_id, "due": d.get("due"),
        "assignedBy": d.get("assigned_by"), "status": o.status, "note": o.closed_note,
        "createdAt": o.created_at.isoformat(), "updatedAt": o.updated_at.isoformat(),
    }


async def add_family_task(session: AsyncSession, *, family_id: str, subject_id: str, title: str, assignee: str, due: datetime | None, by: str | None) -> OpenLoop:
    return await store.open_loop(
        session, family_id=family_id, subject_id=subject_id, kind="family_task", title=title.strip()[:300],
        detail={"due": due.isoformat() if due else None, "assigned_by": by, "max_wakes": 1},
        owner_id=assignee, wake_at=due, alert_rule="dashboard", dedupe_key=f"family_task:{assignee}:{slug(title)[:60]}",
    )


async def family_tasks(session: AsyncSession, family_id: str, *, days_done: int = 14) -> list[dict]:
    since = clock.now() - timedelta(days=days_done)
    q = select(OpenLoop).where(
        OpenLoop.family_id == family_id, OpenLoop.kind == "family_task",
        (OpenLoop.status == "open") | (OpenLoop.updated_at >= since),
    ).order_by(OpenLoop.created_at.desc())
    return [family_task_json(o) for o in (await session.execute(q)).scalars()]


# ── spending ───────────────────────────────────────────────────────────────────


async def spending(session: AsyncSession, family_id: str, month: str | None = None) -> dict:
    from app.tasks.models import Task
    from app.tasks.runtime import rupees

    month = month or clock.ist().strftime("%Y-%m")
    y, m = (int(x) for x in month.split("-"))
    start = datetime(y, m, 1, tzinfo=clock.IST)
    end = datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=clock.IST)
    tasks = list(
        (
            await session.execute(
                select(Task).where(Task.family_id == family_id, Task.created_at >= start, Task.created_at < end).order_by(Task.created_at)
            )
        ).scalars()
    )
    items, by_service, by_person, by_kind = [], defaultdict(float), defaultdict(float), defaultdict(float)
    for t in tasks:
        r = t.result or {}
        placed = t.status == "done" and bool(r.get("placed") or r.get("booked")) and not t.cancel_requested
        if not placed:
            continue
        amt = rupees(r.get("total") or r.get("fare")) or rupees((t.details or {}).get("confirmed_total")) or 0.0
        items.append({"at": t.created_at.isoformat(), "service": t.service, "kind": t.kind, "goal": t.goal, "subjectId": t.subject_id, "total": amt})
        by_service[t.service] += amt
        by_person[t.subject_id] += amt
        by_kind[t.kind] += amt
    return {
        "month": month, "total": round(sum(by_service.values()), 2), "count": len(items),
        "byService": dict(by_service), "byPerson": dict(by_person), "byKind": dict(by_kind), "items": items,
    }
