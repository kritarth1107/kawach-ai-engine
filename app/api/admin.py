"""Admin console routes (/v2/admin), separate from the public /v2 API and secured differently.

Only the admin API service (kavach-admin-api) may call them: its Google identity (ADMIN_API_SA) plus its own secret
(ENGINE_ADMIN_SECRET), never the backend's X-Kavach-Secret. Unset either and the routes answer 503 (switched off). On a
host without Google identities (the AWS copy, private network) ENGINE_CALLER_CHECK=off leaves the secret only; never on
Cloud Run.
Conversations, care records and exports are reached only after the admin API checked break-glass permission, a
reason and wrote its audit entry.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

logger = logging.getLogger(__name__)


def _decode(part: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def _audiences() -> set[str]:
    return {a.strip().rstrip("/") for a in os.environ.get("ENGINE_AUDIENCES", "").split(",") if a.strip()}


def caller_email(authorization: str | None, *, on_cloud_run: bool | None = None, verify=None) -> str:
    """The Google identity that called. Cloud Run IAM already checked the token; this re-checks who it is."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="no caller token")
    token = authorization.split(" ", 1)[1].strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="bad caller token")
    on_cloud_run = bool(os.environ.get("K_SERVICE")) if on_cloud_run is None else on_cloud_run
    auds = _audiences()
    stripped = parts[2] in ("", "SIGNATURE_REMOVED_BY_GOOGLE")  # Cloud Run checked it and removed the signature
    if not stripped:
        try:
            if verify is None:
                from google.auth.transport import requests as g_requests
                from google.oauth2 import id_token

                payload = id_token.verify_oauth2_token(token, g_requests.Request(), audience=None)
            else:
                payload = verify(token)
        except Exception:
            raise HTTPException(status_code=401, detail="bad caller token") from None
    elif on_cloud_run:
        # Only behind Cloud Run IAM (the service is deployed --no-allow-unauthenticated): it verified this token.
        payload = _decode(parts[1])
        if float(payload.get("exp", 0)) < time.time():
            raise HTTPException(status_code=401, detail="bad caller token")
    else:
        raise HTTPException(status_code=401, detail="bad caller token")
    if not auds or str(payload.get("aud", "")).rstrip("/") not in auds:  # no audiences configured = closed
        raise HTTPException(status_code=401, detail="bad caller token")
    if payload.get("email_verified") is False:
        raise HTTPException(status_code=403, detail="caller not allowed")
    return str(payload.get("email", "")).lower()


def caller_check_on() -> bool:
    """The Google identity check on admin and score-posting calls. ENGINE_CALLER_CHECK=off is for a host without Google
    identities (the AWS copy: the engine on a private network, its callers holding ENGINE_ADMIN_SECRET). On Cloud Run
    (K_SERVICE set) the check always stays on, whatever the setting says."""
    if os.environ.get("ENGINE_CALLER_CHECK", "on").strip().lower() != "off":
        return True
    if os.environ.get("K_SERVICE"):
        logger.error("ENGINE_CALLER_CHECK=off ignored on Cloud Run: the caller check stays on")
        return True
    return False


def admin_secret_ok(request: Request) -> None:
    """The shared admin secret (at least 32 characters; unset = admin routes off)."""
    secret = os.environ.get("ENGINE_ADMIN_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="admin routes are switched off")
    if not hmac.compare_digest(request.headers.get("x-admin-secret", ""), secret):
        raise HTTPException(status_code=401, detail="bad admin secret")


async def verify_admin_caller(request: Request) -> None:
    allowed = os.environ.get("ADMIN_API_SA", "").lower()
    check = caller_check_on()
    if check and not allowed:
        raise HTTPException(status_code=503, detail="admin routes are switched off")
    admin_secret_ok(request)
    if check and caller_email(request.headers.get("authorization")) != allowed:
        raise HTTPException(status_code=403, detail="caller not allowed")


router = APIRouter(prefix="/v2/admin", tags=["admin"], dependencies=[Depends(verify_admin_caller)])
DB = Annotated[AsyncSession, Depends(get_db)]


@router.get("/health")
async def admin_health(session: DB) -> dict:
    from app.care.models import FamilyRoster
    from app.llm.provider import llm_provider_label
    from app.rag.embeddings import embedding_provider_label
    from app.care.models import CareEvent

    # the nightly dream's ledger event (app/care/dream.py), not the retired v1 profile render (trust plan A1)
    last_dream = (await session.execute(select(func.max(CareEvent.at)).where(CareEvent.kind == "dream"))).scalar_one_or_none()
    families = (await session.execute(select(func.count()).select_from(FamilyRoster))).scalar_one()
    from app.db.session import SessionLocal
    from app.learn.jobs import last_backup

    try:
        backup = await last_backup(SessionLocal)
    except Exception:  # noqa: BLE001 - the health view must not fail on the backup lookup
        backup = None
    return {"status": "ok", "llm": llm_provider_label(), "embeddings": embedding_provider_label(),
            "lastDream": last_dream.isoformat() if last_dream else None, "families": families, "backup": backup}


@router.get("/nudge-rates")
async def admin_nudge_rates(session: DB, days: int = 7) -> dict:
    """Reply rate of sent nudges by kind, hour and person. The admin overview shows this."""
    from app.care import nudges

    return await nudges.reply_rates(session, days=days)


@router.get("/spend")
async def admin_spend(days: int = 30) -> dict:
    from app.api.brain import llm_spend

    return await llm_spend(days=days)


@router.get("/models")
async def admin_models() -> dict:
    """Which models each role uses (first = primary, rest = fallbacks), their estimated prices, and today's caps."""
    from app.llm import router as llm_router
    from app.llm import spend
    from app.llm.provider import llm_provider_label
    from app.rag.embeddings import embedding_provider_label

    roles = sorted(set(llm_router.DEFAULT_ROUTES) | llm_router.configured_roles())
    configured = llm_router.configured_roles()
    out = {}
    for role in roles:
        out[role] = {
            "configured": role in configured,
            "routes": [{"provider": r.provider, "model": r.model, "location": r.location,
                        "priceInr": [round(x, 2) for x in spend.price_inr(r.model)]} for r in llm_router.routes_for(role)],
        }
    return {"llm": llm_provider_label(), "embeddings": embedding_provider_label(), "roles": out,
            "today": round(await spend.spent_today(), 2), "softCap": spend.soft_cap(), "hardCap": spend.hard_cap(),
            "usdInr": spend.USD_INR}


@router.get("/metrics")
async def admin_metrics(session: DB, days: int = 7) -> dict:
    from app.api.brain import agent_metrics

    return await agent_metrics(session=session, days=days)


@router.get("/learn")
async def admin_learn(weeks: int = 8) -> dict:
    from app.api.brain import learn_overview

    return await learn_overview(weeks=weeks)


@router.get("/flywheel")
async def admin_flywheel(session: DB, days: int = 14) -> dict:
    """The flywheel page: task completion/time/reopen/cost, conversations, check-ins, per family, plus learning signals
    (correction cases this week, rule proposals waiting, fine-tuning readiness) and the browser credit."""
    from app.api.brain import agent_credits
    from app.learn import corrections, flywheel, tuning_readiness

    out = await flywheel.compute(session, days=max(1, min(days, 90)))
    cases = await corrections.cases(session, days=7, limit=500)
    kinds: dict[str, int] = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    ready = await tuning_readiness.readiness(session)
    try:
        credit = await agent_credits()
    except Exception:  # noqa: BLE001
        credit = {"usd": None, "warnAt": None}
    from sqlalchemy import func, select

    from app.learn.models import RuleProposal

    waiting = (await session.execute(select(func.count()).select_from(RuleProposal).where(RuleProposal.status == "proposed"))).scalar_one()
    return {**out, "learning": {"correctionsThisWeek": kinds, "ruleProposalsWaiting": waiting,
                                "tuning": {k: ready[k] for k in ("ready", "goodReplies", "consented", "families", "missing")}},
            "credit": credit}


class By(BaseModel):
    by: str = Field(min_length=1, max_length=64)


class RuleBy(By):
    approve: bool


@router.post("/learn/playbooks/{version}/{action}")
async def admin_playbook(version: int, action: str, body: By) -> dict:
    from app.api import brain

    if action == "approve":
        return await brain.learn_approve(version, brain.PlaybookAction(by=body.by))
    if action == "block":
        return await brain.learn_block(version, brain.PlaybookAction(by=body.by))
    raise HTTPException(status_code=404, detail="no such action")


@router.post("/learn/rules/{rule_id}")
async def admin_rule(rule_id: int, body: RuleBy) -> dict:
    from app.api import brain

    return await brain.learn_rule_decision(rule_id, brain.RuleDecision(by=body.by, approve=body.approve))


@router.get("/families/{family_id}/conversation")
async def admin_conversation(family_id: str, thread: str, session: DB, limit: int = 80) -> dict:
    from app.care.models import Turn

    rows = (await session.execute(
        select(Turn).where(Turn.family_id == family_id, Turn.thread_id == thread).order_by(Turn.id.desc()).limit(max(1, min(limit, 300)))
    )).scalars().all()
    return {"turns": [{"at": t.at.isoformat(), "role": t.role, "speaker": t.speaker_id, "text": t.text,
                       "voice": bool((t.meta or {}).get("modality") == "voice")} for t in reversed(rows)]}


@router.get("/families/{family_id}/care-record")
async def admin_care_record(family_id: str, subject: str, session: DB) -> dict:
    from app.care import links
    from app.care.models import CareFact
    from app.care.versions import MemoryVersion

    rows = (await session.execute(
        select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject).order_by(CareFact.recorded_at.desc()).limit(400)
    )).scalars().all()
    # the change that made each row, for the "Why?" link
    made_rows = (await session.execute(select(MemoryVersion.ref, func.min(MemoryVersion.id)).where(
        MemoryVersion.family_id == family_id, MemoryVersion.subject_id == subject, MemoryVersion.kind == "fact",
        MemoryVersion.ref.in_([str(f.id) for f in rows])).group_by(MemoryVersion.ref))).tuples().all() if rows else []
    made: dict[str | None, int] = {ref: version_id for ref, version_id in made_rows}
    linked = links.by_from(await links.links_for(session, family_id, subject, statuses=("active", "pending")))
    return {"facts": [{"domain": f.domain, "key": f.key, "text": f.text, "status": f.status, "source": f.source_kind,
                       "confidence": f.confidence, "recordedAt": f.recorded_at.isoformat(),
                       "validTo": f.valid_to.isoformat() if f.valid_to else None, "versionId": made.get(str(f.id)),
                       "links": [{"text": links.phrase(ln), "status": ln.status} for ln in linked.get(f.key, [])] if f.status == "active" else []}
                      for f in rows]}


@router.get("/why")
async def admin_why(session: DB, family: str, task: str | None = None, version: int | None = None, loop: str | None = None,
                    event: int | None = None, trace_id: Annotated[str | None, Query(alias="trace")] = None) -> dict:
    """The why-trail (D-13) of a task, a care-record change, an alert or question (open loop), a ledger event or a trace,
    within one family: the message, model, role and cost, tool calls, confirms, outcome and what was learned."""
    from app.care import why

    if not any((task, version, loop, event, trace_id)):
        raise HTTPException(status_code=400, detail="give task, version, loop, event or trace")
    try:
        out = await why.chain(session, family_id=family, task_id=task, version_id=version, loop_id=loop, event_id=event, trace_id=trace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad id") from None
    if out is None:
        raise HTTPException(status_code=404, detail="nothing like that in this family")
    return out


@router.get("/families/{family_id}/why")
async def admin_why_recent(family_id: str, session: DB, days: int = 14) -> dict:
    """What can be asked "why?" about in this family: recent tasks, care-record and link changes, alerts and questions."""
    from app.care import why

    return await why.recent(session, family_id, days=days)


@router.get("/families/{family_id}/export")
async def admin_export(family_id: str, subject: str, session: DB) -> dict:
    """Everything Saheli holds about one person in one family, for that person's own data request."""
    from app.care.models import CareFact, Turn

    from app.care.models import CareLink

    facts = (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject))).scalars().all()
    turns = (await session.execute(select(Turn).where(Turn.family_id == family_id, Turn.thread_id == subject).order_by(Turn.id))).scalars().all()
    rows = (await session.execute(select(CareLink).where(CareLink.family_id == family_id, CareLink.subject_id == subject))).scalars().all()
    return {
        "careRecord": [{"domain": f.domain, "key": f.key, "value": f.value, "text": f.text, "status": f.status, "source": f.source_kind,
                        "recordedAt": f.recorded_at.isoformat(), "validFrom": f.valid_from.isoformat(),
                        "validTo": f.valid_to.isoformat() if f.valid_to else None} for f in facts],
        "links": [{"from": ln.from_ref, "relation": ln.relation, "to": ln.to_ref, "label": ln.label, "status": ln.status, "source": ln.source_kind,
                   "validFrom": ln.valid_from.isoformat(), "validTo": ln.valid_to.isoformat() if ln.valid_to else None} for ln in rows],
        "conversation": [{"at": t.at.isoformat(), "role": t.role, "text": t.text} for t in turns],
    }


@router.get("/latency")
async def admin_latency(session: DB, hours: int = 1) -> dict:
    """How long Saheli took to answer (the why-trail's turn time, message turns only) over the last hours: count, p50,
    p90 and the slowest, in ms. Read by the hourly live check (quality plan F8); sims and shadow families are left out.
    steps: where the time went, p50/p90 of each step over the turns that recorded it (waiting for the family's previous
    turn, building the context, the brain's model calls, tools, the reply checks, after the reply)."""
    import math
    from datetime import timedelta

    from app.care.why import TraceTurn
    from app.core import clock

    since = clock.now() - timedelta(hours=max(1, min(hours, 168)))
    rows = (await session.execute(select(TraceTurn.ms, TraceTurn.timing).where(
        TraceTurn.at >= since, TraceTurn.kind == "message", ~TraceTurn.family_id.startswith("shadow:"),
        ~TraceTurn.family_id.startswith("sim-")).order_by(TraceTurn.ms))).all()
    ms = [int(r.ms or 0) for r in rows]

    def pct(values: list[int], q: float) -> int | None:
        values = sorted(values)
        return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))] if values else None  # nearest rank

    timed = [(int(r.ms or 0), r.timing) for r in rows if isinstance(r.timing, dict)]
    parts: dict[str, list[int]] = {k: [] for k in ("lock", "context", "brain", "otherModels", "tools", "check", "after", "own")}
    for total, t in timed:
        model = t.get("model") or []
        parts["lock"].append(int(t.get("lockMs", 0)))
        parts["context"].append(int(t.get("contextMs", 0)))
        parts["brain"].append(sum(int(m) for r, m in model if r in ("brain", "brain_hard")))
        parts["otherModels"].append(sum(int(m) for r, m in model if r not in ("brain", "brain_hard")))
        parts["tools"].append(sum(int(m) for _, m in t.get("tool") or []))
        parts["check"].append(int(t.get("checkMs", 0)))
        parts["after"].append(int(t.get("afterMs", 0)))
        parts["own"].append(max(0, total - int(t.get("lockMs", 0))))  # the turn's own time, without waiting its turn
    steps = {k: {"p50Ms": pct(v, 0.5), "p90Ms": pct(v, 0.9)} for k, v in parts.items()} if timed else {}
    return {"hours": max(1, min(hours, 168)), "turns": len(ms), "p50Ms": pct(ms, 0.5), "p90Ms": pct(ms, 0.9), "maxMs": ms[-1] if ms else None,
            "timedTurns": len(timed), "steps": steps}


@router.get("/telemetry")
async def admin_telemetry(session: DB, hours: int = 24) -> dict:
    """Where Saheli's tokens, model calls, money and time go (founder 2026-10-17: the admin is for debugging infra and
    AI). Built from the why-trail (one row per turn: models with tokens and cost, timing with cached tokens). Message
    turns (a person wrote) and Saheli's own messages (system) apart; sims and shadow families left out. No text."""
    import math
    from collections import defaultdict
    from datetime import timedelta

    from app.care.why import TraceTurn
    from app.core import clock

    hours = max(1, min(hours, 168))
    since = clock.now() - timedelta(hours=hours)
    rows = (await session.execute(select(TraceTurn.at, TraceTurn.kind, TraceTurn.family_id, TraceTurn.ms, TraceTurn.cost_inr, TraceTurn.models,
                                         TraceTurn.timing, TraceTurn.tools).where(
        TraceTurn.at >= since, ~TraceTurn.family_id.startswith("shadow:"), ~TraceTurn.family_id.startswith("sim-")))).all()

    def pct(values: Sequence[int | float], q: float) -> int | float | None:
        v = sorted(values)
        return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))] if v else None

    def summary(turns: list) -> dict:
        calls, tin, tout, cost, ms, cached, prompt = [], [], [], [], [], 0, 0
        for r in turns:
            models = r.models or []
            calls.append(sum(int(m.get("calls") or 0) for m in models))
            tin.append(sum(int(m.get("tokensIn") or 0) for m in models))
            tout.append(sum(int(m.get("tokensOut") or 0) for m in models))
            cost.append(float(r.cost_inr or 0))
            ms.append(int(r.ms or 0))
            t = r.timing if isinstance(r.timing, dict) else {}
            cached += int(t.get("cacheRead") or 0)
            prompt += int(t.get("promptTok") or 0)
        n = len(turns)
        avg = lambda v: round(sum(v) / n, 2) if n else None  # noqa: E731
        return {
            "turns": n, "costInr": round(sum(cost), 2),
            "callsPerTurn": {"avg": avg(calls), "p90": pct(calls, 0.9)},
            "tokensInPerTurn": {"avg": avg(tin), "p50": pct(tin, 0.5), "p90": pct(tin, 0.9)},
            "tokensOutPerTurn": {"avg": avg(tout), "p90": pct(tout, 0.9)},
            "costPerTurn": {"avg": avg(cost), "p90": pct(cost, 0.9)},
            "ms": {"p50": pct(ms, 0.5), "p90": pct(ms, 0.9), "max": max(ms) if ms else None},
            "cache": {"readTokens": cached, "promptTokens": prompt, "hitPct": round(100 * cached / prompt, 1) if prompt else None},
        }

    msg = [r for r in rows if r.kind == "message"]
    own = [r for r in rows if r.kind == "system"]
    roles: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokensIn": 0, "tokensOut": 0, "costInr": 0.0, "ms": []})
    models: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokensIn": 0, "tokensOut": 0, "costInr": 0.0})
    tools: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        for m in r.models or []:
            for key, box in ((m.get("role") or "?", roles), (m.get("model") or "?", models)):
                b = box[key]
                b["calls"] += int(m.get("calls") or 0)
                b["tokensIn"] += int(m.get("tokensIn") or 0)
                b["tokensOut"] += int(m.get("tokensOut") or 0)
                b["costInr"] += float(m.get("costInr") or 0)
        t = r.timing if isinstance(r.timing, dict) else {}
        for role, ms in t.get("model") or []:
            roles[role]["ms"].append(int(ms))
        for name, ms in t.get("tool") or []:
            tools[name].append(int(ms))
    by_role = sorted(({"role": k, "calls": v["calls"], "tokensIn": v["tokensIn"], "tokensOut": v["tokensOut"], "costInr": round(v["costInr"], 2),
                       "p50Ms": pct(v["ms"], 0.5), "p90Ms": pct(v["ms"], 0.9)} for k, v in roles.items()), key=lambda x: -x["costInr"])
    by_model = sorted(({"model": k, **{**v, "costInr": round(v["costInr"], 2)}} for k, v in models.items()), key=lambda x: -x["costInr"])
    by_tool = sorted(({"tool": k, "calls": len(v), "p50Ms": pct(v, 0.5), "p90Ms": pct(v, 0.9)} for k, v in tools.items()), key=lambda x: -x["calls"])[:15]
    fams: dict[str, dict] = defaultdict(lambda: {"turns": 0, "costInr": 0.0, "tokensIn": 0})
    for r in rows:
        f = fams[r.family_id]
        f["turns"] += 1
        f["costInr"] += float(r.cost_inr or 0)
        f["tokensIn"] += sum(int(m.get("tokensIn") or 0) for m in r.models or [])
    by_family = sorted(({"family": k[:8], **{**v, "costInr": round(v["costInr"], 2)}} for k, v in fams.items()), key=lambda x: -x["costInr"])[:15]
    hourly: dict[str, list] = defaultdict(list)
    for r in msg:
        hourly[clock.ist(r.at).strftime("%Y-%m-%dT%H:00")].append(r)
    series = [{"hour": h, "turns": len(v), "tokensIn": round(sum(sum(int(m.get("tokensIn") or 0) for m in x.models or []) for x in v) / len(v)),
               "costInr": round(sum(float(x.cost_inr or 0) for x in v), 2), "p90Ms": pct([int(x.ms or 0) for x in v], 0.9)} for h, v in sorted(hourly.items())]
    slow = sorted(msg, key=lambda r: -(r.ms or 0))[:10]
    slowest = [{"at": r.at.isoformat(), "family": r.family_id[:8], "ms": int(r.ms or 0), "calls": sum(int(m.get("calls") or 0) for m in r.models or []),
                "tokensIn": sum(int(m.get("tokensIn") or 0) for m in r.models or []), "costInr": round(float(r.cost_inr or 0), 2),
                "timing": {k: v for k, v in (r.timing or {}).items() if k.endswith("Ms")} if isinstance(r.timing, dict) else {}} for r in slow]
    return {"hours": hours, "messages": summary(msg), "proactive": summary(own), "byRole": by_role, "byModel": by_model, "byTool": by_tool,
            "byFamily": by_family, "hourly": series, "slowest": slowest}


@router.get("/job-runs")
async def admin_job_runs(session: DB, days: int = 3) -> dict:
    """The latest run of each scheduled job, and its last 24 hours (runs, failures, p90 time)."""
    from datetime import timedelta

    from app.care.why import JobRun
    from app.core import clock

    since = clock.now() - timedelta(days=max(1, min(days, 14)))
    rows = (await session.execute(select(JobRun).where(JobRun.at >= since).order_by(JobRun.at.desc()))).scalars().all()
    day = clock.now() - timedelta(hours=24)
    out: dict[str, dict] = {}
    for r in rows:
        j = out.setdefault(r.job, {"job": r.job, "lastAt": r.at.isoformat(), "lastMs": r.ms, "lastOk": r.ok, "lastCode": r.code, "runs24h": 0, "failed24h": 0, "ms": []})
        if r.at >= day:
            j["runs24h"] += 1
            j["failed24h"] += 0 if r.ok else 1
            j["ms"].append(r.ms)
    for j in out.values():
        ms = sorted(j.pop("ms"))
        j["p90Ms"] = ms[int(0.9 * (len(ms) - 1))] if ms else None
    return {"jobs": list(out.values())}


@router.get("/live")
async def admin_live(session: DB, minutes: int = 30) -> dict:
    """What is going on right now (founder 2026-10-13: "watch live, every session ongoing"): conversations in the last
    minutes with their pace, cost and mood, orders and rides running, check-ins due within the hour, and alerts of the
    day. No message text here: reading a conversation stays behind break-glass."""
    from datetime import timedelta

    from app.care.models import CareEvent, OpenLoop, Turn
    from app.care.why import TraceTurn
    from app.core import clock
    from app.tasks.models import Task as CareTask

    now = clock.now()
    since = now - timedelta(minutes=max(5, min(minutes, 240)))
    real = lambda col: ~col.startswith("shadow:") & ~col.startswith("sim-")  # noqa: E731
    turns = (await session.execute(select(Turn.family_id, Turn.thread_id, Turn.role, Turn.at).where(Turn.at >= since, real(Turn.family_id))
                                   .order_by(Turn.at))).all()
    convs: dict[tuple[str, str], dict] = {}
    for fid, thread, role, at in turns:
        if thread == "saheli-scheduler":
            continue
        c = convs.setdefault((fid, thread), {"familyId": fid, "personId": thread, "startedAt": at.isoformat(), "messages": 0, "replies": 0,
                                             "lastAt": at.isoformat(), "lastRole": role})
        c["messages" if role == "user" else "replies"] += 1
        c["lastAt"], c["lastRole"] = at.isoformat(), role
    traces = (await session.execute(select(TraceTurn.family_id, TraceTurn.thread_id, TraceTurn.ms, TraceTurn.cost_inr, TraceTurn.models, TraceTurn.kind)
                                    .where(TraceTurn.at >= since, real(TraceTurn.family_id)))).all()
    outreach: dict[str, int] = {}
    for fid, thread, ms, cost, models, kind in traces:
        c = convs.get((fid, thread))
        if kind == "system":
            outreach[fid] = outreach.get(fid, 0) + 1
        if not c:
            continue
        c.setdefault("_ms", []).append(ms or 0)
        c["costInr"] = round(c.get("costInr", 0) + float(cost or 0), 2)
        for m in models or []:
            if isinstance(m, dict) and m.get("role") == "brain" and m.get("model"):
                c["model"] = m["model"]
    if convs:
        moods = (await session.execute(select(CareEvent.family_id, CareEvent.subject_id, CareEvent.payload, CareEvent.at)
                                       .where(CareEvent.kind == "mood", CareEvent.at >= now - timedelta(hours=24),
                                              CareEvent.family_id.in_({k[0] for k in convs})).order_by(CareEvent.at))).all()
        last_mood = {(f, s): p for f, s, p, _ in moods}
    else:
        last_mood = {}
    out_convs = []
    for (fid, thread), c in convs.items():
        ms = c.pop("_ms", [])
        mood = last_mood.get((fid, thread)) or {}
        waiting = c["lastRole"] == "user"
        out_convs.append({**c, "avgReplyMs": int(sum(ms) / len(ms)) if ms else None, "waitingForSaheli": waiting,
                          "waitingSeconds": int((now - datetime_from(c["lastAt"])).total_seconds()) if waiting else None,
                          "mood": {"score": mood.get("score"), "word": mood.get("word")} if mood else None,
                          "active": (now - datetime_from(c["lastAt"])).total_seconds() < 300})
    out_convs.sort(key=lambda c: c["lastAt"], reverse=True)

    tasks = (await session.execute(select(CareTask).where(CareTask.status.in_(("queued", "running", "needs_input", "awaiting_confirm")),
                                                          real(CareTask.family_id)).order_by(CareTask.updated_at.desc()).limit(50))).scalars().all()
    out_tasks = [{"taskId": str(t.id), "familyId": t.family_id, "personId": t.subject_id, "service": t.service, "kind": t.kind,
                  "status": t.status, "phase": t.phase, "inputNeeded": t.input_needed, "startedAt": t.created_at.isoformat(),
                  "updatedAt": t.updated_at.isoformat(), "runs": t.runs, "hasLiveView": bool(t.agent_session),
                  "minutes": int((now - t.created_at).total_seconds() // 60)} for t in tasks]

    due = (await session.execute(select(OpenLoop.family_id, OpenLoop.owner_id, OpenLoop.kind, OpenLoop.wake_at)
                                 .where(OpenLoop.status == "open", OpenLoop.wake_at >= now, OpenLoop.wake_at <= now + timedelta(minutes=60),
                                        real(OpenLoop.family_id)).order_by(OpenLoop.wake_at).limit(50))).all()
    alerts = (await session.execute(select(CareEvent.family_id, CareEvent.subject_id, CareEvent.kind, CareEvent.at)
                                    .where(CareEvent.kind.in_(("alert_whatsapp", "alert_dashboard", "crisis", "emergency_alone")),
                                           CareEvent.at >= now - timedelta(hours=24), real(CareEvent.family_id)).order_by(CareEvent.at.desc()).limit(50))).all()
    return {
        "now": now.isoformat(),
        "windowMinutes": int((now - since).total_seconds() // 60),
        "conversations": out_convs,
        "outreach": [{"familyId": f, "count": n} for f, n in outreach.items()],
        "tasks": out_tasks,
        "comingUp": [{"familyId": f, "personId": o, "kind": k, "at": w.isoformat()} for f, o, k, w in due],
        "alerts": [{"familyId": f, "personId": s, "kind": k, "at": a.isoformat()} for f, s, k, a in alerts],
    }


def datetime_from(iso: str):
    from datetime import datetime as _dt

    return _dt.fromisoformat(iso)


@router.get("/tasks/{task_id}/live-url")
async def admin_task_live_url(task_id: str, session: DB) -> dict:
    """The browser view of a running order or ride (the admin API asks for break-glass and a reason first)."""
    import uuid as _uuid

    from app.brain import tools
    from app.tasks.models import Task as CareTask

    try:
        t = await session.get(CareTask, _uuid.UUID(task_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="bad id") from None
    if not t:
        raise HTTPException(status_code=404, detail="no such task")
    if not t.agent_session:
        return {"url": None}
    try:
        return {"url": await tools.task_agent().live_url(t.agent_session)}
    except Exception:  # noqa: BLE001 — the browser may already be closed
        return {"url": None}


@router.get("/trainer-runs")
async def admin_trainer_runs(session: DB, limit: int = 30) -> dict:
    """The nightly trainer's runs, newest first, each with its full report (eval/trainer.py)."""
    from app.quality.models import TrainerRun

    rows = (await session.execute(select(TrainerRun).order_by(TrainerRun.started_at.desc()).limit(max(1, min(limit, 90))))).scalars().all()
    return {"runs": [{"id": r.id, "day": r.day, "startedAt": r.started_at.isoformat(), "finishedAt": r.finished_at.isoformat() if r.finished_at else None,
                      "status": r.status, "dryRun": r.dry_run, "spentInr": round(float(r.spent_inr or 0), 2), "capInr": float(r.cap_inr or 0),
                      "cases": r.cases, "tried": r.tried, "kept": r.kept, "prs": r.prs or [], "error": r.error or "", "taskId": r.task_id or "",
                      "report": r.report or {}} for r in rows]}


@router.get("/db-stats")
async def admin_db_stats(session: DB) -> dict:
    """Postgres (Aurora) as the engine sees it: size, connections, cache, activity, the biggest tables and the vector
    indexes. Catalog numbers only; no row content and no query text."""
    from sqlalchemy import text as sql

    async def one(q: str) -> dict:
        r = (await session.execute(sql(q))).mappings().first()
        return dict(r) if r else {}

    async def many(q: str) -> list[dict]:
        return [dict(r) for r in (await session.execute(sql(q))).mappings().all()]

    db = await one("""SELECT version() AS version, current_database() AS name, pg_database_size(current_database()) AS bytes,
        pg_postmaster_start_time() AS started_at, current_setting('max_connections')::int AS max_connections,
        current_setting('shared_buffers') AS shared_buffers, current_setting('work_mem') AS work_mem,
        current_setting('statement_timeout') AS statement_timeout, (SELECT count(*) FROM pg_stat_activity) AS connections""")
    stat = await one("""SELECT xact_commit, xact_rollback, blks_read, blks_hit, tup_returned, tup_fetched, tup_inserted, tup_updated,
        tup_deleted, conflicts, temp_files, temp_bytes, deadlocks, stats_reset FROM pg_stat_database WHERE datname = current_database()""")
    hit = (stat.get("blks_hit") or 0) / max(1, (stat.get("blks_hit") or 0) + (stat.get("blks_read") or 0))
    states = await many("""SELECT coalesce(state, 'background') AS state, coalesce(application_name, '') AS app, count(*) AS n,
        max(extract(epoch FROM now() - coalesce(query_start, backend_start)))::int AS oldest_s
        FROM pg_stat_activity WHERE datname = current_database() GROUP BY 1, 2 ORDER BY n DESC""")
    waits = await many("""SELECT wait_event_type AS type, wait_event AS event, count(*) AS n FROM pg_stat_activity
        WHERE wait_event IS NOT NULL AND state = 'active' GROUP BY 1, 2 ORDER BY n DESC LIMIT 10""")
    tables = await many("""SELECT c.relname AS name, pg_total_relation_size(c.oid) AS total_bytes, pg_relation_size(c.oid) AS table_bytes,
        pg_indexes_size(c.oid) AS index_bytes, greatest(c.reltuples, 0)::bigint AS rows_est, s.n_live_tup AS live, s.n_dead_tup AS dead,
        s.seq_scan, s.idx_scan, s.n_tup_ins AS inserts, s.n_tup_upd AS updates, s.n_tup_del AS deletes,
        greatest(s.last_autovacuum, s.last_vacuum) AS last_vacuum, greatest(s.last_autoanalyze, s.last_analyze) AS last_analyze
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE c.relkind = 'r' AND n.nspname = 'public' ORDER BY pg_total_relation_size(c.oid) DESC LIMIT 40""")
    count = await one("SELECT count(*) AS tables FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind = 'r' AND n.nspname = 'public'")
    vectors = await many("""SELECT i.relname AS name, t.relname AS table_name, am.amname AS method, pg_relation_size(i.oid) AS bytes,
        coalesce(s.idx_scan, 0) AS scans FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_class t ON t.oid = x.indrelid
        JOIN pg_am am ON am.oid = i.relam LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid
        WHERE am.amname IN ('hnsw', 'ivfflat') ORDER BY bytes DESC""")
    unused = await many("""SELECT s.indexrelname AS name, s.relname AS table_name, pg_relation_size(s.indexrelid) AS bytes FROM pg_stat_user_indexes s
        JOIN pg_index x ON x.indexrelid = s.indexrelid WHERE s.idx_scan = 0 AND NOT x.indisunique AND NOT x.indisprimary
        ORDER BY pg_relation_size(s.indexrelid) DESC LIMIT 10""")
    ext = await many("SELECT extname AS name, extversion AS version FROM pg_extension ORDER BY extname")
    locks = await one("SELECT count(*) FILTER (WHERE NOT granted) AS waiting, count(*) AS held FROM pg_locks")
    for r in [db, stat, *states, *tables, *vectors]:
        for k, v in list(r.items()):
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
            elif v is not None and not isinstance(v, (int, float, str, bool)):
                r[k] = float(v) if str(v).replace(".", "", 1).isdigit() else str(v)
    return {"database": db, "activity": {**stat, "cacheHitRatio": round(hit, 4)}, "connections": states, "waits": waits, "locks": locks,
            "tables": tables, "tableCount": count.get("tables", 0), "vectorIndexes": vectors, "unusedIndexes": unused, "extensions": ext}


@router.get("/families/{family_id}/week")
async def admin_family_week(family_id: str, session: DB, days: int = 7) -> dict:
    """The family page's numbers from the conversation itself (turns), per person and per day: messages in, Saheli's
    replies, messages she started, voice notes and photos, and when each person last wrote. Counts only, no text."""
    from datetime import timedelta

    from sqlalchemy import text as sql

    from app.core import clock

    since = clock.now() - timedelta(days=max(1, min(days, 90)))
    rows = (await session.execute(sql("""
        SELECT thread_id, role, (meta ? 'voice_confidence') AS voice, (coalesce(jsonb_array_length(CASE WHEN jsonb_typeof(meta->'images') = 'array' THEN meta->'images' END), 0) > 0) AS photo,
               coalesce((meta->>'proactive')::boolean, false) AS proactive, to_char(at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD') AS day, count(*) AS n, max(at) AS last_at
        FROM turns WHERE family_id = :f AND at >= :s AND thread_id <> 'saheli-scheduler' AND role IN ('user', 'assistant')
        GROUP BY 1, 2, 3, 4, 5, 6"""), {"f": family_id, "s": since})).mappings().all()
    last_in = (await session.execute(sql("""SELECT thread_id, max(at) AS at FROM turns WHERE family_id = :f AND role = 'user' AND thread_id <> 'saheli-scheduler'
        GROUP BY 1"""), {"f": family_id})).mappings().all()
    blank = {"messagesIn": 0, "replies": 0, "proactive": 0, "voiceNotes": 0, "photos": 0}
    people: dict[str, dict] = {}
    daily: dict[str, dict] = {}
    total = dict(blank)
    for r in rows:
        n = int(r["n"])
        keys = (["messagesIn"] + (["voiceNotes"] if r["voice"] else []) + (["photos"] if r["photo"] else [])) if r["role"] == "user" else (["proactive"] if r["proactive"] else ["replies"])
        for target in (people.setdefault(r["thread_id"], dict(blank)), daily.setdefault(r["day"], dict(blank)), total):
            for k in keys:
                target[k] += n
    for r in last_in:
        people.setdefault(r["thread_id"], dict(blank))["lastInAt"] = r["at"].isoformat()
    return {"days": days, "total": total, "people": people, "daily": [{"day": d, **v} for d, v in sorted(daily.items())]}


# ── Insights (founder 21 Oct: "build everything") ──────────────────────────────────
# Counts and money only: no message text leaves these routes (reasons are cut before any quoted words).

_TEST_PREFIXES = ("sim-", "shadow:", "replay-", "founder-", "lab-", "eval-", "world-", "fw-")


def _is_test(fid: str) -> bool:
    return fid.startswith(_TEST_PREFIXES)


def _reason(r: str) -> str:
    import re

    return re.sub(r"[\"“'].*$", "", r or "").strip(" :") or "—"


@router.get("/insights/family-costs")
async def insights_family_costs(session: DB, days: int = 30) -> dict:
    """Model cost per family and per day from each turn's trace (live families named by id; test families grouped), and
    the model spend not tied to a family (night jobs, extraction, judge) as the ledger minus the traced turns."""
    from datetime import timedelta

    from sqlalchemy import text as sql

    from app.core import clock

    since = clock.now() - timedelta(days=max(1, min(days, 120)))
    rows = (await session.execute(sql("""SELECT family_id, kind, to_char(at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD') AS day, count(*) AS turns,
        sum(cost_inr) AS cost, count(DISTINCT thread_id) AS people FROM trace_turns WHERE at >= :s GROUP BY 1, 2, 3"""), {"s": since})).mappings().all()
    ledger = (await session.execute(sql("SELECT day::text AS day, role, sum(cost_inr) AS cost, sum(calls) AS calls FROM llm_spend WHERE day >= :d GROUP BY 1, 2"),
                                    {"d": since.date()})).mappings().all()
    fams: dict[str, dict] = {}
    for r in rows:
        key = "tests and simulations" if _is_test(r["family_id"]) else r["family_id"]
        f = fams.setdefault(key, {"familyId": key, "test": key == "tests and simulations", "costInr": 0.0, "turns": 0, "byKind": {}, "daily": {}})
        c = float(r["cost"] or 0)
        f["costInr"] += c
        f["turns"] += int(r["turns"])
        k = f["byKind"].setdefault(r["kind"], {"turns": 0, "costInr": 0.0})
        k["turns"] += int(r["turns"])
        k["costInr"] += c
        f["daily"][r["day"]] = round(f["daily"].get(r["day"], 0) + c, 2)
    traced_daily: dict[str, float] = {}
    for f in fams.values():
        f["costInr"] = round(f["costInr"], 2)
        f["perTurn"] = round(f["costInr"] / f["turns"], 3) if f["turns"] else 0
        for d, v in f["daily"].items():
            traced_daily[d] = traced_daily.get(d, 0) + v
    ledger_daily: dict[str, float] = {}
    roles: dict[str, dict] = {}
    for r in ledger:
        ledger_daily[r["day"]] = ledger_daily.get(r["day"], 0) + float(r["cost"] or 0)
        x = roles.setdefault(r["role"], {"costInr": 0.0, "calls": 0})
        x["costInr"] = round(x["costInr"] + float(r["cost"] or 0), 2)
        x["calls"] += int(r["calls"] or 0)
    background = {d: round(max(0.0, v - traced_daily.get(d, 0)), 2) for d, v in ledger_daily.items()}
    return {"days": days, "families": sorted(fams.values(), key=lambda f: -f["costInr"]), "ledgerDaily": {k: round(v, 2) for k, v in ledger_daily.items()},
            "backgroundDaily": background, "byRole": roles,
            "note": "The engine's own estimate (tokens × list price); Google's bill is about 1.45× higher. Background = night jobs, extraction, the judge and tests not traced to a family turn."}


@router.get("/insights/people")
async def insights_people(session: DB, days: int = 120) -> dict:
    """Per person: first and last message to Saheli, days active, messages in the last 7 and 30 days, Saheli's messages
    since their last reply, and briefs sent. For the growth funnel and the attention list. Live families only."""
    from datetime import timedelta

    from sqlalchemy import text as sql

    from app.core import clock

    since = clock.now() - timedelta(days=max(7, min(days, 365)))
    rows = (await session.execute(sql("""
        WITH t AS (SELECT family_id, thread_id, role, at, coalesce((meta->>'proactive')::boolean, false) AS proactive FROM turns
                   WHERE at >= :s AND thread_id <> 'saheli-scheduler' AND role IN ('user', 'assistant'))
        SELECT family_id, thread_id,
          min(at) FILTER (WHERE role = 'user') AS first_in, max(at) FILTER (WHERE role = 'user') AS last_in,
          max(at) FILTER (WHERE role = 'assistant') AS last_out,
          count(DISTINCT (at AT TIME ZONE 'Asia/Kolkata')::date) FILTER (WHERE role = 'user') AS active_days,
          count(*) FILTER (WHERE role = 'user' AND at >= now() - interval '7 days') AS in7,
          count(*) FILTER (WHERE role = 'user' AND at >= now() - interval '30 days') AS in30,
          count(*) FILTER (WHERE role = 'user') AS in_all
        FROM t GROUP BY 1, 2"""), {"s": since})).mappings().all()
    unanswered = (await session.execute(sql("""
        SELECT t.family_id, t.thread_id, count(*) AS n FROM turns t
        LEFT JOIN LATERAL (SELECT max(at) AS last_in FROM turns u WHERE u.family_id = t.family_id AND u.thread_id = t.thread_id AND u.role = 'user') u ON true
        WHERE t.role = 'assistant' AND coalesce((t.meta->>'proactive')::boolean, false) AND t.at >= now() - interval '14 days'
          AND NOT (t.meta ? 'reminder_id') AND t.thread_id <> 'saheli-scheduler'
          AND (u.last_in IS NULL OR t.at > u.last_in) GROUP BY 1, 2"""))).mappings().all()  # reminders never stop (founder 16 Oct): not counted
    briefs = (await session.execute(sql("""SELECT family_id, count(*) AS n, max(at) AS last FROM care_events WHERE kind = 'daily_brief'
        AND at >= now() - interval '7 days' GROUP BY 1"""))).mappings().all()
    un = {(r["family_id"], r["thread_id"]): int(r["n"]) for r in unanswered}
    iso = lambda v: v.isoformat() if v else None  # noqa: E731
    people = [{"familyId": r["family_id"], "personId": r["thread_id"], "firstInAt": iso(r["first_in"]), "lastInAt": iso(r["last_in"]),
               "lastOutAt": iso(r["last_out"]), "activeDays": int(r["active_days"] or 0), "in7": int(r["in7"]), "in30": int(r["in30"]),
               "inAll": int(r["in_all"]), "unansweredSinceLastReply": un.get((r["family_id"], r["thread_id"]), 0)}
              for r in rows if not _is_test(r["family_id"])]
    return {"people": people, "briefs": {r["family_id"]: {"sent7": int(r["n"]), "lastAt": iso(r["last"])} for r in briefs if not _is_test(r["family_id"])}}


@router.get("/insights/quality")
async def insights_quality(session: DB, days: int = 30) -> dict:
    """The judge's pass rate per day and kind, the most common issues, replay cases open and fixed, and the trainer's nights."""
    from collections import Counter
    from datetime import timedelta

    from sqlalchemy import text as sql

    from app.core import clock

    since = clock.now() - timedelta(days=max(1, min(days, 120)))
    rows = (await session.execute(sql("""SELECT day, kind, passed, issues, family_id FROM reply_verdicts WHERE at >= :s"""), {"s": since})).mappings().all()
    daily: dict[str, dict] = {}
    issues: Counter = Counter()
    by_kind: dict[str, dict] = {}
    worst_fams: Counter = Counter()
    for r in rows:
        if _is_test(r["family_id"]):
            continue
        d = daily.setdefault(r["day"], {"day": r["day"], "judged": 0, "passed": 0})
        d["judged"] += 1
        d["passed"] += 1 if r["passed"] else 0
        k = by_kind.setdefault(r["kind"], {"judged": 0, "passed": 0})
        k["judged"] += 1
        k["passed"] += 1 if r["passed"] else 0
        if not r["passed"]:
            worst_fams[r["family_id"]] += 1
            for i in (r["issues"] or []):
                issues[str(i)] += 1
    cases = (await session.execute(sql("""SELECT status, kind, to_char(created_at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD') AS day, count(*) AS n
        FROM replay_cases GROUP BY 1, 2, 3"""))).mappings().all()
    case_status: Counter = Counter()
    case_kinds: Counter = Counter()
    case_daily: dict[str, int] = {}
    for c in cases:
        case_status[c["status"]] += int(c["n"])
        if c["status"] == "open":
            case_kinds[c["kind"]] += int(c["n"])
        case_daily[c["day"]] = case_daily.get(c["day"], 0) + int(c["n"])
    trainer = (await session.execute(sql("""SELECT day, status, spent_inr, tried, kept, prs FROM trainer_runs ORDER BY started_at DESC LIMIT 30"""))).mappings().all()
    return {"daily": sorted(daily.values(), key=lambda x: x["day"]), "byKind": by_kind, "issues": issues.most_common(20),
            "familiesWithMostFailures": worst_fams.most_common(10), "cases": dict(case_status), "openByKind": case_kinds.most_common(20),
            "casesCreatedDaily": dict(sorted(case_daily.items())[-days:]),
            "trainer": [{"day": t["day"], "status": t["status"], "spentInr": round(float(t["spent_inr"] or 0), 2), "tried": t["tried"], "kept": t["kept"], "prs": t["prs"] or []} for t in trainer]}


@router.get("/insights/outreach")
async def insights_outreach(session: DB, days: int = 14) -> dict:
    """Everything Saheli started: nudges (planned, sent, held and why, answered), reminders (sent, failed), briefs, and the
    pre-send check's holds. Reasons are cut before any quoted words, so no message text is shown."""
    from collections import Counter
    from datetime import timedelta

    from sqlalchemy import text as sql

    from app.core import clock

    since = clock.now() - timedelta(days=max(1, min(days, 60)))
    ledger = (await session.execute(sql("""SELECT family_id, kind, status, reason, to_char(planned_at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD') AS day,
        replied_at IS NOT NULL AS replied, reaction FROM nudge_ledger WHERE planned_at >= :s"""), {"s": since})).mappings().all()
    events = (await session.execute(sql("""SELECT family_id, kind, to_char(at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD') AS day, summary FROM care_events
        WHERE at >= :s AND kind IN ('reminder_sent', 'reminder_failed', 'daily_brief', 'nudge_held', 'checkin', 'alert_whatsapp', 'alert_dashboard')"""), {"s": since})).mappings().all()
    nudges = {"total": 0, "sent": 0, "skipped": 0, "failed": 0, "answered": 0, "reactions": Counter(), "skipReasons": Counter(), "byKind": {}}
    daily: dict[str, dict] = {}
    blank = {"nudgesSent": 0, "nudgesSkipped": 0, "nudgesAnswered": 0, "remindersSent": 0, "remindersFailed": 0, "briefs": 0, "held": 0, "alerts": 0}
    for r in ledger:
        if _is_test(r["family_id"]):
            continue
        d = daily.setdefault(r["day"], dict(blank))
        nudges["total"] += 1
        k = nudges["byKind"].setdefault(r["kind"], {"sent": 0, "skipped": 0, "answered": 0})
        if r["status"] in ("sent", "skipped", "failed"):
            nudges[r["status"]] += 1
        if r["status"] == "sent":
            d["nudgesSent"] += 1
            k["sent"] += 1
            if r["replied"]:
                nudges["answered"] += 1
                d["nudgesAnswered"] += 1
                k["answered"] += 1
        elif r["status"] == "skipped":
            d["nudgesSkipped"] += 1
            k["skipped"] += 1
            nudges["skipReasons"][_reason(r["reason"])[:80]] += 1
        if r["reaction"]:
            nudges["reactions"][r["reaction"]] += 1
    held: Counter = Counter()
    fail_reasons: Counter = Counter()
    for e in events:
        if _is_test(e["family_id"]):
            continue
        d = daily.setdefault(e["day"], dict(blank))
        if e["kind"] == "reminder_sent":
            d["remindersSent"] += 1
        elif e["kind"] == "reminder_failed":
            d["remindersFailed"] += 1
            fail_reasons[_reason((e["summary"] or "").split("—")[-1])[:80]] += 1
        elif e["kind"] == "daily_brief":
            d["briefs"] += 1
        elif e["kind"] == "nudge_held":
            d["held"] += 1
            held[_reason((e["summary"] or "")[:120])[:80]] += 1
        elif e["kind"].startswith("alert_"):
            d["alerts"] += 1
    nudges["reactions"] = dict(nudges["reactions"])
    nudges["skipReasons"] = nudges["skipReasons"].most_common(15)
    return {"days": days, "nudges": nudges, "daily": [{"day": k, **v} for k, v in sorted(daily.items())], "heldReasons": held.most_common(10),
            "reminderFailReasons": fail_reasons.most_common(10)}


@router.get("/insights/orders")
async def insights_orders(session: DB, days: int = 60) -> dict:
    """Orders and rides per store: how many, done, failed, cancelled, why they failed, how long a done one took, and agent runs."""
    from collections import Counter
    from datetime import timedelta

    from sqlalchemy import text as sql

    from app.core import clock

    since = clock.now() - timedelta(days=max(1, min(days, 180)))
    rows = (await session.execute(sql("""SELECT family_id, service, kind, status, runs, created_at, updated_at, details->>'fail_code' AS code,
        details->'metrics'->>'failure_code' AS code2, to_char(created_at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD') AS day FROM care_tasks WHERE created_at >= :s"""),
        {"s": since})).mappings().all()
    stores: dict[str, dict] = {}
    codes: Counter = Counter()
    daily: dict[str, dict] = {}
    for r in rows:
        if _is_test(r["family_id"]):
            continue
        s = stores.setdefault(r["service"], {"service": r["service"], "kind": r["kind"], "total": 0, "done": 0, "failed": 0, "cancelled": 0, "open": 0,
                                             "runs": 0, "minutes": [], "codes": Counter()})
        s["total"] += 1
        s["runs"] += int(r["runs"] or 0)
        st = r["status"]
        bucket = "done" if st in ("done", "placed", "completed") else "failed" if st == "failed" else "cancelled" if st == "cancelled" else "open"
        s[bucket] += 1
        d = daily.setdefault(r["day"], {"day": r["day"], "total": 0, "done": 0, "failed": 0, "cancelled": 0})
        d["total"] += 1
        if bucket != "open":
            d[bucket] += 1
        if bucket == "done" and r["updated_at"] and r["created_at"]:
            s["minutes"].append((r["updated_at"] - r["created_at"]).total_seconds() / 60)
        code = r["code"] or r["code2"]
        if bucket == "failed":
            s["codes"][code or "unknown"] += 1
            codes[code or "unknown"] += 1
    out = []
    for s in stores.values():
        m = sorted(s.pop("minutes"))
        s["medianMinutes"] = round(m[len(m) // 2], 1) if m else None
        s["successRate"] = round(s["done"] / max(1, s["done"] + s["failed"]) * 100) if (s["done"] + s["failed"]) else None
        s["codes"] = s["codes"].most_common(6)
        out.append(s)
    return {"days": days, "stores": sorted(out, key=lambda x: -x["total"]), "failCodes": codes.most_common(15), "daily": sorted(daily.values(), key=lambda x: x["day"])}


class ReportReplyIn(BaseModel):
    family_id: str = Field(min_length=4, max_length=80)
    contains: str = Field(default="", max_length=200)  # part of Saheli's reply, to find it
    reply_id: int | None = None
    note: str = Field(min_length=8, max_length=600)  # what was wrong and what should have happened
    kind: str = Field(default="", max_length=40)


@router.post("/replay-cases/from-reply")
async def admin_case_from_reply(body: ReportReplyIn, session: DB) -> dict:
    """A bad reply the team reports (founder 21 Oct: "turn bugs I report into cases straight away"): the reply is found
    (by id, or by a part of its text in the last 14 days), its moment is snapshotted and anonymised like every other
    case, and it becomes an open replay case the nightly trainer and every deploy see."""
    from datetime import timedelta

    from app.care.models import FamilyRoster
    from app.core import clock
    from app.learn.models import ReplyLog
    from app.quality import replay, snapshot

    q = select(ReplyLog).where(ReplyLog.family_id == body.family_id)
    if body.reply_id:
        q = q.where(ReplyLog.id == body.reply_id)
    elif body.contains.strip():
        q = q.where(ReplyLog.at >= clock.now() - timedelta(days=14), ReplyLog.text.ilike(f"%{body.contains.strip()}%"))
    else:
        raise HTTPException(400, "give reply_id or part of the reply's text")
    rows = list((await session.execute(q.order_by(ReplyLog.at.desc()).limit(3))).scalars())
    if not rows:
        raise HTTPException(404, "no reply found")
    r = rows[0]
    roster = await session.get(FamilyRoster, r.family_id)
    if roster is None:
        raise HTTPException(404, "family not found")
    snap = await snapshot.take(session, roster, r.thread_id, r.at, message=r.user_text, speaker_role=r.speaker_role, trace=r.trace, kind=r.kind)
    kind = (body.kind.strip() or "founder_report")[:40]
    made = await replay.add_snapshot(session, source="founder", source_ref=f"founder:reply:{r.id}", kind=kind, day=clock.ist(r.at).strftime("%Y-%m-%d"),
                                     snap=snap, message=r.user_text, bad_reply=r.text, assertion=[{"type": "grounded"}],
                                     rubric=f"Reported by the Kavach team: {body.note.strip()}", family_id=r.family_id, subject_id=r.thread_id, thread_id=r.thread_id)
    await session.commit()
    return {"created": bool(made), "replyId": r.id, "at": r.at.isoformat(), "kind": kind, "matches": len(rows),
            "note": None if made else "a case for this reply already exists"}
