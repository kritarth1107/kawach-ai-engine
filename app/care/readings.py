"""Readings (weight, BMI, body fat, BP, sugar…) that reach the person's Health Records, from wherever they were heard: a
re-read of the chat, or a one-time recovery from what was kept only as a note (founder 2026-10-13: "if someone has sent
weight or BMI or anything those should go to health records"). The brain logs them live with log_vital."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.care.models import CareEvent, CareFact, MemoryNote
from app.core import clock
from app.llm import router

logger = logging.getLogger(__name__)

KINDS = ("bp", "sugar", "weight", "height", "bmi", "body_fat", "muscle_mass", "waist", "temperature", "spo2", "pulse", "other")


def _num(value) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))  # data check on a number, never on people's words
    return float(m.group()) if m else None


def _label(r: dict) -> str:
    return str(r.get("name") or "").strip().lower() if r.get("kind") == "other" else r["kind"]


async def _seen(session: AsyncSession, family_id: str, subject_id: str, r: dict, at: datetime) -> bool:
    """The same reading (kind and number) already logged within three days of when it was measured."""
    rows = (await session.execute(select(CareEvent.payload).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "vital",
        CareEvent.at >= at - timedelta(days=3), CareEvent.at <= at + timedelta(days=3)))).scalars()
    want = (_label(r), str(r.get("value")).strip() if r["kind"] == "bp" else _num(r.get("value")))
    for p in rows:
        p = p or {}
        if not p.get("kind"):
            continue
        got = (_label(p), str(p.get("value")).strip() if p["kind"] == "bp" else _num(p.get("value")))
        if got == want:
            return True
    return False


async def save(session: AsyncSession, host, family_id: str, subject_id: str, r: dict, *, source: str) -> bool:
    """Log one reading (engine ledger + backend Health Records) unless it is already there. host None: the live backend."""
    kind = str(r.get("kind") or "")
    if kind not in KINDS or _num(r.get("value")) is None or (kind == "other" and not str(r.get("name") or "").strip()):
        return False
    at = clock.now()
    if r.get("taken_on"):
        try:
            day = datetime.strptime(str(r["taken_on"])[:10], "%Y-%m-%d").date()
        except ValueError:
            day = None
        if day and day > clock.ist().date():
            return False
        if day and day < clock.ist().date():
            at = datetime.combine(day, time(12, 0), tzinfo=clock.IST)
    clean = {"kind": kind, "value": str(r["value"]).strip()[:30], "unit": str(r.get("unit") or "").strip()[:12] or None,
             "name": str(r.get("name") or "").strip()[:40] or None if kind == "other" else None}
    if await _seen(session, family_id, subject_id, clean, at):
        return False
    label = clean["name"] if kind == "other" else kind
    summary = f"{label} {clean['value']}{(' ' + clean['unit']) if clean['unit'] else ''} (from {source})"
    await store.record_event(session, family_id=family_id, subject_id=subject_id, kind="vital", summary=summary,
                             payload={**{k: v for k, v in clean.items() if v}, "source": source}, actor_id="saheli", at=at)
    if host is None:
        from app.brain.host import LiveHost

        host = LiveHost()
    try:
        await host.call("log_vitals", {**{k: v for k, v in clean.items() if v}, "note": f"from {source}",
                                       "taken_on": clock.ist(at).date().isoformat() if at.date() != clock.now().date() else None},
                        family_id=family_id, subject_id=subject_id, actor_id=subject_id)
    except Exception:  # noqa: BLE001 — the engine keeps it; the next recovery run sends it again
        logger.exception("log_vitals failed family=%s", family_id)
    return True


RECOVER_PROMPT = """You find measurements of a person's body that were kept only as words, so they can go to their health
records. Below are what is saved about them (facts and notes, each with the day it was written) and the readings already
in their records. List every measurement that is NOT already in the readings: weight, height, BMI, body fat, muscle mass,
waist, BP, sugar, pulse, SpO2, temperature, or other with its name (e.g. visceral fat). Height in cm (5 ft 3 in is 160
cm; "5.3ft" means 5 ft 3 in). taken_on is the day it was measured if said, else the day it was written. Only what is
written there; nothing guessed. JSON only:
{"readings": [{"kind": "weight", "name": null, "value": "73.5", "unit": "kg", "taken_on": "YYYY-MM-DD"}]}"""


async def recover(session: AsyncSession, family_id: str, subject_id: str, *, apply: bool, host=None) -> list[dict]:
    """One person: readings kept only in facts or notes, logged now (apply) or listed (dry run)."""
    facts = (await session.execute(select(CareFact).where(CareFact.family_id == family_id, CareFact.subject_id == subject_id,
                                                          CareFact.status == "active"))).scalars().all()
    notes = (await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id, MemoryNote.subject_id == subject_id))).scalars().all()
    have = (await session.execute(select(CareEvent.at, CareEvent.summary).where(
        CareEvent.family_id == family_id, CareEvent.subject_id == subject_id, CareEvent.kind == "vital"))).all()
    saved = "\n".join(f"[{clock.ist(f.recorded_at).date()}] {f.text}" for f in facts if f.text)
    saved += "\n" + "\n".join(f"[note {n.title}, updated {clock.ist(n.updated_at).date()}]\n{n.body_md[:3000]}" for n in notes)
    if not saved.strip():
        return []
    readings = "\n".join(f"[{clock.ist(a).date()}] {s}" for a, s in have) or "(none)"
    reply = await router.complete("extract", system_stable=RECOVER_PROMPT, max_tokens=1500, effort="low",
                                  messages=[{"role": "user", "content": [{"type": "text", "text": f"SAVED\n{saved}\n\nREADINGS ALREADY IN THEIR RECORDS\n{readings}"}]}])
    m = re.search(r"\{.*\}", reply.text or "", re.S)
    try:
        found = [r for r in (json.loads(m.group()) if m else {}).get("readings") or [] if isinstance(r, dict)]
    except ValueError:
        return []
    out = []
    for r in found[:20]:
        if apply:
            if await save(session, host, family_id, subject_id, r, source="saved notes"):
                out.append(r)
        elif str(r.get("kind")) in KINDS and _num(r.get("value")) is not None:
            out.append(r)
    return out
