"""The context snapshot of a replay case (trust plan C3.2) and its anonymisation (C3.3).

A snapshot holds what is needed to run the turn again on a fresh family: the household, the care record (active and
pending facts, with source and pin), the memory notes, the last turns of the people involved, the day's events up to
then, the message and the time - all as they stood at that moment (app.quality.asof). The pre-deploy runner
(eval/replay_cases.py) seeds a family from it and runs the real brain.

Real families' cases are stored anonymised: names of the household and of everyone in the record become [PERSONn] (the
same placeholder for the same name in every field, the snapshot, the bad reply and the assertions alike), phone
numbers, emails, addresses and pincodes become placeholders, unsaved capitalised names are scrubbed, and person and family
ids become p1, p2, ... and case-<hash>. Words an assertion must check (names never to use, rejected items, medicine
names) are kept as they are unless they are a person's name, so the case still tests the same thing.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care.models import CareEvent, FamilyRoster, Turn
from app.core import clock
from app.quality.asof import memory_at

TURNS = 30
PERSON_KEYS = ("name", "person", "doctor", "doctor_name", "contact", "helper", "caregiver")
PEOPLE_DOMAINS = ("family", "contact", "doctor", "helper", "person", "caregiver")


async def take(session: AsyncSession, roster: FamilyRoster, thread_id: str, at: datetime, *, message: str, speaker_role: str = "",
               trace: list | None = None, kind: str = "reply") -> dict:
    from app.quality.judge import memory_block, names_of, people_of

    elder, members = people_of(roster)
    names = names_of(elder, members)
    people = [p for p in dict.fromkeys([elder.get("id") or "", thread_id]) if p]
    mem = await memory_at(session, roster.family_id, people, at)
    facts = [{"subject": sid, "domain": f.domain, "key": f.key, "value": dict(f.value or {}), "text": f.text, "status": f.status,
              "source_kind": f.source_kind, "pinned": f.pinned_at is not None} for sid, rows in mem.facts.items() for f in rows]
    rejected = [{"subject": sid, "domain": f.domain, "key": f.key, "value": dict(f.value or {}), "text": f.text, "status": f.status}
                for sid, rows in mem.rejected.items() for f in rows]
    notes = [{"subject": n.subject_id, "slug": n.slug, "title": n.title, "body": n.body_md} for n in mem.notes]
    turns = []
    for th in people:
        rows = list((await session.execute(select(Turn).where(Turn.family_id == roster.family_id, Turn.thread_id == th, Turn.at < at)
                                           .order_by(Turn.id.desc()).limit(TURNS))).scalars())[::-1]
        turns += [{"thread": t.thread_id, "role": t.role, "text": (t.text or "")[:1500], "at": t.at.isoformat(), "speaker": t.speaker_id} for t in rows]
    day = clock.ist(at).strftime("%Y-%m-%d")
    evs = list((await session.execute(select(CareEvent).where(CareEvent.family_id == roster.family_id, CareEvent.subject_id.in_(people),
                                                              CareEvent.day == day, CareEvent.at <= at).order_by(CareEvent.at).limit(80))).scalars())
    events = [{"subject": e.subject_id, "kind": e.kind, "summary": (e.summary or "")[:300], "at": e.at.isoformat()} for e in evs]
    return {"v": 1, "time": at.isoformat(), "family_id": roster.family_id, "kind": kind, "elder": elder, "members": members,
            "speaker": {"id": thread_id, "name": names.get(thread_id), "role": speaker_role}, "facts": facts, "rejected": rejected, "notes": notes,
            "turns": turns, "events": events, "message": message or "", "trace": (trace or [])[-30:],
            "rendered": await memory_block(mem, elder, thread_id, names)}


def names_in(snap: dict) -> list[str]:
    """Every person's name the snapshot knows: the household and the record's people (doctors, contacts, family)."""
    out = [str(p.get("name") or "") for p in [snap.get("elder") or {}, *(snap.get("members") or [])]]
    for f in snap.get("facts") or []:
        if f.get("domain") not in PEOPLE_DOMAINS:
            continue  # a medicine's or a dish's "name" is not a person's
        v = f.get("value") or {}
        out += [str(v[k]) for k in PERSON_KEYS if isinstance(v.get(k), str)]
    return [n for n in dict.fromkeys(x.strip() for x in out) if len(n) >= 2]


def keep_words(snap: dict, assertion: list[dict]) -> list[str]:
    """Words the assertions check (never-use names, rejected items, medicine names) that are not a person's name."""
    from app.care import rejected as care_rejected

    words: list[str] = []
    for a in assertion or []:
        words += [str(w) for w in a.get("words") or []]
        if a.get("medicine"):
            words.append(str(a["medicine"]))
    for f in snap.get("facts") or []:
        v = f.get("value") or {}
        if f.get("domain") == "naming":
            words += [str(w) for w in v.get("avoid") or []]
        if f.get("domain") == "medicine" and isinstance(v.get("name"), str):
            words.append(v["name"])
    words += care_rejected.terms([SimpleNamespace(value=r.get("value") or {}) for r in snap.get("rejected") or []])  # type: ignore[arg-type]
    person = {w.lower() for n in names_in(snap) for w in n.split()}
    return [w for w in dict.fromkeys(words) if w and w.lower() not in person and not any(p in w.lower().split() for p in person)]


def anonymise(obj, *, names: list[str], keep: list[str], ids: dict[str, str]):
    """The same mapping over every string inside (dicts, lists): ids, then names and numbers, keeping `keep` words."""
    from app.learn.anonymise import anonymise as scrub

    keep = sorted(set(keep), key=len, reverse=True)

    def one(text: str) -> str:
        for real, fake in ids.items():
            if real:
                text = text.replace(real, fake)
        held: dict[str, str] = {}

        def hold(m: re.Match) -> str:  # each match keeps its own spelling (metformin in a key, Metformin in a sentence)
            tok = "\ue000" + str(len(held)) + "\ue001"
            held[tok] = m.group(0)
            return tok

        for w in keep:
            text = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", hold, text, flags=re.IGNORECASE)
        text = scrub(text, names=names, brand_medicines=False)
        for tok, w in held.items():
            text = text.replace(tok, w)
        return text

    def walk(x):
        if isinstance(x, str):
            return one(x)
        if isinstance(x, list):
            return [walk(i) for i in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return walk(obj)


def ids_for(snap: dict) -> dict[str, str]:
    people = [str(p.get("id") or "") for p in [snap.get("elder") or {}, *(snap.get("members") or [])]]
    out = {pid: f"p{i + 1}" for i, pid in enumerate(dict.fromkeys(p for p in people if p))}
    fid = str(snap.get("family_id") or "")
    if fid:
        out[fid] = "case-" + hashlib.sha256(fid.encode()).hexdigest()[:8]
    return dict(sorted(out.items(), key=lambda kv: -len(kv[0])))


def anonymised_case(snap: dict, *, message: str, bad_reply: str, assertion: list[dict], rubric: str) -> dict:
    """{"context", "message", "bad_reply", "assertion", "rubric"} with one consistent mapping."""
    names, keep, ids = names_in(snap), keep_words(snap, assertion), ids_for(snap)
    whole = anonymise({"context": snap, "message": message, "bad_reply": bad_reply, "assertion": assertion, "rubric": rubric},
                      names=names, keep=keep, ids=ids)
    return whole if isinstance(whole, dict) else {}

