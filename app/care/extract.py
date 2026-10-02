"""The slow memory path.

During a turn the brain saves what it notices. Afterwards this reads the conversation again with
fresh eyes and catches what it missed. Health facts it finds are never put into effect on its word:
they wait as pending for a caregiver. Nightly, the narrative notes are rewritten into clean files.
"""

from __future__ import annotations

import json
import logging
import re

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import digest, store
from app.care.domains import DOMAINS, HEALTH_DOMAINS, fact_key, slug
from app.care.models import CareFact, MemoryNote, OpenLoop, Turn
from app.care.redact import scrub_secrets
from app.core import clock
from app.llm import router

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = f"""You maintain the care memory for an elderly person's family. You read a conversation between Saheli (their care companion) and family members, and list what should be remembered that the care record below does not already say.

Only include what someone actually said or clearly confirmed. Do not infer diagnoses, guess doses, or record one-off chit-chat. Skip anything already in the care record unless it changed.

Domains: {", ".join(sorted(DOMAINS))}.

Return only JSON:
{{"facts": [{{"domain": "...", "name": "...", "details": {{}}, "sentence": "one plain sentence", "about": "person id or omit for the care recipient", "evidence": "the words that show it"}}],
  "stops": [{{"domain": "...", "name": "...", "reason": "...", "evidence": "..."}}],
  "notes": [{{"about": "person id or family", "topic": "short topic", "text": "what to remember about their life"}}]}}
Use empty lists when there is nothing."""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def families_with_new_turns(session: AsyncSession, limit: int = 50) -> list[str]:
    rows = await session.execute(
        select(Turn.family_id).where(Turn.extracted.is_(False), Turn.role == "user").group_by(Turn.family_id).limit(limit)
    )
    return [r[0] for r in rows]


async def extract_family(session: AsyncSession, family_id: str) -> dict:
    roster = await store.roster(session, family_id)
    if not roster:
        return {"skipped": "no roster"}
    elder = roster.elder
    pending = list(
        (
            await session.execute(
                select(Turn).where(Turn.family_id == family_id, Turn.extracted.is_(False)).order_by(Turn.id).limit(120)
            )
        ).scalars()
    )
    if not pending:
        return {"skipped": "nothing new"}
    names = {m["id"]: m.get("name") or m["id"] for m in roster.members} | {elder["id"]: elder.get("name") or "elder"}
    transcript = "\n".join(
        f"{clock.ist(t.at).strftime('%d %b %H:%M')} {'Saheli' if t.role == 'assistant' else names.get(t.thread_id, t.thread_id)} "
        f"[{t.thread_id}]: {scrub_secrets(t.text)}"
        for t in pending
        if t.thread_id != "saheli-scheduler"
    )
    record = digest.care_record(elder.get("name", ""), await store.facts(session, family_id, elder["id"]))
    people = "\n".join(f"{pid}: {n}" for pid, n in names.items())
    reply = await router.complete(
        "extract",
        system_stable=EXTRACT_PROMPT,
        messages=[{"role": "user", "content": [{"type": "text", "text": f"PEOPLE\n{people}\n\n{record}\n\nCONVERSATION\n{transcript}"}]}],
        max_tokens=4000,
        effort="low",
    )
    data = parse_json(reply.text)
    counts = {"facts": 0, "pending": 0, "stops": 0, "notes": 0}
    member_ids = set(names)

    for f in data.get("facts") or []:
        domain = f.get("domain")
        if domain not in DOMAINS or not f.get("name") or not f.get("sentence"):
            continue
        subject = f.get("about") if f.get("about") in member_ids else elder["id"]
        key = fact_key(domain, f["name"])
        existing = await store.active_fact(session, family_id, subject, key)
        if domain in HEALTH_DOMAINS and not (existing and existing.value == (f.get("details") or {})):
            # A health fact from a re-read waits for a caregiver, new or changed.
            row = CareFact(
                family_id=family_id, subject_id=subject, domain=domain, key=key, value=f.get("details") or {},
                text=f["sentence"], status="pending", source_kind="inferred", confidence=0.6,
                valid_from=clock.now(), recorded_at=clock.now(), supersedes=existing.id if existing else None,
                note=f"heard: {str(f.get('evidence', ''))[:200]}",
            )
            session.add(row)
            await session.flush()
            await store.open_loop(
                session, family_id=family_id, subject_id=subject, kind="confirm_fact",
                title=f"Confirm: {f['sentence']}", detail={"fact_id": str(row.id), "key": key, "new": row.value},
                dedupe_key=f"confirm:{subject}:{key}",
            )
            counts["pending"] += 1
            continue
        w = await store.write_fact(
            session, family_id=family_id, subject_id=subject, domain=domain, key=key, value=f.get("details") or {},
            text=f["sentence"], source_kind="inferred", confidence=0.7, note=f"heard: {str(f.get('evidence', ''))[:200]}",
        )
        counts["facts"] += w.result in ("created", "superseded")
        counts["pending"] += w.result == "pending"

    for s in data.get("stops") or []:
        if s.get("domain") in DOMAINS and s.get("name"):
            w = await store.stop_fact(
                session, family_id=family_id, subject_id=elder["id"], key=fact_key(s["domain"], s["name"]),
                reason=str(s.get("reason") or "mentioned in conversation"), source_kind="inferred",
            )
            counts["stops"] += bool(w)

    for n in data.get("notes") or []:
        if not n.get("topic") or not n.get("text"):
            continue
        subject = "family" if n.get("about") == "family" else (n.get("about") if n.get("about") in member_ids else elder["id"])
        existing = {x.slug: x for x in await store.notes(session, family_id, [subject])}
        s_ = slug(n["topic"])
        line = f"- {clock.ist_day()}: {scrub_secrets(n['text']).strip()}"
        if s_ in existing and n["text"].strip().lower() in existing[s_].body_md.lower():
            continue
        body = (existing[s_].body_md.rstrip() + "\n" + line) if s_ in existing else line
        await store.upsert_note(session, family_id=family_id, subject_id=subject, slug=s_, title=n["topic"].strip().title(), body_md=body)
        counts["notes"] += 1

    await session.execute(update(Turn).where(Turn.id.in_([t.id for t in pending])).values(extracted=True))
    await store.record_event(session, family_id=family_id, subject_id=elder["id"], kind="memory_extract", summary=json.dumps(counts))
    return counts


CONSOLIDATE_PROMPT = """You keep a person's memory notes for their care companion. Rewrite the note below into a clean, well-organised markdown note: merge duplicates, keep every distinct fact, keep dates where they matter, put newer information in place of what it corrects (and say "earlier: …" when the change matters), and drop one-time codes. No headings above level 3. Write only the note."""


async def consolidate_family(session: AsyncSession, family_id: str, *, min_lines: int = 12) -> int:
    rewritten = 0
    for n in list((await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id))).scalars()):
        if n.body_md.count("\n") + 1 < min_lines:
            continue
        reply = await router.complete(
            "extract",
            system_stable=CONSOLIDATE_PROMPT,
            messages=[{"role": "user", "content": [{"type": "text", "text": f"# {n.title}\n{n.body_md}"}]}],
            max_tokens=3000,
            effort="low",
        )
        body = scrub_secrets(reply.text.strip())
        if len(body) < 0.3 * len(n.body_md):
            logger.warning("consolidation dropped too much, kept old note family=%s slug=%s", family_id, n.slug)
            continue
        await store.record_event(
            session, family_id=family_id, subject_id=n.subject_id, kind="note_rewritten", summary=n.title,
            payload={"slug": n.slug, "previous": n.body_md},
        )
        await store.upsert_note(session, family_id=family_id, subject_id=n.subject_id, slug=n.slug, title=n.title, body_md=body)
        rewritten += 1
    return rewritten


async def expire_stale_confirmations(session: AsyncSession, *, days: int = 14) -> int:
    from datetime import timedelta

    res = await session.execute(
        update(OpenLoop)
        .where(OpenLoop.kind == "confirm_fact", OpenLoop.status == "open", OpenLoop.created_at < clock.now() - timedelta(days=days))
        .values(status="expired", updated_at=clock.now(), closed_note="no caregiver answer")
    )
    return res.rowcount or 0
