"""The slow memory path.

During a turn the brain saves what it notices. Afterwards this reads the conversation again with
fresh eyes and catches what it missed. Health facts it finds are never put into effect on its word:
they wait as pending for a caregiver. Nightly, the narrative notes are rewritten into clean files.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import dedupe, digest, store
from app.care.domains import DOMAINS, HEALTH_DOMAINS, NONE_DOMAINS, fact_key, is_none_key, merged_value, slug
from app.care.models import CareFact, MemoryNote, Turn
from app.brain.host import ShadowHost
from app.care.redact import scrub_secrets
from app.core import clock
from app.llm import router

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = f"""You maintain the care memory for a person cared for by Saheli, their care companion: an elder whose family looks after them, or someone caring for themselves (then the person and their caregiver are the same, and PEOPLE lists only them). You read a conversation between Saheli and the people in PEOPLE, and list what should be remembered that the care record below does not already say.

Nothing they say about their life is lost: a plan, a worry, a person they mention, a place, a preference, a habit, a goal, something they did. If it is not a fact for the record, it is a note or a plan.

Only include what one of these people actually said or clearly confirmed. Saheli's lines are context only: never take a fact from what Saheli said (her reminders and summaries repeat the record). Do not infer diagnoses, guess doses or times, or record one-off chit-chat. Skip anything already in the care record unless it actually changed; a restatement of the same medicine is not a change. A fact about a caregiver's own life (their gym, their medicine, their diet) is about them: give their person id in "about", or their name when only one person in PEOPLE has it. Omit "about" only for the care recipient. Someone who is not in PEOPLE is not given a medicine, dose, reading, plan or reminder, and it is not saved on the care recipient: put a note instead.

Domains (put each fact in the one that fits; no_order is only for an item never to buy again):
{chr(10).join(f"- {d}: {what}" for d, what in sorted(DOMAINS.items()))}

When they say they have none (no regular medicines, no allergies, no long-term conditions), give that domain with "none": true and a sentence, e.g. {{"domain": "medicine", "none": true, "sentence": "Takes no regular medicines."}}.

A fact may carry links a family member said: a medicine is "for" a condition or "prescribed_by" a doctor, an appointment is "with_doctor" a doctor, a person is "related_to" another (as: son, neighbour). Give a link only when it was said, even for a fact already in the record (then repeat the fact as it is, with the link).

Return only JSON:
{{"facts": [{{"domain": "...", "name": "...", "none": false, "details": {{}}, "sentence": "one plain sentence", "about": "person id or omit for the care recipient", "evidence": "the words that show it", "links": [{{"relation": "for|prescribed_by|with_doctor|related_to", "to_domain": "condition|doctor|contact|family|person", "to_name": "...", "as": "related_to only"}}]}}],
  "stops": [{{"domain": "...", "name": "...", "reason": "...", "about": "person id or omit for the care recipient", "evidence": "..."}}],
  "notes": [{{"about": "person id or family", "topic": "short topic", "text": "what to remember about their life"}}],
  "readings": [{{"about": "person id", "kind": "weight|height|bmi|body_fat|muscle_mass|waist|bp|sugar|pulse|spo2|temperature|other", "name": "for other", "value": "73.5", "unit": "kg", "taken_on": "YYYY-MM-DD"}}],
  "plans": [{{"about": "person id", "what": "what will happen, one line", "when": "YYYY-MM-DDTHH:MM or null", "ask_after": "what to ask afterwards"}}],
  "moods": [{{"about": "person id", "score": 1, "word": "one English word, e.g. cheerful, calm, tired, worried, sad, irritated, lonely", "summary": "how they seemed and why, one line", "evidence": "their words"}}]}}
readings: every measurement a person gave in these messages (typed, said, or that Saheli read out from their photo): weight, height (cm), BMI, body fat, BP, sugar, pulse, SpO2, temperature, muscle mass, waist, or other with its name. taken_on is YYYY-MM-DD when it was measured.
plans: things a person said they will do or that will happen (a call, a visit, a delivery, "trainer will send a diet plan tonight", an exam); Saheli follows up afterwards. when is YYYY-MM-DDTHH:MM IST if they said when, else null.
moods: how each person (never Saheli) seemed in this conversation, only when it shows in what they wrote; one per person, the latest; score 1 very low, 2 low, 3 okay, 4 good, 5 very good.
Use empty lists when there is nothing."""


BATCH = 120  # turns read at once


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


host_for_readings = None  # tests swap in a sim host


def filed_on(about, elder: dict, members: list[dict]) -> str | None:
    """The member this is about. None means `about` names someone who is not a member: do not file it on the care recipient."""
    from app.care.members import UnknownMember, resolve_member

    if not str(about or "").strip():
        return elder["id"]
    try:
        return resolve_member(str(about), elder=elder, members=members)
    except UnknownMember:
        return None


async def extract_family(session: AsyncSession, family_id: str) -> dict:
    roster = await store.roster(session, family_id)
    if not roster:
        return {"skipped": "no roster"}
    elder = roster.elder
    pending = list(
        (
            await session.execute(
                select(Turn).where(Turn.family_id == family_id, Turn.extracted.is_(False)).order_by(Turn.id).limit(BATCH)
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
        if t.thread_id != "saheli-scheduler" and not (t.role == "assistant" and (t.meta or {}).get("proactive"))
    )
    from app.care import links

    from app.care import rejected as care_rejected

    record = digest.care_record(elder.get("name", ""), await store.facts(session, family_id, elder["id"]),
                                links=await links.links_for(session, family_id, elder["id"]),
                                rejected=await care_rejected.rejected(session, family_id, elder["id"]))
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

    async def save_links(f: dict, subject: str, key: str, with_fact=None) -> None:
        # what a re-read heard about how it connects: a health link always waits for a caregiver (app.care.links)
        if isinstance(f.get("links"), list) and f["links"]:
            got = await links.save_many(session, family_id=family_id, subject_id=subject, from_ref=key, items=f["links"][:6],
                                        source_kind="inferred", stated_by=None, source_ref=None, with_fact=with_fact)
            counts["links"] = counts.get("links", 0) + len(got["links"])

    for f in data.get("facts") or []:
        domain = f.get("domain")
        if f.get("none") is True and domain in NONE_DOMAINS:
            f = {**f, "name": "none", "details": {"none": True}, "links": []}  # "has none": never a medicine or an item to avoid
        if domain not in DOMAINS or not f.get("name") or not f.get("sentence"):
            continue
        subject = filed_on(f.get("about"), elder, roster.members)
        if subject is None:
            counts["not_member"] = counts.get("not_member", 0) + 1
            continue
        if is_none_key(fact_key(domain, f["name"])) and [x for x in await store.facts(session, family_id, subject, domains=[domain], statuses=("active", "pending"))
                                                         if not is_none_key(x.key)]:
            continue  # the record lists real ones: "none" is for an empty list
        key = fact_key(domain, f["name"])
        same = await dedupe.check(session, family_id=family_id, subject_id=subject, domain=domain, key=key, text=f["sentence"],
                                  value=f.get("details") or {})
        if same and same.rejected:
            continue  # the family said this is wrong: a re-read never proposes it again (trust plan A3)
        if same:  # said before in other words: the saved fact, with every detail of both
            key, f = same.fact.key, {**f, "sentence": same.text}
        existing = await store.active_fact(session, family_id, subject, key)
        details = f.get("details") or {}
        if existing and store._same(existing.value, merged_value(existing.value, details)):
            await save_links(f, subject, existing.key)  # nothing new in the fact; a link said about it may be
            continue  # a restatement, nothing new
        if existing:
            key = existing.key
        dup = (await session.execute(select(CareFact.id).where(CareFact.family_id == family_id, CareFact.subject_id == subject, CareFact.key == key, CareFact.status == "pending"))).first()
        if domain in HEALTH_DOMAINS and dup:
            await save_links(f, subject, key, with_fact=dup[0])
            continue  # already waiting for a caregiver
        if domain in HEALTH_DOMAINS:
            # A health fact from a re-read waits for a caregiver, new or changed.
            row = CareFact(
                family_id=family_id, subject_id=subject, domain=domain, key=key, value=merged_value(existing.value if existing else None, details),
                text=f["sentence"], status="pending", source_kind="inferred", confidence=0.6,
                valid_from=clock.now(), recorded_at=clock.now(), supersedes=existing.id if existing else None,
                note=f"heard: {str(f.get('evidence', ''))[:200]}",
            )
            session.add(row)
            await session.flush()
            from app.care import versions

            await versions.record_fact(session, row, "pending")  # the proposal shows in memory history like any other
            await store.open_loop(
                session, family_id=family_id, subject_id=subject, kind="confirm_fact",
                title=f"Confirm: {f['sentence']}", detail={"fact_id": str(row.id), "key": key, "new": row.value},
                dedupe_key=f"confirm:{subject}:{key}",
            )
            counts["pending"] += 1
            await save_links(f, subject, key, with_fact=row.id)
            continue
        w = await store.write_fact(
            session, family_id=family_id, subject_id=subject, domain=domain, key=key, value=f.get("details") or {},
            text=f["sentence"], source_kind="inferred", confidence=0.7, note=f"heard: {str(f.get('evidence', ''))[:200]}",
        )
        counts["facts"] += w.result in ("created", "superseded")
        counts["pending"] += w.result == "pending"
        await save_links(f, subject, w.fact.key, with_fact=w.fact.id if w.result == "pending" else None)

    for s in data.get("stops") or []:
        if s.get("domain") in DOMAINS and s.get("name"):
            subject = filed_on(s.get("about"), elder, roster.members)
            if subject is None:
                counts["not_member"] = counts.get("not_member", 0) + 1
                continue
            w = await store.stop_fact(
                session, family_id=family_id, subject_id=subject, key=fact_key(s["domain"], s["name"]),
                reason=str(s.get("reason") or "mentioned in conversation"), source_kind="inferred",
            )
            counts["stops"] += bool(w)

    for n in data.get("notes") or []:
        if not n.get("topic") or not n.get("text"):
            continue
        if n.get("about") == "family":
            subject = "family"
        else:
            subject = filed_on(n.get("about"), elder, roster.members)
            if subject is None:
                subject = "family"  # a note about someone who is not a member: not a medicine, and not the care recipient's fact
        existing = {x.slug: x for x in await store.notes(session, family_id, [subject])}
        s_ = slug(n["topic"])
        line = f"- {clock.ist_day()}: {scrub_secrets(n['text']).strip()}"
        if s_ in existing and n["text"].strip().lower() in existing[s_].body_md.lower():
            continue
        body = (existing[s_].body_md.rstrip() + "\n" + line) if s_ in existing else line
        await store.upsert_note(session, family_id=family_id, subject_id=subject, slug=s_, title=n["topic"].strip().title(), body_md=body)
        counts["notes"] += 1

    from app.care import readings

    for r in (data.get("readings") or [])[:12]:
        who = filed_on(r.get("about"), elder, roster.members) if isinstance(r, dict) else None
        if who is None:
            counts["not_member"] = counts.get("not_member", 0) + 1
            continue
        if await readings.save(session, ShadowHost() if family_id.startswith("shadow:") else host_for_readings, family_id,
                               who, r, source="re-read"):
            counts["readings"] = counts.get("readings", 0) + 1

    for pl in (data.get("plans") or [])[:8]:
        if not isinstance(pl, dict) or not str(pl.get("what") or "").strip():
            continue
        subject = filed_on(pl.get("about"), elder, roster.members)
        if subject is None:
            counts["not_member"] = counts.get("not_member", 0) + 1
            continue
        when = None
        try:
            when = datetime.fromisoformat(f"{pl['when']}:00+05:30") if pl.get("when") else None
        except (TypeError, ValueError):
            when = None
        now = clock.now()
        if when and when < now - timedelta(hours=6):
            continue  # long past: nothing to follow up
        wake_at = (when + timedelta(hours=2)) if when else clock.ist(now).replace(hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1)
        what = str(pl["what"]).strip()[:200]
        await store.open_loop(
            session, family_id=family_id, subject_id=subject, kind="plan", owner_id=subject,
            title=f"{what}" + (f" (ask: {str(pl.get('ask_after'))[:120]})" if pl.get("ask_after") else ""),
            detail={"what": what, "when": pl.get("when"), "max_wakes": 1, "expire_at": (wake_at + timedelta(days=2)).isoformat()},
            wake_at=wake_at, dedupe_key=f"plan:{subject}:{slug(what)[:60]}",
        )
        counts["plans"] = counts.get("plans", 0) + 1

    routine_of = {filed_on(f.get("about"), elder, roster.members)
                  for f in (data.get("facts") or []) if isinstance(f, dict) and f.get("domain") in ("routine", "activity")}
    routine_of.discard(None)
    if routine_of:
        # a routine with a time they mentioned becomes an activity with check-ins (founder 2026-10-13)
        try:
            from app.brain.host import LiveHost
            from app.care import activities

            host = ShadowHost() if family_id.startswith("shadow:") else LiveHost()
            for pid in sorted(routine_of):
                made = await activities.adopt(session, host, family_id, pid)
                if made:
                    counts["activities"] = counts.get("activities", 0) + len(made)
        except Exception:  # noqa: BLE001 — the re-read's facts are kept either way
            logger.exception("adopt activities after re-read failed family=%s", family_id)

    from app.care import mood

    said_at = {t.thread_id: t.at for t in pending if t.role == "user"}  # each person's last message in this batch
    for m in data.get("moods") or []:
        score = mood.score_of(m.get("score")) if isinstance(m, dict) else None
        if not score:
            continue
        subject = filed_on(m.get("about"), elder, roster.members)
        if subject is None:
            counts["not_member"] = counts.get("not_member", 0) + 1
            continue
        when = said_at.get(subject) or pending[-1].at
        if clock.now() - when > mood.STALE:
            continue  # a backlog read late: yesterday's mood is not today's
        if await mood.last(session, family_id, subject, since=pending[0].at):
            continue  # Saheli noted their mood in this conversation already
        word, summary = str(m.get("word") or ""), str(m.get("summary") or mood.SCALE[score])
        # this read sees only the newest lines: how they are is read again from the whole day and the problems reported
        # (live 2026-10-13: "no fever" became "relieved" while loose motions went on)
        seen = await mood.assess(session, family_id, str(subject), hint={"score": score, "word": word, "summary": summary}, now=when)
        if seen:
            score, word, summary = seen["score"], seen["word"], seen["summary"]
        if await mood.record(session, family_id=family_id, subject_id=subject, score=score, word=word,
                             summary=summary, source="reread", at=when):
            counts["moods"] = counts.get("moods", 0) + 1

    await session.execute(update(Turn).where(Turn.id.in_([t.id for t in pending])).values(extracted=True))
    await store.record_event(session, family_id=family_id, subject_id=elder["id"], kind="memory_extract", summary=json.dumps(counts))
    if len(pending) >= BATCH:
        counts["more"] = True  # a full batch: the caller reads the next one
    return counts


CONSOLIDATE_PROMPT = """You keep a person's memory notes for their care companion. Rewrite the note below into a clean, well-organised markdown note: merge duplicates, keep every distinct fact, keep dates where they matter, put newer information in place of what it corrects (and say "earlier: …" when the change matters), and drop one-time codes. No headings above level 3. Write only the note."""


# Notes with their own shape, kept small by their writers: dated history (diary, rollups) and the card rebuilt nightly
# from the record. A rewrite would merge dated lines away (the rollups read "- YYYY-MM-DD:" lines).
KEEP_SHAPE = {"diary", "weekly", "monthly", "profile-card"}


async def consolidate_family(session: AsyncSession, family_id: str, *, min_lines: int = 12) -> int:
    rewritten = 0
    for n in list((await session.execute(select(MemoryNote).where(MemoryNote.family_id == family_id))).scalars()):
        if n.slug in KEEP_SHAPE or n.body_md.count("\n") + 1 < min_lines:
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


async def expire_stale_confirmations(session: AsyncSession, *, days: int = 14) -> dict:
    """Pending health facts: one final ask at day 14, archived at day 17 (app.care.pending, trust plan A4). Before, only the
    question expired and the fact stayed pending forever."""
    from app.care import pending

    return await pending.run(session)