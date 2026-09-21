"""Nightly Instinct 'dreaming' job — consolidate inbox facts into entities."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import SessionLocal, engine
from app.db.migrate import run_instinct_migrations
from app.llm.provider import chat_invoke
from app.models.entities import Elder, EntityStatus, FamilyMemory, MemoryEntity
from app.rag.embeddings import embed_text, embeddings_available
from app.rag.entity_body import propose_slug, render_entity_body
from app.rag.gcs_export import export_memory_to_gcs
from app.rag.memory_hash import content_hash
from app.rag.profile_render import render_memory_profile

logger = logging.getLogger(__name__)

DREAM_SYSTEM = """You consolidate eldercare memory facts into one entity markdown file.

Edit the existing body (do not append duplicates):
- Merge new facts into dated bullets
- Shorten verbose lines; generalise repeated examples into traits
- Apply dated corrections; remove superseded lines
- Drop incidental detail that does not help future care
- For medication/condition/symptom: family/clinician/document sources outrank elder on conflicts
- For preference/mood: elder source outranks others
- Unresolvable conflicts → add a "needs-review" bullet, set status needs-review

Return ONLY JSON:
{
  "body_md": "full markdown file with YAML frontmatter",
  "aliases": ["alias1", "alias2"],
  "links": [{"target_slug": "person-medha", "relation": "related"}],
  "status": "active|needs-review",
  "review_by": "YYYY-MM-DD or null"
}"""


async def _ensure_schema() -> None:
    async with engine.begin() as conn:
        await run_instinct_migrations(conn)


async def _resolve_entity_for_fact(
    session: AsyncSession,
    fact: FamilyMemory,
    entities: list[MemoryEntity],
) -> MemoryEntity | None:
    topic = (fact.topic or "general").lower()
    for entity in entities:
        aliases = [a.lower() for a in (entity.aliases or [])]
        if topic in aliases or any(topic in a for a in aliases):
            return entity
        if topic in entity.slug:
            return entity
    return None


async def _dream_entity(
    session: AsyncSession,
    entity: MemoryEntity,
    facts: list[FamilyMemory],
) -> None:
    fact_lines = "\n".join(
        f"- ({f.source_role or 'elder'} {f.created_at.date() if f.created_at else 'unknown'}): {f.content}"
        for f in facts
    )
    prompt = f"Entity slug: {entity.slug}\nKind: {entity.kind}\n\nCurrent body:\n{entity.body_md}\n\nNew facts:\n{fact_lines}"
    try:
        raw = await chat_invoke(DREAM_SYSTEM, prompt)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
    except Exception:
        logger.warning("Dream LLM failed for %s — appending facts only", entity.slug)
        entity.body_md = (entity.body_md or "") + "\n" + fact_lines
        entity.dirty = False
        return

    body_md = str(data.get("body_md") or entity.body_md)
    entity.body_md = body_md[:16000]
    entity.aliases = list({*(entity.aliases or []), *(data.get("aliases") or [])})[:32]
    entity.status = str(data.get("status") or entity.status)
    review = data.get("review_by")
    if review:
        try:
            entity.review_by = date.fromisoformat(str(review)[:10])
        except ValueError:
            pass
    entity.version = int(entity.version or 1) + 1
    entity.dirty = False
    entity.sources = list({*(entity.sources or []), *[str(f.id) for f in facts]})
    if embeddings_available():
        try:
            entity.body_embedding = await embed_text(entity.body_md[:8000])
        except Exception:
            pass

    for fact in facts:
        fact.entity_id = entity.id


async def dream_elder(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    care_context: str = "",
    life_context: str = "",
) -> dict:
    inbox = (
        await session.execute(
            select(FamilyMemory).where(
                FamilyMemory.family_id == family_id,
                FamilyMemory.elder_id == elder_id,
                FamilyMemory.entity_id.is_(None),
                FamilyMemory.forgotten_at.is_(None),
                FamilyMemory.superseded_by.is_(None),
            )
        )
    ).scalars().all()

    stale_entities = (
        await session.execute(
            select(MemoryEntity).where(
                MemoryEntity.family_id == family_id,
                MemoryEntity.elder_id == elder_id,
                MemoryEntity.review_by.isnot(None),
                MemoryEntity.review_by < date.today(),
                MemoryEntity.kind.in_(["medication", "condition", "symptom"]),
            )
        )
    ).scalars().all()
    for entity in stale_entities:
        entity.status = EntityStatus.needs_review.value
        entity.dirty = True

    entities = (
        await session.execute(
            select(MemoryEntity).where(
                MemoryEntity.family_id == family_id,
                MemoryEntity.elder_id == elder_id,
            )
        )
    ).scalars().all()

    touched: dict[uuid.UUID, list[FamilyMemory]] = {}
    for fact in inbox:
        entity = await _resolve_entity_for_fact(session, fact, entities)
        if not entity:
            kind = "health" if fact.category.value == "health" else "preference"
            if fact.category.value == "family":
                kind = "person"
            slug = propose_slug(kind, fact.topic, fact.content[:40])
            entity = MemoryEntity(
                family_id=family_id,
                elder_id=elder_id,
                slug=slug,
                kind=kind,
                title=fact.topic or slug,
                aliases=[fact.topic, slug],
                body_md=render_entity_body(
                    slug=slug,
                    kind=kind,
                    title=fact.topic or slug,
                    aliases=[fact.topic or slug],
                    bullets=[f"- {fact.content} ({fact.created_at.date() if fact.created_at else 'unknown'})"],
                ),
                sources=[str(fact.id)],
            )
            session.add(entity)
            await session.flush()
            entities.append(entity)
        touched.setdefault(entity.id, []).append(fact)

    for entity_id, facts in touched.items():
        entity = await session.get(MemoryEntity, entity_id)
        if entity:
            await _dream_entity(session, entity, facts)

    await render_memory_profile(
        session,
        family_id=family_id,
        elder_id=elder_id,
        life_context=life_context,
        care_context=care_context,
    )
    gcs = await export_memory_to_gcs(session, family_id=family_id, elder_id=elder_id)
    await session.commit()
    return {
        "inbox_processed": len(inbox),
        "entities_touched": len(touched),
        "gcs": gcs,
    }


async def dream_all_elders() -> dict:
    from app.jobs.memory_context import fetch_dream_context

    await _ensure_schema()
    results: list[dict] = []
    async with SessionLocal() as session:
        elders = (await session.execute(select(Elder))).scalars().all()
        for elder in elders:
            try:
                life_context, care_context = await fetch_dream_context(
                    family_id=elder.family_id,
                    elder_id=elder.id,
                )
                stats = await dream_elder(
                    session,
                    family_id=elder.family_id,
                    elder_id=elder.id,
                    life_context=life_context,
                    care_context=care_context,
                )
                results.append({"elder_id": str(elder.id), **stats})
            except Exception:
                logger.exception("Dream failed for elder %s", elder.id)
                await session.rollback()
    return {"elders": len(results), "results": results}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Kavach memory dream job")
    parser.add_argument("--all", action="store_true", help="Dream all elders")
    parser.add_argument("--elder-id", type=str, help="Single elder UUID")
    args = parser.parse_args()

    if args.all:
        out = asyncio.run(dream_all_elders())
        print(json.dumps(out, indent=2))
        return

    if args.elder_id:
        async def _one() -> None:
            await _ensure_schema()
            async with SessionLocal() as session:
                elder = await session.get(Elder, uuid.UUID(args.elder_id))
                if not elder:
                    raise SystemExit("Elder not found")
                from app.jobs.memory_context import fetch_dream_context

                life_context, care_context = await fetch_dream_context(
                    family_id=elder.family_id,
                    elder_id=elder.id,
                )
                out = await dream_elder(
                    session,
                    family_id=elder.family_id,
                    elder_id=elder.id,
                    life_context=life_context,
                    care_context=care_context,
                )
                print(json.dumps(out, indent=2))

        asyncio.run(_one())
        return

    parser.print_help()


if __name__ == "__main__":
    main()
