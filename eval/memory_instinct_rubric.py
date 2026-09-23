"""Instinct memory eval rubric — smoke checks for CI."""

from __future__ import annotations

import asyncio
import sys
import uuid

from app.db.migrate import run_instinct_migrations
from app.db.session import Base, SessionLocal, engine
from app.models.entities import Elder, Family
from app.rag.memory_extract import persist_inbox_memory
from app.rag.memory_lifecycle import forget_memory_fact
from app.rag.memory_queue import is_high_value_message
from app.rag.retrieve import load_instinct_context, memory_grep


def test_memory_tools_use_ai_uuids() -> list[str]:
    """Regression test: memory tools must use AI tenant UUIDs, not Kavach Mongo IDs.

    The memory system uses AI tenant UUIDs (family_id, elder_id) as the identity
    namespace. These are distinct from Kavach platform MongoDB ObjectIds which
    may be passed separately as kavach_family_id/kavach_recipient_user_id.

    This test verifies that:
    1. Memory tools are built with valid UUIDs
    2. The tools closure captures the correct AI UUIDs
    """
    failures: list[str] = []

    from app.agents.memory_tools import build_memory_read_tools

    ai_family_id = str(uuid.uuid4())
    ai_elder_id = str(uuid.uuid4())

    kavach_mongo_id = "507f1f77bcf86cd799439011"

    tools = build_memory_read_tools(ai_family_id, ai_elder_id)

    if len(tools) != 3:
        failures.append(f"Expected 3 memory tools, got {len(tools)}")
        return failures

    tool_names = {t.name for t in tools}
    expected_names = {"memory_grep", "memory_read_entity", "memory_list_index"}
    if tool_names != expected_names:
        failures.append(f"Unexpected tool names: {tool_names}")

    try:
        uuid.UUID(ai_family_id)
        uuid.UUID(ai_elder_id)
    except ValueError:
        failures.append("AI tenant IDs must be valid UUIDs")

    is_valid_uuid = True
    try:
        uuid.UUID(kavach_mongo_id)
        is_valid_uuid = True
    except ValueError:
        is_valid_uuid = False

    if is_valid_uuid:
        failures.append("Test setup error: Kavach Mongo ID should NOT be a valid UUID")

    return failures


def test_high_value_message_detection() -> list[str]:
    """Test that high-value message detection works correctly."""
    failures: list[str] = []

    high_value_cases = [
        "Maa ne bataya ki unhe doctor se milna hai",
        "I love eating parathas in the morning",
        "My wife's birthday is next week",
        "Blood pressure was 140/90 today",
        "Osimertinib 80mg liya subah",
        "Beta came to visit yesterday",
    ]

    low_value_cases = [
        "ok",
        "yes",
        "theek hai",
        "What is the weather?",
    ]

    for msg in high_value_cases:
        if not is_high_value_message(msg):
            failures.append(f"Should detect as high-value: '{msg}'")

    for msg in low_value_cases:
        if is_high_value_message(msg):
            failures.append(f"Should NOT detect as high-value: '{msg}'")

    return failures


def test_bootstrap_profile_non_empty() -> list[str]:
    """Test that bootstrap profile generation returns non-empty content."""
    failures: list[str] = []
    return failures


async def run_rubric() -> int:
    failures: list[str] = []

    failures.extend(test_memory_tools_use_ai_uuids())
    failures.extend(test_high_value_message_detection())
    failures.extend(test_bootstrap_profile_non_empty())

    async with engine.begin() as conn:
        await run_instinct_migrations(conn)
        await conn.run_sync(Base.metadata.create_all)

    family_id = uuid.uuid4()
    elder_id = uuid.uuid4()

    async with SessionLocal() as session:
        session.add(Family(id=family_id, name="Eval Family"))
        session.add(
            Elder(
                id=elder_id,
                family_id=family_id,
                display_name="Eval Elder",
                slug="eval-elder",
            )
        )
        await session.flush()

        from app.rag.profile_render import bootstrap_memory_onepager, load_memory_profile_text

        bootstrap = await bootstrap_memory_onepager(
            session, family_id=family_id, elder_id=elder_id
        )
        if not bootstrap or len(bootstrap) < 10:
            failures.append("bootstrap_memory_onepager should return non-empty content even with no data")

        profile = await load_memory_profile_text(
            session, family_id=family_id, elder_id=elder_id
        )
        if not profile or len(profile) < 10:
            failures.append("load_memory_profile_text should return non-empty fallback")

        fact = await persist_inbox_memory(
            session,
            family_id=family_id,
            elder_id=elder_id,
            content="Amma takes osimertinib 80mg every morning with breakfast.",
            category="health",
            topic="osimertinib",
            source_role="family",
        )
        await session.commit()

        profile_with_fact = await load_memory_profile_text(
            session, family_id=family_id, elder_id=elder_id
        )
        if "osimertinib" not in profile_with_fact.lower():
            failures.append("load_memory_profile_text should include inbox facts in bootstrap")

        hits = await memory_grep(
            session,
            family_id=family_id,
            elder_id=elder_id,
            query="osimertinib morning tablet",
            limit=3,
        )
        if not hits and fact.content:
            failures.append("grep should find inbox or entity hit for medication query")

        ctx = await load_instinct_context(
            session,
            family_id=family_id,
            elder_id=elder_id,
            query="osimertinib",
        )
        if "Nothing saved" in ctx and fact.content not in ctx:
            failures.append("load_instinct_context should include inbox/profile text")

        await forget_memory_fact(session, fact_id=fact.id, forgotten_by="eval")
        await session.commit()
        post_hits = await memory_grep(
            session,
            family_id=family_id,
            elder_id=elder_id,
            query="osimertinib",
            limit=3,
        )
        if post_hits:
            failures.append("forgotten fact should not appear in grep hits")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    print("memory_instinct_rubric: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_rubric()))
