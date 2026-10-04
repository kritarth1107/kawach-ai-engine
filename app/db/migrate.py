"""Incremental schema migrations for Instinct memory layer."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def run_instinct_migrations(conn: AsyncConnection) -> None:
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    alters = [
        "ALTER TABLE family_memories ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)",
        "ALTER TABLE family_memories ADD COLUMN IF NOT EXISTS source_role VARCHAR(32) DEFAULT 'elder'",
        "ALTER TABLE family_memories ADD COLUMN IF NOT EXISTS superseded_by UUID",
        "ALTER TABLE family_memories ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ",
        "ALTER TABLE family_memories ADD COLUMN IF NOT EXISTS forgotten_by VARCHAR(128)",
        "ALTER TABLE family_memories ADD COLUMN IF NOT EXISTS entity_id UUID",
    ]
    for stmt in alters:
        await conn.execute(text(stmt))

    await conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_family_memories_content_hash
            ON family_memories (elder_id, content_hash)
            WHERE forgotten_at IS NULL AND content_hash IS NOT NULL
            """
        )
    )

    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS memory_entities (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                family_id UUID NOT NULL REFERENCES families(id) ON DELETE CASCADE,
                elder_id UUID NOT NULL REFERENCES elders(id) ON DELETE CASCADE,
                slug VARCHAR(128) NOT NULL,
                kind VARCHAR(32) NOT NULL,
                title VARCHAR(255) NOT NULL,
                aliases TEXT[],
                status VARCHAR(32) DEFAULT 'active',
                owner_user_id VARCHAR(128),
                review_by DATE,
                body_md TEXT DEFAULT '',
                body_embedding vector(768),
                sources JSONB DEFAULT '[]',
                version INTEGER DEFAULT 1,
                dirty BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (elder_id, slug)
            )
            """
        )
    )

    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS memory_entity_links (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                from_entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
                to_entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
                relation VARCHAR(64) DEFAULT 'related',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
    )

    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS memory_profiles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                family_id UUID NOT NULL REFERENCES families(id) ON DELETE CASCADE,
                elder_id UUID NOT NULL REFERENCES elders(id) ON DELETE CASCADE,
                body_md TEXT DEFAULT '',
                token_estimate INTEGER DEFAULT 0,
                rendered_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (family_id, elder_id)
            )
            """
        )
    )

    await conn.execute(
        text(
            """
            DO $$ BEGIN
                ALTER TABLE family_memories
                ADD CONSTRAINT fk_family_memories_entity
                FOREIGN KEY (entity_id) REFERENCES memory_entities(id) ON DELETE SET NULL;
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$
            """
        )
    )

    await conn.execute(
        text(
            """
            DO $$ BEGIN
                ALTER TABLE family_memories
                ADD CONSTRAINT fk_family_memories_superseded
                FOREIGN KEY (superseded_by) REFERENCES family_memories(id) ON DELETE SET NULL;
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$
            """
        )
    )

    await conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_memory_entities_aliases ON memory_entities USING GIN (aliases)"
        )
    )


# Columns and indexes added to Saheli v2 tables after they first shipped. create_all() makes new tables but never
# changes existing ones, so every startup (and the test, sim and job setups) runs these idempotent statements.
V2_ALTERS = [
    "ALTER TABLE reply_log ADD COLUMN IF NOT EXISTS trace JSONB",
    "ALTER TABLE playbook_versions ADD COLUMN IF NOT EXISTS scope JSONB DEFAULT '[]'::jsonb",
    "CREATE INDEX IF NOT EXISTS ix_turns_thread_id ON turns (family_id, thread_id, id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_service_logins_profile ON service_logins (profile_id) WHERE profile_id IS NOT NULL",
]


async def run_v2_migrations(conn: AsyncConnection) -> None:
    for stmt in V2_ALTERS:
        try:
            await conn.execute(text(f"SAVEPOINT v2m"))
            await conn.execute(text(stmt))
            await conn.execute(text("RELEASE SAVEPOINT v2m"))
        except Exception:  # noqa: BLE001 — a table that does not exist yet is created by create_all with the column
            await conn.execute(text("ROLLBACK TO SAVEPOINT v2m"))
