"""Schema changes applied at every start: the engine, the night and restore jobs, tests and sims.

Safe while live traffic uses the tables and when several processes start at once. (D-13 review: one transaction that
ALTERed every table at each start queued behind live queries, could deadlock with another start, and swallowed the
error, so a missing column only showed later as every query on its table failing.)
- The catalog is read first: a table, column, index or constraint that is already there takes no lock at all.
- A missing one is added in its own short transaction with a lock timeout and retried, so a start never holds one
  table's lock while waiting for another, and live queries never queue behind it for long.
- On the live database, indexes are built CONCURRENTLY after start (reads and writes go on); a half-built one is rebuilt.
- A column the code needs that is still missing after the retries stops the start, so the old revision keeps serving.
  An index or constraint that could not be made is logged and tried again at the next start.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

log = logging.getLogger(__name__)

LOCK_TIMEOUT = "3s"  # DDL that blocks traffic while it waits for its lock
INDEX_LOCK_TIMEOUT = "60s"  # a concurrent index build blocks nobody while it waits
TRIES = 5

EXTENSIONS = ("vector", "pg_trgm")

# Every pgvector column and the text it is made from (app.jobs.reembed rebuilds them). Their dimension follows EMBED_DIM.
VECTOR_COLUMNS = [
    ("memory_vectors", "embedding", "text", "key"),
    ("family_memories", "embedding", "content", "id"),
    ("memory_entities", "body_embedding", "body_md", "id"),
    ("memory_snippets", "embedding", "content", "id"),
    ("document_chunks", "embedding", "content", "id"),
]
VECTOR_DIM = ("SELECT a.atttypmod FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
              "JOIN pg_namespace n ON n.oid = c.relnamespace "
              "WHERE n.nspname = current_schema() AND c.relname = :t AND a.attname = :c AND NOT a.attisdropped")


def embed_dim() -> int:
    from app.core.config import get_settings

    return int(get_settings().embed_dim)

# ── v1 memory layer (family_memories and its entity tables) ──

INSTINCT_COLUMNS = [
    ("family_memories", "content_hash", "VARCHAR(64)"),
    ("family_memories", "source_role", "VARCHAR(32) DEFAULT 'elder'"),
    ("family_memories", "superseded_by", "UUID"),
    ("family_memories", "forgotten_at", "TIMESTAMPTZ"),
    ("family_memories", "forgotten_by", "VARCHAR(128)"),
    ("family_memories", "entity_id", "UUID"),
]

INSTINCT_TABLES = [
    ("memory_entities", """
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
            body_embedding vector({dim}),
            sources JSONB DEFAULT '[]',
            version INTEGER DEFAULT 1,
            dirty BOOLEAN DEFAULT FALSE,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (elder_id, slug)
        )"""),
    ("memory_entity_links", """
        CREATE TABLE IF NOT EXISTS memory_entity_links (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            from_entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
            to_entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
            relation VARCHAR(64) DEFAULT 'related',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )"""),
    ("memory_profiles", """
        CREATE TABLE IF NOT EXISTS memory_profiles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            family_id UUID NOT NULL REFERENCES families(id) ON DELETE CASCADE,
            elder_id UUID NOT NULL REFERENCES elders(id) ON DELETE CASCADE,
            body_md TEXT DEFAULT '',
            token_estimate INTEGER DEFAULT 0,
            rendered_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (family_id, elder_id)
        )"""),
]

# (name, table, "UNIQUE" or "", what follows "ON table")
INSTINCT_INDEXES = [
    ("uq_family_memories_content_hash", "family_memories", "UNIQUE",
     "(elder_id, content_hash) WHERE forgotten_at IS NULL AND content_hash IS NOT NULL"),
    ("ix_memory_entities_aliases", "memory_entities", "", "USING GIN (aliases)"),
]

INSTINCT_CONSTRAINTS = [
    ("fk_family_memories_entity", "family_memories",
     "FOREIGN KEY (entity_id) REFERENCES memory_entities(id) ON DELETE SET NULL"),
    ("fk_family_memories_superseded", "family_memories",
     "FOREIGN KEY (superseded_by) REFERENCES family_memories(id) ON DELETE SET NULL"),
]

# ── Saheli v2: added after their tables first shipped (create_all makes new tables, never changes existing ones) ──

COLUMNS = [  # every one is required: the models select it
    ("reply_log", "trace", "JSONB"),
    ("playbook_versions", "scope", "JSONB DEFAULT '[]'::jsonb"),
    # D-13 why-trail: the trace of the work that wrote each row
    ("care_events", "trace_id", "VARCHAR(160)"),
    ("memory_versions", "trace_id", "VARCHAR(160)"),
    ("open_loops", "trace_id", "VARCHAR(160)"),
    ("care_tasks", "trace_id", "VARCHAR(160)"),
    # regular nudges learn: what the writer saw, how the person took it, the nightly grade
    ("nudge_ledger", "context", "TEXT"),
    ("nudge_ledger", "reaction", "VARCHAR(16)"),
    ("nudge_ledger", "grade", "JSONB"),
    ("nudge_ledger", "wamid", "VARCHAR(120)"),
    # trust plan A3: a correction from a person stays until a person changes it
    ("care_facts", "pinned_at", "TIMESTAMPTZ"),
    ("care_facts", "pinned_by", "VARCHAR(64)"),
]

INDEXES = [
    ("ix_turns_thread_id", "turns", "", "(family_id, thread_id, id)"),
    ("uq_service_logins_profile", "service_logins", "UNIQUE", "(profile_id) WHERE profile_id IS NOT NULL"),
    ("ix_care_events_trace", "care_events", "", "(trace_id) WHERE trace_id IS NOT NULL"),
    ("ix_memory_versions_trace", "memory_versions", "", "(trace_id) WHERE trace_id IS NOT NULL"),
    ("ix_open_loops_trace", "open_loops", "", "(trace_id) WHERE trace_id IS NOT NULL"),
    ("ix_care_tasks_trace", "care_tasks", "", "(trace_id) WHERE trace_id IS NOT NULL"),
    ("ix_care_events_task", "care_events", "", "((payload->>'task_id')) WHERE (payload->>'task_id') IS NOT NULL"),
    # nutrition (NUTRITION_PLAN_2026-10-14 §N2): the tables come from create_all, their indexes from here
    ("ix_nutrition_foods_family_name", "nutrition_foods", "", "(family_id, lower(name))"),
    ("ix_nutrition_foods_aliases", "nutrition_foods", "", "USING GIN (aliases)"),
    ("ix_nutrition_foods_name_trgm", "nutrition_foods", "", "USING GIN (lower(name) gin_trgm_ops)"),
    ("ix_nutrition_foods_barcode", "nutrition_foods", "", "(barcode) WHERE barcode IS NOT NULL"),
    ("uq_nutrition_foods_generic", "nutrition_foods", "UNIQUE",
     "(lower(name), coalesce(lower(brand), ''), coalesce(lower(restaurant), '')) WHERE family_id IS NULL"),
    ("ix_nutrition_entries_day", "nutrition_entries", "", "(family_id, subject_id, day)"),
    ("ix_nutrition_entries_eaten", "nutrition_entries", "", "(family_id, subject_id, eaten_at)"),
    ("ix_nutrition_entries_meal", "nutrition_entries", "", "(meal_id)"),
    ("ix_nutrition_entries_source", "nutrition_entries", "", "(family_id, source_ref) WHERE source_ref IS NOT NULL"),
    ("ix_nutrition_jobs_live", "nutrition_jobs", "", "(family_id, subject_id, state)"),
    ("ix_nutrition_jobs_sweep", "nutrition_jobs", "", "(state, updated_at) WHERE state IN ('queued', 'resolving', 'needs_input')"),
    ("ix_nutrition_pantry_family", "nutrition_pantry", "", "(family_id, status)"),
    ("uq_nutrition_pantry_task_item", "nutrition_pantry", "UNIQUE", "(family_id, task_id, lower(name)) WHERE task_id IS NOT NULL"),
]

COLUMN_THERE = ("SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = :t AND column_name = :c")
CONSTRAINT_THERE = ("SELECT 1 FROM pg_constraint k JOIN pg_namespace n ON n.oid = k.connamespace "
                    "WHERE n.nspname = current_schema() AND k.conname = :n")
INDEX_STATE = ("SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
               "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = current_schema() AND c.relname = :n")


@dataclass
class Catalog:
    extensions: set[str]
    tables: set[str]
    columns: set[tuple[str, str]]
    indexes: dict[str, bool]  # name → valid (False: left half-built by a failed concurrent build)
    constraints: set[str]


async def read_catalog(conn: AsyncConnection) -> Catalog:
    async def rows(sql: str) -> list:
        return list((await conn.execute(text(sql))).all())

    return Catalog(
        extensions={r[0] for r in await rows("SELECT extname FROM pg_extension")},
        tables={r[0] for r in await rows("SELECT tablename FROM pg_tables WHERE schemaname = current_schema()")},
        columns={(r[0], r[1]) for r in await rows(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = current_schema()")},
        indexes={r[0]: bool(r[1]) for r in await rows(
            "SELECT c.relname, i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = current_schema()")},
        constraints={r[0] for r in await rows(
            "SELECT k.conname FROM pg_constraint k JOIN pg_namespace n ON n.oid = k.connamespace "
            "WHERE n.nspname = current_schema()")},
    )


def _index_sql(name: str, table: str, unique: str, body: str, *, concurrently: bool = False) -> str:
    return " ".join(p for p in ("CREATE", unique, "INDEX", "CONCURRENTLY" if concurrently else "",
                                f"IF NOT EXISTS {name} ON {table} {body}") if p)


def _short(e: BaseException | None) -> str:
    return " ".join(str(getattr(e, "orig", None) or e).split())[:300]


async def _retry(what: str, run, done=None) -> BaseException | None:
    """Run a schema step up to TRIES times (a lock timeout, a deadlock, or another start that made it first). Returns
    the last error, or None once it ran or done() says the thing is there."""
    err: BaseException | None = None
    for i in range(TRIES):
        try:
            await run()
            return None
        except Exception as e:  # noqa: BLE001
            err = e
            if done is not None:
                try:
                    if await done():
                        return None
                except Exception:  # noqa: BLE001
                    pass
            if i + 1 < TRIES:
                log.warning("schema: %s failed (try %d of %d): %s", what, i + 1, TRIES, _short(e))
                await asyncio.sleep(min(0.5 * 2 ** i, 4) + random.random() * 0.3)
    log.error("schema: %s failed after %d tries: %s", what, TRIES, _short(err))
    return err


async def _alter(engine: AsyncEngine, what: str, ddl: str, there: str, params: dict, *, required: bool) -> None:
    async def run() -> None:
        async with engine.begin() as conn:
            await conn.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
            await conn.execute(text(ddl))
        log.info("schema: added %s", what)

    async def done() -> bool:
        async with engine.connect() as conn:
            return (await conn.execute(text(there), params)).first() is not None

    err = await _retry(what, run, done)
    if err is not None and required:
        raise RuntimeError(f"schema: {what} is missing and could not be added: {_short(err)}") from err


async def ensure_columns(engine: AsyncEngine, columns: list[tuple[str, str, str]], cat: Catalog) -> None:
    """Add the missing ones (required: a start without them would fail every query on the table)."""
    for table, column, kind in columns:
        if table in cat.tables and (table, column) not in cat.columns:
            await _alter(engine, f"column {table}.{column}", f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {kind}",
                         COLUMN_THERE, {"t": table, "c": column}, required=True)


async def vector_dims(conn: AsyncConnection, cat: Catalog) -> dict[tuple[str, str], int]:
    """The dimension of each pgvector column there is (-1: none declared)."""
    out = {}
    for table, column, _, _ in VECTOR_COLUMNS:
        if (table, column) in cat.columns:
            out[(table, column)] = int((await conn.execute(text(VECTOR_DIM), {"t": table, "c": column})).scalar() or -1)
    return out


async def ensure_vector_dims(engine: AsyncEngine, cat: Catalog, dim: int | None = None) -> list[str]:
    """Vector columns follow EMBED_DIM (a new embedder, e.g. Bedrock on the AWS copy). A column of another dimension holds
    another embedder's vectors, which mean nothing to the new one: with EMBED_DIM_MIGRATE=on the column is changed and
    emptied (one short lock each, retried; recall uses words until `python -m app.jobs.reembed` refills it). Without it
    the start stops, so a wrong EMBED_DIM never clears a live index and the old revision keeps serving."""
    dim = dim or embed_dim()
    async with engine.connect() as conn:
        have = await vector_dims(conn, cat)
    wrong = [(t, c, d) for (t, c), d in have.items() if d not in (-1, dim)]
    if not wrong:
        return []
    if os.getenv("EMBED_DIM_MIGRATE", "off").strip().lower() != "on":
        cols = ", ".join(f"{t}.{c} vector({d})" for t, c, d in wrong)
        raise RuntimeError(f"schema: {cols} but EMBED_DIM is {dim}. Set EMBED_DIM_MIGRATE=on once to change them (their vectors "
                           "are cleared), then run python -m app.jobs.reembed")
    changed = []
    for table, column, d in wrong:
        await _alter(engine, f"vector {table}.{column} {d}→{dim}",
                     f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector({dim}) USING NULL",
                     VECTOR_DIM + f" AND a.atttypmod = {dim}", {"t": table, "c": column}, required=True)
        log.warning("schema: %s.%s is now vector(%d) and empty: run python -m app.jobs.reembed", table, column, dim)
        changed.append(f"{table}.{column}")
    return changed


async def ensure_schema(engine: AsyncEngine) -> None:
    """The live path (engine start, night and restore jobs): tables, columns and constraints now. Indexes come from
    ensure_indexes(), which the engine runs after it starts serving."""
    from app.db.session import Base

    async def tables() -> None:
        async with engine.begin() as conn:
            await conn.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
            cat = await read_catalog(conn)
            for ext in EXTENSIONS:
                if ext not in cat.extensions:
                    await conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext}"))
            await conn.run_sync(Base.metadata.create_all)  # checks first: an existing table is never touched
            for name, ddl in INSTINCT_TABLES:
                if name not in cat.tables:
                    await conn.execute(text(ddl.replace("{dim}", str(embed_dim()))))

    err = await _retry("tables", tables)
    if err is not None:
        raise RuntimeError(f"schema: tables could not be created: {_short(err)}") from err
    async with engine.connect() as conn:
        cat = await read_catalog(conn)
    await ensure_columns(engine, INSTINCT_COLUMNS + COLUMNS, cat)
    await ensure_vector_dims(engine, cat)
    for name, table, ddl in INSTINCT_CONSTRAINTS:
        if table in cat.tables and name not in cat.constraints:
            await _alter(engine, f"constraint {name}", f"ALTER TABLE {table} ADD CONSTRAINT {name} {ddl}",
                         CONSTRAINT_THERE, {"n": name}, required=False)


async def build_index(engine: AsyncEngine, name: str, table: str, unique: str, body: str) -> None:
    """CREATE INDEX CONCURRENTLY outside a transaction (reads and writes go on); a half-built one is dropped first."""
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text(f"SET lock_timeout = '{INDEX_LOCK_TIMEOUT}'"))
        try:
            if (await conn.execute(text(INDEX_STATE), {"n": name})).scalar() is False:
                await conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {name}"))
            await conn.execute(text(_index_sql(name, table, unique, body, concurrently=True)))
        finally:
            await conn.execute(text("RESET lock_timeout"))
    log.info("schema: built index %s", name)


async def ensure_indexes(engine: AsyncEngine, indexes: list[tuple[str, str, str, str]] | None = None) -> None:
    """Build missing or half-built indexes. Never raises: a missing index only slows queries, and the next start tries
    again."""
    try:
        async with engine.connect() as conn:
            cat = await read_catalog(conn)
        for name, table, unique, body in (INSTINCT_INDEXES + INDEXES if indexes is None else indexes):
            if table not in cat.tables or cat.indexes.get(name) is True:
                continue

            async def valid(n: str = name) -> bool:
                async with engine.connect() as c:
                    return (await c.execute(text(INDEX_STATE), {"n": n})).scalar() is True

            await _retry(f"index {name}", lambda n=name, t=table, u=unique, b=body: build_index(engine, n, t, u, b), valid)
    except Exception:  # noqa: BLE001
        log.exception("schema: index check failed; the next start tries again")


# ── tests, sims and evals: a fresh or private database without live traffic, inside the caller's transaction ──


async def _apply(conn: AsyncConnection, columns, indexes, constraints=()) -> None:
    cat = await read_catalog(conn)
    for table, column, kind in columns:
        if table in cat.tables and (table, column) not in cat.columns:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {kind}"))
    for name, table, unique, body in indexes:
        if table in cat.tables and name not in cat.indexes:
            await conn.execute(text(_index_sql(name, table, unique, body)))
    for name, table, ddl in constraints:
        if table in cat.tables and name not in cat.constraints:
            await conn.execute(text(f"ALTER TABLE {table} ADD CONSTRAINT {name} {ddl}"))


async def run_v2_migrations(conn: AsyncConnection) -> None:
    await _apply(conn, COLUMNS, INDEXES)


async def run_instinct_migrations(conn: AsyncConnection) -> None:
    cat = await read_catalog(conn)
    for ext in EXTENSIONS:
        if ext not in cat.extensions:
            await conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext}"))
    for name, ddl in INSTINCT_TABLES:
        if name not in cat.tables:
            await conn.execute(text(ddl.replace("{dim}", str(embed_dim()))))
    await _apply(conn, INSTINCT_COLUMNS, INSTINCT_INDEXES, INSTINCT_CONSTRAINTS)
