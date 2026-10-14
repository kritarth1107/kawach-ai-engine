"""Rebuild every embedding from the text it was made from, after a change of embedder (EMBED_PROVIDER, EMBED_MODEL or
EMBED_DIM). The AWS copy runs it once after the restore (Bedrock vectors replace the restored Vertex ones):

    python -m app.jobs.reembed               # every vector column, every row
    python -m app.jobs.reembed --missing     # only rows without a vector (resume, or after the API changed columns)
    python -m app.jobs.reembed --dry-run     # counts only
    python -m app.jobs.reembed --table memory_vectors --batch 64

The schema is brought up to date first (app.db.migrate). This job never changes a vector column: if EMBED_DIM
disagrees it exits. The API changes columns, once, with EMBED_DIM_MIGRATE set to the target dimension. Each batch is
committed on its own, so a stopped run resumes with --missing. Counts only in the output, never text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db.migrate import VECTOR_COLUMNS, read_catalog

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]
SOURCE_CHARS = 8000  # what an entity body was embedded from before (app.jobs.dream); the embedders cut longer text anyway


def _literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


async def reembed(sessions: async_sessionmaker, *, tables: list[str] | None = None, missing: bool = False, batch: int = 64,
                  dry_run: bool = False, embed: Embedder | None = None) -> dict:
    """{table.column: {"rows": n to do, "embedded": n done, "empty": n with no text (left without a vector)}}."""
    if embed is None:
        from app.rag.embeddings import embed_texts, embeddings_available, embedding_provider_label

        if not dry_run and not embeddings_available():
            raise RuntimeError("no embedder configured (EMBED_PROVIDER / GCP_PROJECT_ID)")
        logger.info("re-embed with %s", embedding_provider_label())
        embed = embed_texts
    async with sessions() as s:
        cat = await read_catalog(await s.connection())
    out: dict[str, dict] = {}
    for table, column, source, key in VECTOR_COLUMNS:
        if tables and table not in tables:
            continue
        if (table, column) not in cat.columns or (table, source) not in cat.columns:
            continue
        only = f" AND {column} IS NULL" if missing else ""
        async with sessions() as s:
            rows = int((await s.execute(text(f"SELECT count(*) FROM {table} WHERE true{only}"))).scalar() or 0)
        stat = {"rows": rows, "embedded": 0, "empty": 0}
        out[f"{table}.{column}"] = stat
        if dry_run or not rows:
            continue
        last = None
        while True:
            after = f" AND {key}::text > :last" if last is not None else ""
            async with sessions() as s:
                got = (await s.execute(text(f"SELECT {key}::text AS row_key, coalesce({source}, '') AS body FROM {table} "
                                            f"WHERE true{only}{after} ORDER BY {key}::text LIMIT :n"),
                                       {"last": last, "n": batch})).all()
                if not got:
                    break
                last = got[-1].row_key
                todo = [(r.row_key, r.body[:SOURCE_CHARS]) for r in got if r.body.strip()]
                stat["empty"] += len(got) - len(todo)
                vecs = await embed([t for _, t in todo]) if todo else []
                if len(vecs) != len(todo):
                    raise RuntimeError(f"{table}: {len(todo)} texts gave {len(vecs)} vectors")
                for (k, _), v in zip(todo, vecs):
                    await s.execute(text(f"UPDATE {table} SET {column} = CAST(:v AS vector) WHERE {key}::text = :k"),
                                    {"v": _literal(v), "k": k})
                await s.commit()
                stat["embedded"] += len(todo)
            logger.info("re-embed %s.%s: %d of %d", table, column, stat["embedded"] + stat["empty"], rows)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Rebuild every embedding with the configured embedder")
    p.add_argument("--missing", action="store_true", help="only rows without a vector")
    p.add_argument("--dry-run", action="store_true", help="counts only")
    p.add_argument("--table", action="append", help="limit to this table (repeatable)")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    async def run() -> dict:
        from app.db.migrate import ensure_schema
        from app.db.session import SessionLocal, engine

        await ensure_schema(engine)
        return await reembed(SessionLocal, tables=args.table, missing=args.missing, batch=max(1, args.batch), dry_run=args.dry_run)

    print(json.dumps(asyncio.run(run()), indent=2))


if __name__ == "__main__":
    main()
