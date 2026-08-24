"""Initialize database schema + pgvector extension."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from sqlalchemy import text

from app.db.session import Base, engine
from app.models import entities  # noqa: F401


async def main() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized (tables + pgvector).")


if __name__ == "__main__":
    asyncio.run(main())
