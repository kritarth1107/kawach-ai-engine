import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
# A turn holds its connection while the model thinks; size the pool to the instance's concurrency
# (and keep the total across instances under the Cloud SQL connection limit).
engine = create_async_engine(
    settings.database_url, echo=settings.debug, pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")), max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "15")), pool_timeout=20,
)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
