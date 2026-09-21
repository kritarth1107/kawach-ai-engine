"""Export entity bodies and profile to GCS for caregiver review."""

from __future__ import annotations

import logging
import os
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import MemoryEntity, MemoryProfile

logger = logging.getLogger(__name__)


async def export_memory_to_gcs(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> dict:
    bucket_name = os.getenv("MEMORY_GCS_BUCKET", "").strip()
    if not bucket_name:
        return {"exported": False, "reason": "MEMORY_GCS_BUCKET not set"}

    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        return {"exported": False, "reason": "google-cloud-storage not installed"}

    prefix = f"families/{family_id}/elders/{elder_id}/memory"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    count = 0

    entities = (
        await session.execute(
            select(MemoryEntity).where(
                MemoryEntity.family_id == family_id,
                MemoryEntity.elder_id == elder_id,
            )
        )
    ).scalars().all()
    for entity in entities:
        path = f"{prefix}/{entity.slug}.md"
        bucket.blob(path).upload_from_string(entity.body_md or "", content_type="text/markdown")
        count += 1

    profile = (
        await session.execute(
            select(MemoryProfile).where(
                MemoryProfile.family_id == family_id,
                MemoryProfile.elder_id == elder_id,
            )
        )
    ).scalar_one_or_none()
    if profile and profile.body_md:
        bucket.blob(f"{prefix}/PROFILE.md").upload_from_string(
            profile.body_md,
            content_type="text/markdown",
        )
        count += 1

    return {"exported": True, "files": count, "prefix": f"gs://{bucket_name}/{prefix}"}


async def delete_memory_gcs_prefix(
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
) -> None:
    bucket_name = os.getenv("MEMORY_GCS_BUCKET", "").strip()
    if not bucket_name:
        return
    try:
        from google.cloud import storage  # type: ignore

        prefix = f"families/{family_id}/elders/{elder_id}/memory/"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix):
            blob.delete()
    except Exception:
        logger.exception("GCS memory delete failed")
