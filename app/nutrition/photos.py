"""Meal and label photos (plan §N2 Photos): a private bucket in asia-south1, `families/<f>/nutrition/<id>.jpg`, kept 90 days
(the bucket's lifecycle rule) and deleted with the entry. Only photos used for an entry are kept. Off until
NUTRITION_PHOTO_BUCKET names the bucket: then nothing is stored and entries carry no photo.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os

logger = logging.getLogger(__name__)


def bucket_name() -> str:
    return os.getenv("NUTRITION_PHOTO_BUCKET", "").strip()


def _bucket():
    from google.cloud import storage  # type: ignore[attr-defined]

    return storage.Client().bucket(bucket_name())


async def put(family_id: str, name: str, image: dict) -> str | None:
    """Store one photo; returns its ref (gs path) or None (not configured, or it failed: the entry is saved without it)."""
    if not bucket_name() or not image.get("data"):
        return None
    path = f"families/{family_id}/nutrition/{name}.jpg"

    def upload() -> None:
        _bucket().blob(path).upload_from_string(base64.b64decode(image["data"]), content_type=image.get("mime") or "image/jpeg")

    try:
        await asyncio.to_thread(upload)
    except Exception:  # noqa: BLE001
        logger.warning("nutrition photo not stored family=%s", family_id, exc_info=True)
        return None
    return f"gs://{bucket_name()}/{path}"


async def delete(ref: str | None) -> None:
    if not ref or not ref.startswith("gs://") or not bucket_name():
        return
    path = ref.split("/", 3)[-1]

    def remove() -> None:
        _bucket().blob(path).delete()

    try:
        await asyncio.to_thread(remove)
    except Exception:  # noqa: BLE001
        logger.warning("nutrition photo not deleted %s", ref, exc_info=True)


async def signed_url(ref: str | None, minutes: int = 15) -> str | None:
    if not ref or not ref.startswith("gs://") or not bucket_name():
        return None
    from datetime import timedelta

    path = ref.split("/", 3)[-1]

    def sign() -> str:
        return _bucket().blob(path).generate_signed_url(expiration=timedelta(minutes=minutes), version="v4")

    try:
        return await asyncio.to_thread(sign)
    except Exception:  # noqa: BLE001
        logger.warning("nutrition photo url failed %s", ref, exc_info=True)
        return None
