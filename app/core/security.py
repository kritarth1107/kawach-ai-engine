from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

api_key_header = APIKeyHeader(name="X-Kavach-Secret", auto_error=False)


async def verify_api_secret(api_key: str | None = Security(api_key_header)) -> None:
    settings = get_settings()
    if not api_key or api_key != settings.kawach_api_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Kavach-Secret")
