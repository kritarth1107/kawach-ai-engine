#!/usr/bin/env python3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import uvicorn

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    host = settings.database_url.split("@")[-1].split("/")[0]
    print(f"Saheli memory: {host}")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
