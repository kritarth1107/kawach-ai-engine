from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_ai"
    kawach_api_secret: str = "dev-secret"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: str = "*"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_top_k: int = 8

    # Grok / xAI (primary)
    xai_api_key: str = ""
    grok_chat_model: str = "grok-3-mini"
    grok_embedding_model: str = "text-embedding-3-large"
    grok_cli_timeout_seconds: int = 120

    # Optional OpenAI fallback (embeddings only)
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"


@lru_cache
def get_settings() -> Settings:
    return Settings()
