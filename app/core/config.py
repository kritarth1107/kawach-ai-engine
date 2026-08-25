from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=True)


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

    # LLM provider: azure | ollama | xai
    llm_provider: str = "azure"

    # Azure AI Foundry / OpenAI-compatible (chat + embeddings)
    azure_openai_api_key: str = ""
    azure_openai_base_url: str = ""
    azure_chat_deployment: str = ""
    azure_embedding_deployment: str = "text-embedding-3-small"
    azure_embedding_model: str = "text-embedding-3-small"

    # Self-hosted Gemma via Ollama (OpenAI-compatible /v1)
    ollama_base_url: str = ""
    ollama_model: str = "gemma4:e4b"

    # Legacy xAI / Grok (optional fallback)
    xai_api_key: str = ""
    grok_chat_model: str = "grok-3-mini"
    grok_embedding_model: str = "text-embedding-3-large"
    grok_cli_timeout_seconds: int = 120

    # Legacy OpenAI fallback (embeddings only)
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"


@lru_cache
def get_settings() -> Settings:
    return Settings()
