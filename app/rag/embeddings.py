"""Embedding clients — Vertex AI (GCP), Amazon Bedrock (the AWS copy), Azure Foundry, xAI/OpenAI fallback.

EMBED_PROVIDER picks one: "" (today's order: Vertex when GCP_PROJECT_ID is set, then Azure, xAI, OpenAI) | vertex |
bedrock. Bedrock runs in EMBED_REGION (India only) with EMBED_MODEL:
- amazon.titan-embed-text-v2:0: EMBED_DIM 256, 512 or 1024 (1024 recalled best in the 14 Oct multilingual check)
- cohere.embed-multilingual-v3: EMBED_DIM 1024; queries and memories are embedded as such (input_type)
A different embedder means different vectors: the API changes the columns once with EMBED_DIM_MIGRATE set to the new
dimension (app.db.migrate), then `python -m app.jobs.reembed` refills them. A job never changes the columns.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.core.config import get_settings

XAI_API_BASE = "https://api.x.ai/v1"
BEDROCK_DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"
TITAN_DIMS = (256, 512, 1024)
COHERE_BATCH = 96  # texts per Cohere call
COHERE_CHARS = 2048  # Cohere's limit per text (truncate="END" cuts the rest)
TITAN_CHARS = 30_000  # well under Titan v2's 8k-token input


def _valid_api_key(key: str) -> bool:
    k = key.strip()
    if not k:
        return False
    if k in {"xai-...", "sk-...", "sk-your-key"}:
        return False
    if k.endswith("..."):
        return False
    return True


def _provider() -> str:
    return get_settings().embed_provider.strip().lower()


def _vertex_configured() -> bool:
    settings = get_settings()
    return bool(settings.gcp_project_id.strip())


class BedrockEmbeddings:
    """Amazon Bedrock embeddings with the two calls the engine uses (aembed_query, aembed_documents). Credentials come
    from the default AWS chain (the ECS task role, an SSO profile locally)."""

    def __init__(self, model: str, region: str, dim: int, client: Any = None) -> None:
        if model.startswith("amazon.titan-embed-text-v2") and dim not in TITAN_DIMS:
            raise ValueError(f"EMBED_DIM {dim} is not one Titan v2 makes {TITAN_DIMS}")
        if model.startswith("cohere.embed-multilingual-v3") and dim != 1024:
            raise ValueError(f"EMBED_DIM {dim}: cohere.embed-multilingual-v3 makes 1024")
        self.model, self.region, self.dim = model, region, dim
        self._client = client

    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client("bedrock-runtime", region_name=self.region,
                                        config=Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=60))
        return self._client

    def _invoke(self, body: dict) -> dict:
        resp = self.client().invoke_model(modelId=self.model, body=json.dumps(body), contentType="application/json",
                                          accept="application/json")
        raw = resp["body"].read() if hasattr(resp["body"], "read") else resp["body"]
        return json.loads(raw)

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        if self.model.startswith("cohere."):
            out: list[list[float]] = []
            for i in range(0, len(texts), COHERE_BATCH):
                chunk = [(t or " ")[:COHERE_CHARS] for t in texts[i:i + COHERE_BATCH]]
                out += self._invoke({"texts": chunk, "input_type": kind, "truncate": "END"})["embeddings"]
            return out
        return [self._invoke({"inputText": (t or " ")[:TITAN_CHARS], "dimensions": self.dim, "normalize": True})["embedding"]
                for t in texts]

    def _checked(self, vecs: list[list[float]]) -> list[list[float]]:
        for v in vecs:
            if len(v) != self.dim:
                raise RuntimeError(f"{self.model} returned {len(v)} dimensions, EMBED_DIM is {self.dim}")
        return vecs

    async def aembed_query(self, text: str) -> list[float]:
        return self._checked(await asyncio.to_thread(self._embed, [text], "search_query"))[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._checked(await asyncio.to_thread(self._embed, list(texts), "search_document"))


_bedrock: BedrockEmbeddings | None = None


def _bedrock_embeddings() -> BedrockEmbeddings:
    global _bedrock
    settings = get_settings()
    model = settings.embed_model.strip() or BEDROCK_DEFAULT_MODEL
    if _bedrock is None or (_bedrock.model, _bedrock.region, _bedrock.dim) != (model, settings.embed_region, settings.embed_dim):
        _bedrock = BedrockEmbeddings(model, settings.embed_region, settings.embed_dim)
    return _bedrock


def embeddings_available() -> bool:
    settings = get_settings()
    provider = _provider()
    if provider == "bedrock":
        return True
    if provider == "vertex":
        return _vertex_configured()
    return (
        _vertex_configured()
        or (
            _valid_api_key(settings.azure_openai_api_key)
            and bool(settings.azure_embedding_deployment)
        )
        or _valid_api_key(settings.xai_api_key)
        or _valid_api_key(settings.openai_api_key)
    )


def get_embeddings() -> Any:
    settings = get_settings()
    provider = _provider()
    if provider == "bedrock":
        return _bedrock_embeddings()

    if _vertex_configured():
        from langchain_google_vertexai import VertexAIEmbeddings

        from app.llm import google_wif

        # On AWS the same workload-identity credentials as Gemini (kavach-brain). On GCP, None uses the default chain.
        creds = google_wif.credentials()
        return VertexAIEmbeddings(
            model_name=settings.vertex_embedding_model,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
            **({"credentials": creds} if creds is not None else {}),
        )
    if provider == "vertex":
        raise RuntimeError("EMBED_PROVIDER=vertex needs GCP_PROJECT_ID.")

    from langchain_openai import OpenAIEmbeddings

    if _valid_api_key(settings.azure_openai_api_key) and settings.azure_embedding_deployment:
        return OpenAIEmbeddings(
            model=settings.azure_embedding_model,
            api_key=settings.azure_openai_api_key,
            base_url=settings.azure_openai_base_url,
            deployment=settings.azure_embedding_deployment,
        )

    if _valid_api_key(settings.xai_api_key):
        return OpenAIEmbeddings(
            model=settings.grok_embedding_model,
            api_key=settings.xai_api_key,
            base_url=XAI_API_BASE,
        )

    if _valid_api_key(settings.openai_api_key):
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    raise RuntimeError(
        "RAG embeddings need GCP_PROJECT_ID (Vertex), EMBED_PROVIDER=bedrock, Azure Foundry, or XAI/OPENAI API keys."
    )


def embedding_provider_label() -> str:
    settings = get_settings()
    provider = _provider()
    if provider == "bedrock":
        return f"bedrock:{settings.embed_model.strip() or BEDROCK_DEFAULT_MODEL}@{settings.embed_region}"
    if _vertex_configured():
        return f"vertex:{settings.vertex_embedding_model}"
    if provider == "vertex":
        return "none"
    if _valid_api_key(settings.azure_openai_api_key) and settings.azure_embedding_deployment:
        return f"azure:{settings.azure_embedding_deployment}"
    if _valid_api_key(settings.xai_api_key):
        return f"xai:{settings.grok_embedding_model}"
    if _valid_api_key(settings.openai_api_key):
        return f"openai:{settings.openai_embedding_model}"
    return "none"


async def embed_text(text: str) -> list[float]:
    """A search query's embedding."""
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)


async def embed_texts(texts: list[str], *, query: bool = False) -> list[list[float]]:
    """Memories' embeddings (query=True: these are search queries; matters for Cohere, the others embed both alike)."""
    embeddings = get_embeddings()
    if query and isinstance(embeddings, BedrockEmbeddings):
        return [await embeddings.aembed_query(t) for t in texts]
    return await embeddings.aembed_documents(texts)
