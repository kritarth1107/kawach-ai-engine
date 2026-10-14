"""The AWS copy runs without Google (founder 14 Oct): every Google path has a switch whose default is today's behaviour.

- embeddings: EMBED_PROVIDER=bedrock (Titan v2 or Cohere multilingual v3 in Mumbai), EMBED_DIM followed by the vector
  columns only with EMBED_DIM_MIGRATE=on, and `python -m app.jobs.reembed` rebuilds every vector from its text;
- web search: WEB_SEARCH_PROVIDER=tavily, Tavily's results answered by Saheli's own model routes under the same prompt;
- engine admin auth: ENGINE_CALLER_CHECK=off (the admin secret only), never on Cloud Run;
- the legacy chat path (the doctor brief): LLM_PROVIDER=router goes through MODEL_ROUTES.
Invented text only."""

import io
import json
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import admin, quality
from app.core.config import get_settings
from app.db import migrate
from app.jobs import reembed
from app.llm import router
from app.llm.router import LLMReply
from app.rag import embeddings
from tests.conftest import TEST_DB


@pytest.fixture
def settings_env(monkeypatch):
    """Set env, then read the settings afresh (they are cached)."""
    def set_(**kw):
        for k, v in kw.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, str(v))
        get_settings.cache_clear()
        embeddings._bedrock = None
    yield set_
    get_settings.cache_clear()
    embeddings._bedrock = None


class FakeBedrock:
    """bedrock-runtime invoke_model: records each request, answers vectors of the asked size."""

    def __init__(self, dim: int):
        self.dim, self.calls = dim, []

    def invoke_model(self, *, modelId, body, contentType=None, accept=None):
        req = json.loads(body)
        self.calls.append((modelId, req))
        if modelId.startswith("cohere."):
            out = {"embeddings": [[float(len(t))] * self.dim for t in req["texts"]]}
        else:
            out = {"embedding": [0.5] * req["dimensions"]}
        return {"body": io.BytesIO(json.dumps(out).encode())}


# ── embeddings ───────────────────────────────────────────────────────────────


def test_today_is_unchanged_without_the_switch(settings_env):
    settings_env(EMBED_PROVIDER=None, GCP_PROJECT_ID="kavach-test")
    assert embeddings.embedding_provider_label() == "vertex:text-multilingual-embedding-002"
    assert embeddings.embeddings_available()


def test_vertex_embeddings_use_the_workload_identity(settings_env, monkeypatch):
    settings_env(EMBED_PROVIDER="vertex", GCP_PROJECT_ID="sunny-ship-508214-q1", GCP_REGION="asia-south1")
    sentinel = object()
    got = {}

    class Fake:
        def __init__(self, **kw):
            got.update(kw)

    monkeypatch.setattr("langchain_google_vertexai.VertexAIEmbeddings", Fake)
    monkeypatch.setattr("app.llm.google_wif.credentials", lambda: sentinel)
    embeddings.get_embeddings()
    assert got["credentials"] is sentinel
    assert got["project"] == "sunny-ship-508214-q1" and got["location"] == "asia-south1"
    assert got["model_name"] == "text-multilingual-embedding-002"


def test_bedrock_needs_no_google(settings_env):
    settings_env(EMBED_PROVIDER="bedrock", GCP_PROJECT_ID=None, EMBED_DIM=1024, EMBED_MODEL=None)
    assert embeddings.embeddings_available()
    assert embeddings.embedding_provider_label() == "bedrock:amazon.titan-embed-text-v2:0@ap-south-1"
    assert isinstance(embeddings.get_embeddings(), embeddings.BedrockEmbeddings)


async def test_titan_asks_for_the_configured_dimension():
    fake = FakeBedrock(512)
    e = embeddings.BedrockEmbeddings("amazon.titan-embed-text-v2:0", "ap-south-1", 512, client=fake)
    vecs = await e.aembed_documents(["Ghutne mein dard", "शुगर की दवा"])
    assert [len(v) for v in vecs] == [512, 512]
    assert [r["dimensions"] for _, r in fake.calls] == [512, 512] and all(r["normalize"] for _, r in fake.calls)


async def test_cohere_embeds_queries_and_memories_as_such_in_batches():
    fake = FakeBedrock(1024)
    e = embeddings.BedrockEmbeddings("cohere.embed-multilingual-v3", "ap-south-1", 1024, client=fake)
    await e.aembed_documents([f"memory {i}" for i in range(100)])
    await e.aembed_query("wahi dard phir se")
    kinds = [(len(r["texts"]), r["input_type"]) for _, r in fake.calls]
    assert kinds == [(96, "search_document"), (4, "search_document"), (1, "search_query")]


def test_a_dimension_the_model_cannot_make_is_refused():
    with pytest.raises(ValueError):
        embeddings.BedrockEmbeddings("amazon.titan-embed-text-v2:0", "ap-south-1", 768)
    with pytest.raises(ValueError):
        embeddings.BedrockEmbeddings("cohere.embed-multilingual-v3", "ap-south-1", 512)


async def test_vectors_of_the_wrong_size_never_reach_the_index():
    fake = FakeBedrock(1024)
    e = embeddings.BedrockEmbeddings("cohere.embed-multilingual-v3", "ap-south-1", 1024, client=fake)
    fake.dim = 384
    with pytest.raises(RuntimeError):
        await e.aembed_documents(["x"])


async def test_memory_search_embeds_the_question_as_a_query(monkeypatch):
    from app.care import memory_index

    seen = []

    async def fake_embed_texts(texts, *, query=False):
        seen.append(query)
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(embeddings, "embeddings_available", lambda: True)
    monkeypatch.setattr(memory_index, "_broken_until", 0.0)
    monkeypatch.setenv("MEMORY_EMBEDDINGS", "on")
    await memory_index.embed(["Aaj ghutne mein dard hai"], query=True)
    await memory_index.embed(["(active) Knee pain on stairs"])
    assert seen == [True, False]


# ── EMBED_DIM: the vector columns follow it, only when asked ─────────────────


@pytest.fixture
async def scratch(monkeypatch):
    """A committed scratch table with a vector column (the DDL runs in its own transactions)."""
    name = f"zz_vec_{uuid.uuid4().hex[:8]}"
    eng = create_async_engine(TEST_DB)
    async with eng.begin() as c:
        await c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await c.execute(text(f"CREATE TABLE {name} (id TEXT PRIMARY KEY, body TEXT, emb vector(3))"))
        await c.execute(text(f"INSERT INTO {name} VALUES ('a', 'Kamla ji ko bhajan pasand hai', '[1,2,3]'), "
                             f"('b', 'Peanut allergy', '[4,5,6]'), ('c', '', NULL)"))
    cols = [(name, "emb", "body", "id")]
    monkeypatch.setattr(migrate, "VECTOR_COLUMNS", cols)
    monkeypatch.setattr(reembed, "VECTOR_COLUMNS", cols)
    try:
        yield eng, name
    finally:
        async with eng.begin() as c:
            await c.execute(text(f"DROP TABLE IF EXISTS {name}"))
        await eng.dispose()


async def _dim(eng, name):
    async with eng.connect() as c:
        return (await c.execute(text(migrate.VECTOR_DIM), {"t": name, "c": "emb"})).scalar()


async def test_a_changed_dimension_stops_the_start_unless_asked(scratch, monkeypatch):
    eng, name = scratch
    monkeypatch.delenv("EMBED_DIM_MIGRATE", raising=False)
    async with eng.connect() as c:
        cat = await migrate.read_catalog(c)
    assert await migrate.ensure_vector_dims(eng, cat, dim=3) == []  # the same size: nothing to do
    with pytest.raises(RuntimeError, match="EMBED_DIM_MIGRATE"):
        await migrate.ensure_vector_dims(eng, cat, dim=5)
    assert await _dim(eng, name) == 3  # the live vectors were not touched


async def test_asked_the_column_follows_and_its_old_vectors_go(scratch, monkeypatch):
    eng, name = scratch
    monkeypatch.setenv("EMBED_DIM_MIGRATE", "on")
    async with eng.connect() as c:
        cat = await migrate.read_catalog(c)
    assert await migrate.ensure_vector_dims(eng, cat, dim=5) == [f"{name}.emb"]
    assert await _dim(eng, name) == 5
    async with eng.connect() as c:
        assert (await c.execute(text(f"SELECT count(*) FROM {name} WHERE emb IS NOT NULL"))).scalar() == 0


async def test_reembed_rebuilds_every_vector_from_its_text(scratch, monkeypatch):
    eng, name = scratch
    monkeypatch.setenv("EMBED_DIM_MIGRATE", "on")
    async with eng.connect() as c:
        cat = await migrate.read_catalog(c)
    await migrate.ensure_vector_dims(eng, cat, dim=4)
    sessions = async_sessionmaker(eng, expire_on_commit=False)
    asked: list[list[str]] = []

    async def fake(texts):
        asked.append(texts)
        return [[float(len(t)), 0.0, 0.0, 1.0] for t in texts]

    dry = await reembed.reembed(sessions, dry_run=True, embed=fake)
    assert dry == {f"{name}.emb": {"rows": 3, "embedded": 0, "empty": 0}} and asked == []
    out = await reembed.reembed(sessions, batch=1, embed=fake)
    assert out[f"{name}.emb"] == {"rows": 3, "embedded": 2, "empty": 1}
    async with eng.connect() as c:
        got = dict((await c.execute(text(f"SELECT id, emb::text FROM {name}"))).all())
    assert got["a"] == f"[{len('Kamla ji ko bhajan pasand hai')},0,0,1]" and got["c"] is None
    again = await reembed.reembed(sessions, missing=True, embed=fake)
    assert again[f"{name}.emb"] == {"rows": 1, "embedded": 0, "empty": 1}  # only the row with no text is left


def test_every_vector_column_the_models_declare_is_rebuilt():
    from app.db.session import Base
    from pgvector.sqlalchemy import Vector

    declared = {(t.name, c.name) for t in Base.metadata.tables.values() for c in t.columns if isinstance(c.type, Vector)}
    assert declared <= {(t, c) for t, c, _, _ in migrate.VECTOR_COLUMNS}


# ── web search ───────────────────────────────────────────────────────────────


class Answerer:
    def __init__(self):
        self.calls = []

    async def complete(self, route, **kw):
        self.calls.append(kw)
        user = kw["messages"][0]["content"][0]["text"]
        if "WEB RESULTS" not in user:
            return LLMReply(text="paracetamol side effects elderly", tool_calls=[], model="fake")
        return LLMReply(text="Paracetamol is usually gentle on the stomach [1]. Check with your doctor first.", tool_calls=[], model="fake")


@pytest.fixture
def answerer(monkeypatch):
    a = Answerer()
    router.register_provider("fake", a)
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["fake:m"] for r in ("worker", "classify", "web_search", "nutrition_search")}))
    router.reset_breakers()
    yield a
    router._providers.pop("fake", None)
    router.reset_breakers()


async def test_tavily_answers_under_the_same_prompt(monkeypatch, answerer):
    from app.brain import websearch

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    asked = []

    async def fake_tavily(query, *, max_results=5):
        asked.append(query)
        return [{"title": "Paracetamol | Health guide", "url": "https://example.org/paracetamol", "content": "Generally well tolerated."}]

    monkeypatch.setattr(websearch, "tavily_search", fake_tavily)
    out = await websearch.search("paracetamol side effects for elders")
    assert asked == ["paracetamol side effects for elders"]  # the question itself, no extra model call
    assert out["answer"] == "Paracetamol is usually gentle on the stomach. Check with your doctor first."
    assert out["sources"] == [{"title": "Paracetamol | Health guide", "url": "https://example.org/paracetamol"}]
    prompt = answerer.calls[-1]["messages"][0]["content"][0]["text"]
    assert websearch.PROMPT in prompt and "Generally well tolerated." in prompt  # same rules, the results as the only source


async def test_a_lookup_without_a_question_gets_its_query_written(monkeypatch, answerer):
    from app.brain import websearch

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    asked = []

    async def fake_tavily(query, *, max_results=5):
        asked.append(query)
        return [{"title": "t", "url": "https://example.org/x", "content": "c"}]

    monkeypatch.setattr(websearch, "tavily_search", fake_tavily)
    got = await websearch.grounded("Per 100 g nutrients of the food below, as JSON...\nFOOD: paneer", role="nutrition_search")
    assert asked == ["paracetamol side effects elderly"] and got.model == "tavily+fake" and got.sources


async def test_tavily_without_its_key_fails_loudly(monkeypatch):
    from app.brain import websearch

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        await websearch.tavily_search("x")


def test_gemini_is_the_default_only_where_a_google_project_is_set(monkeypatch):
    from app.brain import websearch

    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "some-project")
    assert websearch.provider() == "gemini"
    # AWS: no Google project, so no call to Vertex (live 14 Oct every search tried it and failed); Tavily once its key is set
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(websearch, "get_settings", lambda: type("S", (), {"gcp_project_id": ""})())
    assert websearch.provider() == "off"
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert websearch.provider() == "tavily"
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "gemini")
    assert websearch.provider() == "gemini", "an explicit choice wins"


# ── engine admin auth without Google ─────────────────────────────────────────


class _Req:
    def __init__(self, headers):
        self.headers = headers


@pytest.fixture
def admin_env(monkeypatch):
    monkeypatch.setenv("ENGINE_ADMIN_SECRET", "s" * 40)
    monkeypatch.delenv("ADMIN_API_SA", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)


async def test_caller_check_off_is_the_secret_only(admin_env, monkeypatch):
    monkeypatch.setenv("ENGINE_CALLER_CHECK", "off")
    assert await admin.verify_admin_caller(_Req({"x-admin-secret": "s" * 40})) is None
    with pytest.raises(HTTPException) as e:
        await admin.verify_admin_caller(_Req({"x-admin-secret": "wrong"}))
    assert e.value.status_code == 401
    monkeypatch.setenv("ENGINE_ADMIN_SECRET", "short")
    with pytest.raises(HTTPException) as e:
        await admin.verify_admin_caller(_Req({"x-admin-secret": "short"}))
    assert e.value.status_code == 503


async def test_caller_check_never_off_on_cloud_run(admin_env, monkeypatch):
    monkeypatch.setenv("ENGINE_CALLER_CHECK", "off")
    monkeypatch.setenv("K_SERVICE", "kawach-ai-engine")
    with pytest.raises(HTTPException) as e:  # no ADMIN_API_SA: switched off, as without the setting
        await admin.verify_admin_caller(_Req({"x-admin-secret": "s" * 40}))
    assert e.value.status_code == 503
    monkeypatch.setenv("ADMIN_API_SA", "admin@example.iam.gserviceaccount.com")
    with pytest.raises(HTTPException) as e:
        await admin.verify_admin_caller(_Req({"x-admin-secret": "s" * 40}))
    assert e.value.status_code == 401  # a Google identity is still required


async def test_scores_are_posted_with_the_secret_when_the_check_is_off(admin_env, monkeypatch):
    monkeypatch.setenv("ENGINE_CALLER_CHECK", "off")
    assert await quality.verify_eval_poster(_Req({"x-admin-secret": "s" * 40})) == "unchecked"
    with pytest.raises(HTTPException):
        await quality.verify_eval_poster(_Req({}))
    monkeypatch.setenv("ENGINE_CALLER_CHECK", "on")
    monkeypatch.delenv("EVAL_POSTER_SA", raising=False)
    with pytest.raises(HTTPException) as e:
        await quality.verify_eval_poster(_Req({"x-admin-secret": "s" * 40}))
    assert e.value.status_code == 503  # today's behaviour


# ── the legacy chat path (the doctor brief) ──────────────────────────────────


async def test_the_doctor_brief_path_goes_through_the_model_routes(settings_env, answerer):
    from app.llm import provider

    settings_env(LLM_PROVIDER="router", GCP_PROJECT_ID=None)
    out = await provider.chat_invoke("You write a doctor brief.", "WEB RESULTS none here")
    assert out.startswith("Paracetamol") and answerer.calls[-1]["system_stable"] == ["You write a doctor brief."]
    assert provider.llm_provider_label() == "router:fake:m@global"


def test_roles_left_on_google_are_named(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["bedrock:in.moonshotai.kimi-k3@ap-south-1"] for r in router.ROLES_IN_USE if r != "speech"}))
    assert router.google_bound_roles() == ["speech"]
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["bedrock:openai.gpt-oss-120b-1:0@ap-south-1"] for r in router.ROLES_IN_USE}))
    assert router.google_bound_roles() == []
