"""Drift guard for "one memory is the truth" (trust plan A1, step 5): the brain v2 paths never write the retired Instinct
tables (family_memories, memory_entities, memory_profiles). A turn that remembers, notes and recalls, and the nightly
dream, leave them untouched; and no live module writes them (only the memory move reads them, and the model file
defines them so the old rows stay readable until the backups age out)."""

import pathlib
import re

import pytest
from sqlalchemy import func, select

from app.brain.loop import TurnRequest, run_turn
from app.care import store
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.models.entities import FamilyMemory, MemoryEntity, MemoryProfile
from app.sim.world import SimHost

FAM = "fam-nolegacy"
ELDER = {"id": "elder-nl", "name": "Kamla", "role": "elder"}
SON = {"id": "son-nl", "name": "Ankit", "role": "primary caregiver"}
APP = pathlib.Path(__file__).resolve().parents[1] / "app"


class Scripted:
    def __init__(self, items):
        self.items = list(items)

    async def complete(self, route, **kw):
        return self.items.pop(0) if self.items else LLMReply(text="{}", tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    def use(items):
        router.register_provider("fake", Scripted(items))

    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield use
    router._providers.pop("fake", None)
    router.reset_breakers()


async def counts(db) -> tuple[int, int, int]:
    return (
        (await db.execute(select(func.count()).select_from(FamilyMemory))).scalar_one(),
        (await db.execute(select(func.count()).select_from(MemoryEntity))).scalar_one(),
        (await db.execute(select(func.count()).select_from(MemoryProfile))).scalar_one(),
    )


async def test_a_v2_turn_that_remembers_notes_and_recalls_writes_no_instinct_row(db, at, model):
    at("2026-10-14 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    await db.commit()
    before = await counts(db)
    model([
        LLMReply(text="", tool_calls=[
            ToolCall("c1", "remember", {"domain": "diet", "name": "low_salt", "details": {}, "sentence": "Low salt diet"}),
            ToolCall("c2", "note", {"topic": "family", "text": "Her grandson Arjun studies in Pune"}),
            ToolCall("c3", "recall", {"query": "grandson"}),
        ], model="fake"),
        LLMReply(text="Theek hai, yaad rakhungi.", tool_calls=[], model="fake"),
    ])
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER, SON],
                                                    text="Doctor ne namak kam bola. Arjun Pune mein padhta hai.", message_ref="nl1"))
    assert out.reply
    assert [f for f in await store.facts(db, FAM, ELDER["id"]) if f.key == "diet:low_salt"]
    assert await counts(db) == before


async def test_the_nightly_dream_writes_no_instinct_row(db, at, model):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.care import dream

    at("2026-10-14 02:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="aaj mandir gayi thi", speaker_id=ELDER["id"])
    await db.commit()
    before = await counts(db)
    model([])
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    await dream.dream_all(sessions, budget_s=30)
    assert await counts(db) == before


def test_the_v1_dream_no_longer_runs_with_the_nightly_job():
    src = (APP / "jobs" / "dream.py").read_text()
    main = src[src.index("def main()"):]
    assert "dream_all_elders" not in main


# Modules allowed to name the Instinct models: the definitions, the memory move (reads), the legacy-writes count (reads).
ALLOWED = {"models/entities.py", "care/memory_unify.py", "api/one_memory.py"}
# The old brain's modules, deleted after the move (founder D9). Until then they are never imported by a v2 path.
V1 = re.compile(r"^(rag/|agents/(?!tool_client\.py)|api/chat\.py|api/memory\.py|jobs/dream\.py|jobs/memory_context\.py|api/documents\.py)")
# Shared v2 infrastructure that lives in the old folders: the embedder, the backend tool client, the schema helper.
SHARED = ("app.rag.embeddings", "app.agents.tool_client", "app.jobs.dream import _ensure_v2_schema")
WRITES = re.compile(r"\b(FamilyMemory|MemoryEntity|MemoryProfile)\s*\(|insert\(\s*(FamilyMemory|MemoryEntity|MemoryProfile)\b|"
                    r"update\(\s*(FamilyMemory|MemoryEntity|MemoryProfile)\b|(family_memories|memory_entities|memory_profiles)[^\n]*(INSERT|UPDATE)", re.I)


def test_no_live_module_writes_or_imports_the_instinct_tables():
    bad = []
    for path in APP.rglob("*.py"):
        rel = path.relative_to(APP).as_posix()
        if rel in ALLOWED or V1.match(rel):
            continue
        text = path.read_text()
        if re.search(r"\b(FamilyMemory|MemoryEntity|MemoryProfile)\b", text) or WRITES.search(text):
            bad.append(rel)
        for m in re.finditer(r"from app\.(rag|agents|jobs\.dream|api\.chat|api\.memory)\b[\w.]* import ([\w, ]+)", text):
            if any(shared in m.group(0) for shared in SHARED):
                continue
            bad.append(f"{rel} imports {m.group(0)}")
    assert not bad, bad


def test_the_memory_move_only_reads_the_instinct_tables():
    text = (APP / "care" / "memory_unify.py").read_text()
    assert not WRITES.search(text)
    assert not re.search(r"session\.(add|delete)\(\s*(FamilyMemory|MemoryEntity|MemoryProfile)", text)
