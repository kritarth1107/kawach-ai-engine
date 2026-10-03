"""The engine's HTTP endpoints the backend and Cloud Scheduler call."""

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api import brain as brain_api
from app.brain import tools
from app.care import store
from app.core.security import verify_api_secret
from app.db import session as db_session
from app.db.session import get_db
from app.llm import router, spend
from app.llm.router import LLMReply, ModelUnavailable, ToolCall
from app.main import app
from app.sim.agent import CART, FakeAgent
from app.sim.world import SimHost
from app.tasks import runtime
from app.tasks.models import Task

FAM = "fam-api"
ELDER = {"id": "elder-a", "name": "Kamla", "role": "elder"}
CG = {"id": "cg-a", "name": "Asha", "role": "primary caregiver"}


class Model:
    def __init__(self):
        self.replies, self.down = [], False

    async def complete(self, route, *, messages, **kw):
        if self.down:
            raise ModelUnavailable(503, "down")
        return self.replies.pop(0) if self.replies else LLMReply(text="Theek hai ji.", tool_calls=[], model="fake")


@pytest.fixture
async def api(db, monkeypatch):
    model = Model()
    router.register_provider("fake", model)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "worker": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    monkeypatch.setattr(db_session, "SessionLocal", sessions)  # jobs open their own sessions
    host = SimHost()
    monkeypatch.setattr(brain_api, "LiveHost", lambda: host)
    monkeypatch.setattr(brain_api, "ShadowHost", lambda: host)

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        c.model, c.host, c.sessions = model, host, sessions
        yield c
    app.dependency_overrides.clear()
    router._providers.pop("fake", None)
    spend.reset()


def turn_body(text, ref, speaker=ELDER, mode="live"):
    return {"family_id": FAM, "elder": ELDER, "speaker": speaker, "members": [ELDER, CG], "text": text, "message_ref": ref, "mode": mode}


async def test_turn_live_and_duplicate(api, at):
    at("2026-10-02 10:00")
    r = await api.post("/v2/turn", json=turn_body("Namaste", "w1"))
    assert r.status_code == 200 and r.json()["reply"] == "Theek hai ji."
    r2 = (await api.post("/v2/turn", json=turn_body("Namaste", "w1"))).json()
    assert r2["duplicate"] and r2["reply"] == "Theek hai ji."


async def test_turn_model_down_is_503_so_backend_falls_back(api, at):
    at("2026-10-02 10:00")
    api.model.down = True
    r = await api.post("/v2/turn", json=turn_body("Namaste", "w2"))
    assert r.status_code == 503


async def test_turn_rejects_bad_input(api):
    r = await api.post("/v2/turn", json={"family_id": FAM, "text": "hi"})
    assert r.status_code == 422


async def test_shadow_turn_uses_separate_family(api, at, db):
    at("2026-10-02 10:00")
    r = await api.post("/v2/turn", json=turn_body("Namaste", "w3", mode="shadow"))
    assert r.status_code == 200
    assert await store.recent_turns(db, "shadow:" + FAM, ELDER["id"]) and not await store.recent_turns(db, FAM, ELDER["id"])


async def test_daily_job_opens_refill_once(api, at, db):
    at("2026-10-01 08:00")
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ELDER, speaker=CG, members=[ELDER, CG])
    await tools.run(c, "remember", {"domain": "medicine", "name": "Amlodipine", "details": {"dose": "5 mg", "times": ["08:00", "20:00"]}, "sentence": "Amlodipine 5 mg twice"})
    await tools.run(c, "set_stock", {"medicine": "Amlodipine", "count": 6})
    await db.commit()
    at("2026-10-02 10:00")
    assert (await api.post("/v2/jobs/daily")).json() == {"refills": 1}
    assert (await api.post("/v2/jobs/daily")).json() == {"refills": 0}  # asked once; stays on the dashboard


async def test_weekly_checkin_is_once_per_day_even_if_retried(api, at, db):
    at("2026-10-04 18:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, CG])
    await db.commit()
    api.model.replies = [
        LLMReply(text="", tool_calls=[ToolCall("c1", "send_message", {"to": CG["id"], "text": "Asha, aap kaisi hain is hafte?"})], model="fake"),
        LLMReply(text="none", tool_calls=[], model="fake"),
        # If the retry ran a second turn, it would send this (different words, so only the job's own dedupe stops it).
        LLMReply(text="", tool_calls=[ToolCall("c2", "send_message", {"to": CG["id"], "text": "Neend aur sehat kaisi chal rahi hai aajkal, Asha ji? Thoda apna dhyan rakhiye."})], model="fake"),
        LLMReply(text="none", tool_calls=[], model="fake"),
    ]
    await api.post("/v2/jobs/weekly")
    await api.post("/v2/jobs/weekly")  # Cloud Scheduler retry
    assert len(api.host.world.sent) == 1
    # The retry is a duplicate turn: no second brain run (and no second model bill).
    runs = [t for t in await store.recent_turns(db, FAM, "saheli-scheduler") if t.role == "user"]
    assert len(runs) == 1


async def test_tasks_job_advances_orders(api, at, db, monkeypatch):
    at("2026-10-02 10:00")
    agent = FakeAgent(script={"prepare": [CART]})
    monkeypatch.setattr(tools, "task_agent", lambda: agent)
    await store.save_roster(db, FAM, ELDER, [ELDER, CG])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER["id"], requested_by=ELDER["id"], service="zepto", kind="order",
                             goal="Atta", details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    s1 = (await api.post("/v2/jobs/tasks")).json()
    s2 = (await api.post("/v2/jobs/tasks")).json()
    assert s1["started"] == 1 and s2["finished"] == 1
    await db.refresh(t)
    assert t.status == "awaiting_confirm"


async def test_extract_job_survives_a_failing_family(api, at, db):
    at("2026-10-02 10:00")
    await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="Mujhe doodh se allergy hai", speaker_id=ELDER["id"])
    await store.add_turn(db, family_id="fam-other", thread_id="e2", role="user", text="hello", speaker_id="e2")
    await db.commit()
    api.model.down = True
    r = await api.post("/v2/jobs/extract")
    assert r.status_code == 200 and r.json()["families"] >= 1


async def test_metrics_and_spend_endpoints(api, at):
    at("2026-10-02 10:00")
    m = (await api.get("/v2/agents/metrics", params={"days": 999})).json()
    assert m["days"] == 90 and "rows" in m and "channels" in m
    s = (await api.get("/v2/llm/spend")).json()
    assert {"today", "softCap", "hardCap", "rows"} <= set(s)


async def test_dash_refill_refused_for_allergy_is_422(api, at, db):
    at("2026-10-02 10:00")
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ELDER, speaker=CG, members=[ELDER, CG])
    await tools.run(c, "remember", {"domain": "medicine", "name": "Lactulose syrup milk", "details": {"dose": "10 ml", "times": ["21:00"]}, "sentence": "Lactulose 10 ml at night"})
    await tools.run(c, "remember", {"domain": "allergy", "name": "milk", "details": {"allergen": "milk"}, "sentence": "Allergic to milk"})
    await db.commit()
    key = [f.key for f in await store.facts(db, FAM, ELDER["id"]) if f.domain == "medicine"][0]
    r = await api.post(f"/v2/dash/{FAM}/{ELDER['id']}/stock/{key.split(':', 1)[1]}/order", json={"actor": {"id": CG["id"], "name": "Asha"}, "service": "apollo"})
    assert r.status_code == 422 and "allergy" in r.json()["detail"]
    assert not list((await db.execute(Task.__table__.select())).all())
