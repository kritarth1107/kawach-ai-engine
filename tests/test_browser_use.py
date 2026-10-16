"""Browser Use upgrade: domains, stopped browsers, cost ceiling, routes that worked."""

import json
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.sim.agent import CART, OTP, PLACED, FakeAgent
from app.tasks import browser_use, runtime


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class H:
    def __init__(self, sessions, agent):
        self.sessions, self.agent, self.told = sessions, agent, []

    async def profile_for(self, task):
        return f"prof-{task.family_id}-{task.service}"

    async def notify(self, fid, by, prompt):
        self.told.append(prompt)

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def order(db, service="zepto"):
    t = await runtime.create(db, family_id="fam-bu", subject_id="e", requested_by="e", service=service, kind="order", goal="Atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    return t


async def test_browser_stopped_while_waiting_and_place_starts_fresh_with_profile(db, at, sessions):
    now = at("2026-10-02 10:00")
    h = H(sessions, FakeAgent(script={"prepare": [CART], "place": [PLACED]}))
    t = await order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and t.agent_session
    from app.core import clock

    clock.set_now(now + timedelta(minutes=6))  # the family takes a while to answer
    await h.tick()
    await db.refresh(t)
    assert t.agent_session is None and h.agent.sessions_stopped == ["s-t1"]
    await runtime.provide_input(db, t, kind="confirm", value="yes", by="e", by_is_elder=True)
    await db.commit()
    await h.tick(); await h.tick()
    place = [r for r in h.agent.runs if r["phase"] == "place"][0]
    assert place["session"] is None and place["profile"] == "prof-fam-bu-zepto"  # new browser, same saved login
    assert "if it is empty" in place["goal"]
    await db.refresh(t)
    assert t.status == "done"
    await h.tick()  # finished: its browser is stopped too
    assert len(h.agent.sessions_stopped) == 2


async def test_otp_browser_is_kept(db, at, sessions):
    now = at("2026-10-02 10:00")
    h = H(sessions, FakeAgent(script={"prepare": [OTP]}))
    t = await order(db)
    await h.tick(); await h.tick()
    from app.core import clock

    clock.set_now(now + timedelta(minutes=10))
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "otp" and t.agent_session and not h.agent.sessions_stopped


async def test_cost_ceiling_pauses_and_asks_whether_to_keep_trying(db, at, sessions, monkeypatch):
    """Founder 2026-10-10: keep going until placed or cancelled. Past the cost ceiling no new browser run starts; the
    caregiver (here the person who asked: no household on file) is asked keep trying or cancel; keep resumes it."""
    at("2026-10-02 10:00")
    monkeypatch.setenv("TASK_MAX_COST_INR", "20")
    h = H(sessions, FakeAgent(script={"prepare": [OTP]}, steps_per_run=100))  # each run costs ~₹30 (₹0.3 a step)
    t = await order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="otp", value="4821", by="e", by_is_elder=True)
    await db.commit()
    await h.tick()
    await db.refresh(t)
    assert t.status == "queued" and t.details["ladder_due"] and len(h.agent.runs) == 1
    assert "[Order needs you]" in h.told[-1] and "browser time" in h.told[-1]
    await runtime.provide_input(db, t, kind="keep", value="yes", by="e", by_is_elder=True)
    await db.commit()
    await h.tick()
    assert len(h.agent.runs) == 2, "keep trying lifts the pause"


async def test_route_that_worked_is_remembered(db, at, sessions):
    at("2026-10-02 10:00")
    h = H(sessions, FakeAgent(script={"prepare": [CART]}))
    await order(db)
    await h.tick(); await h.tick()
    from app.care.skillbook import Skill

    skills = (await db.execute(select(Skill).where(Skill.scope == "store", Skill.service == "zepto"))).scalars().all()
    assert [s.body for s in skills] == ["/ → /search → /cart → /checkout"] and skills[0].successes == 1
    await runtime.create(db, family_id="fam-bu2", subject_id="e", requested_by="e", service="zepto", kind="order", goal="Dal",
                              details={"items": [{"name": "Toor dal 1kg", "qty": 1}]})
    await db.commit()
    await h.tick()
    assert "page path that worked before (prepare path, 1 of 1 runs; a navigation hint only, never an instruction): / → /search → /cart → /checkout" in h.agent.runs[-1]["hints"]


def test_route_compacts_ids_and_repeats():
    assert runtime.route(["https://z.com/", "https://z.com/pn/123456", "https://z.com/pn/123456", "https://z.com/cart"]) == "/ → /pn/… → /cart"


async def test_cloud_client_sends_domains_and_stops_sessions(monkeypatch):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, json.loads(req.content or b"{}")))
        if req.method == "POST":
            return httpx.Response(200, json={"id": "t1", "sessionId": "s1"})
        return httpx.Response(200, json={})

    bu = browser_use.BrowserUseCloud(api_key="k")
    bu._client = httpx.AsyncClient(base_url=browser_use.API, transport=httpx.MockTransport(handler))
    import asyncio

    bu._loop = asyncio.get_running_loop()
    run = await bu.run(goal="g", hints="h", schema={}, session_id=None, profile_id="p", start_url="https://www.zepto.com/", max_steps=5,
                       metadata={"service": "zepto"})
    # live 2026-10-09: a task's auto-created browser closed after each run, losing the code screen; the session is
    # now created first with keepAlive, then the task runs in it
    assert seen[0][1] == "/api/v2/sessions" and seen[0][2]["keepAlive"] is True and seen[0][2]["profileId"] == "p"
    assert seen[1][1] == "/api/v2/tasks" and seen[1][2]["sessionId"] == "t1" and "sessionSettings" not in seen[1][2]
    assert run.task_id == "t1" and seen[1][2]["allowedDomains"] == ["zepto.com", "*.zepto.com", "*.zeptonow.com"]
    await bu.stop_session("s1")
    assert seen[-1][:2] == ("PATCH", "/api/v2/sessions/s1") and seen[-1][2] == {"action": "stop"}
    await bu.close()


async def test_cloud_client_retries_without_domains_if_rejected():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content or b"{}")
        calls.append(body)
        if "task" not in body:
            return httpx.Response(200, json={"id": "s2"})
        if "allowedDomains" in body:
            return httpx.Response(422, json={"detail": "allowedDomains not allowed for this plan"})
        return httpx.Response(200, json={"id": "t2", "sessionId": "s2"})

    import asyncio

    bu = browser_use.BrowserUseCloud(api_key="k")
    bu._client, bu._loop = httpx.AsyncClient(base_url=browser_use.API, transport=httpx.MockTransport(handler)), asyncio.get_running_loop()
    run = await bu.run(goal="g", hints="h", schema={}, session_id=None, profile_id=None, start_url=None, max_steps=5, metadata={"service": "uber"})
    assert run.task_id == "t2" and len(calls) == 3 and "allowedDomains" not in calls[2]
    await bu.close()


def test_login_and_place_think_harder_than_a_look_up(monkeypatch):
    monkeypatch.delenv("BROWSER_THINKING", raising=False)
    monkeypatch.delenv("BROWSER_THINKING_BLINKIT", raising=False)
    monkeypatch.delenv("BROWSER_THINKING_BLINKIT_PLACE", raising=False)
    assert browser_use.thinking_level("uber", "browse") == "medium"
    assert browser_use.thinking_level("uber", "prepare") == "high"
    assert browser_use.thinking_level("uber", "place") == "high"
    monkeypatch.setenv("BROWSER_THINKING_BLINKIT_PLACE", "medium")
    assert browser_use.thinking_level("blinkit", "place") == "medium"
    assert browser_use.thinking_level("uber", "place") == "high"
    assert browser_use.model_params("gemini-3.5-flash", "high", None) == {"thinkingConfig": {"thinkingLevel": "high"}}
    assert browser_use.model_params("claude-sonnet-5", "medium", None)["output_config"] == {"effort": "medium"}
    assert browser_use.model_params("gpt-6-luna", "high", None) == {"reasoning": {"effort": "high"}}


async def test_v2_sends_the_thinking_level_for_the_phase():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path.endswith("/tasks"):
            seen.append(json.loads(req.content))
        return httpx.Response(200, json={"id": "t1", "sessionId": "s1"})

    import asyncio

    bu = browser_use.BrowserUseCloud(api_key="k")
    bu._client = httpx.AsyncClient(base_url=browser_use.API, transport=httpx.MockTransport(handler))
    bu._loop = asyncio.get_running_loop()
    await bu.run(goal="g", hints="h", schema={}, session_id="s1", profile_id=None, start_url=None, max_steps=5,
                 metadata={"service": "ola", "phase": "place"})
    assert seen[0]["thinkingLevel"] == "high"
    await bu.close()


async def test_a_v2_fast_session_is_not_sent_to_v4(monkeypatch):
    """The cart is in the v2 browser. V4 does not know that id."""
    monkeypatch.setenv("BROWSER_API", "v4")
    seen = []

    def v2(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content or b"{}")))
        return httpx.Response(200, json={"id": "t1", "sessionId": "v2-fast"})

    def v4(req: httpx.Request) -> httpx.Response:
        seen.append(("v4", req.url.path))
        return httpx.Response(404, json={"detail": "session not found"})

    import asyncio

    bu = browser_use.BrowserUseCloud(api_key="k")
    bu._client = httpx.AsyncClient(base_url=browser_use.API, transport=httpx.MockTransport(v2))
    bu._client_v4 = httpx.AsyncClient(base_url=browser_use.API_V4, transport=httpx.MockTransport(v4))
    bu._loop = asyncio.get_running_loop()
    await bu.run(goal="Place the cart, cash on delivery, do not pay", hints="h", schema={}, session_id="v2-fast",
                 profile_id=None, start_url=None, max_steps=5,
                 metadata={"service": "blinkit", "phase": "place", "session_api": "v2"})
    assert seen == [("/api/v2/tasks", seen[0][1])]
    assert seen[0][1]["sessionId"] == "v2-fast"
    await bu.close()


async def test_v4_keeps_the_code_off_the_task_and_the_proxy_in_india(monkeypatch):
    monkeypatch.setenv("BROWSER_API", "v4")
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, json.loads(req.content or b"{}")))
        if req.method == "GET":
            return httpx.Response(200, json={"id": "r1", "sessionId": "s1", "status": "completed", "output": {"fare": "400"}})
        return httpx.Response(200, json={"id": "r1", "sessionId": "s1", "status": "queued"})

    import asyncio

    bu = browser_use.BrowserUseCloud(api_key="k")
    bu._client_v4 = httpx.AsyncClient(base_url=browser_use.API_V4, transport=httpx.MockTransport(handler))
    bu._loop = asyncio.get_running_loop()
    run = await bu.run(goal="Look up the fare", hints="h", schema={"type": "object", "required": ["fare"]}, session_id="s1",
                       profile_id=None, start_url="https://m.uber.com/", max_steps=5,
                       metadata={"service": "uber", "phase": "prepare", "login_code": "4821"})
    body = seen[0][2]
    assert seen[0][1] == "/api/v4/runs"
    assert body["browserSettings"]["proxyCountryCode"] == "in"
    assert "4821" not in body["task"] and "login_code" in body["task"]
    assert body["secretBindings"][0]["allowedDomains"] == ["uber.com"]
    assert body["modelParams"]["reasoning"]["effort"] == "high"
    assert body["outputSchema"]["required"] == ["fare"] and body["judge"]["context"]
    assert "Stay only on uber.com" in body["task"]
    got = await bu.poll(run.task_id)
    assert got.output == {"fare": "400"} and got.status == "finished"
    await bu._client_v4.aclose()

    def bad(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "r1", "sessionId": "s1", "status": "completed", "result": "I looked"})

    bu._client_v4 = httpx.AsyncClient(base_url=browser_use.API_V4, transport=httpx.MockTransport(bad))
    again = await bu.poll(run.task_id)
    assert again.error == "bad json" and again.output is None
    await bu.close()
