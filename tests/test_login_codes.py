"""A login code they send goes in whole, and a code that does not go in is never asked for again and again (PharmEasy,
live 2026-10-13: its four code boxes took three digits from the agent, the agent restarted the login and asked for new
codes until the number was blocked, and another family was asked for a code six times)."""

from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.sim.agent import CART, OTP, FakeAgent
from app.tasks import codeentry, runtime
from app.tasks.models import Task

FAM, ME = "fam-lc", "self-lc"
NOT_TAKEN = {"logged_in": False, "needs_otp": False, "blocked": False, "problem": "code not accepted: the boxes kept 3 digits"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


@dataclass
class CdpAgent(FakeAgent):
    async def cdp_url(self, session_id):
        return "http://cdp.test"


class H:
    def __init__(self, sessions, agent):
        self.sessions, self.agent, self.told = sessions, agent, []

    async def profile_for(self, task):
        return {"profileId": f"prof-{task.service}", "loginPhone": "9000033333"}

    async def notify(self, family_id, requested_by, prompt):
        self.told.append(prompt)

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def _order(db) -> Task:
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="pharmeasy", kind="order", goal="Almox 250mg",
                             details={"items": [{"name": "Almox 250mg strip of 10", "qty": 1}], "for_set": True})
    await db.commit()
    return t


async def _code(db, t, value="4821"):
    await db.refresh(t)
    assert t.input_needed == "otp"
    assert await runtime.provide_input(db, t, kind="otp", value=value, by=ME, by_is_elder=True) == "code received; continuing"
    await db.commit()


async def test_the_code_is_typed_into_the_boxes_by_code_and_the_agent_only_carries_on(db, at, sessions, monkeypatch):
    at("2026-10-13 02:12")
    typed = []

    async def type_code(cdp, code, *, hosts=(), **kw):
        typed.append((cdp, code, hosts))
        return True

    monkeypatch.setattr(codeentry, "type_code", type_code)
    agent = CdpAgent(script={"prepare": [OTP, CART]})
    h = H(sessions, agent)
    t = await _order(db)
    await h.tick(); await h.tick()
    await _code(db, t)
    await h.tick()
    assert typed == [("http://cdp.test", "4821", ("pharmeasy.in", "pharmeasy.in"))]
    goal = agent.runs[-1]["goal"]
    assert "already typed into the code boxes" in goal and "4821" not in goal and "do not restart the login" in goal
    await db.refresh(t)
    assert any("typed into the code boxes" in x["note"] for x in t.history) and not any("4821" in x["note"] for x in t.history)


async def test_a_v2_browser_is_not_told_to_ask_for_a_v4_secret(db, at, sessions, monkeypatch):
    """The fast path's browser is v2. V4's login_code secret does not exist there, so the digits go in that run's goal."""
    at("2026-10-16 22:40")
    monkeypatch.setenv("BROWSER_API", "v4")
    agent = FakeAgent(script={"prepare": [OTP, CART]})
    h = H(sessions, agent)
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    t.agent_session = "v2-fast"
    t.details = {**(t.details or {}), "session_api": "v2", "browser_api": "v2", "fast_session": True}
    await db.commit()
    await _code(db, t)
    await h.tick()
    goal = agent.runs[-1]["goal"]
    assert agent.runs[-1]["api"] == "v2" and agent.runs[-1]["session"] == "v2-fast"
    assert "4821" in goal and "secret named login_code" not in goal


async def test_a_v4_browser_gets_the_code_as_a_secret_not_in_the_goal(db, at, sessions, monkeypatch):
    at("2026-10-16 22:41")
    monkeypatch.setenv("BROWSER_API", "v4")
    agent = FakeAgent(script={"prepare": [OTP, CART]})
    h = H(sessions, agent)
    t = await _order(db)
    await h.tick(); await h.tick()
    await _code(db, t)
    await h.tick()
    goal = agent.runs[-1]["goal"]
    assert agent.runs[-1]["api"] == "v4"
    assert "4821" not in goal and "secret named login_code" in goal


async def test_without_the_page_the_agent_types_it_key_by_key_and_never_asks_for_another(db, at, sessions):
    at("2026-10-13 02:12")
    agent = FakeAgent(script={"prepare": [OTP, CART]})  # no way into its page here: the agent types it
    h = H(sessions, agent)
    t = await _order(db)
    await h.tick(); await h.tick()
    await _code(db, t)
    await h.tick()
    goal = agent.runs[-1]["goal"]
    assert "4821" in goal and "type the digits one key at a time" in goal and "never paste it" in goal
    assert "Never ask for a new code and never restart the login yourself" in goal
    assert "Never ask the site for another code and never restart the login by yourself" in agent.runs[-1]["hints"]


async def test_a_code_that_does_not_go_in_is_asked_once_more_then_it_stops_and_says_so(db, at, sessions):
    at("2026-10-13 02:30")
    agent = FakeAgent(script={"prepare": [OTP, OTP, NOT_TAKEN]})
    h = H(sessions, agent)
    t = await _order(db)
    await h.tick(); await h.tick()
    await _code(db, t, "1111")
    await h.tick(); await h.tick()  # it came back asking for a code: that one did not go in
    await db.refresh(t)
    assert t.input_needed == "otp" and "did not go in on PharmEasy" in h.told[-1] and "newest code" in h.told[-1]
    await _code(db, t, "2222")
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "failed" and "did not take the login code (2 codes tried)" in h.told[-1]
    assert "no more codes are asked for" in h.told[-1] and "PharmEasy app" in h.told[-1]
    assert sum(1 for x in h.told[:-1] if "code" in x and "ask" in x.lower()) == 2, "two asks for a code, never six"


def test_a_code_that_did_not_go_in_is_news_not_the_same_ask_again():
    t = Task(status="needs_input", phase="prepare", input_needed="otp", details={"codes_sent": 1})
    first = runtime._stage(t)
    t.details = {"codes_sent": 1, "code_fails": 1}
    assert runtime._stage(t) != first, "the ask after a code that did not go in must reach them"
