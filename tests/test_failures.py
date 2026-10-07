"""What a person gets when parts of the system fail: models, the backend host, a single tool."""

import pytest

from app.brain.loop import TurnRequest, run_turn
from app.care import store
from app.llm import router, spend
from app.llm.router import AllModelsFailed, LLMReply, ModelUnavailable, ToolCall
from app.sim.world import SimHost

FAM = "fam-fail"
ELDER = {"id": "elder-x", "name": "Kamla", "role": "elder"}
SON = {"id": "cg-x", "name": "Ankit", "role": "primary caregiver"}


class Flaky:
    """Plays the scripted replies; an Exception item is raised instead (the model went down)."""

    def __init__(self, items):
        self.items = list(items)

    async def complete(self, route, **kw):
        item = self.items.pop(0) if self.items else ModelUnavailable(503, "down")
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def model(monkeypatch):
    def use(items):
        router.register_provider("fake", Flaky(items))

    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield use
    router._providers.pop("fake", None)
    router.reset_breakers()


def _req(text, ref):
    return TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text=text, message_ref=ref)


async def _seed(db):
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="Haan beta, dawai le li hai", speaker_id=ELDER["id"])
    await db.commit()


async def test_model_down_before_anything_raises_for_fallback(db, at, model):
    at("2026-10-02 10:00")
    await _seed(db)
    model([ModelUnavailable(503, "down")])
    with pytest.raises(AllModelsFailed):
        await run_turn(db, SimHost(), _req("Mera BP 130/80 hai", "a1"))


async def test_model_down_after_a_save_keeps_it_and_says_so(db, at, model):
    at("2026-10-02 10:00")
    await _seed(db)
    model([
        LLMReply(text="", tool_calls=[ToolCall("c1", "remember", {"domain": "diet", "name": "low_salt", "details": {},
                                                                   "sentence": "Low salt diet"})], model="fake"),
        ModelUnavailable(503, "down"),
    ])
    out = await run_turn(db, SimHost(), _req("Doctor ne kaha namak kam karo", "a2"))
    assert "नोट कर ली" in out.reply  # she writes Hinglish: Hindi, in Devanagari
    assert [f for f in await store.facts(db, FAM, ELDER["id"]) if f.key == "diet:low_salt"]
    # The same message again (a retry) is a duplicate: answered from the stored reply, nothing redone.
    again = await run_turn(db, SimHost(), _req("Doctor ne kaha namak kam karo", "a2"))
    assert again.duplicate and again.reply == out.reply


async def test_backend_host_error_reaches_the_model_not_the_person(db, at, model):
    at("2026-10-02 10:00")
    await _seed(db)

    class DownHost(SimHost):
        async def call(self, tool, args, **kw):
            if tool == "send_whatsapp":
                raise RuntimeError("backend 502")
            return await super().call(tool, args, **kw)

    model([
        LLMReply(text="", tool_calls=[ToolCall("c1", "send_message", {"to": SON["id"], "text": "अंकित, मम्मी आपको याद कर रही हैं।"})], model="fake"),
        LLMReply(text="अभी अंकित तक संदेश नहीं पहुँचा, थोड़ी देर में फिर कोशिश करती हूँ।", tool_calls=[], model="fake"),
    ])
    out = await run_turn(db, DownHost(), _req("Ankit ko bolo phone kare", "a3"))
    failed = [a for a in out.actions if a["tool"] == "send_message"][0]
    assert not failed["ok"] and "backend 502" in failed["error"]
    assert "नहीं पहुँचा" in out.reply


async def test_unknown_tool_and_bad_args_do_not_crash(db, at, model):
    at("2026-10-02 10:00")
    await _seed(db)
    model([
        LLMReply(text="", tool_calls=[ToolCall("c1", "fly_to_moon", {}), ToolCall("c2", "log_dose", {"bogus": 1})], model="fake"),
        LLMReply(text="Theek hai ji.", tool_calls=[], model="fake"),
    ])
    out = await run_turn(db, SimHost(), _req("dawai le li", "a4"))
    assert out.reply == "Theek hai ji." and all(not a.get("ok", True) for a in out.actions if a["tool"] == "log_dose")
