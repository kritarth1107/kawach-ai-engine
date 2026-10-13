"""A tool call written into the reply as text did nothing and would reach the person as code (world probe run, 2026-10-13:
a Marwari "stop calling me Amma" got back "🏹 Stafford: default_api:remember{...}"). The brain is told once that nothing
was done and calls the tool properly; whatever still reads like code never reaches the person."""

import pytest

from app.brain import guards, tools
from app.brain.loop import TurnRequest, run_turn
from app.care import store
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost

FAM = "fam-leak"
ME = {"id": "self-l", "name": "Manisha", "role": "primary caregiver"}
LEAKED = ('🏹 Stafford: default_api:remember{"details":{"avoid":["अम्मा"],"name":"मनीषा"},"domain":"naming","name":"address_as",'
          '"sentence":"मनीषा ने \'अम्मा\' के बजाय उनके नाम \'मनीषा\' से बुलाना पसंद है।"} BMD: default_api:remember')


class Script:
    def __init__(self, items):
        self.items, self.seen = list(items), []

    async def complete(self, route, **kw):
        msgs = kw.get("messages") or []
        self.seen.append(dict(msgs[-1]) if msgs else {})  # the last message as it was at this call
        return self.items.pop(0) if self.items else LLMReply(text="ठीक है सा।", tool_calls=[], model="fake")


@pytest.fixture
def brain(monkeypatch):
    holder = {}

    def use(items):
        holder["s"] = Script(items)
        router.register_provider("fake", holder["s"])
        return holder["s"]

    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield use
    router._providers.pop("fake", None)
    router.reset_breakers()


def test_the_leaked_reply_and_other_tool_text_are_recognised_but_words_are_not():
    names = set(tools.names())
    assert guards.leaked_tool_call(LEAKED, names)
    assert guards.leaked_tool_call('remember{"domain": "diet", "name": "low_salt"}', names)
    assert guards.leaked_tool_call("log_dose(medicine='Metformin', outcome='taken')", names)
    assert guards.leaked_tool_call("<tool_call>{}</tool_call>", names)
    assert not guards.leaked_tool_call("I will remember that you like your name, Manisha ji.", names)
    assert not guards.leaked_tool_call("ठीक है, मैं याद रखूँगी कि आपको मनीषा कहूँ।", names)


async def test_a_tool_call_written_as_text_is_sent_back_and_then_done_properly(db, at, brain):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, ME, [ME])
    await db.commit()
    s = brain([
        LLMReply(text=LEAKED, tool_calls=[], model="fake"),
        LLMReply(text="", tool_calls=[ToolCall("c1", "remember", {"domain": "naming", "name": "address_as", "details": {"name": "मनीषा", "avoid": ["अम्मा"]},
                                                                   "sentence": "Call her मनीषा, not अम्मा"})], model="fake"),
        LLMReply(text="ठीक है मनीषा जी, अब से नाम से ही बुलाऊँगी।", tool_calls=[], model="fake"),
    ])
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ME, speaker=ME, members=[ME], text="म्हाने अम्मा मत कह्यो", message_ref="leak1"))
    assert "default_api" not in out.reply and "{" not in out.reply
    assert [a["tool"] for a in out.actions] == ["remember"]
    assert any(f.key == "naming:address_as" for f in await store.facts(db, FAM, ME["id"]))
    nudge = s.seen[1]["content"][0]["text"]
    assert "NOTHING was done" in nudge


async def test_tool_text_that_comes_back_again_never_reaches_the_person(db, at, brain):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, ME, [ME])
    await db.commit()
    brain([LLMReply(text=LEAKED, tool_calls=[], model="fake"), LLMReply(text=LEAKED, tool_calls=[], model="fake")])
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ME, speaker=ME, members=[ME], text="म्हाने अम्मा मत कह्यो", message_ref="leak2"))
    assert "default_api" not in out.reply
    assert out.reply  # a short, true acknowledgement instead of code
