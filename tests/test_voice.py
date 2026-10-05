"""Voice notes: the brain knows a message was spoken (a transcript that can be wrong) and that its reply is read aloud;
a medicine, allergy or condition heard in an unsure transcript waits for a confirmation, whoever said it."""

import json

import pytest

from app.brain import tools
from app.brain.loop import TurnRequest, run_turn, turn_context
from app.care import store
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost

FAM = "fam-voice"
ELDER = {"id": "e-voice", "name": "Kamla", "role": "elder"}
SON = {"id": "c-voice", "name": "Ankit", "role": "primary caregiver"}


class Scripted:
    def __init__(self, items):
        self.items, self.seen = list(items), []

    async def complete(self, route, **kw):
        self.seen.append(kw)
        return self.items.pop(0) if self.items else LLMReply(text="Theek hai.", tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    holder = {}

    def use(items):
        holder["m"] = Scripted(items)
        router.register_provider("fake", holder["m"])
        return holder["m"]

    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield use
    router._providers.pop("fake", None)
    router.reset_breakers()


def req(text, speaker=ELDER, **kw):
    return TurnRequest(family_id=FAM, elder=ELDER, speaker=speaker, members=[ELDER, SON], text=text, message_ref=kw.pop("ref", None), **kw)


def everything(kw: dict) -> str:
    return json.dumps(kw, default=str, ensure_ascii=False)


async def test_text_message_has_no_voice_block(db, at):
    at("2026-10-02 10:00")
    dynamic, _ = await turn_context(db, req("BP 130/80 aaya"))
    assert "VOICE NOTE" not in dynamic


async def test_voice_note_tells_the_brain_it_is_a_transcript_spoken_back(db, at):
    at("2026-10-02 10:00")
    dynamic, _ = await turn_context(db, req("aaj bp ek sau tees by assi aaya", modality="voice", voice_confidence=0.92))
    assert "THIS MESSAGE WAS A VOICE NOTE" in dynamic and "read aloud" in dynamic and "UNSURE" not in dynamic


@pytest.mark.parametrize("conf,text", [(0.41, "metformin ek hazaar kar do"), (None, "haan")])
async def test_unsure_voice_note_asks_to_repeat_before_changing_medicines(db, at, conf, text):
    at("2026-10-02 10:00")
    dynamic, _ = await turn_context(db, req(text, modality="voice", voice_confidence=conf))
    assert "THE TRANSCRIPT IS UNSURE" in dynamic and "say it again" in dynamic


async def test_turn_marks_the_voice_note_in_history_and_prompt(db, at, model):
    at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    m = model([LLMReply(text="Bahut accha, Kamla ji.", tool_calls=[], model="fake")])
    out = await run_turn(db, SimHost(), req("dawai le li", modality="voice", voice_confidence=0.9, ref="v1"))
    assert out.reply
    assert "(voice note, transcript) dawai le li" in everything(m.seen[0])
    turns = await store.recent_turns(db, FAM, ELDER["id"])
    user = next(t for t in turns if t.role == "user")
    assert user.meta.get("voice") is True and user.meta.get("voice_confidence") == 0.9


async def test_unsure_voice_note_from_a_caregiver_cannot_change_a_dose_directly(db, at, model):
    at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    await store.write_fact(db, family_id=FAM, subject_id=ELDER["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}, text="Metformin 500 mg at 08:00",
                           source_kind="caregiver_said", stated_by=SON["id"])
    await db.commit()
    change = ToolCall("c1", "remember", {"domain": "medicine", "name": "Metformin", "details": {"name": "Metformin", "dose": "1000 mg"},
                                         "sentence": "Metformin 1000 mg"})
    model([LLMReply(text="", tool_calls=[change], model="fake"), LLMReply(text="Aapne 1000 mg kaha? Pakka?", tool_calls=[], model="fake")])
    out = await run_turn(db, SimHost(), req("metformin hazaar", speaker=SON, modality="voice", voice_confidence=0.4, ref="v2"))
    act = await store.active_fact(db, FAM, ELDER["id"], "medicine:metformin")
    assert act.value["dose"] == "500 mg" and out.actions[0]["ok"]
    pending = await store.facts(db, FAM, ELDER["id"], statuses=("pending",))
    assert pending and pending[0].value["dose"] == "1000 mg"


async def test_sure_voice_note_from_a_caregiver_applies_as_usual(db, at, model):
    at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    change = ToolCall("c1", "remember", {"domain": "medicine", "name": "Metformin",
                                         "details": {"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}, "sentence": "Metformin 500 mg at 08:00"})
    model([LLMReply(text="", tool_calls=[change], model="fake"), LLMReply(text="Save kar diya.", tool_calls=[], model="fake")])
    await run_turn(db, SimHost(), req("metformin paanch sau subah aath baje", speaker=SON, modality="voice", voice_confidence=0.95, ref="v3"))
    assert (await store.active_fact(db, FAM, ELDER["id"], "medicine:metformin")).value["dose"] == "500 mg"


async def test_unsure_flag_does_not_touch_non_health_facts(db, at):
    at("2026-10-02 10:00")
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ELDER, speaker=SON, members=[ELDER, SON], voice_unsure=True)
    out, err = await tools.run(c, "remember", {"domain": "dish", "name": "poha", "details": {}, "sentence": "Makes poha on Sundays"})
    assert not err and json.loads(out)["result"] == "created"


# ── voice replies preference (WhatsApp tool; the dashboard sets the same backend value) ──

VIEWER = {"id": "v-voice", "name": "Neel", "role": "family member"}


def c(db, speaker, host):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker=speaker, members=[ELDER, SON, VIEWER])


async def test_voice_replies_set_and_read_by_the_right_people(db, at):
    at("2026-10-02 10:00")
    host = SimHost()
    out, err = await tools.run(c(db, ELDER, host), "voice_replies", {"mode": "always"})  # she asks for voice herself
    assert not err and host.world.voice_modes[ELDER["id"]] == "always"
    out, err = await tools.run(c(db, ELDER, host), "voice_replies", {})
    assert not err and json.loads(out)["mode"] == "always"
    out, err = await tools.run(c(db, SON, host), "voice_replies", {"mode": "never", "about": SON["id"]})  # caregiver, for himself
    assert not err and host.world.voice_modes[SON["id"]] == "never"
    out, err = await tools.run(c(db, ELDER, host), "voice_replies", {"mode": "never", "about": SON["id"]})
    assert err  # the elder sets only her own
    out, err = await tools.run(c(db, VIEWER, host), "voice_replies", {"mode": "never"})
    assert err and host.world.voice_modes[ELDER["id"]] == "always"  # a view-only member cannot change it
    sys = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker={"id": "saheli", "role": "system"}, members=[ELDER, SON])
    out, err = await tools.run(sys, "voice_replies", {"mode": "never"})
    assert err


async def test_voice_preference_is_not_written_in_shadow_mode(db, at):
    from app.brain.host import WRITE_TOOLS

    assert "set_voice_preference" in WRITE_TOOLS and "get_voice_preference" not in WRITE_TOOLS
