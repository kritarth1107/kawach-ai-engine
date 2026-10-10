"""Media reaches the brain as itself (live 2026-10-10: an air-cooler photo with "ye kya h" reached the brain as text only and
Saheli called it a prescription), and Saheli's internet look-up (web_search) with its daily limit."""

import json

from app.brain import tools, websearch
from app.brain.loop import TurnRequest, run_turn
from app.care import store
from app.llm import router
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-ms"
ELDER = {"id": "e-ms", "name": "Kamla", "role": "elder"}
SON = {"id": "c-ms", "name": "Ravi", "role": "primary caregiver"}


class SeeingBrain:
    def __init__(self):
        self.seen = []

    async def complete(self, route, *, messages, **kw):
        last = messages[-1]["content"] if messages[-1]["role"] == "user" else []
        self.seen.append([p.get("type") + ":" + (p.get("mime") or p.get("text", "")[:80]) for p in last])
        return LLMReply(text="Yeh ek air cooler hai.", tool_calls=[], model="fake")


async def test_a_photo_with_words_reaches_the_brain_as_media(db, at, monkeypatch):
    brain = SeeingBrain()
    router.register_provider("fake", brain)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        at("2026-10-10 22:00")
        await store.save_roster(db, FAM, ELDER, [ELDER, SON])
        res = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text="Ye kya h",
                                                       images=[{"mime": "image/jpeg", "data": "AAAA"}], message_ref="m1"))
        parts = brain.seen[0]
        assert parts[0].startswith("text:[10 Oct 22:00] (they sent a image, attached; look at it) Ye kya h")
        assert "image:image/jpeg" in parts
        turns = await store.recent_turns(db, FAM, ELDER["id"], limit=3)
        assert turns[0].meta["media"] == ["image/jpeg"] and res.reply == "Yeh ek air cooler hai."
        # a video too large to open: the brain is told, nothing is attached
        await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text="",
                                                  media_note="They sent a video too large to look at (40 MB).", message_ref="m2"))
        assert brain.seen[-1][0].endswith("(They sent a video too large to look at (40 MB).)")
    finally:
        router._providers.pop("fake", None)


def test_saheli_is_told_to_look_and_to_ask_when_there_is_no_question():
    from app.brain import persona

    assert "look at it (watch, listen, read) yourself" in persona.PERSONA
    assert "never guess what it is from memory" in persona.PERSONA


def ctx(db, speaker=ELDER):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ELDER, speaker=speaker, members=[ELDER, SON], message_ref="m")


async def test_web_search_answers_with_sources_and_is_logged(db, at, monkeypatch):
    at("2026-10-10 22:10")
    asked = []

    async def fake(q):
        asked.append(q)
        return {"answer": "No rain expected in Raipur tomorrow; sunny, 32°C.", "sources": [{"title": "timeanddate.com", "url": "https://x"}]}

    monkeypatch.setattr(websearch, "search", fake)
    out, err = await tools.run(ctx(db), "web_search", {"question": "Raipur weather tomorrow"})
    r = json.loads(out)
    assert not err and "No rain" in r["answer"] and r["sources"][0]["title"] == "timeanddate.com" and asked == ["Raipur weather tomorrow"]
    ev = await store.events(db, FAM, ELDER["id"], kinds=["web_search"])
    assert ev and ev[-1].summary == "Looked up: Raipur weather tomorrow"


async def test_web_search_has_a_daily_limit_and_fails_plainly(db, at, monkeypatch):
    at("2026-10-10 22:20")
    monkeypatch.setenv("WEB_SEARCH_PER_FAMILY_DAY", "1")

    async def ok(q):
        return {"answer": "x", "sources": []}

    monkeypatch.setattr(websearch, "search", ok)
    _, err = await tools.run(ctx(db), "web_search", {"question": "a"})
    assert not err
    out, err = await tools.run(ctx(db), "web_search", {"question": "b"})
    assert err and "used up" in out

    async def down(q):
        raise RuntimeError("quota")

    monkeypatch.setenv("WEB_SEARCH_PER_FAMILY_DAY", "5")
    monkeypatch.setattr(websearch, "search", down)
    out, err = await tools.run(ctx(db), "web_search", {"question": "c"})
    assert err and "do not guess" in out
