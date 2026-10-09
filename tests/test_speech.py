"""Voice-note script (founder 2026-10-09: speak like a person, not read a script): facts cannot drift."""

from app.care import speech
from app.llm import router
from app.llm.router import LLMReply


def test_same_points_keeps_numbers_and_length():
    assert speech.same_points("बीपी १२८ बाय ८० है", "अच्छा [short pause] बीपी 128 बाय 80 है")
    assert not speech.same_points("BP 130 by 80", "BP 130 by 90")
    assert not speech.same_points("दवाई ले ली?", "दवाई ले ली? " + "बहुत सारी नई बातें " * 6)
    assert speech.plain("अच्छा [short pause] ठीक [medium pause] है") == "अच्छा ठीक है"


class Fake:
    def __init__(self, text):
        self.text = text

    async def complete(self, route, **kw):
        return LLMReply(text=self.text, tool_calls=[], model="fake")


async def test_prepare_uses_the_script_or_falls_back(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTES", '{"speech": ["fake:m"]}')
    router.reset_breakers()
    try:
        router.register_provider("fake", Fake('{"script": "हाँ जी, [short pause] 2 बजे शेलकल 500 की 1 गोली ले लीजिए।", "mood": "gentle"}'))
        out = await speech.prepare("2 बजे शेलकल 500 की 1 गोली ले लीजिए।", "Hindi")
        assert out["prepared"] and out["mood"] == "gentle" and "[short pause]" in out["script"]
        # a changed number → the text as written
        router.register_provider("fake", Fake('{"script": "3 बजे शेलकल 500 की 1 गोली ले लीजिए।", "mood": "gentle"}'))
        out = await speech.prepare("2 बजे शेलकल 500 की 1 गोली ले लीजिए।", "Hindi")
        assert not out["prepared"] and out["script"].startswith("2 बजे")
    finally:
        router._providers.pop("fake", None)
