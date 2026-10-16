"""The "I'll call you X" line is only for the speaker's own new name, and only when the draft needs replacing.

Live 16 Oct 20:38: a caregiver wrote "don't speak in Hindi"; a naming save about her mother ("Maa") made the reply
"I'll call you Maa." Nightly w0333: a draft that saved the name also repeated the prompt's own rule text.
"""

from types import SimpleNamespace

from app.brain import guards
from app.brain.loop import _asked_name
from app.brain.persona import PERSONA


def _ctx(speaker: str, actions: list[dict]):
    return SimpleNamespace(speaker={"id": speaker}, actions=actions)


def _naming(subject: str, name: str) -> dict:
    return {"tool": "remember", "ok": True, "subject": subject,
            "args": {"domain": "naming", "name": "address_as", "details": {"name": name}}}


def test_a_name_saved_for_someone_else_is_not_the_speakers_name():
    assert _asked_name(_ctx("cg-1", [_naming("elder-1", "Maa")])) == ""


def test_the_speakers_own_new_name_is_found():
    assert _asked_name(_ctx("cg-1", [_naming("elder-1", "Maa"), _naming("cg-1", "Medha")])) == "Medha"


def test_a_draft_that_copies_the_prompt_is_caught_and_a_normal_one_is_not():
    rule = " ".join(PERSONA.split()[40:52])
    assert guards.echoes_prompt(f"Theek hai. {rule}", PERSONA)
    assert not guards.echoes_prompt("Okay, I'll call you Medha and keep writing in English.", PERSONA)
    assert not guards.echoes_prompt("ok", PERSONA)
