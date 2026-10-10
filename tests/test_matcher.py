"""The AI product matcher's contract (the model's answer is parsed and checked; nothing is matched by words in code), and
the order flow's rule that answers are data from the brain (yes / no, option ids), not words."""

import json

import pytest

from app.llm.router import LLMReply
from app.tasks import matcher, runtime


@pytest.fixture
def real_choose(monkeypatch, fake_matcher):
    """Undo the autouse fake: these tests drive the real choose() with a scripted model."""
    import importlib

    monkeypatch.setattr(matcher, "choose", importlib.reload(matcher).choose)
    return matcher.choose


class Model:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    async def __call__(self, role, **kw):
        self.prompts.append((role, kw["messages"][0]["content"][0]["text"]))
        if isinstance(self.answer, Exception):
            raise self.answer
        return LLMReply(text=json.dumps(self.answer), tool_calls=[], model="fake")


LISTINGS = [[{"name": "Modern Kitchens Butter Muruku", "pack": "150 g", "price": "₹35"},
             {"name": "Haldiram Peri Peri Murukku", "pack": "150 g", "price": "₹40"},
             {"name": "Peri Peri Murukku Gift Box", "pack": "4 x 150 g", "price": "₹199", "available": False}]]


async def test_the_models_choice_is_parsed_and_checked(monkeypatch, real_choose):
    model = Model({"picks": [{"item": 1, "exact": [2, 3, 9], "closest": 1, "why": "peri peri asked; butter is another kind"}]})
    monkeypatch.setattr(matcher.router, "complete", model)
    out = await real_choose([{"name": "Peri Peri Muruku", "qty": 1, "must_match": ["peri peri"]}], LISTINGS)
    assert out == [{"exact": [1], "closest": None, "why": "peri peri asked; butter is another kind"}], "unavailable / unknown listings dropped"
    role, text = model.prompts[0]
    assert role == "classify" and '"must_match": ["peri peri"]' in text and "[unavailable]" in text


async def test_only_another_variant_comes_back_as_closest_not_exact(monkeypatch, real_choose):
    monkeypatch.setattr(matcher.router, "complete", Model({"picks": [{"item": 1, "exact": [], "closest": 1, "why": "only butter"}]}))
    out = await real_choose([{"name": "Peri Peri Muruku", "qty": 1}], [LISTINGS[0][:1]])
    assert out == [{"exact": [], "closest": 0, "why": "only butter"}]


async def test_no_answer_means_the_person_picks(monkeypatch, real_choose):
    monkeypatch.setattr(matcher.router, "complete", Model(RuntimeError("model down")))
    assert await real_choose([{"name": "milk", "qty": 1}], [LISTINGS[0]]) is None


def test_items_stay_as_the_brain_gave_them():
    item = runtime.clean_item({"name": "Khakhra", "qty": "2", "must_match": ["methi", ""], "max_price": "100", "cheapest": True, "extra": "x"})
    assert item == {"name": "Khakhra", "qty": 2, "must_match": ["methi"], "max_price": 100.0, "cheapest": True}


async def test_a_confirm_is_yes_or_no_from_the_brain_not_words(db, at):
    from app.tasks.models import Task  # noqa: F401

    at("2026-10-10 21:00")
    t = await runtime.create(db, family_id="fam-m", subject_id="e", requested_by="e", service="blinkit", kind="order", goal="milk",
                             details={"items": [{"name": "milk", "qty": 1}]})
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "confirm"
    t.details = {**t.details, "cart_fp": "fp"}
    await db.commit()
    assert "pass value yes" in await runtime.provide_input(db, t, kind="confirm", value="haan", by="e", by_is_elder=True)
    assert (await runtime.provide_input(db, t, kind="confirm", value="yes", by="e", by_is_elder=True)).startswith("confirmed; placing")
