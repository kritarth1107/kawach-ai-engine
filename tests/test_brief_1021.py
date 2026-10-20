"""The nightly brief (live 17-20 Oct): held by an unanswered nudge, sent as one paragraph with ** bold, the same fixed
sections for everyone, and a caregiver's own web searches listed as his mother's day. Synthetic data only."""

from types import SimpleNamespace

from app.brain.tools.trips import _asker
from app.care import brief

KNOWN = "SCHEDULE:\n  07:30 Thyronorm: taken\n  21:00 Atorva: not marked\nMaa did not write to Saheli today."


def test_the_layout_survives_the_no_invention_check():
    text = ("🌙 **Maa's day — Mon 20 Oct**\n\n**Medicines**\n• 07:30 Thyronorm — taken\n• 21:00 Atorva — not marked\n\n"
            "**Health & mood**\n• No update today")
    out = brief.keep(text, KNOWN)
    assert "\n" in out and out.count("\n\n") == 2, out
    assert "**" not in out and "*Medicines*" in out and "• 07:30 Thyronorm — taken" in out


def test_sections_follow_their_choices():
    assert brief.sections_for({}, "Maa")[0] == "Medicines" and len(brief.sections_for({}, "Maa")) == 7
    left = brief.sections_for({"sections": {"leave_out": ["food", "orders"]}}, "Maa")
    assert "Food" not in left and "Orders & rides" not in left and "Medicines" in left
    assert brief.sections_for({"sections": {"only": ["medicines", "needs_you"]}}, "Maa") == ["Medicines", "Needs you"]
    p = brief.prompt_for({"id": "c1", "name": "Arun"}, {"id": "e1", "name": "Maa"}, "2026-10-20", KNOWN,
                         setting_value={"sections": {"leave_out": ["food"], "note": "keep it short"}})
    assert "never one paragraph" in p and "Food;" not in p and "keep it short" in p and "*Title*" in p


def test_a_search_is_filed_on_whoever_asked():
    members = [{"id": "c1", "role": "primary caregiver"}, {"id": "e1", "role": "elder"}]
    ctx = SimpleNamespace(speaker={"id": "c1"}, elder_id="e1", members=members, is_system=False)
    assert _asker(ctx) == "c1"
    assert _asker(SimpleNamespace(speaker={"id": "saheli"}, elder_id="e1", members=members, is_system=True)) == "e1"
    assert _asker(SimpleNamespace(speaker={"id": "stranger"}, elder_id="e1", members=members, is_system=False)) == "e1"
