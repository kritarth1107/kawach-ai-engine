"""The nightly trainer's locks and bookkeeping (eval/trainer.py). No model calls, no GitHub."""

import asyncio
import base64
import json
import re

import pytest

from eval import trainer
from eval.trainer import Edit


@pytest.fixture()
def texts():
    t = trainer.originals()
    yield t
    trainer.use_texts(t)


def _first_line(text: str) -> str:
    return next(line for line in text.splitlines() if len(line) > 40 and not trainer.PROTECTED.search(line))


def test_only_instruction_files_and_exact_text(texts):
    line = _first_line(texts["core.md"])
    assert trainer.lock(Edit("health.md", "x", "y", "", "procedure"), texts, []) is not None
    assert trainer.lock(Edit("../tools/base.py", "x", "y", "", "procedure"), texts, []) is not None
    assert "exactly once" in trainer.lock(Edit("core.md", "not in the file at all", "y", "", "procedure"), texts, [])
    assert trainer.lock(Edit("core.md", line, line + " Check the plan first.", "", "procedure"), texts, []) is None


def test_safety_text_is_never_touched(texts):
    line = _first_line(texts["core.md"])
    assert "emergency" in trainer.lock(Edit("core.md", line, line + " Send an alert to the caregiver.", "", "knowledge"), texts, [])
    assert "emergency" in trainer.lock(Edit("core.md", line, line + " Skip the 500 mg dose.", "", "knowledge"), texts, [])


def test_size_and_rejected_edits(texts):
    line = _first_line(texts["core.md"])
    assert "characters" in trainer.lock(Edit("core.md", line, line + " word" * 200, "", "knowledge"), texts, [])
    gone = re.sub(r"\W+", " ", (line + " Check the plan first.").lower()).strip()
    assert "closed" in trainer.lock(Edit("core.md", line, line + " Check the plan first.", "", "procedure"), texts, [gone])


def test_an_edit_reaches_the_prompt_and_is_undone(texts):
    from app.brain import persona

    line = _first_line(texts["core.md"])
    trainer.use_texts(trainer.apply(texts, [Edit("core.md", line, line + " TRAINER-MARK.", "", "procedure")]))
    assert "TRAINER-MARK." in persona.for_groups(set())
    trainer.use_texts(texts)
    assert "TRAINER-MARK." not in persona.for_groups(set())
    assert trainer.prefix_tokens(texts) < trainer.PREFIX_TOKENS


def test_kinds_most_cases_first_health_and_busy_left_out():
    cases = ([{"id": f"a{i}", "kind": "spoiler", "status": "open"} for i in range(3)] + [{"id": "b", "kind": "quote", "status": "open"}]
             + [{"id": "h", "kind": "health", "status": "open"}, {"id": "f", "kind": "spoiler", "status": "fixed"}]
             + [{"id": "c", "kind": "repeat", "status": "open"}])
    got = trainer.pick(cases, groups=2, per=2, kinds=None, busy={"repeat"})
    assert list(got) == ["spoiler", "quote"] and len(got["spoiler"]) == 2


def test_app_token_is_a_signed_jwt(monkeypatch):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()).decode()
    monkeypatch.setenv("TRAINER_APP_ID", "12345")
    monkeypatch.setenv("TRAINER_APP_KEY", pem)
    head, body, sig = trainer.GitHub().app_jwt().split(".")
    pad = lambda s: base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))  # noqa: E731
    assert json.loads(pad(body))["iss"] == "12345"
    key.public_key().verify(pad(sig), f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())


def test_open_prs_block_their_kind_and_closed_ones_are_remembered():
    gh = trainer.GitHub()
    meta = lambda k, r: f'<!-- trainer {json.dumps({"kind": k, "replace": r})} -->'  # noqa: E731
    prs = [{"state": "open", "labels": [{"name": trainer.LABEL}], "body": meta("quote", "a")},
           {"state": "closed", "merged_at": None, "labels": [{"name": trainer.LABEL}], "body": meta("spoiler", "Never tell the END!")},
           {"state": "closed", "merged_at": "2026-10-21", "labels": [{"name": trainer.LABEL}], "body": meta("repeat", "merged")},
           {"state": "closed", "merged_at": None, "labels": [], "body": meta("other", "not ours")}]

    async def call(method, path, body=None, *, bearer=None):
        return prs

    gh.call = call  # type: ignore[method-assign]
    busy, rejected = asyncio.run(gh.trainer_prs())
    assert busy == {"quote"} and rejected == ["never tell the end"]


def test_replay_copies_keep_their_time_and_do_not_crash_on_bare_patterns():
    """21 Oct trainer run: every real-chat case replayed at the wrong hour (the anonymiser masked the fraction of each
    timestamp as a PIN code), and patterns copied without their text crashed the turn (KeyError 'title')."""
    from datetime import datetime, timezone

    from app.care.patterns import context_block
    from app.learn.anonymise import PINCODE
    from eval.replay_cases import _t

    assert PINCODE.sub("[PIN]", "2026-10-19T18:02:23.123456+00:00") == "2026-10-19T18:02:23.123456+00:00"
    assert PINCODE.sub("[PIN]", "Delhi 110001") == "Delhi [PIN]"
    assert _t("2026-10-19T18:02:23.[PIN]+00:00", datetime(2020, 1, 1, tzinfo=timezone.utc)).hour == 18
    assert context_block([{"key": "x", "subject_id": "p1"}], {}) == ""
    assert "Walks less" in context_block([{"key": "x"}, {"title": "Walks less", "subject_id": "p1"}], {"p1": "Maa"})


def test_the_editor_hears_how_much_room_is_left(texts):
    assert "characters of room" in trainer.room(texts)


def test_tool_descriptions_are_editable_on_one_source_line_but_not_safety_tools(texts):
    """Founder 21 Oct: the trainer may change a tool's description too (not alerts, emergencies, doses, reminders)."""
    from pathlib import Path

    assert not any(k.startswith("tool:") and trainer.PROTECTED_TOOLS.search(k) for k in texts)
    assert "tool:alert_caregiver" not in texts and "tool:log_dose" not in texts
    root = Path(trainer.__file__).resolve().parents[1]
    for key, desc in texts.items():
        if not key.startswith("tool:"):
            continue
        src = (root / trainer.tool_source(key[5:])).read_text()
        chunk = next((desc[i:i + 40] for i in range(0, max(1, len(desc) - 40), 20)
                      if src.count(desc[i:i + 40]) == 1 and desc.count(desc[i:i + 40]) == 1 and not trainer.PROTECTED.search(desc[i:i + 40])
                      and not any(ch in desc[i:i + 40] for ch in "\"\\\n{}")), None)
        if chunk:
            break
    assert chunk, "some tool description has a plain chunk"
    assert trainer.lock(Edit(key, chunk, chunk + " Check the dates asked", "", "knowledge"), texts, []) is None
    assert "quotes" in trainer.lock(Edit(key, chunk, chunk + ' say "x"', "", "knowledge"), texts, [])
    from app.brain.tools import base

    trainer.use_texts(trainer.apply(texts, [Edit(key, chunk, chunk + " TRAINER-TOOL-MARK", "", "knowledge")]))
    assert "TRAINER-TOOL-MARK" in base._TOOLS[key[5:]][0].description
    trainer.use_texts(texts)
    assert "TRAINER-TOOL-MARK" not in base._TOOLS[key[5:]][0].description
