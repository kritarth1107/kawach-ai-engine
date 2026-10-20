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
