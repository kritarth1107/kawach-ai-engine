"""Nothing from the night is emailed (founder, 17 Oct 2026: only a crisis alert emails the team)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "post_eval_scores", Path(__file__).parents[1] / "scripts" / "post_eval_scores.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[union-attr]


def test_the_night_is_never_emailed_even_with_a_key(monkeypatch, capsys):
    def urlopen(req, timeout=30):
        raise AssertionError(f"nothing may be sent: {req.full_url}")

    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    assert mod.email("Nightly score NOT STORED", "the night FAILED") is False
    assert "Nightly score NOT STORED" in capsys.readouterr().out


def test_a_score_that_cannot_be_posted_is_printed_not_emailed(monkeypatch, tmp_path, capsys):
    scores = tmp_path / "scores.json"
    scores.write_text('{"passed": false, "failures": ["recall produced no result"]}')

    def urlopen(req, timeout=30):
        raise AssertionError(f"nothing may be sent: {req.full_url}")

    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.delenv("ID_TOKEN", raising=False)
    monkeypatch.delenv("ENGINE_ADMIN_SECRET", raising=False)
    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    assert mod.main([str(scores), "--run-id", "1", "--commit", "abc"]) == 0
    assert "recall produced no result" in capsys.readouterr().out
