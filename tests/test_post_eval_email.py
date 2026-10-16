"""The nightly team email must name itself. Resend's Cloudflare answers 403 to Python-urllib."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "post_eval_scores", Path(__file__).parents[1] / "scripts" / "post_eval_scores.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[union-attr]


def test_nightly_email_names_itself(monkeypatch):
    seen = {}

    class Resp:
        status = 200

        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(req, timeout=30):
        seen["ua"] = req.get_header("User-agent")
        return Resp()

    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    assert mod.email("Nightly", "ok") is True
    assert seen["ua"] == "kavach-nightly"
    assert "Python-urllib" not in seen["ua"]
