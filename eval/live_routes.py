"""The live engine's model routes, for the nightly world and the replay gate.

Copied from the kavach-engine task in ap-south-1. Deploys swap the image only, so this file must not be edited to change
what families get, and nothing here writes the task definition back.

Text roles start at gpt-oss-120b. Photos start at Kimi K3. The fallbacks are the ones already on the task. Paid eval
calls Bedrock in ap-south-1 and never Vertex (not sunny-ship, not kavach-care).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Exact MODEL_ROUTES from kavach-engine (read 14 Oct 2026). sim is eval-only: the person's next line in a probe.
LIVE: dict[str, list[str]] = {
    "brain": ["bedrock:openai.gpt-oss-120b-1:0@ap-south-1", "bedrock:in.moonshotai.kimi-k3@ap-south-1"],
    "classify": ["bedrock:qwen.qwen3-next-80b-a3b@ap-south-1", "bedrock:openai.gpt-oss-120b-1:0@ap-south-1"],
    "extract": ["bedrock:openai.gpt-oss-120b-1:0@ap-south-1", "bedrock:qwen.qwen3-next-80b-a3b@ap-south-1"],
    "judge": ["bedrock:in.moonshotai.kimi-k3@ap-south-1", "bedrock:openai.gpt-oss-120b-1:0@ap-south-1"],
    "judge_fast": ["bedrock:qwen.qwen3-next-80b-a3b@ap-south-1", "bedrock:openai.gpt-oss-120b-1:0@ap-south-1"],
    "learn": ["bedrock:in.moonshotai.kimi-k3@ap-south-1", "bedrock:openai.gpt-oss-120b-1:0@ap-south-1"],
    "nutrition": ["bedrock:openai.gpt-oss-120b-1:0@ap-south-1", "bedrock:in.moonshotai.kimi-k3@ap-south-1"],
    "speech": ["bedrock:qwen.qwen3-next-80b-a3b@ap-south-1", "bedrock:openai.gpt-oss-120b-1:0@ap-south-1"],
    "vision": ["bedrock:in.moonshotai.kimi-k3@ap-south-1", "bedrock:qwen.qwen3-vl-235b-a22b@ap-south-1"],
    "worker": ["bedrock:openai.gpt-oss-120b-1:0@ap-south-1", "bedrock:in.moonshotai.kimi-k3@ap-south-1"],
}
SIM = "bedrock:openai.gpt-oss-120b-1:0@ap-south-1"
VERTEX_PROJECTS = {"kavach-care", "sunny-ship-508214-q1"}
# A refusal of the account (policy, auth, 5xx). A 429 is capacity, not a refusal: the run still judges what it got.
DOWN = re.compile(
    r"PERMISSION_DENIED|UNAUTHENTICATED|AccessDenied|UnrecognizedClient|ExpiredToken|InvalidClient"
    r"|\b(401|403|500|502|503|504)\b|INTERNAL|UNAVAILABLE|refused by policy|terms of use"
)


def for_eval() -> dict[str, list[str]]:
    routes = {k: list(v) for k, v in LIVE.items()}
    routes["sim"] = [SIM]
    return routes


def is_provider_down(text: str) -> bool:
    return bool(DOWN.search(text or ""))


def vertex_routes(routes: dict) -> list[str]:
    bad = []
    for role, specs in routes.items():
        for spec in specs:
            if str(spec).startswith(("gemini:", "claude:")):
                bad.append(f"{role}={spec}")
    return bad


def apply() -> dict[str, list[str]]:
    """Point this process at the live Bedrock routes. Refuses a Vertex project or a Gemini/Claude route."""
    project = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or ""
    if project in VERTEX_PROJECTS:
        print("Refusing: nightly and replay call Bedrock in ap-south-1, never Vertex (sunny-ship or kavach-care).")
        raise SystemExit(2)
    raw = os.getenv("MODEL_ROUTES", "").strip()
    if raw:
        try:
            routes = json.loads(raw)
        except json.JSONDecodeError:
            print("Refusing: MODEL_ROUTES is not valid JSON.")
            raise SystemExit(2) from None
    else:
        routes = for_eval()
        os.environ["MODEL_ROUTES"] = json.dumps(routes)
    bad = vertex_routes(routes)
    if bad:
        print("Refusing: a nightly or replay route still calls Vertex: " + ", ".join(bad[:8]))
        raise SystemExit(2)
    os.environ.setdefault("WEB_SEARCH_PROVIDER", "off")
    os.environ.setdefault("MEMORY_EMBEDDINGS", "off")
    os.environ.setdefault("LLM_PROVIDER", "router")
    return routes


async def ping() -> str | None:
    """One short brain call. Returns why the provider refused it, or None when it answered (a timeout is not a refusal)."""
    apply()
    from app.llm import router

    try:
        await router.complete(
            "brain",
            system_stable="Reply with one word.",
            messages=[{"role": "user", "content": [{"type": "text", "text": "ok?"}]}],
            max_tokens=200,
            effort="low",
            timeout_s=60,
            essential=False,
        )
    except Exception as exc:  # noqa: BLE001 — the caller decides from the text
        text = str(exc)
        if is_provider_down(text):
            return text[:200]
        return None
    return None


def main(argv: list[str]) -> int:
    if "--preflight" not in argv:
        apply()
        print(json.dumps(for_eval()))
        return 0
    why = asyncio.run(ping())
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"down={'true' if why else 'false'}\n")
    if why:
        dest = Path(os.environ.get("NIGHTLY_OUT", "runs/nightly"))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "provider_down.json").write_text(json.dumps({"why": why}))
        print(f"::warning::Bedrock refused the preflight call; this run sends nothing else: {why}")
    else:
        print("preflight ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
