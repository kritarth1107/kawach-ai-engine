"""Model routes for the nightly world and the replay gate.

EVAL is the Bedrock route the gate calls. It stays on Bedrock while sunny-ship refuses Gemini. LIVE is a copy of the
production task and may later point at Gemini in kavach-care; the gate never calls that project. Gemini replay is
allowed only in sunny-ship, and only when a preflight call answers. Deploys swap the image only: nothing here writes
the task definition back.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

SUNNY = "sunny-ship-508214-q1"
CARE = "kavach-care"
# The gate and the nightly. Bedrock ap-south-1, copied from the task on 14 Oct before the brain moved back to Gemini.
# Do not point this at kavach-care: no test or eval traffic goes there.
EVAL: dict[str, list[str]] = {
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
# What families get. The gate does not follow this when it names Gemini in kavach-care.
LIVE: dict[str, list[str]] = {k: list(v) for k, v in EVAL.items()}
SIM = "bedrock:openai.gpt-oss-120b-1:0@ap-south-1"
VERTEX_PROJECTS = {CARE, SUNNY}
# A refusal of the account (policy, auth, 5xx). A 429 is capacity, not a refusal: the run still judges what it got.
DOWN = re.compile(
    r"PERMISSION_DENIED|UNAUTHENTICATED|AccessDenied|UnrecognizedClient|ExpiredToken|InvalidClient"
    r"|\b(401|403|500|502|503|504)\b|INTERNAL|UNAVAILABLE|refused by policy|terms of use"
)


def project() -> str:
    return os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or ""


def for_eval() -> dict[str, list[str]]:
    """Bedrock routes. Never Gemini in kavach-care, even after LIVE moves there."""
    routes = {k: list(v) for k, v in EVAL.items()}
    routes["sim"] = [SIM]
    return routes


def _calls_care(routes: dict) -> bool:
    if project() == CARE:
        return True
    return any(CARE in str(spec) for specs in routes.values() for spec in specs)


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
    """Point this process at Bedrock. Gemini is kept only for sunny-ship; kavach-care is refused."""
    raw = os.getenv("MODEL_ROUTES", "").strip()
    if raw:
        try:
            routes = json.loads(raw)
        except json.JSONDecodeError:
            print("Refusing: MODEL_ROUTES is not valid JSON.")
            raise SystemExit(2) from None
    else:
        routes = for_eval()
    if _calls_care(routes):
        print("Refusing: replay and nightly never call Gemini in kavach-care.")
        raise SystemExit(2)
    if vertex_routes(routes):
        if project() != SUNNY:
            print("Gemini replay runs only in sunny-ship, and only when that project answers. This run stays on Bedrock.")
            routes = for_eval()
        # sunny-ship Gemini stays. preflight drops it if the project refuses.
    os.environ["MODEL_ROUTES"] = json.dumps(routes)
    os.environ.setdefault("WEB_SEARCH_PROVIDER", "off")
    os.environ.setdefault("MEMORY_EMBEDDINGS", "off")
    os.environ.setdefault("LLM_PROVIDER", "router")
    return routes


def use_bedrock() -> dict[str, list[str]]:
    """Drop a refused sunny-ship route and continue on Bedrock."""
    os.environ["MODEL_ROUTES"] = json.dumps(for_eval())
    return apply()


async def _ping_once() -> str | None:
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


async def ping() -> str | None:
    """One short brain call. A refused sunny-ship Gemini route falls back to Bedrock and is tried once more."""
    routes = apply()
    why = await _ping_once()
    if why and vertex_routes(routes):
        print(f"sunny-ship refused; the gate runs on Bedrock ({why[:160]})")
        use_bedrock()
        why = await _ping_once()
    return why


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
