"""Browser Use Cloud: goal-driven browser agents on stealth Chrome with an India proxy.

The agent works from a goal plus a service's skill hints, not from scripted clicks, so a changed
page does not stop it. Follow-up tasks run in the same session (same tab, same login), which is how
a task pauses for an OTP and then carries on.

Cost and safety (docs.browser-use.com, API v2):
- each run may only visit the store's own domains (allowedDomains), so a stray link cannot lead it off-site
- each task's browser is created on its own with keepAlive (a browser auto-created by a task closes when that run
  ends, losing an OTP screen); the runtime stops it when the task ends or waits long on a person (stop_session),
  and the family's saved profile keeps the login for the next run
- one HTTP client is reused per event loop (no client per call)
- step URLs come back with each run, so a run that worked can teach the next one its route
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

API = "https://api.browser-use.com/api/v2"

# Domains each store's agent may visit (its own site and sub-domains). Extend here when a store moves its login.
DOMAINS = {
    "swiggy": ["swiggy.com", "*.swiggy.com"], "instamart": ["swiggy.com", "*.swiggy.com"], "zepto": ["zepto.com", "*.zepto.com", "*.zeptonow.com"],
    "blinkit": ["blinkit.com", "*.blinkit.com"], "zomato": ["zomato.com", "*.zomato.com"],
    "apollo": ["apollopharmacy.in", "*.apollopharmacy.in", "*.apollo247.com"], "1mg": ["1mg.com", "*.1mg.com"],
    "pharmeasy": ["pharmeasy.in", "*.pharmeasy.in"], "uber": ["uber.com", "*.uber.com"], "ola": ["olacabs.com", "*.olacabs.com"],
    "rapido": ["rapido.bike", "*.rapido.bike"],
}


def allowed_domains(service: str | None, start_url: str | None) -> list[str] | None:
    if os.getenv("BROWSER_ALLOWED_DOMAINS", "on") != "on":
        return None
    if service in DOMAINS:
        return DOMAINS[service]
    host = urlparse(start_url or "").hostname
    return [host, f"*.{host.removeprefix('www.')}"] if host else None


@dataclass
class AgentRun:
    task_id: str
    session_id: str
    status: str  # created | started | finished | failed | stopped
    output: dict | None = None
    error: str | None = None
    live_url: str | None = None
    steps: int = 0
    path: list[str] = field(default_factory=list)  # pages visited, in order (for learning a route that worked)


class BrowserAgent(Protocol):
    async def run(self, *, goal: str, hints: str, schema: dict, session_id: str | None, profile_id: str | None,
                  start_url: str | None, max_steps: int, metadata: dict, llm: str | None = None, flash: bool = False) -> AgentRun: ...

    async def poll(self, task_id: str) -> AgentRun: ...

    async def stop(self, task_id: str, *, end_session: bool = False) -> None: ...

    async def live_url(self, session_id: str) -> str | None: ...

    async def new_profile(self, name: str) -> str: ...

    async def stop_session(self, session_id: str) -> None: ...


class BrowserUseCloud:
    def __init__(self, api_key: str | None = None, llm: str | None = None) -> None:
        self.key = api_key or os.getenv("BROWSER_USE_API_KEY", "")
        self.llm = llm or os.getenv("BROWSER_AGENT_LLM", "browser-use-2.0")
        self._client: httpx.AsyncClient | None = None
        self._loop = None

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30, base_url=API, headers={"X-Browser-Use-API-Key": self.key})
            self._loop = loop
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        r = await self._http().request(method, path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"browser-use {method} {path.split('/')[1]} {r.status_code}: {r.text[:200]}")
        return r.json() if r.content else {}

    async def run(self, *, goal, hints, schema, session_id, profile_id, start_url, max_steps, metadata, llm=None, flash=False) -> AgentRun:
        body: dict = {
            "task": goal,
            "llm": llm or self.llm,
            "maxSteps": max_steps,
            "structuredOutput": json.dumps(schema),
            "systemPromptExtension": hints[:9500],
            "metadata": {k: str(v)[:100] for k, v in list(metadata.items())[:10]},
            "vision": True,
        }
        if flash:
            body["flashMode"] = True  # fewer reasoning steps per action: faster on simple, well-known pages
        if start_url:
            body["startUrl"] = start_url
        domains = allowed_domains(metadata.get("service"), start_url)
        if domains:
            body["allowedDomains"] = domains
        if not session_id:
            # An auto-created session (sessionSettings on the task) closes as soon as the task finishes, so the login
            # page waiting for the code was gone when the code came (live 2026-10-09: Instamart asked for a second code).
            # A session created on its own stays alive for the follow-up runs; release_sessions stops it when done.
            sess = await self._req("POST", "/sessions", {"proxyCountryCode": "in", "keepAlive": True, "persistMemory": True,
                                                         **({"profileId": profile_id} if profile_id else {})})
            session_id = sess["id"]
        body["sessionId"] = session_id
        try:
            out = await self._req("POST", "/tasks", body)
        except RuntimeError as exc:
            if "allowedDomains" in body and (" 400" in str(exc) or " 422" in str(exc)) and "allowed" in str(exc).lower():
                logger.warning("browser-use rejected allowedDomains, retrying without: %s", exc)
                body.pop("allowedDomains")
                out = await self._req("POST", "/tasks", body)
            else:
                raise
        return AgentRun(task_id=out["id"], session_id=out["sessionId"], status="created")

    async def poll(self, task_id: str) -> AgentRun:
        t = await self._req("GET", f"/tasks/{task_id}")
        raw = t.get("output")
        parsed = None
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"text": raw}
        elif isinstance(raw, dict):
            parsed = raw
        elif raw:
            parsed = {"text": str(raw)}
        return AgentRun(
            task_id=task_id,
            session_id=t.get("sessionId", ""),
            status=t.get("status", ""),
            output=parsed,
            error=None if t.get("status") != "failed" else str(t.get("output") or "failed")[:300],
            steps=len(t.get("steps") or []),
            path=[str(st.get("url") or "") for st in (t.get("steps") or []) if isinstance(st, dict) and st.get("url")],
        )

    async def stop(self, task_id: str, *, end_session: bool = False) -> None:
        await self._req("PATCH", f"/tasks/{task_id}", {"action": "stop_task_and_session" if end_session else "stop"})

    async def live_url(self, session_id: str) -> str | None:
        s = await self._req("GET", f"/sessions/{session_id}")
        return s.get("liveUrl")

    async def new_profile(self, name: str) -> str:
        return (await self._req("POST", "/profiles", {"name": name[:100]}))["id"]

    async def stop_session(self, session_id: str) -> None:
        """Stop the cloud browser (it keeps running, and billing, after a task finishes)."""
        await self._req("PATCH", f"/sessions/{session_id}", {"action": "stop"})
