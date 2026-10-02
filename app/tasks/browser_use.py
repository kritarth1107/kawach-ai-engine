"""Browser Use Cloud: goal-driven browser agents on stealth Chrome with an India proxy.

The agent works from a goal plus a service's skill hints, not from scripted clicks, so a changed
page does not stop it. Follow-up tasks run in the same session (same tab, same login), which is how
a task pauses for an OTP or a confirmation and then carries on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

API = "https://api.browser-use.com/api/v2"


@dataclass
class AgentRun:
    task_id: str
    session_id: str
    status: str  # created | started | finished | failed | stopped
    output: dict | None = None
    error: str | None = None
    live_url: str | None = None
    steps: int = 0


class BrowserAgent(Protocol):
    async def run(self, *, goal: str, hints: str, schema: dict, session_id: str | None, profile_id: str | None,
                  start_url: str | None, max_steps: int, metadata: dict) -> AgentRun: ...

    async def poll(self, task_id: str) -> AgentRun: ...

    async def stop(self, task_id: str, *, end_session: bool = False) -> None: ...

    async def live_url(self, session_id: str) -> str | None: ...

    async def new_profile(self, name: str) -> str: ...


class BrowserUseCloud:
    def __init__(self, api_key: str | None = None, llm: str | None = None) -> None:
        self.key = api_key or os.getenv("BROWSER_USE_API_KEY", "")
        self.llm = llm or os.getenv("BROWSER_AGENT_LLM", "browser-use-2.0")

    async def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.request(method, f"{API}{path}", headers={"X-Browser-Use-API-Key": self.key}, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"browser-use {method} {path.split('/')[1]} {r.status_code}: {r.text[:200]}")
        return r.json() if r.content else {}

    async def run(self, *, goal, hints, schema, session_id, profile_id, start_url, max_steps, metadata) -> AgentRun:
        body: dict = {
            "task": goal,
            "llm": self.llm,
            "maxSteps": max_steps,
            "structuredOutput": json.dumps(schema),
            "systemPromptExtension": hints[:9500],
            "metadata": {k: str(v)[:100] for k, v in list(metadata.items())[:10]},
            "vision": True,
        }
        if start_url:
            body["startUrl"] = start_url
        if session_id:
            body["sessionId"] = session_id
        else:
            body["sessionSettings"] = {"proxyCountryCode": "in", **({"profileId": profile_id} if profile_id else {})}
        out = await self._req("POST", "/tasks", body)
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
        )

    async def stop(self, task_id: str, *, end_session: bool = False) -> None:
        await self._req("PATCH", f"/tasks/{task_id}", {"action": "stop_task_and_session" if end_session else "stop"})

    async def live_url(self, session_id: str) -> str | None:
        s = await self._req("GET", f"/sessions/{session_id}")
        return s.get("liveUrl")

    async def new_profile(self, name: str) -> str:
        return (await self._req("POST", "/profiles", {"name": name[:100]}))["id"]
