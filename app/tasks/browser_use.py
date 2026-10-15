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
API_V4 = "https://api.browser-use.com/api/v4"
THINKING = ("disabled", "low", "medium", "high")

# Domains each store's agent may visit (its own site and sub-domains). Extend here when a store moves its login.
DOMAINS = {
    "swiggy": ["swiggy.com", "*.swiggy.com"], "instamart": ["swiggy.com", "*.swiggy.com"], "zepto": ["zepto.com", "*.zepto.com", "*.zeptonow.com"],
    "blinkit": ["blinkit.com", "*.blinkit.com"], "zomato": ["zomato.com", "*.zomato.com"],
    "apollo": ["apollopharmacy.in", "*.apollopharmacy.in", "*.apollo247.com"], "1mg": ["1mg.com", "*.1mg.com"],
    "pharmeasy": ["pharmeasy.in", "*.pharmeasy.in"], "uber": ["uber.com", "*.uber.com"], "ola": ["olacabs.com", "*.olacabs.com"],
    "rapido": ["rapido.bike", "*.rapido.bike"],
    "indigo": ["goindigo.in", "*.goindigo.in"], "airindia": ["airindia.com", "*.airindia.com"],
    "airindiaexpress": ["airindiaexpress.com", "*.airindiaexpress.com"], "akasa": ["akasaair.com", "*.akasaair.com"],
    "spicejet": ["spicejet.com", "*.spicejet.com"],
}


def _pairs(raw: str) -> dict[str, str]:
    out = {}
    for bit in (raw or "").split(","):
        if ":" not in bit:
            continue
        key, val = bit.split(":", 1)
        if val.strip() in THINKING:
            out[key.strip()] = val.strip()
    return out


def thinking_level(service: str | None, phase: str | None) -> str:
    """Login and place think harder than a look-up. BROWSER_THINKING is phase:level pairs; a service env overrides it,
    and BROWSER_THINKING_<SERVICE>_<PHASE> overrides one phase."""
    phase = phase or "browse"
    default = "high" if phase in ("prepare", "otp", "place") else "medium"
    levels = {**_pairs(os.getenv("BROWSER_THINKING", "")), **_pairs(os.getenv(f"BROWSER_THINKING_{(service or '').upper()}", ""))}
    one = os.getenv(f"BROWSER_THINKING_{(service or '').upper()}_{(phase or '').upper()}", "").strip()
    if one in THINKING:
        return one
    return levels.get(phase, default)


def api_version(service: str | None, phase: str | None) -> str:
    """BROWSER_API is v2 or v4. BROWSER_API_<SERVICE> and BROWSER_API_<SERVICE>_<PHASE> override it."""
    for key in (f"BROWSER_API_{(service or '').upper()}_{(phase or '').upper()}", f"BROWSER_API_{(service or '').upper()}", "BROWSER_API"):
        val = os.getenv(key, "").strip().lower()
        if val in ("v2", "v4"):
            return val
    return "v2"


def v4_model(service: str | None, phase: str | None) -> str:
    for key in (f"BROWSER_MODEL_{(service or '').upper()}_{(phase or '').upper()}", f"BROWSER_MODEL_{(service or '').upper()}", "BROWSER_MODEL"):
        val = os.getenv(key, "").strip()
        if val:
            return val
    return "gpt-6-luna"


def model_params(model: str, level: str, service: str | None) -> dict:
    """Provider-native reasoning for this model. BROWSER_MODEL_PARAMS_<SERVICE> or BROWSER_MODEL_PARAMS replaces it."""
    raw = os.getenv(f"BROWSER_MODEL_PARAMS_{(service or '').upper()}", "") or os.getenv("BROWSER_MODEL_PARAMS", "")
    if raw.strip():
        try:
            got = json.loads(raw)
        except json.JSONDecodeError:
            got = None
        if isinstance(got, dict):
            return got
    name = (model or "").lower()
    if name.startswith("gemini"):
        return {"thinkingConfig": {"thinkingLevel": level}}
    if name.startswith("claude"):
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
    return {"reasoning": {"effort": level}}


def secret_hosts(service: str | None, start_url: str | None) -> list[str]:
    """Bare hostnames a secret may be typed into. V4 rejects a wildcard."""
    hosts: list[str] = []
    for domain in allowed_domains(service, start_url) or []:
        host = domain.removeprefix("*.").strip().lower()
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _cost(payload: dict) -> float | None:
    """USD the service reports for this run, when the payload has it."""
    for key in ("costUsd", "cost", "totalCostUsd", "billedUsd"):
        if payload.get(key) is None:
            continue
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            return None
    return None


def _json_object(raw) -> tuple[dict | None, bool]:
    """(object, bad). Bad when a finished reply is not a JSON object."""
    if isinstance(raw, dict):
        return raw, False
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            got = json.loads(raw)
        except json.JSONDecodeError:
            return None, True
        if isinstance(got, dict):
            return got, False
    return None, True


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
    cost: float | None = None


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
        self._client_v4: httpx.AsyncClient | None = None
        self._v4: set[str] = set()
        self._loop = None

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30, base_url=API, headers={"X-Browser-Use-API-Key": self.key})
            self._loop = loop
        return self._client

    def _http_v4(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client_v4 is None or self._loop is not loop or self._client_v4.is_closed:
            self._client_v4 = httpx.AsyncClient(timeout=30, base_url=API_V4, headers={"X-Browser-Use-API-Key": self.key})
            self._loop = loop
        return self._client_v4

    async def close(self) -> None:
        for client in (self._client, self._client_v4):
            if client is not None and not client.is_closed:
                await client.aclose()
        self._client = self._client_v4 = None

    async def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        r = await self._http().request(method, path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"browser-use {method} {path.split('/')[1]} {r.status_code}: {r.text[:200]}")
        return r.json() if r.content else {}

    async def _req_v4(self, method: str, path: str, body: dict | None = None) -> dict:
        r = await self._http_v4().request(method, path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"browser-use {method} v4 {r.status_code}: {r.text[:200]}")
        return r.json() if r.content else {}

    async def run(self, *, goal, hints, schema, session_id, profile_id, start_url, max_steps, metadata, llm=None, flash=False) -> AgentRun:
        meta = dict(metadata or {})
        secret = str(meta.pop("login_code", "") or "")
        service, phase = meta.get("service"), str(meta.get("phase") or "")
        if api_version(service, phase) == "v4":
            return await self._run_v4(goal=goal, schema=schema, session_id=session_id, profile_id=profile_id, start_url=start_url,
                                      secret=secret, service=service, phase=phase)
        body: dict = {
            "task": goal,
            "llm": llm or self.llm,
            "maxSteps": max_steps,
            "structuredOutput": json.dumps(schema),
            "systemPromptExtension": hints[:9500],
            "metadata": {k: str(v)[:100] for k, v in list(meta.items())[:10]},
            "vision": True,
            "thinkingLevel": thinking_level(service, phase),
        }
        if flash:
            body["flashMode"] = True  # fewer reasoning steps per action: faster on simple, well-known pages
        if start_url:
            body["startUrl"] = start_url
        domains = allowed_domains(service, start_url)
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

    async def _run_v4(self, *, goal, schema, session_id, profile_id, start_url, secret, service, phase) -> AgentRun:
        """V4 run. The store's domains are in the task and in judge, the proxy is India, and a login code is a secret
        typed only on those hosts. The reply is still checked as JSON after the run."""
        hosts = secret_hosts(service, start_url)
        stay = f"Stay only on {', '.join(hosts)}. Do not open any other site. Do not pay." if hosts else "Do not pay."
        if secret and hosts:
            stay += " When the site asks for the login code, use the secret named login_code."
        model = v4_model(service, phase)
        level = thinking_level(service, phase)
        body: dict = {
            "task": f"{goal}\n{stay}\nReply with only a JSON object.",
            "model": model,
            "modelParams": model_params(model, level, service),
            "judge": {"context": "Done only if the reply is the JSON asked for and the browser stayed on the store's own site. "
                                 "Paying, or leaving that site, is a failure."},
            "browserSettings": {"proxyCountryCode": "in", **({"profileId": profile_id} if profile_id and not session_id else {})},
        }
        if schema:
            body["outputSchema"] = schema
        if session_id:
            body["sessionId"] = session_id
        if secret and hosts:
            body["secretBindings"] = [{"alias": "login_code", "source": {"type": "inline", "value": secret}, "allowedDomains": hosts}]
        out = await self._req_v4("POST", "/runs", body)
        self._v4.add(out["id"])
        return AgentRun(task_id=out["id"], session_id=out["sessionId"], status="created")

    async def poll(self, task_id: str) -> AgentRun:
        if task_id in self._v4:
            return await self._poll_v4(task_id)
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
            cost=_cost(t),
        )

    async def _poll_v4(self, task_id: str) -> AgentRun:
        t = await self._req_v4("GET", f"/runs/{task_id}")
        status = {"completed": "finished", "cancelled": "stopped", "failed": "failed"}.get(t.get("status") or "", "started")
        raw = t.get("output")
        if raw in (None, "") and isinstance(t.get("result"), str):
            raw = t["result"]
        parsed, bad = _json_object(raw)
        return AgentRun(
            task_id=task_id, session_id=t.get("sessionId", ""), status=status, output=parsed,
            error="bad json" if bad and status == "finished" else (None if status != "failed" else str(t.get("error") or "failed")[:300]),
            cost=_cost(t),
        )

    async def downloads(self, session_id: str, *, api: str = "v2") -> list:
        """Files the browser saved (a boarding pass). api is v2 or v4."""
        if not session_id:
            return []
        got = await (self._req_v4 if api == "v4" else self._req)("GET", f"/browsers/{session_id}/downloads")
        return list(got.get("downloads") or got.get("files") or [])

    async def stop(self, task_id: str, *, end_session: bool = False) -> None:
        if task_id in self._v4:
            await self._req_v4("POST", f"/runs/{task_id}/cancel")
            return
        await self._req("PATCH", f"/tasks/{task_id}", {"action": "stop_task_and_session" if end_session else "stop"})

    async def live_url(self, session_id: str) -> str | None:
        s = await self._req("GET", f"/sessions/{session_id}")
        return s.get("liveUrl")

    async def new_profile(self, name: str) -> str:
        return (await self._req("POST", "/profiles", {"name": name[:100]}))["id"]

    async def stop_session(self, session_id: str) -> None:
        """Stop the cloud browser. A finished run keeps billing until this. V4's browser id is not the session id."""
        if not session_id:
            return
        try:
            await self._req("PATCH", f"/sessions/{session_id}", {"action": "stop"})
            return
        except RuntimeError:
            logger.warning("v2 session stop missed %s", session_id[:8])
        try:
            data = await self._req_v4("GET", "/browsers?limit=50")
        except RuntimeError:
            return
        for item in data.get("items") or []:
            if item.get("status") != "active":
                continue
            if item.get("agentSessionId") != session_id and item.get("id") != session_id:
                continue
            await self._req_v4("PATCH", f"/browsers/{item['id']}", {"action": "stop"})

    async def open_session(self, profile_id: str | None) -> str:
        """A kept-alive cloud browser on the family's profile, for fixed fast steps and the agent runs after them."""
        sess = await self._req("POST", "/sessions", {"proxyCountryCode": "in", "keepAlive": True, "persistMemory": True,
                                                     **({"profileId": profile_id} if profile_id else {})})
        return sess["id"]

    async def cdp_url(self, session_id: str) -> str | None:
        return (await self._req("GET", f"/browsers/{session_id}")).get("cdpUrl")
