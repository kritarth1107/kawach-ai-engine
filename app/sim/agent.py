"""A fake browser agent for simulations and tests: each run finishes with a scripted report."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from app.tasks.browser_use import AgentRun

CART = {
    "logged_in": True, "needs_otp": False, "blocked": False, "problem": "",
    "items": [{"name": "Aashirvaad Atta 5kg", "qty": 1, "price": "₹289", "available": True}],
    "total": "₹318", "fees": "₹29", "cod_available": True, "eta": "18 min",
}
PLACED = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "placed": True, "order_id": "IM-55821", "payment_method": "Cash on Delivery", "eta": "17 min", "total": "₹318"}
OTP = {"logged_in": False, "needs_otp": True, "otp_sent_to": "+91 ******9888", "blocked": False, "problem": "login needs OTP"}
CANCELLED = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "cancelled": True}
FARES = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "options": [{"type": "Auto", "fare": "₹142", "eta": "4 min"}, {"type": "Uber Go", "fare": "₹236", "eta": "6 min"}], "surge": False}
BOOKED = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": "", "booked": True, "ride_id": "UB-9921", "driver": "Ramesh, KA01AB1234", "status": "driver assigned"}


@dataclass
class FakeAgent:
    """script: per phase, the reports to return in order (the last one repeats)."""

    script: dict[str, list[dict]] = field(default_factory=dict)
    finish_after_polls: int = 1
    steps_per_run: int = 12
    runs: list[dict] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    sessions_stopped: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=lambda: ["https://shop.test/", "https://shop.test/search?q=atta", "https://shop.test/cart", "https://shop.test/checkout"])
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))
    _polls: dict[str, int] = field(default_factory=dict)
    _out: dict[str, dict] = field(default_factory=dict)

    async def run(self, *, goal, hints, schema, session_id, profile_id, start_url, max_steps, metadata, llm=None, flash=False) -> AgentRun:
        tid = f"t{next(self._ids)}"
        phase = metadata.get("phase", "prepare")
        # "browse:zepto" scripts one store; "browse" every store
        queue = self.script.get(f"{phase}:{metadata.get('service')}") or self.script.get(phase) or [CART]
        out = queue.pop(0) if len(queue) > 1 else queue[0]
        self._out[tid] = out
        self.runs.append({"task": tid, "phase": phase, "goal": goal, "session": session_id, "profile": profile_id, "hints": hints, "llm": llm,
                          "agent": metadata.get("agent"), "service": metadata.get("service"), "flash": flash})
        return AgentRun(task_id=tid, session_id=session_id or f"s-{tid}", status="created")

    async def poll(self, task_id: str) -> AgentRun:
        if task_id in self.stopped:
            return AgentRun(task_id=task_id, session_id="", status="stopped", output=None)
        self._polls[task_id] = self._polls.get(task_id, 0) + 1
        if self._polls[task_id] < self.finish_after_polls:
            return AgentRun(task_id=task_id, session_id="", status="started")
        out = self._out[task_id]
        if "__fail__" in out:  # the run itself crashed or timed out: no report
            return AgentRun(task_id=task_id, session_id="", status="failed", output=None, error=out["__fail__"], steps=self.steps_per_run)
        return AgentRun(task_id=task_id, session_id="", status="finished", output=dict(out), steps=self.steps_per_run, path=list(self.path))

    async def stop(self, task_id: str, *, end_session: bool = False) -> None:
        self.stopped.append(task_id)

    async def live_url(self, session_id: str) -> str | None:
        return None

    async def new_profile(self, name: str) -> str:
        return f"prof-{name}"

    async def stop_session(self, session_id: str) -> None:
        self.sessions_stopped.append(session_id)
