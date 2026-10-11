"""The why-trail (D-13): every piece of work carries one trace id, so "why did this happen?" has an answer.

- An inbound WhatsApp message's trace id is its own message id (so a care fact's source_ref is already its trace).
- Scheduler, dashboard and nightly work get a generated one: "dash:…", "job:…", "turn:…".
- A task (an order or a ride) keeps the trace of the message that started it. Everything the task does later (its
  runs, the confirm, placing, the outcome, the updates sent about it, the model calls) is filed under that same trace,
  with `via` naming the turn that moved it (the "yes" that confirmed it), so one order reads as one trail.

Like versions.attribution, it lives in a context variable: memory writes, ledger events, open loops, tasks, backend
calls and model spend pick it up without being passed it. This module has no database imports, so anything can use it.
"""

from __future__ import annotations

import contextlib
import contextvars
import re
import uuid
from dataclasses import dataclass

MAX_ID = 160  # the width of the trace_id columns (and of care_facts.source_ref)


@dataclass(frozen=True)
class Trace:
    id: str | None = None
    task_id: str | None = None  # the task this work is for
    via: str | None = None  # the turn's own trace, when its work is filed under a task's trace
    family_id: str | None = None


_current: contextvars.ContextVar[Trace] = contextvars.ContextVar("why_trace", default=Trace())
_spent: contextvars.ContextVar[list | None] = contextvars.ContextVar("why_spent", default=None)


def current() -> str | None:
    return _current.get().id


def task_id() -> str | None:
    return _current.get().task_id


def via() -> str | None:
    return _current.get().via


def family_id() -> str | None:
    return _current.get().family_id


def new(kind: str) -> str:
    return f"{kind}:{uuid.uuid4().hex[:20]}"


def clip(trace_id: str | None) -> str | None:
    return trace_id[:MAX_ID] if trace_id else None


@contextlib.contextmanager
def use(trace_id: str | None, *, task_id: str | None = None, via: str | None = None, family_id: str | None = None):
    """Work inside is filed under this trace (inner blocks override outer ones)."""
    token = _current.set(Trace(clip(trace_id), str(task_id) if task_id else None, clip(via), family_id or _current.get().family_id))
    try:
        yield
    finally:
        _current.reset(token)


def start(trace_id: str, *, family_id: str | None = None) -> None:
    """Set the trace for the rest of this request (each web request runs in its own context; nothing to undo)."""
    _current.set(Trace(clip(trace_id), None, None, family_id))


def detach() -> None:
    """Forget the trace and the spend collector (run inside a copied context for background work that is not part of
    the current turn: it must not be filed under it or counted in its cost)."""
    _current.set(Trace())
    _spent.set(None)


@contextlib.contextmanager
def scope():
    """Whatever is bound inside (bind_task) is undone on the way out."""
    token = _current.set(_current.get())
    try:
        yield
    finally:
        _current.reset(token)


def of_task(task) -> str:
    """A task's trace: the one it was started under, or (a task from before traces existed) one named after it."""
    return getattr(task, "trace_id", None) or f"task:{task.id}"


def _task_trace(task) -> Trace:
    here = _current.get()
    mine = of_task(task)
    return Trace(clip(mine), str(task.id), clip(here.id) if here.id and here.id != mine else here.via, task.family_id)


def bind_task(task) -> None:
    """From here on (until the enclosing scope() ends), file the work under the task's own trace."""
    _current.set(_task_trace(task))


@contextlib.contextmanager
def for_task(task):
    """File the work inside under the task's own trace; the turn that is moving it (if any) is kept as `via`."""
    token = _current.set(_task_trace(task))
    try:
        yield
    finally:
        _current.reset(token)


# ── what the model calls of one turn cost (the per-trace ledger is in app.llm.spend) ──


@contextlib.contextmanager
def collect_spend():
    """Collect every model call made inside: [(role, model, cost_inr, tokens_in, tokens_out), …]."""
    calls: list = []
    token = _spent.set(calls)
    try:
        yield calls
    finally:
        _spent.reset(token)


def note_spend(role: str, model: str, cost: float, tokens_in: int, tokens_out: int) -> None:
    calls = _spent.get()
    if calls is not None:
        calls.append((role, model, cost, tokens_in, tokens_out))


# ── nothing secret is ever stored in a trace ─────────────────────────────────

SECRET_KEYS = {"otp", "code", "password", "passcode", "pin", "cvv", "token", "secret", "card", "card_number", "upi_pin"}
# A login code sent on its own ("482913", "here 482913"). Not a number with a unit or a currency (a dose, an amount).
BARE_CODE = re.compile(r"(?<![\d.,₹])(?<!rs )(?<!rs\.)(?<!inr )\b\d{4,8}\b(?![.,]\d)(?!\s?(?:mg|mcg|ml|gm|g|iu|units?|kcal|rs|/-|%))", re.I)
SHORT_WORDS = 4
CODE_WORD = re.compile(r"(?i)\b(otp|code|pin|password|passcode|cvv)\b")  # the existing scrubber's own words


def scrub(text: str | None, known: set[str] | tuple = ()) -> str:
    """A stored trace's copy of a message, reply or argument: the existing scrubber (codes next to 'otp', 'code', 'pin'
    …), every secret value seen in this turn (a code passed to a task), and a bare 4–8 digit run in a short message or
    in one that speaks of a code anywhere (a login code sent on its own, or 'the code from the store is …')."""
    from app.care.redact import scrub_secrets

    out = scrub_secrets(text or "")
    for s in sorted({str(k) for k in known if k and len(str(k)) >= 4}, key=len, reverse=True):
        out = out.replace(s, "••••")
    return BARE_CODE.sub("••••", out) if len(out.split()) <= SHORT_WORDS or CODE_WORD.search(out) else out


def mask(value, known: set[str] | tuple = ()):
    """A copy of tool arguments (or results) with secrets masked: keys that name a secret, a task answer that is a
    login code, and every string scrubbed."""
    if isinstance(value, dict):
        coded = str(value.get("kind") or "").lower() in ("otp", "code")
        return {k: "••••" if str(k).lower() in SECRET_KEYS or (coded and k == "value") else mask(v, known) for k, v in value.items()}
    if isinstance(value, list):
        return [mask(v, known) for v in value]
    if isinstance(value, str):
        return scrub(value, known)
    return value


def secrets_in(args: dict | None) -> set[str]:
    """Values in tool arguments that are secrets: a key that names one, or a task answer that is a login code."""
    found: set[str] = set()
    a = args or {}
    for k, v in a.items():
        if isinstance(v, (str, int)) and str(k).lower() in SECRET_KEYS:
            found.add(str(v))
        elif isinstance(v, dict):
            found |= secrets_in(v)
    if str(a.get("kind") or "").lower() in ("otp", "code") and a.get("value"):
        found.add(str(a["value"]))
    return found


def summarize(args: dict | None, known: set[str] | tuple = (), limit: int = 240) -> str:
    """A short, scrubbed one-line summary of a tool's arguments for the trail."""
    def flat(v) -> str:
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {'••••' if str(k).lower() in SECRET_KEYS else flat(x)}" for k, x in v.items()) + "}"
        if isinstance(v, list):
            return "[" + ", ".join(flat(x) for x in v[:6]) + (", …" if len(v) > 6 else "") + "]"
        return str(v)

    a = dict(args or {})
    if str(a.get("kind") or "").lower() in ("otp", "code") and "value" in a:
        a["value"] = "••••"
    text = ", ".join(f"{k}={'••••' if str(k).lower() in SECRET_KEYS else flat(v)}" for k, v in a.items())
    text = scrub(" ".join(text.split()), known)
    return text if len(text) <= limit else text[: limit - 1] + "…"
