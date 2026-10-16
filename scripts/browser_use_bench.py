"""Guest Browser Use comparison. It does not call the API until the founder says "start testing Browser Use".

Nothing is ordered or booked. Each task stops at the pay page, the fare, or the booking-reference form.
A missing API key still prints the plan and calls nothing. A live run stops when $45 has been spent or
$58 is left, and at most four browsers run at once. Every session is stopped when its run ends.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

PHRASE = "start testing Browser Use"
FLOOR_USD = 58.0
MAX_SPEND_USD = 45.0
CONCURRENCY = 4
_ENV = ("BROWSER_API", "BROWSER_MODEL", "BROWSER_AGENT_LLM", "BROWSER_THINKING")
_HIGH = "browse:high,prepare:high,otp:high,place:high"

# service, phase, what the run may do. Every line forbids paying.
RUNS = (
    ("blinkit", "browse", "Search for milk, add one to the cart, and open the pay page. Do not pay. Do not place the order."),
    ("zepto", "browse", "Search for milk, add one to the cart, and open the pay page. Do not pay. Do not place the order."),
    ("instamart", "browse", "Search for milk, add one to the cart, and open the pay page. Do not pay. Do not place the order."),
    ("apollo", "browse", "Search for paracetamol and report the first results. Do not add to the cart. Do not pay."),
    ("1mg", "browse", "Search for paracetamol and report the first results. Do not add to the cart. Do not pay."),
    ("uber", "browse", "Look up a fare. Do not book. Do not pay."),
    ("ola", "browse", "Look up a fare. Do not book. Do not pay."),
    ("rapido", "browse", "Look up a fare. Do not book. Do not pay."),
    ("indigo", "prepare", "Open web check-in and stop at the booking-reference form. Do not submit a booking. Do not pay."),
    ("airindia", "prepare", "Open web check-in and stop at the booking-reference form. Do not submit a booking. Do not pay."),
    ("airindiaexpress", "prepare", "Open web check-in and stop at the booking-reference form. Do not submit a booking. Do not pay."),
    ("akasa", "prepare", "Open web check-in and stop at the booking-reference form. Do not submit a booking. Do not pay."),
    ("spicejet", "prepare", "Open web check-in and stop at the booking-reference form. Do not submit a booking. Do not pay."),
)

# api, model. Every case thinks at high. v2 is last so a short budget still covers the v4 models.
CASES = (
    ("v4", "gpt-6-luna"),
    ("v4", "gemini-3.5-flash"),
    ("v4", "claude-sonnet-5"),
    ("v2", "browser-use-2.0"),
)
# Grocery carts and airline check-in pages also try Opus. Pharmacies and ride fares do not.
OPUS = ("v4", "claude-opus-5")
OPUS_SERVICES = frozenset({
    "blinkit", "zepto", "instamart",
    "indigo", "airindia", "airindiaexpress", "akasa", "spicejet",
})

_DONE = frozenset({"finished", "failed", "stopped"})
_PAID = ("placed", "booked", "paid", "pressed_place", "checked_in")


def allowed() -> bool:
    return os.environ.get("BROWSER_USE_BENCH") == PHRASE


def jobs() -> list[tuple[str, str, str, str, str]]:
    """Service, phase, task, api, model. V4 models rotate across stores before v2, then Opus where it applies."""
    waves: list[tuple[str, str]] = list(CASES[:-1])
    out: list[tuple[str, str, str, str, str]] = []
    for api, model in waves:
        for service, phase, task in RUNS:
            out.append((service, phase, task, api, model))
    for service, phase, task in RUNS:
        if service in OPUS_SERVICES:
            out.append((service, phase, task, OPUS[0], OPUS[1]))
    api, model = CASES[-1]
    for service, phase, task in RUNS:
        out.append((service, phase, task, api, model))
    return out


def succeeded(row: dict) -> bool:
    """Finished, and nothing was placed, booked, or paid."""
    if row.get("status") != "finished":
        return False
    out = row.get("output")
    if not isinstance(out, dict):
        return False
    return not any(out.get(key) is True for key in _PAID)


def _restore(saved: dict[str, str | None]) -> None:
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _over_budget(start: float | None, now: float | None, *, floor: float, max_spend: float) -> str | None:
    if now is None:
        return "balance unread"
    if now <= floor:
        return f"${now:.2f} left"
    if start is not None and start - now >= max_spend:
        return f"${start - now:.2f} spent"
    return None


async def execute(agent, *, sleep=asyncio.sleep, polls: int = 40, balance=None,
                  floor: float = FLOOR_USD, max_spend: float = MAX_SPEND_USD, concurrency: int = CONCURRENCY,
                  on_batch=None) -> list[dict]:
    """Guest runs. No family profile. At most `concurrency` at once. Stops at the credit floor."""
    from app.tasks.skills import SKILLS, schema_for

    saved = {key: os.environ.get(key) for key in _ENV}
    os.environ["BROWSER_THINKING"] = _HIGH
    lock = asyncio.Lock()
    rows: list[dict] = []
    try:
        start = await balance() if balance else None
    except Exception:
        _restore(saved)
        raise

    async def one(service: str, phase: str, task: str, api: str, model: str) -> dict:
        skill = SKILLS[service]
        t0 = time.monotonic()
        async with lock:
            os.environ["BROWSER_API"] = api
            os.environ["BROWSER_MODEL"] = model
            if api == "v2":
                os.environ["BROWSER_AGENT_LLM"] = model
            try:
                run = await agent.run(
                    goal=task,
                    hints=f"{skill['hints']} Do not pay. Do not place an order. Do not book.",
                    schema=schema_for(skill["kind"], phase),
                    session_id=None,
                    profile_id=None,
                    start_url=skill["start_url"],
                    max_steps=15 if phase == "prepare" else 25,
                    metadata={"service": service, "phase": phase, "bench": "guest"},
                    llm=model if api == "v2" else None,
                )
            except Exception as exc:  # noqa: BLE001 — one rejected model must not stop the rest
                return {"api": api, "model": model, "service": service, "phase": phase, "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"[:200], "output": None,
                        "seconds": round(time.monotonic() - t0, 1), "cost": None}
        try:
            for _ in range(polls):
                if run.status in _DONE:
                    break
                await sleep(5)
                run = await agent.poll(run.task_id, api=api)
            else:
                await agent.stop(run.task_id, end_session=True, api=api, session_id=getattr(run, "session_id", None))
                run.status = "stopped"
        finally:
            if getattr(run, "session_id", None):
                try:
                    await agent.stop_session(run.session_id, api=api)
                except Exception as exc:  # noqa: BLE001 — a stuck session is reported, the next run still starts
                    print(f"session stop failed {service} {model}: {type(exc).__name__}", file=sys.stderr)
        return {"api": api, "model": model, "service": service, "phase": phase, "status": run.status,
                "output": run.output, "error": getattr(run, "error", None),
                "seconds": round(time.monotonic() - t0, 1), "cost": getattr(run, "cost", None)}

    def _hold(reason: str, waiting) -> None:
        for service, phase, _task, api, model in waiting:
            rows.append({"api": api, "model": model, "service": service, "phase": phase,
                         "status": "not_run", "reason": reason, "output": None, "seconds": None, "cost": None})

    try:
        pending = jobs()
        if balance and (reason := _over_budget(start, start, floor=floor, max_spend=max_spend)):
            _hold(reason, pending)
            pending = []
        while pending:
            batch, pending = pending[:concurrency], pending[concurrency:]
            rows.extend(await asyncio.gather(*(one(*job) for job in batch)))
            if on_batch:
                on_batch(rows)
            if balance and pending:
                reason = _over_budget(start, await balance(), floor=floor, max_spend=max_spend)
                if reason:
                    _hold(reason, pending)
                    break
    finally:
        _restore(saved)
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not allowed():
        print("Not running. The founder has not said to start testing Browser Use.", file=sys.stderr)
        return 2
    planned = jobs()
    print(f"{len(planned)} guest runs. Nothing is ordered or booked. Stop at ${MAX_SPEND_USD:.0f} spent or ${FLOOR_USD:.0f} left.")
    for service, phase, task, api, model in planned:
        print(f"{api} {model} {service} {phase}: {task}")
    if "--run" not in argv:
        print("Ready. Pass --run with this phrase to start. Nothing was called.")
        return 0
    if not os.environ.get("BROWSER_USE_API_KEY"):
        print("No BROWSER_USE_API_KEY. Nothing was called.", file=sys.stderr)
        return 2
    from app.tasks.browser_use import BrowserUseCloud
    from app.tasks.credits import balance

    cloud = BrowserUseCloud()
    before = asyncio.run(balance(cloud))
    if before is None or before <= FLOOR_USD:
        print(f"Not running. Balance ${before}. Nothing was called.", file=sys.stderr)
        return 2
    out_path = os.environ.get("BENCH_OUT")

    def _save(got: list[dict]) -> None:
        if not out_path:
            return
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"before_usd": before, "rows": got}, fh, default=str)
        last = got[-1]
        print(f"so far {len(got)} last {last.get('status')} {last.get('service')} {last.get('model')}", flush=True)

    rows = asyncio.run(execute(cloud, balance=lambda: balance(cloud), polls=120, on_batch=_save))
    after = asyncio.run(balance(cloud))
    for row in rows:
        print(f"{row['status']} {row['api']} {row['model']} {row['service']} {row.get('seconds')}s cost={row.get('cost')}")
    spent = None if before is None or after is None else round(before - after, 2)
    print(f"Balance ${before} -> ${after}. Spent ${spent}.")
    return 0 if rows and any(row["status"] == "finished" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
