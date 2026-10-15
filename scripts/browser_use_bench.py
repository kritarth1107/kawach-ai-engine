"""Guest Browser Use comparison. It does not call the API until the founder says "start testing Browser Use".

Nothing is ordered or booked. Each task stops at the pay page, the fare, or the booking-reference form.
The account's remaining credit is for live orders until that sentence is set in BROWSER_USE_BENCH and this
script is started with --run. A missing API key still prints the plan and calls nothing.
"""

from __future__ import annotations

import asyncio
import os
import sys

PHRASE = "start testing Browser Use"
_ENV = ("BROWSER_API", "BROWSER_MODEL", "BROWSER_AGENT_LLM")

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

# api, model. v2 uses thinkingLevel high on prepare. v4 uses the model with that effort.
CASES = (
    ("v2", "browser-use-2.0"),
    ("v4", "gpt-6-luna"),
    ("v4", "gemini-3.5-flash"),
    ("v4", "claude-sonnet-5"),
)

_DONE = frozenset({"finished", "failed", "stopped"})


def allowed() -> bool:
    return os.environ.get("BROWSER_USE_BENCH") == PHRASE


def _restore(saved: dict[str, str | None]) -> None:
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


async def execute(agent, *, sleep=asyncio.sleep, polls: int = 40) -> list[dict]:
    """One guest run per store and case. No family profile. The session is stopped when the run ends."""
    from app.tasks.skills import SKILLS, schema_for

    saved = {key: os.environ.get(key) for key in _ENV}
    rows: list[dict] = []
    try:
        for service, phase, task in RUNS:
            skill = SKILLS[service]
            for api, model in CASES:
                os.environ["BROWSER_API"] = api
                os.environ["BROWSER_MODEL"] = model
                if api == "v2":
                    os.environ["BROWSER_AGENT_LLM"] = model
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
                for _ in range(polls):
                    if run.status in _DONE:
                        break
                    await sleep(5)
                    run = await agent.poll(run.task_id)
                else:
                    await agent.stop(run.task_id, end_session=True)
                    run.status = "stopped"
                if getattr(run, "session_id", None):
                    await agent.stop_session(run.session_id)
                rows.append({"api": api, "model": model, "service": service, "phase": phase,
                             "status": run.status, "output": run.output})
    finally:
        _restore(saved)
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not allowed():
        print("Not running. The founder has not said to start testing Browser Use.", file=sys.stderr)
        return 2
    print(f"{len(RUNS)} guest flows x {len(CASES)} cases. Nothing is ordered or booked.")
    for service, phase, task in RUNS:
        for api, model in CASES:
            print(f"{api} {model} {service} {phase}: {task}")
    if "--run" not in argv:
        print("Ready. Pass --run with this phrase to start. Nothing was called.")
        return 0
    if not os.environ.get("BROWSER_USE_API_KEY"):
        print("No BROWSER_USE_API_KEY. Nothing was called.", file=sys.stderr)
        return 2
    from app.tasks.browser_use import BrowserUseCloud

    rows = asyncio.run(execute(BrowserUseCloud()))
    for row in rows:
        print(f"{row['status']} {row['api']} {row['model']} {row['service']}")
    print("Set the results into Shared/kavach-ordering/BROWSER_USE_V4_2026-10-16.md after the runs.")
    return 0 if rows and all(row["status"] == "finished" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
