"""Other ways to make a trip when a ride app cannot book it, or the fare is more than they want.

The brain passes the places and the time. A live look-up returns the page text. An AI proposes the options; code
keeps one only when its fare and its route are written in that page text. Nothing is filled in for a city, and a
missing fact is left out rather than guessed. One option is marked, from those facts: at night, for an elder, the
lowest fare that is door to door.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

AFTER_LANDING = timedelta(minutes=45)  # bags and the walk out; the same for every airport
NOT_FOUND = ("I could not check the airport taxi or the bus just now, so I will not guess a fare or a route. "
             "Nothing was booked and nothing was paid.")

KINDS = {
    "airport_taxi": "Airport prepaid taxi",
    "airport_bus": "Airport bus",
    "metro": "Metro",
    "city_bus": "City bus",
    "auto": "Auto",
    "own_app": "A cab on their own app",
}

_FIELD = re.compile(r"(\w+)=([^\s].*?)(?=\s+\w+=|$)")
_NUM = re.compile(r"\d{2,}")


def pickup_after_landing(landing: datetime) -> datetime:
    return landing + AFTER_LANDING


def _fields(line: str) -> dict[str, str]:
    return {k: v.strip() for k, v in _FIELD.findall(line)}


def _numbers_in(text: str, raw: str) -> bool:
    return all(n in raw for n in _NUM.findall(text or ""))


def options_from(proposed: str, raw: str, *, at: datetime | None, elder: bool) -> list[dict]:
    """Options whose fare and route are in the page text. A number that is only in the proposal is dropped."""
    night = bool(at) and (at.hour >= 21 or at.hour < 8)
    kept: list[dict] = []
    for line in (proposed or "").splitlines():
        if not line.strip().startswith("OPTION "):
            continue
        f = _fields(line)
        kind = f.get("kind") or ""
        if kind not in KINDS:
            continue
        fare_s = f.get("fare") or "unknown"
        fare = int(fare_s) if fare_s.isdigit() and fare_s in raw else None
        if fare_s not in ("unknown", "") and fare is None:
            continue
        route = f.get("route") or "none"
        if route != "none" and route not in raw:
            continue
        door_s = f.get("door") or "unknown"
        door = True if door_s == "yes" else False if door_s == "no" else None
        steps = f.get("steps") or ""
        pay = f.get("pay") or ""
        if not _numbers_in(steps, raw):
            steps = ""
        if pay not in ("", "unknown") and not _numbers_in(pay, raw):
            pay = ""
        title = KINDS[kind] + (f" {route}" if route != "none" else "")
        safety = ""
        if door is False and elder and night:
            safety = "An onward ride is needed at that hour."
        kept.append({
            "kind": kind, "title": title, "fare": fare, "door": door, "route": None if route == "none" else route,
            "pay": pay or None, "steps": steps or None, "safety": safety or None,
            "door_line": {True: "Door to door", False: "Not door to door"}.get(door, "Whether it is door to door was not in the look-up"),
        })
    _mark(kept, elder=elder, night=night)
    return kept


def _mark(options: list[dict], *, elder: bool, night: bool) -> None:
    priced = [o for o in options if o.get("fare") is not None]
    if elder and night:
        doors = [o for o in priced if o.get("door") is True]
        pool, why = (doors, "lowest door-to-door fare at that hour") if doors else (priced, "lowest fare found; not door to door")
    else:
        pool, why = (priced or options), ("lowest fare found" if priced else "the one the look-up could describe")
    if not pool:
        return
    best = min(pool, key=lambda o: o["fare"] if o.get("fare") is not None else 10 ** 9)
    best["recommended"] = True
    best["why"] = why


def messages(options: list[dict]) -> list[str]:
    """One short message per option, then the choice. The brain sends each in their language and adds nothing."""
    out = []
    for i, o in enumerate(options, start=1):
        bit = f"{i}) {o['title']}"
        if o.get("fare") is not None:
            bit += f", about ₹{o['fare']}"
        bit += f". {o['door_line']}."
        if o.get("safety"):
            bit += f" {o['safety']}"
        if o.get("pay"):
            bit += f" Pay: {o['pay']}."
        if o.get("steps"):
            bit += f" {o['steps']}"
        if o.get("recommended"):
            bit += f" I'd take this one: {o['why']}."
        out.append(bit)
    if out:
        out.append("Which feels right?")
    return out


async def fetch(origin: str, destination: str, at: datetime | None) -> tuple[str, list[dict]]:
    """The page text of one live look-up, and its sources. Raises when search is off or fails."""
    from app.brain import websearch

    when = at.strftime("%H:%M") if at else ""
    query = f"{origin} to {destination} {when} airport prepaid taxi bus metro fare route".strip()
    if websearch.provider() == "tavily":
        rows = await websearch.tavily_search(query, max_results=5)
        raw = "\n".join(f"{r.get('title') or ''}\n{r.get('content') or ''}" for r in rows)
        sources = [{"title": (r.get("title") or "")[:80], "url": r["url"]} for r in rows if r.get("url")]
        if not raw.strip():
            raise RuntimeError("no web results")
        return raw, sources[:4]
    if websearch.provider() in ("off", "none", "unavailable"):
        raise RuntimeError("web search is off")
    # Grounded search does not return the page text, so a fare in its answer cannot be checked.
    raise RuntimeError("the look-up has no page text to check")


async def judge(raw: str) -> str:
    """OPTION lines proposed from the page text. Tests replace this; the check still drops a number the page does not have."""
    from app.llm import router

    system = (
        "From the web page text, write one line per real way to make the trip. Each line exactly: "
        "OPTION kind=<airport_taxi|airport_bus|metro|city_bus|auto|own_app> fare=<digits only, or unknown> "
        "door=<yes|no|unknown> route=<the route or line id as written, or none> pay=<how they pay, or unknown> "
        "steps=<one short sentence they can follow>. A fare or a route that is not written in the page text is "
        "forbidden: use unknown or none. No other text."
    )
    reply = await router.complete("classify", system_stable=system, effort="low", max_tokens=600,
                                  messages=[{"role": "user", "content": [{"type": "text", "text": raw[:6000]}]}])
    return (reply.text or "").strip()
