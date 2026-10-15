"""Who a fact is about. A name resolves only when one member has it. Anyone else is not the care recipient."""

from __future__ import annotations

import unicodedata


class UnknownMember(Exception):
    def __init__(self, about: str, matches: list[str]):
        self.about = about
        self.matches = matches
        super().__init__(about)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().casefold()


def _people(elder: dict, members: list[dict]) -> list[dict]:
    out, seen = [], set()
    for person in [elder, *members]:
        pid = person.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            out.append(person)
    return out


def resolve_member(about: str | None, *, elder: dict, members: list[dict]) -> str:
    """The member id. Blank means the care recipient. A name is used only when one member has it."""
    people = _people(elder, members)
    ids = {p["id"] for p in people}
    raw = (about or "").strip()
    if not raw:
        return elder["id"]
    if raw in ids:
        return raw
    key = _fold(raw)
    full = [p for p in people if _fold(p.get("name") or "") == key]
    if len(full) == 1:
        return full[0]["id"]
    if len(full) > 1:
        raise UnknownMember(raw, [p["id"] for p in full])
    first = [p for p in people if (p.get("name") or "").split()[:1] and _fold((p.get("name") or "").split()[0]) == key]
    if len(first) == 1:
        return first[0]["id"]
    raise UnknownMember(raw, [p["id"] for p in first])
