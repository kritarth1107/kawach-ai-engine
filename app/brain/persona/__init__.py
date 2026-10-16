"""Saheli's standing instructions. Identical for every family so it stays in the prompt cache.

core.md is sent on every turn. One section file is sent only with that area's tools.
"""

from pathlib import Path

from app.care import restricted

_DIR = Path(__file__).parent


def _text(name: str) -> str:
    return (_DIR / name).read_text()


# The long standing text stays for tests and for anything that still reads the full instructions.
PERSONA = _text("standing.md") + "\n" + restricted.persona_block() + "\n"

# Sent on every turn. The domain sections are sent only with their tool group.
PERSONA_CORE = _text("core.md")

_BLOCKS = {
    "order": _text("order.md"),
    "nutrition": _text("nutrition.md"),
    "setup": _text("setup.md"),
    "health": _text("health.md"),
    "photos": _text("photos.md"),
    "memory": _text("memory.md"),
}


def for_groups(groups: set[str] | frozenset[str]) -> str:
    """Core rules always. A domain section only when that tool group is on this call."""
    parts = [PERSONA_CORE]
    for name in ("order", "nutrition", "setup", "health", "photos", "memory"):
        if name in groups:
            parts.append(_BLOCKS[name])
    parts.append(restricted.persona_block())
    return "\n\n".join(parts)


REPLY_FORMAT = _text("reply.md")
