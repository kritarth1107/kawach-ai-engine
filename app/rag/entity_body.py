"""Instinct-style entity body_md rendering."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any


def render_entity_frontmatter(
    *,
    slug: str,
    kind: str,
    title: str,
    aliases: list[str],
    status: str = "active",
    owner: str | None = None,
    review_by: date | None = None,
    sources: list[uuid.UUID] | None = None,
) -> str:
    alias_line = ", ".join(aliases[:24]) if aliases else title
    lines = [
        "---",
        f"id: {slug}",
        f"kind: {kind}",
        f"aliases: [{alias_line}]",
        f"status: {status}",
    ]
    if owner:
        lines.append(f"owner: {owner}")
    if review_by:
        lines.append(f"review_by: {review_by.isoformat()}")
    if sources:
        lines.append(f"sources: [{', '.join(str(s) for s in sources[:20])}]")
    lines.append("---")
    return "\n".join(lines)


def render_entity_body(
    *,
    slug: str,
    kind: str,
    title: str,
    aliases: list[str] | None = None,
    status: str = "active",
    owner: str | None = None,
    review_by: date | None = None,
    sources: list[uuid.UUID] | None = None,
    bullets: list[str] | None = None,
    links: list[tuple[str, str]] | None = None,
) -> str:
    front = render_entity_frontmatter(
        slug=slug,
        kind=kind,
        title=title,
        aliases=aliases or [title],
        status=status,
        owner=owner,
        review_by=review_by,
        sources=sources,
    )
    body_lines = list(bullets or [])
    if links:
        related = ", ".join(f"[[{target}]]" for target, _rel in links)
        if related:
            body_lines.append(f"- Related: {related}")
    if not body_lines:
        body_lines.append(f"- **{title}:** No details recorded yet.")
    return front + "\n" + "\n".join(body_lines)


def propose_slug(kind: str, topic: str, title: str | None = None) -> str:
    base = (topic or title or "general").lower().strip()
    base = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in base)
    base = "-".join(part for part in base.split("-") if part)[:48] or "general"
    prefix = {
        "medication": "med",
        "person": "person",
        "condition": "cond",
        "symptom": "symptom",
        "preference": "pref",
        "procedure": "proc",
        "visit": "visit",
        "episode": "episode",
        "organisation": "org",
    }.get(kind, kind[:4])
    return f"{prefix}-{base}"[:120]


def parse_wikilinks(body_md: str) -> list[str]:
    import re

    return re.findall(r"\[\[([^\]]+)\]\]", body_md or "")
