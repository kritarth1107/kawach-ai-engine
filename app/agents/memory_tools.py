"""Read-only Instinct memory tools for the elder WhatsApp agent."""

from __future__ import annotations

import json
import uuid

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.db.session import SessionLocal
from app.rag.profile_render import load_memory_profile_text
from app.rag.retrieve import load_entity_body, memory_grep


class GrepArgs(BaseModel):
    query: str = Field(description="Search query for people, medicines, or history")


class SlugArgs(BaseModel):
    slug: str = Field(description="Entity slug from memory_grep results")


class EmptyArgs(BaseModel):
    pass


def build_memory_read_tools(family_id: str, elder_id: str) -> list[StructuredTool]:
    fid = uuid.UUID(family_id)
    eid = uuid.UUID(elder_id)

    async def memory_grep_tool(query: str) -> str:
        async with SessionLocal() as session:
            hits = await memory_grep(session, family_id=fid, elder_id=eid, query=query, limit=5)
            return json.dumps(
                [{"slug": h.slug, "kind": h.kind, "title": h.title, "snippet": h.snippet[:280]} for h in hits]
            )

    async def memory_read_entity(slug: str) -> str:
        async with SessionLocal() as session:
            entity = await load_entity_body(session, elder_id=eid, slug=slug)
            if not entity:
                return json.dumps({"error": "Entity not found"})
            return json.dumps({"slug": entity.slug, "kind": entity.kind, "body_md": entity.body_md[:4000]})

    async def memory_list_index() -> str:
        async with SessionLocal() as session:
            profile = await load_memory_profile_text(session, family_id=fid, elder_id=eid)
            index_part = profile.split("## Entity index")
            return index_part[-1][:2500] if len(index_part) > 1 else profile[:2500]

    return [
        StructuredTool.from_function(
            coroutine=memory_grep_tool,
            name="memory_grep",
            description="Search Saheli's saved memory about people, medicines, preferences, and history.",
            args_schema=GrepArgs,
        ),
        StructuredTool.from_function(
            coroutine=memory_read_entity,
            name="memory_read_entity",
            description="Read the full memory file for one entity slug from memory_grep.",
            args_schema=SlugArgs,
        ),
        StructuredTool.from_function(
            coroutine=memory_list_index,
            name="memory_list_index",
            description="List what Saheli knows — entity index from the memory profile.",
            args_schema=EmptyArgs,
        ),
    ]
