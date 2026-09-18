"""Saheli caregiver agent tools — executed on kavach-backend."""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agents.tool_client import execute_backend_tool


class QueryArgs(BaseModel):
    query: str = Field(description="Search query")


class LabNameArgs(BaseModel):
    name: str = Field(description="Lab test name e.g. TSH, HbA1c")


class OrderArgs(BaseModel):
    message: str = Field(description="Full order request from the caregiver")


class SwiggySearchArgs(BaseModel):
    query: str = Field(description="Dish or restaurant search")
    addressId: str | None = Field(default=None)


class InstamartSearchArgs(BaseModel):
    query: str = Field(description="Product search")
    addressId: str | None = Field(default=None)


class LimitArgs(BaseModel):
    limit: int = Field(default=10)


class PartnerOnlyArgs(BaseModel):
    partner: str = Field(default="swiggy")


def build_caregiver_tools(
    family_id: str,
    elder_id: str,
    actor_user_id: str,
) -> list[StructuredTool]:
    async def _run(tool: str, args: dict) -> str:
        result = await execute_backend_tool(
            tool=tool,
            args=args,
            family_id=family_id,
            elder_id=elder_id,
            actor_user_id=actor_user_id,
        )
        return json.dumps(result, default=str)

    async def get_care_timeline(limit: int = 40) -> str:
        return await _run("get_care_timeline", {"limit": limit})

    async def search_lab_reports(query: str) -> str:
        return await _run("search_lab_reports", {"query": query})

    async def get_lab_value(name: str) -> str:
        return await _run("get_lab_value", {"name": name})

    async def get_elder_messages(limit: int = 12) -> str:
        return await _run("get_elder_messages", {"limit": limit})

    async def list_partner_addresses(partner: str = "swiggy") -> str:
        return await _run("list_partner_addresses", {"partner": partner})

    async def search_swiggy_food(query: str, addressId: str | None = None) -> str:
        return await _run("search_swiggy_food", {"query": query, "addressId": addressId})

    async def search_instamart(query: str, addressId: str | None = None) -> str:
        return await _run("search_instamart", {"query": query, "addressId": addressId})

    async def suggest_order(message: str) -> str:
        return await _run("suggest_order", {"message": message})

    async def recall_memories(limit: int = 10) -> str:
        return await _run("recall_memories", {"limit": limit})

    return [
        StructuredTool.from_function(coroutine=get_care_timeline, name="get_care_timeline", description="Fetch care timeline.", args_schema=LimitArgs),
        StructuredTool.from_function(coroutine=search_lab_reports, name="search_lab_reports", description="Search saved lab reports.", args_schema=QueryArgs),
        StructuredTool.from_function(coroutine=get_lab_value, name="get_lab_value", description="Get specific lab value with date.", args_schema=LabNameArgs),
        StructuredTool.from_function(coroutine=get_elder_messages, name="get_elder_messages", description="Recent elder Saheli messages.", args_schema=LimitArgs),
        StructuredTool.from_function(coroutine=list_partner_addresses, name="list_partner_addresses", description="List Swiggy/Instamart addresses.", args_schema=PartnerOnlyArgs),
        StructuredTool.from_function(coroutine=search_swiggy_food, name="search_swiggy_food", description="Search Swiggy food catalog.", args_schema=SwiggySearchArgs),
        StructuredTool.from_function(coroutine=search_instamart, name="search_instamart", description="Search Instamart products.", args_schema=InstamartSearchArgs),
        StructuredTool.from_function(coroutine=suggest_order, name="suggest_order", description="Create order basket for approval.", args_schema=OrderArgs),
        StructuredTool.from_function(coroutine=recall_memories, name="recall_memories", description="Recall elder family memories.", args_schema=LimitArgs),
    ]
