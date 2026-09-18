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


class ResolvePartnerArgs(BaseModel):
    message: str = Field(description="Caregiver order message")


class PreviewOrderItem(BaseModel):
    name: str
    quantity: int = Field(default=1, ge=1, le=20)


class PreviewOrderArgs(BaseModel):
    partner: str = Field(description="swiggy or instamart")
    addressId: str = Field(description="Partner address id from list_partner_addresses")
    items: list[PreviewOrderItem] = Field(description="Cart lines with live catalog names")
    notes: str | None = Field(default=None)


class PlaceCodArgs(BaseModel):
    previewId: str = Field(description="Preview id from preview_order")


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

    async def resolve_order_partner(message: str) -> str:
        return await _run("resolve_order_partner", {"message": message})

    async def search_swiggy_food(query: str, addressId: str | None = None) -> str:
        return await _run("search_swiggy_food", {"query": query, "addressId": addressId})

    async def search_instamart(query: str, addressId: str | None = None) -> str:
        return await _run("search_instamart", {"query": query, "addressId": addressId})

    async def preview_order(
        partner: str,
        addressId: str,
        items: list[PreviewOrderItem],
        notes: str | None = None,
    ) -> str:
        return await _run(
            "preview_order",
            {
                "partner": partner,
                "addressId": addressId,
                "items": [item.model_dump() for item in items],
                "notes": notes,
            },
        )

    async def place_cod_order(previewId: str) -> str:
        return await _run("place_cod_order", {"previewId": previewId})

    async def recall_memories(limit: int = 10) -> str:
        return await _run("recall_memories", {"limit": limit})

    return [
        StructuredTool.from_function(coroutine=get_care_timeline, name="get_care_timeline", description="Fetch care timeline.", args_schema=LimitArgs),
        StructuredTool.from_function(coroutine=search_lab_reports, name="search_lab_reports", description="Search saved lab reports.", args_schema=QueryArgs),
        StructuredTool.from_function(coroutine=get_lab_value, name="get_lab_value", description="Get specific lab value with date.", args_schema=LabNameArgs),
        StructuredTool.from_function(coroutine=get_elder_messages, name="get_elder_messages", description="Recent elder Saheli messages.", args_schema=LimitArgs),
        StructuredTool.from_function(coroutine=resolve_order_partner, name="resolve_order_partner", description="Pick Swiggy Food vs Instamart from caregiver message.", args_schema=ResolvePartnerArgs),
        StructuredTool.from_function(coroutine=list_partner_addresses, name="list_partner_addresses", description="List saved delivery addresses for swiggy or instamart.", args_schema=PartnerOnlyArgs),
        StructuredTool.from_function(coroutine=search_swiggy_food, name="search_swiggy_food", description="Search Swiggy Food dishes/restaurants. Requires addressId.", args_schema=SwiggySearchArgs),
        StructuredTool.from_function(coroutine=search_instamart, name="search_instamart", description="Search Instamart grocery products. Requires addressId.", args_schema=InstamartSearchArgs),
        StructuredTool.from_function(coroutine=preview_order, name="preview_order", description="Build COD order preview with live MCP prices.", args_schema=PreviewOrderArgs),
        StructuredTool.from_function(coroutine=place_cod_order, name="place_cod_order", description="Place COD order after caregiver confirms preview card.", args_schema=PlaceCodArgs),
        StructuredTool.from_function(coroutine=recall_memories, name="recall_memories", description="Recall elder family memories.", args_schema=LimitArgs),
    ]
