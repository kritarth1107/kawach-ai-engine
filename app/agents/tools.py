"""Saheli agent tools — executed on kavach-backend."""

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
    message: str = Field(description="Full order request from the user")


class SessionQueryArgs(BaseModel):
    sessionId: str = Field(description="Active order session id from ensure_order_session")
    query: str = Field(description="Product or dish search query")


class SessionOnlyArgs(BaseModel):
    sessionId: str = Field(description="Active order session id")


class ResolveCatalogArgs(BaseModel):
    sessionId: str
    query: str | None = Field(default=None)
    candidateIndex: int | None = Field(default=None, description="0-based index from search_catalog results")
    candidateId: str | None = Field(default=None)


class CartLineItem(BaseModel):
    query: str | None = Field(default=None, description="Product name to resolve and add")
    candidateIndex: int | None = Field(default=None)
    candidateId: str | None = Field(default=None)
    quantity: int = Field(default=1, ge=1, le=20)


class AddToCartArgs(BaseModel):
    sessionId: str
    items: list[CartLineItem] = Field(description="One or more items to add in a single call")


class SelectAddressArgs(BaseModel):
    sessionId: str
    addressId: str = Field(description="Partner address id from list_partner_addresses")


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
    message: str = Field(description="Order message")


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


class ScheduleTitleArgs(BaseModel):
    title: str | None = Field(default=None, description="Schedule item title hint")
    scheduleId: str | None = Field(default=None)
    dateKey: str | None = Field(default=None, description="YYYY-MM-DD in IST")
    note: str | None = Field(default=None)


class LogCheckInArgs(BaseModel):
    mood: str | None = Field(default=None)
    meals: str | None = Field(default=None)
    sleep: str | None = Field(default=None)
    pain: str | None = Field(default=None)
    note: str | None = Field(default=None)


class LogDoseArgs(BaseModel):
    medicineName: str = Field(description="Medicine name")
    quantity: str | None = Field(default=None)
    note: str | None = Field(default=None)


class SaveMemoryArgs(BaseModel):
    content: str = Field(description="Family news or detail worth remembering")


class LogVitalsArgs(BaseModel):
    kind: str = Field(description="Vital type e.g. Blood pressure, Blood sugar")
    value: str = Field(description="Reading e.g. 140/90 or 110")
    unit: str | None = Field(default=None)
    note: str | None = Field(default=None)


class GetOrderStatusArgs(BaseModel):
    orderId: str | None = Field(default=None, description="Optional order id; latest order if omitted")


def _backend_tools(
    family_id: str,
    elder_id: str,
    actor_user_id: str,
):
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

    async def get_family_members() -> str:
        return await _run("get_family_members", {})

    async def suggest_order(message: str) -> str:
        return await _run("suggest_order", {"message": message})

    async def ensure_order_session(message: str) -> str:
        return await _run("ensure_order_session", {"message": message})

    async def search_catalog(sessionId: str, query: str) -> str:
        return await _run("search_catalog", {"sessionId": sessionId, "query": query})

    async def resolve_catalog_item(
        sessionId: str,
        query: str | None = None,
        candidateIndex: int | None = None,
        candidateId: str | None = None,
    ) -> str:
        return await _run(
            "resolve_catalog_item",
            {
                "sessionId": sessionId,
                "query": query,
                "candidateIndex": candidateIndex,
                "candidateId": candidateId,
            },
        )

    async def add_to_order_cart(sessionId: str, items: list[CartLineItem]) -> str:
        return await _run(
            "add_to_order_cart",
            {
                "sessionId": sessionId,
                "items": [item.model_dump() for item in items],
            },
        )

    async def get_order_cart(sessionId: str) -> str:
        return await _run("get_order_cart", {"sessionId": sessionId})

    async def submit_order_cart(sessionId: str) -> str:
        return await _run("submit_order_cart", {"sessionId": sessionId})

    async def select_order_address(sessionId: str, addressId: str) -> str:
        return await _run(
            "select_order_address",
            {"sessionId": sessionId, "addressId": addressId},
        )

    async def get_today_schedule(dateKey: str | None = None) -> str:
        return await _run("get_today_schedule", {"dateKey": dateKey} if dateKey else {})

    async def get_missed_tasks(dateKey: str | None = None) -> str:
        return await _run("get_missed_tasks", {"dateKey": dateKey} if dateKey else {})

    async def log_check_in(
        mood: str | None = None,
        meals: str | None = None,
        sleep: str | None = None,
        pain: str | None = None,
        note: str | None = None,
    ) -> str:
        return await _run(
            "log_check_in",
            {"mood": mood, "meals": meals, "sleep": sleep, "pain": pain, "note": note},
        )

    async def log_dose(
        medicineName: str,
        quantity: str | None = None,
        note: str | None = None,
    ) -> str:
        return await _run(
            "log_dose",
            {"medicineName": medicineName, "quantity": quantity, "note": note},
        )

    async def mark_schedule_completed(
        title: str | None = None,
        scheduleId: str | None = None,
        dateKey: str | None = None,
        note: str | None = None,
    ) -> str:
        return await _run(
            "mark_schedule_completed",
            {"title": title, "scheduleId": scheduleId, "dateKey": dateKey, "note": note},
        )

    async def get_order_status(orderId: str | None = None) -> str:
        args = {"orderId": orderId} if orderId else {}
        return await _run("get_order_status", args)

    async def log_vitals(
        kind: str,
        value: str,
        unit: str | None = None,
        note: str | None = None,
    ) -> str:
        return await _run(
            "log_vitals",
            {"kind": kind, "value": value, "unit": unit, "note": note},
        )

    async def save_memory(content: str) -> str:
        return await _run("save_memory", {"content": content})

    async def quick_order(message: str) -> str:
        return await _run("quick_order", {"message": message})

    async def confirm_and_place_order(sessionId: str) -> str:
        return await _run("confirm_and_place_order", {"sessionId": sessionId})

    return locals()


def build_caregiver_tools(
    family_id: str,
    elder_id: str,
    actor_user_id: str,
) -> list[StructuredTool]:
    tools = _backend_tools(family_id, elder_id, actor_user_id)

    return [
        StructuredTool.from_function(coroutine=tools["get_care_timeline"], name="get_care_timeline", description="Fetch care timeline.", args_schema=LimitArgs),
        StructuredTool.from_function(coroutine=tools["search_lab_reports"], name="search_lab_reports", description="Search saved lab reports.", args_schema=QueryArgs),
        StructuredTool.from_function(coroutine=tools["get_lab_value"], name="get_lab_value", description="Get specific lab value with date.", args_schema=LabNameArgs),
        StructuredTool.from_function(coroutine=tools["get_elder_messages"], name="get_elder_messages", description="Recent elder Saheli messages.", args_schema=LimitArgs),
        StructuredTool.from_function(coroutine=tools["resolve_order_partner"], name="resolve_order_partner", description="Pick Swiggy Food vs Instamart from caregiver message.", args_schema=ResolvePartnerArgs),
        StructuredTool.from_function(coroutine=tools["list_partner_addresses"], name="list_partner_addresses", description="List saved delivery addresses for swiggy or instamart.", args_schema=PartnerOnlyArgs),
        StructuredTool.from_function(coroutine=tools["search_swiggy_food"], name="search_swiggy_food", description="Search Swiggy Food dishes/restaurants. Requires addressId.", args_schema=SwiggySearchArgs),
        StructuredTool.from_function(coroutine=tools["search_instamart"], name="search_instamart", description="Search Instamart grocery products. Requires addressId.", args_schema=InstamartSearchArgs),
        StructuredTool.from_function(coroutine=tools["preview_order"], name="preview_order", description="Build COD order preview with live MCP prices.", args_schema=PreviewOrderArgs),
        StructuredTool.from_function(coroutine=tools["place_cod_order"], name="place_cod_order", description="Place COD order after caregiver confirms preview card.", args_schema=PlaceCodArgs),
        StructuredTool.from_function(coroutine=tools["recall_memories"], name="recall_memories", description="Recall elder family memories.", args_schema=LimitArgs),
        StructuredTool.from_function(
            coroutine=tools["get_family_members"],
            name="get_family_members",
            description="List family members with saved phone numbers, roles, and relationships.",
            args_schema=LimitArgs,
        ),
    ]


def build_elder_whatsapp_tools(
    family_id: str,
    elder_id: str,
    actor_user_id: str,
) -> list[StructuredTool]:
    tools = _backend_tools(family_id, elder_id, actor_user_id)

    return [
        StructuredTool.from_function(
            coroutine=tools["get_today_schedule"],
            name="get_today_schedule",
            description="Today's medicines, meals, and appointments for the elder.",
            args_schema=ScheduleTitleArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["get_missed_tasks"],
            name="get_missed_tasks",
            description="Schedule items still due or missed today.",
            args_schema=ScheduleTitleArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["log_check_in"],
            name="log_check_in",
            description="Log a wellness check-in when the elder shares mood, meals, sleep, or how they feel.",
            args_schema=LogCheckInArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["log_dose"],
            name="log_dose",
            description="Log that the elder took a medicine.",
            args_schema=LogDoseArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["mark_schedule_completed"],
            name="mark_schedule_completed",
            description="Mark a schedule item done (medicine, meal, task).",
            args_schema=ScheduleTitleArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["quick_order"],
            name="quick_order",
            description=(
                "Phase-2 one-shot order: pass the elder's full order message. "
                "Returns confirm card data (items, price, partner, address, sessionId). "
                "Never invent prices — only read tool output."
            ),
            args_schema=OrderArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["confirm_and_place_order"],
            name="confirm_and_place_order",
            description=(
                "After elder says haan/confirm/yes on a quick_order card, place the order "
                "with the sessionId from quick_order."
            ),
            args_schema=SessionOnlyArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["resolve_order_partner"],
            name="resolve_order_partner",
            description="Fallback only: pick Swiggy (food) vs Instamart (groceries) from the elder's message.",
            args_schema=ResolvePartnerArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["list_partner_addresses"],
            name="list_partner_addresses",
            description="List saved delivery addresses. Call after resolve_order_partner.",
            args_schema=PartnerOnlyArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["ensure_order_session"],
            name="ensure_order_session",
            description=(
                "Start or resume an order session when the elder clearly wants to order. "
                "Pass their full request (e.g. 'diet coke from instamart'). Returns sessionId."
            ),
            args_schema=OrderArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["select_order_address"],
            name="select_order_address",
            description="Select delivery address for an order session when multiple addresses exist.",
            args_schema=SelectAddressArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["search_catalog"],
            name="search_catalog",
            description="Search live catalog within an order session. Returns ranked candidates with confidence.",
            args_schema=SessionQueryArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["add_to_order_cart"],
            name="add_to_order_cart",
            description=(
                "Add one or more items to the order cart. Pass query per item; "
                "if ambiguous, ask the elder to pick from disambiguation options. "
                "Supports batch: milk + bread + eggs in one call."
            ),
            args_schema=AddToCartArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["get_order_cart"],
            name="get_order_cart",
            description="Show current cart contents and total before submitting.",
            args_schema=SessionOnlyArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["submit_order_cart"],
            name="submit_order_cart",
            description="Submit the cart for approval/checkout after elder confirms.",
            args_schema=SessionOnlyArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["get_order_status"],
            name="get_order_status",
            description="Latest order status when elder asks where their order is.",
            args_schema=GetOrderStatusArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["log_vitals"],
            name="log_vitals",
            description="Log BP, sugar, or other vitals the elder reports in chat.",
            args_schema=LogVitalsArgs,
        ),
        StructuredTool.from_function(
            coroutine=tools["resolve_catalog_item"],
            name="resolve_catalog_item",
            description="Pick a numbered catalog option after disambiguation (candidateIndex 0-based).",
            args_schema=ResolveCatalogArgs,
        ),
    ]
