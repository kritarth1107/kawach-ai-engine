"""Tool-using caregiver Saheli agent."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompts import CAREGIVER_SAHELI_SYSTEM
from app.agents.tools import build_caregiver_tools
from app.llm.provider import chat_invoke_messages, get_caregiver_chat_llm
from app.models.entities import Elder


def _extract_order_connect(tool_results: list[dict]) -> tuple[dict | None, dict | None]:
    order_payload = None
    connect_payload = None
    for row in tool_results:
        if not isinstance(row, dict):
            continue
        result = row.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                continue
        if not isinstance(result, dict):
            continue
        inner = result.get("result") if isinstance(result.get("result"), dict) else result
        if not isinstance(inner, dict):
            continue
        kind = inner.get("kind") or result.get("status")
        if kind == "order" and inner.get("orderId"):
            order_payload = inner
        elif kind in ("connect_required", "connect") or result.get("status") == "connect_required":
            connect_payload = inner if inner.get("connectPartner") else result
    return order_payload, connect_payload


async def run_caregiver_agent(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    care_record_context: str | None = None,
    elder_thread_context: str | None = None,
    labs_context: str | None = None,
    session_context: str | None = None,
    order_context: str | None = None,
    history_messages: list | None = None,
    actor_user_id: str,
    kavach_family_id: str | None = None,
    kavach_recipient_user_id: str | None = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    elder = await session.get(Elder, elder_id)
    elder_name = elder.display_name if elder else "Care recipient"

    platform_block = ""
    if care_record_context:
        platform_block += f"\n- Kavach care timeline:\n{care_record_context[:4000]}"
    if elder_thread_context:
        platform_block += f"\n- Elder messages:\n{elder_thread_context[:2000]}"
    if labs_context:
        platform_block += f"\n- Saved labs:\n{labs_context[:3000]}"
    if session_context:
        platform_block += f"\n- Session:\n{session_context[:1500]}"
    if order_context:
        platform_block += f"\n- Ordering note:\n{order_context[:1000]}"

    system = f"""{CAREGIVER_SAHELI_SYSTEM}

Care recipient: {elder_name}
{platform_block}

When the caregiver wants food or groceries, call list_partner_addresses first if needed, then search_swiggy_food or search_instamart, then suggest_order with their exact request.
Quote lab values with dates only — never say high/low/normal.
"""

    platform_family_id = kavach_family_id or str(family_id)
    platform_recipient_id = kavach_recipient_user_id or str(elder_id)
    tools = build_caregiver_tools(platform_family_id, platform_recipient_id, actor_user_id)
    llm = get_caregiver_chat_llm().bind_tools(tools)

    messages: list = [SystemMessage(content=system)]
    if history_messages:
        messages.extend(history_messages)
    messages.append(HumanMessage(content=message))

    tool_trace: list[dict] = []
    tool_results_raw: list[dict] = []

    for _ in range(max_iterations):
        response: AIMessage = await llm.ainvoke(messages)
        if not getattr(response, "tool_calls", None):
            reply = response.content if isinstance(response.content, str) else str(response.content)
            order_payload, connect_payload = _extract_order_connect(tool_results_raw)
            return {
                "reply": reply.strip(),
                "order": order_payload,
                "connect": connect_payload,
                "tool_trace": tool_trace,
            }

        messages.append(response)
        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args") or {}
            tool_trace.append({"tool": tool_name, "status": "started"})
            selected = next((t for t in tools if t.name == tool_name), None)
            if not selected:
                result_str = json.dumps({"error": f"Unknown tool {tool_name}"})
            else:
                try:
                    result = await selected.ainvoke(tool_args)
                    result_str = result if isinstance(result, str) else json.dumps(result)
                    tool_results_raw.append({"tool": tool_name, "result": json.loads(result_str) if result_str.startswith("{") else result_str})
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)})
            tool_trace.append({"tool": tool_name, "status": "done"})
            messages.append(
                ToolMessage(content=result_str, tool_call_id=call["id"]),
            )

    fallback = await chat_invoke_messages(messages)
    content = fallback.content if isinstance(fallback.content, str) else str(fallback.content)
    order_payload, connect_payload = _extract_order_connect(tool_results_raw)
    return {
        "reply": content.strip(),
        "order": order_payload,
        "connect": connect_payload,
        "tool_trace": tool_trace,
    }
