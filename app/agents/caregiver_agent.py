"""Tool-using caregiver Saheli agent."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompts import CAREGIVER_SAHELI_SYSTEM, ELDER_WHATSAPP_AGENT_SYSTEM
from app.agents.tools import build_caregiver_tools, build_elder_whatsapp_tools
from app.llm.provider import chat_invoke_messages, get_caregiver_chat_llm
from app.models.entities import Elder
from app.rag.retrieve import format_family_memories, retrieve_family_memories

_CASUAL_OFFER = re.compile(
    r"\banything you (want|need) to know\b|\bdo you need (any|some)? (info|information|help)\b",
    re.I,
)
_ORDER_HINT = re.compile(r"\b(order|swiggy|instamart|zepto)\b", re.I)
_GENERIC_ORDER_REPLY = re.compile(
    r"tell me what to order|what would you like to order|what do you want to order|from swiggy, instamart",
    re.I,
)
_ORDER_TOOL_NAMES = {
    "resolve_order_partner",
    "ensure_order_session",
    "search_catalog",
    "add_to_order_cart",
    "get_order_cart",
    "submit_order_cart",
    "select_order_address",
    "list_partner_addresses",
    "resolve_catalog_item",
}


def _sanitize_elder_reply_without_tools(
    message: str,
    reply: str,
    tool_trace: list[dict],
) -> str:
    """Strip hallucinated order replies when no ordering tools ran."""
    tool_names = {row.get("tool") for row in tool_trace if row.get("status") == "done"}
    had_order_tools = bool(tool_names & _ORDER_TOOL_NAMES)
    if _CASUAL_OFFER.search(message) and _ORDER_HINT.search(reply):
        return (
            "That's sweet of you to ask! I don't need anything right now — "
            "tell me how you're doing or share any news."
        )
    if not had_order_tools and _ORDER_HINT.search(message):
        if _GENERIC_ORDER_REPLY.search(reply) or "₹" in reply:
            return ""
    return reply


def _parse_tool_result(raw: object) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _tool_inner(result: dict) -> dict:
    inner = result.get("result")
    return inner if isinstance(inner, dict) else result


def _stringify_ai_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if block.get("type") == "text" and isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


ORDER_AGENT_PLAYBOOK = """
You handle food and grocery ordering conversationally for the caregiver.

Ordering playbook:
1. Call resolve_order_partner when the caregiver wants to order.
2. Swiggy Food = restaurant meals. Instamart = groceries/products — never call Instamart a restaurant.
3. Call list_partner_addresses for that partner. Pick address (ask if multiple).
4. Call search_swiggy_food or search_instamart with addressId. Quote real prices from results only — never guess ₹50.
5. When cart is ready, call preview_order — tell caregiver to confirm on the card.
6. Only call place_cod_order after they explicitly confirm on the card (not from chat text alone).

If partner not connected, explain they must connect Swiggy Food or Instamart separately in Integrations.
"""


def _extract_tool_payloads(
    tool_results: list[dict],
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    order_payload = None
    connect_payload = None
    order_preview = None
    order_flow = None
    for row in tool_results:
        if not isinstance(row, dict):
            continue
        result = _parse_tool_result(row.get("result"))
        if not result:
            continue
        inner = _tool_inner(result)
        kind = inner.get("kind") or result.get("status")
        if kind == "order_flow" or result.get("status") in ("order_flow", "added", "disambiguation_required"):
            order_flow = inner.get("orderFlow") or inner.get("order_flow") or inner
        elif kind == "order_preview" or inner.get("previewId"):
            order_preview = inner
        elif kind == "order_placed" and inner.get("orderId"):
            order_payload = inner
        elif kind == "order" and inner.get("orderId"):
            order_payload = inner
        elif kind in ("connect_required", "connect") or result.get("status") == "connect_required":
            connect_payload = inner if inner.get("connectPartner") else result
    return order_payload, connect_payload, order_preview, order_flow


def _friendly_tool_error(raw: object) -> str | None:
    if not isinstance(raw, str):
        raw = str(raw)
    lower = raw.lower()
    if "timeout" in lower or "timed out" in lower:
        return "That search timed out — please try again in a minute."
    if "not connected" in lower:
        return "That delivery partner isn't connected yet — your caregiver can link it in Integrations."
    if "session expired" in lower or ("timed out" in lower and "basket" in lower):
        return "That order basket timed out — tell me again what you'd like to order and I'll start fresh."
    if raw.strip():
        return f"Sorry — {raw.strip()}"
    return None


def _prompt_message_from_tools(tool_results: list[dict]) -> str | None:
    for row in reversed(tool_results):
        if not isinstance(row, dict):
            continue
        result = _parse_tool_result(row.get("result"))
        if not result:
            continue
        inner = _tool_inner(result)
        err = inner.get("error") or result.get("error")
        if err:
            msg = inner.get("message") or result.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            friendly = _friendly_tool_error(err)
            if friendly:
                return friendly
        if result.get("status") == "session_expired":
            msg = inner.get("message") or result.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        if inner.get("kind") == "prompt" and isinstance(inner.get("message"), str):
            return inner["message"].strip()
        if result.get("status") == "prompt" and isinstance(inner.get("message"), str):
            return inner["message"].strip()
        if inner.get("kind") == "order_flow" or result.get("status") == "order_flow":
            msg = inner.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        if result.get("status") == "disambiguation_required":
            msg = inner.get("message") or result.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            query = inner.get("query") or result.get("query")
            candidates = inner.get("candidates") or result.get("candidates")
            if isinstance(candidates, list) and candidates:
                lines = [f'Which "{query}" did you mean?']
                for i, row in enumerate(candidates[:5], start=1):
                    if isinstance(row, dict):
                        name = row.get("name", "Option")
                        price = row.get("pricePaise")
                        price_txt = f" ₹{int(price) // 100}" if isinstance(price, (int, float)) and price else ""
                        lines.append(f"{i}. {name}{price_txt}")
                return "\n".join(lines)
        if result.get("status") == "added" and isinstance(inner.get("message"), str):
            return inner["message"].strip()
    return None


async def run_elder_whatsapp_agent(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    elder_id: uuid.UUID,
    message: str,
    care_record_context: str | None = None,
    schedule_context: str | None = None,
    channel_context: str | None = None,
    order_context: str | None = None,
    companion_profile: dict[str, Any] | None = None,
    history_messages: list | None = None,
    actor_user_id: str,
    kavach_family_id: str | None = None,
    kavach_recipient_user_id: str | None = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    profile = companion_profile or {}
    child_name = profile.get("child_name") or profile.get("childName") or "Saheli"
    lang = profile.get("preferred_language") or profile.get("preferredLanguage") or "english"

    platform_block = ""
    if schedule_context:
        platform_block += f"\n- Today's schedule:\n{schedule_context[:2500]}"
    if care_record_context:
        platform_block += f"\n- Care timeline:\n{care_record_context[:2500]}"
    if order_context:
        platform_block += f"\n- Ordering note:\n{order_context[:800]}"
    if channel_context:
        platform_block += f"\n- Channel rules:\n{channel_context[:1200]}"

    system = f"""{ELDER_WHATSAPP_AGENT_SYSTEM}

You are {child_name}. Language preference: {lang}.
{platform_block}

Ordering playbook:
1. resolve_order_partner FIRST — speak its message field verbatim when partner unavailable
2. list_partner_addresses → ensure_order_session(message) → keep sessionId
3. select_order_address if needed
4. add_to_order_cart with all items in one batch call
5. If disambiguation_required, ask elder to pick 1/2/3 then resolve_catalog_item or add with candidateIndex
6. get_order_cart → elder confirms → submit_order_cart
7. get_order_status when elder asks where their order is
8. log_vitals for BP/sugar; save_memory for family news worth remembering
Never re-ask what to order when item + partner are already stated.
If a tool returns session_expired, call ensure_order_session again with the full order request — never reuse old sessionIds.
Never order for check-ins or "anything you want to know?".
"""

    platform_family_id = kavach_family_id or str(family_id)
    platform_recipient_id = kavach_recipient_user_id or str(elder_id)
    tools = build_elder_whatsapp_tools(platform_family_id, platform_recipient_id, actor_user_id)
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
            reply = _stringify_ai_content(response.content)
            prompt_message = _prompt_message_from_tools(tool_results_raw)
            if prompt_message:
                reply = prompt_message
            reply = _sanitize_elder_reply_without_tools(message, reply, tool_trace)
            order_payload, connect_payload, order_preview, order_flow = _extract_tool_payloads(
                tool_results_raw,
            )
            return {
                "reply": reply.strip(),
                "order": order_payload,
                "connect": connect_payload,
                "order_preview": order_preview,
                "order_flow": order_flow,
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
                    parsed = (
                        json.loads(result_str)
                        if isinstance(result_str, str) and result_str.startswith("{")
                        else result_str
                    )
                    tool_results_raw.append({"tool": tool_name, "result": parsed})
                except Exception as exc:
                    parsed = {"error": str(exc)}
                    result_str = json.dumps(parsed)
                    tool_results_raw.append({"tool": tool_name, "result": parsed})
            tool_trace.append({"tool": tool_name, "status": "done"})
            messages.append(
                ToolMessage(content=result_str, tool_call_id=call["id"]),
            )

    fallback = await chat_invoke_messages(messages)
    reply = _stringify_ai_content(fallback.content)
    prompt_message = _prompt_message_from_tools(tool_results_raw)
    if prompt_message:
        reply = prompt_message
    reply = _sanitize_elder_reply_without_tools(message, reply, tool_trace)
    order_payload, connect_payload, order_preview, order_flow = _extract_tool_payloads(
        tool_results_raw,
    )
    return {
        "reply": reply.strip(),
        "order": order_payload,
        "connect": connect_payload,
        "order_preview": order_preview,
        "order_flow": order_flow,
        "tool_trace": tool_trace,
    }


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

    memories = await retrieve_family_memories(
        session,
        family_id=family_id,
        elder_id=elder_id,
        query=message,
        limit=12,
    )
    memory_block = format_family_memories(memories)

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
- Long-term memories about {elder_name}:
{memory_block}
{platform_block}

{ORDER_AGENT_PLAYBOOK}
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
            reply = _stringify_ai_content(response.content)
            prompt_message = _prompt_message_from_tools(tool_results_raw)
            if prompt_message:
                reply = prompt_message
            order_payload, connect_payload, order_preview, _order_flow = _extract_tool_payloads(
                tool_results_raw,
            )
            return {
                "reply": reply.strip(),
                "order": order_payload,
                "connect": connect_payload,
                "order_preview": order_preview,
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
    reply = _stringify_ai_content(fallback.content)
    prompt_message = _prompt_message_from_tools(tool_results_raw)
    if prompt_message:
        reply = prompt_message
    order_payload, connect_payload, order_preview, _order_flow = _extract_tool_payloads(
        tool_results_raw,
    )
    return {
        "reply": reply.strip(),
        "order": order_payload,
        "connect": connect_payload,
        "order_preview": order_preview,
        "tool_trace": tool_trace,
    }


async def stream_caregiver_agent(session: AsyncSession, **kwargs):
    """Run caregiver agent; emit tool events and stream final LLM tokens via astream."""
    elder_id = kwargs["elder_id"]
    family_id = kwargs["family_id"]
    message = kwargs["message"]
    actor_user_id = kwargs["actor_user_id"]
    kavach_family_id = kwargs.get("kavach_family_id")
    kavach_recipient_user_id = kwargs.get("kavach_recipient_user_id")
    care_record_context = kwargs.get("care_record_context")
    elder_thread_context = kwargs.get("elder_thread_context")
    labs_context = kwargs.get("labs_context")
    session_context = kwargs.get("session_context")
    order_context = kwargs.get("order_context")
    history_messages = kwargs.get("history_messages")
    max_iterations = kwargs.get("max_iterations", 5)

    elder = await session.get(Elder, elder_id)
    elder_name = elder.display_name if elder else "Care recipient"
    memories = await retrieve_family_memories(
        session, family_id=family_id, elder_id=elder_id, query=message, limit=12,
    )
    memory_block = format_family_memories(memories)

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
- Long-term memories about {elder_name}:
{memory_block}
{platform_block}

{ORDER_AGENT_PLAYBOOK}
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

    tool_results_raw: list[dict] = []
    reply = ""

    for _ in range(max_iterations):
        response: AIMessage | None = None
        if hasattr(llm, "astream"):
            accumulated = ""
            async for chunk in llm.astream(messages):
                if getattr(chunk, "tool_calls", None):
                    response = chunk if isinstance(chunk, AIMessage) else AIMessage(content=accumulated, tool_calls=chunk.tool_calls)
                    break
                delta = _stringify_ai_content(getattr(chunk, "content", chunk))
                if delta:
                    accumulated += delta
                    yield {"type": "token", "delta": delta}
            if response is None and accumulated:
                response = AIMessage(content=accumulated)
        else:
            response = await llm.ainvoke(messages)

        if not response or not getattr(response, "tool_calls", None):
            reply = _stringify_ai_content(response.content if response else "")
            prompt_message = _prompt_message_from_tools(tool_results_raw)
            if prompt_message:
                reply = prompt_message
            break

        messages.append(response)
        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args") or {}
            yield {"type": "tool_start", "id": tool_name, "name": tool_name}
            selected = next((t for t in tools if t.name == tool_name), None)
            if not selected:
                result_str = json.dumps({"error": f"Unknown tool {tool_name}"})
            else:
                try:
                    result = await selected.ainvoke(tool_args)
                    result_str = result if isinstance(result, str) else json.dumps(result)
                    tool_results_raw.append(
                        {
                            "tool": tool_name,
                            "result": json.loads(result_str) if result_str.startswith("{") else result_str,
                        }
                    )
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)})
            yield {"type": "tool_result", "id": tool_name, "name": tool_name}
            messages.append(ToolMessage(content=result_str, tool_call_id=call["id"]))

    if not reply:
        fallback = await chat_invoke_messages(messages)
        reply = _stringify_ai_content(fallback.content)
        prompt_message = _prompt_message_from_tools(tool_results_raw)
        if prompt_message:
            reply = prompt_message

    order_payload, connect_payload, order_preview, _order_flow = _extract_tool_payloads(
        tool_results_raw,
    )
    if order_payload:
        yield {"type": "tool_result", "id": "order", "order": order_payload}
    if connect_payload:
        yield {"type": "tool_result", "id": "connect", "connect": connect_payload}
    if order_preview:
        yield {"type": "tool_result", "id": "order_preview", "order_preview": order_preview}
    yield {
        "type": "done",
        "reply": reply.strip(),
        "order": order_payload,
        "connect": connect_payload,
        "order_preview": order_preview,
    }
