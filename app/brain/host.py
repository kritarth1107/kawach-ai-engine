"""Where the brain's side effects go.

LiveHost calls the backend executor. ShadowHost lets reads through and fakes writes, so the
new brain can run beside the old one on real traffic without doing anything twice.
SimHost (app/sim) plays the whole outside world for simulated families.
"""

from __future__ import annotations

from typing import Protocol

from app.agents.tool_client import execute_backend_tool

# Backend tools that change something in the world or in the family's records.
WRITE_TOOLS = {
    "record_review",
    "connector_place",
    "set_voice_preference",
    "sync_medicine_schedule",
    "mark_schedule_completed",
    "mark_schedule_missed",
    "log_vitals",
    "log_dose",
    "log_check_in",
    "log_symptom",
    "log_appointment_notes",
    "create_reminder",
    "cancel_reminder",
    "notify_caregivers",
    "trigger_emergency_escalation",
    "save_memory",
    "book_ride",
    "cancel_ride",
    "browser_order",
    "browse_and_shop",
    "quick_order",
    "place_cod_order",
    "confirm_and_place_order",
    "claim_schedule_rows",
    "send_whatsapp",
}


class ToolHost(Protocol):
    shadow: bool

    async def call(self, tool: str, args: dict, *, family_id: str, subject_id: str, actor_id: str) -> dict: ...


class LiveHost:
    shadow = False

    async def call(self, tool: str, args: dict, *, family_id: str, subject_id: str, actor_id: str) -> dict:
        return await execute_backend_tool(
            tool=tool, args=args, family_id=family_id, elder_id=subject_id, actor_user_id=actor_id
        )


class ShadowHost:
    shadow = True

    def __init__(self) -> None:
        self.would_have: list[dict] = []

    async def call(self, tool: str, args: dict, *, family_id: str, subject_id: str, actor_id: str) -> dict:
        if tool in WRITE_TOOLS:
            self.would_have.append({"tool": tool, "args": args})
            return {"ok": True, "shadow": True, "note": "not executed (shadow mode)"}
        return await execute_backend_tool(
            tool=tool, args=args, family_id=family_id, elder_id=subject_id, actor_user_id=actor_id
        )
