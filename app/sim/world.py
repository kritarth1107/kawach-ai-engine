"""A fake outside world for simulated families: schedules, reminders, alerts, orders, rides, records.

SimHost implements the backend tool surface in memory, so the brain runs unchanged and every side
effect is inspectable by graders. Service behaviour (success, stall, OTP, surge, CAPTCHA) is scriptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core import clock


@dataclass
class World:
    schedules: dict[str, dict] = field(default_factory=dict)  # scheduleId -> row
    reminders_sent: list[dict] = field(default_factory=list)  # {dateKey, scheduleId, delivered, at}
    marks: list[dict] = field(default_factory=list)
    vitals: list[dict] = field(default_factory=list)
    symptoms: list[dict] = field(default_factory=list)
    reminders: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)  # proactive WhatsApp messages Saheli started
    emergencies: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    rides: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    # scripted service outcomes, consumed in order: "ok" | "otp" | "captcha" | "stall" | "surge" | "fail"
    service_script: list[str] = field(default_factory=list)
    # what the backend already held for the family before Saheli v2 (export_care_record)
    backend_record: dict = field(default_factory=dict)


class SimHost:
    shadow = False

    def __init__(self, world: World | None = None) -> None:
        self.world = world or World()

    def _next_outcome(self) -> str:
        return self.world.service_script.pop(0) if self.world.service_script else "ok"

    async def call(self, tool: str, args: dict, *, family_id: str, subject_id: str, actor_id: str) -> dict:
        w = self.world
        w.calls.append({"tool": tool, "args": args, "at": clock.now().isoformat()})
        day = clock.ist_day()
        if tool == "sync_medicine_schedule":
            key = args["key"]
            keep = set(args.get("times") or []) if args.get("active", True) else set()
            for sid, row in w.schedules.items():
                if row["sourceKey"] == key:
                    row["active"] = row["time"] in keep
            have = {r["time"] for r in w.schedules.values() if r["sourceKey"] == key}
            for t in keep - have:
                sid = f"sch-{len(w.schedules) + 1}"
                w.schedules[sid] = {"scheduleId": sid, "sourceKey": key, "title": args.get("name"), "dose": args.get("dose"), "time": t, "active": True}
            for row in w.schedules.values():
                if row["sourceKey"] == key and row["active"]:
                    row.update(title=args.get("name"), dose=args.get("dose"))
            return {"active": sorted(r["time"] for r in w.schedules.values() if r["sourceKey"] == key and r["active"])}
        if tool == "get_reminder_log":
            d = args.get("dateKey") or day
            return {
                "dateKey": d,
                "attempts": [r for r in w.reminders_sent if r["dateKey"] == d],
                "scheduled": [{"scheduleId": r["scheduleId"], "item": f"{r['title']} {r.get('dose') or ''} at {r['time']}"} for r in w.schedules.values() if r["active"]],
            }
        if tool in ("mark_schedule_completed", "mark_schedule_missed"):
            w.marks.append({"tool": tool, **args, "day": day})
            return {"ok": True}
        if tool == "log_vitals":
            w.vitals.append({**args, "day": day})
            return {"ok": True}
        if tool in ("log_symptom", "log_check_in", "log_dose", "log_appointment_notes", "save_memory"):
            w.symptoms.append({"tool": tool, **args, "day": day})
            return {"ok": True}
        if tool == "create_reminder":
            w.reminders.append(args)
            return {"ok": True, "reminderId": f"rem-{len(w.reminders)}"}
        if tool == "notify_caregivers":
            w.alerts.append({**args, "at": clock.now().isoformat()})
            return {"notifiedCount": 1}
        if tool == "trigger_emergency_escalation":
            w.emergencies.append({**args, "at": clock.now().isoformat()})
            return {"escalated": True}
        if tool in ("get_today_schedule",):
            sent = {r["scheduleId"] for r in w.reminders_sent if r["dateKey"] == day and r.get("delivered")}
            return {"dateKey": day, "items": [{"title": r["title"], "time": r["time"], "status": "reminded" if r["scheduleId"] in sent else "upcoming"} for r in w.schedules.values() if r["active"]]}
        if tool in ("search_lab_reports",):
            q = (args.get("query") or "").lower()
            return {"labs": [r for r in w.records if q in str(r).lower()][:5]}
        if tool == "get_lab_trends":
            return {"points": []}
        if tool in ("browser_order", "browse_and_shop", "quick_order"):
            outcome = self._next_outcome()
            order = {"goal": args.get("goal") or args.get("message"), "outcome": outcome, "at": clock.now().isoformat()}
            w.orders.append(order)
            if outcome == "ok":
                return {"status": "placed", "orderId": f"ord-{len(w.orders)}", "payment": "COD", "eta": "25 min"}
            if outcome == "otp":
                return {"status": "needs_otp", "message": "The store sent an OTP to the family phone."}
            return {"status": "failed", "reason": outcome}
        if tool == "book_ride":
            outcome = self._next_outcome()
            w.rides.append({**args, "outcome": outcome})
            if outcome == "ok":
                return {"status": "searching", "rideId": f"ride-{len(w.rides)}"}
            if outcome == "surge":
                return {"status": "needs_confirmation", "fare": "₹420", "note": "surge pricing"}
            return {"status": "failed", "reason": outcome}
        if tool == "ride_status":
            return {"status": w.rides[-1]["outcome"] if w.rides else "none"}
        if tool == "cancel_ride":
            if w.rides:
                w.rides[-1]["cancelled"] = True
            return {"cancelled": bool(w.rides)}
        if tool == "export_care_record":
            return dict(w.backend_record)
        if tool == "send_whatsapp":
            w.sent.append({**args, "at": clock.now().isoformat()})
            return {"delivered": True}
        if tool == "claim_schedule_rows":
            return {"claimed": len(args.get("scheduleIds") or [])}
        return {"ok": True, "note": f"sim: {tool} not modelled"}

    def fire_due_reminders(self, *, delivered: bool = True) -> list[dict]:
        """What the backend scheduler would do at this minute: send every active dose reminder that is due."""
        now = clock.ist()
        hhmm = now.strftime("%H:%M")
        day = clock.ist_day()
        fired = []
        for r in self.world.schedules.values():
            if r["active"] and r["time"] == hhmm and not any(
                s["scheduleId"] == r["scheduleId"] and s["dateKey"] == day for s in self.world.reminders_sent
            ):
                entry = {"dateKey": day, "scheduleId": r["scheduleId"], "item": f"{r['title']} at {r['time']}", "delivered": delivered, "at": now.isoformat()}
                self.world.reminders_sent.append(entry)
                fired.append(entry)
        return fired
