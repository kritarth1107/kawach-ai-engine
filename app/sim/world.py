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
    templates: list[dict] = field(default_factory=list)  # template notices sent to people whose chat was closed
    team_alerts: list[dict] = field(default_factory=list)  # emails to the Kavach care team
    other_caregivers: int = 1  # caregivers other than the person an alert is about (0: someone caring for themselves)
    closed: set[str] = field(default_factory=set)  # people whose WhatsApp chat is closed (no message from them in 24 h)
    emergencies: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    rides: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    voice_modes: dict[str, str] = field(default_factory=dict)  # person id -> auto | always | never
    # scripted service outcomes, consumed in order: "ok" | "otp" | "captcha" | "stall" | "surge" | "fail"
    service_script: list[str] = field(default_factory=list)
    # what the backend already held for the family before Saheli v2 (export_care_record)
    backend_record: dict = field(default_factory=dict)
    # regular nudges (nudge_gate): the admin flag, a pause, flows in progress per person, their last inbound message (ISO),
    # and messages the backend sent them itself (reminders, dose nudges, the morning schedule): {to, text, at}
    nudge_mode: str = "off"
    meals_mode: str = "off"  # nudge.meals
    escalations: dict[str, str] = field(default_factory=dict)  # person → none | pending | deferred | sent | replied
    paused: bool = False
    flows: dict[str, str] = field(default_factory=dict)
    inbound: dict[str, str] = field(default_factory=dict)
    outbound: list[dict] = field(default_factory=list)


class SimHost:
    shadow = False

    def __init__(self, world: World | None = None) -> None:
        self.world = world or World()

    def _next_outcome(self) -> str:
        return self.world.service_script.pop(0) if self.world.service_script else "ok"

    async def call(self, tool: str, args: dict, *, family_id: str, subject_id: str, actor_id: str) -> dict:
        w = self.world
        w.calls.append({"tool": tool, "args": args, "at": clock.now().isoformat(), "actor": actor_id})
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
        if tool == "get_voice_preference":
            return {"mode": w.voice_modes.get(subject_id, "auto"), **getattr(w, "speech", {}).get(subject_id, {})}
        if tool == "set_voice_preference":
            out = {"ok": True}
            if "mode" in args:
                w.voice_modes[subject_id] = args["mode"]
                out["mode"] = args["mode"]
            if any(k in args for k in ("language", "dialect", "script")):
                if not hasattr(w, "speech"):
                    w.speech = {}
                w.speech[subject_id] = {k: args[k] for k in ("language", "dialect", "script") if k in args}
                out.update(w.speech[subject_id])
            return out
        if tool == "create_reminder":
            w.reminders.append(args)
            return {"ok": True, "reminderId": f"rem-{len(w.reminders)}"}
        if tool == "list_reminders":
            # the backend's shape: times and weekdays at the top level, schedule as a sentence
            return {"reminders": [{"reminderId": f"rem-{i + 1}", "text": r.get("text"), "times": (r.get("schedule") or {}).get("times") or [],
                                   "daysOfWeek": (r.get("schedule") or {}).get("days") or [], "schedule": "daily", "status": "active"}
                                  for i, r in enumerate(w.reminders)]}
        if tool == "notify_caregivers":
            # the backend never alerts the person the alert is about: a self-care family has no one else (others 0)
            w.alerts.append({**args, "at": clock.now().isoformat()})
            return {"notifiedCount": 1 if w.other_caregivers else 0, "others": w.other_caregivers}
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
        if tool == "emergency_link":
            return {"url": f"https://kavach.test/e/{subject_id}"}
        if tool == "export_care_record":
            return dict(w.backend_record)
        if tool == "send_whatsapp":
            if args.get("to") in w.closed:
                # the backend: a closed chat gets the template (the message waits for their next one), or nothing
                if not args.get("template"):
                    return {"delivered": False, "reason": "outside_window"}
                w.templates.append({"to": args.get("to"), **args["template"], "waiting": args.get("text"), "at": clock.now().isoformat()})
                return {"delivered": True, "via": "template", "waiting": True}
            w.sent.append({**args, "at": clock.now().isoformat()})
            # the backend returns the wamid Meta gave it (delivery ledger, B1.1)
            return {"delivered": True, "wamid": f"wamid.sim{len(w.sent)}", "wamids": [f"wamid.sim{len(w.sent)}"]}
        if tool == "alert_care_team":
            w.team_alerts.append({**args, "at": clock.now().isoformat()})
            return {"sent": True, "to": 1}
        if tool == "whatsapp_window":
            return {"open": args.get("to") not in w.closed}
        if tool == "nudge_gate":
            to = args.get("to") or subject_id
            outs = sorted([{"text": m.get("text", ""), "at": m["at"]} for m in [*w.sent, *w.outbound] if m.get("to") == to and m.get("at")],
                          key=lambda m: m["at"], reverse=True)[:5]
            flow = w.flows.get(to)
            reason = "paused" if w.paused else (f"active_flow:{flow}" if flow else ("window_closed" if to in w.closed else None))
            return {"ok": reason is None, "reason": reason, "mode": w.nudge_mode, "meals_mode": w.meals_mode, "window_open": to not in w.closed, "paused": w.paused,
                    "active_flow": f"active_flow:{flow}" if flow else None, "last_inbound_at": w.inbound.get(to),
                    "last_outbound_at": outs[0]["at"] if outs else None, "last_outbound_preview": outs}
        if tool == "escalation_status":
            return {"unanswered": 0, "state": w.escalations.get(args.get("to") or subject_id, "none")}
        if tool == "send_template":
            if args.get("to") not in w.closed:
                return {"sent": False, "reason": "chat open"}
            w.templates.append({"to": args.get("to"), **(args.get("template") or {}), "at": clock.now().isoformat()})
            return {"sent": True}
        if tool == "send_song":
            w.sent.append({**args, "song": True, "at": clock.now().isoformat()})
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
