"""Regular nudges. The pieces live in planner, compose, ledger and followups; this package keeps the old names."""

from __future__ import annotations

import time as _time
from datetime import timedelta

from app.care.nudges.compose import COMPOSE as COMPOSE
from app.care.nudges.compose import Context as Context
from app.care.nudges.compose import Option as Option
from app.care.nudges.compose import build_context as build_context
from app.care.nudges.compose import candidates as candidates
from app.care.nudges.compose import check as check
from app.care.nudges.compose import short_ref as short_ref
from app.care.nudges.compose import write_and_check as write_and_check
from app.care.nudges.followups import followup_ladder_skips as followup_ladder_skips
from app.care.nudges.followups import reminder_followups as reminder_followups
from app.care.nudges.ledger import GRADE as GRADE
from app.care.nudges.ledger import Learned as Learned
from app.care.nudges.ledger import MODE_TTL as MODE_TTL
from app.care.nudges.ledger import WINDOW as WINDOW
from app.care.nudges.ledger import _everyone as _everyone
from app.care.nudges.ledger import _iso as _iso
from app.care.nudges.ledger import _log as _log
from app.care.nudges.ledger import _mode as _mode
from app.care.nudges.ledger import _review as _review
from app.care.nudges.ledger import asked_open as asked_open
from app.care.nudges.ledger import cached_mode as cached_mode
from app.care.nudges.ledger import learn as learn
from app.care.nudges.ledger import learn_meals as learn_meals
from app.care.nudges.ledger import learned as learned
from app.care.nudges.ledger import ledger_rows as ledger_rows
from app.care.nudges.ledger import note_reply as note_reply
from app.care.nudges.ledger import pull_due as pull_due
from app.care.nudges.ledger import read_reactions as read_reactions
from app.care.nudges.ledger import reply_hint as reply_hint
from app.care.nudges.ledger import reply_rates as reply_rates
from app.care.nudges.ledger import review as review
from app.care.nudges.planner import Clock as Clock
from app.care.nudges.planner import Person as Person
from app.care.nudges.planner import load_work as load_work
from app.care.nudges.planner import record_appointment_tap as record_appointment_tap
from app.care.nudges.planner import run as run
from app.core import clock

def _export(mod_name: str) -> None:
    import importlib
    mod = importlib.import_module(f"app.care.nudges.{mod_name}")
    for name, value in vars(mod).items():
        if name.startswith("__"):
            continue
        globals()[name] = value

_export("ledger")
_export("planner")
_export("compose")
_export("followups")
globals()["_time"] = _time
globals()["timedelta"] = timedelta
globals()["clock"] = clock
