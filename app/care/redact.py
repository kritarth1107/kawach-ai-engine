"""Secrets that must never be stored in memory."""

from __future__ import annotations

import re

OTP = re.compile(r"(?i)\b(otp|code|pin|password|cvv)\b[^0-9]{0,15}(\d{4,8})\b|\b(\d{4,8})\b(?=[^0-9]{0,15}\b(is your|otp)\b)")


def scrub_secrets(text: str) -> str:
    """One-time codes never enter memory."""
    return OTP.sub(lambda m: m.group(0).replace(m.group(2) or m.group(3) or "", "••••"), text or "")
