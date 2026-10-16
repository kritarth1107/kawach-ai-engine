"""One CloudWatch line on stdout.

CLOUDWATCH_EMF defaults off. The line stays in the logs either way. The _aws block,
which is what turns the line into custom metrics, is written only when the flag is on.
A cancelled hedge is not a call.
"""
from __future__ import annotations

import json
import os
import time


def metrics_on() -> bool:
    return os.environ.get("CLOUDWATCH_EMF", "off").strip().lower() in {"1", "on", "true"}


def note(namespace: str, role: str, *, errors: int = 0, throttles: int = 0, refusals: int = 0, latency_ms: float | None = None,
         region: str | None = None) -> None:
    metrics = [
        {"Name": "Calls", "Unit": "Count"},
        {"Name": "Errors", "Unit": "Count"},
        {"Name": "Throttles", "Unit": "Count"},
        {"Name": "Refusals", "Unit": "Count"},
    ]
    body: dict = {
        "namespace": namespace,
        "Role": role,
        "Calls": 1,
        "Errors": errors,
        "Throttles": throttles,
        "Refusals": refusals,
    }
    # Region is only set when the call left Mumbai. The Role-only series stays the one dashboards already use.
    dims: list = [["Role"], []]
    if region:
        body["Region"] = region
        dims = [["Role", "Region"], ["Role"], []]
    if latency_ms is not None:
        metrics.append({"Name": "LatencyMs", "Unit": "Milliseconds"})
        body["LatencyMs"] = round(latency_ms, 1)
    if metrics_on():
        body = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": namespace,
                    "Dimensions": dims,
                    "Metrics": metrics,
                }],
            },
            **body,
        }
    print(json.dumps(body), flush=True)
