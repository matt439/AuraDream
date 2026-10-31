"""Driving the app through TestClient: submit, follow the stream, wait."""
from __future__ import annotations

import json
import time

from backend.config import settings

ORIGIN = f"http://127.0.0.1:{settings.port}"

# The analyser's comparison, in the order written - which is not alphabetical,
# so a sheet sorted by label would put `original` last.
VARIANTS = [
    {"label": "rewrite", "prompt": "A woman in a yellow raincoat holds a red umbrella."},
    {"label": "original", "prompt": "a red umbrella held by a woman in a yellow raincoat"},
    {"label": "reordered", "prompt": "a woman in a yellow raincoat holding a red umbrella"},
]
TEXTS = [variant["prompt"] for variant in VARIANTS]
LABELS = [variant["label"] for variant in VARIANTS]


def stream_job(client, job_id):
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/stream") as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            events.append(event)
            if event["type"] == "done":
                break
    return events


def run_one(client, **payload):
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, **payload}
    job_id = client.post("/api/generate", json=body).json()["job_id"]
    return job_id, stream_job(client, job_id)


def until(predicate, timeout=5.0, *, interval=0.01, message="timed out"):
    """Poll `predicate` until it answers something truthy, and return that.

    The timeout is in seconds, not milliseconds: CI's runners are slow, and a
    test racing a clock needs the margin (AGENTS.md, Tests).
    """
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(interval)


def wait_for(client, job_id, predicate, timeout=5.0):
    def matched():
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        return snapshot if predicate(snapshot) else None

    return until(matched, timeout, message="timed out waiting for the job")


def wait_for_status(client, timeout=5.0, **expected):
    """Poll /api/status until every given field has its value; return the body."""
    def matched():
        body = client.get("/api/status").json()
        return body if all(body.get(key) == value for key, value in expected.items()) else None

    return until(matched, timeout, message=f"status never reached {expected}")


def run_grid(client, **payload):
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, **payload}
    response = client.post("/api/grid", json=body)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    return job_id, stream_job(client, job_id)
