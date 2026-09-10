"""Driving the app through TestClient: submit, follow the stream, wait."""
from __future__ import annotations

import json
import time

from backend.config import settings

ORIGIN = f"http://127.0.0.1:{settings.port}"


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


def wait_for(client, job_id, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the job")


def run_grid(client, **payload):
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, **payload}
    response = client.post("/api/grid", json=body)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    return job_id, stream_job(client, job_id)
