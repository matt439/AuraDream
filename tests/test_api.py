"""End-to-end through the HTTP layer, with a backend that draws nothing."""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from backend.backends.base import GenerationBackend
from backend.config import settings
from backend.jobs import JobManager
from backend.model_defaults import defaults_for, hints_for
from backend.schemas import GenerateRequest

ORIGIN = f"http://127.0.0.1:{settings.port}"


class FakeBackend(GenerationBackend):
    """Runs the whole job machinery at a few milliseconds per step."""

    name = "fake"

    def __init__(self, step_seconds: float = 0.0):
        self.step_seconds = step_seconds
        self.requests: list[GenerateRequest] = []
        self.seeds: list[int] = []
        self.loaded = True

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def is_ready(self) -> bool:
        return self.loaded

    def generate(self, request, seed, on_progress=None, should_cancel=None):
        self.requests.append(request)
        self.seeds.append(seed)
        for step in range(request.steps):
            if self.step_seconds:
                time.sleep(self.step_seconds)
            if on_progress is not None:
                on_progress(step + 1, request.steps)
        return [Image.new("RGB", (16, 16)) for _ in range(request.num_images)]

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_id": settings.model_id,
            "quantization": settings.quantization,
            "offload": settings.offload,
            "detail": None,
            "pipeline": "FakePipeline",
            "cfg_threshold": None,
        }


@pytest.fixture
def fake(store, monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(main, "backend", backend)
    # A fresh manager per test: shutting the app down drains its one worker
    # thread for good, and there is nothing to load, so nothing to wait on.
    monkeypatch.setattr(main, "jobs", JobManager(backend))
    return backend


@pytest.fixture
def client(fake):
    # base_url has to be a host the app answers to - which is the point of the
    # middleware, and why TestClient's default "testserver" would 400.
    with TestClient(main.app, base_url=ORIGIN) as c:
        yield c


def _stream(client, job_id):
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


def _run(client, **payload):
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, **payload}
    job_id = client.post("/api/generate", json=body).json()["job_id"]
    return job_id, _stream(client, job_id)


# ------------------------------------------------------------------ defaults


def test_a_bare_request_takes_the_checkpoints_profile(client, fake):
    # The schema used to carry Turbo's 8 / 0.0 / 1.15 whatever the checkpoint,
    # so `curl -d '{"prompt": "..."}'` ran FLUX at 8 steps.
    _run(client)
    profile = defaults_for(settings.model_id)
    request = fake.requests[-1]
    assert (request.steps, request.guidance_scale, request.mu) == (
        profile["steps"],
        profile["guidance_scale"],
        profile["mu"],
    )


@pytest.mark.parametrize(
    "model_id",
    ["krea/Krea-2-Turbo", "krea/Krea-2-Raw", "black-forest-labs/FLUX.1-Krea-dev", "nobody/whatever"],
)
def test_every_profile_explains_its_three_controls(model_id):
    # The hints are format strings over the profile's own numbers, so they can
    # never quote a value the controls are not seeded with - and a mu of None or
    # a renamed key would blow up here rather than on the first hover.
    hints = hints_for(model_id)
    assert set(hints) == {"steps", "guidance_scale", "mu"}
    assert all(text and "{" not in text for text in hints.values())


def test_status_hints_quote_the_loaded_pipelines_threshold(client, fake, monkeypatch):
    # Where guidance switches on is the pipeline's answer, not the id's guess.
    # The tooltip and the greyed-out negative prompt field read the same number.
    described = dict(fake.describe(), cfg_threshold=7.0)
    monkeypatch.setattr(fake, "describe", lambda: described)
    body = client.get("/api/status").json()
    assert body["defaults"]["cfg_threshold"] == 7.0
    assert "7" in body["hints"]["guidance_scale"]


def test_an_explicit_null_mu_is_not_filled_in(client, fake):
    # What the UI sends to mean "no shift of ours"; filling it from the profile
    # would put Turbo's 1.15 back on and leave the record unable to reproduce.
    _run(client, mu=None)
    assert fake.requests[-1].mu is None


def test_explicit_values_win(client, fake):
    _run(client, steps=3, guidance_scale=7.5, mu=0.5)
    request = fake.requests[-1]
    assert (request.steps, request.guidance_scale, request.mu) == (3, 7.5, 0.5)


# ---------------------------------------------------------------- the stream


def test_a_sweep_reports_every_image_and_finishes(client, fake):
    _, events = _run(client, runs=2, num_images=2, seed=7)
    images = [e for e in events if e["type"] == "image"]
    assert len(images) == 4
    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    # seed + run * batch, then +1 per image in the batch.
    assert [i["image"]["seed"] for i in images] == [7, 8, 9, 10]
    assert fake.seeds == [7, 9]


def test_a_stream_opened_after_the_end_still_terminates(client):
    job_id, _ = _run(client)
    events = _stream(client, job_id)  # attach again, job long finished
    assert len(events) == 1 and events[0]["type"] == "done"
    assert events[0]["images"]


def test_active_jobs_are_listed_without_their_images(client, fake):
    fake.step_seconds = 0.02
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 20}
    ).json()["job_id"]
    live = client.get("/api/jobs").json()
    assert [job["job_id"] for job in live] == [job_id]
    assert "images" not in live[0]
    _stream(client, job_id)
    assert client.get("/api/jobs").json() == []


def test_every_stream_sees_the_terminal_event(store):
    """One shared queue split the events between two readers."""

    async def scenario():
        manager = JobManager(FakeBackend(step_seconds=0.01))
        job = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=4, num_images=2)
        )

        async def collect(stream):
            seen = []
            async for event in stream:
                seen.append(event)
                if event["type"] == "done":
                    return seen
            return seen

        return await asyncio.gather(
            collect(manager.events(job)), collect(manager.events(job))
        )

    first, second = asyncio.run(scenario())
    for seen in (first, second):
        assert seen[-1]["type"] == "done"
        assert len(seen[-1]["images"]) == 2


# -------------------------------------------------------------------- guards


def test_unload_is_refused_while_a_job_is_live(client, fake):
    fake.step_seconds = 0.02
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 30}
    ).json()["job_id"]
    response = client.post("/api/model/unload")
    assert response.status_code == 409
    assert fake.loaded  # 200 here used to free nothing and reload mid-sweep

    _stream(client, job_id)
    assert client.post("/api/model/unload").status_code == 200
    assert not fake.loaded


def test_a_cross_origin_write_is_refused(client):
    refused = client.post("/api/model/unload", headers={"Origin": "http://evil.example"})
    assert refused.status_code == 403
    # The page's own origin is exactly what a same-origin fetch announces.
    assert client.post("/api/model/unload", headers={"Origin": ORIGIN}).status_code == 200
    # Reads are not writes: this middleware is not a CORS policy.
    assert client.get("/api/status", headers={"Origin": "http://evil.example"}).status_code == 200


def test_a_rebound_host_header_is_refused(client):
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 400
    allowed = {"Host": f"localhost:{settings.port}"}
    assert client.get("/api/status", headers=allowed).status_code == 200


def test_the_pixel_budget_is_enforced(client):
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 2816, "height": 2816}
    )
    assert response.status_code == 422
