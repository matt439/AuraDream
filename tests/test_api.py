"""End-to-end through the HTTP layer, with a backend that draws nothing."""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import grid, main
from backend.backends.base import GenerationBackend
from backend.config import settings
from backend.jobs import JobManager
from backend.model_defaults import defaults_for, hints_for
from backend.schemas import GenerateRequest

ORIGIN = f"http://127.0.0.1:{settings.port}"


class FakeBackend(GenerationBackend):
    """Runs the whole job machinery at a few milliseconds per step."""

    name = "fake"

    def __init__(self, step_seconds: float = 0.0, cfg_threshold: float | None = None, loaded: bool = True):
        self.step_seconds = step_seconds
        self.requests: list[GenerateRequest] = []
        self.seeds: list[int] = []
        self.loaded = loaded
        self.loads = 0
        # None: apply everything as asked, like the base class. A number: behave
        # like a pipeline whose negative prompt switches on above it.
        self.cfg_threshold = cfg_threshold
        # Calls (1-based) that raise instead of rendering, and a hook run after
        # each render - for a cancel that lands after the last step.
        self.fail_on: set[int] = set()
        self.after_generate = None

    def applied(self, request):
        if self.cfg_threshold is None:
            return request
        return request.applied(negative_prompt_supported=True, cfg_threshold=self.cfg_threshold)

    def load(self) -> None:
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def is_ready(self) -> bool:
        return self.loaded

    def generate(self, request, seed, on_progress=None, should_cancel=None):
        from backend.backends import Cancelled

        if not self.loaded:
            # The real backend's contract: loading happens in main.ensure_loaded
            # and nowhere else, so a job that reaches here unloaded is a bug.
            raise RuntimeError("model is not loaded")
        self.requests.append(request)
        self.seeds.append(seed)
        if len(self.requests) in self.fail_on:
            raise RuntimeError(f"cell {len(self.requests)} blew up")
        for step in range(request.steps):
            if self.step_seconds:
                time.sleep(self.step_seconds)
            if on_progress is not None:
                on_progress(step + 1, request.steps)
            # The contract every backend signs up to: poll between steps.
            if should_cancel is not None and should_cancel():
                raise Cancelled
        if self.after_generate is not None:
            self.after_generate()
        # At the size that was asked for: the row records the image's own
        # dimensions, so anything matching on them - resume, the duration
        # estimate - needs a double that does not quietly render 16x16.
        return [
            Image.new("RGB", (request.width, request.height)) for _ in range(request.num_images)
        ]

    def describe(self) -> dict:
        return {
            "model_id": settings.model_id,
            "quantization": settings.quantization,
            "offload": settings.offload,
            "cfg_threshold": None,
        }


@pytest.fixture
def fake(store, monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(main, "backend", backend)
    monkeypatch.setattr(main, "_load_state", {"state": "unloaded", "detail": None})
    monkeypatch.setattr(main, "_load_lock", asyncio.Lock())
    # A fresh manager per test: shutting the app down drains its one worker
    # thread for good. Wired the way main.py wires it, so the single load path
    # the whole app relies on is the one under test.
    monkeypatch.setattr(main, "jobs", JobManager(backend, main.ensure_loaded))
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


def test_status_says_whether_the_pipeline_takes_a_negative_prompt(client, fake, monkeypatch):
    # The profile's guess until a pipeline is loaded, then the parameter list.
    # Derived, not a literal: the suite runs under whatever KREA_MODEL_ID the
    # shell has, and klein's profile says False.
    guess = defaults_for(settings.model_id)["negative_prompt_supported"]
    assert client.get("/api/status").json()["defaults"]["negative_prompt_supported"] is guess
    described = dict(fake.describe(), negative_prompt_supported=not guess)
    monkeypatch.setattr(fake, "describe", lambda: described)
    assert client.get("/api/status").json()["defaults"]["negative_prompt_supported"] is not guess


def test_the_first_generate_loads_the_weights_exactly_once(client, fake):
    fake.loaded = False
    first = client.post("/api/generate", json={"prompt": "x", "width": 256, "height": 256}).json()
    second = client.post("/api/generate", json={"prompt": "y", "width": 256, "height": 256}).json()
    assert _stream(client, first["job_id"])[-1]["status"] == "done"
    assert _stream(client, second["job_id"])[-1]["status"] == "done"
    # Two back-to-back requests share one load, and the status agrees.
    assert fake.loads == 1
    assert client.get("/api/status").json()["state"] == "ready"


def test_a_batch_that_multiplies_past_the_budget_is_refused(client):
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 2048, "height": 2048, "num_images": 8}
    )
    assert response.status_code == 422
    assert "Runs" in response.json()["detail"]
    assert client.post(
        "/api/generate", json={"prompt": "x", "width": 1024, "height": 1024, "num_images": 8}
    ).status_code == 200


def test_the_origin_check_understands_a_default_port(client, monkeypatch):
    import dataclasses

    monkeypatch.setattr(main, "settings", dataclasses.replace(settings, port=80))
    # A browser serialises an origin on port 80 without the port.
    assert client.post("/api/model/unload", headers={"Origin": "http://127.0.0.1"}).status_code != 403
    assert client.post("/api/model/unload", headers={"Origin": "http://127.0.0.1:7860"}).status_code == 403


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


# --------------------------------------------------------------- cancellation


def _wait_for(client, job_id, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the job")


def test_cancel_mid_sweep_keeps_what_finished(client, fake):
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 5, "runs": 8}
    ).json()["job_id"]
    # Poll for the first image rather than sleeping a guessed interval.
    _wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    events = _stream(client, job_id)
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1 and done[0]["status"] == "cancelled"
    # Landed within a step: fewer pipeline calls than runs, and everything that
    # finished before it is kept and reported.
    assert 1 <= len(done[0]["images"]) < 8
    assert len(fake.requests) < 8
    assert client.get("/api/gallery").json()["total"] == len(done[0]["images"])


def test_cancelling_a_queued_job_ends_it_at_once(client, fake):
    fake.step_seconds = 0.02
    ahead = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 40}
    ).json()["job_id"]
    queued = client.post(
        "/api/generate", json={"prompt": "y", "width": 256, "height": 256}
    ).json()["job_id"]
    started = time.perf_counter()
    assert client.post(f"/api/jobs/{queued}/cancel").json()["status"] == "cancelled"
    events = _stream(client, queued)
    # One terminal event, delivered without waiting for the job ahead.
    assert [e["type"] for e in events] == ["done"]
    assert time.perf_counter() - started < 0.5
    _stream(client, ahead)
    assert [r.prompt for r in fake.requests] == ["x"]


def test_a_cancel_after_the_last_step_does_not_relabel_a_complete_run(client, fake):
    # The flag is polled between steps; a cancel landing during the decode or
    # the encode of the final image changes nothing and used to still flip the
    # status to "cancelled", so a script treating that as "resume it" re-queued
    # a finished grid.
    job_id = None
    fake.after_generate = lambda: client.post(f"/api/jobs/{job_id}/cancel")
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 2}
    ).json()["job_id"]
    events = _stream(client, job_id)
    assert events[-1]["status"] == "done"
    assert len(events[-1]["images"]) == 1


def test_one_failed_cell_does_not_end_the_grid(client, fake):
    fake.fail_on = {2}
    _, events = _grid(client, steps_axis="6,8,10", seed=1)
    done = events[-1]
    assert done["status"] == "done"
    assert done["failures"] == 1
    assert "1 cell(s) failed" in done["error"] and "blew up" in done["error"]
    # The two other cells rendered; the failed one is what resume will find missing.
    assert [r.steps for r in fake.requests] == [6, 8, 10]
    assert sorted(r["steps"] for r in done["images"]) == [6, 10]


def test_three_failures_in_a_row_end_the_job(client, fake):
    fake.fail_on = {1, 2, 3}
    _, events = _grid(client, steps_axis="6,8,10,12", seed=1)
    assert events[-1]["status"] == "error"
    assert len(fake.requests) == 3


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


# ---------------------------------------------------------------- grid search


def _grid(client, **payload):
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, **payload}
    response = client.post("/api/grid", json=body)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    return job_id, _stream(client, job_id)


def test_a_grid_runs_every_cell_of_every_seed(client, fake):
    _, events = _grid(client, steps_axis="6,8", cfg_axis="0,1", seeds=2, seed=42)

    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    # Seeds outermost, then steps, then CFG - the order a half-finished grid
    # depends on to still be a usable comparison.
    assert [(r.steps, r.guidance_scale) for r in fake.requests] == [
        (6, 0.0), (6, 1.0), (8, 0.0), (8, 1.0),
        (6, 0.0), (6, 1.0), (8, 0.0), (8, 1.0),
    ]
    assert fake.seeds == [42, 42, 42, 42, 43, 43, 43, 43]
    assert len(events[-1]["images"]) == 8


def test_a_grid_says_which_cell_is_running(client, fake):
    _, events = _grid(client, steps_axis="6,8", seed=1)
    labels = [event["label"] for event in events if event.get("label")]
    # The axes the grid did not vary are filled from the default checkpoint's
    # profile, so the expected label follows whichever checkpoint that is.
    profile = defaults_for(settings.model_id)
    assert grid._label({}, 6, profile["guidance_scale"], profile["mu"]) in labels
    # A plain request varies nothing, so it labels nothing.
    _, plain = _run(client, runs=2)
    assert not [event for event in plain if event.get("label")]


def test_the_preview_expands_the_same_grid_it_would_run(client):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6-10x2",
        "mu_axis": "none,1.15",
        "seeds": 2,
    }
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["values"]["steps"] == [6, 8, 10]
    assert preview["values"]["mu"] == [None, 1.15]
    assert (preview["cells"], preview["images"]) == (12, 12)
    # Nothing in the gallery at this size yet, so it promises no duration at all
    # rather than guessing one.
    assert preview["seconds"] is None


def test_a_grid_varies_the_prompt_through_its_placeholders(client, fake):
    _, events = _grid(
        client,
        prompt="{size} dog sitting on {ground}",
        prompt_axis="size = big | small | none\nground = grass",
        seed=7,
    )
    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    assert [request.prompt for request in fake.requests] == [
        "big dog sitting on grass",
        "small dog sitting on grass",
        # `none` is a value: the placeholder and the space it sat in both go.
        "dog sitting on grass",
    ]
    # One prompt per cell, but one seed for all of them - the comparison is the
    # same noise under a different word.
    assert fake.seeds == [7, 7, 7]


def test_the_preview_shows_the_prompts_it_would_render(client):
    preview = client.post(
        "/api/grid/preview",
        json={
            "prompt": "{size} dog on {ground}",
            "width": 256,
            "height": 256,
            "prompt_axis": "size = big | small\nground = grass | dirt",
            "steps_axis": "6,8",
        },
    ).json()
    # The finished text, not the variables behind it: the substitution is the
    # half of a prompt grid that is easy to get wrong.
    assert preview["values"]["prompt"] == [
        "big dog on grass",
        "big dog on dirt",
        "small dog on grass",
        "small dog on dirt",
    ]
    assert (preview["cells"], preview["images"]) == (8, 8)
    assert preview["summary"].startswith("4 prompts x 2 steps")


def test_a_prompt_axis_that_does_not_line_up_is_a_422(client):
    body = {"prompt": "a {size} dog", "width": 256, "height": 256}
    unused = client.post("/api/grid/preview", json={**body, "prompt_axis": "colour = red"})
    assert unused.status_code == 422
    assert "{colour}" in unused.json()["detail"]
    # No variables at all with placeholders in the prompt is the easiest
    # mistake in the feature - the box forgotten, or cleared - and used to
    # render the literal braces into every image of the night.
    unfilled = client.post("/api/grid/preview", json={**body, "prompt_axis": ""})
    assert unfilled.status_code == 422
    assert "{size}" in unfilled.json()["detail"]
    # A prompt with no placeholders and no variables is the image Generate
    # would have made.
    plain = client.post("/api/grid/preview", json={"prompt": "a dog", "width": 256, "height": 256})
    assert plain.status_code == 200
    assert plain.json()["values"]["prompt"] == ["a dog"]


def test_resume_skips_only_the_prompt_it_already_rendered(client, store):
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(
            prompt="big dog on grass", width=256, height=256, steps=8, guidance_scale=0.0, mu=None
        ),
        seed=5,
        duration_ms=100,
        model_id=settings.model_id,
    )
    preview = client.post(
        "/api/grid/preview",
        json={
            "prompt": "{size} dog on grass",
            "width": 256,
            "height": 256,
            "prompt_axis": "size = big | small",
            "steps_axis": "8",
            "cfg_axis": "0",
            "mu_axis": "none",
            "seed": 5,
            "resume": True,
        },
    ).json()
    # The row matches one prompt of two; the other has never been rendered and
    # the identical settings under it must not be skipped with it.
    assert (preview["skipped"], preview["cells"]) == (1, 1)


def test_the_preview_estimates_from_what_the_gallery_measured(client, store):
    # Half a second per step, measured at this size on this checkpoint. The
    # fake backend renders instantly, so the row has to be a real one to
    # estimate from - and a row from another checkpoint must not count.
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256, steps=8),
        seed=2,
        duration_ms=40000,
        model_id="some/other-checkpoint",
    )
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256, steps=8),
        seed=1,
        duration_ms=4000,
        model_id=settings.model_id,
    )
    preview = client.post(
        "/api/grid/preview",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "8", "seeds": 4},
    ).json()
    # 0.5s x 8 steps x 4 cells. Per step, not per image: a grid whose cells
    # disagree about the step count cannot be costed from an image average.
    assert preview["seconds"] == pytest.approx(16.0)


def test_a_bad_axis_is_refused_before_anything_runs(client, fake):
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "eight"},
    )
    assert response.status_code == 422
    assert "not a number" in response.json()["detail"]
    assert not fake.requests


def test_a_grid_out_of_the_schemas_range_names_the_field(client):
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "90-120x10"},
    )
    assert response.status_code == 422
    assert "steps" in response.json()["detail"]


def test_runs_are_not_a_grid_axis(client):
    # They would multiply the night behind the count the preview promised.
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "runs": 4},
    )
    assert response.status_code == 422


def test_every_image_of_a_run_carries_its_job(client, fake):
    job_id, events = _grid(client, steps_axis="6,8", seed=5)
    records = events[-1]["images"]
    assert {r["job_id"] for r in records} == {job_id}
    # And that id is what the gallery filter selects on, so a night's work is
    # one search away rather than however far down the grid it has sunk.
    page = client.get(f"/api/gallery?search=job:{job_id}").json()
    assert page["total"] == len(records)


def test_a_resumed_grid_skips_what_the_gallery_already_holds(client, fake):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6,8,10",
        "seed": 11,
    }
    _stream(client, client.post("/api/grid", json={**body, "steps_axis": "6"}).json()["job_id"])
    assert len(fake.requests) == 1

    preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    assert (preview["cells"], preview["skipped"]) == (2, 1)

    _stream(client, client.post("/api/grid", json={**body, "resume": True}).json()["job_id"])
    # The 6-step cell was not rendered twice, and the order of what remains is
    # the order the grid would have reached it in.
    assert [r.steps for r in fake.requests] == [6, 8, 10]


def test_resuming_a_finished_grid_refuses_rather_than_running_nothing(client, fake):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6,8",
        "seed": 3,
    }
    _stream(client, client.post("/api/grid", json=body).json()["job_id"])
    response = client.post("/api/grid", json={**body, "resume": True})
    # An empty job would report "done" having drawn nothing, which reads as a
    # failure; 409 says the work is already there.
    assert response.status_code == 409
    assert len(fake.requests) == 2


def test_a_grid_publishes_the_seed_it_drew(client, fake):
    # A grid with a blank Seed box draws its own base and every image carries a
    # *derived* seed - so the base has to come back in the response, or resume
    # has nothing to key on.
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "seeds": 2},
    )
    accepted = response.json()
    assert isinstance(accepted["seed"], int)
    assert client.get(f"/api/jobs/{accepted['job_id']}").json()["seed"] == accepted["seed"]
    events = _stream(client, accepted["job_id"])
    # The base is the *oldest* image's seed, not the newest's.
    assert events[-1]["images"][0]["seed"] == accepted["seed"]
    assert events[-1]["images"][-1]["seed"] == accepted["seed"] + 1
    # And resuming with it finds every cell done.
    resume = client.post(
        "/api/grid",
        json={
            "prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "seeds": 2,
            "seed": accepted["seed"], "resume": True,
        },
    )
    assert resume.status_code == 409


def test_a_plain_request_publishes_its_seed_too(client):
    accepted = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "seed": 77}
    ).json()
    assert accepted["seed"] == 77
    assert accepted["run_id"]


def test_a_resumed_grid_is_one_run_under_two_jobs(client, fake):
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, "steps_axis": "6,8,10", "seed": 11}
    first = client.post("/api/grid", json=body).json()
    images = _stream(client, first["job_id"])[-1]["images"]
    # Pretend the night died after one cell: lose the other two, then resume.
    for record in images[1:]:
        client.delete(f"/api/images/{record['id']}")
    second = client.post("/api/grid", json={**body, "resume": True}).json()
    _stream(client, second["job_id"])
    assert second["job_id"] != first["job_id"]
    # The run id is the whole grid's, so both halves carry it - and the sheet's
    # `run:` filter returns the complete matrix where `job:` would return a part.
    assert second["run_id"] == first["run_id"]
    page = client.get(f"/api/gallery?search=run:{second['run_id']}").json()
    assert page["total"] == 3
    assert {item["run_id"] for item in page["items"]} == {first["run_id"]}
    assert len({item["job_id"] for item in page["items"]}) == 2


def test_the_record_carries_what_ran_not_what_was_asked(client, fake):
    # Turbo at CFG 0 never sees a negative prompt; the row and the resume key
    # used to carry it anyway, so clearing the greyed-out box before a resume
    # matched nothing and re-rendered the whole grid.
    fake.cfg_threshold = 0.0
    body = {
        "prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "cfg_axis": "0,1",
        "seed": 5, "negative_prompt": "blurry",
    }
    images = _stream(client, client.post("/api/grid", json=body).json()["job_id"])[-1]["images"]
    by_cfg = {record["guidance_scale"]: record["negative_prompt"] for record in images}
    assert by_cfg == {0.0: None, 1.0: "blurry"}
    assert {record["quantization"] for record in images} == {settings.quantization}
    # Resubmitted with the box cleared, the CFG 0 cells count as done and only
    # the guided cells - whose negative prompt did change - are outstanding.
    preview = client.post(
        "/api/grid/preview", json={**body, "negative_prompt": None, "resume": True}
    ).json()
    assert (preview["skipped"], preview["cells"]) == (2, 2)
    # And resubmitted as it was, everything is.
    assert client.post("/api/grid", json={**body, "resume": True}).status_code == 409


def test_resume_without_a_seed_is_refused(client, fake):
    # It would draw a fresh random base, match none of the night it is meant to
    # continue, and render the whole grid again while reporting a resume.
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "resume": True},
    )
    assert response.status_code == 422
    # A model validator lands in FastAPI's own structured 422 body rather than
    # in one of ours, which the UI flattens back into a sentence.
    assert "seed" in str(response.json()["detail"])
    assert not fake.requests


def test_a_bad_gallery_filter_is_a_422(client):
    response = client.get("/api/gallery?search=seed:tuesday")
    assert response.status_code == 422
    assert "not a number" in response.json()["detail"]


def test_the_pixel_budget_is_enforced(client):
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 2816, "height": 2816}
    )
    assert response.status_code == 422
