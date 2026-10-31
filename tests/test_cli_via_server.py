"""`generate --via-server`: the command line scripting the running server.

No local backend is built or probed; every request goes through the server's
own routes, here the test client's, and is checked against the local path.
"""
from __future__ import annotations

import io

import pytest

from backend import gallery_lock, generate, generate_remote

from tests.cli_helpers import (  # noqa: F401  (fixtures by name)
    SIZE,
    _mask_file,
    _rewrite_record,
    _to_refine,
    isolated_execution_environment,
)


# ----------------------------------------------------- script the running server


@pytest.fixture
def via_server(client, fake, monkeypatch):
    import os
    from types import SimpleNamespace

    for name in ("AURADREAM_MODEL_ID", "AURADREAM_QUANT", "AURADREAM_OFFLOAD"):
        # Record the original absence too: apply_env may set a flag directly
        # during this invocation, after the fixture cleared its environment.
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AURADREAM_HOST", "127.0.0.1")
    monkeypatch.setenv("AURADREAM_PORT", "7860")
    calls = []

    def request(base, path, *, body=None, data=None, headers=None, timeout=30):
        assert base == "http://127.0.0.1:7860"
        method = "POST" if body is not None or data is not None else "GET"
        calls.append((method, path, body, headers or {}))
        response = client.request(method, path, json=body, content=data, headers=headers)
        if response.status_code >= 400:
            raise generate_remote._ServerError(str(response.json()["detail"]), response.status_code)
        return response.json()

    def no_local_backend(*args):
        pytest.fail("--via-server tried to construct/probe a local generation backend")

    monkeypatch.setattr(generate_remote, "_server_json", request)
    monkeypatch.setattr(generate_remote, "_SERVER_POLL_SECONDS", 0.01)
    monkeypatch.setattr(generate, "_project", no_local_backend)
    monkeypatch.setattr(generate, "_server_holds_gpu", no_local_backend)
    return SimpleNamespace(client=client, backend=fake, calls=calls, request=request)


def test_via_server_queues_the_plain_body_and_prints_each_url_seed_once(via_server, capsys):
    from backend import main

    argv = ["a fox 🦊", "--via-server", *SIZE, "--batch", "2", "--runs", "3", "--seed", "41",
            "--steps", "7", "--cfg", "2.5", "--true-cfg", "1.5", "--no-mu", "-n", "rain", "-q"]
    assert generate.main(argv) == 0
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert len(lines) == 6 and all(line.startswith("http://127.0.0.1:7860/outputs/") for line in lines)
    assert [int(line.rsplit("  seed ", 1)[1]) for line in lines] == list(range(41, 47))
    assert "Queued job" in out.err and "Queued job" not in out.out
    submitted = next(call for call in via_server.calls if call[1] == "/api/generate")
    body, headers = submitted[2:]
    assert {key: body[key] for key in ("prompt", "steps", "guidance_scale", "true_cfg", "mu", "num_images", "runs", "seed")} == {
        "prompt": "a fox 🦊", "steps": 7, "guidance_scale": 2.5, "true_cfg": 1.5,
        "mu": None, "num_images": 2, "runs": 3, "seed": 41,
    }
    assert headers["X-AuraDream-Model-Revision"] == str(main.model_selection.revision())
    assert via_server.backend.seeds == [41, 43, 45]


def test_via_server_refuses_configured_recipe_after_status_before_any_reference_or_generate(via_server, capsys):
    from backend import main
    from backend.execution import ExecutionSpec
    main.model_selection.select(
        "black-forest-labs/FLUX.1-Krea-dev", quantization="fp8", offload="model",
        loader_execution=ExecutionSpec(compilation="regional-v1"),
    )
    assert generate.main(["a fox", "--via-server", *SIZE, "--reference", "99", "-q"]) == 2
    assert [call[1] for call in via_server.calls] == ["/api/status"]
    assert via_server.backend.loads == 0 and via_server.backend.requests == []
    assert "not available" in capsys.readouterr().err


def test_via_server_explicit_eager_flags_remain_explicit_in_the_request(via_server):
    assert generate.main(["a fox", "--via-server", *SIZE, "--steps", "1", "--compilation", "eager",
                          "--attention", "native", "--step-cache", "off", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert "execution" in body and body["execution"] is None
    assert "execution" in via_server.backend.requests[0].model_fields_set


def test_via_server_uses_the_selected_servers_profile_when_flags_are_omitted(via_server, capsys):
    from backend import main

    main.model_selection.select("cagliostrolab/animagine-xl-4.0")
    expected = via_server.client.get("/api/status").json()["defaults"]
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert not {"steps", "guidance_scale", "true_cfg", "mu", "seed", "strength"} & body.keys()
    ran = via_server.backend.requests[0]
    assert (ran.steps, ran.guidance_scale) == (expected["steps"], expected["guidance_scale"])


def test_via_server_grid_body_uses_the_same_axes_and_seed_order(via_server, capsys):
    assert generate.main(["a {animal}", "--via-server", *SIZE, "--grid-prompt", "animal = fox | wolf",
                          "--grid-steps", "6,8", "--grid-cfg", "1.5,2", "--seed-list", "11,99",
                          "--batch", "2", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/grid")
    assert (body["prompt_axis"], body["steps_axis"], body["cfg_axis"], body["seed_axis"], body["runs"]) == (
        "animal = fox | wolf", "6,8", "1.5,2", "11,99", 1,
    )
    assert via_server.backend.seeds == [seed for seed in (11, 99) for _ in range(8)]
    assert [request.prompt for request in via_server.backend.requests[:8]] == ["a fox"] * 4 + ["a wolf"] * 4


def test_via_server_random_prompt_mode_matches_the_direct_expansion(via_server, capsys):
    from backend import grid

    assert generate.main([
        "a {size} fox", "--via-server", *SIZE, "--grid-prompt", "size = small | large | none",
        "--grid-prompt-mode", "random", "--grid-steps", "6,8", "--seed-list", "42,80", "--batch", "2", "-q",
    ]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/grid")
    assert body["prompt_mode"] == "random"
    assert via_server.backend.seeds == [42, 42, 43, 43, 80, 80, 81, 81]
    expected = grid.parse_variables(body["prompt_axis"])
    for request, seed in zip(via_server.backend.requests, via_server.backend.seeds):
        assert request.num_images == 1
        assert request.prompt == grid.substitute("a {size} fox", grid.random_assignment(expected, seed))


def test_size_axis_and_crop_anchor_via_server_match_direct_requests(via_server, store):
    assert generate.main(["a fox", "--via-server", *SIZE, "--grid-size", "256x256,512x256",
                          "--grid-steps", "4,8", "--seed", "13", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/grid")
    assert body["size_axis"] == "256x256,512x256"
    assert [(r.width, r.height, r.steps) for r in via_server.backend.requests] == [
        (w, h, steps) for w, h in ((256, 256), (512, 256)) for steps in (4, 8)
    ]
    source = _to_refine(store)
    assert generate.main(["a fox", "--via-server", *SIZE, "--refine", str(source.id), "--strength", "0.5",
                          "--crop-anchor", "bottom-left", "--seed", "3", "-q"]) == 0
    assert via_server.backend.requests[-1].crop_anchor == "bottom-left"


def test_via_server_resume_and_plain_repeat_budget_use_grid_semantics(via_server, capsys):
    argv = ["a fox", "--via-server", *SIZE, "--runs", "3", "--seed", "91", "-q"]
    assert generate.main([*argv, "--budget", "10"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/grid")
    assert (body["seeds"], body["runs"], body["budget_minutes"]) == (3, 1, 10)
    capsys.readouterr()
    assert generate.main([*argv, "--resume"]) == 0
    assert "Nothing to do" in capsys.readouterr().err
    assert len(via_server.backend.requests) == 3


@pytest.mark.parametrize("flags,word", [
    (["--dry-run"], "makes no network requests"),
    (["--ignore-server"], "are alternatives"),
    (["--resume"], "needs --seed"),
    (["--grid-steps", "6,8", "--runs", "2"], "a grid steps its own seeds"),
    (["--grid-lora-scale", "0,1"], "needs --lora"),
    (["--grid-strength", "0.4,0.6"], "needs --refine"),
])
def test_via_server_refuses_incompatible_modes_before_network(via_server, capsys, flags, word):
    with pytest.raises(SystemExit) as raised:
        generate.main(["a fox", "--via-server", *SIZE, *flags])
    assert raised.value.code == 2 and word in capsys.readouterr().err
    assert via_server.calls == []


@pytest.mark.parametrize("flag,value", [("--model", "a/different-model"), ("--quant", "none"), ("--offload", "none")])
def test_via_server_model_flags_assert_the_server_selection_without_switching(via_server, capsys, flag, value):
    assert generate.main(["a fox", "--via-server", *SIZE, flag, value]) == 2
    assert f"{flag} asks for {value}" in capsys.readouterr().err
    assert [(call[0], call[1]) for call in via_server.calls] == [("GET", "/api/status")]
    assert via_server.backend.requests == []


def test_via_server_matching_model_flags_and_environment_are_accepted(via_server, monkeypatch):
    status = via_server.client.get("/api/status").json()
    monkeypatch.setenv("AURADREAM_QUANT", status["quantization"])
    assert generate.main(["a fox", "--via-server", *SIZE, "--model", status["model_id"],
                          "--offload", status["offload"], "-q"]) == 0


def test_via_server_from_image_keeps_sampling_and_explicit_overrides(via_server, store, capsys):
    from tests.api_helpers import run_one

    _, events = run_one(via_server.client, prompt="a fox", width=256, height=256, steps=17,
                        guidance_scale=2.5, seed=72, mu=None)
    image = events[-1]["images"][0]
    path = store.settings.outputs_dir / image["filename"]
    _rewrite_record(path, lambda meta: meta.update(quantization="none"))
    via_server.calls.clear()
    assert generate.main(["a wolf", "--via-server", "--from-image", str(path), "--seed", "99", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["prompt"], body["seed"], body["steps"], body["guidance_scale"], body["mu"]) == ("a wolf", 99, 17, 2.5, None)
    assert "the PNG was made with quant=none" in capsys.readouterr().err


def test_via_server_refuses_carried_grid_labels_instead_of_silently_losing_them(via_server, store, capsys):
    from tests.api_helpers import stream_job

    accepted = via_server.client.post("/api/grid", json={"prompt": "a {animal}", "prompt_axis": "animal = fox | wolf",
                                                       "width": 256, "height": 256, "seed": 1}).json()
    image = stream_job(via_server.client, accepted["job_id"])[-1]["images"][0]
    with pytest.raises(SystemExit) as raised:
        generate.main(["--via-server", "--from-image", str(store.settings.outputs_dir / image["filename"])])
    assert raised.value.code == 2 and "cannot carry a PNG's grid/comparison labels" in capsys.readouterr().err
    assert via_server.calls == []


def test_via_server_refine_and_mask_use_the_server_gallery_and_upload_route(via_server, store, tmp_path, capsys):
    source = _to_refine(store)
    path, _ = _mask_file(tmp_path)
    assert generate.main(["a fox", "--via-server", "--refine", str(source.id), "--scale", "2", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["refine"], body["width"], body["height"]) == (source.id, 512, 768)
    via_server.calls.clear()
    assert generate.main(["a scarf", "--via-server", "--refine", str(source.id), "--mask", str(path),
                          "--strength", "0.6", "-q"]) == 0
    assert any(call[1] == f"/api/masks?source={source.id}" for call in via_server.calls)
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["width"], body["height"], body["strength"]) == (256, 384, 0.6)
    assert len(body["mask"]) == 64 and via_server.backend.requests[-1].mask == body["mask"]
    # The PNG's embedded mask can restore a swept server copy without the
    # CLI writing to any local masks directory itself.
    image = store.list_images()[0][0]
    store.mask_path(body["mask"]).unlink()
    assert generate.main(["--via-server", "--from-image", str(store.settings.outputs_dir / image.filename), "-q"]) == 0
    assert store.mask_path(body["mask"]).is_file()


def test_via_server_outpaint_uses_server_presets_and_keeps_its_spec(via_server, store):
    source = _to_refine(store)
    assert generate.main(["a wider view", "--via-server", "--refine", str(source.id), "--outpaint", "16:9",
                          "--band", "16", "--seed", "3", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["width"], body["height"]) == (688, 384)
    assert body["outpaint"] == {"left": 208, "top": 0, "right": 224, "bottom": 0, "band": 16, "fill": "edge-box3-r16"}
    assert "mask" not in body
    image = store.list_images()[0][0]
    assert generate.main(["--via-server", "--from-image", str(store.settings.outputs_dir / image.filename), "-q"]) == 0
    assert via_server.backend.requests[-1].outpaint == via_server.backend.requests[-2].outpaint


def test_via_server_from_image_verifies_carried_references_against_the_server_gallery(via_server, store, capsys):
    from tests.api_helpers import run_one

    source = _to_refine(store)
    _, events = run_one(via_server.client, prompt="a reference portrait", width=256, height=256,
                        reference=[source.id], seed=10)
    path = store.settings.outputs_dir / events[-1]["images"][0]["filename"]
    assert generate.main(["--via-server", "--from-image", str(path), "-q"]) == 0
    assert via_server.backend.requests[-1].reference == [source.id]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    count = len(via_server.backend.requests)
    assert generate.main(["--via-server", "--from-image", str(path), "-q"]) == 2
    assert "not verified as the PNG's reference" in capsys.readouterr().err
    assert len(via_server.backend.requests) == count
    # An explicit source selection is the user's own choice of server ids.
    assert generate.main(["--via-server", "--from-image", str(path), "--reference", str(source.id), "-q"]) == 0


def test_via_server_from_image_drops_a_mismatched_refine_source_as_the_local_cli_does(via_server, store, capsys):
    from tests.api_helpers import run_one

    source = _to_refine(store)
    _, events = run_one(via_server.client, prompt="a refined portrait", width=256, height=384,
                        refine=source.id, strength=0.6, seed=10)
    path = store.settings.outputs_dir / events[-1]["images"][0]["filename"]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--via-server", "--from-image", str(path), "-q"]) == 0
    assert via_server.backend.requests[-1].refine is None
    assert "running as a plain generation" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        generate.main(["--via-server", "--from-image", str(path), "--scale", "2", "-q"])
    assert "name the source with --refine" in capsys.readouterr().err


def test_via_server_reports_job_failures_and_budget_cancellation(via_server, capsys):
    via_server.backend.fail_on = {1}
    assert generate.main(["a fox", "--via-server", *SIZE, "--runs", "2", "-q"]) == 1
    assert "failed" in capsys.readouterr().err
    via_server.backend.fail_on = set()
    assert generate.main(["a fox", "--via-server", *SIZE, "--runs", "3", "--budget", "0.000001", "-q"]) == 130
    assert "stopped by budget" in capsys.readouterr().err


def test_via_server_ctrl_c_cancels_only_its_owned_job(via_server, monkeypatch, capsys):
    via_server.backend.step_seconds = 0.01
    request = via_server.request
    interrupted = []

    def ctrl_c(base, path, **kwargs):
        if path.startswith("/api/jobs/") and not interrupted:
            interrupted.append(1)
            raise KeyboardInterrupt
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate_remote, "_server_json", ctrl_c)
    assert generate.main(["a fox", "--via-server", *SIZE, "--steps", "30", "--runs", "3", "-q"]) == 130
    cancels = [call for call in via_server.calls if call[1].endswith("/cancel")]
    assert len(cancels) == 1 and "Cancelling job" in capsys.readouterr().err


def test_via_server_does_not_retry_or_fall_back_after_an_unconfirmed_submission(via_server, monkeypatch, capsys):
    request = via_server.request
    attempted = []

    def dropped_answer(base, path, **kwargs):
        if path == "/api/generate":
            attempted.append(path)
            raise generate_remote._ServerError("connection closed")
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate_remote, "_server_json", dropped_answer)
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 1
    assert attempted == ["/api/generate"]
    assert "Submission was not confirmed" in capsys.readouterr().err


def test_via_server_a_lost_owned_job_is_a_runtime_failure_and_is_never_resubmitted(via_server, monkeypatch, capsys):
    request = via_server.request

    def restarted(base, path, **kwargs):
        if path.startswith("/api/jobs/"):
            raise generate_remote._ServerError("unknown job", 404)
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate_remote, "_server_json", restarted)
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 1
    assert "unknown job (job " in capsys.readouterr().err
    assert len([call for call in via_server.calls if call[1] == "/api/generate"]) == 1


def test_via_server_refuses_a_stale_model_revision_before_generation(via_server, monkeypatch, capsys):
    from backend import main

    request = via_server.request

    def switched(base, path, **kwargs):
        if path == "/api/generate":
            main.model_selection.select("cagliostrolab/animagine-xl-4.0")
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate_remote, "_server_json", switched)
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 2
    assert "checkpoint changed" in capsys.readouterr().err
    assert via_server.backend.requests == []


def test_server_json_sends_json_and_revision_without_retrying_the_transport(monkeypatch):
    import json
    import urllib.request

    calls = []

    def respond(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(b'{"job_id":"one"}')

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    assert generate_remote._server_json("http://127.0.0.1:9999", "/api/generate", body={"prompt": "狐狸 🦊"},
                                 headers={"X-AuraDream-Model-Revision": "9"}) == {"job_id": "one"}
    [(request, timeout)] = calls
    assert request.get_method() == "POST" and json.loads(request.data) == {"prompt": "狐狸 🦊"}
    assert request.get_header("X-auradream-model-revision") == "9"
    assert request.get_header("Content-type") == "application/json" and timeout == 30


def test_server_json_reports_http_validation_and_network_errors_without_tracebacks(monkeypatch):
    import json
    import urllib.error
    import urllib.request

    def invalid(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 422, "Unprocessable", {},
                                     io.BytesIO(json.dumps({"detail": [{"loc": ["body", "width"], "msg": "invalid size"}]}).encode()))

    monkeypatch.setattr(urllib.request, "urlopen", invalid)
    with pytest.raises(generate_remote._ServerError, match="width: invalid size") as error:
        generate_remote._server_json("http://127.0.0.1:9999", "/api/generate", body={"width": 5})
    assert error.value.status == 422

    def offline(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("no listener"))

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(generate_remote._ServerError, match="no usable answer") as error:
        generate_remote._server_json("http://127.0.0.1:9999", "/api/status")
    assert error.value.status == 0


def test_gan_via_server_carries_the_choice_and_refuses_before_mask_upload(via_server, store, monkeypatch, tmp_path, capsys):
    from backend import model_selection, refine

    model_selection.select("LyliaEngine/waiIllustriousSDXL_v170")
    source = _to_refine(store)
    argv = ["a fox", "--via-server", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--seed", "7", "-q"]
    assert generate.main(argv) == 0
    assert via_server.backend.requests[-1].upscaler == refine.ANIME6B
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert body["upscaler"] == refine.ANIME6B
    mask, _ = _mask_file(tmp_path)
    calls = len(via_server.calls)
    assert generate.main([*argv, "--mask", str(mask)]) == 2
    assert "ordinary Refine" in capsys.readouterr().err
    assert not any(call[1].startswith("/api/masks") or call[1] == "/api/generate" for call in via_server.calls[calls:])
