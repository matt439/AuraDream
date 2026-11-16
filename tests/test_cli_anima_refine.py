"""The measured Anima CLI refuses known mismatches before any weight load."""
from __future__ import annotations

import dataclasses

from PIL import Image
import pytest

from backend import config, generate, refine_batch, refine_qualification, schemas, schemas_generate, then_refine
from tests import fakes
from tests.cli_helpers import cli, isolated_execution_environment  # noqa: F401 (fixtures by name)


MODEL = "circlestone-labs/Anima"
MODELS = (MODEL, "circlestone-labs/Anima@aesthetic-v1.1", "Gazingstars123/Anima-2.9B")
READERS = (config, refine_batch, schemas, schemas_generate, then_refine, fakes)


@pytest.fixture(params=MODELS)
def turbo_cli(cli, monkeypatch, request):
    for module in READERS:
        monkeypatch.setattr(module, "settings", dataclasses.replace(
            module.settings, model_id=request.param, quantization="fp8", offload="model", device="cuda",
        ))
    cli.measured_refine = refine_qualification.envelope_for(request.param)
    return cli


def source(store):
    measured = refine_qualification.envelope_for(config.settings.model_id)
    assert measured is not None
    return store.save_image(
        Image.new("RGB", (1024, 1024), "navy"),
        schemas.GenerateRequest(prompt="keeper", width=1024, height=1024, steps=measured.steps, guidance_scale=measured.guidance_scale),
        seed=4471, duration_ms=1, model_id=measured.model_id, quantization="fp8", run_id="cli-source",
    )


def pass_args(image_id):
    measured = refine_qualification.envelope_for(config.settings.model_id)
    assert measured is not None
    return ["keeper", "--refine", str(image_id), "--scale", "1", "--steps", str(measured.steps),
            "--cfg", str(measured.guidance_scale), "-q"]


def test_cli_default_and_png_replay_keep_the_measured_source_strength(turbo_cli, store):
    first = source(store)
    measured = turbo_cli.measured_refine
    assert generate.main([*pass_args(first.id), "--seed", "7001"]) == 0
    ran = turbo_cli.requests[-1]
    assert (ran.refine, ran.strength, ran.width, ran.height, ran.steps, ran.guidance_scale) == (
        first.id, measured.default_strength, 1024, 1024, measured.steps, measured.guidance_scale,
    )
    saved = store.list_images()[0][0]
    assert (saved.refine, saved.strength) == (first.id, measured.default_strength)
    assert generate.main(["--from-image", str(store.settings.outputs_dir / saved.filename), "--seed", "7002", "-q"]) == 0
    assert (turbo_cli.requests[-1].refine, turbo_cli.requests[-1].strength) == (first.id, measured.default_strength)


@pytest.mark.parametrize("change, fragment", [
    ({"quantization": "int8"}, "fp8"),
    ({"offload": "sequential"}, "model offload"),
])
def test_cli_refine_loader_mismatch_is_cold(turbo_cli, store, monkeypatch, capsys, change, fragment):
    first = source(store)
    for module in READERS:
        monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, **change))
    assert generate.main(pass_args(first.id)) == 2
    assert fragment in capsys.readouterr().err
    assert turbo_cli.loads == 0 and turbo_cli.requests == []


def test_cli_later_grid_cell_is_refused_before_load(turbo_cli, store, capsys):
    first = source(store)
    measured = turbo_cli.measured_refine
    assert generate.main([*pass_args(first.id), "--grid-steps", f"{measured.steps},{measured.steps + 4}"]) == 2
    assert f"{measured.steps} steps" in capsys.readouterr().err
    assert turbo_cli.loads == 0 and turbo_cli.requests == []


def test_cli_then_refine_checks_known_loader_before_load(turbo_cli, monkeypatch, capsys):
    measured = turbo_cli.measured_refine
    for module in READERS:
        monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, quantization="int8"))
    assert generate.main([
        "keeper", "--width", "1024", "--height", "1024", "--steps", str(measured.steps), "--cfg", str(measured.guidance_scale),
        "--then-refine-scale", "1", "--then-refine-strength", str(measured.default_strength),
        "--then-refine-steps", str(measured.steps), "-q",
    ]) == 2
    assert "fp8" in capsys.readouterr().err
    assert turbo_cli.loads == 0 and turbo_cli.requests == []
