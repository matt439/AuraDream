"""The bounded Turbo CLI refuses known mismatches before any weight load."""
from __future__ import annotations

import dataclasses

from PIL import Image
import pytest

from backend import config, generate, refine_batch, schemas, schemas_generate, then_refine
from tests import fakes
from tests.cli_helpers import cli, isolated_execution_environment  # noqa: F401 (fixtures by name)


MODEL = "circlestone-labs/Anima"
READERS = (config, refine_batch, schemas, schemas_generate, then_refine, fakes)


@pytest.fixture
def turbo_cli(cli, monkeypatch):
    for module in READERS:
        monkeypatch.setattr(module, "settings", dataclasses.replace(
            module.settings, model_id=MODEL, quantization="fp8", offload="model", device="cuda",
        ))
    return cli


def source(store):
    return store.save_image(
        Image.new("RGB", (1024, 1024), "navy"),
        schemas.GenerateRequest(prompt="keeper", width=1024, height=1024, steps=8, guidance_scale=1),
        seed=4471, duration_ms=1, model_id=MODEL, quantization="fp8", run_id="cli-source",
    )


def pass_args(image_id):
    return ["keeper", "--refine", str(image_id), "--scale", "1", "--steps", "8", "--cfg", "1", "-q"]


def test_cli_default_and_png_replay_keep_the_measured_source_strength(turbo_cli, store):
    first = source(store)
    assert generate.main([*pass_args(first.id), "--seed", "7001"]) == 0
    ran = turbo_cli.requests[-1]
    assert (ran.refine, ran.strength, ran.width, ran.height, ran.steps, ran.guidance_scale) == (
        first.id, .25, 1024, 1024, 8, 1,
    )
    saved = store.list_images()[0][0]
    assert (saved.refine, saved.strength) == (first.id, .25)
    assert generate.main(["--from-image", str(store.settings.outputs_dir / saved.filename), "--seed", "7002", "-q"]) == 0
    assert (turbo_cli.requests[-1].refine, turbo_cli.requests[-1].strength) == (first.id, .25)


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
    assert generate.main([*pass_args(first.id), "--grid-steps", "8,12"]) == 2
    assert "8 steps" in capsys.readouterr().err
    assert turbo_cli.loads == 0 and turbo_cli.requests == []


def test_cli_then_refine_checks_known_loader_before_load(turbo_cli, monkeypatch, capsys):
    for module in READERS:
        monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, quantization="int8"))
    assert generate.main([
        "keeper", "--width", "1024", "--height", "1024", "--steps", "8", "--cfg", "1",
        "--then-refine-scale", "1", "--then-refine-strength", ".25", "--then-refine-steps", "8", "-q",
    ]) == 2
    assert "fp8" in capsys.readouterr().err
    assert turbo_cli.loads == 0 and turbo_cli.requests == []
