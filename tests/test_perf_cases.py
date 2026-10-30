"""The fixed speed comparison cases (`tools/perf_cases.py`) stay what the plans say.

docs/plan-step-caching.md, "Fixed comparison cases": every case comes out of
`applied()` unchanged, the two arms of a recipe differ only in the execution
extension, the prompts stay their authors' words, and a dry run counts six
images per shape while a nonempty recipe is still refused before any work.
"""
from __future__ import annotations

import dataclasses
import os
import signal

import pytest

import backend.backends as backends
from backend import config, generate, grid, guides
from backend.execution import UNAVAILABLE, ExecutionSpec
from backend.model_defaults import defaults_for
from backend.schemas import GenerateRequest
from tests.fakes import FakeBackend
from tools import perf_cases

# Each speed plan's two arms: the recipe it adds over Eager/Native/Off.
RECIPES = {
    "PERF#1": {"compilation": "regional-v1"},
    "PERF#3": {"attention_backend": "cudnn-v1"},
    "PERF#4": {"step_cache": "fbc-005-v1"},
}
# Shape A's scene case, both seeds: the id its card PNGs will carry.
RUN_ID_SCENE_A = "18602667a40f"


def _request(prompt: str, shape: str) -> GenerateRequest:
    width, height = perf_cases.SHAPES[shape]
    return GenerateRequest(
        prompt=perf_cases.PROMPTS[prompt], width=width, height=height, steps=perf_cases.STEPS,
        guidance_scale=perf_cases.CFG, num_images=perf_cases.BATCH,
    )


def _cells(request: GenerateRequest) -> list[grid.Cell]:
    return [grid.Cell(request=request, seed=seed) for seed in perf_cases.SEEDS]


def test_the_cases_are_three_prompts_two_seeds_two_shapes():
    assert perf_cases.CHECKPOINT == "black-forest-labs/FLUX.1-Krea-dev"
    assert (perf_cases.QUANTIZATION, perf_cases.OFFLOAD) == ("fp8", "model")
    assert perf_cases.SHAPES == {"A": (1024, 1024), "B": (1360, 768)}
    assert perf_cases.SEEDS == (4471, 8823)
    assert len(perf_cases.cases()) == 6 and len(set(perf_cases.cases())) == 6


def test_the_settings_are_the_krea_dev_profiles_own():
    profile = defaults_for(perf_cases.CHECKPOINT)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (perf_cases.STEPS, perf_cases.CFG, None)


@pytest.mark.parametrize("name", list(perf_cases.PROMPTS))
def test_every_prompt_is_its_authors_example_verbatim(name):
    # An edit to a guide fails here instead of drifting the cases.
    guide = getattr(guides, perf_cases.SOURCES[name])
    assert perf_cases.PROMPTS[name] in guide["examples"]


@pytest.mark.parametrize("prompt,shape", perf_cases.cases())
def test_applied_leaves_every_case_as_written(prompt, shape):
    profile = defaults_for(perf_cases.CHECKPOINT)
    request = _request(prompt, shape)
    for seed in perf_cases.SEEDS:
        applied = request.model_copy(update={"seed": seed}).applied(
            negative_prompt_supported=profile["negative_prompt_supported"],
            cfg_threshold=profile["cfg_threshold"], mu_supported=profile["mu_supported"],
        )
        assert applied.prompt == perf_cases.PROMPTS[prompt]
        assert (applied.width, applied.height) == perf_cases.SHAPES[shape]
        assert (applied.steps, applied.guidance_scale, applied.seed) == (perf_cases.STEPS, perf_cases.CFG, seed)
        assert applied.num_images == 1
        assert not applied.negative_prompt and applied.mu is None and applied.true_cfg is None
        assert applied.execution is None and applied.lora is None and applied.refine is None


@pytest.mark.parametrize("plan", list(RECIPES))
@pytest.mark.parametrize("prompt,shape", perf_cases.cases())
def test_a_recipes_arms_differ_only_in_the_execution_extension(plan, prompt, shape):
    request = _request(prompt, shape)
    eager = _cells(request.model_copy(update={"execution": ExecutionSpec()}))
    # GenerateRequest and applied() refuse a nonempty recipe today, so the
    # arm is a copy keyed without applied(), as test_execution.py does.
    recipe = _cells(request.model_copy(update={"execution": ExecutionSpec(**RECIPES[plan])}))
    execution_at = 12  # grid.cell_key's execution_key element
    for plain_cell, recipe_cell in zip(eager, recipe):
        plain_key, recipe_key = grid.cell_key(plain_cell), grid.cell_key(recipe_cell)
        assert plain_key[execution_at] is None and recipe_key[execution_at] is not None
        assert plain_key[:execution_at] + plain_key[execution_at + 1:] == (
            recipe_key[:execution_at] + recipe_key[execution_at + 1:]
        )
    legacy = grid.run_id(_cells(request), perf_cases.CHECKPOINT, perf_cases.QUANTIZATION)
    assert grid.run_id(eager, perf_cases.CHECKPOINT, perf_cases.QUANTIZATION) == legacy
    assert grid.run_id(recipe, perf_cases.CHECKPOINT, perf_cases.QUANTIZATION) != legacy


def test_the_legacy_run_id_of_the_first_case_is_pinned():
    # The id a card run's PNGs will carry; widening CellKey must not move it.
    cells = _cells(_request("scene", "A"))
    assert grid.run_id(cells, perf_cases.CHECKPOINT, perf_cases.QUANTIZATION) == RUN_ID_SCENE_A


@pytest.mark.parametrize("prompt,shape", perf_cases.cases())
def test_the_fp8_activation_pair_differs_only_by_run_id(prompt, shape):
    # PERF#2 compares a quantization, not a recipe: the cells and their keys
    # are the same (no key carries the quantization); only the run is its own.
    cells = _cells(_request(prompt, shape))
    assert grid.run_id(cells, perf_cases.CHECKPOINT, "fp8") != grid.run_id(cells, perf_cases.CHECKPOINT, "fp8-dynamic")


@pytest.fixture
def krea_cli(store, monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(backends, "get_backend", lambda name="diffusers": fake)
    monkeypatch.setattr(config, "settings", dataclasses.replace(
        store.settings, model_id=perf_cases.CHECKPOINT, quantization=perf_cases.QUANTIZATION,
        offload=perf_cases.OFFLOAD,
    ))
    monkeypatch.setattr(generate, "_server_holds_gpu", lambda host, port, timeout=2.0: None)
    # apply_env writes --model/--quant/--offload into os.environ: put each back.
    for name in ("AURADREAM_MODEL_ID", "AURADREAM_QUANT", "AURADREAM_OFFLOAD", "AURADREAM_COMPILATION",
                 "AURADREAM_ATTENTION", "AURADREAM_LOADER_EXECUTION_ERROR"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)
    handler = signal.getsignal(signal.SIGINT)
    yield fake
    signal.signal(signal.SIGINT, handler)


@pytest.mark.parametrize("shape", list(perf_cases.SHAPES))
def test_a_dry_run_counts_six_images_per_shape_and_loads_nothing(krea_cli, store, capsys, shape):
    total = 0
    for prompt in perf_cases.PROMPTS:
        assert generate.main([*perf_cases.generate_args(prompt, shape), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert f"model    : {perf_cases.CHECKPOINT} (quant=fp8" in out
        output = next(line for line in out.splitlines() if line.startswith("output   : "))
        assert output.startswith("output   : 2 image(s) - 2 run(s) x 1")
        total += int(output.split()[2])
    assert total == 6
    assert krea_cli.loads == 0 and krea_cli.requests == []
    assert store.list_images()[1] == 0


@pytest.mark.parametrize("plan", list(RECIPES))
def test_a_nonempty_recipe_arm_is_refused_before_any_work(krea_cli, store, capsys, plan):
    # When a recipe is admitted, this becomes a dry-run count for its arm.
    args = perf_cases.generate_args("scene", "A", recipe=RECIPES[plan])
    with pytest.raises(SystemExit) as stopped:
        generate.main([*args, "--dry-run"])
    assert stopped.value.code == 2
    assert UNAVAILABLE in capsys.readouterr().err
    assert krea_cli.loads == 0 and store.list_images()[1] == 0


def test_the_printed_commands_name_every_case_once(capsys):
    assert perf_cases.main(["--step-cache", "fbc-005-v1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6 and all("--step-cache fbc-005-v1" in line for line in lines)
    assert all("--seed-list 4471,8823" in line for line in lines)

