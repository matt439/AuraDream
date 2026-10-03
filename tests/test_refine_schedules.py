"""Refine advisories held to installed pipelines' real CPU schedule slicing."""
from __future__ import annotations

from types import SimpleNamespace

import diffusers
import pytest

from backend import grid, refine
from backend.schemas import GenerateRequest

from test_profiles import CASES


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_profiled_start_index_matches_the_real_siblings_scheduler(model_id, class_name):
    sibling = refine.sibling(getattr(diffusers, class_name))
    if sibling is None:
        return
    sdxl = class_name == "StableDiffusionXLPipeline"
    scheduler = (
        diffusers.EulerDiscreteScheduler()
        if sdxl else diffusers.FlowMatchEulerDiscreteScheduler()
    )
    pipe = SimpleNamespace(scheduler=scheduler)
    for steps in (4, 8, 10, 28):
        scheduler.set_timesteps(steps, device="cpu")
        for strength in (0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.9, 1.0):
            if not refine.leaves_a_step(steps, strength):
                continue
            timesteps, count = sibling.get_timesteps(pipe, steps, strength, "cpu")
            assert refine.start_index(steps, strength, model_id) == scheduler.begin_index
            assert refine.steps_run(steps, strength, model_id) == count == len(timesteps)
            assert timesteps.equal(scheduler.timesteps[scheduler.begin_index:])


@pytest.mark.parametrize("model_id, steps, axis, expected", [
    ("black-forest-labs/FLUX.1-Krea-dev", 28, "0.05-0.15x0.01", 7),
    ("black-forest-labs/FLUX.1-schnell", 4, "0.5-0.9x0.1", 2),
    ("Tongyi-MAI/Z-Image-Turbo", 8, "0.2-0.5x0.05", 4),
    ("KBlueLeaf/Kohaku-XL-Zeta", 10, "0.4,0.45,0.5", 1),
])
def test_duplicate_advisory_uses_integer_start_without_changing_resume_keys(
    model_id, steps, axis, expected,
):
    request = GenerateRequest(prompt="a fox", steps=steps, seed=1, refine=3,
                              strength=0.5, num_images=2)
    cells = grid.expand(request, strength=axis)
    keys = [grid.cell_key(cell) for cell in cells]
    run = grid.run_id(cells, model_id, "fp8")
    assert grid.duplicate_count(cells, lambda r: r, model_id) == expected * 2
    assert len(set(keys)) == len(cells)
    assert [grid.cell_key(cell) for cell in cells] == keys
    assert grid.run_id(cells, model_id, "fp8") == run
    # An off-catalogue family cannot promise equivalence from a guessed index.
    assert grid.duplicate_count(cells, lambda r: r, "nobody/custom") == 0


def test_duplicate_advisory_keeps_sources_seeds_and_prompts_distinct():
    request = GenerateRequest(prompt="a fox", steps=10, refine=3, strength=0.41)
    cells = [grid.Cell(request, 1)]
    for change in ({"refine": 4}, {"prompt": "a cat"}, {"steps": 11}):
        cells.append(grid.Cell(request.model_copy(update=change), 1))
    cells.append(grid.Cell(request, 2))
    assert grid.duplicate_count(cells, lambda r: r, "black-forest-labs/FLUX.1-Krea-dev") == 0
