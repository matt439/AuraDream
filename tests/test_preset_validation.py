"""Saved drafts use existing generation/grid rules without submitting work."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import subprocess
import sys

import pytest

from backend import grid, model_selection, schemas_generate
from backend.model_defaults import defaults_for
from backend.preset_validation import validate_payload
from backend.schemas_generate import GenerateRequest, GridRequest
from backend.schemas_presets import PresetGrid, PresetPayload
from backend.seeds import MAX_SEED


FLUX = "black-forest-labs/FLUX.1-Krea-dev"
TURBO = "krea/Krea-2-Turbo"
SDXL = "cagliostrolab/animagine-xl-4.0"


def draft(**values):
    return PresetPayload(prompt="  a fox\nunder\tstars  ", width=512, height=512, **values)


def test_profile_defaults_remain_unowned_and_zero_stays_explicit():
    payload = draft()
    turbo = validate_payload(payload, TURBO)
    flux = validate_payload(payload, FLUX)
    assert (turbo.effective.steps, turbo.effective.guidance_scale, turbo.effective.mu) == (8, 0.0, 1.15)
    assert (flux.effective.steps, flux.effective.guidance_scale, flux.effective.mu) == (28, 3.5, None)
    assert turbo.saved == flux.saved == payload
    assert payload.steps is payload.guidance_scale is payload.mu is None
    zero = validate_payload(draft(guidance_scale=0.0, mu=0.0), TURBO)
    assert zero.effective.guidance_scale == zero.effective.mu == 0.0
    assert zero.saved.guidance_scale == zero.saved.mu == 0.0
    assert zero.effective.prompt == payload.prompt


@pytest.mark.parametrize("seed", [None, 0, MAX_SEED])
def test_generate_keeps_seed_and_batch_repeat_without_drawing(monkeypatch, seed):
    def forbidden():
        pytest.fail("Validation must not draw or adopt a seed")

    monkeypatch.setattr(grid, "random_seed", forbidden)
    payload = draft(seed=seed, num_images=2, runs=3)
    result = validate_payload(payload, FLUX)
    assert result.saved.seed == result.effective.seed == seed
    assert (result.cells, result.images) == (3, 6)
    assert result.saved.num_images == result.effective.num_images == 2
    assert result.saved.runs == result.effective.runs == 3


@pytest.mark.parametrize("model_id,guidance,true_cfg,negative_active", [
    (TURBO, 0.0, None, False), (TURBO, 0.5, None, True),
    (FLUX, 3.5, None, False), (FLUX, 0.0, 1.5, True),
    (SDXL, 1.0, None, False), (SDXL, 7.0, None, True),
    ("black-forest-labs/FLUX.2-klein-4B", 1.0, None, False),
])
def test_negative_summary_matches_ordinary_applied(model_id, guidance, true_cfg, negative_active):
    payload = draft(negative_prompt="  not a dog\n\t", guidance_scale=guidance, true_cfg=true_cfg, mu=0.0)
    profile = defaults_for(model_id)
    result = validate_payload(payload, model_id)
    ordinary = GenerateRequest(
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        width=payload.width, height=payload.height, steps=profile["steps"],
        guidance_scale=guidance, true_cfg=true_cfg, mu=0.0,
    ).applied(
        negative_prompt_supported=profile["negative_prompt_supported"],
        cfg_threshold=profile["cfg_threshold"], mu_supported=profile["mu_supported"],
        true_cfg_supported=profile["true_cfg_supported"],
    )
    assert result.effective == ordinary
    assert bool(result.effective.negative_prompt) is negative_active
    assert ("negative_prompt" in result.inactive) is not negative_active
    assert result.saved.negative_prompt == "  not a dog\n\t"


def test_unused_sampling_is_visible_and_raw_values_survive():
    result = validate_payload(draft(mu=0.0, true_cfg=2.0), SDXL)
    assert result.saved.mu == 0.0 and result.saved.true_cfg == 2.0
    assert result.effective.mu is result.effective.true_cfg is None
    assert result.inactive == ["mu", "true_cfg"]
    ignored = validate_payload(draft(guidance_scale=2.0), "black-forest-labs/FLUX.1-schnell")
    assert "guidance_scale" in ignored.inactive


def test_cross_grid_matches_existing_expansion_without_adopting_seeds(monkeypatch):
    def forbidden():
        pytest.fail("Cold validation must not draw seeds")

    monkeypatch.setattr(grid, "random_seed", forbidden)
    axes = PresetGrid(
        prompt_axis="size = large | small\n", steps_axis="6, 8", cfg_axis="0, 0.5",
        mu_axis="none, 0", size_axis="512x512, 256x512", seeds=2,
    )
    payload = PresetPayload(
        mode="grid", prompt="a {size} fox", negative_prompt="not {size}",
        width=512, height=512, num_images=2, runs=7, grid=axes,
    )
    result = validate_payload(payload, TURBO)
    ordinary = GridRequest(
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        width=512, height=512, steps=8, guidance_scale=0.0, mu=1.15,
        seed=0, num_images=2, runs=1, **axes.model_dump(),
    )
    cells = grid.expand(
        ordinary, prompt=axes.prompt_axis, steps=axes.steps_axis,
        cfg=axes.cfg_axis, mu=axes.mu_axis, size=axes.size_axis, seeds=axes.seeds,
    )
    assert (result.cells, result.images) == (len(cells), len(cells) * 2) == (64, 128)
    assert result.values["prompt"] == ["a large fox", "a small fox"]
    assert result.effective_values["negative_prompt"] == [None, "not large", "not small"]
    assert result.values["seed"] == []
    assert result.effective.seed is result.saved.seed is None
    assert result.effective.runs == 1 and result.saved.runs == 7
    assert isinstance(result.effective, GridRequest) and result.effective.resume is False
    assert result.saved.grid == axes
    assert result.prompt_sample is False
    assert "negative_prompt" in result.inactive


@pytest.mark.parametrize("seed,seed_axis,sample", [(None, "", True), (42, "", False), (None, "42,80", False)])
def test_random_grid_preserves_per_image_semantics_and_labels_samples(monkeypatch, seed, seed_axis, sample):
    monkeypatch.setattr(grid, "random_seed", lambda: pytest.fail("Must not draw a seed"))
    axes = PresetGrid(
        prompt_axis="size = small | large | none", prompt_mode="random",
        steps_axis="6,8", seed_axis=seed_axis, seeds=2,
    )
    payload = PresetPayload(
        mode="grid", prompt="a {size} fox", width=512, height=512,
        seed=seed, num_images=2, grid=axes,
    )
    result = validate_payload(payload, TURBO)
    assert (result.cells, result.images) == (8, 8)
    assert result.prompt_sample is sample
    assert result.saved.seed == result.effective.seed == seed
    assert result.values["seed"] == ([42, 80] if seed_axis else [])
    assert result.saved.grid == axes
    bases = [42, 80] if seed_axis else [seed or 0, (seed or 0) + 2]
    expected = {
        grid.substitute("a {size} fox", grid.random_assignment({"size": ["small", "large", ""]}, base + offset))
        for base in bases for offset in range(2)
    }
    assert set(result.values["prompt"]) == expected


@pytest.mark.parametrize("axes,message", [
    ({"steps_axis": "101"}, "Steps"),
    ({"cfg_axis": "21"}, "CFG"),
    ({"mu_axis": "11"}, "Mu"),
    ({"size_axis": "513x512"}, "multiple of 16"),
    ({"seed_axis": str(MAX_SEED + 1)}, "Seed list"),
    ({"prompt_axis": "size = small | large"}, "nothing in the prompt says"),
    ({"steps_axis": "1-64", "cfg_axis": "1-20", "seeds": 2}, "a grid may queue"),
])
def test_invalid_or_oversize_axes_refuse(axes, message):
    payload = draft(mode="grid", grid=PresetGrid(**axes))
    with pytest.raises(ValueError, match=message):
        validate_payload(payload, TURBO)


@pytest.mark.parametrize("model,axes,message", [
    (SDXL, {"mu_axis": "none"}, "Mu axis is unavailable"),
    ("black-forest-labs/FLUX.1-schnell", {"cfg_axis": "0"}, "CFG axis is ignored"),
])
def test_unusable_active_axes_refuse_instead_of_silently_disappearing(model, axes, message):
    payload = draft(mode="grid", grid=PresetGrid(**axes))
    with pytest.raises(ValueError, match=message):
        validate_payload(payload, model)


def test_grid_checks_live_mu_answer_and_live_negative_normalization():
    payload = draft(mode="grid", grid=PresetGrid(mu_axis="0, 1"))

    def live(request):
        return request.applied(negative_prompt_supported=False, cfg_threshold=1.0, mu_supported=False)

    with pytest.raises(ValueError, match="Mu axis is unavailable"):
        validate_payload(payload, FLUX, applier=live)
    generated = validate_payload(draft(negative_prompt="kept in draft", true_cfg=2.0), FLUX, applier=live)
    assert generated.effective.negative_prompt is None
    assert generated.saved.negative_prompt == "kept in draft"
    assert "negative_prompt" in generated.inactive


@pytest.mark.parametrize("profile_model,live_true_cfg", [(SDXL, True), (FLUX, False)])
def test_true_cfg_inactive_summary_uses_live_normalization(profile_model, live_true_cfg):
    def live(request):
        return request.applied(
            negative_prompt_supported=True, cfg_threshold=1.0,
            true_cfg_supported=live_true_cfg,
        )

    # Equal draft values cannot reveal which guidance knob ran; the pure
    # capability probe must still distinguish profile/live disagreement.
    result = validate_payload(
        draft(negative_prompt="blur", guidance_scale=2.0, true_cfg=2.0),
        profile_model, applier=live,
    )
    assert result.effective.negative_prompt == "blur"
    assert ("true_cfg" in result.inactive) is not live_true_cfg


def test_requested_grid_values_are_distinct_from_normalized_sampling():
    result = validate_payload(draft(mode="grid", mu=0.0, grid=PresetGrid()), SDXL)
    assert result.saved.mu == 0.0
    assert result.values["mu"] == [0.0]
    assert result.effective_values["mu"] == [None]
    assert result.effective.mu is None
    assert "mu" in result.inactive
    # The HTTP base contract deliberately omits execution-only grid fields.
    encoded = result.model_dump()
    assert "resume" not in encoded["effective"]
    assert encoded["saved"]["grid"]["seeds"] == 1


def test_pixel_and_batch_budgets_use_the_existing_request_guard(monkeypatch):
    monkeypatch.setattr(schemas_generate, "settings", dataclasses.replace(
        schemas_generate.settings, max_pixels=512 * 512, max_batch_pixels=512 * 512,
    ))
    with pytest.raises(ValueError, match="in one batch exceeds"):
        validate_payload(draft(num_images=2), FLUX)
    payload = draft(mode="grid", grid=PresetGrid(size_axis="1024x1024"))
    with pytest.raises(ValueError, match="pixel budget"):
        validate_payload(payload, FLUX)


def test_validation_uses_supplied_profile_without_mutating_selection(monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", (FLUX, 99))
    result = validate_payload(draft(), TURBO)
    assert result.effective.steps == 8 and result.effective.mu == 1.15
    assert model_selection._selection == (FLUX, 99)


def test_model_copies_cannot_bypass_strict_saved_contract():
    with pytest.raises(ValueError):
        validate_payload(draft().model_copy(update={"seed": MAX_SEED + 1}), FLUX)
    with pytest.raises(ValueError):
        validate_payload(draft().model_copy(update={"steps": 0}), FLUX)


def test_editor_checkpoint_refuses_text_to_image_setup():
    with pytest.raises(ValueError, match="instruction"):
        validate_payload(draft(), "black-forest-labs/FLUX.1-Kontext-dev")


def test_fresh_process_validation_does_not_import_frameworks_or_hub():
    completed = subprocess.run(
        [sys.executable, "-c", "\n".join([
            "import sys",
            "from backend.preset_validation import validate_payload",
            "from backend.schemas_presets import PresetPayload, PresetGrid",
            "draft = PresetPayload(mode='grid', prompt='a {size} fox', grid=PresetGrid(",
            "    prompt_axis='size = large | small', prompt_mode='random'))",
            "result = validate_payload(draft, 'krea/Krea-2-Turbo')",
            "assert result.effective.seed is result.saved.seed is None",
            "assert result.prompt_sample is True",
            "assert not {'torch', 'diffusers', 'transformers', 'huggingface_hub'}.intersection(sys.modules)",
        ])],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
