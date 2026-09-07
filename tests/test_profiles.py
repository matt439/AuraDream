"""The facts the UI and CLI hang off, checked against the real classes.

`cfg_threshold()` and `negative_prompt_supported` are derived from the loaded
pipeline's `__call__` signature, and `model_defaults` hard-codes the same two
answers per checkpoint for the unloaded case - so the profile's guess and the
pipeline's answer have to agree, and until now only a human kept them in step.
The classes import without weights or a GPU; the id -> class mapping is the
one assumption this cannot verify offline (the loader reads it from the repo's
model_index.json).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import diffusers
import pytest

from backend import generate, launch, sizes
from backend.backends import diffusers_backend as module
from backend.backends.diffusers_backend import DiffusersBackend
from backend.model_defaults import _FALLBACK, _profile_for, defaults_for

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("black-forest-labs/FLUX.1-Krea-dev", "FluxPipeline"),
    ("krea/Krea-2-Turbo", "Krea2Pipeline"),
    ("krea/Krea-2-Raw", "Krea2Pipeline"),
    ("black-forest-labs/FLUX.2-klein-4B", "Flux2KleinPipeline"),
    ("black-forest-labs/FLUX.2-klein-base-4B", "Flux2KleinPipeline"),
]


def _loaded_like(class_name: str) -> DiffusersBackend:
    cls = getattr(diffusers, class_name)
    backend = DiffusersBackend()
    # The real signature under a stand-in of the same *name*: the threshold
    # rule keys on the class name, and constructing a pipeline needs weights.
    backend._pipe = type(class_name, (), {})()
    backend._call_params = set(inspect.signature(cls.__call__).parameters)
    return backend


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_the_profile_guess_matches_the_real_pipeline(model_id, class_name):
    backend = _loaded_like(class_name)
    profile = defaults_for(model_id)
    assert backend.cfg_threshold() == profile["cfg_threshold"]
    assert ("negative_prompt" in backend._call_params) is profile["negative_prompt_supported"]
    # And every family has what load() insists on.
    assert module._REQUIRED_PARAMS <= backend._call_params


def test_every_catalogue_entry_has_a_profile():
    # The catalogue and the profile table are joined by substring only; an
    # entry that missed every token would print the fallback's "28 steps" in
    # the menu and run a distilled model at 28 steps. By identity, since the
    # flux profile is value-identical to the fallback in every number.
    repos = [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]
    for repo in repos:
        assert _profile_for(repo) is not _FALLBACK, repo


def test_the_size_table_is_the_same_in_the_browser_and_the_cli():
    # completed_cells keys on width and height, so a preset that differs
    # between the two makes --resume unable to continue a browser grid.
    src = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const SIZES = \{(.*?)\n\};", src, re.S).group(1)
    found = {}
    for aspect, body in re.findall(r"'([\d:]+)':\s*\{([^}]*)\}", block):
        found[aspect] = {
            int(tier): tuple(int(n) for n in re.findall(r"\d+", dims))
            for tier, dims in re.findall(r"(\d): \[([^\]]*)\]", body)
        }
    assert found == sizes.SIZES
    assert generate.SIZES is sizes.SIZES


def test_the_readme_size_table_matches_the_code():
    # The fourth copy. Rows read `| 16:9 | 1360×768 | 1920×1088 | 2736×1536 |`.
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    table = re.search(r"## Sizes.*?\n\n(\|.*?)\n\n", text, re.S).group(1)
    found = {}
    for line in table.splitlines()[2:]:
        aspect, *cells = [cell.strip() for cell in line.strip("|").split("|")]
        found[aspect] = {
            tier: tuple(int(n) for n in re.findall(r"\d+", cell))
            for tier, cell in zip(sizes.TIERS, cells)
        }
    assert found == sizes.SIZES


def test_the_size_guard_rails_are_the_largest_preset():
    # config.py used to derive these in prose and then hard-code rounder
    # numbers; the HTML's `max` on the two boxes is the same bound by hand.
    from backend.config import settings

    assert settings.max_side == max(max(size) for size in sizes.every_size()) == 2736
    assert settings.max_pixels == max(w * h for w, h in sizes.every_size()) == 1776 * 2368
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    bounds = re.findall(r'id="(?:width|height)"[^>]*max="(\d+)"', html)
    assert bounds == [str(settings.max_side)] * 2
    # Every preset passes the budget it defines.
    from backend.schemas import GenerateRequest

    for width, height in sizes.every_size():
        GenerateRequest(prompt="x", width=width, height=height).check_budget()
