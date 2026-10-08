"""Browser loading options reach the same loader and metadata as launch flags."""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import model_selection
from backend.backends import diffusers_backend as module, loading
from backend.config import Settings
from backend.launch import CHECKPOINTS


@pytest.fixture(autouse=True)
def selection(monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", (None, 0))


def test_runtime_options_are_one_frozen_copy_and_model_only_switch_retains_them():
    startup = Settings(model_id=CHECKPOINTS[0].repo, quantization="fp8", offload="model")
    model_selection.select(CHECKPOINTS[1].repo, quantization="int8", offload="sequential")
    selected = model_selection.current_settings(startup)
    assert (selected.model_id, selected.quantization, selected.offload) == (
        CHECKPOINTS[1].repo, "int8", "sequential",
    )
    assert (startup.model_id, startup.quantization, startup.offload) == (
        CHECKPOINTS[0].repo, "fp8", "model",
    )
    with pytest.raises(FrozenInstanceError):
        selected.quantization = "none"
    model_selection.select(CHECKPOINTS[2].repo)
    next_selection = model_selection.current_settings(startup)
    assert (next_selection.quantization, next_selection.offload) == ("int8", "sequential")
    assert model_selection.revision() == 2
    assert selected.model_id == CHECKPOINTS[1].repo


def test_windows_inventory_keeps_all_ordinary_modes_but_refuses_nunchaku(monkeypatch):
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="win32"))
    startup = Settings()
    assert [item["value"] for item in model_selection.options(startup)["quantizations"]] == [
        "fp8", "int8", "none",
    ]
    assert model_selection.available(startup, "nunchaku") == []
    with pytest.raises(ValueError, match="WSL 2"):
        model_selection.validate(startup, CHECKPOINTS[0].lite_repo("nvfp4"), quantization="nunchaku")


def test_nunchaku_formats_are_explicit_and_never_offer_the_startup_plain_repo(monkeypatch):
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")
    monkeypatch.setitem(model_selection.imported_modules, "diffusers.utils.constants", SimpleNamespace(
        DIFFUSERS_TRUST_REMOTE_KERNELS=True,
    ))
    startup = Settings(model_id=CHECKPOINTS[0].repo)
    for fmt in ("nvfp4", "int4"):
        choices = model_selection.available(startup, "nunchaku", fmt)
        assert choices == [
            {"model_id": checkpoint.lite_repo(fmt), "label": checkpoint.name}
            for checkpoint in CHECKPOINTS if checkpoint.lite
        ]
        target = model_selection.validate(
            startup, CHECKPOINTS[0].lite_repo(fmt), quantization="nunchaku", format=fmt,
        )
        model_selection.select(target.model_id, quantization=target.quantization)
        assert model_selection.format_for(startup) == fmt
    with pytest.raises(ValueError, match="catalogue"):
        model_selection.validate(
            startup, CHECKPOINTS[0].lite_repo("int4"), quantization="nunchaku", format="nvfp4",
        )
    with pytest.raises(ValueError, match="catalogue"):
        model_selection.validate(startup, startup.model_id, quantization="nunchaku")
    with pytest.raises(ValueError, match="catalogue"):
        model_selection.validate(startup, CHECKPOINTS[0].lite_repo("nvfp4"), quantization="fp8")


def test_explicit_remote_kernel_refusal_is_respected_before_selection(monkeypatch):
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "false")
    with pytest.raises(ValueError, match="currently disables"):
        model_selection.validate(Settings(), CHECKPOINTS[0].lite_repo("int4"), quantization="nunchaku")
    assert model_selection.revision() == 0


def test_previously_imported_diffusers_kernel_refusal_wins_over_later_environment(monkeypatch):
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")
    monkeypatch.setitem(model_selection.imported_modules, "diffusers.utils.constants", SimpleNamespace(
        DIFFUSERS_TRUST_REMOTE_KERNELS=False,
    ))
    with pytest.raises(ValueError, match="disabled when Diffusers was imported"):
        model_selection.validate(Settings(), CHECKPOINTS[0].lite_repo("int4"), quantization="nunchaku")
    assert model_selection.revision() == 0


def test_cold_nunchaku_validation_does_not_import_diffusers(monkeypatch):
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")
    monkeypatch.delenv("DIFFUSERS_DISABLE_REMOTE_CODE", raising=False)
    monkeypatch.delitem(model_selection.imported_modules, "diffusers.utils.constants", raising=False)
    selected = model_selection.validate(
        Settings(), CHECKPOINTS[0].lite_repo("int4"), quantization="nunchaku",
    )
    assert selected.quantization == "nunchaku"
    assert "diffusers.utils.constants" not in model_selection.imported_modules


def test_remote_code_disabled_blocks_nunchaku_even_with_kernel_opt_in(monkeypatch):
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")
    monkeypatch.setenv("DIFFUSERS_DISABLE_REMOTE_CODE", "true")
    with pytest.raises(ValueError, match="currently disables"):
        model_selection.validate(Settings(), CHECKPOINTS[0].lite_repo("int4"), quantization="nunchaku")
    assert model_selection.revision() == 0


@pytest.mark.parametrize("options", [
    {"quantization": "fp16"}, {"offload": "cpu"}, {"format": "fp4"},
])
def test_invalid_loading_options_never_select(options):
    with pytest.raises(ValueError):
        model_selection.validate(Settings(), CHECKPOINTS[0].repo, **options)
    assert model_selection.revision() == 0


def test_explicit_custom_startup_repo_is_tied_to_its_known_quantization():
    startup = Settings(model_id="my-own/checkpoint", quantization="int8")
    assert startup.model_id in {item["model_id"] for item in model_selection.available(startup, "int8")}
    assert startup.model_id not in {item["model_id"] for item in model_selection.available(startup, "fp8")}
    assert model_selection.validate(startup, startup.model_id).model_id == startup.model_id
    with pytest.raises(ValueError, match="catalogue"):
        model_selection.validate(startup, "another/checkpoint")


def test_loader_uses_selected_quantizer_and_placement_without_mutating_startup(monkeypatch):
    import diffusers
    import torch

    startup = replace(module.settings, model_id=CHECKPOINTS[0].repo, quantization="fp8", offload="model")
    monkeypatch.setattr(module, "settings", startup)
    model_selection.select(CHECKPOINTS[1].repo, quantization="int8", offload="sequential")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = {}
    quant = object()

    class Pipeline:
        def __call__(self, generator=None, callback_on_step_end=None):
            pass

    pipe = Pipeline()

    def quantization_config(settings):
        seen["quantizer"] = settings
        return quant

    def from_pretrained(model_id, **kwargs):
        seen["loaded"] = (model_id, kwargs)
        return pipe

    monkeypatch.setattr(loading, "quantization_config", quantization_config)
    monkeypatch.setattr(loading, "unused_components", lambda model_id: (None, {}))
    monkeypatch.setattr(loading, "component_overrides", lambda *args, **kwargs: {})
    monkeypatch.setattr(loading, "place_pipeline", lambda pipe, settings: seen.update(placement=settings))
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", from_pretrained)
    backend = module.DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert seen["loaded"][0] == CHECKPOINTS[1].repo
    assert seen["loaded"][1]["quantization_config"] is quant
    for stage in ("quantizer", "placement"):
        assert (seen[stage].model_id, seen[stage].quantization, seen[stage].offload) == (
            CHECKPOINTS[1].repo, "int8", "sequential",
        )
    assert (startup.model_id, startup.quantization, startup.offload) == (CHECKPOINTS[0].repo, "fp8", "model")


def test_lora_refusal_tracks_runtime_quantization_before_loading():
    backend = module.DiffusersBackend()
    model_selection.select(CHECKPOINTS[0].lite_repo("nvfp4"), quantization="nunchaku")
    assert "4-bit SVDQuant" in backend.lora_unavailable_reason()
    assert not backend.is_ready()


def test_each_process_starts_its_own_revision():
    # A page still sending the last process's revision must be refused by a
    # restarted server, so the count never starts at the same number twice.
    script = "from backend import model_selection; print(model_selection.revision())"
    root = Path(__file__).resolve().parents[1]
    before = int(time.time() * 1000)
    first, second = (
        int(subprocess.run(
            [sys.executable, "-c", script], cwd=root, capture_output=True, text=True,
            check=True, timeout=60,
        ).stdout) for _ in range(2)
    )
    assert before <= first < second < 2**53
