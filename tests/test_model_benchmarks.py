"""Picker timing evidence must not cross variants or claim unmeasured settings."""
import pytest

from backend import model_benchmarks, model_info
from backend.launch import CHECKPOINTS
from backend.model_defaults import defaults_for


@pytest.fixture(autouse=True)
def windows_samples(monkeypatch):
    monkeypatch.setattr(model_benchmarks.sys, "platform", "win32")


def test_timing_comparisons_use_the_checkpoints_normal_steps():
    for model_id, samples in model_benchmarks._SAMPLES.items():
        assert model_id in {entry.repo for entry in CHECKPOINTS}
        assert all(sample.steps == defaults_for(model_id)["steps"] for sample in samples)
    detail = model_info.details_for("black-forest-labs/FLUX.1-Krea-dev")
    assert detail["speed_label"] == "Medium"
    assert detail["speed_basis"] == "Measured"
    assert "28 steps, CFG 3.5" in detail["speed_note"]
    assert "41.6–42.6 s" in detail["speed_note"]
    assert "first image after loading took 60.8 s" in detail["speed_note"]


@pytest.mark.parametrize("model_id", [
    "krea/Krea-2-Raw", "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/FLUX.2-klein-base-4B", "black-forest-labs/FLUX.2-klein-base-9B",
])
def test_unmeasured_variants_do_not_inherit_a_family_timing(model_id):
    detail = model_info.details_for(model_id)
    assert detail["speed_label"] == "Unmeasured"
    assert detail["speed_basis"] == "No timing"
    assert detail["speed_summary"] == f"{defaults_for(model_id)['steps']} default steps"


def test_custom_checkpoint_has_no_invented_speed_or_default_step_advice():
    detail = model_info.details_for("custom/checkpoint")
    assert detail["speed_label"] == "Unmeasured"
    assert detail["speed_summary"] == "No recorded timing"


def test_anima_releases_keep_their_own_measurements():
    turbo = model_info.details_for("circlestone-labs/Anima")
    aesthetic = model_info.details_for("circlestone-labs/Anima@aesthetic-v1.1")
    assert turbo["speed_label"] == "Fast"
    assert "5.68 / 4.40 s" in turbo["speed_note"]
    assert aesthetic["speed_label"] == "Medium"
    assert "23.229 / 22.582 s" in aesthetic["speed_note"]
    assert "CPU test suite" in aesthetic["speed_note"]


def test_speed_tooltip_retains_noncomparable_size_and_cold_cpu_caveats():
    wai = model_info.details_for("LyliaEngine/waiIllustriousSDXL_v170")
    assert "1.38 MP" in wai["speed_summary"]
    assert "1024×1344" in wai["speed_note"]
    animagine = model_info.details_for("cagliostrolab/animagine-xl-4.0")
    assert animagine["speed_label"] == "Medium"
    assert animagine["speed_basis"] == "Upper bound"
    assert "17.2 s" in animagine["speed_note"]
    assert "2026-10-05" in animagine["speed_note"]
    assert "First image of a process" in animagine["speed_note"]
    assert "CPU about 75% busy" in animagine["speed_note"]
    assert "upper-bound" in animagine["speed_note"]
    assert model_info.details_for("cagliostrolab/animagine-xl-4.0", "int8")["speed_basis"] == "FP8 reference"


@pytest.mark.parametrize(("quantization", "offload"), [
    ("int8", "model"), ("none", "model"), ("fp8", "sequential"), ("fp8", "none"),
])
def test_different_loader_settings_are_visibly_only_a_reference(quantization, offload):
    detail = model_info.details_for("Minthy/RouWei-0.8", quantization, offload)
    assert detail["speed_basis"] == "FP8 reference"
    assert "different setup" in detail["speed_note"]
    assert "FP8, Windows, model offload" in detail["speed_note"]


def test_windows_cannot_claim_wsl_only_timing_as_its_own():
    detail = model_info.details_for("krea/Krea-2-Turbo", "fp8", "model")
    assert detail["speed_basis"] == "NVFP4 reference"
    assert detail["speed_label"] == "Fast"
    assert "Nunchaku NVFP4, Linux/WSL" in detail["speed_note"]


def test_lite_format_and_platform_select_the_actual_recorded_sample(monkeypatch):
    klein = next(entry for entry in CHECKPOINTS if entry.repo == "black-forest-labs/FLUX.2-klein-4B")
    native = model_info.details_for(klein.repo)
    assert "10.2 s" in native["speed_note"]
    monkeypatch.setattr(model_benchmarks.sys, "platform", "linux")
    lite = model_info.details_for(klein.lite_repo("nvfp4"))
    assert lite["speed_basis"] == "Measured"
    assert "4.0 s" in lite["speed_note"]
    int4 = model_info.details_for(klein.lite_repo("int4"))
    assert int4["speed_basis"] == "NVFP4 reference"
    native_linux = model_info.details_for(klein.repo)
    assert native_linux["speed_basis"] == "FP8 reference"


def test_no_offload_has_its_own_slower_sample_where_recorded(monkeypatch):
    monkeypatch.setattr(model_benchmarks.sys, "platform", "linux")
    krea = next(entry for entry in CHECKPOINTS if entry.repo == "krea/Krea-2-Turbo")
    detail = model_info.details_for(krea.lite_repo("nvfp4"), "nunchaku", "none")
    assert detail["speed_label"] == "Medium"
    assert detail["speed_basis"] == "Measured"
    assert "26 s" in detail["speed_note"]
    assert "no offload" in detail["speed_note"]
    assert "spilled into system RAM" in detail["speed_note"]


def test_qwen_true_cfg_and_first_image_cost_remain_explicit(monkeypatch):
    monkeypatch.setattr(model_benchmarks.sys, "platform", "linux")
    qwen = next(entry for entry in CHECKPOINTS if entry.repo == "Qwen/Qwen-Image")
    detail = model_info.details_for(qwen.lite_repo("nvfp4"))
    assert detail["speed_label"] == "Slow"
    assert "True CFG 4 with a negative prompt" in detail["speed_note"]
    assert "first image of a session took 210 s" in detail["speed_note"]
    assert "Fast ≤15 s, Medium >15–45 s, Slow >45 s" in detail["speed_note"]
