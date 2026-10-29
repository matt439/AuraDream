"""INP#11 stage 4's measured envelope as pure data: refusals, no callers yet."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from backend import grid
from backend import instruction_edit as edit
from backend.execution import ExecutionSpec
from tests.test_instruction_edit_identity import applied, request, stored

KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"
ENVELOPE = {
    "quantization": "fp8", "offload": "model", "device": "cuda", "platform": "win32",
    "loader_execution": None, "diffusers_version": "0.40.0",
}


def refusal(model_id=KONTEXT, **changes):
    return edit.loader_refusal(model_id, **{**ENVELOPE, **changes})


def test_the_envelope_is_what_stage_3_measured():
    envelope = edit.QUALIFIED_ENVELOPE
    assert (envelope.model_id, envelope.revision, envelope.recipe) == (
        edit.KONTEXT_V1.model_id, edit.KONTEXT_V1.revision, edit.KONTEXT_V1.name,
    )
    assert (envelope.quantization, envelope.offload, envelope.platform, envelope.diffusers_version) == (
        "fp8", "model", "win32", "0.40.0",
    )
    assert envelope.sizes == (((1, 1), (512, 512)),)
    with pytest.raises(AttributeError):
        envelope.quantization = "int8"  # type: ignore[misc]


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
@pytest.mark.parametrize("loader_execution", [None, {}, ExecutionSpec()])
def test_only_the_envelope_is_unrefused(device, loader_execution):
    assert refusal(device=device, loader_execution=loader_execution) is None


@pytest.mark.parametrize(("model_id", "changes", "named"), [
    (KONTEXT, {"quantization": "int8"}, "int8 quantization"),
    (KONTEXT, {"quantization": "none"}, "BF16 (no quantization)"),
    (KONTEXT, {"quantization": "nunchaku"}, "Nunchaku 4-bit"),
    (KONTEXT, {"offload": "sequential"}, "sequential offload"),
    (KONTEXT, {"offload": "none"}, "no offload"),
    (KONTEXT, {"platform": "linux"}, "Linux (WSL)"),
    (KONTEXT, {"device": "cpu"}, "the device 'cpu'"),
    (KONTEXT, {"device": "cuda:1"}, "the device 'cuda:1'"),
    (KONTEXT, {"loader_execution": ExecutionSpec(compilation="regional-v1")}, "regional-v1 compilation"),
    (KONTEXT, {"loader_execution": {"attention_backend": "cudnn-v1"}}, "cudnn-v1 attention"),
    (KONTEXT, {"loader_execution": {"step_cache": "fbc-005-v1"}}, "an unreadable performance configuration"),
    (KONTEXT, {"loader_execution": {"compilation": "future"}}, "an unreadable performance configuration"),
    (KONTEXT, {"diffusers_version": "0.41.0"}, "diffusers 0.41.0"),
    (KONTEXT, {"diffusers_version": "0.40.0.dev0"}, "diffusers 0.40.0.dev0"),
    (KONTEXT, {"diffusers_version": None}, "a missing diffusers"),
    (KONTEXT.lower(), {}, f"the checkpoint {KONTEXT.lower()!r}"),
    ("black-forest-labs/FLUX.1-Krea-dev", {}, "the checkpoint 'black-forest-labs/FLUX.1-Krea-dev'"),
])
def test_loader_refusal_names_each_unmeasured_choice(model_id, changes, named):
    reason = refusal(model_id, **changes)
    assert reason == edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice=named)
    assert reason.startswith("FLUX.1 Kontext [dev] is qualified on this card only at fp8 with model offload")
    assert reason.endswith(f"{named} is not measured.")


def test_the_first_unmeasured_choice_is_the_one_named():
    reason = refusal(quantization="int8", offload="none", platform="linux")
    assert reason is not None and "int8 quantization is not measured" in reason


def test_installed_diffusers_version_reads_metadata_only(monkeypatch):
    def missing(name):
        assert name == "diffusers"
        raise edit.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(edit.metadata, "version", missing)
    assert edit.installed_diffusers_version() is None
    monkeypatch.setattr(edit.metadata, "version", lambda name: "0.40.0")
    assert edit.installed_diffusers_version() == "0.40.0"


ASPECTS = [
    ((1024, 1024), (1, 1), (512, 512)), ((768, 512), (3, 2), (576, 384)),
    ((512, 768), (2, 3), (384, 576)), ((1024, 768), (4, 3), (512, 384)),
    ((768, 1024), (3, 4), (384, 512)), ((1920, 1080), (16, 9), (512, 288)),
    ((1080, 1920), (9, 16), (288, 512)),
]


@pytest.mark.parametrize(("source", "aspect", "working"), ASPECTS)
def test_only_square_geometry_is_qualified(source, aspect, working):
    reason = edit.geometry_refusal(edit.geometry(*source))
    if aspect == (1, 1):
        assert reason is None
    else:
        assert reason == (
            "Only square sources are qualified for instruction editing (512x512); "
            f"this {aspect[0]}:{aspect[1]} source would render at {working[0]}x{working[1]}, "
            "which is not measured."
        )


@pytest.mark.parametrize("size", [(256, 256), (512, 512), (1024, 1024), (3000, 3000)])
def test_aspect_not_source_size_is_the_qualified_dimension(size):
    assert edit.geometry_refusal(edit.geometry(*size)) is None


def test_geometry_refusal_takes_only_a_resolved_geometry():
    with pytest.raises(TypeError):
        edit.geometry_refusal((512, 512))  # type: ignore[arg-type]


@pytest.mark.parametrize(("source", "aspect", "working"), ASPECTS)
def test_geometry_table_unchanged(source, aspect, working):
    # The envelope narrows what runs, never what binds or inspects.
    resolved = edit.geometry(*source)
    assert resolved.working_size == resolved.output_size == working
    assert resolved.pipeline_kwargs()["max_area"] == working[0] * working[1]
    with pytest.raises(ValueError, match="exact source aspect"):
        edit.geometry(1000, 999)


def test_edit_run_id_and_cell_key_unchanged():
    # Pinned before stage 4 from the stage-1/2 identity fixture; a 3:2 row
    # outside the envelope keys and binds exactly as it did.
    square = grid.Cell(request(), 7)
    assert grid.cell_key(square) == (
        "make it night", None, 28, 2.5, None, None, None, None, None, None, None,
        '{"edit_source":{"filename":"source.png","hash":"' + "ab" * 32
        + '","height":1024,"width":1024},"recipe":"kontext-v1","source":17}',
        None, "512x512@center", 7,
    )
    assert grid.run_id([square], KONTEXT, "fp8") == "a83cde9e6cb2"
    wide = stored(edit_source={"filename": "source.png", "hash": "cd" * 32, "width": 768, "height": 512})
    wide_cell = grid.Cell(request(edit=wide, width=576, height=384), 7)
    assert grid.run_id([wide_cell], KONTEXT, "fp8") == "d36d4cfce2f8"
    assert grid.cell_key(grid.Cell(applied(request()), 7)) == grid.cell_key(square)


def test_envelope_module_import_is_cold():
    root = str(Path(__file__).resolve().parents[1])
    script = """
import importlib.abc
import sys
sys.path.insert(0, ROOT)
class RefuseWarm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'diffusers', 'huggingface_hub'} or fullname in {
            'backend.config', 'backend.storage', 'backend.main',
        }:
            raise AssertionError('unexpected warm import: ' + fullname)
sys.meta_path.insert(0, RefuseWarm())
from backend import instruction_edit as edit
assert edit.loader_refusal(
    edit.KONTEXT_V1.model_id, quantization='fp8', offload='model', device='cuda', platform='win32',
    loader_execution=None, diffusers_version='0.40.0',
) is None
assert edit.geometry_refusal(edit.geometry(768, 512)) is not None
edit.installed_diffusers_version()
assert not {'torch', 'diffusers', 'huggingface_hub'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "ROOT = " + repr(root) + "\n" + script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_option_refusals_name_each_unmeasured_choice_alone():
    # The Models picker holds the editor's quantization and offload options by
    # these words: each the envelope's refusal of that one choice, the rest of
    # the loader taken as measured; a measured choice and any other checkpoint
    # have none.
    holds = edit.option_refusals(
        KONTEXT, quantizations=["fp8", "int8", "none", "nunchaku"], offloads=["model", "sequential", "none"],
    )
    assert holds == {
        "quantization": {
            "int8": refusal(quantization="int8"), "none": refusal(quantization="none"),
            "nunchaku": refusal(quantization="nunchaku"),
        },
        "offload": {"sequential": refusal(offload="sequential"), "none": refusal(offload="none")},
    }
    assert holds["quantization"]["int8"] == edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="int8 quantization")
    assert edit.option_refusals(KONTEXT, quantizations=["fp8"], offloads=["model"]) == {"quantization": {}, "offload": {}}
    for other in (KONTEXT.lower(), "black-forest-labs/FLUX.1-Krea-dev", "Qwen/Qwen-Image-Edit"):
        assert edit.option_refusals(other, quantizations=["int8"], offloads=["none"]) == {}
