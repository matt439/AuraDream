"""Reference images and LoRA adapters: the two conditioning inputs.

Both are all-or-nothing - a pipeline that takes no `image`, or an adapter no
file backs, renders a plain text-to-image and *looks* like a success - so what
is covered here is the refusal, the drop that keeps the row honest when
something gets past it, and the resume key that has to tell the two apart.
"""
from __future__ import annotations

import dataclasses
import inspect
import json

import pytest
from PIL import Image

from backend import grid, loras
from backend.backends import diffusers_backend as backend_module
from backend.backends.diffusers_backend import DiffusersBackend
from backend.schemas import GenerateRequest

# `tests/` has no __init__.py; pytest puts this directory on sys.path.
from test_klein import KleinLikePipeline


class ReferencePipeline(KleinLikePipeline):
    """klein's shape: `image` first, and no negative prompt."""

    def __call__(self, image=None, prompt=None, width=None, height=None,
                 num_inference_steps=None, guidance_scale=None, num_images_per_prompt=1,
                 generator=None, callback_on_step_end=None, sigmas=None):
        self.calls.append(dict(prompt=prompt, image=image))
        out = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [out] * num_images_per_prompt})()


def _loaded(pipe) -> DiffusersBackend:
    backend = DiffusersBackend()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    """An adapter directory with one file in it."""
    directory = tmp_path / "loras"
    directory.mkdir()
    (directory / "style.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        loras, "settings", dataclasses.replace(loras.settings, loras_dir=directory)
    )
    return directory


# ------------------------------------------------------------------ dropping


def test_references_are_dropped_where_the_pipeline_takes_none():
    # The record has to describe the image that was made. A pipeline with no
    # `image` parameter renders the prompt alone, so a row claiming two
    # references would misdirect Reuse and split the resume key forever.
    backend = _loaded(KleinLikePipeline())
    assert backend.describe()["reference_supported"] is False
    assert backend.applied(GenerateRequest(prompt="a fox", reference=[3, 4])).reference == []


def test_references_reach_a_pipeline_that_takes_them(store, monkeypatch):
    record = store.save_image(
        Image.new("RGB", (64, 64)),
        GenerateRequest(prompt="seed image", width=256, height=256),
        seed=1, duration_ms=10, model_id="test/model",
    )
    pipe = ReferencePipeline()
    backend = _loaded(pipe)
    assert backend.describe()["reference_supported"] is True

    request = GenerateRequest(
        prompt="the same fox, running", width=256, height=256, steps=4, reference=[record.id]
    )
    assert backend.applied(request).reference == [record.id]
    backend.generate(request, seed=1)
    # Decoded and handed over in the order asked for; klein resizes each to
    # 1 MP and crops it to the VAE's grid itself.
    assert [image.size for image in pipe.calls[0]["image"]] == [(64, 64)]


def test_a_missing_reference_is_named_rather_than_guessed(store):
    with pytest.raises(ValueError, match="no gallery image with id 404"):
        store.reference_paths([404])
    record = store.save_image(
        Image.new("RGB", (64, 64)), GenerateRequest(prompt="x", width=256, height=256),
        seed=1, duration_ms=10, model_id="test/model",
    )
    (store.settings.outputs_dir / record.filename).unlink()
    with pytest.raises(ValueError, match="missing from the gallery directory"):
        store.reference_paths([record.id])


def test_an_adapter_is_refused_where_it_cannot_be_applied(lora_dir, monkeypatch):
    # A pipeline with no LoRA mixin says so by name; all three families here
    # have one, but the loader is model-agnostic and an arbitrary repo may not.
    assert "no LoRA loader" in _loaded(KleinLikePipeline()).lora_unavailable_reason()

    backend = _loaded(SwapPipeline())
    assert backend.lora_unavailable_reason() is None
    # Nunchaku's linears are 4-bit SVDQuant, not the layers PEFT wraps: refused
    # by name, up front, the way the Nunchaku preflight refuses.
    monkeypatch.setattr(
        backend_module,
        "settings",
        dataclasses.replace(backend_module.settings, quantization="nunchaku"),
    )
    assert "nunchaku" in backend.lora_unavailable_reason()
    assert backend.lora_supported() is False
    # And dropped from the row if anything gets past that refusal.
    ran = backend.applied(GenerateRequest(prompt="a fox", lora="style.safetensors", lora_scale=0.5))
    assert ran.lora is None
    # The weight goes back to the default with it: a scale with no adapter is
    # not a setting, and left alone it would split the resume key on a number
    # no pixel depends on.
    assert ran.lora_scale == 1.0


# -------------------------------------------------------------------- the key


def test_the_resume_key_separates_conditioned_images_from_plain_ones():
    plain = grid.Cell(request=GenerateRequest(prompt="a fox", seed=1), seed=1)
    referenced = grid.Cell(request=GenerateRequest(prompt="a fox", seed=1, reference=[7]), seed=1)
    adapted = grid.Cell(
        request=GenerateRequest(prompt="a fox", seed=1, lora="style.safetensors", lora_scale=0.5),
        seed=1,
    )
    assert len({grid.cell_key(c) for c in (plain, referenced, adapted)}) == 3
    assert grid.cell_key(referenced)[6] == "7"
    assert grid.cell_key(adapted)[7] == "style.safetensors@0.5"


def test_a_grid_that_uses_neither_keeps_the_run_id_its_images_carry():
    # The id is a hash of the cell keys, so every element added to the key
    # would rename every run ever made - unless the name drops the trailing
    # optional elements while they are None, which is what this pins.
    cells = grid.expand(
        GenerateRequest(
            prompt="a {size} fox", negative_prompt="blurry", width=256, height=256, steps=8,
            guidance_scale=2.0, mu=1.15, seed=42,
        ),
        prompt="size = big | small", steps="6,8", seeds=2,
    )
    assert grid.run_id(cells, "krea/Krea-2-Turbo", "fp8") == "0ebe5101c492"


def test_conditioning_changes_the_run_id():
    def cells(**extra):
        return grid.expand(
            GenerateRequest(prompt="a fox", width=256, height=256, seed=1, **extra), steps="4,8"
        )

    plain = grid.run_id(cells(), "m", "fp8")
    assert grid.run_id(cells(lora="style.safetensors"), "m", "fp8") != plain
    assert grid.run_id(cells(reference=[3]), "m", "fp8") != plain


def test_a_weight_axis_needs_an_adapter_to_be_an_axis():
    # Without one the weight does nothing, so the axis must collapse rather
    # than multiply the night by a number no pixel depends on.
    base = GenerateRequest(prompt="a fox", width=256, height=256, seed=1)
    assert len(grid.expand(base, lora_scale="0,0.5,1")) == 1
    cells = grid.expand(base.model_copy(update={"lora": "style.safetensors"}), lora_scale="0-1x0.5")
    assert [cell.request.lora_scale for cell in cells] == [0.0, 0.5, 1.0]
    assert "LoRA 0.5" in cells[1].label


# ------------------------------------------------------------------- the swap


class SwapPipeline(KleinLikePipeline):
    """Records what `load_lora_weights`/`set_adapters`/`unload` were asked for."""

    def __init__(self, fail: bool = False):
        super().__init__()
        self.loaded: list[str] = []
        self.weights: list[list[float]] = []
        self.unloads = 0
        self.fail = fail

    def load_lora_weights(self, path, adapter_name=None):
        if self.fail:
            raise RuntimeError("bad adapter")
        self.loaded.append(path)

    def set_adapters(self, names, weights):
        self.weights.append(list(weights))

    def unload_lora_weights(self):
        self.unloads += 1


def test_a_weight_change_costs_a_reweighting_not_a_reload(lora_dir):
    backend = _loaded(SwapPipeline())
    pipe = backend._pipe
    for weight in (0.25, 0.5, 0.5, 1.0):
        backend._apply_lora(
            pipe, GenerateRequest(prompt="x", lora="style.safetensors", lora_scale=weight)
        )
    # Loaded once; re-weighted for each new value and not for the repeat, which
    # is what makes `0-1x0.25` an ordinary sweep rather than five reloads.
    assert len(pipe.loaded) == 1
    assert pipe.weights == [[0.25], [0.5], [1.0]]
    # Clearing it takes the adapter off the live pipeline.
    backend._apply_lora(pipe, GenerateRequest(prompt="x"))
    assert pipe.unloads == 1 and backend._lora is None


def test_a_failed_injection_leaves_no_half_applied_adapter(lora_dir):
    backend = _loaded(SwapPipeline(fail=True))
    with pytest.raises(RuntimeError):
        backend._apply_lora(backend._pipe, GenerateRequest(prompt="x", lora="style.safetensors"))
    # The cell fails either way; what must not happen is every cell after it
    # running under an adapter nothing recorded.
    assert backend._lora is None and backend._pipe.unloads == 1


def test_an_adapter_name_never_leaves_its_directory(lora_dir):
    assert loras.path_for("style.safetensors").name == "style.safetensors"
    assert [entry["name"] for entry in loras.available()] == ["style.safetensors"]
    with pytest.raises(ValueError, match="no LoRA named"):
        loras.path_for("missing.safetensors")
    # A path is refused by the schema, before it can reach the directory at all.
    for bad in ("../keys.safetensors", "sub/style.safetensors", "C:\\keys.safetensors"):
        with pytest.raises(ValueError):
            GenerateRequest(prompt="x", lora=bad)


# ----------------------------------------------------------------- the record


def test_the_row_and_the_png_carry_both(store, lora_dir):
    request = GenerateRequest(
        prompt="a fox", width=256, height=256, reference=[4, 9],
        lora="style.safetensors", lora_scale=0.75,
    )
    record = store.save_image(
        Image.new("RGB", (256, 256)), request, seed=3, duration_ms=10, model_id="test/model"
    )
    assert (record.reference, record.lora, record.lora_scale) == ([4, 9], "style.safetensors", 0.75)
    with Image.open(store.settings.outputs_dir / record.filename) as png:
        meta = json.loads(png.info["local-diffusion"])
        assert meta["reference"] == [4, 9]
        assert (meta["lora"], meta["lora_scale"]) == ("style.safetensors", 0.75)
        assert "LoRA: style.safetensors:0.75" in png.info["parameters"]
        assert "Reference: 4,9" in png.info["parameters"]
    # The key built from the row is the key built from the cell, or a resumed
    # night would render every conditioned cell twice.
    done = store.completed_cells(["a fox"], 256, 256, "test/model")
    assert grid.cell_keys(grid.Cell(request=request, seed=3))[0] in done
