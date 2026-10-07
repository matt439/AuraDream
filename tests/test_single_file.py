"""Pinned conversion, atomic cache publication and the normal quantized load."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import diffusers
import huggingface_hub
import pytest
import torch

from backend.backends import diffusers_backend as module
from backend.backends import loading, single_file
from backend.schemas import GenerateRequest


RECIPES = {
    single_file.ILLUSTRIOUS_ID: single_file.RECIPE,
    single_file.WAI_ID: single_file.WAI_RECIPE,
    single_file.ROUWEI_ID: single_file.ROUWEI_RECIPE,
}


@pytest.fixture(params=list(RECIPES))
def model_id(request):
    return request.param


@pytest.fixture
def conversion(monkeypatch, tmp_path):
    monkeypatch.setattr(single_file, "cache_root", lambda: tmp_path)
    calls = []

    def convert(path, model_id=single_file.ILLUSTRIOUS_ID):
        calls.append((path, model_id))
        path.mkdir()
        (path / "model_index.json").write_text("{}")

    monkeypatch.setattr(single_file, "_convert", convert)
    return calls


def test_conversion_is_published_once_and_reused_across_quantizers(conversion, model_id):
    first = single_file.resolve_model(model_id, "fp8")
    assert Path(first, "model_index.json").is_file()
    assert json.loads(Path(first, "local-diffusion-conversion.json").read_text()) == RECIPES[model_id]
    for quant in ("int8", "none", "fp8"):
        assert single_file.resolve_model(model_id, quant) == first
    assert single_file.converted_tree(model_id) == first
    assert len(conversion) == 1


def test_failed_conversion_is_not_reused(monkeypatch, conversion, tmp_path, model_id):
    convert = single_file._convert

    def fail(path, model_id):
        convert(path, model_id)
        raise RuntimeError("disk full")

    monkeypatch.setattr(single_file, "_convert", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        single_file.resolve_model(model_id, "fp8")
    assert not list(tmp_path.glob("*/model_index.json"))
    monkeypatch.setattr(single_file, "_convert", convert)
    result = single_file.resolve_model(model_id, "fp8")
    assert Path(result, "model_index.json").is_file()
    assert len(conversion) == 2


def test_a_failed_conversion_reports_its_own_error_past_an_open_file(monkeypatch, conversion, tmp_path, model_id):
    # On Windows a file still open stops the work directory's cleanup: the
    # caller still hears why the conversion failed, and the next load sweeps.
    convert = single_file._convert
    held = []

    def fail(path, model_id):
        convert(path, model_id)
        held.append((path / "model_index.json").open("rb"))
        raise RuntimeError("disk full")

    monkeypatch.setattr(single_file, "_convert", fail)
    try:
        with pytest.raises(RuntimeError, match="disk full"):
            single_file.resolve_model(model_id, "fp8")
    finally:
        for handle in held:
            handle.close()
    monkeypatch.setattr(single_file, "_convert", convert)
    result = single_file.resolve_model(model_id, "fp8")
    assert [path.name for path in tmp_path.iterdir() if path.is_dir()] == [Path(result).name]
    assert len(conversion) == 2


def test_an_interrupted_conversions_work_directory_is_swept(conversion, tmp_path, model_id):
    target, _, _ = single_file._target(model_id)
    stale = tmp_path / f"{target.name}-k3j9_0q1"
    (stale / "tree").mkdir(parents=True)
    (stale / "tree" / "model_index.json").write_text("{}")
    # Named like one but not one: a longer or shorter suffix, a copy of a
    # published tree, a file, a name that matches only if its dots match
    # anything - and the other checkpoint's work, which its own lock covers.
    longer = tmp_path / f"{target.name}-k3j9_0q1x"
    (longer / "tree").mkdir(parents=True)
    shorter = tmp_path / f"{target.name}-old"
    (shorter / "tree").mkdir(parents=True)
    copy = tmp_path / f"{target.name}-backup01"
    copy.mkdir()
    (copy / "model_index.json").write_text("{}")
    note = tmp_path / f"{target.name}-notes123"
    note.write_text("mine")
    undotted = tmp_path / f"{single_file._target(single_file.ILLUSTRIOUS_ID)[0].name.replace('.', 'x')}-k3j9_0q1"
    (undotted / "tree").mkdir(parents=True)
    other_id = single_file.WAI_ID if model_id == single_file.ILLUSTRIOUS_ID else single_file.ILLUSTRIOUS_ID
    other = tmp_path / f"{single_file._target(other_id)[0].name}-k3j9_0q1"
    (other / "tree").mkdir(parents=True)
    first = single_file.resolve_model(model_id, "fp8")
    assert not stale.exists()
    assert (longer / "tree").is_dir() and (shorter / "tree").is_dir() and (copy / "model_index.json").is_file()
    assert note.is_file() and (undotted / "tree").is_dir() and (other / "tree").is_dir()
    # Also once converted: what an earlier interrupted attempt left goes on
    # the next load, which converts nothing.
    (stale / "tree").mkdir(parents=True)
    assert single_file.resolve_model(model_id, "fp8") == first
    assert not stale.exists()
    assert len(conversion) == 1


def test_the_sweep_never_fails_a_load(monkeypatch, conversion, tmp_path, model_id, caplog):
    target, _, _ = single_file._target(model_id)
    # Work with a file another process still holds, and work whose listing is
    # refused: neither stops the load, and neither is reported gone.
    held = tmp_path / f"{target.name}-held_0a1"
    (held / "tree").mkdir(parents=True)
    unlisted = tmp_path / f"{target.name}-unlist_1"
    (unlisted / "tree").mkdir(parents=True)
    refused = {unlisted}
    rmtree, iterdir = shutil.rmtree, Path.iterdir

    def held_rmtree(path, *args, ignore_errors=False, **kwargs):
        if Path(path) != held:
            return rmtree(path, *args, ignore_errors=ignore_errors, **kwargs)
        if not ignore_errors:
            raise PermissionError(32, "The process cannot access the file", str(held / "tree"))

    def refusing_iterdir(self):
        if self in refused:
            raise PermissionError(13, "Access is denied", str(self))
        return iterdir(self)

    monkeypatch.setattr(shutil, "rmtree", held_rmtree)
    monkeypatch.setattr(Path, "iterdir", refusing_iterdir)
    first = single_file.resolve_model(model_id, "fp8")
    assert first == str(target)
    assert held.is_dir() and unlisted.is_dir()
    assert "could not remove an interrupted conversion yet" in caplog.text
    assert "removed an interrupted conversion" not in caplog.text
    # Nor does a cache directory that cannot be listed.
    refused.add(tmp_path)
    assert single_file.resolve_model(model_id, "fp8") == first
    assert len(conversion) == 1


def test_the_sweep_leaves_a_link_named_like_work_alone(conversion, tmp_path, model_id, caplog):
    target, _, _ = single_file._target(model_id)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "tree").mkdir(parents=True)
    link = tmp_path / f"{target.name}-linked01"
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(elsewhere), str(link))
    else:
        link.symlink_to(elsewhere, target_is_directory=True)
    single_file.resolve_model(model_id, "fp8")
    assert link.exists() and (elsewhere / "tree").is_dir()
    assert "interrupted" not in caplog.text


def test_the_sweep_waits_for_a_conversion_holding_the_lock(conversion, tmp_path, model_id):
    # Another process's conversion, mid-save: it holds the lock, and its work
    # directory is live until it lets go.
    import threading

    from huggingface_hub.utils import WeakFileLock

    target, _, _ = single_file._target(model_id)
    live = tmp_path / f"{target.name}-live0001"
    answer: list[str] = []
    with WeakFileLock(tmp_path / f"{target.name}.lock"):
        (live / "tree").mkdir(parents=True)
        worker = threading.Thread(target=lambda: answer.append(single_file.resolve_model(model_id, "fp8")))
        worker.start()
        worker.join(0.5)
        waited, kept = worker.is_alive(), live.is_dir()
    # Joined before any assert, so a failure never leaves it running into the next test.
    worker.join(10)
    assert waited and kept
    assert answer == [str(target)]
    assert not live.exists()


def test_ordinary_models_pass_through_and_nunchaku_refuses_before_conversion(conversion, model_id):
    assert single_file.resolve_model("cagliostrolab/animagine-xl-4.0", "fp8") == "cagliostrolab/animagine-xl-4.0"
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        single_file.resolve_model(model_id, "nunchaku")
    assert not conversion


def test_single_file_caches_cannot_be_confused(conversion, model_id):
    assert single_file.converted_tree(model_id) is None
    others = [single_file.resolve_model(other, "fp8") for other in RECIPES if other != model_id]
    assert single_file.converted_tree(model_id) is None
    tree = single_file.resolve_model(model_id, "fp8")
    assert len(set([tree, *others])) == len(RECIPES)
    assert single_file.converted_tree(model_id.upper()) == tree
    marker = Path(tree, "local-diffusion-conversion.json")
    marker.write_text(Path(others[0], marker.name).read_text(), encoding="utf-8")
    assert single_file.converted_tree(model_id) is None


def test_conversion_uses_only_pinned_weights_and_fixed_epsilon_euler_a(monkeypatch, tmp_path, model_id):
    downloads = []
    kwargs_seen = {}

    source = tmp_path / "checkpoint.safetensors"
    source.write_bytes(b"checkpoint fixture")
    recipe = RECIPES[model_id]
    if "source_sha256" in recipe:
        monkeypatch.setitem(recipe, "source_sha256", hashlib.sha256(source.read_bytes()).hexdigest())

    def download(**kwargs):
        downloads.append(kwargs)
        return str(source) if "filename" in kwargs else str(tmp_path)

    pipe = SimpleNamespace(scheduler=diffusers.EulerDiscreteScheduler())

    def save(path, **kwargs):
        assert isinstance(pipe.scheduler, diffusers.EulerAncestralDiscreteScheduler)
        assert pipe.scheduler.config.prediction_type == "epsilon"
        assert kwargs == {"safe_serialization": True}

    pipe.save_pretrained = save

    def load(path, **kwargs):
        assert path == str(source)
        kwargs_seen.update(kwargs)
        return pipe

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_single_file", load)
    single_file._convert(tmp_path, model_id)
    config, weights = downloads
    assert weights == recipe["source"]
    assert config["revision"] == single_file.SDXL_CONFIG["revision"]
    assert config["allow_patterns"] == ["*.json", "tokenizer/*.txt", "tokenizer_2/*.txt"]
    assert kwargs_seen == {
        "config": str(tmp_path), "local_files_only": True,
        "dtype": torch.bfloat16, "add_watermarker": False,
    }


@pytest.mark.parametrize("model_id", [single_file.WAI_ID, single_file.ROUWEI_ID])
def test_hash_mismatch_refuses_before_loading_or_publishing(monkeypatch, tmp_path, model_id):
    source = tmp_path / "bad.safetensors"
    source.write_bytes(b"wrong checkpoint")
    monkeypatch.setattr(single_file, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(source))

    def load(*args, **kwargs):
        raise AssertionError("must verify checkpoint before the loader runs")

    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_single_file", load)
    with pytest.raises(RuntimeError, match="SHA-256"):
        single_file.resolve_model(model_id, "fp8")
    assert single_file.converted_tree(model_id) is None
    assert not list((tmp_path / "cache").glob("*/model_index.json"))


@pytest.mark.parametrize("quant", ["fp8", "int8", "none"])
def test_backend_quantizes_the_converted_tree_but_keeps_the_public_id(monkeypatch, tmp_path, quant, model_id):
    tree = str(tmp_path / "converted")
    configured = dataclasses.replace(
        module.settings, model_id=model_id, quantization=quant, device="cpu", offload="none",
    )
    monkeypatch.setattr(module, "settings", configured)
    monkeypatch.setattr(single_file, "resolve_model", lambda model, mode: tree)
    quant_config = object() if quant != "none" else None

    def quantize(settings):
        assert settings.model_id == tree and settings.quantization == quant
        return quant_config

    def unused(model):
        assert model == tree
        return "StableDiffusionXLPipeline", {}

    class Pipe:
        scheduler = diffusers.EulerAncestralDiscreteScheduler()

        def to(self, device):
            return self

        def __call__(self, generator=None, callback_on_step_end=None, negative_prompt=""):
            pass

    def load(model, **kwargs):
        assert model == tree
        assert kwargs.get("quantization_config") is quant_config
        return Pipe()

    monkeypatch.setattr(loading, "quantization_config", quantize)
    monkeypatch.setattr(loading, "unused_components", unused)
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", load)
    backend = module.DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert module.settings.model_id == model_id
    assert backend.describe()["model_id"] == model_id
    assert backend.applied(GenerateRequest(
        prompt="a landscape", mu=1.15,
    )).mu is None
