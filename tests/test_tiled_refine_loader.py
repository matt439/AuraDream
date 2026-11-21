"""The private candidate uses the already-cached revision and normal ownership."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import sys

import huggingface_hub
import pytest

from backend import tiled_refine_contract as tiled
from backend.backends import diffusers_backend as module
from tests.test_pipeline_ownership import synthetic_loader


@pytest.mark.parametrize("returned_revision", [tiled.REVISION, "other-revision"])
def test_private_candidate_loads_only_the_owned_pinned_snapshot(
    synthetic_loader, monkeypatch, tmp_path, returned_revision,
):
    backend, _, events = synthetic_loader
    selection = replace(
        module.settings, model_id=tiled.MODEL_ID, quantization=tiled.QUANTIZATION,
        offload=tiled.OFFLOAD, device="cuda:0", loader_execution=None, loader_execution_error=None,
    )
    monkeypatch.setattr(module, "current_settings", lambda _: selection)
    monkeypatch.setattr(backend, "refine_host", lambda: ("win32", "0.42.1", "NVIDIA GeForce RTX 5080", True))
    monkeypatch.setattr(module.card_facts, "probe", lambda device: events.append(("probe", device)))
    monkeypatch.setattr(sys.modules["torch"].cuda, "is_available", lambda: True)
    snapshot = tmp_path / "models--cagliostrolab--animagine-xl-4.0" / "snapshots" / returned_revision

    def local_snapshot(model_id, *, revision, local_files_only, allow_patterns):
        assert (model_id, revision, local_files_only) == (tiled.MODEL_ID, tiled.REVISION, True)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", local_snapshot)
    monkeypatch.setattr(module.loading, "unused_components", lambda _: ("StableDiffusionXLPipeline", {}))
    monkeypatch.setattr(
        module.loading, "resolve_model", lambda *args, **kwargs: pytest.fail("the private pin reached Hub head"),
    )

    class StableDiffusionXLPipeline:
        config = SimpleNamespace(_name_or_path=str(snapshot))

        def maybe_free_model_hooks(self):
            pytest.fail("the candidate load returned denoising hooks")

        def __call__(self, generator=None, callback_on_step_end=None):
            pytest.fail("the candidate load performed inference")

    def from_pretrained(path, **kwargs):
        assert path == str(snapshot) and "quantization_config" in kwargs
        events.append("pinned-load")
        return StableDiffusionXLPipeline()

    monkeypatch.setattr(sys.modules["diffusers.pipelines.pipeline_utils"].DiffusionPipeline, "from_pretrained",
                        from_pretrained)
    if returned_revision != tiled.REVISION:
        with pytest.raises(ValueError, match="exact pinned Animagine snapshot"):
            backend._load_tiled_refine_candidate()
        assert not backend.is_ready() and "pinned-load" not in events
        return
    backend._load_tiled_refine_candidate()
    assert backend.is_ready() and backend._require_tiled_refine_owner() is backend._pipe
    assert events.count("pinned-load") == 1
    assert events.index("shards-enter") < events.index("pinned-load") < events.index("shards-exit")
    assert events.index("pinned-load") < events.index("place") < events.index("ready-log")
    backend.unload()
    assert backend._tiled_refine_candidate is None and not backend.is_ready()

def test_offline_candidate_skips_the_alternate_root_weights_and_refuses_a_missing_component(tmp_path, monkeypatch):
    """Exercise the Hub's actual local tree filter before the loader opens weights."""
    from huggingface_hub._snapshot_download import snapshot_download
    from huggingface_hub._tree_cache import TreeCacheEntry, write_tree_cache
    from huggingface_hub.errors import IncompleteSnapshotError

    cache = tmp_path / "hub"
    repository = cache / "models--cagliostrolab--animagine-xl-4.0"
    snapshot = repository / "snapshots" / tiled.REVISION
    files = ("model_index.json", "unet/diffusion_pytorch_model.safetensors", "text_encoder_2/model.safetensors")
    for key in files:
        file = snapshot / key
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"fixture")
    write_tree_cache(str(repository), tiled.REVISION, {
        key: TreeCacheEntry(7, "0" * 40)
        for key in (*files, ".gitattributes", "README.md", "animagine-xl-4.0-opt.safetensors")
    })
    def local_only(model_id, *, revision, local_files_only, allow_patterns):
        assert local_files_only is True
        return snapshot_download(model_id, revision=revision, allow_patterns=allow_patterns,
                                 cache_dir=cache, local_files_only=True)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", local_only)
    backend = module.DiffusersBackend()
    monkeypatch.setattr(backend, "_require_tiled_refine_loader", lambda **kwargs: None)
    monkeypatch.setattr(module.card_facts, "probe", lambda device: None)
    loaded = []
    monkeypatch.setattr(backend, "_load_prepared", lambda execution, *, tiled_refine_snapshot: loaded.append(
        tiled_refine_snapshot,
    ))
    backend._load_tiled_refine_candidate()
    assert loaded == [str(snapshot)]
    (snapshot / files[-1]).unlink()
    with pytest.raises(IncompleteSnapshotError, match="text_encoder_2/model.safetensors"):
        backend._load_tiled_refine_candidate()
    assert loaded == [str(snapshot)]


def test_public_resolver_reuses_complete_pin_or_fetches_configs_before_selected_weights(tmp_path, monkeypatch):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = tmp_path / "models--cagliostrolab--animagine-xl-4.0" / "snapshots" / tiled.REVISION
    calls = []

    def download(model_id, *, revision, allow_patterns):
        assert model_id == tiled.MODEL_ID and revision == tiled.REVISION
        calls.append(allow_patterns)
        for name in allow_patterns:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        (snapshot / "model_index.json").write_text(
            '{"_class_name":"StableDiffusionXLPipeline"}', encoding="utf-8",
        )
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(module.loading, "hub_file", lambda _, filename: str(snapshot / filename))
    assert module.DiffusersBackend._resolve_public_tiled_refine_snapshot() == str(snapshot)
    assert calls == [list(module._TILED_CONFIG_FILES), list(module._TILED_RECIPE_FILES)]
    assert all("*" not in name for name in calls[-1])
    assert module.DiffusersBackend._resolve_public_tiled_refine_snapshot() == str(snapshot)
    assert len(calls) == 2


def test_an_unavailable_pin_leaves_ordinary_animagine_load_available(monkeypatch):
    backend = module.DiffusersBackend()
    selection = replace(
        module.settings, model_id=tiled.MODEL_ID, quantization=tiled.QUANTIZATION,
        offload=tiled.OFFLOAD, device="cuda:0", loader_execution=None, loader_execution_error=None,
    )
    monkeypatch.setattr(module, "current_settings", lambda _: selection)
    monkeypatch.setattr(backend, "refine_host", lambda: ("win32", "0.42.1", "NVIDIA GeForce RTX 5080", True))
    attempted = []

    def unavailable():
        attempted.append("pin")
        raise OSError("offline")

    def load(execution, *, tiled_refine_snapshot=None):
        assert execution is None
        attempted.append(tiled_refine_snapshot)

    monkeypatch.setattr(backend, "_resolve_public_tiled_refine_snapshot", unavailable)
    monkeypatch.setattr(backend, "_load_prepared", load)
    backend.load()
    assert attempted == ["pin", None]
    selection = replace(selection, quantization="int8")
    backend.load()
    assert attempted == ["pin", None, None]
