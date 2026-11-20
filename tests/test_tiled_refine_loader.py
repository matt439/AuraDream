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
