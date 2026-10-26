"""Bounded, offline GAN preprocessing; never downloads or executes GPU work."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from backend import refine, storage
from backend.backends.base import Cancelled


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}


@pytest.fixture
def asset(tmp_path, monkeypatch):
    payload = b"tiny pinned fixture"
    target = tmp_path / "cache" / "weights.pth"
    monkeypatch.setattr(refine, "ANIME6B_BYTES", len(payload))
    monkeypatch.setattr(refine, "ANIME6B_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(refine, "_anime6b_path", lambda: target)
    return payload, target


def test_download_is_verified_published_and_then_checked_offline(asset, monkeypatch):
    import urllib.request

    payload, target = asset
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: calls.append(request.full_url) or Response(payload, len(payload)))
    assert refine.anime6b_weights() == target
    assert target.read_bytes() == payload and calls == [refine.ANIME6B_URL]
    assert not list(target.parent.glob("*.part"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("cached weights caused a network request"))
    assert refine.anime6b_weights() == target


@pytest.mark.parametrize("damage", ["short", "oversize", "hash", "header"])
def test_failed_download_publishes_nothing_and_preserves_other_files(asset, monkeypatch, damage):
    import urllib.request

    payload, target = asset
    target.parent.mkdir()
    unrelated = target.parent / "unrelated.bin"
    unrelated.write_bytes(b"leave alone")
    body = payload[:-1] if damage == "short" else payload + b"extra" if damage == "oversize" else b"x" * len(payload) if damage == "hash" else payload
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response(body, 1 if damage == "header" else None))
    with pytest.raises(ValueError):
        refine.anime6b_weights()
    assert not target.exists() and not list(target.parent.glob("*.part"))
    assert unrelated.read_bytes() == b"leave alone"


def test_corrupt_existing_cache_is_refused_without_replacement(asset, monkeypatch):
    import urllib.request

    payload, target = asset
    target.parent.mkdir()
    target.write_bytes(b"x" * len(payload))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("existing artifact was replaced"))
    with pytest.raises(ValueError, match="SHA256"):
        refine.anime6b_weights()
    assert target.read_bytes() == b"x" * len(payload)


def test_cancel_during_download_cleans_only_its_own_partial(asset, monkeypatch):
    import urllib.request

    payload, target = asset
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response(payload))
    polls = iter([False, True])
    with pytest.raises(Cancelled):
        refine.anime6b_weights(lambda: next(polls))
    assert not target.exists() and not list(target.parent.glob("*.part"))


def test_source_bounds_refuse_before_decode_or_weight_access(monkeypatch):
    class Header:
        size = (4097, 256)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def convert(self, *args): pytest.fail("oversized source was decoded")

    monkeypatch.setattr(storage, "reference_paths", lambda *a, **k: (Path("header-only.png"),))
    monkeypatch.setattr(Image, "open", lambda *a: Header())
    monkeypatch.setattr(refine, "anime6b_weights", lambda *a: pytest.fail("oversized source fetched weights"))
    with pytest.raises(ValueError, match="4.19 MP"):
        refine.source_image(1, 256, 256, upscaler=refine.ANIME6B)
    with pytest.raises(ValueError, match="4.19 MP"):
        refine.check_upscaler_source(1)
    with pytest.raises(ValueError, match="4.19 MP"):
        refine.anime6b_upscale(SimpleNamespace(size=(4096, 4096)))
    refine.check_upscaler_size(2048, 2048)


def test_unreadable_source_is_a_worded_refusal(monkeypatch):
    monkeypatch.setattr(storage, "reference_paths", lambda *a, **k: (Path("broken.png"),))
    monkeypatch.setattr(Image, "open", lambda *a: (_ for _ in ()).throw(OSError("broken fixture")))
    with pytest.raises(ValueError, match="source image 1 could not be read"):
        refine.check_upscaler_source(1)


def test_rrdb_construction_preserves_rng_and_exact_official_parameter_names():
    before = torch.random.get_rng_state()
    net = refine._anime6b_net()
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(net.body) == 6
    assert net.body[0].rdb1.conv1.weight.shape == (32, 64, 3, 3)
    assert net.body[5].rdb3.conv5.weight.shape == (64, 192, 3, 3)
    assert set(net.state_dict()) >= {"conv_first.weight", "body.5.rdb3.conv5.weight", "conv_up2.weight", "conv_last.bias"}


def fake_net(monkeypatch, *, cancel_after=None):
    calls = []
    class Net(torch.nn.Module):
        def load_state_dict(self, state, strict):
            assert state == {} and strict is True
        def forward(self, tile):
            assert not torch.is_grad_enabled()
            calls.append(tuple(tile.shape))
            return torch.nn.functional.interpolate(tile, scale_factor=4, mode="nearest")
    monkeypatch.setattr(refine, "anime6b_weights", lambda *a: Path("fixture.pth"))
    def load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"params_ema": {}}
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(refine, "_anime6b_net", Net)
    return calls, lambda: cancel_after is not None and len(calls) >= cancel_after


def test_tiled_preprocess_crops_context_and_preserves_rgb_with_no_grad(monkeypatch):
    calls, cancel = fake_net(monkeypatch)
    pixels = np.arange(300 * 12 * 3, dtype=np.uint8).reshape((12, 300, 3))
    source = Image.fromarray(pixels)
    result = refine.anime6b_upscale(source, cancel, device="cpu")
    assert result.size == (1200, 48) and result.mode == "RGB"
    assert np.array_equal(np.asarray(result), np.asarray(source.resize(result.size, Image.Resampling.NEAREST)))
    assert len(calls) == 2 and all(max(shape[2:]) <= 276 for shape in calls)


def test_cancel_after_tile_discards_output_before_next_tile(monkeypatch):
    calls, cancel = fake_net(monkeypatch, cancel_after=1)
    with pytest.raises(Cancelled):
        refine.anime6b_upscale(Image.new("RGB", (300, 12)), cancel, device="cpu")
    assert len(calls) == 1


def test_gan_then_target_resize_runs_before_source_is_handed_to_the_vae(store, monkeypatch):
    from backend.schemas import GenerateRequest

    request = GenerateRequest(prompt="source", width=256, height=256)
    saved = store.save_image(Image.new("RGB", (256, 256), "red"), request, seed=1,
                             duration_ms=1, model_id="fixture", quantization="none")
    seen = []
    def upscale(source, cancel, *, device):
        seen.append((source.size, device))
        return Image.new("RGB", (1024, 1024), "blue")
    monkeypatch.setattr(refine, "anime6b_upscale", upscale)
    result = refine.source_image(saved.id, 384, 256, upscaler=refine.ANIME6B, device="cpu")
    assert seen == [((256, 256), "cpu")]
    assert result.size == (384, 256) and result.getpixel((0, 0)) == (0, 0, 255)


def test_post_load_guard_refuses_a_backend_that_drops_the_requested_upscaler(fake, monkeypatch):
    from backend import model_selection
    from backend.schemas import GenerateRequest

    model_selection.select("LyliaEngine/waiIllustriousSDXL_v170")
    request = GenerateRequest(prompt="x", refine=1, upscaler=refine.ANIME6B)
    monkeypatch.setattr(fake, "applied", lambda given: given.model_copy(update={"upscaler": None}))
    with pytest.raises(ValueError, match="cannot preserve"):
        fake.check_refine_supported(request)


def test_direct_backend_generation_refuses_unsupported_gan_before_normalizing(monkeypatch):
    from backend.backends.diffusers_backend import DiffusersBackend
    from backend.schemas import GenerateRequest

    backend = DiffusersBackend()
    backend._pipe = object()
    monkeypatch.setattr(backend, "describe", lambda: {"model_id": "cagliostrolab/animagine-xl-4.0"})
    monkeypatch.setattr(backend, "applied", lambda *a: pytest.fail("explicit unsupported choice was normalized"))
    with pytest.raises(ValueError, match="only for WAI"):
        backend.generate(GenerateRequest(prompt="x", refine=1, upscaler=refine.ANIME6B), seed=1)
