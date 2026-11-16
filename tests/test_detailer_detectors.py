"""Pinned downloads/cold status and geometry boundaries, without detector weights."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from backend import detailer_detectors as detectors
from backend.backends.base import Cancelled


@pytest.fixture
def small_asset(monkeypatch):
    body = b"pinned test detector bytes"
    asset = detectors.Asset("model.xml", "https://example.invalid/model.xml", len(body),
                            hashlib.sha256(body).hexdigest(), "test-revision")
    detector = detectors.Detector("anime-face-lbp-v1", "Test faces", "face", (asset,), "Test only", "test-recipe")
    monkeypatch.setattr(detectors, "DETECTORS", (detector,))
    return body, asset


def test_status_is_cold_and_does_not_create_cache(tmp_path, monkeypatch, small_asset):
    cache = tmp_path / "absent"
    monkeypatch.setattr(detectors, "_runtime_problem", lambda: "runtime missing")
    monkeypatch.setattr(detectors.urllib.request, "urlopen", lambda *a, **k: pytest.fail("status downloaded"))
    before = sys.modules.get("cv2")
    status = detectors.DetailerDetectors(cache).status()[0]
    assert status["available"] is False
    assert status["download_bytes"] == len(small_asset[0])
    assert status["reason"] == "runtime missing"
    assert not cache.exists()
    assert sys.modules.get("cv2") is before


def test_runtime_is_optional_and_single_distribution(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(detectors.importlib.metadata, "version", missing)
    assert "uv sync" in detectors._runtime_problem()
    monkeypatch.setattr(detectors.importlib.metadata, "version", lambda name: detectors.RUNTIME_VERSION)
    assert "one OpenCV" in detectors._runtime_problem()


def test_prepare_is_explicit_and_verified_atomic(tmp_path, monkeypatch, small_asset):
    body, asset = small_asset
    monkeypatch.setattr(detectors, "_runtime_problem", lambda: None)
    calls = []
    def answer(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(body)
    monkeypatch.setattr(detectors.urllib.request, "urlopen", answer)
    adapter = detectors.DetailerDetectors(tmp_path)
    adapter.prepare("anime-face-lbp-v1")
    assert (tmp_path / asset.name).read_bytes() == body
    assert not list(tmp_path.glob("*.part"))
    adapter.prepare("anime-face-lbp-v1")
    assert len(calls) == 1
    # Presence/size is status's cold promise; prepare detects same-size damage.
    (tmp_path / asset.name).write_bytes(b"x" * len(body))
    cached = adapter.status()[0]
    assert cached["available"] and cached["prepare_available"]
    assert cached["download_bytes"] == 0 and cached["asset_bytes"] == len(body)
    adapter.prepare("anime-face-lbp-v1")
    assert len(calls) == 2
    assert (tmp_path / asset.name).read_bytes() == body


@pytest.mark.parametrize("response", [b"short", b"x" * 25, b"much too long" * 20])
def test_bad_download_never_replaces_previous_asset(tmp_path, monkeypatch, small_asset, response):
    body, asset = small_asset
    (tmp_path / asset.name).write_bytes(b"old")
    monkeypatch.setattr(detectors, "_runtime_problem", lambda: None)
    monkeypatch.setattr(detectors.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(response))
    with pytest.raises(detectors.DetailerDetectorError):
        detectors.DetailerDetectors(tmp_path).prepare("anime-face-lbp-v1")
    assert (tmp_path / asset.name).read_bytes() == b"old"
    assert not list(tmp_path.glob("*.part"))


def test_cancellation_cleans_partial_download(tmp_path, monkeypatch, small_asset):
    body, asset = small_asset
    monkeypatch.setattr(detectors.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    with pytest.raises(Cancelled):
        detectors.DetailerDetectors._download(tmp_path / asset.name, asset, lambda: True)
    assert not (tmp_path / asset.name).exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_budget_is_failure(tmp_path, monkeypatch, small_asset):
    body, asset = small_asset
    times = iter([0.0, detectors.DOWNLOAD_BUDGET + 1.0])
    monkeypatch.setattr(detectors.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(detectors.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    with pytest.raises(detectors.DetailerDetectorError, match="timed out"):
        detectors.DetailerDetectors._download(tmp_path / asset.name, asset, lambda: False)
    assert not list(tmp_path.glob("*.part"))


def fake_cv(monkeypatch, boxes, failure=None):
    class Cascade:
        def empty(self):
            return False
        def detectMultiScale(self, *args, **kwargs):
            if failure:
                raise failure
            return boxes
    cv = SimpleNamespace(
        __version__="5.0.0", COLOR_RGB2BGR=1, COLOR_BGR2GRAY=2, INTER_LINEAR=1,
        cvtColor=lambda image, code: image,
        equalizeHist=lambda image: image,
        resize=lambda image, size, interpolation: np.zeros((size[1], size[0], 3), dtype=np.uint8),
        CascadeClassifier=lambda path: Cascade(),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv)


def test_detect_maps_canvas_and_orders_native_output(tmp_path, monkeypatch, small_asset):
    body, asset = small_asset
    (tmp_path / asset.name).write_bytes(body)
    monkeypatch.setattr(detectors, "_runtime_problem", lambda: None)
    fake_cv(monkeypatch, [(100, 300, 20, 30), (200, 10, 40, 20)])
    adapter = detectors.DetailerDetectors(tmp_path)
    regions, facts = adapter.detect("anime-face-lbp-v1", Image.new("RGB", (3072, 1536)))
    assert [region.box for region in regions] == [(400, 20, 480, 60), (200, 600, 240, 660)]
    assert facts[0].asset_sha256 == asset.sha256
    assert facts[0].provider == "CPU"
    assert "numpy=" in facts[0].preprocessing


def test_missing_or_changed_asset_never_imports_or_detects(tmp_path, monkeypatch, small_asset):
    monkeypatch.setattr(detectors, "_runtime_problem", lambda: None)
    with pytest.raises(detectors.DetailerDetectorError, match="missing or changed"):
        detectors.DetailerDetectors(tmp_path).detect("anime-face-lbp-v1", Image.new("RGB", (64, 64)))


@pytest.mark.parametrize("boxes", [[(0, 0, 10, 0)], [(float("nan"), 1, 10, 10)], [(100, 100, 10, 10)]])
def test_invalid_detector_output_is_error_not_empty(tmp_path, monkeypatch, small_asset, boxes):
    body, asset = small_asset
    (tmp_path / asset.name).write_bytes(body)
    monkeypatch.setattr(detectors, "_runtime_problem", lambda: None)
    fake_cv(monkeypatch, boxes)
    with pytest.raises(detectors.DetailerDetectorError):
        detectors.DetailerDetectors(tmp_path).detect("anime-face-lbp-v1", Image.new("RGB", (64, 64)))


def test_native_engine_failure_and_empty_success_differ(tmp_path, monkeypatch, small_asset):
    body, asset = small_asset
    (tmp_path / asset.name).write_bytes(body)
    monkeypatch.setattr(detectors, "_runtime_problem", lambda: None)
    adapter = detectors.DetailerDetectors(tmp_path)
    fake_cv(monkeypatch, [], RuntimeError("engine failed"))
    with pytest.raises(detectors.DetailerDetectorError, match="engine failed"):
        adapter.detect("anime-face-lbp-v1", Image.new("RGB", (64, 64)))
    fake_cv(monkeypatch, [])
    regions, facts = adapter.detect("anime-face-lbp-v1", Image.new("RGB", (64, 64)))
    assert regions == [] and len(facts) == 1
