"""The converted-tree cache contract the Anima, NetaYume and NewBie loaders share.

Each converts its pinned checkpoint once into an atomic, recipe-marked tree
under `cache_root()`. These tests hold all three to the same sweep, lock,
failure and refusal behaviour; each loader's own conversion, reuse and
pipeline tests stay in its `test_<name>_loading.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.backends import anima_loading, netayume_loading, newbie_loading

# Another checkpoint of the same family, which passes through untouched.
OTHERS = {
    anima_loading: "SeeSee21/Z-Anime",
    netayume_loading: "Alpha-VLLM/Lumina-Image-2.0",
    newbie_loading: "Alpha-VLLM/Lumina-Image-2.0",
}


@pytest.fixture(params=list(OTHERS), ids=lambda module: module.__name__.rsplit(".", 1)[-1])
def module(request):
    return request.param


@pytest.fixture
def conversion(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "cache_root", lambda: tmp_path)
    calls = []

    # Anima's conversion also takes the variant's model id.
    def convert(path, *args):
        calls.append(path)
        path.mkdir()
        (path / "model_index.json").write_text("{}")

    monkeypatch.setattr(module, "_convert", convert)
    return calls


def test_other_models_pass_through_and_nunchaku_refuses_before_conversion(module, conversion):
    other = OTHERS[module]
    assert module.resolve_model(other, "fp8") == other
    assert module.converted_tree(other) is None
    with pytest.raises(RuntimeError, match="not a Nunchaku Lite"):
        module.resolve_model(module.MODEL_ID, "nunchaku")
    assert not conversion


def test_failed_conversion_leaves_no_published_tree_and_retry_works(module, monkeypatch, conversion, tmp_path):
    convert = module._convert

    def fail(path, *args):
        convert(path, *args)
        raise RuntimeError("disk full")

    monkeypatch.setattr(module, "_convert", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        module.resolve_model(module.MODEL_ID, "none")
    assert module.converted_tree(module.MODEL_ID) is None
    assert not list(tmp_path.glob("*/model_index.json"))
    monkeypatch.setattr(module, "_convert", convert)
    assert Path(module.resolve_model(module.MODEL_ID, "none"), "model_index.json").is_file()
    assert len(conversion) == 2


def test_a_failed_conversion_reports_its_own_error_past_an_open_file(module, monkeypatch, conversion, tmp_path):
    # On Windows a file still open stops the work directory's cleanup: the
    # caller still hears why the conversion failed, and the next load sweeps.
    convert = module._convert
    held = []

    def fail(path, *args):
        convert(path, *args)
        held.append((path / "model_index.json").open("rb"))
        raise RuntimeError("disk full")

    monkeypatch.setattr(module, "_convert", fail)
    try:
        with pytest.raises(RuntimeError, match="disk full"):
            module.resolve_model(module.MODEL_ID, "none")
    finally:
        for handle in held:
            handle.close()
    monkeypatch.setattr(module, "_convert", convert)
    result = module.resolve_model(module.MODEL_ID, "none")
    assert [path.name for path in tmp_path.iterdir() if path.is_dir()] == [Path(result).name]
    assert len(conversion) == 2


def test_an_interrupted_conversions_work_directory_is_swept(module, conversion, tmp_path):
    target, _, _ = module._target()
    stale = tmp_path / f"{target.name}-k3j9_0q1"
    (stale / "tree" / "transformer").mkdir(parents=True)
    # Named like one but not one: a longer or shorter suffix and a copy of a
    # published tree.
    longer = tmp_path / f"{target.name}-k3j9_0q1x"
    (longer / "tree").mkdir(parents=True)
    shorter = tmp_path / f"{target.name}-old"
    (shorter / "tree").mkdir(parents=True)
    copy = tmp_path / f"{target.name}-backup01"
    copy.mkdir()
    (copy / "model_index.json").write_text("{}")
    first = module.resolve_model(module.MODEL_ID, "fp8")
    assert not stale.exists()
    assert (longer / "tree").is_dir() and (shorter / "tree").is_dir() and (copy / "model_index.json").is_file()
    # Once converted, the next load still sweeps and converts nothing.
    (stale / "tree").mkdir(parents=True)
    assert module.resolve_model(module.MODEL_ID, "fp8") == first
    assert not stale.exists()
    assert len(conversion) == 1


def test_the_sweep_waits_for_a_conversion_holding_the_lock(module, conversion, tmp_path):
    # Another process's conversion, mid-save: it holds the lock, and its work
    # directory is live until it lets go.
    import threading

    from huggingface_hub.utils import WeakFileLock

    target, _, _ = module._target()
    live = tmp_path / f"{target.name}-live0001"
    answer: list[str] = []
    with WeakFileLock(tmp_path / f"{target.name}.lock"):
        (live / "tree").mkdir(parents=True)
        worker = threading.Thread(target=lambda: answer.append(module.resolve_model(module.MODEL_ID, "fp8")))
        worker.start()
        worker.join(0.5)
        waited, kept = worker.is_alive(), live.is_dir()
    # Joined before any assert, so a failure never leaves it running into the next test.
    worker.join(10)
    assert waited and kept
    assert answer == [str(target)]
    assert not live.exists()


@pytest.mark.parametrize("marker_text", [None, "stale recipe"])
def test_incomplete_or_stale_cache_is_never_overwritten(module, conversion, marker_text):
    target, marker, _ = module._target()
    target.mkdir()
    (target / "model_index.json").write_text("{}")
    if marker_text is not None:
        marker.write_text(marker_text)
    assert module.converted_tree(module.MODEL_ID) is None
    with pytest.raises(RuntimeError, match="Incomplete converted model cache"):
        module.resolve_model(module.MODEL_ID, "fp8")
    # Nothing is written into it: no marker appears and none is replaced.
    assert (marker.read_text() if marker.exists() else None) == marker_text
    assert not conversion
