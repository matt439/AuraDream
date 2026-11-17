"""Optional writing models keep exact source pins and share verified downloads."""
from __future__ import annotations

import dataclasses
import hashlib
import re

import huggingface_hub
import pytest

from backend import assistant_models, critic_runtime


def test_catalogue_models_are_immutable_and_keep_the_critic_default():
    assert [model.key for model in assistant_models.MODELS] == ["qwen3-8b", "qwen3.5-4b"]
    assert critic_runtime.MODEL["repo_id"] == "Qwen/Qwen3-4B-GGUF"
    assert critic_runtime.MODEL_SIZE == 4280404704
    for model in assistant_models.MODELS:
        assert assistant_models.get_model(model.key) is model
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision)
        assert re.fullmatch(r"[0-9a-f]{64}", model.sha256)
        assert model.context == 8192
        assert not model.filename.startswith(("mmproj-", "mtp-"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.label = "changed"
        source = model.source
        source["repo_id"] = "other/weights"
        assert model.source["repo_id"] == model.repo_id
        sampling = model.sampling
        sampling["temperature"] = 0
        assert model.sampling["temperature"] > 0


def test_profiles_use_each_authors_separate_recipe():
    qwen3 = assistant_models.get_model("qwen3-8b")
    qwen35 = assistant_models.get_model("qwen3.5-4b")
    assert (qwen3.repo_id, qwen3.size) == ("Qwen/Qwen3-8B-GGUF", 5027783488)
    assert (qwen35.repo_id, qwen35.size) == ("ggml-org/Qwen3.5-4B-GGUF", 4482403264)
    assert (qwen3.enable_thinking, qwen3.reasoning_budget) == (True, 2048)
    assert (qwen35.enable_thinking, qwen35.reasoning_budget) == (False, 0)
    assert qwen3.sampling == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repeat_penalty": 1.0,
    }
    assert qwen35.sampling == {**qwen3.sampling, "temperature": 0.7, "top_p": 0.8}


@pytest.mark.parametrize("key", ["default", "../mine.gguf", "Qwen/other", "QWEN3-8B", ""])
def test_unknown_model_keys_never_become_paths(key):
    with pytest.raises(ValueError, match="unknown assistant model"):
        assistant_models.get_model(key)


@pytest.mark.parametrize("model", assistant_models.MODELS)
def test_cached_checks_only_the_selected_revision(monkeypatch, tmp_path, model):
    cached = tmp_path / model.filename
    cached.write_bytes(b"not hashed by a status read")
    checked = []

    def inspect(repo_id, filename, *, revision):
        checked.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", inspect)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: pytest.fail("downloaded"))
    monkeypatch.setattr(critic_runtime, "_sha256", lambda path: pytest.fail("hashed"))
    assert model.cached() == cached
    assert checked == [model.source]
    cached.unlink()
    assert model.cached() is None


@pytest.mark.parametrize("model", assistant_models.MODELS)
def test_download_verifies_the_chosen_pin_once_without_default_leakage(monkeypatch, tmp_path, model):
    payload = model.key.encode()
    gguf = tmp_path / model.filename
    gguf.write_bytes(payload)
    selected = dataclasses.replace(model, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    calls = []
    hashes = []
    real_hash = critic_runtime._sha256
    monkeypatch.setattr(critic_runtime, "install_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(critic_runtime, "_sha256", lambda path: hashes.append(path) or real_hash(path))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None)

    def fetch(**kwargs):
        calls.append(kwargs)
        return str(gguf)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    assert selected.download() == gguf
    assert selected.download() == gguf
    assert calls == [selected.source, selected.source]
    assert hashes == [gguf]
    assert (tmp_path / "runtime" / "verified" / f"{selected.sha256}.json").is_file()
    assert not (tmp_path / "runtime" / "verified" / f"{critic_runtime.MODEL_SHA256}.json").exists()


def test_a_bad_optional_model_is_refetched_once_at_its_own_revision(monkeypatch, tmp_path):
    model = assistant_models.MODELS[1]
    selected = dataclasses.replace(model, size=4, sha256=hashlib.sha256(b"good").hexdigest())
    gguf = tmp_path / model.filename
    gguf.write_bytes(b"bad!")
    calls = []
    monkeypatch.setattr(critic_runtime, "install_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: str(gguf))

    def fetch(**kwargs):
        calls.append(kwargs)
        if kwargs.get("force_download"):
            assert not gguf.exists()
            gguf.write_bytes(b"good")
        return str(gguf)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    assert selected.download() == gguf
    assert gguf.read_bytes() == b"good"
    assert calls == [selected.source, {**selected.source, "force_download": True}]


def test_each_models_stamp_holds_to_its_own_digest(monkeypatch, tmp_path):
    shared = tmp_path / "shared.gguf"
    shared.write_bytes(b"same")
    good_digest = hashlib.sha256(b"same").hexdigest()
    monkeypatch.setattr(critic_runtime, "install_root", lambda: tmp_path / "runtime")
    critic_runtime.verify_model(shared, size=4, sha256=good_digest)
    with pytest.raises(critic_runtime.CriticError, match="sha256"):
        critic_runtime.verify_model(shared, size=4, sha256=hashlib.sha256(b"else").hexdigest())
    with pytest.raises(critic_runtime.CriticError, match="bytes"):
        critic_runtime.verify_model(shared, size=5, sha256=good_digest)
