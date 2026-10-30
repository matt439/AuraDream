"""Cold download facts require complete selected files, including every shard.

All caches here are tiny fake Hub trees; no model is loaded or downloaded.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import time

import pytest
from huggingface_hub import constants

from backend import main, model_info
from backend.backends import anima_loading, loading, netayume_loading, newbie_loading, single_file
from backend.launch import CHECKPOINTS, FORMATS


COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40


def _json(value):
    return json.dumps(value, sort_keys=True).encode()


@pytest.fixture
def cache_factory(tmp_path, monkeypatch):
    data = {"observed_at": "2026-10-18", "models": {}, "sources": {}}
    monkeypatch.setattr(model_info, "download_references", lambda: data)
    monkeypatch.setattr(constants, "HF_HOME", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    model_info.invalidate_cache_facts()

    def source(repo="example/model", revision="main", files=None, indexes=None):
        if files is None:
            index = {"_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"],
                     "text_encoder": ["transformers", "T5EncoderModel"],
                     "tokenizer": ["transformers", "T5TokenizerFast"]}
            shards = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
            files = {"model_index.json": _json(index), "transformer/config.json": b"{}",
                     "transformer/diffusion_pytorch_model.safetensors": b"transformer",
                     "text_encoder/config.json": b"{}", "text_encoder/model.safetensors.index.json": _json(shards),
                     "text_encoder/model-00001-of-00002.safetensors": b"shard-one",
                     "text_encoder/model-00002-of-00002.safetensors": b"shard-two",
                     "tokenizer/tokenizer_config.json": b"{}", "tokenizer/tokenizer.json": b"vocabulary",
                     "tokenizer/spiece.model": b"sentencepiece"}
            indexes = {"model_index.json": index, "text_encoder/model.safetensors.index.json": shards}
        key = f"{repo}@{revision}"
        data["sources"][key] = {"repo_id": repo, "revision": revision, "commit": COMMIT,
                                "files": {name: len(body) for name, body in files.items()}, "indexes": indexes or {}}
        snapshot = tmp_path / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / COMMIT

        def write(names=None, *, commit=COMMIT):
            target = snapshot.parent / commit
            for name in files if names is None else names:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[name])
            if revision == "main":
                ref = snapshot.parent.parent / "refs" / "main"
                ref.parent.mkdir(parents=True, exist_ok=True)
                ref.write_text(commit, encoding="utf-8")
            return target

        return {"source": key, "files": list(files)}, write, snapshot

    def pipeline(model_id="example/model", repo=None, revision="main"):
        selected, write, snapshot = source(repo or model_id, revision)
        data["models"][model_id] = {"kind": "pipeline", "sources": [selected]}
        return write, snapshot

    yield data, source, pipeline
    model_info.invalidate_cache_facts()


@pytest.mark.parametrize("missing", ["all", "weights", "last-shard", "tokenizer"])
def test_a_descriptor_or_index_does_not_prove_the_model_is_cached(cache_factory, missing):
    data, _, pipeline = cache_factory
    write, _ = pipeline()
    names = data["models"]["example/model"]["sources"][0]["files"]
    if missing != "all":
        omitted = (lambda name: name.endswith(".safetensors")) if missing == "weights" else (
            lambda name: name.endswith("00002-of-00002.safetensors")) if missing == "last-shard" else (
            lambda name: name == "tokenizer/spiece.model")
        write([name for name in names if not omitted(name)])
    facts = model_info.download_facts("example/model")
    assert facts["cache_state"] == ("absent" if missing == "all" else "partial")
    assert facts["cache"] != "On disk"
    assert "full" in facts["download_note"] and "not the amount still missing" in facts["download_note"]


def test_complete_shards_tokenizers_and_configs_are_checked_without_network(cache_factory, monkeypatch):
    _, _, pipeline = cache_factory
    write, _ = pipeline()
    write()

    def refuse(*args, **kwargs):
        raise AssertionError("cache facts never download or load")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", refuse)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    facts = model_info.download_facts("example/model")
    assert (facts["cache_state"], facts["cache"]) == ("complete", "On disk")


@pytest.mark.parametrize("damage", ["truncated-weight", "same-size-index-change"])
def test_incomplete_or_changed_files_are_not_reported_on_disk(cache_factory, damage):
    _, _, pipeline = cache_factory
    write, snapshot = pipeline()
    write()
    if damage == "truncated-weight":
        (snapshot / "transformer/diffusion_pytorch_model.safetensors").write_bytes(b"short")
    else:
        path = snapshot / "text_encoder/model.safetensors.index.json"
        path.write_bytes(path.read_bytes().replace(b'"a"', b'"z"'))
    assert model_info.download_facts("example/model")["cache_state"] == "partial"


def test_an_inventory_for_another_main_revision_is_unknown(cache_factory):
    _, _, pipeline = cache_factory
    write, _ = pipeline()
    write(commit=OTHER_COMMIT)
    assert model_info.download_facts("example/model")["cache_state"] == "unknown"


def test_a_pinned_external_component_is_required_at_its_selected_revision(cache_factory):
    data, source, pipeline = cache_factory
    write, _ = pipeline()
    write()
    external, external_write, _ = source("example/shared", revision=COMMIT)
    data["models"]["example/model"]["sources"].append(external)
    external_write(commit=OTHER_COMMIT)
    assert model_info.download_facts("example/model")["cache_state"] == "partial"
    external_write()
    model_info.invalidate_cache_facts()
    assert model_info.download_facts("example/model")["cache_state"] == "complete"


def test_nunchaku_checks_its_own_repository_and_all_tokenizer_assets(cache_factory):
    _, _, pipeline = cache_factory
    native_write, _ = pipeline("example/native")
    native_write()
    lite = "lite-infer/example-nunchaku-lite-nvfp4-bnb4-text-encoder"
    lite_write, _ = pipeline(lite)
    assert model_info.download_facts(lite)["cache_state"] == "absent"
    lite_write()
    model_info.invalidate_cache_facts()
    assert model_info.download_facts(lite)["cache_state"] == "complete"


def _converted(cache_factory, monkeypatch):
    data, source, _ = cache_factory
    one, write_one, _ = source("example/single", COMMIT, {"release.safetensors": b"checkpoint"})
    shared, write_shared, _ = source("example/config", COMMIT, {"tokenizer/tokenizer.json": b"vocab", "unet/config.json": b"{}"})
    output = Path(constants.HF_HOME) / "converted"
    marker = output / "local-diffusion-conversion.json"
    recipe = {"version": 1, "source": "pinned fixture"}
    recipe_text = json.dumps(recipe, sort_keys=True, indent=2) + "\n"
    index = {"_class_name": "StableDiffusionXLPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}
    files = {"local-diffusion-conversion.json": recipe_text.encode(), "model_index.json": _json(index),
             "unet/config.json": b"{}", "unet/diffusion_pytorch_model.safetensors": b"converted-weights",
             "tokenizer/tokenizer_config.json": b"{}", "tokenizer/tokenizer.json": b"converted-vocab"}
    data["models"]["example/converted"] = {"kind": "converted", "sources": [one, shared],
        "converted": {"recipe": recipe, "files": {name: len(body) for name, body in files.items()},
                      "indexes": {"model_index.json": index}}}
    monkeypatch.setattr(model_info, "_conversion_target", lambda model_id: (output, marker, recipe_text))

    def write_output(names=None):
        for name in files if names is None else names:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name])
    return write_one, write_shared, write_output, output


def test_a_single_file_alone_is_partial_but_all_inputs_are_cached_before_conversion(cache_factory, monkeypatch):
    write_one, write_shared, _, _ = _converted(cache_factory, monkeypatch)
    write_one()
    assert model_info.download_facts("example/converted")["cache_state"] == "partial"
    write_shared()
    model_info.invalidate_cache_facts()
    facts = model_info.download_facts("example/converted")
    assert facts["cache_state"] == "complete" and "still converts" in facts["cache_note"]


@pytest.mark.parametrize("missing", ["weights", "tokenizer", "marker"])
def test_a_conversion_marker_and_descriptor_do_not_prove_complete_output(cache_factory, monkeypatch, missing):
    _, _, write_output, output = _converted(cache_factory, monkeypatch)
    write_output()
    path = {"weights": "unet/diffusion_pytorch_model.safetensors", "tokenizer": "tokenizer/tokenizer.json",
            "marker": "local-diffusion-conversion.json"}[missing]
    (output / path).unlink()
    assert model_info.download_facts("example/converted")["cache_state"] == "partial"


def test_a_complete_conversion_needs_no_original_downloads(cache_factory, monkeypatch):
    _, _, write_output, _ = _converted(cache_factory, monkeypatch)
    write_output()
    assert model_info.download_facts("example/converted")["cache_state"] == "complete"


@pytest.mark.parametrize("output_exists", [False, True])
def test_a_conversion_without_output_sizes_uses_verified_source_files(cache_factory, monkeypatch, output_exists):
    data, _, _ = cache_factory
    write_one, write_shared, write_output, _ = _converted(cache_factory, monkeypatch)
    converted = data["models"]["example/converted"]["converted"]
    del converted["files"], converted["indexes"]
    if output_exists:
        write_output(["local-diffusion-conversion.json", "model_index.json"])
    first = model_info.download_facts("example/converted")
    assert first["cache_state"] == ("unknown" if output_exists else "absent")
    write_one()
    model_info.invalidate_cache_facts()
    assert model_info.download_facts("example/converted")["cache_state"] == ("unknown" if output_exists else "partial")
    write_shared()
    model_info.invalidate_cache_facts()
    facts = model_info.download_facts("example/converted")
    assert facts["cache_state"] == "complete" and "source files are cached" in facts["cache_note"]


def test_download_reference_can_record_its_own_observation_date(cache_factory):
    data, _, pipeline = cache_factory
    pipeline()
    data["models"]["example/model"]["observed_at"] = "2026-10-24"
    assert "checked 2026-10-24" in model_info.download_facts("example/model")["download_note"]


def test_a_changed_converter_recipe_is_unknown(cache_factory, monkeypatch):
    data, _, _ = cache_factory
    _, _, write_output, _ = _converted(cache_factory, monkeypatch)
    write_output()
    data["models"]["example/converted"]["converted"]["recipe"] = {"version": 99}
    assert model_info.download_facts("example/converted")["cache_state"] == "unknown"


def test_custom_or_unavailable_manifests_are_unknown(cache_factory):
    data, _, _ = cache_factory
    assert model_info.download_facts("unlisted/custom")["cache_state"] == "unknown"
    data["models"]["example/unavailable"] = {"kind": "pipeline", "sources": [{"source": "missing", "files": None}]}
    assert model_info.download_facts("example/unavailable")["download"] == "Unknown"
    assert model_info.download_facts("example/unavailable")["cache_state"] == "unknown"


@pytest.mark.parametrize("fail", [False, True])
def test_the_models_api_refreshes_facts_after_successful_and_failed_load_attempts(client, fake, cache_factory, monkeypatch, fail):
    _, _, pipeline = cache_factory
    model_id = main.current_model_id(main.settings)
    write, _ = pipeline(model_id)
    before = client.get("/api/models").json()
    assert before["details"][model_id]["cache_state"] == "absent" and fake.loads == 0
    original_load = fake.load
    fake.unload()  # the shared fake begins ready; this test needs an actual load attempt

    def load():
        if fail:
            write(["model_index.json"])
            raise RuntimeError("fixture stopped after a partial download")
        write()
        original_load()
    monkeypatch.setattr(fake, "load", load)
    response = client.post("/api/model/load")
    assert response.status_code == 200  # accepted; the daemon load finishes later
    deadline = time.monotonic() + 5
    while True:
        status = client.get("/api/status").json()
        if status["cache_revision"] > before["cache_revision"] and status["state"] == ("error" if fail else "ready"):
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.01)
    after = client.get("/api/models").json()
    assert after["cache_revision"] == before["cache_revision"] + 1
    assert after["details"][model_id]["cache_state"] == ("partial" if fail else "complete")
    assert client.get("/api/status").json()["cache_revision"] == after["cache_revision"]


def test_retained_references_cover_every_catalogue_and_lite_selection():
    # Source/selection integrity, independently of the tiny fake-cache fixtures.
    data = json.loads(Path(model_info.__file__).with_name("model-downloads.json").read_text(encoding="utf-8"))
    # The editor's pin is tests/test_instruction_edit_catalogue.py's.
    expected = {entry.repo for entry in CHECKPOINTS} | {
        entry.lite_repo(fmt) for entry in CHECKPOINTS if entry.lite for fmt, _ in FORMATS}
    assert set(data["models"]) == expected
    for model_id, recipe in data["models"].items():
        assert recipe["sources"]
        for selected in recipe["sources"]:
            source = data["sources"][selected["source"]]
            assert source["url"].startswith("https://huggingface.co/api/models/")
            assert len(source["commit"]) == 40
            if selected["files"] is None:
                assert recipe["unknown"]
                continue
            assert selected["files"] and len(set(selected["files"])) == len(selected["files"])
            for name in selected["files"]:
                assert model_info._relative_file(name) and source["files"][name] > 0
                # Converters can intentionally fetch an unused JSON index via
                # '*.json' while loading an explicit, different single file
                # (NewBie's Gemma). Pipeline-used shard maps require all shards.
                if recipe["kind"] == "pipeline" and name.endswith(".safetensors.index.json"):
                    shards = source["indexes"][name]["weight_map"].values()
                    assert {str(PurePosixPath(name).parent / shard) for shard in shards} <= set(selected["files"])
        if "ERNIE-Image-Turbo" in model_id:
            assert all(not name.startswith(("pe/", "pe_tokenizer/")) for selected in recipe["sources"] for name in selected["files"])


def test_retained_custom_loader_inputs_follow_the_actual_pins():
    data = json.loads(Path(model_info.__file__).with_name("model-downloads.json").read_text(encoding="utf-8"))
    expected = {
        single_file.ILLUSTRIOUS_ID: [single_file.ILLUSTRIOUS_SOURCE, single_file.SDXL_CONFIG],
        single_file.WAI_ID: [single_file.WAI_SOURCE, single_file.SDXL_CONFIG],
        single_file.ROUWEI_ID: [single_file.ROUWEI_SOURCE, single_file.SDXL_CONFIG],
        anima_loading.MODEL_ID: [anima_loading.SOURCE, anima_loading.SHARED],
        anima_loading.AESTHETIC_MODEL_ID: [anima_loading.AESTHETIC_SOURCE, anima_loading.SHARED],
        anima_loading.PREVIEW_MODEL_ID: [anima_loading.PREVIEW_SOURCE, anima_loading.SHARED],
        netayume_loading.MODEL_ID: [netayume_loading.SOURCE, netayume_loading.SHARED],
        newbie_loading.MODEL_ID: [newbie_loading.SOURCE],
        loading.NOOB_MODEL_ID: [{"repo_id": loading.NOOB_MODEL_ID, "revision": loading.NOOB_REVISION}],
        "SeeSee21/Z-Anime": [{"repo_id": "SeeSee21/Z-Anime", "revision": loading.NESTED_MODELS["SeeSee21/Z-Anime"][0]},
                             {"repo_id": loading.Z_ANIME_TEXT_ENCODER["pretrained_model_name_or_path"],
                              "revision": loading.Z_ANIME_TEXT_ENCODER["revision"]}],
    }
    for model_id, inputs in expected.items():
        selected = data["models"][model_id]["sources"]
        assert [item["source"] for item in selected] == [f"{item['repo_id']}@{item['revision']}" for item in inputs]
        for actual, source in zip(selected, inputs):
            if "filename" in source:
                assert source["filename"] in actual["files"]
