"""`tools/model_downloads.py` (GEN#65): the offline check of
backend/model-downloads.json, and its refresh, here through a stand-in Hub.
Nothing reaches the network or downloads a file."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from backend import model_info
from backend.launch import CHECKPOINTS, FORMATS

ROOT = Path(__file__).resolve().parent.parent


def _tool():
    spec = importlib.util.spec_from_file_location("model_downloads", ROOT / "tools" / "model_downloads.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Its dataclass resolves its own module by name.
    sys.modules.setdefault("model_downloads", module)
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _data() -> dict:
    return json.loads(tool.DATA.read_text(encoding="utf-8"))


def test_the_retained_file_passes_its_check_with_every_size_known(capsys):
    data = _data()
    assert tool.check(data) == []
    assert tool.unknown_sizes(data) == []
    assert tool.main(["--check"]) == 0
    assert "0 problem(s), 0 unknown size(s)" in capsys.readouterr().out
    # What Models shows: a size for every id the catalogue writes.
    for entry in CHECKPOINTS:
        for model_id in [entry.repo, *(entry.lite_repo(fmt) for fmt, _ in FORMATS if entry.lite)]:
            assert model_info.download_reference_bytes(model_id), model_id


def test_the_file_keeps_its_own_layout():
    text = tool.DATA.read_text(encoding="utf-8")
    assert tool.dumps(json.loads(text)) == text.replace("\r\n", "\n")


def test_lite_sources_keep_the_ids_the_app_asks_for_and_record_the_rename():
    data = _data()
    for entry in CHECKPOINTS:
        for fmt, _ in FORMATS:
            if not entry.lite:
                continue
            [item] = data["models"][entry.lite_repo(fmt)]["sources"]
            source = data["sources"][item["source"]]
            assert source["repo_id"] == entry.lite_repo(fmt)
            assert source["moved_to"] == entry.lite_alias(fmt)
            assert entry.lite_alias(fmt) not in data["models"]


def test_selections_follow_diffusers_and_flux3s_own_rules():
    data = _data()
    for model_id in ("black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-dev", "krea/Krea-2-Raw",
                     "black-forest-labs/FLUX.2-klein-9B"):
        [item] = data["models"][model_id]["sources"]
        source = data["sources"][item["source"]]
        derived = tool.diffusers_selection(source["repo_id"], list(source["files"]), source["indexes"]["model_index.json"])
        assert derived == sorted(item["files"]), model_id
    for model_id in ("black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein-base"):
        [item] = data["models"][model_id]["sources"]
        selected = item["files"]
        assert selected == tool.flux3_selection(list(data["sources"][item["source"]]["files"]))
        assert "modular_model_index.json" in selected and "model.safetensors" not in selected
        assert not any(name.startswith("variants/") for name in selected)


def _break(data, how):
    krea = data["models"]["black-forest-labs/FLUX.1-Krea-dev"]
    source = data["sources"][krea["sources"][0]["source"]]
    if how == "null without a note":
        krea["sources"][0]["files"] = None
    elif how == "stale note":
        krea["unknown"] = "was unreadable"
    elif how == "missing shard":
        krea["sources"][0]["files"].remove("transformer/diffusion_pytorch_model-00002-of-00003.safetensors")
    elif how == "descriptor without contents":
        del source["indexes"]["transformer/diffusion_pytorch_model.safetensors.index.json"]
    elif how == "alias as a key":
        entry = next(entry for entry in CHECKPOINTS if entry.lite)
        data["models"][entry.lite_alias("int4")] = data["models"].pop(entry.lite_repo("int4"))
    elif how == "orphan source":
        data["sources"]["example/orphan@main"] = copy.deepcopy(source) | {"repo_id": "example/orphan"}
    elif how == "pinned commit":
        pinned = data["sources"][data["models"]["SeeSee21/Z-Anime"]["sources"][0]["source"]]
        pinned["commit"] = "f" * 40
    elif how == "flux3 selection":
        data["models"]["black-forest-labs/flux-3-image-klein"]["sources"][0]["files"].append("model.safetensors")
    elif how == "unselected file":
        krea["sources"][0]["files"].append("not/there.json")


@pytest.mark.parametrize("how", ["null without a note", "stale note", "missing shard", "descriptor without contents",
                                 "alias as a key", "orphan source", "pinned commit", "flux3 selection",
                                 "unselected file"])
def test_the_check_names_each_kind_of_damage(how):
    data = _data()
    _break(data, how)
    assert tool.check(data), how


INDEX = {"_class_name": "FluxPipeline", "_diffusers_version": "0.30.0",
         "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
         "transformer": ["diffusers", "FluxTransformer2DModel"], "vae": ["diffusers", "AutoencoderKL"]}
SHARDS = {"metadata": {}, "weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                                          "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}}
LISTED = {"README.md": 10, "flux1.safetensors": 900, "model_index.json": 300,
          "scheduler/scheduler_config.json": 20, "transformer/config.json": 30,
          "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 500,
          "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 400,
          "transformer/diffusion_pytorch_model.safetensors.index.json": 60,
          "vae/config.json": 40, "vae/diffusion_pytorch_model.safetensors": 70}


class StandIn:
    """The Hub's metadata answers, by repository; a descriptor read is logged."""

    def __init__(self, listings, refused=()):
        self.listings, self.refused, self.read = listings, set(refused), []

    def listing(self, repo_id, revision):
        if repo_id in self.refused:
            response = httpx.Response(403, request=httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}"))
            raise GatedRepoError("403 Client Error.", response=response)
        return self.listings[repo_id]

    def descriptor(self, repo_id, commit, name):
        self.read.append(name)
        assert name.endswith(".json"), "only descriptors are read"
        return {"model_index.json": INDEX,
                "transformer/diffusion_pytorch_model.safetensors.index.json": SHARDS}[name]


def _small(*, files=None, revision="main", commit="a" * 40):
    key = f"example/model@{revision}"
    return {
        "observed_at": "2026-10-18", "selection_basis": ["x"],
        "models": {"example/model": {"kind": "pipeline", "unknown": "not readable",
                                     "sources": [{"source": key, "files": files}]}},
        "sources": {key: {"repo_id": "example/model", "revision": revision, "commit": commit,
                          "files": dict(LISTED), "indexes": {},
                          "url": "https://huggingface.co/api/models/example/model?blobs=true"}},
    }


def test_a_refresh_derives_an_unknown_selection_and_reads_only_descriptors():
    hub = StandIn({"example/model": tool.Listing("example/model", "b" * 40, dict(LISTED))})
    fresh, report = tool.refresh(_small(), hub, today="2026-11-12")
    recipe = fresh["models"]["example/model"]
    assert "unknown" not in recipe and recipe["observed_at"] == "2026-11-12"
    assert recipe["sources"][0]["files"] == sorted(set(LISTED) - {"README.md", "flux1.safetensors"})
    source = fresh["sources"]["example/model@main"]
    assert source["commit"] == "b" * 40 and "moved_to" not in source
    assert set(source["indexes"]) == {"model_index.json", "transformer/diffusion_pytorch_model.safetensors.index.json"}
    assert set(hub.read) == set(source["indexes"])
    assert any("files selected" in line for line in report)


def test_a_renamed_repository_keeps_its_id_and_records_where_it_moved():
    data = _small(files=sorted(set(LISTED) - {"README.md", "flux1.safetensors"}))
    del data["models"]["example/model"]["unknown"]
    data["sources"]["example/model@main"]["indexes"] = {
        "model_index.json": INDEX, "transformer/diffusion_pytorch_model.safetensors.index.json": SHARDS}
    hub = StandIn({"example/model": tool.Listing("example/model-renamed", "a" * 40, dict(LISTED))})
    fresh, report = tool.refresh(data, hub, today="2026-11-12")
    source = fresh["sources"]["example/model@main"]
    assert (source["repo_id"], source["moved_to"]) == ("example/model", "example/model-renamed")
    assert fresh["models"]["example/model"]["sources"] == data["models"]["example/model"]["sources"]
    assert hub.read == []  # the same commit: its descriptors stand
    assert tool._source_problems("example/model@main", source) == []


def test_an_unchanged_partial_record_is_left_alone():
    # Kontext's source lists only its component files, read from the cache.
    data = _small(files=["model_index.json"])
    del data["models"]["example/model"]["unknown"]
    data["sources"]["example/model@main"]["files"] = {"model_index.json": 300}
    data["sources"]["example/model@main"]["indexes"] = {"model_index.json": INDEX}
    hub = StandIn({"example/model": tool.Listing("example/model", "a" * 40, dict(LISTED))})
    fresh, report = tool.refresh(data, hub, today="2026-11-12")
    assert fresh == data and report == []


def test_an_unreadable_repository_stays_unknown_with_its_reason():
    hub = StandIn({}, refused={"example/model"})
    fresh, report = tool.refresh(_small(), hub, today="2026-11-12")
    recipe = fresh["models"]["example/model"]
    assert recipe["sources"][0]["files"] is None
    assert "2026-11-12" in recipe["unknown"] and "GatedRepoError 403" in recipe["unknown"]
    assert any("not readable" in line for line in report)


def test_a_pinned_revision_must_still_be_its_commit():
    pin = "c" * 40
    hub = StandIn({"example/model": tool.Listing("example/model", "d" * 40, dict(LISTED))})
    with pytest.raises(SystemExit, match="pinned"):
        tool.refresh(_small(revision=pin, commit=pin), hub, today="2026-11-12")


def test_a_selected_file_that_is_gone_stops_the_refresh():
    data = _small(files=["model_index.json", "vae/gone.safetensors"])
    hub = StandIn({"example/model": tool.Listing("example/model", "b" * 40, dict(LISTED))})
    with pytest.raises(SystemExit, match="no longer has"):
        tool.refresh(data, hub, today="2026-11-12")
