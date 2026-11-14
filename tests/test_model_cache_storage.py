"""Inventory real scratch Hub layouts, including shared references and old revisions."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from backend import model_cache_storage as cache, model_info

CURRENT = "1" * 40
OLD = "2" * 40
HELPER = "3" * 40


def write(root: Path, name: str, content: bytes) -> Path:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    write(hub, f"models--demo--shared/snapshots/{CURRENT}/shared.bin", b"a" * 11)
    write(hub, f"models--demo--shared/snapshots/{CURRENT}/extra.bin", b"b" * 7)
    write(hub, f"models--demo--shared/snapshots/{OLD}/old.bin", b"c" * 5)
    write(hub, "models--demo--shared/refs/main", CURRENT.encode())
    write(hub, f"models--helper--critic/snapshots/{HELPER}/critic.gguf", b"d" * 13)
    write(hub, "models--demo--shared/blobs/partial.incomplete", b"e" * 3)
    converted = tmp_path / "converted"
    write(converted, "a-ready/model.bin", b"f" * 17)
    write(converted, "old-recipe/model.bin", b"g" * 4)
    reference = {
        "models": {
            "model/a": {"kind": "converted", "sources": [{"source": "source", "files": ["shared.bin"]}]},
            "model/b": {"kind": "pipeline", "sources": [{"source": "source", "files": ["shared.bin", "extra.bin"]}]},
        },
        "sources": {"source": {"repo_id": "demo/shared", "commit": CURRENT, "revision": "main"}},
    }
    monkeypatch.setattr(model_info, "download_references", lambda: reference)
    monkeypatch.setattr(cache, "_catalogue_model_ids", lambda: list(reference["models"]))
    monkeypatch.setattr(model_info, "_conversion_target", lambda model: (converted / "a-ready", None, None))
    monkeypatch.setattr(cache, "hf_home_dir", lambda name: converted)
    return hub, converted, reference


def test_measured_bytes_share_files_without_doubling_totals_and_keep_old_revisions(inventory):
    hub, converted, _ = inventory
    before = {path: path.read_bytes() for root in (hub, converted) for path in root.rglob("*") if path.is_file()}
    result = cache.snapshot(hub)
    models = {item["model_id"]: item for item in result["models"]}
    assert models["model/a"] == {"model_id": "model/a", "entry": "model/a", "label": None, "variant": None,
                                  "reference_complete": True, "cached_bytes": 11,
                                  "revision_bytes": 11, "other_revision_bytes": 0, "shared_bytes": 11,
                                  "exclusive_bytes": 0, "converted_bytes": 17, "converted_directory": "a-ready"}
    assert models["model/b"]["cached_bytes"] == 18
    assert models["model/b"]["shared_bytes"] == 11
    assert models["model/b"]["exclusive_bytes"] == 7
    assert result["hub_bytes"] == 36
    assert result["unreferenced_bytes"] == 18  # old image revision plus the critic
    assert result["incomplete_bytes"] == 3
    assert result["converted_bytes"] == 21
    assert result["converted_unreferenced_bytes"] == 4
    assert result["free_bytes"] > 0
    repos = {item["repo_id"]: item for item in result["repositories"]}
    assert repos["helper/critic"]["revisions"][0]["catalogue_models"] == []
    revisions = {item["revision"]: item for item in repos["demo/shared"]["revisions"]}
    assert revisions[CURRENT]["catalogue_models"] == ["model/a", "model/b"]
    assert revisions[OLD]["catalogue_models"] == []
    assert revisions[OLD]["other_revision_of"] == []  # beside the recorded one: an old download
    assert result["warnings"] == []
    assert before == {path: path.read_bytes() for root in (hub, converted) for path in root.rglob("*") if path.is_file()}


def test_unknown_file_selection_never_claims_a_measured_model_total(inventory):
    hub, _, references = inventory
    references["models"]["model/a"]["sources"][0]["files"] = None
    result = cache.snapshot(hub)
    model = result["models"][0]
    assert model["reference_complete"] is False and model["cached_bytes"] is None
    assert model["revision_bytes"] == 18
    assert "not unused" in result["note"]


def test_a_catalogue_repository_cached_only_at_an_unrecorded_commit_is_unknown_not_zero(inventory):
    """GEN#71: a download from before the recorded commit, or after a later
    upstream one, is the checkpoint's at an unknown selection, never 0 bytes
    and never another app's repository."""
    hub, _, references = inventory
    references["sources"]["source"]["commit"] = "9" * 40
    result = cache.snapshot(hub)
    models = {item["model_id"]: item for item in result["models"]}
    for model in ("model/a", "model/b"):
        assert models[model]["cached_bytes"] is None and models[model]["revision_bytes"] == 0
        assert models[model]["other_revision_bytes"] == 23  # both cached revisions
        assert models[model]["reference_complete"] is True
    revisions = {item["revision"]: item for item in next(
        repo for repo in result["repositories"] if repo["repo_id"] == "demo/shared")["revisions"]}
    assert revisions[CURRENT]["other_revision_of"] == revisions[OLD]["other_revision_of"] == ["model/a", "model/b"]
    assert revisions[CURRENT]["catalogue_models"] == []
    assert result["unreferenced_bytes"] == 13  # the critic alone
    assert "other_revision_bytes" in result["note"]


def test_lite_rows_and_a_renamed_lite_repository_are_named_by_their_entry(inventory, monkeypatch):
    """GEN#72 stage 0: a Nunchaku lite id's row names Krea 2 Turbo and its
    format, and lite-infer's renamed repository (no recipe of its own) is
    the written id's other revision: never its own row, never unreferenced."""
    hub, _, references = inventory
    entry = model_info.catalogue_entry("krea/Krea-2-Turbo")
    assert entry is not None and entry.lite
    nvfp4, int4, alias = entry.lite_repo("nvfp4"), entry.lite_repo("int4"), entry.lite_alias("int4")
    assert nvfp4 and int4 and alias and alias != int4
    lite_commit, alias_commit = "6" * 40, "7" * 40
    for model in (nvfp4, int4):
        references["models"][model] = {"kind": "pipeline", "sources": [
            {"source": f"{model}@main", "files": ["tokenizer/tokenizer.json"]}]}
        references["sources"][f"{model}@main"] = {"repo_id": model, "commit": lite_commit, "revision": "main"}
    write(hub, f"models--{nvfp4.replace('/', '--')}/snapshots/{lite_commit}/tokenizer/tokenizer.json", b"t" * 19)
    write(hub, f"models--{alias.replace('/', '--')}/snapshots/{alias_commit}/tokenizer/tokenizer.json", b"r" * 23)
    # The alias at the int4 id's own recorded commit is still the alias's: its
    # file list is not the one the reference retains.
    write(hub, f"models--{alias.replace('/', '--')}/snapshots/{lite_commit}/model_index.json", b"m" * 2)
    monkeypatch.setattr(cache, "_catalogue_model_ids", lambda: [*references["models"], entry.repo])
    result = cache.snapshot(hub)
    models = {item["model_id"]: item for item in result["models"]}
    assert alias not in models
    assert {key: models[nvfp4][key] for key in ("entry", "label", "variant", "cached_bytes", "other_revision_bytes")} == {
        "entry": "krea/Krea-2-Turbo", "label": "Krea 2 Turbo", "variant": "nvfp4",
        "cached_bytes": 19, "other_revision_bytes": 0}
    assert {key: models[int4][key] for key in ("entry", "label", "variant", "cached_bytes", "revision_bytes", "other_revision_bytes")} == {
        "entry": "krea/Krea-2-Turbo", "label": "Krea 2 Turbo", "variant": "int4",
        "cached_bytes": None, "revision_bytes": 0, "other_revision_bytes": 25}
    assert {key: models[entry.repo][key] for key in ("entry", "label", "variant")} == {
        "entry": "krea/Krea-2-Turbo", "label": "Krea 2 Turbo", "variant": None}
    assert result["unreferenced_bytes"] == 18  # unchanged: old image revision plus the critic
    repos = {item["repo_id"]: item for item in result["repositories"]}
    assert all(revision["other_revision_of"] == [int4] and revision["catalogue_models"] == []
               and revision["unreferenced_bytes"] == 0 for revision in repos[alias]["revisions"])
    assert result["warnings"] == []


def test_missing_cache_is_zero_without_creating_it(inventory, tmp_path):
    missing = tmp_path / "absent" / "hub"
    result = cache.snapshot(missing)
    assert not missing.exists()
    assert result["hub_bytes"] == result["incomplete_bytes"] == 0
    assert result["repositories"] == []
    assert all(item["cached_bytes"] == 0 for item in result["models"])
    assert result["free_bytes"] > 0


def test_a_catalogue_entry_without_a_retained_reference_is_present_as_unknown(inventory, monkeypatch):
    hub, _, _ = inventory
    monkeypatch.setattr(cache, "_catalogue_model_ids", lambda: ["model/a", "model/b", "model/new"])
    result = cache.snapshot(hub)
    model = next(item for item in result["models"] if item["model_id"] == "model/new")
    assert model["cached_bytes"] is None and not model["reference_complete"]


def test_corrupt_repo_is_reported_without_leaking_raw_paths(inventory):
    hub, _, _ = inventory
    write(hub, "models--broken--repo/snapshots", b"not a directory")
    result = cache.snapshot(hub)
    assert result["warnings"] and all(str(hub) not in message for message in result["warnings"])
    assert result["hub_bytes"] == 36


def test_route_inspects_only_on_demand_on_a_daemon_thread(client, monkeypatch):
    calls = []

    def inspect():
        calls.append(threading.current_thread())
        return {"models": [], "read_only": True}

    monkeypatch.setattr(cache, "snapshot", inspect)
    assert calls == []
    response = client.get("/api/models/storage")
    assert response.status_code == 200 and response.json() == {"models": [], "read_only": True}
    assert len(calls) == 1 and calls[0].daemon
    assert calls[0].name == "model-cache-storage"
    assert client.delete("/api/models/storage").status_code == 405
    assert len(calls) == 1


def test_scan_reservation_is_nonblocking_and_recovers(client, monkeypatch):
    cache._scan_lock.acquire()
    try:
        response = client.get("/api/models/storage")
        assert response.status_code == 409 and "already running" in response.json()["detail"]
    finally:
        cache._scan_lock.release()
    monkeypatch.setattr(cache, "_snapshot", lambda _: {"models": []})
    assert client.get("/api/models/storage").status_code == 200


def test_route_read_failure_has_plain_words_and_no_filesystem_details(client, monkeypatch):
    def unreadable():
        raise OSError("private path and environment text")

    monkeypatch.setattr(cache, "snapshot", unreadable)
    response = client.get("/api/models/storage")
    assert response.status_code == 503
    assert "private" not in response.text and "drive" in response.json()["detail"]
