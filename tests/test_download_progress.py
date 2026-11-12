"""Load status samples only selected cache files and preserves uncertain totals."""
from __future__ import annotations

import asyncio
import errno
import threading
import time

import httpx
import pytest

from backend import download_progress, main, model_info
from backend.error_text import download_failure, generation_error
from tests.api_helpers import wait_for_status
from tests.test_model_downloads import cache_factory, OTHER_COMMIT


@pytest.fixture(autouse=True)
def clear_samples():
    download_progress._sample.cache_clear()
    yield
    download_progress._sample.cache_clear()


def test_progress_counts_selected_files_and_repo_partials_without_other_snapshots(cache_factory):
    data, _, pipeline = cache_factory
    write, snapshot = pipeline()
    names = data["models"]["example/model"]["sources"][0]["files"]
    write(names[:-1])
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    (blobs / "active.incomplete").write_bytes(b"part")
    unrelated = snapshot.parent / OTHER_COMMIT
    unrelated.mkdir()
    (unrelated / "old-extra.bin").write_bytes(b"unrelated" * 100)
    result = download_progress.sample("example/model")
    source = data["sources"]["example/model@main"]
    total = sum(source["files"].values())
    assert result["total_bytes"] == total
    assert result["downloaded_bytes"] == min(total, total - source["files"][names[-1]] + 4)
    assert result["cache_state"] == "partial"
    assert result["cache_free_bytes"] > 0 and result["approximate"]
    assert model_info.download_reference_bytes("example/model") == total


def test_complete_cached_model_is_loading_weights_instead_of_downloading(cache_factory):
    _, _, pipeline = cache_factory
    write, _ = pipeline()
    write()
    result = download_progress.sample("example/model")
    assert result["cache_state"] == "complete"
    assert result["downloaded_bytes"] == result["total_bytes"]


def test_changed_mutable_revision_has_no_stale_exact_total(cache_factory):
    _, _, pipeline = cache_factory
    write, _ = pipeline()
    write(commit=OTHER_COMMIT)
    result = download_progress.sample("example/model")
    assert result["total_bytes"] is None and result["cache_state"] != "complete"


def test_missing_file_map_keeps_total_unknown_and_counts_only_source_blobs(cache_factory):
    data, _, pipeline = cache_factory
    _, snapshot = pipeline()
    data["models"]["example/model"]["sources"][0]["files"] = None
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "weight").write_bytes(b"cached")
    result = download_progress.sample("example/model")
    assert result["total_bytes"] is None and result["downloaded_bytes"] == 6
    assert result["cache_state"] == "partial"


def test_unknown_model_is_unknown_without_any_filesystem_creation(cache_factory):
    result = download_progress.sample("custom/unknown")
    assert result["total_bytes"] is None and result["cache_state"] == "unknown"


@pytest.mark.parametrize("failure", [OSError(errno.ENOSPC, "private filename"), TimeoutError("secret URL"), httpx.ConnectError("secret URL"), httpx.ReadTimeout("secret URL")])
def test_load_download_failures_are_plain_and_retain_retry_reuse(failure):
    wrapper = RuntimeError("loader wrapper")
    wrapper.__cause__ = failure
    detail, kind, url = main._load_failure(wrapper)
    assert kind in {"disk_full", "connection"} and url is None
    assert "Retry load" in detail and "reused" in detail
    assert "secret" not in detail and "private" not in detail
    assert generation_error(wrapper, loading=True) == detail
    assert generation_error(wrapper) == "RuntimeError: loader wrapper"


def test_words_alone_and_http_status_errors_are_not_classified_as_network():
    assert download_failure(RuntimeError("disk full and timeout")) is None
    assert download_failure(httpx.HTTPStatusError("not allowed", request=httpx.Request("GET", "https://example.invalid"), response=httpx.Response(403))) is None


def test_api_load_refusal_status_and_retry_agree(client, fake, monkeypatch):
    headers = {"Origin": "http://127.0.0.1:7860"}
    assert client.post("/api/model/unload", headers=headers).status_code == 200
    original_load = fake.load

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "secret cache path")

    monkeypatch.setattr(fake, "load", full_disk)
    response = client.post("/api/model/load", headers=headers)
    assert response.status_code == 200
    answer = wait_for_status(client, state="error")
    assert answer["state"] == "error" and answer["detail_kind"] == "disk_full"
    assert "secret" not in answer["detail"] and "cache drive" in answer["detail"]
    monkeypatch.setattr(fake, "load", original_load)
    assert client.post("/api/model/load", headers=headers).status_code == 200
    answer = wait_for_status(client, state="ready")
    assert answer["state"] == "ready" and answer["detail"] is None and answer["detail_kind"] is None


def test_status_samples_off_loop_and_retries_when_selection_changes(client, fake, monkeypatch):
    calls = []
    revision = [0]

    def sample(model):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            off_loop = True
        else:
            off_loop = False
        calls.append((model, off_loop, threading.current_thread().name))
        revision[0] = 1
        return {"downloaded_bytes": len(calls), "total_bytes": 9, "cache_state": "partial", "cache_free_bytes": 123, "approximate": True}

    monkeypatch.setattr(download_progress, "sample", sample)
    monkeypatch.setattr(main.model_selection, "revision", lambda: revision[0])
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["model_download"]["downloaded_bytes"] == 2
    assert response.json()["model_revision"] == 1
    assert len(calls) == 2 and all(off_loop for _, off_loop, _ in calls)
