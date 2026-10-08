"""Browser bundle admission and background completion against a scratch gallery."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

from backend import api_bundles, bundles, main
from tests.api_helpers import run_one


def _finish(client, operation_id):
    until = time.monotonic() + 10
    while time.monotonic() < until:
        state = client.get("/api/gallery/bundles/status").json()
        if state["id"] == operation_id and state["state"] != "running":
            return state
        time.sleep(0.01)
    raise AssertionError("bundle operation did not finish")


def test_browser_export_download_and_uploaded_merge(client, store):
    job_id, _ = run_one(client, seed=19)
    record = store.list_images()[0][0]
    store.set_favorite(record.id, True)
    preview = client.get("/api/gallery/bundles/preview").json()
    assert preview["records"] == preview["images"] == 1
    response = client.post("/api/gallery/bundles/export", json={"expected": 1})
    assert response.status_code == 202, response.text
    result = _finish(client, response.json()["id"])
    assert result["state"] == "done", result
    assert result["result"]["cleared"] is True
    assert store.list_images()[1] == 0
    assert client.get(f"/api/jobs/{job_id}").json()["images"] == []
    name = result["result"]["name"]
    listing = client.get("/api/gallery/bundles").json()
    assert any(item["name"] == name for item in listing["bundles"])
    download = client.get(f"/api/gallery/bundles/files/{name}")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert "attachment" in download.headers["content-disposition"]

    run_one(client, seed=20)
    existing = store.list_images()[0][0]
    response = client.post("/api/gallery/bundles/upload", content=download.content,
                           headers={"content-type": "application/zip"})
    assert response.status_code == 202, response.text
    merged = _finish(client, response.json()["id"])
    assert merged["state"] == "done", merged
    assert merged["result"]["imported"] == 1
    restored = {row.seed: row for row in store.list_images()[0]}
    assert restored[19].favorite and restored[19].id > existing.id
    assert restored[20] == existing
    duplicate = client.post("/api/gallery/bundles/import", json={"name": name})
    again = _finish(client, duplicate.json()["id"])
    assert again["result"]["imported"] == 0
    assert store.list_images()[1] == 2


def test_running_bundle_blocks_mutations_but_status_stays_available(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def held(expected):
        entered.set()
        assert release.wait(5)
        return {"cleared": True, "deleted_ids": []}

    monkeypatch.setattr(bundles, "export_bundle", held)
    start = client.post("/api/gallery/bundles/export", json={"expected": 0})
    try:
        assert start.status_code == 202
        assert entered.wait(2)
        assert client.get("/api/status").status_code == 200
        assert client.get("/api/gallery/bundles/status").json()["state"] == "running"
        assert client.get("/api/gallery").status_code == 409
        assert client.post("/api/generate", json={"prompt": "a fox"}).status_code == 409
        assert client.post("/api/images/1/favorite").status_code == 409
        assert client.post("/api/gallery/bundles/export", json={"expected": 0}).status_code == 409
    finally:
        release.set()
    assert _finish(client, start.json()["id"])["state"] == "done"
    assert client.get("/api/gallery").status_code == 200


def test_bundles_refuse_jobs_loading_and_foreign_origins(client, monkeypatch):
    response = client.post("/api/gallery/bundles/export", json={"expected": 0},
                           headers={"origin": "https://example.invalid"})
    assert response.status_code == 403
    monkeypatch.setattr(main.jobs, "busy", lambda: True)
    assert client.post("/api/gallery/bundles/export", json={"expected": 0}).status_code == 409
    monkeypatch.setattr(main.jobs, "busy", lambda: False)
    monkeypatch.setitem(main._load_state, "state", "loading")
    assert client.post("/api/gallery/bundles/export", json={"expected": 0}).status_code == 409
    assert main.bundle_manager.state["state"] == "idle"


def test_bad_upload_and_changed_preview_leave_gallery_unchanged(client, store):
    run_one(client)
    before = store.list_images()[0]
    for route, arguments in [
        ("export", {"json": {"expected": 0}}),
        ("upload", {"content": b"not an archive"}),
    ]:
        response = client.post(f"/api/gallery/bundles/{route}", **arguments)
        assert response.status_code == 202
        assert _finish(client, response.json()["id"])["state"] == "failed"
        assert store.list_images()[0] == before
    assert client.post("/api/gallery/bundles/import", json={"name": "missing.zip"}).status_code == 404
    assert client.post("/api/gallery/bundles/import", json={"name": "../outside.zip"}).status_code == 422


def test_admission_counts_an_image_until_its_stream_finishes():
    async def scenario():
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
        started, release = asyncio.Event(), asyncio.Event()

        async def streaming(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            started.set()
            await release.wait()
            await send({"type": "http.response.body", "body": b"image"})

        async def send(message):
            pass

        middleware = api_bundles.GalleryAdmission(streaming, lambda: manager)
        task = asyncio.create_task(middleware({"type": "http", "path": "/outputs/a.png", "method": "GET"}, None, send))
        await started.wait()
        with pytest.raises(HTTPException) as caught:
            manager.reserve("export")
        assert caught.value.status_code == 409
        assert manager.readers == 1
        release.set()
        await task
        assert manager.readers == 0

    asyncio.run(scenario())


def test_cancelling_supervisor_keeps_reservation_until_worker_finishes(tmp_path):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
        upload = tmp_path / "upload.zip"
        upload.write_bytes(b"still in use")

        def held():
            entered.set()
            assert release.wait(5)
            assert upload.exists()
            return {"imported": 1}

        manager.reserve("import")
        manager.start(held, cleanup=upload)
        while not entered.is_set():
            await asyncio.sleep(0.01)
        manager.task.cancel()
        await asyncio.sleep(0.01)
        assert manager.running and upload.exists()
        release.set()
        await manager.finish()
        assert manager.state["state"] == "done"
        assert not upload.exists()

    asyncio.run(scenario())
