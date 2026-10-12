"""Browser bundle admission and background completion against a scratch gallery."""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time

import pytest
from fastapi import HTTPException

from backend import api_bundles, bundle_crypto, bundles, main
from tests.api_helpers import ORIGIN, run_one
from tests.bundle_helpers import _holding_child


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

    def held(expected, password=None, progress=None):
        progress("writing", 1, 3)
        entered.set()
        assert release.wait(5)
        return {"cleared": True, "deleted_ids": []}

    monkeypatch.setattr(bundles, "export_bundle", held)
    start = client.post("/api/gallery/bundles/export", json={"expected": 0})
    try:
        assert start.status_code == 202
        assert entered.wait(2)
        assert client.get("/api/status").status_code == 200
        status = client.get("/api/gallery/bundles/status").json()
        assert status["state"] == "running" and status["progress"] == {"phase": "writing", "done": 1, "total": 3}
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


@pytest.mark.parametrize("busy", ["card", "switching", "load-lock"])
def test_bundles_refuse_while_the_card_or_a_model_change_is_busy(client, monkeypatch, busy):
    if busy == "card":
        # Expand or the critic holding the gate through run_when_idle.
        monkeypatch.setattr(main.jobs, "card", lambda: ("critic", "ticket"))
    elif busy == "switching":
        monkeypatch.setattr(main, "_switching", True)
    else:
        monkeypatch.setattr(main._load_lock, "locked", lambda: True)
    assert client.post("/api/gallery/bundles/export", json={"expected": 0}).status_code == 409
    upload = client.post("/api/gallery/bundles/upload", content=b"PK",
                         headers={"content-type": "application/zip"})
    assert upload.status_code == 409
    assert main.bundle_manager.state["state"] == "idle"
    assert not list(main.storage.bundle_staging_dir().glob(".bundle-upload-*"))


def test_another_process_holding_the_gallery_fails_the_operation_readably(client, store):
    child = _holding_child(store.settings.db_path)
    try:
        response = client.post("/api/gallery/bundles/export", json={"expected": 0})
        assert response.status_code == 202, response.text
        result = _finish(client, response.json()["id"])
    finally:
        child.stdin.close()
        child.wait(timeout=15)
    assert result["state"] == "failed"
    assert result["error"] == "Another Local Diffusion process is using the gallery; try again when it finishes."


def test_a_password_without_a_utf8_form_is_refused_before_any_work(client, store):
    run_one(client, seed=41)
    exported = _finish(client, client.post("/api/gallery/bundles/export",
                                           json={"expected": 1, "password": "pass"}).json()["id"])
    name = exported["result"]["name"]
    header = base64.b64encode(client.get(f"/api/gallery/bundles/files/{name}").content[:64]).decode()
    before = main.bundle_manager.state
    # A lone surrogate survives JSON decoding (json.dumps escapes it) but has
    # no UTF-8 form: one fixed 422 on every route, never a 500 or codec text.
    bad = "a\ud800b"
    for route, body in [("password", {"header": header, "password": bad}),
                        ("export", {"expected": 0, "password": bad}),
                        ("import", {"name": name, "password": bad})]:
        response = client.post(f"/api/gallery/bundles/{route}", content=json.dumps(body),
                               headers={"content-type": "application/json"})
        assert response.status_code == 422, (route, response.text)
        assert response.json()["detail"] == "The bundle password contains characters that cannot be used."
    assert main.bundle_manager.state == before


def test_a_disconnected_upload_fails_readably_and_leaves_no_staging(fake, store):
    # TestClient sends a whole body before the app runs, so this drives the
    # ASGI app directly: one chunk, then the client goes away.
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/api/gallery/bundles/upload", "raw_path": b"/api/gallery/bundles/upload",
             "query_string": b"", "root_path": "", "client": ("127.0.0.1", 50000), "server": None,
             "headers": [(b"host", ORIGIN.split("//", 1)[1].encode()), (b"content-type", b"application/zip"),
                         (b"content-length", b"1000000")]}
    messages = [{"type": "http.request", "body": b"PK" * 1000, "more_body": True}, {"type": "http.disconnect"}]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(main.app(scope, receive, send))
    starts = [m["status"] for m in sent if m["type"] == "http.response.start"]
    assert starts and starts[0] != 202
    state = main.bundle_manager.state
    assert state["state"] == "failed" and state["error"] == api_bundles.UPLOAD_STOPPED
    assert not list(store.bundle_staging_dir().glob(".bundle-upload-*"))


def test_an_error_without_a_message_is_reported_readably():
    manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
    manager.reserve("import")
    manager.fail(asyncio.CancelledError())
    assert manager.state["error"].startswith("The gallery bundle operation stopped unexpectedly")


def test_a_worker_error_is_shown_as_a_sentence():
    # BundleError and CryptoError say lowercase clauses; the page shows the
    # status's text as it is.
    manager = api_bundles.BundleManager(lambda: main.jobs, lambda: False)
    for raised, shown in [
        (bundles.BundleError("the bundle is damaged: File is not a zip file"),
         "The bundle is damaged: File is not a zip file."),
        (bundle_crypto.CryptoError("the encrypted bundle is damaged or was changed"),
         "The encrypted bundle is damaged or was changed."),
        (api_bundles.UPLOAD_STOPPED, api_bundles.UPLOAD_STOPPED),
    ]:
        manager.reserve("import")
        manager.fail(raised)
        assert manager.state["error"] == shown


def test_a_refusal_answered_over_http_is_a_sentence_too(client, store):
    # The preview and Import refuse through main's BundleError handler; the
    # dialog shows its detail as it is, so it reads as the worker's would.
    (store.settings.outputs_dir / "x.png.part").write_bytes(b"")
    preview = client.get("/api/gallery/bundles/preview")
    assert preview.status_code == 422
    assert preview.json()["detail"] == "Finish or recover the incomplete image write first: x.png.part."
    named = client.post("/api/gallery/bundles/import", json={"name": "x.txt"})
    assert named.status_code == 422
    assert named.json()["detail"] == f"Choose a bundle filename ending in .zip or {bundle_crypto.SUFFIX}."


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


def test_password_bundle_export_check_and_upload(client, store):
    run_one(client, seed=31)
    response = client.post("/api/gallery/bundles/export", json={"expected": 1, "password": "sésame"})
    result = _finish(client, response.json()["id"])
    assert result["state"] == "done", result
    name = result["result"]["name"]
    assert name.endswith(".ldbundle")
    assert [b["encrypted"] for b in client.get("/api/gallery/bundles").json()["bundles"] if b["name"] == name] == [True]
    download = client.get(f"/api/gallery/bundles/files/{name}")
    assert download.headers["content-type"] == "application/octet-stream"

    header = base64.b64encode(download.content[:64]).decode()
    wrong = client.post("/api/gallery/bundles/password", json={"header": header, "password": "nope"})
    assert wrong.status_code == 400 and "nope" not in wrong.text
    assert client.post("/api/gallery/bundles/password", json={"header": header, "password": "sésame"}).json() == {"ok": True}

    def upload(password):
        headers = {"content-type": "application/octet-stream"}
        if password is not None:
            headers["X-Bundle-Password"] = base64.b64encode(password.encode()).decode()
        started = client.post("/api/gallery/bundles/upload", content=download.content, headers=headers)
        assert started.status_code == 202, started.text
        return _finish(client, started.json()["id"])

    refused = upload("nope")
    # The worker's refusal reads as the password check's does: a sentence.
    assert wrong.json()["detail"] == "The bundle password is incorrect."
    assert refused["state"] == "failed" and refused["error"] == "The bundle password is incorrect."
    assert upload(None)["state"] == "failed"
    assert store.list_images()[1] == 0
    assert upload("sésame")["result"]["imported"] == 1
    saved = client.post("/api/gallery/bundles/import", json={"name": name, "password": "sésame"})
    assert _finish(client, saved.json()["id"])["result"]["present"] == 1
    # Uploads are staged on the images' volume and removed afterwards.
    staging = store.bundle_staging_dir()
    assert staging.parent == store.settings.outputs_dir.resolve().parent
    assert not list(staging.iterdir())
    assert not list(store.settings.db_path.parent.glob(".bundle-*"))


def test_an_upload_in_many_batches_arrives_byte_for_byte(client, store, monkeypatch):
    run_one(client, seed=31)
    exported = _finish(client, client.post("/api/gallery/bundles/export", json={"expected": 1}).json()["id"])
    blob = client.get(f"/api/gallery/bundles/files/{exported['result']['name']}").content
    # Small batches: each write overlaps the next batch's arrival.
    monkeypatch.setattr(api_bundles, "_UPLOAD_BUFFER", 4096)
    written = []
    real = bundles.import_bundle
    monkeypatch.setattr(bundles, "import_bundle",
                        lambda path, *a: written.append(path.read_bytes()) or real(path, *a))
    pieces = (blob[i:i + 1000] for i in range(0, len(blob), 1000))
    response = client.post("/api/gallery/bundles/upload", content=pieces,
                           headers={"content-type": "application/zip"})
    assert response.status_code == 202, response.text
    merged = _finish(client, response.json()["id"])
    assert merged["state"] == "done", merged
    assert written == [blob] and merged["result"]["imported"] == 1


def test_the_preview_counts_the_recycle_bin_among_its_records(client, store):
    _, events = run_one(client, seed=19, num_images=2)
    first = events[-1]["images"][0]["id"]
    assert client.post("/api/images/bin", json={"ids": [first]}).status_code == 200
    preview = client.get("/api/gallery/bundles/preview").json()
    assert (preview["records"], preview["images"], preview["binned"]) == (2, 2, 1)
