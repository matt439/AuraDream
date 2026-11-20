"""Saved-bundle views never enter live gallery maintenance or persist their secrets."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from backend import bundle_browse_archive, bundle_leases, bundle_views, bundles, main
from backend.api_bundle_browse import VIEW_HEADER
from backend.bundle_manager import BROWSE_PREFIX
from tests.bundle_helpers import bundle_snapshot, save_source


def test_browse_session_is_private_read_only_and_independent(client, store, monkeypatch):
    archived = save_source("archived fox", seed=201)
    store.set_favorite(archived.id, True)
    path = Path(bundles.export_bundle(1, "secret")["path"])
    save_source("current gallery", seed=202)
    before = bundle_snapshot(store)
    status = main.bundle_manager.status()
    wrong = client.post(BROWSE_PREFIX, json={"name": path.name, "password": "incorrect"})
    assert wrong.status_code == 422 and "password is incorrect" in wrong.json()["detail"]
    assert not bundle_leases.downloads.leased(path)
    invalid = client.post(BROWSE_PREFIX, json={"name": path.name, "password": {"private": "secret"}})
    assert invalid.status_code == 422 and "secret" not in invalid.text
    assert client.post(BROWSE_PREFIX, json={"name": "../chosen.zip"}).status_code == 422

    # An independent live-gallery maintenance operation cannot block a read
    # or strand its lease. No view request is counted against that operation.
    monkeypatch.setitem(main.bundle_manager.state, "state", "running")
    opened = client.post(BROWSE_PREFIX, json={"name": path.name, "password": "secret"})
    assert opened.status_code == 200, opened.text
    headers = {VIEW_HEADER: opened.json()["token"]}
    assert bundle_leases.downloads.leased(path)
    page = client.post(BROWSE_PREFIX + "/images", headers=headers, json={"starred": True, "prompt": "FOX"})
    assert page.status_code == 200 and page.json()["total"] == 1
    found = page.json()["images"][0]
    assert found["settings"]["seed"] == 201 and found["prompt"] == "archived fox"
    thumb = client.get(BROWSE_PREFIX + f"/thumb/{archived.id}", headers=headers)
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/webp"
    assert thumb.content.startswith(b"RIFF")
    assert client.get(BROWSE_PREFIX + "/thumb/9999", headers=headers).status_code == 404
    refused = client.post(BROWSE_PREFIX + "/images", json={})
    assert refused.status_code == 410 and refused.json()["code"] == "bundle_view_closed"
    bad_filter = client.post(BROWSE_PREFIX + "/images", headers=headers, json={"offset": "secret query"})
    assert bad_filter.status_code == 422 and "secret query" not in bad_filter.text
    monkeypatch.setitem(main.bundle_manager.state, "state", "idle")
    assert client.delete("/api/gallery/bundles/files/" + path.name).json()["code"] == "bundle_in_use"
    closed = client.delete(BROWSE_PREFIX, headers=headers)
    assert closed.status_code == 204
    assert client.get(BROWSE_PREFIX + f"/thumb/{archived.id}", headers=headers).status_code == 410
    assert not bundle_leases.downloads.leased(path)
    for response in [opened, page, thumb, closed, refused, invalid, bad_filter, wrong]:
        assert "no-store" in response.headers["cache-control"]
    assert status == main.bundle_manager.status()
    assert bundle_snapshot(store) == before


def test_expiry_releases_even_an_abandoned_view_and_capacity(monkeypatch):
    views = bundle_views.BundleViews()
    exited = threading.Event()

    @contextmanager
    def opened(*args, **kwargs):
        try:
            yield object()
        finally:
            exited.set()

    monkeypatch.setattr(bundle_browse_archive, "open_view", opened)
    monkeypatch.setattr(bundle_views, "MAX_VIEWS", 1)
    # One reader occupies the bounded slot until it is closed.
    views.open("test", "a.zip", None, views.reserve("test"))
    with views.read("test"):
        with pytest.raises(bundle_views.ViewsFull):
            views.reserve("second")
    views.close("test")
    assert exited.is_set()
    # No further client request is required for an abandoned slot to expire.
    views = bundle_views.BundleViews(idle_seconds=0.01)
    exited.clear()
    views.open("test", "a.zip", None, views.reserve("test"))
    assert exited.wait(10)
    with pytest.raises(bundle_views.ViewClosed):
        with views.read("test"):
            pytest.fail("expired entry can still be read")
    views.shutdown()


def test_shutdown_interrupts_pending_verification_and_drains_it(monkeypatch):
    views = bundle_views.BundleViews()
    started, checking, exited = threading.Event(), threading.Event(), threading.Event()

    @contextmanager
    def slow(name, password, *, cancelled):
        started.set()
        assert checking.wait(10)
        try:
            assert cancelled()
            raise bundles.BundleError("the bundle view was closed")
            yield  # pragma: no cover
        finally:
            exited.set()

    monkeypatch.setattr(bundle_browse_archive, "open_view", slow)
    with ThreadPoolExecutor(2) as pool:
        opening = pool.submit(views.open, "test", "a.zip", None, views.reserve("test"))
        assert started.wait(10)
        stopping = pool.submit(views.shutdown)
        try:
            assert views._stopping.wait(10)
        finally:
            checking.set()
        stopping.result(timeout=10)
        with pytest.raises(bundles.BundleError):
            opening.result(timeout=10)
    assert exited.is_set()


def test_caller_can_revoke_an_open_before_its_receipt_and_reuse_the_slot(client, monkeypatch):
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    token = "ab" * 32
    headers = {VIEW_HEADER: token}

    @contextmanager
    def held(name, password, *, cancelled):
        entered.set()
        try:
            assert release.wait(10)
            assert cancelled()
            raise bundles.BundleError("the bundle view was closed")
            yield  # pragma: no cover
        finally:
            cleaned.set()

    original = bundle_browse_archive.open_view
    monkeypatch.setattr(bundle_browse_archive, "open_view", held)
    with ThreadPoolExecutor(1) as pool:
        opening = pool.submit(client.post, BROWSE_PREFIX, headers=headers, json={"name": "a.zip"})
        try:
            assert entered.wait(10)
            collision = client.post(BROWSE_PREFIX, headers=headers, json={"name": "b.zip"})
            assert collision.status_code == 409 and "already open" in collision.json()["detail"]
            assert client.delete(BROWSE_PREFIX, headers=headers).status_code == 204
            assert client.post(BROWSE_PREFIX + "/images", headers=headers, json={}).status_code == 410
        finally:
            release.set()
        assert opening.result(timeout=10).status_code == 422
    assert cleaned.is_set()
    monkeypatch.setattr(bundle_browse_archive, "open_view", original)
    # Repeated failed opens and unopened DELETE requests retain no tokens.
    assert client.post(BROWSE_PREFIX, headers={VIEW_HEADER: "short"}, json={"name": "a.zip"}).status_code == 422
    for _ in range(3):
        assert client.post(BROWSE_PREFIX, headers=headers, json={"name": "missing.zip"}).status_code == 404
        assert client.delete(BROWSE_PREFIX, headers=headers).status_code == 204
    views = main.app.state.bundle_views
    assert views._slots == 0 and not views._opening and not views._views
