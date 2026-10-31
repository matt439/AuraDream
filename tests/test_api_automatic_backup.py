"""Default-off app wiring and actual verified exports over synthetic galleries."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import api_backup, api_bundles, backup_preferences, backup_receipts, backup_settings, bundles, main
from backend.backup_policy import BackupPolicy
from backend.jobs import JobManager
from backend.schemas import GenerateRequest
from tests.api_helpers import ORIGIN
from tests.backup_helpers import save_v1_policy
from tests.bundle_helpers import _destination, _save, _snapshot

PREFIX = api_backup.PREFIX


def body(status, **changes):
    chosen = dict(revision=status["revision"], enabled=True, minimum_age_hours=24,
                  keep_last=3, allow_unencrypted=True, destination=status["destination"])
    chosen.update(changes)
    return chosen


def configured(store):
    return backup_settings.configure(
        store.settings.db_path, revision="missing", enabled=True, directory=store.bundle_directory(),
        minimum_age_hours=24, keep_last=3, allow_unencrypted=True,
    )


def finish(client):
    until = time.monotonic() + 15
    while time.monotonic() < until:
        response = client.get(PREFIX)
        assert response.status_code == 200
        status = response.json()
        if status["state"] != "running" and (status["state"] != "succeeded" or status["verified_at"]):
            return status
        time.sleep(0.01)
    raise AssertionError("owned startup backup did not finish")


def reset_managers(fake, monkeypatch):
    monkeypatch.setattr(main, "jobs", JobManager(fake, main.ensure_loaded))
    monkeypatch.setattr(main, "bundle_manager", api_bundles.BundleManager(lambda: main.jobs, lambda: False))


def test_missing_settings_startup_is_unchanged_and_status_is_cached(client, store, monkeypatch):
    state = client.get(PREFIX).json()
    assert state["enabled"] is False and state["revision"] == "missing"
    assert state["minimum_age_hours"] is state["keep_last"] is state["protection"] is None
    assert state["retention_enabled"] is False and main.bundle_manager.state["state"] == "idle"
    assert not backup_preferences.path_for(store.settings.db_path).exists()
    assert not store.bundle_directory().exists()

    def unexpected(*_args, **_kwargs):
        pytest.fail("status must not read files or acquire locks")

    monkeypatch.setattr(backup_settings, "load", unexpected)
    monkeypatch.setattr(backup_receipts, "load", unexpected)
    monkeypatch.setattr(store, "bundle_directory", unexpected)
    main.bundle_manager.reserve("test-held-maintenance")
    assert client.get(PREFIX).json() == state
    main.bundle_manager.fail("finished synthetic hold")


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_settings_never_export_or_rewrite_on_app_start(fake, store, enabled, monkeypatch):
    path = save_v1_policy(store.settings.db_path, BackupPolicy(enabled, timedelta(hours=1), 2, "unencrypted"))
    before = path.read_bytes()
    monkeypatch.setattr(bundles, "export_bundle", lambda **_kwargs: pytest.fail("legacy settings never activate"))
    with TestClient(main.app, base_url=ORIGIN) as client:
        state = client.get(PREFIX).json()
        assert state["state"] == "needs_confirmation" and not state["enabled"]
        assert state["requires_confirmation"] and main.bundle_manager.task is None
    assert path.read_bytes() == before


def test_explicit_save_arms_next_start_only_and_disable_preserves_scope(client, store):
    _save("synthetic explicit configuration")
    before = _snapshot(store)
    state = client.get(PREFIX).json()
    for change in ({"allow_unencrypted": False}, {"minimum_age_hours": None}, {"keep_last": None}, {"enabled": 1},
                   {"protection": "encrypted"}, {"minimum_age_hours": True}):
        response = client.put(PREFIX, json=body(state, **change))
        assert response.status_code == 422, response.text
        assert not backup_preferences.path_for(store.settings.db_path).exists()
    response = client.put(PREFIX, json=body(state))
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["enabled"] and saved["state"] == "scheduled" and saved["reason"] == "next_start"
    assert not backup_receipts.path_for(store.settings.db_path).exists() and bundles.list_bundles() == []
    assert _snapshot(store) == before
    wire = json.loads(backup_preferences.path_for(store.settings.db_path).read_text())
    assert wire["version"] == 2 and wire["allow_unencrypted"] is True
    assert client.put(PREFIX, json=body(state)).status_code == 409
    assert client.put(PREFIX, json=body(saved, destination="stale destination")).status_code == 409
    response = client.put(PREFIX, json={"revision": saved["revision"], "enabled": False})
    assert response.status_code == 200 and not response.json()["enabled"]
    disabled = json.loads(backup_preferences.path_for(store.settings.db_path).read_text())
    assert disabled["gallery_id"] == wire["gallery_id"] and disabled["destination_id"] == wire["destination_id"]
    assert _snapshot(store) == before


def test_configuration_reserves_without_counting_its_own_request(client):
    state = client.get(PREFIX).json()
    main.bundle_manager.readers = 1
    try:
        assert client.put(PREFIX, json=body(state)).status_code == 409
        assert client.get(PREFIX).status_code == 200
    finally:
        main.bundle_manager.readers = 0
    assert client.put(PREFIX, json=body(state)).status_code == 200


def test_disable_does_not_require_a_usable_destination(client, store, monkeypatch):
    state = client.get(PREFIX).json()
    saved = client.put(PREFIX, json=body(state)).json()

    def invalid_directory():
        raise bundles.BundleError("synthetic invalid destination")

    monkeypatch.setattr(store, "bundle_directory", invalid_directory)
    response = client.put(PREFIX, json={"revision": saved["revision"], "enabled": False})
    assert response.status_code == 200 and not response.json()["enabled"]
    assert backup_settings.load(store.settings.db_path).settings.enabled is False


def test_explicit_reload_reads_external_settings_without_admission_or_export(client, store):
    cached = client.get(PREFIX).json()
    changed = configured(store)  # Another cooperating process published settings.
    assert client.get(PREFIX).json()["revision"] == cached["revision"]
    main.bundle_manager.reserve("synthetic-maintenance")
    try:
        reloaded = client.post(PREFIX + "/reload").json()
        assert reloaded["revision"] == changed.revision and reloaded["enabled"]
        assert main.bundle_manager.state["operation"] == "synthetic-maintenance"
    finally:
        main.bundle_manager.fail("synthetic maintenance ended")
    assert client.put(PREFIX, json={"revision": reloaded["revision"], "enabled": False}).status_code == 200
    assert bundles.list_bundles() == []


def test_cancelled_reload_keeps_its_guard_until_the_owned_reader_finishes(client, monkeypatch):
    service = main.backup_service
    entered, release = threading.Event(), threading.Event()
    original = service.prepare

    def held_reader():
        entered.set()
        assert release.wait(10)
        original()

    monkeypatch.setattr(service, "prepare", held_reader)
    router = api_backup.create_router(lambda: service)
    endpoint = next(route.endpoint for route in router.routes if route.path.endswith("/reload"))

    async def scenario():
        task = asyncio.create_task(endpoint())
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert service.reading and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not service.reading

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("payload", [
    {"password": "private-secret-value"}, {"protection": "private-secret-value"},
    {"enabled": "private-secret-value"}, {"minimum_age_hours": "private-secret-value"},
])
def test_rejected_payload_values_never_appear_in_validation_errors(client, payload):
    state = client.get(PREFIX).json()
    response = client.put(PREFIX, json=body(state, **payload))
    assert response.status_code == 422 and "private-secret-value" not in response.text
    assert "credentials" in response.json()["detail"]


def test_two_starts_export_once_then_skip_young_and_restore_normally(fake, store, monkeypatch, tmp_path):
    source = _save("synthetic first checkpoint", seed=1)
    store.set_favorite(source.id, True)
    child = _save("synthetic source child", seed=2, refine=source.id)
    store.bin_images([child.id])
    mask, _ = store.store_mask(Image.new("L", (256, 256), 128))
    masked = store.save_image(
        Image.new("RGB", (256, 256), (10, 20, 30)),
        GenerateRequest(prompt="synthetic other checkpoint inpaint", width=256, height=256,
                        refine=source.id, strength=0.6, mask=mask),
        seed=3, duration_ms=1, model_id="test/another-model", job_id="synthetic", run_id="synthetic",
    )
    before = _snapshot(store)
    activated = configured(store)
    manual = bundles.export_bundle(keep_gallery=True)
    manual_path = store.bundle_directory() / manual["name"]
    manual_bytes = manual_path.read_bytes()
    with TestClient(main.app, base_url=ORIGIN) as client:
        first = finish(client)
        assert first["state"] == "succeeded" and first["archive_name"] and first["verified_at"]
        assert client.get("/api/gallery/bundles/status").json()["operation"] == "automatic_backup"
    assert _snapshot(store) == before and manual_path.read_bytes() == manual_bytes
    receipts = backup_receipts.load(store.settings.db_path, activated.settings.gallery_id)
    assert len(receipts) == 1
    reset_managers(fake, monkeypatch)
    with TestClient(main.app, base_url=ORIGIN) as client:
        second = finish(client)
        assert second["state"] == "skipped" and second["reason"] == "young"
    assert _snapshot(store) == before and manual_path.read_bytes() == manual_bytes
    assert len(backup_receipts.load(store.settings.db_path, activated.settings.gallery_id)) == 1
    archive = store.bundle_directory() / first["archive_name"]
    _destination(store, monkeypatch, tmp_path)
    restored = bundles.import_bundle(archive)
    assert restored["imported"] == 3 and restored["binned"] == 1
    assert store.get_image(source.id).favorite
    assert store.get_image(child.id, include_binned=True).refine == source.id
    assert store.get_image(masked.id).model_id == "test/another-model"
    assert store.get_image(masked.id).mask == mask and store.mask_path(mask).is_file()


def test_changed_destination_does_not_implicitly_activate_or_adopt_old_set(fake, store, monkeypatch, tmp_path):
    activated = configured(store)
    _save()
    with TestClient(main.app, base_url=ORIGIN) as client:
        assert finish(client)["state"] == "succeeded"
    old_archives = {p.name: p.read_bytes() for p in store.bundle_directory().glob("*.zip")}
    reset_managers(fake, monkeypatch)
    monkeypatch.setattr(store, "settings", replace(store.settings, bundles_dir=tmp_path / "new-destination"))
    with TestClient(main.app, base_url=ORIGIN) as client:
        state = client.get(PREFIX).json()
        assert not state["enabled"] and state["reason"] == "destination_changed"
        assert not store.bundle_directory().exists()
        assert client.put(PREFIX, json=body(state)).status_code == 200
        changed = backup_settings.load(store.settings.db_path).settings
        assert changed.gallery_id == activated.settings.gallery_id
        assert changed.destination_id != activated.settings.destination_id
    assert not store.bundle_directory().exists()
    old_dir = tmp_path / "bundles"
    assert {p.name: p.read_bytes() for p in old_dir.glob("*.zip")} == old_archives


def test_shutdown_drains_admitted_backup_and_never_starts_waiting_preload(fake, store, monkeypatch):
    _save()
    configured(store)
    entered, release = threading.Event(), threading.Event()
    preloads = []
    real_export = bundles.export_bundle

    def export(**kwargs):
        entered.set()
        assert release.wait(15)
        return real_export(**kwargs)

    async def preload():
        preloads.append(True)

    monkeypatch.setattr(bundles, "export_bundle", export)
    monkeypatch.setattr(main, "settings", replace(main.settings, load_on_startup=True))
    monkeypatch.setattr(main, "_load_in_background", preload)

    async def scenario():
        life = main.lifespan(main.app)
        await life.__aenter__()
        assert await asyncio.to_thread(entered.wait, 10)
        closing = asyncio.create_task(life.__aexit__(None, None, None))
        await asyncio.sleep(0)
        assert main.bundle_manager.running and not closing.done() and preloads == []
        release.set()
        await asyncio.wait_for(closing, 15)
        assert preloads == [] and not main.bundle_manager.running

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("fails", [False, True])
def test_preload_waits_for_backup_success_or_failure(fake, store, monkeypatch, fails):
    _save()
    configured(store)
    entered, release, loaded = threading.Event(), threading.Event(), threading.Event()
    real_export = bundles.export_bundle

    def export(**kwargs):
        entered.set()
        assert release.wait(15)
        if fails:
            raise OSError("synthetic failure with private details")
        return real_export(**kwargs)

    async def preload():
        assert not main.bundle_manager.running
        loaded.set()

    monkeypatch.setattr(bundles, "export_bundle", export)
    monkeypatch.setattr(main, "settings", replace(main.settings, load_on_startup=True))
    monkeypatch.setattr(main, "_load_in_background", preload)
    with TestClient(main.app, base_url=ORIGIN) as client:
        assert entered.wait(10) and not loaded.is_set()
        assert client.get(PREFIX).status_code == 200
        release.set()
        state = finish(client)
        assert loaded.wait(10)
        assert state["state"] == ("failed" if fails else "succeeded")
        assert "private details" not in json.dumps(state)


def test_receipt_read_overlapping_configuration_cannot_publish_old_stamp_or_break_preload(fake, store, monkeypatch):
    _save()
    configured(store)
    entered, release, preloaded = threading.Event(), threading.Event(), threading.Event()
    original = backup_receipts.load

    def held_receipt(database, gallery_id):
        # The export's journal reads occur while admission is held. Pause the
        # later finish-time receipt read, after the writer has relinquished it.
        if not main.bundle_manager.running and not entered.is_set():
            entered.set()
            assert release.wait(15)
        return original(database, gallery_id)

    async def preload():
        preloaded.set()

    monkeypatch.setattr(backup_receipts, "load", held_receipt)
    monkeypatch.setattr(main, "settings", replace(main.settings, load_on_startup=True))
    monkeypatch.setattr(main, "_load_in_background", preload)
    try:
        with TestClient(main.app, base_url=ORIGIN) as client:
            assert entered.wait(10) and not main.bundle_manager.running
            before = client.get(PREFIX).json()
            assert before["state"] == "succeeded" and before["verified_at"] is None
            disabled = client.put(PREFIX, json={"revision": before["revision"], "enabled": False})
            assert disabled.status_code == 200 and not disabled.json()["enabled"]
            assert main.backup_service.coordinator is None
            release.set()
            assert preloaded.wait(10)
            final = client.get(PREFIX).json()
            assert not final["enabled"] and final["state"] == "disabled" and final["verified_at"] is None
            assert main._startup_task.done() and main._startup_task.exception() is None
    finally:
        release.set()
