"""Restore one selected archived image through its private browsing view."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from backend import bundle_leases, bundles, storage
from backend.api_bundle_browse import VIEW_HEADER
from backend.bundle_manager import BROWSE_PREFIX, RESTORE_PATH
from tests.bundle_helpers import bundle_snapshot, png_metadata, save_source, use_destination
from tests.test_api_bundles import _finish
from tests.test_inpaint_bundles import _inpaint_gallery


def _open(client, path, password=None):
    response = client.post(BROWSE_PREFIX, json={"name": path.name, "password": password})
    assert response.status_code == 200, response.text
    return {VIEW_HEADER: response.json()["token"]}


def _selected(client, headers, image_id, include=()):
    response = client.post(RESTORE_PATH, headers=headers, json={"image_id": image_id, "include_ids": list(include)})
    assert response.status_code == 202, response.text
    assert response.json()["operation"] == "restore"
    return response.json()["id"]


@pytest.mark.parametrize("password", [None, "private password"])
def test_restore_keeper_and_mask_remaps_the_collision_and_recognizes_a_repeat(
    client, store, tmp_path, monkeypatch, password,
):
    source, keeper, digest, unused = _inpaint_gallery()
    unrelated = save_source("unrelated archive record", 20)
    # A second keeper proves that the source can be reused without being
    # selected a second time once the destination contains the same image.
    second = save_source("second keeper", 21, reference=[source.id])
    path = Path(bundles.export_bundle(password=password, keep_gallery=True)["path"])
    use_destination(store, monkeypatch, tmp_path)
    collision = save_source("unrelated live image", 90)
    assert collision.id == source.id
    headers = _open(client, path, password)
    before = bundle_snapshot(store)

    request = {"image_id": keeper.id}
    assert client.post(RESTORE_PATH + "/preview", json=request).status_code == 410
    preview = client.post(RESTORE_PATH + "/preview", headers=headers, json=request)
    assert "no-store" in preview.headers["cache-control"]
    dependencies = preview.json()["dependencies"]
    assert [(item["kind"], item["id"], item["status"]) for item in dependencies] == [
        ("image", source.id, "available"), ("mask", digest, "available"),
    ]
    refused = _finish(client, _selected(client, headers, keeper.id, [unrelated.id]))
    assert refused["state"] == "failed" and "only replay dependencies" in refused["error"].lower()
    assert bundle_snapshot(store) == before

    outcome = _finish(client, _selected(client, headers, keeper.id, [source.id]))
    assert outcome["state"] == "done", outcome
    result = outcome["result"]
    assert result["imported"] == 2 and not result["replay_unavailable"]
    assert "private password" not in str(outcome) and path.name not in str(outcome)
    restored = store.get_image(result["image_id"])
    assert restored.mask == digest
    local_source = store.get_image(restored.refine)
    assert local_source.id != collision.id and local_source.filename == source.filename
    assert png_metadata(store.settings.outputs_dir / restored.filename)["refine"] == local_source.id
    assert store.mask_path(digest).is_file() and unused not in store._tracked_masks()
    assert {row.prompt for row in store.list_images()[0]} == {keeper.prompt, source.prompt, collision.prompt}
    assert not (store.settings.outputs_dir / unrelated.filename).exists()

    repeated = _finish(client, _selected(client, headers, keeper.id, [source.id]))["result"]
    assert repeated["imported"] == 0 and repeated["present"] == 2 and repeated["image_id"] == restored.id
    preview = client.post(RESTORE_PATH + "/preview", headers=headers, json={"image_id": second.id}).json()
    assert preview["dependencies"][0]["status"] == "present"
    result = _finish(client, _selected(client, headers, second.id))["result"]
    assert result["present"] == 1 and result["imported"] == 1 and not result["replay_unavailable"]
    assert store.get_image(result["image_id"]).reference == [local_source.id]
    store.bin_images([restored.id])
    binned = _finish(client, _selected(client, headers, keeper.id))["result"]
    assert binned["image_id"] == restored.id and binned["image_binned"] and binned["imported"] == 0


def test_missing_archive_source_does_not_bind_to_a_colliding_live_id(client, store, tmp_path, monkeypatch):
    source = save_source()
    keeper = save_source("keeper without its source", 5, refine=source.id, strength=0.6)
    path = Path(bundles.export_bundle(keep_gallery=True, ids=[keeper.id])["path"])
    use_destination(store, monkeypatch, tmp_path)
    assert save_source("different image reusing the original source ID", 18).id == source.id
    headers = _open(client, path)
    preview = client.post(RESTORE_PATH + "/preview", headers=headers, json={"image_id": keeper.id}).json()
    assert preview["dependencies"] == [
        {"kind": "image", "id": source.id, "used_by": keeper.id, "role": "refine", "status": "missing"},
    ]
    result = _finish(client, _selected(client, headers, keeper.id))["result"]
    assert result["replay_unavailable"] and result["unresolved_references"] == 1
    assert store.get_image(result["image_id"]).refine is None


def test_importing_source_later_leaves_a_known_keepers_unresolved_link_visible(client, store):
    source = save_source()
    keeper = save_source("keeper with an omitted source", 6, refine=source.id, strength=0.6)
    path = Path(bundles.export_bundle(expected=2)["path"])
    headers = _open(client, path)
    first = _finish(client, _selected(client, headers, keeper.id))["result"]
    assert first["replay_unavailable"] and first["imported"] == 1
    repeat = _finish(client, _selected(client, headers, keeper.id, [source.id]))["result"]
    assert repeat["replay_unavailable"] and repeat["imported"] == 1 and repeat["present"] == 1
    assert repeat["image_id"] == first["image_id"] and store.get_image(repeat["image_id"]).refine is None


def test_accepted_restore_keeps_its_pinned_input_after_view_closes(client, store, monkeypatch):
    keeper = save_source("binned keeper")
    storage.bin_images([keeper.id])
    archive = Path(bundles.export_bundle(expected=1)["path"])
    headers = _open(client, archive)
    reached, proceed = threading.Event(), threading.Event()
    original = bundles.restore_selection

    def held(*args):
        reached.set()
        assert proceed.wait(10)
        return original(*args)

    monkeypatch.setattr(bundles, "restore_selection", held)
    operation = _selected(client, headers, keeper.id)
    try:
        assert reached.wait(10)
        assert client.delete(BROWSE_PREFIX, headers=headers).status_code == 204
        assert client.post(BROWSE_PREFIX + "/images", headers=headers, json={}).status_code == 410
        assert bundle_leases.downloads.leased(archive)
    finally:
        proceed.set()
    state = _finish(client, operation)
    assert state["state"] == "done", state
    result = state["result"]
    assert result["image_binned"] is True and store.get_image(result["image_id"]) is None
    assert store.get_image(result["image_id"], include_binned=True).filename == keeper.filename
    assert not bundle_leases.downloads.leased(archive)
