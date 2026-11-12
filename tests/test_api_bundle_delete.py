"""Deleting one saved bundle from the Bundles dialog (GAL#26): one confirmed
file, refused while it is leased, while a bundle operation holds the
gallery, or while the automatic backup's journal counts it."""
from __future__ import annotations

from uuid import uuid4

import pytest

from backend import api_bundles, backup_receipts as receipts, bundle_leases, bundles
from tests.bundle_helpers import save_source

URL = "/api/gallery/bundles/files/{}"


@pytest.fixture
def saved(store):
    save_source("kept", 3)
    result = bundles.export_bundle(keep_gallery=True)
    return result["name"]


def _listed(client):
    return [item["name"] for item in client.get("/api/gallery/bundles").json()["bundles"]]


def test_a_saved_bundle_is_deleted_and_leaves_the_list(client, store, saved):
    path = store.bundle_directory() / saved
    size = path.stat().st_size
    assert saved in _listed(client)
    response = client.delete(URL.format(saved))
    assert response.status_code == 200, response.text
    assert response.json() == {"name": saved, "bytes": size}
    assert not path.exists() and saved not in _listed(client)
    # The gallery is untouched: the bundle was a copy.
    assert store.list_images()[1] == 1


def test_a_missing_bundle_is_refused_as_gone(client, store, saved):
    assert client.delete(URL.format(saved)).status_code == 200
    again = client.delete(URL.format(saved))
    assert again.status_code == 404
    assert again.json()["code"] == "bundle_gone"
    assert "nothing was deleted" in again.json()["detail"]


@pytest.mark.parametrize("name", ["notes.txt", "..%2Fgallery.db", "a.db"])
def test_only_a_bundle_filename_in_the_directory_is_taken(client, store, saved, name):
    response = client.delete(URL.format(name))
    assert response.status_code in {404, 422}, response.text
    assert (store.bundle_directory() / saved).exists()


def test_a_bundle_being_downloaded_or_read_is_refused_in_use(client, store, saved):
    path = store.bundle_directory() / saved
    with bundle_leases.reading(path):
        refused = client.delete(URL.format(saved))
    assert refused.status_code == 409
    assert refused.json()["code"] == "bundle_in_use"
    assert path.exists()
    # Once the reader has gone, the same press goes through.
    assert client.delete(URL.format(saved)).status_code == 200


def test_a_running_bundle_operation_refuses_the_delete(client, store, saved, monkeypatch):
    monkeypatch.setattr(api_bundles.BundleManager, "running", property(lambda self: True))
    refused = client.delete(URL.format(saved))
    assert refused.status_code == 409
    assert refused.json()["code"] == "bundle_in_use"
    assert (store.bundle_directory() / saved).exists()


def test_a_file_another_process_holds_is_refused_in_use(client, store, saved, monkeypatch):
    path = store.bundle_directory() / saved

    def held(self, *args, **kwargs):
        raise PermissionError("in use")

    monkeypatch.setattr(type(path), "unlink", held)
    refused = client.delete(URL.format(saved))
    assert refused.status_code == 409 and refused.json()["code"] == "bundle_in_use"


def test_a_copy_the_automatic_backup_counts_is_refused(client, store):
    save_source("kept", 4)
    identity = receipts.ManagedIdentity(str(uuid4()), str(uuid4()), str(uuid4()))
    managed = bundles.export_bundle(keep_gallery=True, managed=identity)["name"]
    plain = bundles.export_bundle(keep_gallery=True)["name"]
    refused = client.delete(URL.format(managed))
    assert refused.status_code == 409 and "code" not in refused.json()
    assert "automatic backup" in refused.json()["detail"]
    assert (store.bundle_directory() / managed).exists()
    # The journal counts only its own copy; another saved bundle still goes.
    assert client.delete(URL.format(plain)).status_code == 200


def test_an_unreadable_journal_refuses_rather_than_guesses(client, store, saved):
    receipts.path_for(store.settings.db_path).write_text("{not json", encoding="utf-8")
    refused = client.delete(URL.format(saved))
    assert refused.status_code == 409 and "receipts could not be read" in refused.json()["detail"]
    assert (store.bundle_directory() / saved).exists()


def test_the_journal_names_only_live_and_pending_copies(store):
    db = store.settings.db_path
    directory = store.bundle_directory()
    assert receipts.counts_archive(db, directory, "missing.zip") is False
    save_source("kept", 5)
    identity = receipts.ManagedIdentity(str(uuid4()), str(uuid4()), str(uuid4()))
    name = bundles.export_bundle(keep_gallery=True, managed=identity)["name"]
    assert receipts.counts_archive(db, directory, name) is True
    assert receipts.counts_archive(db, directory, name.upper()) is True
    assert receipts.counts_archive(db, directory / "elsewhere", name) is False
    assert receipts.counts_archive(db, directory, "other.zip") is False
