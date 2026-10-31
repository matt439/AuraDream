"""Internal managed publications over scratch galleries; no activation or pruning."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from backend import backup_receipts as receipts, bundle_archive, bundle_crypto, bundles, gallery_delete, gallery_lock
from backend.bundle_common import BundleError
from tests.bundle_helpers import _destination, _save, _snapshot


@pytest.fixture
def identity():
    return receipts.ManagedIdentity(str(uuid4()), str(uuid4()), str(uuid4()))


@pytest.mark.parametrize("password", [None, "scratch-only password"])
def test_managed_copy_has_exact_outer_proof_one_readback_and_restores(store, monkeypatch, tmp_path, identity, password):
    source = _save("source", 11)
    refined = _save("refined", 12, refine=source.id, strength=0.35)
    store.set_favorite(refined.id, True)
    gallery_delete.bin_images([source.id])
    before = _snapshot(store)
    db = store.settings.db_path
    real = bundles._verify_archive
    real_append = receipts.append
    verified = []

    def verify(*args, **kwargs):
        verified.append(kwargs.get("managed"))
        return real(*args, **kwargs)

    def append(database, receipt):
        assert os.path.normcase(str(database.resolve())) in gallery_lock._local.held
        assert verified == [identity] and (Path(receipt.directory) / receipt.name).exists()
        real_append(database, receipt)

    monkeypatch.setattr(bundles, "_verify_archive", verify)
    monkeypatch.setattr(receipts, "append", append)
    result = bundles.export_bundle(keep_gallery=True, password=password, managed=identity)
    assert verified == [identity]
    assert result["managed"] is True and result["kept"] and not result["cleared"]
    assert result["deleted_ids"] == [] and _snapshot(store) == before
    receipt, = receipts.load(db, identity.gallery_id)
    archive = Path(result["path"])
    blob = archive.read_bytes()
    assert receipt.sha256 == hashlib.sha256(blob).hexdigest()
    assert receipt.stamp.size == result["bytes"] == len(blob)
    assert receipt.protection == ("encrypted" if password else "unencrypted")
    assert receipts.matches_current(receipt, archive.parent)
    with bundle_archive._open_archive(archive, password) as opened:
        manifest = json.loads(opened.read("manifest.json"))
        assert manifest[receipts.MARKER] == identity.marker()
        assert manifest["version"] == 1
    if password:
        assert blob.startswith(bundle_crypto.MAGIC) and b"source" not in blob
    _destination(store, monkeypatch, tmp_path)
    restored = bundles.import_bundle(archive, password)
    assert restored["imported"] == 2 and restored["binned"] == 1
    records = {row.filename: row for row in store.list_images()[0] + store.list_images(binned=True)[0]}
    assert records[refined.filename].favorite
    assert records[refined.filename].refine == records[source.filename].id


def test_ordinary_exports_keep_original_manifest_and_create_no_journal(store):
    _save()
    result = bundles.export_bundle(keep_gallery=True)
    assert "managed" not in result
    assert not receipts.path_for(store.settings.db_path).exists()
    with zipfile.ZipFile(result["path"]) as archive:
        assert receipts.MARKER not in json.loads(archive.read("manifest.json"))


@pytest.mark.parametrize("kwargs", [{}, {"keep_gallery": True, "ids": []}])
def test_managed_identity_cannot_label_clearing_or_selected_export(store, identity, kwargs):
    _save()
    before = _snapshot(store)
    with pytest.raises(BundleError, match="managed exports"):
        bundles.export_bundle(managed=identity, **kwargs)
    assert _snapshot(store) == before
    assert not receipts.path_for(store.settings.db_path).exists()


def test_marker_mismatch_in_the_existing_readback_refuses_publication(store, monkeypatch, identity):
    _save()
    real = bundles._verify_archive

    def mismatched(*args, **kwargs):
        kwargs["managed"] = replace(identity, operation_id=str(uuid4()))
        return real(*args, **kwargs)

    monkeypatch.setattr(bundles, "_verify_archive", mismatched)
    with pytest.raises(BundleError, match="identity does not match"):
        bundles.export_bundle(keep_gallery=True, managed=identity)
    assert bundles.list_bundles() == []
    assert not list(bundles._directory().glob("*.part"))
    assert not receipts.path_for(store.settings.db_path).exists()


def test_managed_publication_never_overwrites_a_collision(store, monkeypatch, identity):
    _save()
    real = bundles._claim
    existing = []

    def collide(source, target):
        target.write_bytes(b"existing unrelated archive")
        existing.append(target)
        real(source, target)

    monkeypatch.setattr(bundles, "_claim", collide)
    with pytest.raises(FileExistsError):
        bundles.export_bundle(keep_gallery=True, managed=identity)
    assert existing[0].read_bytes() == b"existing unrelated archive"
    assert not list(bundles._directory().glob("*.part"))
    assert not receipts.path_for(store.settings.db_path).exists()


def test_receipt_failure_keeps_previous_journal_and_published_copy_unmanaged(store, monkeypatch, identity):
    _save()
    first = bundles.export_bundle(keep_gallery=True, managed=identity)
    path = receipts.path_for(store.settings.db_path)
    before = path.read_bytes()
    real = receipts.os.replace

    def fail_journal(source, target):
        if Path(target) == path:
            raise OSError("secret-that-must-not-be-echoed")
        return real(source, target)

    monkeypatch.setattr(receipts.os, "replace", fail_journal)
    second = bundles.export_bundle(keep_gallery=True, managed=replace(identity, operation_id=str(uuid4())))
    assert second["managed"] is False
    assert "secret-that-must-not-be-echoed" not in repr(second)
    assert path.read_bytes() == before
    assert Path(first["path"]).exists() and Path(second["path"]).exists()
    assert len(receipts.load(store.settings.db_path, identity.gallery_id)) == 1


def test_duplicate_operation_keeps_existing_receipt_and_new_copy_unmanaged(store, identity):
    _save()
    first = bundles.export_bundle(keep_gallery=True, managed=identity)
    before = receipts.path_for(store.settings.db_path).read_bytes()
    second = bundles.export_bundle(keep_gallery=True, managed=identity)
    assert not second["managed"]
    assert Path(first["path"]).exists() and Path(second["path"]).exists()
    assert receipts.path_for(store.settings.db_path).read_bytes() == before


def test_same_name_replacement_preserving_marker_is_not_owned(store, identity):
    _save()
    result = bundles.export_bundle(keep_gallery=True, managed=identity)
    receipt, = receipts.load(store.settings.db_path, identity.gallery_id)
    path = Path(result["path"])
    with zipfile.ZipFile(path) as source:
        members = {info.filename: source.read(info.filename) for info in source.infolist()}
    replacement = path.with_name("replacement.zip")
    with zipfile.ZipFile(replacement, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.comment = b"same marker and contents, distinct outer archive"
    replacement.replace(path)
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("manifest.json"))[receipts.MARKER] == identity.marker()
    assert not receipts.matches_current(receipt, path.parent)
    # Even supplying current stat facts cannot substitute for the byte digest.
    changed_stamp = receipts.FileStamp.of(path.stat())
    assert not receipts.matches_current(replace(receipt, stamp=changed_stamp), path.parent)


def test_malformed_journal_is_not_rebuilt_from_a_new_publication(store, identity):
    _save()
    journal = receipts.path_for(store.settings.db_path)
    journal.write_bytes(b'{"broken": "keep exact bytes"}')
    result = bundles.export_bundle(keep_gallery=True, managed=identity)
    assert not result["managed"] and Path(result["path"]).exists()
    assert journal.read_bytes() == b'{"broken": "keep exact bytes"}'


def test_reserved_malformed_marker_is_refused_by_import_without_changes(store):
    _save()
    result = bundles.export_bundle(keep_gallery=True)
    path = Path(result["path"])
    with zipfile.ZipFile(path) as source:
        members = {info.filename: source.read(info.filename) for info in source.infolist()}
    manifest = json.loads(members["manifest.json"])
    manifest[receipts.MARKER] = {"version": True, "secret": "do not echo"}
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    before = _snapshot(store)
    with pytest.raises(BundleError, match="managed identity is invalid") as error:
        bundles.import_bundle(path)
    assert "do not echo" not in str(error.value) and _snapshot(store) == before


def test_changed_partial_after_verification_is_never_published_or_receipted(store, monkeypatch, identity):
    _save()
    real = bundles._verify_archive

    def changed(path, *args, **kwargs):
        result = real(path, *args, **kwargs)
        with path.open("ab") as stream:
            stream.write(b"changed after verification")
        return result

    monkeypatch.setattr(bundles, "_verify_archive", changed)
    with pytest.raises(BundleError, match="changed during verification"):
        bundles.export_bundle(keep_gallery=True, managed=identity)
    assert bundles.list_bundles() == []
    assert not receipts.path_for(store.settings.db_path).exists()


def test_existing_unowned_partial_is_never_removed_on_exclusive_open_failure(store, monkeypatch, identity):
    _save()
    directory = bundles._directory()
    directory.mkdir()
    final = directory / "collision.zip"
    partial = directory / "collision.zip.part"
    partial.write_bytes(b"someone else's in-flight file")
    monkeypatch.setattr(bundles, "bundle_path", lambda _name: final)
    with pytest.raises(FileExistsError):
        bundles.export_bundle(keep_gallery=True, managed=identity)
    assert partial.read_bytes() == b"someone else's in-flight file"
    assert not final.exists() and not receipts.path_for(store.settings.db_path).exists()


def test_valid_same_marker_replacement_after_writer_close_is_refused_and_preserved(store, monkeypatch, identity):
    _save()
    real_lstat = Path.lstat
    real_verify = bundles._verify_archive
    replacement_bytes = []
    partials = []
    verified = []

    def replace_before_snapshot(path, *args, **kwargs):
        if path.name.endswith(".zip.part") and not partials:
            partials.append(path)
            with zipfile.ZipFile(path) as archive:
                members = {info.filename: archive.read(info.filename) for info in archive.infolist()}
            other = path.with_name("external-replacement.zip")
            with zipfile.ZipFile(other, "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
                archive.comment = b"valid replacement retaining the original marker"
            other.replace(path)
            replacement_bytes.append(path.read_bytes())
        return real_lstat(path, *args, **kwargs)

    def verify(*args, **kwargs):
        verified.append(True)
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(Path, "lstat", replace_before_snapshot)
    monkeypatch.setattr(bundles, "_verify_archive", verify)
    with pytest.raises(BundleError, match="changed after writing"):
        bundles.export_bundle(keep_gallery=True, managed=identity)
    assert verified == []  # Identity refuses before an unowned file is read back.
    partial, = partials
    assert partial.read_bytes() == replacement_bytes[0]
    assert not receipts.path_for(store.settings.db_path).exists()
    assert bundles.list_bundles() == []
    manifest, _, _ = real_verify(partial, rows=bundles._rows(store.settings.db_path), managed=identity)
    assert manifest[receipts.MARKER] == identity.marker()


def test_unowned_source_recreated_after_successful_claim_is_preserved(store, monkeypatch, identity):
    _save()
    real = bundles._claim
    recreated = []

    def claim_then_recreate(source, target):
        real(source, target)
        # POSIX claim leaves the original hard link; Windows rename removes it.
        # Replace that source entry in either case, preserving the final file.
        source.unlink(missing_ok=True)
        source.write_bytes(b"unrelated writer reused the partial name")
        recreated.append(source)

    monkeypatch.setattr(bundles, "_claim", claim_then_recreate)
    result = bundles.export_bundle(keep_gallery=True, managed=identity)
    assert result["managed"] is True
    source, = recreated
    assert source.read_bytes() == b"unrelated writer reused the partial name"
    receipt, = receipts.load(store.settings.db_path, identity.gallery_id)
    assert receipts.matches_current(receipt, Path(result["path"]).parent)
