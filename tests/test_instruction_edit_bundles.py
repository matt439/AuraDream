"""Versioned portable edit identity over owned synthetic galleries only."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import pytest
from PIL import Image, PngImagePlugin

from backend import backup_receipts, bundle_archive, bundle_images, bundles, reindex, storage
from backend.bundle_common import BundleError
from backend.instruction_edit import descriptor_from_image, edit_key
from tests.bundle_helpers import _destination, _metadata, _save, _snapshot
from tests.instruction_edit_helpers import save_edit


def backup(**kwargs):
    return Path(bundles.export_bundle(keep_gallery=True, **kwargs)["path"])


def rewrite_edit(store, child, value):
    with store._session() as conn:
        conn.execute("UPDATE images SET edit = ? WHERE id = ?", (edit_key(value), child.id))
    path = store.settings.outputs_dir / child.filename
    meta = _metadata(path)
    meta.pop("edit", None)
    if value is not None:
        meta["edit"] = value.model_dump(mode="json")
    bundle_images._rewrite_metadata(path, meta)


def altered_archive(archive, tmp_path, *, version=1, remove_sql=False, remove_png=False):
    with ZipFile(archive) as opened:
        members = {info.filename: opened.read(info) for info in opened.infolist()}
    database = tmp_path / "altered.sqlite3"
    database.write_bytes(members["database.sqlite3"])
    with closing(sqlite3.connect(database)) as conn:
        name, = conn.execute("SELECT filename FROM images WHERE edit IS NOT NULL").fetchone()
        if remove_sql:
            conn.execute("ALTER TABLE images DROP COLUMN edit")
            conn.commit()
    members["database.sqlite3"] = database.read_bytes()
    if remove_png:
        member = "images/" + name
        with Image.open(io.BytesIO(members[member])) as image:
            meta = json.loads(image.info["auradream"])
        meta.pop("edit")
        members[member] = reindex._with_auradream_metadata(members[member], meta)
    manifest = json.loads(members["manifest.json"])
    manifest["version"] = version
    for name, data in members.items():
        if name != "manifest.json":
            manifest["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    members["manifest.json"] = json.dumps(manifest).encode()
    result = tmp_path / "altered.zip"
    with ZipFile(result, "w") as opened:
        for name, data in members.items():
            opened.writestr(name, data)
    return result


@pytest.mark.parametrize("password", [None, "scratch-only secret"])
@pytest.mark.parametrize("selected", [False, True])
def test_whole_selected_and_password_edit_exports_remap_proven_sources(store, monkeypatch, tmp_path, password, selected):
    source = _save()
    child = save_edit(source)
    origin_path = store.settings.outputs_dir / source.filename
    original = origin_path.read_bytes()
    archive = backup(ids=[source.id, child.id] if selected else None, password=password)
    with bundle_archive._open_archive(archive, password) as opened:
        assert json.loads(opened.read("manifest.json"))["version"] == 2
    dest = _destination(store, monkeypatch, tmp_path)
    _save("occupied", 90)
    result = bundles.import_bundle(archive, password)
    assert result["imported"] == 2 and result["unresolved_references"] == 0
    records = {row.filename: row for row in store.list_images()[0]}
    back = records[child.filename]
    assert back.edit.source == records[source.filename].id != source.id
    assert back.edit.edit_source == child.edit.edit_source and not back.edit_missing
    assert _metadata(dest.outputs_dir / child.filename)["edit"] == back.edit.model_dump(mode="json")
    assert bundle_images._pixels(dest.outputs_dir / source.filename).rgb_hash == child.edit.edit_source.hash
    assert origin_path.read_bytes() == original


@pytest.mark.parametrize("managed", [False, True])
def test_edit_free_exports_stay_version_one(store, managed):
    _save()
    identity = backup_receipts.ManagedIdentity(*(str(uuid4()) for _ in range(3))) if managed else None
    archive = backup(managed=identity)
    with ZipFile(archive) as opened:
        assert json.loads(opened.read("manifest.json"))["version"] == 1


def test_managed_edit_export_uses_v2_in_existing_capacity_receipt_and_readback(store, monkeypatch):
    child = save_edit(_save())
    identity = backup_receipts.ManagedIdentity(*(str(uuid4()) for _ in range(3)))
    real = bundles._verify_archive
    calls = []

    def verify(*args, **kwargs):
        calls.append(kwargs.get("managed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(bundles, "_verify_archive", verify)
    archive = backup(managed=identity)
    assert calls == [identity]
    receipt, = backup_receipts.load(store.settings.db_path, identity.gallery_id)
    assert receipt.sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as opened:
        assert json.loads(opened.read("manifest.json"))["version"] == 2
    assert store.get_image(child.id).edit == child.edit


def test_child_only_selection_preserves_unresolved_proof_across_reexport(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    result = bundles.export_bundle(keep_gallery=True, ids=[child.id])
    assert result["links_outside"] == 1 and result["records"] == 1
    _destination(store, monkeypatch, tmp_path)
    unrelated = _save("unrelated", 55)
    imported = bundles.import_bundle(Path(result["path"]))
    assert imported["unresolved_references"] == 1
    back = next(row for row in store.list_images()[0] if row.id != unrelated.id)
    assert back.edit.source is None and back.edit_missing
    assert back.edit.edit_source == child.edit.edit_source
    later = bundles.export_bundle(keep_gallery=True, ids=[back.id])
    assert later["links_outside"] == 1
    with ZipFile(later["path"]) as opened:
        with Image.open(io.BytesIO(opened.read("images/" + back.filename))) as image:
            assert json.loads(image.info["auradream"])["edit"] == back.edit.model_dump(mode="json")


@pytest.mark.parametrize("mismatch", ["hash", "filename", "width"])
def test_included_unproven_source_becomes_null_even_with_mapping(store, monkeypatch, tmp_path, mismatch):
    source = _save()
    child = save_edit(source)
    proof = child.edit.edit_source
    if mismatch == "hash":
        # Keep the complete legacy 16-character prefix equal.
        proof = proof.model_copy(update={"hash": proof.hash[:16] + "0" * 48})
    elif mismatch == "filename":
        proof = proof.model_copy(update={"filename": "another.png"})
    else:
        proof = proof.model_copy(update={"width": 512, "height": 512})
    rewrite_edit(store, child, child.edit.model_copy(update={"edit_source": proof}))
    archive = backup()
    _destination(store, monkeypatch, tmp_path)
    result = bundles.import_bundle(archive)
    assert result["unresolved_references"] == 1
    back = next(row for row in store.list_images()[0] if row.filename == child.filename)
    assert back.edit.source is None and back.edit.edit_source == proof and back.edit_missing


def test_repeat_import_does_not_relink_orphan_or_overwrite_destination_metadata(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    selected = backup(ids=[child.id])
    whole = backup()
    _destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(selected)
    orphan = store.list_images()[0][0]
    before = (store.settings.outputs_dir / orphan.filename).read_bytes()
    result = bundles.import_bundle(whole)
    assert result["imported"] == 1 and result["present"] == 1
    assert store.get_image(orphan.id).edit.source is None
    assert (store.settings.outputs_dir / orphan.filename).read_bytes() == before


@pytest.mark.parametrize("remove_sql,remove_png", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("repeat", [False, True])
def test_recomputed_v1_downgrade_cannot_strip_edit_intent(store, monkeypatch, tmp_path, remove_sql, remove_png, repeat):
    child = save_edit(_save())
    archive = backup()
    changed = altered_archive(archive, tmp_path, remove_sql=remove_sql, remove_png=remove_png)
    if not repeat:
        _destination(store, monkeypatch, tmp_path)
    before = _snapshot(store)
    with pytest.raises(BundleError, match="instruction|version"):
        bundles.import_bundle(changed)
    assert _snapshot(store) == before
    if repeat:
        assert store.get_image(child.id).edit == child.edit


@pytest.mark.parametrize("version", [True, False, 1.0, 2.0, 0, 3, "2"])
def test_manifest_versions_are_strict_integers(store, tmp_path, version):
    save_edit(_save())
    changed = altered_archive(backup(), tmp_path, version=version)
    with pytest.raises(BundleError, match="supported"):
        bundles.import_bundle(changed)


@pytest.mark.parametrize("missing", ["sql", "png"])
def test_export_conflict_refuses_before_publishing_or_cleanup(store, missing):
    child = save_edit(_save())
    if missing == "sql":
        with store._session() as conn:
            conn.execute("UPDATE images SET edit = NULL WHERE id = ?", (child.id,))
    else:
        path = store.settings.outputs_dir / child.filename
        meta = _metadata(path)
        meta.pop("edit")
        bundle_images._rewrite_metadata(path, meta)
    before = _snapshot(store)
    with pytest.raises(BundleError, match="instruction"):
        bundles.export_bundle(expected=2)
    assert _snapshot(store) == before
    assert list(store.bundle_directory().glob("*.zip")) == []


def test_equal_pixels_do_not_make_an_ordinary_row_the_same_as_an_edit(store):
    child = save_edit(_save())
    archive = backup()
    rewrite_edit(store, child, None)
    before = _snapshot(store)
    with pytest.raises(BundleError, match="different instruction"):
        bundles.import_bundle(archive)
    assert _snapshot(store) == before


def test_binned_proven_source_keeps_lineage_but_is_unavailable(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    store.bin_images([source.id])
    archive = backup()
    _destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(archive)
    back = next(row for row in store.list_images()[0] if row.filename == child.filename)
    assert back.edit.source is not None and back.edit_missing and back.edit_binned


def test_source_proof_uses_decoded_rgb_for_nonrgb_png(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    path = store.settings.outputs_dir / source.filename
    meta = _metadata(path)
    chunks = PngImagePlugin.PngInfo()
    chunks.add_text("auradream", json.dumps(meta))
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    rgba.putalpha(71)
    rgba.save(path, pnginfo=chunks)
    rgba.close()
    assert not store.get_image(child.id).edit_missing
    archive = backup()
    _destination(store, monkeypatch, tmp_path)
    result = bundles.import_bundle(archive)
    assert result["unresolved_references"] == 0
    back = next(row for row in store.list_images()[0] if row.filename == child.filename)
    assert back.edit.source is not None and not back.edit_missing


def test_cyclic_link_is_kept_unresolved(store, monkeypatch, tmp_path):
    child = save_edit(_save())
    with Image.open(store.settings.outputs_dir / child.filename) as image:
        proof = descriptor_from_image(child.filename, image)
    rewrite_edit(store, child, child.edit.model_copy(update={"source": child.id, "edit_source": proof}))
    archive = backup()
    _destination(store, monkeypatch, tmp_path)
    result = bundles.import_bundle(archive)
    assert result["unresolved_references"] == 1
    back = next(row for row in store.list_images()[0] if row.filename == child.filename)
    assert back.edit.source is None and back.edit.edit_source == proof


def test_retained_destination_png_is_checked_against_its_own_edit_row(store):
    child = save_edit(_save())
    archive = backup()
    path = store.settings.outputs_dir / child.filename
    meta = _metadata(path)
    meta.pop("edit")
    bundle_images._rewrite_metadata(path, meta)
    before = _snapshot(store)
    with pytest.raises(BundleError, match="instruction"):
        bundles.import_bundle(archive)
    assert _snapshot(store) == before


def test_missing_file_restore_preserves_destination_unresolved_identity(store, monkeypatch, tmp_path):
    source = _save()
    child = save_edit(source)
    selected = backup(ids=[child.id])
    whole = backup()
    _destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(selected)
    orphan = store.list_images()[0][0]
    path = store.settings.outputs_dir / orphan.filename
    path.unlink()
    result = bundles.import_bundle(whole)
    assert result["restored"] == 1 and store.get_image(orphan.id).edit.source is None
    assert _metadata(path)["edit"] == orphan.edit.model_dump(mode="json")


def test_sql_png_instruction_disagreement_refuses_export(store):
    child = save_edit(_save())
    path = store.settings.outputs_dir / child.filename
    meta = _metadata(path)
    meta["prompt"] = "a different instruction"
    bundle_images._rewrite_metadata(path, meta)
    before = _snapshot(store)
    with pytest.raises(BundleError, match="disagree"):
        backup()
    assert _snapshot(store) == before
