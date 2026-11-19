"""Archive carry and source proofs for private Tiled Refine results."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image, ImageDraw

from backend import bundle_images, bundles, storage, tiled_refine_admission
from backend import tiled_refine_contract as tiled
from backend.bundle_common import BundleError
from backend.schemas import GenerateRequest
from tests.bundle_helpers import bundle_snapshot, png_metadata, save_source, use_destination
from tests.instruction_edit_helpers import save_edit


def save_tiled_refine():
    with closing(Image.new("RGB", tiled.ROOT_SIZE, (15, 28, 61))) as root_image:
        ImageDraw.Draw(root_image).rectangle((33, 17, 400, 700), fill=(230, 75, 140))
        root = storage.save_image(root_image, GenerateRequest(prompt="root"), 1, 0, tiled.MODEL_ID)
    spec = tiled_refine_admission.bind_source(tiled.TiledRefineSpec(source=root.id, recipe=tiled.RECIPE_NAME))
    request = GenerateRequest(
        prompt="keep the building", negative_prompt="blur", width=tiled.CANVAS_SIZE[0], height=tiled.CANVAS_SIZE[1],
        steps=tiled.STEPS, guidance_scale=tiled.GUIDANCE_SCALE, strength=tiled.STRENGTH, tiled_refine=spec,
    ).applied(negative_prompt_supported=True, cfg_threshold=1.0)
    with closing(Image.new("RGB", tiled.CANVAS_SIZE, (26, 56, 122))) as result:
        child = storage.save_image(result, request, 2, 0, tiled.MODEL_ID, quantization=tiled.QUANTIZATION)
    return root, child, spec


def backup(**kwargs):
    return Path(bundles.export_bundle(keep_gallery=True, **kwargs)["path"])


def update_proof(child, spec, **fields):
    changed = spec.model_copy(update=fields)
    with storage._session() as conn:
        conn.execute("UPDATE images SET tiled_refine = ? WHERE id = ?", (tiled.stored_json(changed), child.id))
    path = storage.settings.outputs_dir / child.filename
    metadata = png_metadata(path)
    metadata["tiled_refine"] = changed.model_dump(mode="json")
    bundle_images._rewrite_metadata(path, metadata)
    return changed


def downgrade(archive, tmp_path, version, *, drop_database_field):
    with ZipFile(archive) as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["version"] = version
    if drop_database_field:
        database = tmp_path / "downgraded.sqlite3"
        database.write_bytes(members["database.sqlite3"])
        with closing(sqlite3.connect(database)) as conn:
            conn.execute("ALTER TABLE images DROP COLUMN tiled_refine")
            conn.commit()
        data = members["database.sqlite3"] = database.read_bytes()
        manifest["files"]["database.sqlite3"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    members["manifest.json"] = json.dumps(manifest).encode()
    path = tmp_path / "downgraded.zip"
    with ZipFile(path, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    return path


def test_v3_remaps_to_an_already_present_proven_root_and_carries_v2_edits(store, monkeypatch, tmp_path):
    root, child, spec = save_tiled_refine()
    edit_source = save_source("edit root")
    edited = save_edit(edit_source)
    root_backup = backup(ids=[root.id])
    archive = backup(ids=[root.id, child.id, edit_source.id, edited.id])
    for path, version in ((root_backup, 1), (archive, 3)):
        with ZipFile(path) as reader:
            assert json.loads(reader.read("manifest.json"))["version"] == version

    destination = use_destination(store, monkeypatch, tmp_path)
    save_source("occupied", 12)
    bundles.import_bundle(root_backup)
    result = bundles.import_bundle(archive)
    assert result["present"] == 1 and result["imported"] == 3 and result["unresolved_references"] == 0
    records = {row.filename: row for row in store.list_images()[0]}
    mapped = records[root.filename].id
    assert mapped != root.id
    carried = records[child.filename].tiled_refine
    assert carried == spec.model_copy(update={"source": mapped})
    assert png_metadata(destination.outputs_dir / child.filename)["tiled_refine"] == carried.model_dump(mode="json")
    assert records[edited.filename].edit.source == records[edit_source.filename].id
    before = bundle_snapshot(store)
    assert bundles.import_bundle(archive)["present"] == 4
    assert before == bundle_snapshot(store)


def test_selected_result_without_root_reexports_complete_unresolved_intent(store, monkeypatch, tmp_path):
    root, child, spec = save_tiled_refine()
    result = bundles.export_bundle(keep_gallery=True, ids=[child.id])
    assert result["records"] == 1 and result["links_outside"] == 1
    destination = use_destination(store, monkeypatch, tmp_path)
    assert save_source("unrelated").id == root.id
    result = bundles.import_bundle(Path(result["path"]))
    assert result["unresolved_references"] == 1
    row = next(record for record in store.list_images()[0] if record.filename == child.filename)
    carried = spec.model_copy(update={"source": None})
    assert row.tiled_refine == carried
    assert png_metadata(destination.outputs_dir / child.filename)["tiled_refine"] == carried.model_dump(mode="json")
    with pytest.raises(ValueError, match="not been resolved"):
        row.tiled_refine.request()
    with ZipFile(backup(ids=[row.id])) as reader:
        assert json.loads(reader.read("manifest.json"))["version"] == 3


def test_present_root_requires_gallery_provenance_even_when_its_pixels_match(store, monkeypatch, tmp_path):
    root, child, spec = save_tiled_refine()
    root_backup = backup(ids=[root.id])
    archive = backup()
    use_destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(root_backup)
    with store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE filename = ?", ("0" * 16, root.filename))
    result = bundles.import_bundle(archive)
    assert result["present"] == 1 and result["unresolved_references"] == 1
    carried = next(record.tiled_refine for record in store.list_images()[0] if record.filename == child.filename)
    assert carried == spec.model_copy(update={"source": None})


@pytest.mark.parametrize("mismatch", ["root", "prepared"])
def test_matching_database_id_never_substitutes_for_either_proof(store, monkeypatch, tmp_path, mismatch):
    _, child, spec = save_tiled_refine()
    if mismatch == "root":
        # An equal legacy hash prefix still can't prove the root.
        root_hash = spec.root_source.hash[:16] + "0" * 48
        forged = update_proof(child, spec, root_source=spec.root_source.model_copy(update={"hash": root_hash}))
    else:
        forged = update_proof(child, spec, prepared_rgb_sha256="0" * 64)
    archive = backup()
    use_destination(store, monkeypatch, tmp_path)
    result = bundles.import_bundle(archive)
    assert result["unresolved_references"] == 1
    row = next(record for record in store.list_images()[0] if record.filename == child.filename)
    assert row.tiled_refine == forged.model_copy(update={"source": None})


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("drop_database_field", [False, True])
def test_tiled_intent_cannot_be_read_as_an_old_bundle(store, monkeypatch, tmp_path, version, drop_database_field):
    save_tiled_refine()
    archive = downgrade(backup(), tmp_path, version, drop_database_field=drop_database_field)
    use_destination(store, monkeypatch, tmp_path)
    before = bundle_snapshot(store)
    with pytest.raises(BundleError, match="Tiled Refine"):
        bundles.import_bundle(archive)
    assert bundle_snapshot(store) == before


def test_repeat_import_refuses_a_different_preparation_proof_on_equal_pixels(store, monkeypatch, tmp_path):
    _, child, _ = save_tiled_refine()
    archive = backup()
    use_destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(archive)
    local = next(record for record in store.list_images()[0] if record.filename == child.filename)
    update_proof(local, local.tiled_refine, prepared_rgb_sha256="0" * 64)
    before = bundle_snapshot(store)
    with pytest.raises(BundleError, match="different Tiled Refine recipe"):
        bundles.import_bundle(archive)
    assert bundle_snapshot(store) == before
