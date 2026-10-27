"""Crop positions survive recovery without changing legacy centered metadata."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from backend import bundles, png_settings, reindex, storage
from backend.bundle_common import BundleError
from backend.grid import Cell, cell_key
from backend.metadata import PNG_METADATA_KEY
from backend.refine import CROP_ANCHORS
from backend.schemas import GenerateRequest, ImageRecord


def _save(prompt="source", seed=1, **fields):
    return storage.save_image(
        Image.new("RGB", (256, 256), (seed % 255, 40, 90)),
        GenerateRequest(prompt=prompt, width=256, height=256, steps=28, **fields),
        seed=seed, duration_ms=10, model_id="test/model", quantization="fp8",
    )


def _chunks(path):
    with Image.open(path) as image:
        return {key: value for key, value in image.info.items() if isinstance(value, str)}


def _metadata(path):
    return json.loads(_chunks(path)[PNG_METADATA_KEY])


def _rewrite(path, meta):
    path.write_bytes(reindex._with_auradream_metadata(path.read_bytes(), meta))


def _destination(store, monkeypatch, tmp_path):
    mine = store.settings
    target = dataclasses.replace(mine, db_path=tmp_path / "destination.db", outputs_dir=tmp_path / "destination")
    monkeypatch.setattr(storage, "settings", target)
    storage.init_db()
    monkeypatch.setattr(reindex, "settings", target)
    return target


def test_an_old_table_gains_a_nullable_anchor_on_its_next_read(store):
    made = _save()
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        conn.execute("ALTER TABLE images DROP COLUMN crop_anchor")
        conn.commit()
    assert store.get_image(made.id).crop_anchor is None
    with store._session() as conn:
        column = next(row for row in conn.execute("PRAGMA table_info(images)") if row["name"] == "crop_anchor")
        assert column["type"] == "TEXT" and column["notnull"] == 0


@pytest.mark.parametrize("anchor", CROP_ANCHORS)
def test_anchor_round_trips_through_row_png_and_index_file(store, anchor):
    source = _save()
    made = _save("pass", 2, refine=source.id, strength=0.3, crop_anchor=anchor)
    expected = None if anchor == "center" else anchor
    path = store.settings.outputs_dir / made.filename
    meta = _metadata(path)
    assert made.crop_anchor == store.get_image(made.id).crop_anchor == expected
    assert meta.get("crop_anchor") == expected
    assert ("crop_anchor" in meta) == (expected is not None)
    assert (f", Crop anchor: {anchor}" in _chunks(path)["parameters"]) == (expected is not None)
    with store._session() as conn:
        assert conn.execute("SELECT crop_anchor FROM images WHERE id = ?", (made.id,)).fetchone()[0] == expected
    indexed = store.index_file("copy.png", meta, 256, 256, datetime.now(timezone.utc))
    assert indexed.crop_anchor == expected and indexed.refine == source.id


def test_explicit_center_has_exactly_the_legacy_png_metadata(store):
    source = _save()
    implicit = _save("pass", 2, refine=source.id, strength=0.3)
    explicit = _save("pass", 2, refine=source.id, strength=0.3, crop_anchor="center")
    assert _chunks(store.settings.outputs_dir / implicit.filename) == _chunks(store.settings.outputs_dir / explicit.filename)
    plain = _save("plain", 3)
    assert "crop_anchor" not in _metadata(store.settings.outputs_dir / plain.filename)


def test_completed_keys_distinguish_anchors_and_match_request_keys(store):
    source = _save()
    requests = [GenerateRequest(prompt="pass", width=256, height=256, steps=28,
                                refine=source.id, strength=0.3, crop_anchor=anchor)
                for anchor in ("center", "left", "right")]
    for request in requests:
        storage.save_image(Image.new("RGB", (256, 256)), request, 7, 10, "test/model", quantization="fp8")
    keys = store.completed_cells(["pass"], 256, 256, "test/model", "fp8")
    assert keys == {cell_key(Cell(request, 7)) for request in requests}
    assert {key[-2] for key in keys} == {"256x256@center", "256x256@left", "256x256@right"}


@pytest.mark.parametrize("changed", [
    {"crop_anchor": "north"}, {"crop_anchor": 3}, {"crop_anchor": ["top"]},
    {"refine": None}, {"refine": True}, {"mask": "a" * 64},
    {"outpaint": {"left": 16, "top": 0, "right": 0, "bottom": 0, "band": 8}},
])
def test_saved_and_imported_metadata_refuse_invalid_anchor_relationships(store, changed):
    source = _save()
    made = _save("pass", 2, refine=source.id, strength=0.3, crop_anchor="top")
    meta = {**_metadata(store.settings.outputs_dir / made.filename), **changed}
    with pytest.raises(ValueError, match="crop anchor"):
        store.index_file("invalid.png", meta, 256, 256, datetime.now(timezone.utc))
    with pytest.raises(png_settings.PngSettingsError, match="crop anchor"):
        png_settings.record({PNG_METADATA_KEY: json.dumps(meta)}, "invalid.png")
    with store._session() as conn:
        values = dict(conn.execute("SELECT * FROM images WHERE id = ?", (made.id,)).fetchone())
    row_changes = {key: value for key, value in changed.items()}
    if "outpaint" in row_changes:
        row_changes["outpaint"] = "edge-box3-r16:16,0,0,0@8"
    values.update(filename="invalid.png", **row_changes)
    with pytest.raises(ValueError, match="crop anchor"):
        store.import_file(values)
    assert store.list_images()[1] == 2


def test_record_validation_rejects_a_noncenter_crop_without_ordinary_source(store):
    source = _save()
    made = _save("pass", 2, refine=source.id, crop_anchor="bottom")
    for changed in ({"refine": None}, {"mask": "a" * 64}, {"crop_anchor": "north"}):
        with pytest.raises(ValidationError, match="crop"):
            ImageRecord.model_validate({**made.model_dump(), **changed})


@pytest.mark.parametrize("database", [False, True])
@pytest.mark.parametrize("missing_source", [False, True])
def test_reindex_remaps_or_drops_anchor_with_its_source(store, monkeypatch, tmp_path, database, missing_source):
    source = _save()
    made = _save("pass", 2, refine=source.id, strength=0.3, crop_anchor="bottom-right")
    original = store.settings
    if missing_source:
        (original.outputs_dir / source.filename).unlink()
    target = _destination(store, monkeypatch, tmp_path)
    _save("unrelated", 8)  # A source id from the other gallery is unsafe here.
    result = reindex.reindex(original.outputs_dir, database=original.db_path if database else None)
    recovered = next(row for row in store.list_images()[0] if row.filename == made.filename)
    meta = _metadata(target.outputs_dir / made.filename)
    assert result["unresolved_references"] == int(missing_source)
    if missing_source:
        assert recovered.refine is None and recovered.crop_anchor is None
        assert "crop_anchor" not in meta
    else:
        new_source = next(row for row in store.list_images()[0] if row.filename == source.filename)
        assert new_source.id != source.id
        assert recovered.refine == new_source.id and recovered.crop_anchor == "bottom-right"
        assert (meta["refine"], meta["crop_anchor"]) == (new_source.id, "bottom-right")


def test_in_place_reindex_clears_unproven_anchor_in_both_row_and_png(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = _save()
    made = _save("pass", 2, refine=source.id, crop_anchor="left")
    path = store.settings.outputs_dir / made.filename
    meta = _metadata(path)
    meta["refine_source"]["hash"] = "0" * 16
    _rewrite(path, meta)
    with store._session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (made.id,))
    counts = reindex.reindex(store.settings.outputs_dir)
    recovered = next(row for row in store.list_images()[0] if row.filename == made.filename)
    assert counts["unresolved_references"] == 1
    assert recovered.refine is None and recovered.crop_anchor is None
    assert "crop_anchor" not in _metadata(path)


def test_noncenter_png_dry_run_keeps_synthetic_sources_without_writing(store, monkeypatch, tmp_path):
    source = _save()
    made = _save("pass", 2, refine=source.id, crop_anchor="bottom")
    original = store.settings
    before = {path.name: path.read_bytes() for path in original.outputs_dir.glob("*.png")}
    target = _destination(store, monkeypatch, tmp_path)
    counts = reindex.reindex(original.outputs_dir, dry_run=True)
    assert counts["indexed"] == 2 and counts["unresolved_references"] == 0
    assert store.list_images()[1] == 0 and not (target.outputs_dir / made.filename).exists()
    assert {path.name: path.read_bytes() for path in original.outputs_dir.glob("*.png")} == before


@pytest.mark.parametrize("missing_source", [False, True])
def test_in_place_database_recovery_rewrites_changed_source_and_anchor(store, monkeypatch, tmp_path, missing_source):
    removed = _save("removed", 9)
    source = _save()
    made = _save("pass", 2, refine=source.id, crop_anchor="right")
    original = store.settings
    database = tmp_path / "source.db"
    bundles._backup(original.db_path, database)
    (original.outputs_dir / removed.filename).unlink()
    if missing_source:
        (original.outputs_dir / source.filename).unlink()
    target = dataclasses.replace(original, db_path=tmp_path / "recovered.db")
    monkeypatch.setattr(storage, "settings", target)
    monkeypatch.setattr(reindex, "settings", target)
    storage.init_db()
    counts = reindex.reindex(original.outputs_dir, database=database)
    recovered = next(row for row in store.list_images()[0] if row.filename == made.filename)
    meta = _metadata(target.outputs_dir / made.filename)
    assert counts["unresolved_references"] == int(missing_source)
    if missing_source:
        assert recovered.refine is None and recovered.crop_anchor is None
        assert meta["refine"] is None and "crop_anchor" not in meta
    else:
        assert recovered.refine != source.id
        assert (meta["refine"], meta["crop_anchor"]) == (recovered.refine, "right")


@pytest.mark.parametrize("database", [False, True])
def test_reindex_skips_an_invalid_anchor_before_remapping(store, monkeypatch, tmp_path, database):
    source = _save()
    made = _save("pass", 2, refine=source.id, crop_anchor="top-left")
    original = store.settings
    if database:
        with store._session() as conn:
            conn.execute("UPDATE images SET crop_anchor = 'north' WHERE id = ?", (made.id,))
    else:
        path = original.outputs_dir / made.filename
        _rewrite(path, {**_metadata(path), "crop_anchor": "north"})
    _destination(store, monkeypatch, tmp_path)
    counts = reindex.reindex(original.outputs_dir, database=original.db_path if database else None)
    assert counts["invalid" if database else "invalid_crop_anchors"] == 1
    assert [row.prompt for row in store.list_images()[0]] == ["source"]


@pytest.mark.parametrize("source_included", [False, True])
def test_bundle_remaps_or_drops_anchor_with_its_source(store, monkeypatch, tmp_path, source_included):
    source = _save()
    made = _save("pass", 2, refine=source.id, strength=0.3, crop_anchor="top-right")
    archive = bundles.export_bundle(keep_gallery=True, ids=None if source_included else [made.id])
    target = _destination(store, monkeypatch, tmp_path)
    _save("unrelated", 8)
    result = bundles.import_bundle(Path(archive["path"]))
    recovered = next(row for row in store.list_images()[0] if row.filename == made.filename)
    meta = _metadata(target.outputs_dir / made.filename)
    assert result["unresolved_references"] == int(not source_included)
    if source_included:
        new_source = next(row for row in store.list_images()[0] if row.filename == source.filename)
        assert new_source.id != source.id
        assert (recovered.refine, recovered.crop_anchor) == (new_source.id, "top-right")
        assert (meta["refine"], meta["crop_anchor"]) == (new_source.id, "top-right")
    else:
        assert recovered.refine is None and recovered.crop_anchor is None
        assert "crop_anchor" not in meta
    again = bundles.import_bundle(Path(archive["path"]))
    assert again["imported"] == 0 and again["present"] == 1 + int(source_included)


def test_full_bundle_after_source_less_selection_keeps_known_child(store, monkeypatch, tmp_path):
    source = _save()
    made = _save("pass", 2, refine=source.id, strength=0.3, crop_anchor="top-right")
    full = Path(bundles.export_bundle(keep_gallery=True)["path"])
    selection = Path(bundles.export_bundle(keep_gallery=True, ids=[made.id])["path"])
    target = _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(selection)["unresolved_references"] == 1
    known = next(row for row in store.list_images()[0] if row.filename == made.filename)
    assert known.refine is None and known.crop_anchor is None
    picture = target.outputs_dir / known.filename
    before = picture.read_bytes()

    result = bundles.import_bundle(full)
    assert (result["imported"], result["present"]) == (1, 1)
    assert store.get_image(known.id) == known
    assert picture.read_bytes() == before
    assert any(row.filename == source.filename for row in store.list_images()[0])
    repeated = bundles.import_bundle(full)
    assert (repeated["imported"], repeated["present"]) == (0, 2)


def test_repeated_bundle_import_refuses_different_crop_metadata(store, monkeypatch, tmp_path):
    source = _save()
    made = _save("pass", 2, refine=source.id, crop_anchor="top")
    archive = bundles.export_bundle(keep_gallery=True)
    _destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(Path(archive["path"]))
    with store._session() as conn:
        conn.execute("UPDATE images SET crop_anchor = 'bottom' WHERE filename = ?", (made.filename,))
    with pytest.raises(BundleError, match="different crop anchor"):
        bundles.import_bundle(Path(archive["path"]))


@pytest.mark.parametrize("assignment", ["crop_anchor = 'north'", "refine = NULL", "mask = '" + "a" * 64 + "'"])
def test_bundle_refuses_invalid_crop_rows_before_publishing(store, assignment):
    source = _save()
    made = _save("pass", 2, refine=source.id, crop_anchor="top")
    with store._session() as conn:
        conn.execute(f"UPDATE images SET {assignment} WHERE id = ?", (made.id,))
    with pytest.raises(BundleError, match="crop anchor"):
        bundles.export_bundle(keep_gallery=True)
    assert bundles.list_bundles() == []
    assert (store.settings.outputs_dir / made.filename).is_file()


def test_retained_requests_keep_anchor_and_legacy_requests_recover_center(store):
    source = _save()
    request = GenerateRequest(prompt="pass", width=256, height=256, steps=28,
                              refine=source.id, strength=0.3, crop_anchor="left")
    definition = {"job_id": "anchor-job", "run_id": "anchor-run", "created_at": 1.0,
                  "origin": "grid", "model_id": "test/model", "quantization": "fp8",
                  "width": 256, "height": 256, "batch": 1, "comparison": False, "metadata": {}}
    store.retain_run_job(definition, [Cell(request, 7)])
    recovered = store.retained_run_jobs()[0]["cells"][0]
    assert recovered.request.crop_anchor == "left"
    assert cell_key(recovered) == cell_key(Cell(request, 7))
    legacy = request.model_dump(exclude={"crop_anchor"})
    assert GenerateRequest.model_validate(legacy).crop_anchor == "center"


def test_png_settings_fills_panel_without_source_or_anchor(client, fake, store):
    source = _save()
    made = _save("pass", 2, refine=source.id, strength=0.3, crop_anchor="bottom-left")
    response = client.post("/api/png/settings", json={
        "name": made.filename, "chunks": _chunks(store.settings.outputs_dir / made.filename),
    })
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["settings"]["refine"] is None
    assert answer["settings"]["crop_anchor"] == "center"
    assert any("crop anchor is left out" in note for note in answer["notes"])
    assert store.list_images()[1] == 2
