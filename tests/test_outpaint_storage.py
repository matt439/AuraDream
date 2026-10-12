"""An outpaint where its row goes: the column and its migration, the chunk
keys and the A1111 line, bundles (the column, the chunk rewrite, the
invariant, the refusals, the mask built again), reindex's three builders,
and the spec recovered from the A1111 line an older build's import leaves."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from backend import bundle_images, bundles, grid, reindex, storage
from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEY
from backend.schemas import GenerateRequest, Outpaint
from tests.bundle_helpers import _destination, _metadata, _save

SPEC = {"left": 32, "right": 16, "band": 16}
KEY = "edge-box3-r16:32,0,16,0@16"


def _outpainted(source, spec=SPEC, *, seed=2):
    shape = Outpaint(**spec)
    width, height = source.width + shape.left + shape.right, source.height + shape.top + shape.bottom
    request = GenerateRequest(
        prompt="a wider view", width=width, height=height, steps=28, refine=source.id, strength=0.9,
        outpaint=spec,
    )
    (cell,) = storage.stamp_outpaints([grid.Cell(request=request, seed=seed)])
    return storage.save_image(
        Image.new("RGB", (width, height), (10, 200, 10)), cell.request, seed=seed, duration_ms=10,
        model_id="test/model", job_id="job-one", run_id="run-one",
    )


def _gallery(spec=SPEC):
    source = _save("source", 1)
    return source, _outpainted(source, spec)


def _rewrite_png(store, record, *, drop=(), meta=None, parameters=None) -> None:
    """`record`'s PNG with some text chunks dropped or replaced, its pixels as they were."""
    path = store.settings.outputs_dir / record.filename
    with Image.open(path) as image:
        image.load()
        info = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if not isinstance(value, str) or key in drop:
                continue
            if key == PNG_METADATA_KEY and meta is not None:
                value = json.dumps(meta)
            if key == "parameters" and parameters is not None:
                value = parameters
            info.add_itxt(key, value)
        image.save(path, format="PNG", pnginfo=info)


def _as_an_older_build_left_it(store, record) -> None:
    """What an older build's import leaves: no column, no chunk keys - and
    the A1111 line, which its rewrite never touches."""
    with store._session() as conn:
        conn.execute("UPDATE images SET outpaint = NULL WHERE id = ?", (record.id,))
    meta = _metadata(store.settings.outputs_dir / record.filename)
    meta.pop("outpaint")
    meta.pop("outpaint_fill")
    _rewrite_png(store, record, meta=meta)


def _lose_the_database(store):
    for suffix in ("", "-wal", "-shm"):
        Path(str(store.settings.db_path) + suffix).unlink(missing_ok=True)


def _strip_mask(store, record) -> None:
    store.mask_path(record.mask).unlink()
    _rewrite_png(store, record, drop=(PNG_MASK_KEY,))


# --------------------------------------------------------------------- the row


def test_an_older_table_gains_the_column(store):
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        conn.execute("DROP INDEX idx_images_outpaint")
        conn.execute("ALTER TABLE images DROP COLUMN outpaint")
        conn.commit()
        assert "outpaint" not in {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    source, made = _gallery()
    assert store.get_image(made.id).outpaint.key() == KEY
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        assert "idx_images_outpaint" in {row[1] for row in conn.execute("PRAGMA index_list(images)")}


def test_the_row_the_chunk_and_the_parameters_line(store):
    source, made = _gallery()
    assert made.outpaint.key() == KEY and (made.refine, made.width) == (source.id, 304)
    with store._session() as conn:
        assert conn.execute("SELECT outpaint FROM images WHERE id = ?", (made.id,)).fetchone()[0] == KEY
    with Image.open(store.settings.outputs_dir / made.filename) as image:
        meta = json.loads(image.info[PNG_METADATA_KEY])
        parameters = image.info["parameters"]
    assert meta["outpaint"] == {"left": 32, "top": 0, "right": 16, "bottom": 0, "band": 16}
    assert meta["outpaint_fill"] == "edge-box3-r16"
    assert parameters.endswith(", Outpaint: 32/0/16/0, Outpaint band: 16, Outpaint fill: edge-box3-r16")
    # Never without both its source and its mask.
    plain = store.save_image(
        Image.new("RGB", (304, 256)),
        GenerateRequest(prompt="x", width=304, height=256, refine=source.id, outpaint=SPEC),
        seed=3, duration_ms=1, model_id="test/model",
    )
    assert plain.outpaint is None and plain.mask is None


def test_index_file_reads_the_chunk_and_holds_the_invariant(store):
    source, made = _gallery()
    meta = _metadata(store.settings.outputs_dir / made.filename)
    path = store.settings.outputs_dir / made.filename
    copy = path.with_name("copy.png")
    copy.write_bytes(path.read_bytes())
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    assert store.index_file("copy.png", meta, 304, 256, now).outpaint.key() == KEY
    for broken in ({**meta, "refine": None}, {**meta, "mask": None}):
        other = path.with_name(f"copy-{len(str(broken))}-{broken['refine']}.png")
        other.write_bytes(path.read_bytes())
        assert store.index_file(other.name, broken, 304, 256, now).outpaint is None
    malformed = {**meta, "outpaint": {"left": 30, "top": 0, "right": 0, "bottom": 0, "band": 16}}
    path.with_name("copy-bad.png").write_bytes(path.read_bytes())
    assert store.index_file("copy-bad.png", malformed, 304, 256, now).outpaint is None
    assert store.malformed_outpaint(malformed) and not store.malformed_outpaint(meta)


# --------------------------------------------------------------------- reindex


def test_a_png_only_reindex_brings_the_outpaint_back(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["indexed"] == 2 and counts["outpaint_malformed"] == 0
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert (row.outpaint.key(), row.mask) == (KEY, made.mask)


def test_an_unknown_but_well_formed_fill_is_kept_on_reindex(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery({**SPEC, "fill": "edge-box5-r8"})
    _lose_the_database(store)
    reindex.reindex(store.settings.outputs_dir)
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert row.outpaint.key() == "edge-box5-r8:32,0,16,0@16"


def test_a_malformed_outpaint_is_left_off_and_counted(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    meta = _metadata(store.settings.outputs_dir / made.filename)
    meta["outpaint"]["left"] = 30
    # The A1111 line too, which would otherwise give the spec back exactly.
    _rewrite_png(store, made, meta=meta, drop=("parameters",))
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert (counts["outpaint_malformed"], counts["outpaints_recovered"]) == (1, 0)
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert row.outpaint is None and row.mask == made.mask


def test_a_directory_reindex_drops_the_outpaint_with_its_source(store, monkeypatch, tmp_path):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(storage, "settings", dataclasses.replace(mine, db_path=tmp_path / "other.db",
                                                                 outputs_dir=elsewhere))
    storage.init_db()
    source, made = _gallery()
    (elsewhere / source.filename).unlink()
    monkeypatch.setattr(storage, "settings", mine)
    counts = reindex.reindex(elsewhere)
    assert counts["indexed"] == 1 and counts["unresolved_references"] == 1
    (row,) = store.list_images()[0]
    assert (row.refine, row.mask, row.outpaint) == (None, None, None)
    meta = _metadata(mine.outputs_dir / made.filename)
    assert (meta["refine"], meta["mask"], meta["outpaint"], meta["outpaint_fill"]) == (None, None, None, None)


def test_a_png_only_reindex_builds_a_lost_outpaint_mask_again(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _strip_mask(store, made)
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert (counts["outpaint_masks_rebuilt"], counts["unresolved_masks"]) == (1, 0)
    assert store.mask_path(made.mask).is_file()


def test_a_png_only_reindex_recovers_an_outpaint_an_older_build_stripped(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["outpaints_recovered"] == 1
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert row.outpaint.key() == KEY
    assert _metadata(store.settings.outputs_dir / made.filename)["outpaint_fill"] == "edge-box3-r16"


def test_no_recovery_where_the_rebuilt_mask_is_not_the_rows(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    with Image.open(store.settings.outputs_dir / made.filename) as image:
        parameters = image.info["parameters"].replace("Outpaint band: 16", "Outpaint band: 24")
    _rewrite_png(store, made, parameters=parameters)
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["outpaints_recovered"] == 0
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert row.outpaint is None and row.mask == made.mask


def test_a_database_merge_carries_and_drops_the_outpaint(store, monkeypatch, tmp_path):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    _save("already here", 90)
    elsewhere = tmp_path / "backup" / "outputs"
    elsewhere.mkdir(parents=True)
    source_db = elsewhere.parent / "local-diffusion.db"
    monkeypatch.setattr(storage, "settings", dataclasses.replace(mine, db_path=source_db, outputs_dir=elsewhere))
    storage.init_db()
    source, made = _gallery()
    orphan_source = _save("gone source", 5)
    orphan = _outpainted(orphan_source, seed=6)
    with storage._session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (orphan_source.id,))
    monkeypatch.setattr(storage, "settings", mine)

    counts = reindex.reindex(elsewhere, database=source_db)
    assert counts["indexed"] == 3
    rows = {r.filename: r for r in store.list_images(limit=10)[0]}
    carried, dropped = rows[made.filename], rows[orphan.filename]
    assert (carried.outpaint.key(), carried.refine) == (KEY, rows[source.filename].id)
    assert (dropped.refine, dropped.mask, dropped.outpaint) == (None, None, None)
    meta = _metadata(mine.outputs_dir / made.filename)
    assert (meta["outpaint"]["left"], meta["outpaint_fill"]) == (32, "edge-box3-r16")
    assert _metadata(mine.outputs_dir / orphan.filename)["outpaint"] is None


# --------------------------------------------------------------------- bundles


def test_a_bundle_carries_the_column_and_the_chunk(store, monkeypatch, tmp_path):
    source, made = _gallery()
    archive = bundles.export_bundle(keep_gallery=True)
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["masks_missing"]) == (2, 0)
    row = next(r for r in store.list_images()[0] if r.outpaint is not None)
    new_source = next(r for r in store.list_images()[0] if r.filename == source.filename)
    assert (row.outpaint.key(), row.refine, row.mask) == (KEY, new_source.id, made.mask)
    meta = _metadata(store.settings.outputs_dir / row.filename)
    assert (meta["outpaint"]["right"], meta["outpaint_fill"], meta["refine"]) == (16, "edge-box3-r16", new_source.id)


def test_a_bundle_without_the_source_clears_the_outpaint(store, monkeypatch, tmp_path):
    source, made = _gallery()
    archive = bundles.export_bundle(keep_gallery=True, ids=[made.id])
    _destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(Path(archive["path"]))
    (row,) = store.list_images()[0]
    assert (row.refine, row.mask, row.outpaint) == (None, None, None)
    meta = _metadata(store.settings.outputs_dir / row.filename)
    assert (meta["outpaint"], meta["outpaint_fill"]) == (None, None)


@pytest.mark.parametrize("change", ["garbage", "no mask", "no side"])
def test_a_bundle_row_with_a_broken_outpaint_is_refused(store, change):
    source, made = _gallery()
    update = {
        "garbage": ("UPDATE images SET outpaint = '../x' WHERE id = ?", ()),
        "no mask": ("UPDATE images SET mask = NULL WHERE id = ?", ()),
        "no side": ("UPDATE images SET outpaint = 'edge-box3-r16:0,0,0,0@16' WHERE id = ?", ()),
    }[change]
    with store._session() as conn:
        conn.execute(update[0], (made.id,))
    with pytest.raises(bundles.BundleError, match=f"invalid outpaint on image {made.id}"):
        bundles.export_bundle(keep_gallery=True)
    row = {key: getattr(made, key, None) for key in storage._COLUMNS}
    row.update(id=made.id, outpaint="nope", mask=made.mask, refine=source.id, binned_at=None,
               reference=None, variables=None)
    with pytest.raises(bundles.BundleError, match="invalid outpaint"):
        bundle_images._validate_row(row, made.width, made.height, made.content_hash)


def test_an_import_builds_an_outpaints_mask_again_from_its_spec(store, monkeypatch, tmp_path):
    source, made = _gallery()
    _strip_mask(store, made)
    archive = bundles.export_bundle(keep_gallery=True)
    assert archive["masks"] == 0
    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["outpaint_masks_rebuilt"], imported["masks_missing"], imported["masks"]) == (1, 0, 1)
    assert store.mask_path(made.mask).is_file()


def test_an_import_whose_spec_does_not_build_the_rows_mask_counts_it_missing(store, monkeypatch, tmp_path):
    source, made = _gallery()
    _strip_mask(store, made)
    with store._session() as conn:
        conn.execute("UPDATE images SET outpaint = 'edge-box3-r16:32,0,16,0@24' WHERE id = ?", (made.id,))
    archive = bundles.export_bundle(keep_gallery=True)
    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["outpaint_masks_rebuilt"], imported["masks_missing"]) == (0, 1)


def test_an_import_recovers_an_outpaint_an_older_build_stripped(store, monkeypatch, tmp_path):
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    archive = bundles.export_bundle(keep_gallery=True)
    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert imported["outpaints_recovered"] == 1
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert row.outpaint.key() == KEY
    assert _metadata(store.settings.outputs_dir / row.filename)["outpaint_fill"] == "edge-box3-r16"


def test_an_import_does_not_recover_where_the_hash_differs(store, monkeypatch, tmp_path):
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    with Image.open(store.settings.outputs_dir / made.filename) as image:
        parameters = image.info["parameters"].replace("Outpaint: 32/0/16/0", "Outpaint: 16/0/32/0")
    _rewrite_png(store, made, parameters=parameters)
    archive = bundles.export_bundle(keep_gallery=True)
    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert imported["outpaints_recovered"] == 0
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert row.outpaint is None and row.mask == made.mask
