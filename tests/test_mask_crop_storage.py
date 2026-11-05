"""Only the masked area where its row goes (TODO.md INP#9): the two columns
and their migration, the chunk key and the A1111 line, `index_file`,
reindex's two paths (the region proved against the mask, a malformed one
left off, the spec recovered from the A1111 line), bundles (carried,
cleared, refused, recovered), the reruns that carry or drop it, the search
term, and the price read from cropped history of the same tier alone."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from backend import bundle_images, bundles, gallery_reports, grid, mask_crop, reindex, storage
from backend import submission, then_refine
from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEY
from backend.png_settings import record as png_record
from backend.schemas import GenerateRequest, ThenRefine
from backend.schemas_mask_crop import MaskCropSpec
from tests.api_helpers import run_one, stream_job
from tests.bundle_helpers import png_metadata, save_source, use_destination

KEY = "bbox16:32@1"
# The painted rectangle, and the region rule bbox16 draws round it at pad 32.
PAINTED = (48, 64, 112, 96)
BOX, WORKING = (16, 32, 144, 128), (1200, 896)
BOX_TEXT = "16,32,144,128>1200x896"


def _mask(box=PAINTED) -> Image.Image:
    image = Image.new("L", (256, 256), 0)
    image.paste(255, box)
    return image


def _cropped(source, *, pad=32, seed=2, box=PAINTED, region=None, crop=True, duration_ms=10,
             model_id="test/model", steps=28, strength=0.6):
    digest, _ = storage.store_mask(_mask(box))
    request = GenerateRequest(
        prompt="a red scarf", width=256, height=256, steps=steps, refine=source.id, strength=strength,
        mask=digest, mask_crop={"pad": pad} if crop else None,
    )
    return storage.save_image(
        Image.new("RGB", (256, 256), (200, 10, 10)), request, seed=seed, duration_ms=duration_ms,
        model_id=model_id, job_id="job-one", run_id="run-one", mask_region=region,
    )


def _gallery(**kwargs):
    source = save_source("source", 1)
    return source, _cropped(source, **kwargs)


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
    """What an older build's import leaves: no columns, no chunk key - and
    the A1111 line, which its rewrite never touches."""
    with store._session() as conn:
        conn.execute("UPDATE images SET mask_crop = NULL, mask_crop_box = NULL WHERE id = ?", (record.id,))
    meta = png_metadata(store.settings.outputs_dir / record.filename)
    meta.pop("mask_crop")
    _rewrite_png(store, record, meta=meta)


def _with_chunk(store, record, chunk) -> None:
    meta = png_metadata(store.settings.outputs_dir / record.filename)
    meta["mask_crop"] = chunk
    _rewrite_png(store, record, meta=meta)


def _lose_the_database(store):
    for suffix in ("", "-wal", "-shm"):
        Path(str(store.settings.db_path) + suffix).unlink(missing_ok=True)


def _strip_mask(store, record) -> None:
    store.mask_path(record.mask).unlink()
    _rewrite_png(store, record, drop=(PNG_MASK_KEY,))


def _the_pass(store):
    return next(r for r in store.list_images(limit=20)[0] if r.refine is not None)


# --------------------------------------------------------------------- the row


def test_the_golden_region():
    assert mask_crop.region(_mask(), MaskCropSpec()) == mask_crop.Crop(BOX, WORKING)


def test_an_older_table_gains_both_columns(store):
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        conn.execute("ALTER TABLE images DROP COLUMN mask_crop")
        conn.execute("ALTER TABLE images DROP COLUMN mask_crop_box")
        conn.commit()
        assert not {"mask_crop", "mask_crop_box"} & {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    source, made = _gallery()
    record = store.get_image(made.id)
    assert (record.mask_crop.key(), record.mask_crop_box, record.mask_crop_working) == (KEY, BOX, WORKING)


def test_the_row_the_chunk_and_the_parameters_line(store):
    source, made = _gallery()
    assert (made.mask_crop.key(), made.mask_crop_box, made.mask_crop_working) == (KEY, BOX, WORKING)
    with store._session() as conn:
        row = conn.execute("SELECT mask_crop, mask_crop_box FROM images WHERE id = ?", (made.id,)).fetchone()
    assert tuple(row) == (KEY, BOX_TEXT)
    with Image.open(store.settings.outputs_dir / made.filename) as image:
        meta = json.loads(image.info[PNG_METADATA_KEY])
        parameters = image.info["parameters"]
    assert meta["mask_crop"] == {"rule": "bbox16", "pad": 32, "tier": 1, "box": list(BOX), "working": list(WORKING)}
    assert parameters.endswith(", Mask crop: bbox16:32@1")
    # The served record says the region, as lists.
    dumped = made.model_dump(mode="json")
    assert (dumped["mask_crop"], dumped["mask_crop_box"], dumped["mask_crop_working"]) == (
        {"rule": "bbox16", "pad": 32, "tier": 1}, list(BOX), list(WORKING),
    )


def test_the_backends_region_is_what_is_recorded(store):
    # The region the call ran (`last_mask_crop()`), a result like the peak.
    source = save_source("source", 1)
    ran = mask_crop.Crop((0, 0, 256, 256), (1024, 1024))
    made = _cropped(source, region=ran)
    assert (made.mask_crop_box, made.mask_crop_working) == ((0, 0, 256, 256), (1024, 1024))


def test_an_uncropped_inpaint_and_a_plain_render_say_so_by_absence(store):
    source, made = _gallery(crop=False)
    assert (made.mask_crop, made.mask_crop_box, made.mask_crop_working) == (None, None, None)
    for record in (source, made):
        with Image.open(store.settings.outputs_dir / record.filename) as image:
            assert "mask_crop" not in json.loads(image.info[PNG_METADATA_KEY])
            assert "Mask crop" not in image.info["parameters"]
    # Never without its mask: a crop with an outpaint the save never sees
    # (no stamp, so no mask) is no crop either.
    plain = store.save_image(
        Image.new("RGB", (288, 256)),
        GenerateRequest(prompt="x", width=288, height=256, refine=source.id,
                        outpaint={"left": 32}, mask_crop={}),
        seed=3, duration_ms=1, model_id="test/model",
    )
    assert (plain.mask, plain.mask_crop, plain.mask_crop_box) == (None, None, None)


def test_the_record_holds_the_region_to_its_spec_and_picture(store):
    source, made = _gallery()
    base = made.model_dump()
    from backend.schemas import ImageRecord

    for change in ({"mask_crop": None}, {"mask_crop_working": None}, {"mask_crop_box": (0, 0, 272, 256)},
                   {"mask_crop_box": (16, 32, 16, 128)}):
        with pytest.raises(ValueError):
            ImageRecord(**{**base, **change})


def test_the_box_codec_has_one_spelling():
    crop = storage.parse_mask_crop_box(BOX_TEXT)
    assert crop == mask_crop.Crop(BOX, WORKING) and storage.mask_crop_box(crop) == BOX_TEXT
    for bad in ("016,32,144,128>1200x896", "16,32,16,128>1200x896", "16,32,144,128", "x", None, 7):
        with pytest.raises(ValueError):
            storage.parse_mask_crop_box(bad)
        assert storage.stored_mask_crop_box(bad) is None


def test_the_chunk_codec():
    spec = MaskCropSpec()
    chunk = {"mask_crop": storage.mask_crop_chunk(spec, mask_crop.Crop(BOX, WORKING))}
    assert storage.chunk_mask_crop(chunk) == (spec, mask_crop.Crop(BOX, WORKING))
    assert storage.chunk_mask_crop({"mask_crop": {"rule": "bbox16", "pad": 32, "tier": 1}}) == (spec, None)
    for bad in ({"rule": "bbox16", "pad": 30, "tier": 1}, {"rule": "other", "pad": 32, "tier": 1},
                {"rule": "bbox16", "pad": "32", "tier": 1}, {"rule": "bbox16", "pad": 32, "tier": 1, "box": [1]},
                {"rule": "bbox16", "pad": 32, "tier": 1, "box": [16, 32, 144, 128]}, "bbox16:32@1"):
        assert storage.chunk_mask_crop({"mask_crop": bad}) is None and storage.malformed_mask_crop({"mask_crop": bad})
    assert not storage.malformed_mask_crop({}) and storage.chunk_mask_crop({}) is None


def test_index_file_reads_the_chunk_and_holds_the_invariant(store):
    source, made = _gallery()
    path = store.settings.outputs_dir / made.filename
    meta = png_metadata(path)
    now = datetime.now(timezone.utc)
    cases = {
        "copy.png": (meta, (KEY, BOX)),
        "no-mask.png": ({**meta, "mask": None}, (None, None)),
        "no-source.png": ({**meta, "refine": None}, (None, None)),
        "malformed.png": ({**meta, "mask_crop": {**meta["mask_crop"], "pad": 30}}, (None, None)),
        # A region outside the picture is a result dropped, the spec kept.
        "outside.png": ({**meta, "mask_crop": {**meta["mask_crop"], "box": [0, 0, 512, 256]}}, (KEY, None)),
    }
    for name, (chunk, expected) in cases.items():
        path.with_name(name).write_bytes(path.read_bytes())
        record = store.index_file(name, chunk, 256, 256, now)
        assert ((record.mask_crop.key() if record.mask_crop else None), record.mask_crop_box) == expected, name


# --------------------------------------------------------------------- reindex


def test_a_png_only_reindex_brings_the_crop_back(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert (counts["indexed"], counts["mask_crop_malformed"], counts["mask_crops_recovered"]) == (2, 0, 0)
    row = _the_pass(store)
    assert (row.mask_crop.key(), row.mask_crop_box, row.mask_crop_working, row.mask) == (KEY, BOX, WORKING, made.mask)


def test_a_png_only_reindex_proves_the_region_from_the_embedded_mask(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    # The masks directory lost with the database: the PNG's own copy proves it.
    store.mask_path(made.mask).unlink()
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["mask_crop_malformed"] == 0
    assert _the_pass(store).mask_crop_box == BOX


def test_a_region_that_is_not_the_masks_is_left_off_and_counted(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _with_chunk(store, made, {"rule": "bbox16", "pad": 32, "tier": 1, "box": [0, 0, 256, 256],
                              "working": [1024, 1024]})
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["mask_crop_malformed"] == 1
    row = _the_pass(store)
    assert (row.mask_crop, row.mask_crop_box, row.mask) == (None, None, made.mask)
    assert "mask_crop" not in png_metadata(store.settings.outputs_dir / made.filename)


def test_a_malformed_crop_is_left_off_and_counted(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _with_chunk(store, made, {"rule": "bbox16", "pad": 30, "tier": 1})
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    # Left off, never recovered from the A1111 line in its place.
    assert (counts["mask_crop_malformed"], counts["mask_crops_recovered"]) == (1, 0)
    assert _the_pass(store).mask_crop is None


def test_a_region_not_recorded_is_drawn_again_from_the_mask(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _with_chunk(store, made, {"rule": "bbox16", "pad": 32, "tier": 1})
    _lose_the_database(store)
    reindex.reindex(store.settings.outputs_dir)
    row = _the_pass(store)
    assert (row.mask_crop.key(), row.mask_crop_box) == (KEY, BOX)
    assert png_metadata(store.settings.outputs_dir / made.filename)["mask_crop"]["box"] == list(BOX)


def test_without_its_mask_a_recorded_crop_is_kept_as_given(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _strip_mask(store, made)
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["unresolved_masks"] == 1
    row = _the_pass(store)
    assert (row.mask_crop.key(), row.mask_crop_box) == (KEY, BOX)


def test_a_png_only_reindex_recovers_a_crop_an_older_build_stripped(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["mask_crops_recovered"] == 1
    row = _the_pass(store)
    assert (row.mask_crop.key(), row.mask_crop_box, row.mask_crop_working) == (KEY, BOX, WORKING)
    assert png_metadata(store.settings.outputs_dir / made.filename)["mask_crop"]["pad"] == 32


def test_no_recovery_without_the_mask_to_draw_it_from(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    _strip_mask(store, made)
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["mask_crops_recovered"] == 0
    assert _the_pass(store).mask_crop is None


def _parameters(store, record) -> str:
    with Image.open(store.settings.outputs_dir / record.filename) as image:
        return image.info["parameters"]


@pytest.mark.parametrize("chunk", [
    {"rule": "bbox16", "pad": 32, "tier": 1, "box": [0, 0, 256, 256], "working": [1024, 1024]},
    {"rule": "bbox16", "pad": 32, "tier": 1, "box": [0, 0, 64]},
    {"rule": "bbox16", "pad": 30, "tier": 1},
])
def test_a_crop_left_off_stays_off_on_the_next_reindex(store, monkeypatch, chunk):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made = _gallery()
    before = _parameters(store, made)
    _with_chunk(store, made, chunk)
    _lose_the_database(store)
    assert reindex.reindex(store.settings.outputs_dir)["mask_crop_malformed"] == 1
    # Its `Mask crop:` is out of the A1111 line, and nothing else is.
    assert _parameters(store, made) == before.replace(f", Mask crop: {KEY}", "")
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert (counts["mask_crop_malformed"], counts["mask_crops_recovered"]) == (0, 0)
    assert _the_pass(store).mask_crop is None


def test_a_prompt_naming_a_crop_is_never_one(store, monkeypatch, tmp_path):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = save_source("source", 1)
    digest, _ = storage.store_mask(_mask())
    words = f"a red scarf, Mask crop: {KEY}, and more"
    made = storage.save_image(
        Image.new("RGB", (256, 256), (200, 10, 10)),
        GenerateRequest(prompt=words, negative_prompt=f"blurry\nSteps: 9, Mask crop: {KEY}", width=256,
                        height=256, steps=28, refine=source.id, strength=0.6, mask=digest),
        seed=2, duration_ms=10, model_id="test/model", job_id="job-one", run_id="run-one",
    )
    parameters = _parameters(store, made)
    assert f"Mask crop: {KEY}" in parameters
    assert storage.parameters_mask_crop(parameters) is None
    assert storage.recovered_mask_crop(parameters, _mask()) is None
    # Neither the reindex nor an import gives the whole-picture inpaint a crop.
    archive = bundles.export_bundle(keep_gallery=True)
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert (counts["mask_crops_recovered"], _the_pass(store).mask_crop) == (0, None)
    use_destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert imported["mask_crops_recovered"] == 0
    assert next(r for r in store.list_images()[0] if r.refine is not None).mask_crop is None


def test_only_the_settings_line_names_a_crop():
    line = f"Steps: 28, CFG scale: 4.5, Seed: 2, Size: 256x256, Model: m, Mask crop: {KEY}"
    block = f"a prompt, Mask crop: {KEY}\nNegative prompt: \n{line}"
    assert storage.parameters_mask_crop(block) == MaskCropSpec(pad=32)
    assert storage.without_mask_crop_parameters(block) == block[: -len(f", Mask crop: {KEY}")]
    # No settings line, or one without the key: nothing to read or take out.
    for text in (f"a prompt, Mask crop: {KEY}", f"a prompt\nNegative prompt: Mask crop: {KEY}\nSteps: 28"):
        assert storage.parameters_mask_crop(text) is None
        assert storage.without_mask_crop_parameters(text) is None


def test_the_rewritten_a1111_line_keeps_its_text(tmp_path):
    """`reindex._with_auradream_metadata` with an A1111 block: Latin-1 as
    tEXt, anything else as UTF-8 iTXt, each read back as written."""
    from io import BytesIO

    for text in ("a scarf\nNegative prompt: \nSteps: 28", "a scarf ☃ café\nNegative prompt: \nSteps: 28"):
        stream = BytesIO()
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "old")
        Image.new("RGB", (4, 4)).save(stream, format="PNG", pnginfo=info)
        rewritten = reindex._with_auradream_metadata(stream.getvalue(), {"prompt": "new"}, text)
        with Image.open(BytesIO(rewritten)) as image:
            assert (image.info["parameters"], json.loads(image.info[PNG_METADATA_KEY])) == (text, {"prompt": "new"})


def test_a_directory_reindex_drops_the_crop_with_its_source(store, monkeypatch, tmp_path):
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
    assert counts["indexed"] == 1
    (row,) = store.list_images()[0]
    assert (row.refine, row.mask, row.mask_crop, row.mask_crop_box) == (None, None, None, None)
    assert "mask_crop" not in png_metadata(mine.outputs_dir / made.filename)


def _other_gallery(store, monkeypatch, tmp_path):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    elsewhere = tmp_path / "backup" / "outputs"
    elsewhere.mkdir(parents=True)
    source_db = elsewhere.parent / "auradream.db"
    monkeypatch.setattr(storage, "settings", dataclasses.replace(mine, db_path=source_db, outputs_dir=elsewhere))
    storage.init_db()
    return mine, elsewhere, source_db


def test_a_database_merge_carries_and_drops_the_crop(store, monkeypatch, tmp_path):
    mine, elsewhere, source_db = _other_gallery(store, monkeypatch, tmp_path)
    source, made = _gallery()
    orphan_source = save_source("gone source", 5)
    orphan = _cropped(orphan_source, seed=6)
    with storage._session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (orphan_source.id,))
    monkeypatch.setattr(storage, "settings", mine)
    counts = reindex.reindex(elsewhere, database=source_db)
    assert (counts["indexed"], counts["mask_crop_malformed"]) == (3, 0)
    rows = {r.filename: r for r in store.list_images(limit=10)[0]}
    carried, dropped = rows[made.filename], rows[orphan.filename]
    assert (carried.mask_crop.key(), carried.mask_crop_box, carried.mask_crop_working) == (KEY, BOX, WORKING)
    assert (dropped.refine, dropped.mask, dropped.mask_crop, dropped.mask_crop_box) == (None, None, None, None)
    assert png_metadata(mine.outputs_dir / made.filename)["mask_crop"]["box"] == list(BOX)
    assert "mask_crop" not in png_metadata(mine.outputs_dir / orphan.filename)


@pytest.mark.parametrize("change, expected", [
    ("UPDATE images SET mask_crop = 'bbox16:30@1' WHERE id = ?", (None, None, 1)),
    ("UPDATE images SET mask_crop_box = '0,0,256,256>1024x1024' WHERE id = ?", (None, None, 1)),
    # A bad box is a result: drawn again from the mask.
    ("UPDATE images SET mask_crop_box = 'nonsense' WHERE id = ?", (KEY, BOX, 0)),
])
def test_a_database_merge_proves_what_it_copies(store, monkeypatch, tmp_path, change, expected):
    mine, elsewhere, source_db = _other_gallery(store, monkeypatch, tmp_path)
    source, made = _gallery()
    with storage._session() as conn:
        conn.execute(change, (made.id,))
    monkeypatch.setattr(storage, "settings", mine)
    counts = reindex.reindex(elsewhere, database=source_db)
    row = next(r for r in store.list_images(limit=10)[0] if r.filename == made.filename)
    assert ((row.mask_crop.key() if row.mask_crop else None), row.mask_crop_box,
            counts["mask_crop_malformed"]) == expected
    # One left off leaves no `Mask crop:` on the copy's A1111 line to bring it back.
    with Image.open(mine.outputs_dir / made.filename) as image:
        assert ("Mask crop:" in image.info["parameters"]) == (expected[0] is not None)


def test_a_database_merge_recovers_from_the_a1111_line(store, monkeypatch, tmp_path):
    mine, elsewhere, source_db = _other_gallery(store, monkeypatch, tmp_path)
    source, made = _gallery()
    _as_an_older_build_left_it(storage, made)
    monkeypatch.setattr(storage, "settings", mine)
    counts = reindex.reindex(elsewhere, database=source_db)
    assert counts["mask_crops_recovered"] == 1
    row = next(r for r in store.list_images(limit=10)[0] if r.filename == made.filename)
    assert (row.mask_crop.key(), row.mask_crop_box) == (KEY, BOX)
    assert png_metadata(mine.outputs_dir / made.filename)["mask_crop"]["pad"] == 32


# --------------------------------------------------------------------- bundles


def test_a_bundle_carries_the_columns_and_the_chunk(store, monkeypatch, tmp_path):
    source, made = _gallery()
    archive = bundles.export_bundle(keep_gallery=True)
    use_destination(store, monkeypatch, tmp_path)
    save_source("existing", 99)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["masks_missing"], imported["mask_crops_recovered"]) == (2, 0, 0)
    row = next(r for r in store.list_images()[0] if r.mask_crop is not None)
    assert (row.mask_crop.key(), row.mask_crop_box, row.mask_crop_working) == (KEY, BOX, WORKING)
    assert png_metadata(store.settings.outputs_dir / row.filename)["mask_crop"]["box"] == list(BOX)
    # Edit-free and crop-carrying, the bundle stays version 1.
    import zipfile

    from backend.bundle_archive import _MANIFEST

    with zipfile.ZipFile(archive["path"]) as bundle:
        assert json.loads(bundle.read(_MANIFEST))["version"] == 1


def test_a_bundle_without_the_source_clears_the_crop(store, monkeypatch, tmp_path):
    source, made = _gallery()
    archive = bundles.export_bundle(keep_gallery=True, ids=[made.id])
    use_destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(Path(archive["path"]))
    (row,) = store.list_images()[0]
    assert (row.refine, row.mask, row.mask_crop, row.mask_crop_box) == (None, None, None, None)
    assert "mask_crop" not in png_metadata(store.settings.outputs_dir / row.filename)


@pytest.mark.parametrize("change", ["garbage", "bad pad", "no mask", "box without crop", "box outside",
                                    "box misspelled"])
def test_a_bundle_row_with_a_broken_crop_is_refused(store, change):
    source, made = _gallery()
    update = {
        "garbage": "UPDATE images SET mask_crop = '../x' WHERE id = ?",
        "bad pad": "UPDATE images SET mask_crop = 'bbox16:30@1' WHERE id = ?",
        "no mask": "UPDATE images SET mask = NULL WHERE id = ?",
        "box without crop": "UPDATE images SET mask_crop = NULL WHERE id = ?",
        "box outside": "UPDATE images SET mask_crop_box = '0,0,512,256>1024x512' WHERE id = ?",
        "box misspelled": "UPDATE images SET mask_crop_box = '016,32,144,128>1200x896' WHERE id = ?",
    }[change]
    with store._session() as conn:
        conn.execute(update, (made.id,))
    with pytest.raises(bundles.BundleError, match=f"invalid mask crop on image {made.id}"):
        bundles.export_bundle(keep_gallery=True)
    row = {key: getattr(made, key, None) for key in storage._COLUMNS}
    row.update(id=made.id, mask_crop="nope", mask_crop_box=None, mask=made.mask, refine=source.id,
               binned_at=None, reference=None, variables=None, outpaint=None, execution=None, edit=None)
    with pytest.raises(bundles.BundleError, match="invalid mask crop"):
        bundle_images._validate_row(row, made.width, made.height, made.content_hash)


def test_an_import_refuses_a_region_that_is_not_its_masks(store, monkeypatch, tmp_path):
    source, made = _gallery()
    with store._session() as conn:
        conn.execute("UPDATE images SET mask_crop_box = '0,0,256,256>1024x1024' WHERE id = ?", (made.id,))
    archive = bundles.export_bundle(keep_gallery=True)
    use_destination(store, monkeypatch, tmp_path)
    with pytest.raises(bundles.BundleError, match="does not match its mask under rule bbox16"):
        bundles.import_bundle(Path(archive["path"]))
    assert store.list_images()[0] == []


def test_an_import_draws_a_missing_region_again(store, monkeypatch, tmp_path):
    source, made = _gallery()
    with store._session() as conn:
        conn.execute("UPDATE images SET mask_crop_box = NULL WHERE id = ?", (made.id,))
    archive = bundles.export_bundle(keep_gallery=True)
    use_destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(Path(archive["path"]))
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert (row.mask_crop.key(), row.mask_crop_box) == (KEY, BOX)


def test_an_import_keeps_a_crop_whose_mask_is_lost(store, monkeypatch, tmp_path):
    source, made = _gallery()
    _strip_mask(store, made)
    archive = bundles.export_bundle(keep_gallery=True)
    use_destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert imported["masks_missing"] == 1
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert (row.mask_crop.key(), row.mask_crop_box) == (KEY, BOX)


def test_an_import_recovers_a_crop_an_older_build_stripped(store, monkeypatch, tmp_path):
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    archive = bundles.export_bundle(keep_gallery=True)
    use_destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert imported["mask_crops_recovered"] == 1
    row = next(r for r in store.list_images()[0] if r.refine is not None)
    assert (row.mask_crop.key(), row.mask_crop_box, row.mask_crop_working) == (KEY, BOX, WORKING)
    assert png_metadata(store.settings.outputs_dir / row.filename)["mask_crop"]["pad"] == 32


def test_an_import_does_not_recover_a_spec_this_build_refuses(store, monkeypatch, tmp_path):
    source, made = _gallery()
    _as_an_older_build_left_it(store, made)
    with Image.open(store.settings.outputs_dir / made.filename) as image:
        parameters = image.info["parameters"].replace("Mask crop: bbox16:32@1", "Mask crop: bbox16:30@1")
    _rewrite_png(store, made, parameters=parameters)
    archive = bundles.export_bundle(keep_gallery=True)
    use_destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert imported["mask_crops_recovered"] == 0
    assert next(r for r in store.list_images()[0] if r.refine is not None).mask_crop is None


# --------------------------------------------------------------------- reruns


def test_the_rerun_carriers_and_droppers(store):
    source, made = _gallery()
    record = store.get_image(made.id)
    # Re-render, Swap tags and More seeds go through `cell_from_record`.
    cell = grid.cell_from_record(record)
    assert cell.request.mask_crop == MaskCropSpec()
    # (Mass refine drops it with the mask: over HTTP below.) Then refine's stand-in likewise.
    stand = then_refine.stand_in(cell.request, ThenRefine(scale=1.0, strength=0.5))
    assert (stand.mask, stand.mask_crop) == (None, None)
    # The checked inputs key on the crop: two paddings are two checks.
    other = cell.request.model_copy(update={"mask_crop": MaskCropSpec(pad=64)})
    assert submission.input_spec(cell.request) != submission.input_spec(other)
    assert len(submission.NO_INPUTS) == len(submission.input_spec(cell.request))
    assert submission.input_spec(cell.request)[7] == MaskCropSpec()


# --------------------------------------------------------------- search, price


def test_search_and_stats_find_a_cropped_pass(store):
    source, made = _gallery()
    plain = _cropped(source, crop=False, seed=4)
    ids = lambda q: {r.id for r in store.list_images(search=q)[0]}  # noqa: E731
    assert ids("maskcrop:yes") == {made.id}
    assert ids("maskcrop:no inpaint:yes") == {plain.id}
    assert source.id in ids("maskcrop:no")
    with pytest.raises(ValueError, match="maskcrop"):
        store.list_images(search="maskcrop:maybe")
    rates = gallery_reports.keep_rates(by=["maskcrop"])
    assert KEY in {row["maskcrop"] for row in rates}


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


def test_a_cropped_pass_is_priced_by_its_tier_never_by_its_size(store):
    source = save_source("source", 1)
    # Three whole-picture inpaints and two cropped ones, each at 10 steps
    # and strength 0.6 (6 denoised steps on SDXL).
    for seed, ms in ((1, 1000), (2, 1100), (3, 1200)):
        _cropped(source, crop=False, seed=seed, duration_ms=ms, model_id=ANIMAGINE, steps=10)
    for seed, ms in ((4, 4000), (5, 4200)):
        _cropped(source, seed=seed, duration_ms=ms, model_id=ANIMAGINE, steps=10)
    whole = gallery_reports.refine_cost_ms(256, 256, ANIMAGINE, inpaint=True)
    crops = gallery_reports.mask_crop_cost_ms(ANIMAGINE)
    assert list(whole.values()) == [1100] and set(crops) == {(1, steps) for steps in whole}
    assert list(crops.values()) == [4200]
    record = store.get_image(store.list_images(search="maskcrop:yes")[0][0].id)
    cropped = grid.cell_from_record(record)
    uncropped = dataclasses.replace(cropped, request=cropped.request.model_copy(update={"mask_crop": None}))
    assert gallery_reports.estimate_seconds([cropped], None, inpaint_costs=whole, crop_costs=crops) == 4.2
    assert gallery_reports.estimate_seconds([uncropped], None, inpaint_costs=whole, crop_costs=crops) == 1.1
    # No cropped history, no price - never the source size's.
    assert gallery_reports.estimate_seconds([cropped], None, inpaint_costs=whole) is None
    prices, _ = gallery_reports.execution_estimates([cropped, uncropped], ANIMAGINE, disk=False)
    assert prices[(256, 256)][2:] == (whole, crops)
    assert gallery_reports.estimate_seconds([cropped, uncropped], None, by_size=prices) == 5.3
    # A three-tuple from before is read as no cropped history.
    assert gallery_reports.estimate_seconds([cropped], None, by_size={(256, 256): (None, {}, whole)}) is None


# ------------------------------------------------------------------- over HTTP


def _png_bytes(box=(0, 0, 128, 64)) -> bytes:
    import io

    image = Image.new("L", (256, 256), 0)
    image.paste(255, box)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _http_inpainted(client, store):
    run_one(client, width=256, height=256)
    record = client.get("/api/gallery").json()["items"][0]
    response = client.post("/api/masks", params={"source": record["id"]}, content=_png_bytes(),
                           headers={"content-type": "image/png"})
    digest = response.json()["hash"]
    body = {"prompt": "1girl, red hair, long hair", "width": 256, "height": 256, "steps": 10,
            "refine": record["id"], "strength": 0.6, "mask": digest, "mask_crop": {"pad": 32}, "seed": 9}
    _, events = run_one(client, **body)
    assert events[-1]["status"] == "done", events[-1]
    return record, events[-1]["images"][0], digest


def test_a_job_records_the_region_its_backend_ran(client, fake, store):
    record, made, digest = _http_inpainted(client, store)
    expected = mask_crop.region(_mask((0, 0, 128, 64)), MaskCropSpec())
    assert fake.last_mask_crop() == expected
    assert (made["mask_crop"], made["mask_crop_box"], made["mask_crop_working"]) == (
        {"rule": "bbox16", "pad": 32, "tier": 1}, list(expected.box), list(expected.working),
    )
    meta = png_metadata(store.settings.outputs_dir / made["filename"])
    assert meta["mask_crop"]["working"] == list(expected.working)


def test_re_render_swap_tags_and_more_seeds_carry_the_crop(client, fake, store):
    record, made, digest = _http_inpainted(client, store)
    response = client.post("/api/more-seeds", json={"ids": [made["id"]], "count": 2})
    assert response.status_code == 200, response.text
    assert stream_job(client, response.json()["job_id"])[-1]["status"] == "done"
    assert [r.mask_crop for r in fake.requests[-2:]] == [MaskCropSpec()] * 2
    swap = [{"find": "red hair, long hair", "replace": "silver hair"}]
    response = client.post("/api/swap-tags", json={"swaps": swap, "ids": [made["id"]]})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    assert (fake.requests[-1].prompt, fake.requests[-1].mask_crop) == ("1girl, silver hair", MaskCropSpec())
    newest = client.get("/api/gallery").json()["items"][0]
    assert newest["mask_crop_box"] == made["mask_crop_box"]
    # Re-render keys the record with its crop, so the cropped row it made
    # is the one already there: were the crop dropped, the uncropped key
    # would match no row and a render would be queued.
    before = len(fake.requests)
    response = client.post("/api/rerender", json={"ids": [made["id"]], "width": 256, "height": 256})
    assert response.status_code == 409 and "already rendered" in response.text
    assert len(fake.requests) == before
    cell = grid.cell_from_record(store.get_image(made["id"]))
    assert cell.request.mask_crop == MaskCropSpec()
    assert f"{digest}+{KEY}" in grid.cell_key(cell)


def test_mass_refine_drops_the_crop_with_the_mask(client, fake, store):
    record, made, digest = _http_inpainted(client, store)
    response = client.post("/api/refine/batch", json={"ids": [made["id"]], "strength": 0.5, "scale": 1.5,
                                                       "skip_passes": False})
    assert response.status_code == 200, response.text
    stream_job(client, response.json()["job_id"])
    ran = fake.requests[-1]
    assert (ran.refine, ran.mask, ran.mask_crop) == (made["id"], None, None)
    assert client.get("/api/gallery").json()["items"][0]["mask_crop"] is None


def test_load_settings_from_png_drops_the_crop_whatever_it_is(client, fake, store):
    record, made, digest = _http_inpainted(client, store)
    with Image.open(store.settings.outputs_dir / made["filename"]) as handle:
        chunks = {key: value for key, value in handle.info.items() if isinstance(value, str)}
    chunks[PNG_MASK_KEY] = ""
    meta = json.loads(chunks[PNG_METADATA_KEY])
    # A later build's rule, an off-grid padding, a crop with no mask beside
    # it: the panel never takes the crop, so none refuses the file.
    for crop in (meta["mask_crop"], {"rule": "bbox24", "pad": 32, "tier": 1},
                 {"rule": "bbox16", "pad": 30, "tier": 2}, "nonsense"):
        for changed in ({**meta, "mask_crop": crop}, {**meta, "mask_crop": crop, "mask": None}):
            assert png_record({PNG_METADATA_KEY: json.dumps(changed)}, "made.png") == changed
            response = client.post("/api/png/settings", json={
                "name": "made.png", "chunks": {**chunks, PNG_METADATA_KEY: json.dumps(changed)},
            })
            assert response.status_code == 200, response.text
            settings = response.json()["settings"]
            assert (settings["mask"], settings.get("mask_crop"), settings["refine"]) == (None, None, None)
            assert settings["prompt"] == meta["prompt"]
