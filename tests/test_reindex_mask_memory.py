"""Reindex reloads verified masks for crop proof without retaining their pixels."""
from __future__ import annotations

from contextlib import closing
from io import StringIO
from weakref import ref

import pytest
from PIL import Image

from backend import detailer, reindex, storage
from backend.metadata import PNG_MASK_KEY
from tests.bundle_helpers import png_metadata, save_source, use_destination
from tests.test_detailer_carry import chain
from tests.test_mask_crop_storage import BOX, WORKING, _as_an_older_build_left_it, _cropped, _rewrite_png
from tests.test_outpaint_storage import _outpainted, _strip_mask


@pytest.mark.parametrize("database", [False, True])
@pytest.mark.parametrize("directory_count", [24, 96])
def test_reindex_keeps_decoded_mask_memory_bounded_and_preserves_proof(
    store, monkeypatch, tmp_path, database, directory_count,
):
    origin = store.settings
    mask_directory = origin.outputs_dir / storage.MASKS_DIR
    mask_directory.mkdir()
    hashes = set()
    for level in range(1, directory_count + 1):
        with closing(Image.new("L", (128, 128), level)) as mask:
            digest = storage.mask_hash(mask)
            mask.save(mask_directory / f"{digest}.png")
            hashes.add(digest)

    source = save_source("source", seed=101)
    embedded = _cropped(source, seed=102, crop=False)
    depends = _cropped(embedded, seed=103)
    _as_an_older_build_left_it(store, depends)
    _rewrite_png(store, depends, drop=(PNG_MASK_KEY,))
    store.mask_path(embedded.mask).unlink()

    standard = _cropped(source, seed=104, box=(64, 80, 120, 128), crop=False)
    _rewrite_png(store, standard, drop=(PNG_MASK_KEY,))
    outpainted = _outpainted(source, seed=105)
    _strip_mask(store, outpainted)
    _detail_source, detailer_results = chain(store)
    for result in detailer_results:
        _strip_mask(store, result)
    hashes.update((embedded.mask, standard.mask, outpainted.mask, *detailer_results[-1].detailer.plan.masks))

    target = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", target)
    live_masks = []
    peak = 0
    read_result = reindex._read

    def decoded(reference):
        mask = reference()
        if mask is None:
            return False
        try:
            return mask.im is not None
        except ValueError:  # Pillow raises when its pixel buffer has been closed.
            return False

    def track(original):
        def tracked(*args, **kwargs):
            nonlocal peak
            mask = original(*args, **kwargs)
            if mask is not None:
                live_masks[:] = [reference for reference in live_masks if decoded(reference)]
                live_masks.append(ref(mask))
                peak = max(peak, len(live_masks))
            return mask
        return tracked

    def read_after_releasing_masks(*args, **kwargs):
        assert not any(map(decoded, live_masks))
        return read_result(*args, **kwargs)

    for module, name in (
        (reindex, "_open_mask"), (reindex, "_embedded_mask"), (reindex, "region_mask"),
        (detailer, "region_mask"), (storage, "rebuilt_outpaint_mask"),
    ):
        monkeypatch.setattr(module, name, track(getattr(module, name)))
    monkeypatch.setattr(reindex, "_read", read_after_releasing_masks)
    source_db = origin.db_path if database else None
    preview = reindex.reindex(origin.outputs_dir, database=source_db, dry_run=True, out=StringIO())
    assert preview["masks"] == len(hashes) == directory_count + 5
    assert preview["unresolved_masks"] == 0
    assert preview["unresolved_references"] == 0
    assert preview["outpaint_masks_rebuilt"] == 1
    assert preview["detailer_masks_rebuilt"] == 2
    assert preview["mask_crops_recovered"] == 1
    assert not (target.outputs_dir / storage.MASKS_DIR).exists()

    imported = reindex.reindex(origin.outputs_dir, database=source_db, out=StringIO())
    assert imported == preview
    assert peak == 1
    assert not any(map(decoded, live_masks))
    with storage._session() as conn:
        assert {row[0] for row in conn.execute("SELECT hash FROM masks")} == hashes
    installed = list((target.outputs_dir / storage.MASKS_DIR).glob("*.png"))
    assert {path.stem for path in installed} == hashes
    for path in installed:
        assert (mask := storage.open_mask(path, path.stem)) is not None
        mask.close()
    rows = {row.filename: row for row in storage.list_images()[0]}
    restored = rows[depends.filename]
    assert restored.mask == embedded.mask and restored.refine == rows[embedded.filename].id
    assert (restored.mask_crop_box, restored.mask_crop_working) == (BOX, WORKING)
    assert png_metadata(target.outputs_dir / depends.filename)["mask_crop"]["box"] == list(BOX)
    assert rows[standard.filename].mask == standard.mask
    assert rows[outpainted.filename].outpaint == outpainted.outpaint
    for result in detailer_results:
        manifest = rows[result.filename].detailer
        assert manifest.group_id == result.detailer.group_id
        assert manifest.plan.source == result.detailer.plan.source
