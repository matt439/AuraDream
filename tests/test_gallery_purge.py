"""Confirmed permanent deletion of exactly the recycle bin's picked ids."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from backend import gallery_delete, gallery_lock
from tests.bundle_helpers import holding_child, save_source


def test_purge_is_exact_and_normalizes_order_and_duplicates(store):
    picked, other, live = (save_source(seed=seed) for seed in (1, 2, 3))
    store.bin_images([picked.id, other.id])
    result = gallery_delete.purge_images([picked.id, picked.id], [picked.id])
    assert result == {"records": 1, "files": 1, "ids": [picked.id], "masks": 0}
    assert not (store.settings.outputs_dir / picked.filename).exists()
    assert not store._thumb_path(store.settings.outputs_dir / picked.filename).exists()
    assert (store.settings.outputs_dir / other.filename).exists()
    assert (store.settings.outputs_dir / live.filename).exists()
    assert store.list_images(binned=True)[1] == store.list_images()[1] == 1


@pytest.mark.parametrize("change", ["selection", "restore", "delete"])
def test_a_changed_pick_refuses_the_whole_purge(store, change):
    first, second = save_source(seed=1), save_source(seed=2)
    store.bin_images([first.id, second.id])
    expected = [first.id, second.id]
    wanted = expected
    if change == "selection":
        wanted = [second.id, 9999]  # Same count of different identities.
    elif change == "restore":
        store.restore_images([second.id])
    else:
        store.delete_image(second.id)
    with pytest.raises(gallery_delete.PurgeChanged) as changed:
        gallery_delete.purge_images(wanted, expected)
    assert changed.value.not_binned == ([second.id] if change == "restore" else [])
    assert changed.value.missing == ([second.id] if change == "delete" else [])
    assert (store.settings.outputs_dir / first.filename).exists()
    assert any(row.id == first.id for row in store.list_images(binned=True)[0])


def test_every_chunk_is_validated_before_any_requested_row_is_deleted(store):
    # A live pick after the first 500 must refuse the first chunk too. Tiny
    # synthetic images keep this test's entire gallery under a few MiB.
    rows = [save_source(seed=seed) for seed in range(1, 502)]
    store.bin_images([row.id for row in rows[:-1]])
    ids = [row.id for row in rows]
    with pytest.raises(gallery_delete.PurgeChanged) as changed:
        gallery_delete.purge_images(ids, ids[::-1])
    assert changed.value.not_binned == [rows[-1].id]
    assert store.list_images(binned=True)[1] == 500
    assert all((store.settings.outputs_dir / row.filename).exists() for row in rows)
    store.bin_images([rows[-1].id])
    result = gallery_delete.purge_images(ids[::-1], ids)
    assert result["ids"] == ids and result["files"] == 501
    assert store.list_images(binned=True)[1] == 0


def test_purge_keeps_shared_held_and_unsettled_masks(store):
    source = save_source(seed=1)
    masks = [store.store_mask(Image.new("L", (256, 256), level))[0] for level in (64, 128, 192, 255)]
    shared, held, orphan, young = masks
    with store._session() as conn:
        conn.execute("UPDATE masks SET created_at = ? WHERE hash != ?", (
            (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(), young,
        ))
    kept = save_source("kept inpaint", 2, refine=source.id, strength=0.6, mask=shared)
    picked = [save_source("picked inpaint", at + 3, refine=source.id, strength=0.6, mask=digest)
              for at, digest in enumerate(masks)]
    ids = [row.id for row in picked]
    store.bin_images(ids)
    result = gallery_delete.purge_images(ids, ids, in_flight=lambda: {held})
    assert result["masks"] == 1
    assert store.get_image(kept.id).mask == shared
    assert all(store.mask_path(digest).exists() for digest in (shared, held, young))
    with pytest.raises(ValueError, match="no stored mask"):
        store.mask_path(orphan)


def test_purge_fails_fast_when_another_process_has_reserved_the_gallery(store):
    row = save_source()
    store.bin_images([row.id])
    child = holding_child(store.settings.db_path)
    try:
        with pytest.raises(gallery_lock.GalleryBusy):
            gallery_delete.purge_images([row.id], [row.id])
    finally:
        child.stdin.close()
        child.wait(timeout=15)
    assert (store.settings.outputs_dir / row.filename).exists()


def test_an_empty_storage_purge_cannot_mean_the_whole_bin(store):
    row = save_source()
    store.bin_images([row.id])
    with pytest.raises(gallery_delete.PurgeChanged):
        gallery_delete.purge_images([], [])
    assert store.list_images(binned=True)[1] == 1
