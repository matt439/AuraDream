"""Empty recycle bin limited to what was binned more than N days ago
(`GAL#5`): a cutoff fixed by the summary, pinned with its count and ids."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from backend import clear, gallery_delete, gallery_lock, storage
from tests.bundle_helpers import holding_child, save_source


def _age(store, ids, days, *, stamp=None):
    """Backdate these rows' `binned_at` by `days`, or write `stamp` as is."""
    value = stamp if stamp is not None else (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with store._session() as conn:
        conn.executemany("UPDATE images SET binned_at = ? WHERE id = ?", [(value, i) for i in ids])


def test_the_cutoff_is_iso_utc_and_parses_back():
    now = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    cutoff = gallery_delete.bin_cutoff(7, now)
    assert cutoff == "2026-10-12T12:00:00+00:00"
    assert gallery_delete.parse_cutoff(cutoff) == now - timedelta(days=7)
    # Without a zone it is UTC, as every stamp the gallery writes is.
    assert gallery_delete.parse_cutoff("2026-10-12T12:00:00") == now - timedelta(days=7)
    with pytest.raises(ValueError):
        gallery_delete.parse_cutoff("last week")


def test_the_scoped_summary_counts_only_what_is_older_than_the_cutoff(store):
    old, older, fresh, live = (save_source(seed=seed) for seed in (1, 2, 3, 4))
    store.bin_images([old.id, older.id, fresh.id])
    _age(store, [old.id], 8)
    _age(store, [older.id], 40)
    whole = gallery_delete.bin_summary()
    assert (whole["records"], whole["bin_records"], whole["before"]) == (3, 3, None)

    week = gallery_delete.bin_summary(gallery_delete.bin_cutoff(7))
    assert (week["records"], week["bin_records"]) == (2, 3)
    assert week["pin"] == gallery_delete.bin_pin([old.id, older.id]) != whole["pin"]
    assert 0 < week["bytes"] < whole["bytes"]
    month = gallery_delete.bin_summary(gallery_delete.bin_cutoff(30))
    assert (month["records"], month["pin"]) == (1, gallery_delete.bin_pin([older.id]))
    assert gallery_delete.bin_summary(gallery_delete.bin_cutoff(365))["records"] == 0


def test_an_age_scoped_empty_deletes_only_rows_binned_before_the_cutoff(store):
    outputs = store.settings.outputs_dir
    old, fresh, live = (save_source(seed=seed) for seed in (1, 2, 3))
    store.bin_images([old.id, fresh.id])
    _age(store, [old.id], 10)
    before = gallery_delete.bin_cutoff(7)
    found = gallery_delete.bin_summary(before)
    result = gallery_delete.empty_bin(expected=found["records"], pin=found["pin"], before=before)
    assert (result["records"], result["files"], result["ids"], result["masks"]) == (1, 1, [old.id], 0)
    assert not (outputs / old.filename).exists()
    assert not store._thumb_path(outputs / old.filename).exists()
    assert (outputs / fresh.filename).exists() and (outputs / live.filename).exists()
    assert store.get_image(fresh.id, include_binned=True).binned_at is not None
    assert store.get_image(live.id).binned_at is None


def test_the_scoped_empty_is_pinned_to_its_count_and_its_ids(store):
    outputs = store.settings.outputs_dir
    first, second, third = (save_source(seed=seed) for seed in (1, 2, 3))
    store.bin_images([first.id, second.id, third.id])
    _age(store, [first.id, second.id], 10)
    before = gallery_delete.bin_cutoff(7)
    found = gallery_delete.bin_summary(before)
    assert found["records"] == 2

    # A Restore in another tab: one fewer.
    store.restore_images([second.id])
    with pytest.raises(gallery_delete.SelectionChanged) as caught:
        gallery_delete.empty_bin(expected=2, pin=found["pin"], before=before)
    assert (caught.value.found, caught.value.expected) == (1, 2)
    # Binned again, but now (not older than the cutoff), and another image
    # backdated in its place: the same count, other ids.
    store.bin_images([second.id])
    _age(store, [third.id], 10)
    with pytest.raises(gallery_delete.SelectionChanged):
        gallery_delete.empty_bin(expected=2, pin=found["pin"], before=before)
    assert all((outputs / r.filename).exists() for r in (first, second, third))


def test_an_image_ageing_past_the_threshold_while_asked_is_not_taken(store):
    # The cutoff is the summary's, sent back unchanged: an image binned just
    # after it is never in the set, however long the dialog stays open.
    early, late = save_source(seed=1), save_source(seed=2)
    store.bin_images([early.id, late.id])
    now = datetime.now(timezone.utc)
    _age(store, [early.id], 0, stamp=(now - timedelta(days=7, minutes=5)).isoformat())
    _age(store, [late.id], 0, stamp=(now - timedelta(days=7) + timedelta(minutes=5)).isoformat())
    before = gallery_delete.bin_cutoff(7, now)
    found = gallery_delete.bin_summary(before)
    assert found["records"] == 1
    result = gallery_delete.empty_bin(expected=1, pin=found["pin"], before=before)
    assert result["ids"] == [early.id]
    assert store.get_image(late.id, include_binned=True) is not None


def test_an_undatable_stamp_is_kept_by_a_scoped_empty_and_taken_by_the_whole(store):
    odd, old = save_source(seed=1), save_source(seed=2)
    store.bin_images([odd.id, old.id])
    _age(store, [odd.id], 0, stamp="not a date")
    _age(store, [old.id], 30)
    before = gallery_delete.bin_cutoff(7)
    assert gallery_delete.empty_bin(expected=1, before=before)["ids"] == [old.id]
    assert gallery_delete.empty_bin(expected=1)["ids"] == [odd.id]


def test_a_malformed_cutoff_deletes_nothing(store):
    row = save_source()
    store.bin_images([row.id])
    _age(store, [row.id], 30)
    with pytest.raises(ValueError):
        gallery_delete.empty_bin(expected=1, before="a month ago")
    assert (store.settings.outputs_dir / row.filename).exists()


def test_the_scoped_empty_keeps_masks_a_kept_row_names(store):
    source = save_source(seed=1)
    shared, own = (store.store_mask(Image.new("L", (256, 256), level))[0] for level in (64, 192))
    with store._session() as conn:
        conn.execute("UPDATE masks SET created_at = ?", (
            (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
        ))
    kept = save_source("kept, binned today", 2, refine=source.id, strength=0.6, mask=shared)
    old_shared = save_source("old, shared mask", 3, refine=source.id, strength=0.6, mask=shared)
    old_own = save_source("old, own mask", 4, refine=source.id, strength=0.6, mask=own)
    store.bin_images([kept.id, old_shared.id, old_own.id])
    _age(store, [old_shared.id, old_own.id], 10)
    result = gallery_delete.empty_bin(expected=2, before=gallery_delete.bin_cutoff(7))
    assert result["ids"] == sorted([old_shared.id, old_own.id]) and result["masks"] == 1
    assert store.mask_path(shared).exists()
    with pytest.raises(ValueError, match="no stored mask"):
        store.mask_path(own)


def test_the_scoped_empty_fails_fast_when_another_process_has_reserved_the_gallery(store):
    row = save_source()
    store.bin_images([row.id])
    _age(store, [row.id], 30)
    child = holding_child(store.settings.db_path)
    try:
        with pytest.raises(gallery_lock.GalleryBusy):
            gallery_delete.empty_bin(expected=1, before=gallery_delete.bin_cutoff(7))
    finally:
        child.stdin.close()
        child.wait(timeout=15)
    assert (store.settings.outputs_dir / row.filename).exists()


def test_clear_bin_older_than_previews_and_empties_only_the_old(store, capsys, monkeypatch):
    monkeypatch.setattr(clear, "settings", store.settings)
    old, fresh = save_source(seed=1), save_source(seed=2)
    store.bin_images([old.id, fresh.id])
    _age(store, [old.id], 40)

    assert clear.main(["--bin", "--older-than", "30", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 of the 2 images in the recycle bin binned more than 30 day(s) ago" in out
    assert "Would delete 1 record(s)" in out
    assert clear.main(["--bin", "--older-than", "30", "--yes"]) == 0
    assert "Deleted 1 record(s), 1 file(s)." in capsys.readouterr().out
    assert not (store.settings.outputs_dir / old.filename).exists()
    assert (store.settings.outputs_dir / fresh.filename).exists()
    assert clear.main(["--bin", "--older-than", "30", "--yes"]) == 0
    assert "Nothing in the recycle bin was binned more than 30 day(s) ago." in capsys.readouterr().out
    with pytest.raises(SystemExit):
        clear.main(["--older-than", "30"])
    with pytest.raises(SystemExit):
        clear.main(["--bin", "--older-than", "0"])
    assert storage.bin_summary()["records"] == 1
