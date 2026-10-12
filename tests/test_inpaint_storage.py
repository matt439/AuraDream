"""An inpaint's reproduction set in storage: the `masks` table and files,
the `mask` column, the chunk keys and the embedded mask, the resume key, and
the cost history split on the mask."""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from backend import gallery_reports, grid, storage
from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEY
from backend.schemas import GenerateRequest

from test_storage import OLD_SCHEMA, keep

ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


def _mask(size=(256, 256), box=(0, 0, 128, 64)) -> Image.Image:
    image = Image.new("L", size, 0)
    image.paste(255, box)
    return image


def _inpainted(store, source, mask: str, *, strength=0.6, seed=2, duration_ms=1000, model_id=ANIMAGINE):
    request = GenerateRequest(
        prompt="a fox in the snow", width=source.width, height=source.height, steps=10,
        guidance_scale=5.0, refine=source.id, strength=strength, mask=mask,
    )
    return store.save_image(
        Image.new("RGB", (source.width, source.height), (200, 10, 10)), request,
        seed=seed, duration_ms=duration_ms, model_id=model_id,
    )


# ------------------------------------------------------------------ the masks


def test_a_mask_is_stored_once_by_its_size_and_levels(store):
    digest, created = store.store_mask(_mask())
    assert created is True and len(digest) == 64
    assert store.mask_path(digest) == store.masks_dir() / f"{digest}.png"
    assert store.mask_size(digest) == (256, 256)
    assert store.store_mask(_mask().convert("RGB")) == (digest, False)
    # The levels alone do not say the shape: an all-white mask of a portrait
    # and of a landscape are the same bytes and two different masks.
    tall, _ = store.store_mask(Image.new("L", (256, 384), 255))
    wide, _ = store.store_mask(Image.new("L", (384, 256), 255))
    assert tall != wide
    assert (store.mask_size(tall), store.mask_size(wide)) == ((256, 384), (384, 256))
    assert list(store.masks_dir().glob("*.part")) == []


def test_mask_path_names_an_unknown_hash_and_a_missing_file(store):
    with pytest.raises(ValueError, match="no stored mask 0000000000000000"):
        store.mask_path("0" * 64)
    digest, _ = store.store_mask(_mask())
    store.mask_path(digest).unlink()
    with pytest.raises(ValueError, match="missing from the masks directory"):
        store.mask_path(digest)
    # The next upload of the same picture gives the row its file back.
    assert store.store_mask(_mask()) == (digest, True)
    assert store.mask_path(digest).is_file()


def test_a_lost_mask_row_is_given_back_by_its_file(store):
    digest, _ = store.store_mask(_mask())
    with store._session() as conn:
        conn.execute("DELETE FROM masks")
    assert store.mask_size(digest) is None
    assert store.store_mask(_mask()) == (digest, True)
    assert store.mask_size(digest) == (256, 256)


def test_a_replaced_database_gains_the_column_and_the_masks_table(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    db = tmp_path / "local-diffusion.db"
    monkeypatch.setattr(storage, "settings", dataclasses.replace(storage.settings, db_path=db, outputs_dir=outputs))
    storage.init_db()
    storage.store_mask(_mask())
    # An older file restored over the same path under the running server:
    # no `mask` column, no `masks` table, until its next connection.
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(OLD_SCHEMA)
    conn.close()
    source = keep(storage)
    assert source.mask is None
    with storage._session() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "mask" in columns and "masks" in tables
    # The mask file outlived its row; storing it again gives it one.
    digest, created = storage.store_mask(_mask())
    assert created is True and storage.mask_size(digest) == (256, 256)
    record = _inpainted(storage, source, digest)
    assert storage.get_image(record.id).mask == digest


def test_a_mask_partial_is_finished_when_its_row_committed(store):
    digest, _ = store.store_mask(_mask())
    path = store.mask_path(digest)
    partial = path.with_name(path.name + ".part")
    path.replace(partial)
    debris = store.masks_dir() / f"{'1' * 64}.png.part"
    debris.write_bytes(b"cut short")
    old = time.time() - 2 * store.PARTIAL_SETTLE_SECONDS
    for stale in (partial, debris):
        os.utime(stale, (old, old))
    assert store.settle_partials() == {"recovered": 1, "discarded": 1}
    assert path.is_file() and not partial.exists() and not debris.exists()


# ------------------------------------------------------------- the row, the PNG


def test_an_inpaint_is_on_its_row_and_in_its_chunks(store):
    source = keep(store, width=256, height=256)
    digest, _ = store.store_mask(_mask())
    record = _inpainted(store, source, digest)
    assert (record.refine, record.strength, record.mask) == (source.id, 0.6, digest)
    with Image.open(store.settings.outputs_dir / record.filename) as handle:
        info = dict(handle.info)
    meta = json.loads(info[PNG_METADATA_KEY])
    assert (meta["refine"], meta["mask"], meta["mask_blend"]) == (source.id, digest, "overlay")
    assert meta["mask_source"] == {"filename": f"{digest}.png", "hash": digest, "width": 256, "height": 256}
    assert info["parameters"].endswith(f"Refine: {source.id}, Inpaint mask: {digest[:16]}")
    assert base64.b64decode(info[PNG_MASK_KEY]) == store.mask_path(digest).read_bytes()
    # A refine pass with no mask says so by absence, the three together.
    passed = store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="a fox", width=256, height=256, refine=source.id, strength=0.5),
        seed=3, duration_ms=10, model_id=ANIMAGINE,
    )
    with Image.open(store.settings.outputs_dir / passed.filename) as handle:
        info = dict(handle.info)
    meta = json.loads(info[PNG_METADATA_KEY])
    assert (meta["mask"], meta["mask_source"], meta["mask_blend"]) == (None, None, None)
    assert PNG_MASK_KEY not in info and passed.mask is None


def test_an_image_saves_though_its_mask_cannot_be_embedded(store, monkeypatch):
    source = keep(store)
    digest, _ = store.store_mask(_mask())
    store.mask_path(digest).unlink()
    record = _inpainted(store, source, digest)
    # The hash still says what ran; only the backstop copy is missing.
    assert record.mask == digest
    with Image.open(store.settings.outputs_dir / record.filename) as handle:
        assert PNG_MASK_KEY not in handle.info


def test_a_reindexed_row_reads_its_mask_from_the_chunk(store):
    meta = {
        "prompt": "a fox", "steps": 10, "guidance_scale": 5.0, "seed": 1, "model": ANIMAGINE,
        "refine": 3, "strength": 0.6, "mask": "ab" * 32,
    }
    created = datetime.now(timezone.utc)
    record = store.index_file("a.png", meta, 256, 256, created)
    assert (record.refine, record.mask) == (3, "ab" * 32)
    # A hand-written block's mask is read only where it is a hash, and only
    # beside its source.
    assert store.index_file("b.png", {**meta, "mask": "../../keys"}, 256, 256, created).mask is None
    assert store.index_file("c.png", {**meta, "refine": None}, 256, 256, created).mask is None


# --------------------------------------------------------------- the resume key


def test_completed_cells_keys_on_the_mask(store):
    source = keep(store)
    one, _ = store.store_mask(_mask())
    two, _ = store.store_mask(_mask(box=(0, 0, 64, 64)))
    _inpainted(store, source, one, seed=2)
    _inpainted(store, source, two, seed=2)
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="a fox in the snow", width=256, height=256, steps=10, guidance_scale=5.0,
                        refine=source.id, strength=0.6),
        seed=2, duration_ms=10, model_id=ANIMAGINE,
    )
    done = store.completed_cells(["a fox in the snow"], 256, 256, ANIMAGINE)
    assert {key[9] for key in done} == {one, two, None}
    # And the request side builds the very same key.
    cell = grid.Cell(
        request=GenerateRequest(prompt="a fox in the snow", width=256, height=256, steps=10,
                                guidance_scale=5.0, refine=source.id, strength=0.6, mask=one),
        seed=2,
    )
    assert grid.cell_key(cell) in done


def test_an_inpaint_is_priced_from_inpaints_and_a_pass_from_passes(store, monkeypatch):
    source = keep(store, width=256, height=256)
    digest, _ = store.store_mask(_mask())
    _inpainted(store, source, digest, strength=0.6, seed=2, duration_ms=9000)
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="a fox", width=256, height=256, steps=10, refine=source.id, strength=0.6),
        seed=3, duration_ms=4000, model_id=ANIMAGINE,
    )
    # SDXL's classes run int(steps * strength) of the schedule, the inpaint's
    # as the image-to-image one's.
    assert gallery_reports.refine_cost_ms(256, 256, ANIMAGINE) == {6: 4000}
    assert gallery_reports.refine_cost_ms(256, 256, ANIMAGINE, inpaint=True) == {6: 9000}
