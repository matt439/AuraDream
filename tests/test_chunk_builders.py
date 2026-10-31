"""The PNG `auradream` chunk is built in three places, and they keep one key set.

`storage.save_image` builds it from the request that ran, `reindex._database_meta`
from a source database's row, and `bundle_images._metadata` from a bundle's row.
Each normalises its own values (a merged builder would have to take every
field pre-normalised), so only the key set is held equal here: a key added to
one builder and not the others fails this test rather than a later reindex.
"""
from __future__ import annotations

import json

import pytest
from PIL import Image

from backend import bundle_images, reindex, storage
from backend.metadata import PNG_METADATA_KEY
from backend.schemas import GenerateRequest


def _save(prompt="source", seed=1, peak=None, **fields):
    return storage.save_image(
        Image.new("RGB", (256, 256), (seed % 255, 40, 90)),
        GenerateRequest(prompt=prompt, width=256, height=256, steps=28, **fields),
        seed=seed, duration_ms=10, model_id="test/model", quantization="fp8", peak=peak,
    )


def _row(image_id):
    with storage._session() as conn:
        return dict(conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone())


def _chunk(store, record):
    with Image.open(store.settings.outputs_dir / record.filename) as image:
        return json.loads(image.info[PNG_METADATA_KEY])


@pytest.mark.parametrize("kind", ["text-to-image", "refine pass"])
def test_the_three_chunk_builders_write_the_same_keys(store, kind):
    source = _save(peak=(1024, 2048))
    made = source if kind == "text-to-image" else _save(
        "pass", 2, peak=(4096, 8192), refine=source.id, strength=0.3, crop_anchor="top",
    )
    saved = _chunk(store, made)
    row = _row(made.id)
    references = json.loads(row["reference"]) if row["reference"] else []
    rebuilt = reindex._database_meta(
        row, row["width"], row["height"], row["content_hash"], references, row["refine"],
    )
    bundled = bundle_images._metadata(row, {source.id: _row(source.id)})
    assert "vram_peak_allocated" in saved and "vram_peak_reserved" in saved
    assert ("crop_anchor" in saved) == (kind == "refine pass")
    assert set(rebuilt) == set(saved)
    assert set(bundled) == set(saved)
