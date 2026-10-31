"""The peak GPU memory a call held, saved beside its images.

A result, like the duration: on the row and in the chunk, carried by
reindex and bundles, shared whole by a batch's images, and no part of what
reproduces an image - so no key, run id or Resume count may move with it.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from backend import bundles, grid, storage
from backend.schemas import GenerateRequest
from tests.api_helpers import run_one
from tests.bundle_helpers import use_destination, export_all, png_metadata

GIB = 1024**3
PEAK = (int(3.89 * GIB), int(4.70 * GIB))


def _save(peak=None, seed=1, **kwargs):
    return storage.save_image(
        Image.new("RGB", (256, 256), (seed % 256, 24, 80)),
        GenerateRequest(prompt="a lighthouse", width=256, height=256, steps=8, **kwargs),
        seed=seed, duration_ms=1200, model_id="test/model", job_id="job", run_id="run",
        quantization="fp8", peak=peak,
    )


def _chunk(record):
    return png_metadata(storage.settings.outputs_dir / record.filename)


def test_a_peak_rides_on_the_row_and_in_the_chunk(store):
    record = _save(PEAK)
    assert (record.vram_peak_allocated, record.vram_peak_reserved) == PEAK
    assert storage.get_image(record.id).vram_peak_allocated == PEAK[0]
    chunk = _chunk(record)
    assert (chunk["vram_peak_allocated"], chunk["vram_peak_reserved"]) == PEAK
    # Unmeasured is absence, in the chunk as on the row.
    bare = _save(None, seed=2)
    assert bare.vram_peak_allocated is None and bare.vram_peak_reserved is None
    assert "vram_peak_allocated" not in _chunk(bare) and "vram_peak_reserved" not in _chunk(bare)


def test_an_older_database_gains_the_columns_and_reads_none(store):
    old = _save(None)
    with sqlite3.connect(storage.settings.db_path) as conn:
        conn.execute("UPDATE images SET vram_peak_allocated = NULL, vram_peak_reserved = NULL")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    assert {"vram_peak_allocated", "vram_peak_reserved"} <= columns
    assert storage.get_image(old.id).vram_peak_allocated is None


def test_a_png_only_reindex_brings_the_peak_back_and_a_bad_one_is_none(store):
    meta = json.loads(json.dumps(_chunk(_save(PEAK))))
    meta.update(seed=5)
    back = storage.index_file("again.png", meta, 256, 256, storage.datetime.now(storage.timezone.utc))
    assert (back.vram_peak_allocated, back.vram_peak_reserved) == PEAK
    # A chunk written by hand says what it likes; the row is never refused.
    meta.update(seed=6, vram_peak_allocated="lots", vram_peak_reserved=-1)
    odd = storage.index_file("odd.png", meta, 256, 256, storage.datetime.now(storage.timezone.utc))
    assert odd.vram_peak_allocated is None and odd.vram_peak_reserved is None


def test_the_peak_is_no_part_of_a_cells_identity(store):
    measured, unmeasured = _save(PEAK, seed=3), _save(None, seed=3)
    assert grid.cell_key(grid.cell_from_record(measured)) == grid.cell_key(grid.cell_from_record(unmeasured))
    cell = grid.cell_from_record(measured)
    assert grid.run_id([cell], "test/model", "fp8") == grid.run_id([grid.cell_from_record(unmeasured)], "test/model", "fp8")
    # Resume reads the two rows as the same cell.
    done = storage.completed_cells(["a lighthouse"], 256, 256, "test/model", "fp8")
    assert len(done) == 1


def test_a_batch_shares_its_calls_peak_and_a_failed_call_leaves_none(client, fake, store):
    fake.peak = PEAK
    _, events = run_one(client, num_images=3, seed=10)
    rows = store.list_images()[0]
    assert len(rows) == 3 and {(r.vram_peak_allocated, r.vram_peak_reserved) for r in rows} == {PEAK}
    # The fake clears its answer as each call begins: a call that raises
    # never hands its predecessor's peak to the next save.
    fake.fail_on = {len(fake.requests) + 1}
    run_one(client, seed=20)
    assert fake.last_peak() is None
    fake.peak = None
    run_one(client, seed=30)
    newest = store.list_images(search="seed:30")[0]
    assert len(newest) == 1 and newest[0].vram_peak_allocated is None


def test_a_bundle_carries_the_peak_and_a_bad_one_is_dropped(store, monkeypatch, tmp_path):
    measured = _save(PEAK, seed=1)
    odd = _save(None, seed=2)
    with sqlite3.connect(storage.settings.db_path) as conn:
        conn.execute("UPDATE images SET vram_peak_allocated = 'lots' WHERE id = ?", (odd.id,))
    archive = export_all()
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(Path(archive["path"]))["imported"] == 2
    rows = {r.filename: r for r in storage.list_images()[0]}
    assert (rows[measured.filename].vram_peak_allocated, rows[measured.filename].vram_peak_reserved) == PEAK
    assert rows[odd.filename].vram_peak_allocated is None
    assert png_metadata(dest.outputs_dir / measured.filename)["vram_peak_reserved"] == PEAK[1]
    with sqlite3.connect(dest.db_path) as conn:
        kinds = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(images)")}
    assert kinds["vram_peak_allocated"] == kinds["vram_peak_reserved"] == "INTEGER"


def test_load_settings_from_png_ignores_the_peak(client, store):
    measured = _save(PEAK, seed=4)
    unmeasured = _save(None, seed=4)

    def loaded(record):
        with Image.open(storage.settings.outputs_dir / record.filename) as image:
            chunks = {k: v for k, v in image.info.items() if isinstance(v, str)}
        response = client.post("/api/png/settings", json={"chunks": chunks})
        assert response.status_code == 200, response.text
        return {k: v for k, v in response.json().items() if k not in {"source", "notes"}}

    assert loaded(measured) == loaded(unmeasured)


# ------------------------------------------------------------ the real backend


class _Cuda:
    def __init__(self, available=True, fail=False):
        self.available, self.fail, self.calls = available, fail, []

    def is_available(self):
        return self.available

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", device))

    def max_memory_allocated(self, device):
        self.calls.append(("allocated", device))
        if self.fail:
            raise RuntimeError("no device")
        return PEAK[0]

    def max_memory_reserved(self, device):
        self.calls.append(("reserved", device))
        return PEAK[1]


@pytest.fixture
def real(monkeypatch):
    import torch

    from backend.backends.diffusers_backend import DiffusersBackend

    backend = DiffusersBackend()
    backend._pipe = object()
    cuda = _Cuda()
    monkeypatch.setattr(torch, "cuda", cuda)
    order = []

    def body(*args):
        order.append(list(cuda.calls))
        if backend.raise_next:
            raise RuntimeError("out of memory")
        return [Image.new("RGB", (8, 8))]

    backend.raise_next = False
    monkeypatch.setattr(backend, "_generate_images", body)
    return backend, cuda, order


def test_the_backend_resets_before_the_call_and_reads_after(real, monkeypatch):
    backend, cuda, order = real
    monkeypatch.setattr(backend, "_peak_device", lambda: 0)
    request = GenerateRequest(prompt="a fox", width=256, height=256)
    assert len(backend.generate(request, seed=1)) == 1
    assert order == [[("reset", 0)]]
    assert backend.last_peak() == PEAK
    # A call that raises leaves no peak behind it.
    backend.raise_next = True
    with pytest.raises(RuntimeError):
        backend.generate(request, seed=2)
    assert backend.last_peak() is None


def test_a_failed_read_still_returns_the_images(real, monkeypatch):
    backend, cuda, _ = real
    monkeypatch.setattr(backend, "_peak_device", lambda: 0)
    cuda.fail = True
    assert len(backend.generate(GenerateRequest(prompt="a fox", width=256, height=256), seed=1)) == 1
    assert backend.last_peak() is None


def test_no_cuda_device_measures_nothing(real):
    backend, cuda, _ = real
    cuda.available = False
    assert backend._peak_device() is None
    assert len(backend.generate(GenerateRequest(prompt="a fox", width=256, height=256), seed=1)) == 1
    assert backend.last_peak() is None and cuda.calls == []
