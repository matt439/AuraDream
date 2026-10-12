"""The inpaint masks swept where images are removed for good - Empty recycle
bin, the per-image delete, `clear` and `clear --where` - and kept while any
row (live or binned) or any queued or running cell names them, or while they
are younger than the settle window."""
from __future__ import annotations

import os
import time

from PIL import Image

from backend import clear, gallery_delete, main, storage
from backend.schemas import GenerateRequest
from tests.api_helpers import stream_job, wait_for

OLD = "2000-01-01T00:00:00+00:00"


def _mask(box) -> Image.Image:
    image = Image.new("L", (256, 256), 0)
    image.paste(255, box)
    return image


def _source(seed=1):
    return storage.save_image(
        Image.new("RGB", (256, 256), (seed, 24, 80)),
        GenerateRequest(prompt="source", width=256, height=256, steps=28),
        seed=seed, duration_ms=10, model_id="test/model",
    )


def _inpainted(source, digest, seed=2):
    return storage.save_image(
        Image.new("RGB", (256, 256), (200, 10, 10)),
        GenerateRequest(prompt="a red scarf", width=256, height=256, steps=28,
                        refine=source.id, strength=0.6, mask=digest),
        seed=seed, duration_ms=10, model_id="test/model",
    )


def _stored(box, *, old=True) -> str:
    digest, _ = storage.store_mask(_mask(box))
    if old:
        with storage._session() as conn:
            conn.execute("UPDATE masks SET created_at = ? WHERE hash = ?", (OLD, digest))
    return digest


def _held() -> set[str]:
    on_disk = {path.stem for path in storage.masks_dir().glob("*.png")}
    assert on_disk == storage._tracked_masks()
    return on_disk


def test_the_sweep_spares_what_a_row_a_job_or_the_window_holds(store):
    source = _source()
    live = _stored((0, 0, 10, 10))
    binned = _stored((0, 0, 20, 20))
    unused = _stored((0, 0, 30, 30))
    young = _stored((0, 0, 40, 40), old=False)
    queued = _stored((0, 0, 50, 50))
    kept = _inpainted(source, live)
    in_bin = _inpainted(source, binned, seed=3)
    gallery_delete.bin_images([in_bin.id])

    assert gallery_delete.sweep_masks(lambda: {queued}) == 1
    assert _held() == {live, binned, young, queued}
    # Emptying the bin takes the binned inpaint, and then its mask; the
    # queued one goes too once nothing is in flight to read it.
    result = gallery_delete.empty_bin()
    assert result["masks"] == 2 and _held() == {live, young}
    # The per-image delete of the last row naming a mask takes it.
    assert gallery_delete.delete_image(kept.id)
    assert _held() == {young}


def test_a_re_upload_restarts_the_settle_window(store):
    digest = _stored((0, 0, 10, 10))
    storage.store_mask(_mask((0, 0, 10, 10)))
    assert gallery_delete.sweep_masks() == 0 and _held() == {digest}


def test_the_whole_clear_takes_every_mask_and_the_directory(store):
    source = _source()
    _inpainted(source, _stored((0, 0, 10, 10)))
    _stored((0, 0, 20, 20), old=False)
    result = gallery_delete.clear_history()
    assert result["masks"] == 2
    assert not storage.masks_dir().exists() and storage._tracked_masks() == set()


def test_keep_favorites_keeps_a_starred_inpaints_mask(store):
    source = _source()
    starred_mask = _stored((0, 0, 10, 10))
    other_mask = _stored((0, 0, 20, 20))
    young = _stored((0, 0, 30, 30), old=False)
    starred = _inpainted(source, starred_mask)
    _inpainted(source, other_mask, seed=3)
    storage.set_favorite(starred.id, True)
    result = gallery_delete.clear_history(keep_favorites=True)
    assert result["masks"] == 1
    assert _held() == {starred_mask, young}


def test_the_orphan_sweep_takes_a_mask_file_with_no_row_once_settled(store):
    _source()
    directory = storage.masks_dir()
    directory.mkdir(exist_ok=True)
    old, fresh = directory / f"{'a' * 64}.png", directory / f"{'b' * 64}.png"
    for path in (old, fresh):
        _mask((0, 0, 8, 8)).save(path)
    then = time.time() - 2 * storage.PARTIAL_SETTLE_SECONDS
    os.utime(old, (then, then))
    gallery_delete.clear_history(keep_favorites=True)
    assert not old.exists() and fresh.exists()


def test_clear_where_sweeps_the_masks_its_rows_named(store, monkeypatch, capsys):
    monkeypatch.setattr(clear, "settings", store.settings)
    source = _source()
    digest = _stored((0, 0, 10, 10))
    made = _inpainted(source, digest)
    assert clear.main(["--where", f"id:{made.id}", "--yes"]) == 0
    assert "1 unused inpaint mask(s)" in capsys.readouterr().out
    assert _held() == set()


def _queued_inpaint(client, fake, digest, released):
    """An inpaint job over `digest` whose first cell is held before its
    image saves, so nothing in the gallery names the mask yet."""
    run = client.post("/api/generate", json={"prompt": "x", "width": 256, "height": 256}).json()
    stream_job(client, run["job_id"])
    source = client.get("/api/gallery").json()["items"][0]
    fake.after_generate = lambda: released.wait(10)
    job_id = client.post("/api/generate", json={
        "prompt": "a red scarf", "width": 256, "height": 256, "steps": 10, "refine": source["id"],
        "strength": 0.6, "mask": digest, "runs": 4,
    }).json()["job_id"]
    wait_for(client, job_id, lambda s: s["status"] == "running")
    wait_for(client, job_id, lambda s: len(fake.requests) >= 2)
    with storage._session() as conn:
        named = conn.execute("SELECT COUNT(*) FROM images WHERE mask = ?", (digest,)).fetchone()[0]
    assert named == 0, "the held cell must not have saved"
    return job_id


def _end(client, fake, job_id, released):
    client.post(f"/api/jobs/{job_id}/cancel")
    released.set()
    fake.after_generate = None
    stream_job(client, job_id)


def test_a_delete_spares_the_mask_a_queued_cell_will_read(client, fake, store):
    import threading

    other = _source(7)
    digest = _stored((0, 0, 10, 10))
    released = threading.Event()
    job_id = _queued_inpaint(client, fake, digest, released)
    assert main.jobs.masks_in_flight() == {digest}
    # Nothing is saved yet that names it, and it is past the window: only
    # the job holds it - through the route's own sweep.
    assert client.delete(f"/api/images/{other.id}").status_code == 200
    assert digest in _held()
    _end(client, fake, job_id, released)
    assert main.jobs.masks_in_flight() == set()


def test_a_cli_sweep_spares_the_mask_a_server_job_holds(client, fake, store, monkeypatch, capsys):
    import threading

    # The CLI cannot ask the server's queue: it reads the holds the server
    # wrote beside the rows when it queued the job.
    monkeypatch.setattr(clear, "settings", store.settings)
    other = _source(7)
    gallery_delete.bin_images([other.id])
    digest = _stored((0, 0, 10, 10))
    released = threading.Event()
    job_id = _queued_inpaint(client, fake, digest, released)
    with storage._session() as conn:
        assert [tuple(r) for r in conn.execute("SELECT hash, job_id FROM mask_holds")] == [(digest, job_id)]
    assert clear.main(["--bin", "--yes"]) == 0
    assert gallery_delete.sweep_masks(None) == 0
    assert digest in _held()
    _end(client, fake, job_id, released)
    # Released once the job ends (off the loop); then only the row the held
    # cell saved keeps it, and with that row gone the sweep takes it.
    deadline = time.time() + 5
    while time.time() < deadline:
        with storage._session() as conn:
            if conn.execute("SELECT COUNT(*) FROM mask_holds").fetchone()[0] == 0:
                break
        time.sleep(0.02)
    with storage._session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM mask_holds").fetchone()[0] == 0
        made = [row[0] for row in conn.execute("SELECT id FROM images WHERE mask = ?", (digest,))]
    for image_id in made:
        assert gallery_delete.delete_image(image_id)
    assert digest not in _held()


def test_a_server_start_drops_the_holds_a_crashed_one_left(store):
    digest = _stored((0, 0, 10, 10))
    assert storage.hold_masks("gone-job", [digest]) == []
    assert gallery_delete.sweep_masks(None) == 0
    storage.release_masks()
    assert gallery_delete.sweep_masks(None) == 1


def test_a_mask_swept_between_the_check_and_the_hold_is_not_queued(client, fake, store, monkeypatch):
    run = client.post("/api/generate", json={"prompt": "x", "width": 256, "height": 256}).json()
    stream_job(client, run["job_id"])
    source = client.get("/api/gallery").json()["items"][0]
    digest = _stored((0, 0, 10, 10))
    held = storage.hold_masks

    def swept_first(job_id, masks):
        # The sweep's transaction won: the mask is gone when the hold looks.
        gallery_delete.sweep_masks(None)
        return held(job_id, masks)

    monkeypatch.setattr(storage, "hold_masks", swept_first)
    answer = client.post("/api/generate", json={
        "prompt": "a red scarf", "width": 256, "height": 256, "steps": 10, "refine": source["id"],
        "strength": 0.6, "mask": digest,
    })
    assert answer.status_code == 422 and "has just gone" in answer.json()["detail"]
    assert main.jobs.active() == []
    with storage._session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM mask_holds").fetchone()[0] == 0
