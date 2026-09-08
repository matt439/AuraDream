"""Rows rebuilt from the PNGs' own metadata: in place, from elsewhere, idempotently."""
from __future__ import annotations

import dataclasses

from PIL import Image

from backend import reindex, storage
from backend.schemas import GenerateRequest


def _make(store, prompt, seed, **kw):
    return store.save_image(
        Image.new("RGB", (64, 48)),
        GenerateRequest(prompt=prompt, width=256, height=256, steps=6, guidance_scale=0.5, mu=1.15, **kw),
        seed=seed, duration_ms=1234, model_id="krea/Krea-2-Turbo",
        job_id="job1", run_id="run1", quantization="fp8",
    )


def test_reindex_in_place_rebuilds_the_rows(store, monkeypatch, capsys):
    monkeypatch.setattr(reindex, "settings", store.settings)
    a = _make(store, "a fox", 7, negative_prompt="blurry")
    b = _make(store, "a lighthouse", 8)
    (store.settings.outputs_dir / "foreign.png").write_bytes(
        Image.new("RGB", (8, 8)).tobytes()  # not even a PNG
    )
    Image.new("RGB", (8, 8)).save(store.settings.outputs_dir / "plain.png")
    # Lose the index: the file is what survives.
    store.settings.db_path.unlink()

    assert reindex.main(["--dry-run"]) == 0
    assert not store.settings.db_path.exists()
    out = capsys.readouterr().out
    assert "Would index 2 file(s)" in out and "1 without metadata" in out or "without metadata" in out

    assert reindex.main([]) == 0
    items, total = store.list_images()
    assert total == 2
    by_seed = {r.seed: r for r in items}
    back = by_seed[7]
    assert (back.prompt, back.negative_prompt, back.steps, back.guidance_scale, back.mu) == (
        "a fox", "blurry", 6, 0.5, 1.15,
    )
    assert (back.model_id, back.job_id, back.run_id, back.quantization) == (
        "krea/Krea-2-Turbo", "job1", "run1", "fp8",
    )
    assert (back.width, back.height) == (64, 48)  # the file's, not the request's
    assert back.filename == a.filename and back.duration_ms == 0
    # The timestamp is the filename's UTC prefix, to the second.
    assert back.created_at[:16] == a.created_at[:16]
    assert by_seed[8].filename == b.filename
    # And the search index came back with the rows.
    assert store.list_images(search="lighthouse")[1] == 1
    # Idempotent: a second pass adds nothing.
    assert reindex.main([]) == 0
    assert "Indexed 0 file(s); 2 already in the index" in capsys.readouterr().out
    assert store.list_images()[1] == 2


def test_reindex_copies_another_gallery_in(store, monkeypatch, tmp_path):
    # `store` is the storage module itself, so hold on to this gallery's
    # settings before pointing the module at the other one.
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    elsewhere = tmp_path / "wsl-outputs"
    elsewhere.mkdir()
    # Two of this app's PNGs made elsewhere: write them through a second gallery.
    other = dataclasses.replace(mine, db_path=tmp_path / "other.db", outputs_dir=elsewhere)
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    made = [_make(storage, "from wsl", 1), _make(storage, "from wsl too", 2)]
    monkeypatch.setattr(storage, "settings", mine)
    # A name clash with a *different* file already on disk here is not overwritten.
    (store.settings.outputs_dir / made[1].filename).write_bytes(b"something else")

    counts = reindex.reindex(elsewhere)
    assert counts == {"indexed": 1, "present": 0, "no_metadata": 0, "conflicts": 1}
    assert (store.settings.outputs_dir / made[0].filename).exists()
    assert (store.settings.outputs_dir / made[0].filename.replace(".png", ".webp")).exists()
    assert list(store.settings.outputs_dir.glob("*.part")) == []
    items, total = store.list_images()
    assert total == 1 and items[0].prompt == "from wsl"
    # Its source is untouched.
    assert (elsewhere / made[0].filename).exists()
