"""Rows rebuilt from the PNGs' own metadata: in place, from elsewhere, idempotently."""
from __future__ import annotations

import dataclasses
import io
import json
import sys

from PIL import Image, PngImagePlugin

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
    items, total, _ = store.list_images()
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
    # The hash is recomputed from the pixels and lands where it was.
    assert back.content_hash == a.content_hash and len(back.content_hash) == 16
    assert back.true_cfg == a.true_cfg == 0.5  # the scale "blurry" ran at
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
    assert counts == {
        "indexed": 1, "present": 0, "no_metadata": 0, "conflicts": 1,
        "unresolved_references": 0, "masks": 0, "bad_masks": 0, "unresolved_masks": 0,
        "outpaint_malformed": 0, "outpaints_recovered": 0, "outpaint_masks_rebuilt": 0,
        "invalid_upscalers": 0,
    }
    assert (store.settings.outputs_dir / made[0].filename).exists()
    assert (store.settings.outputs_dir / "thumbnails" / made[0].filename.replace(".png", ".webp")).exists()
    assert list(store.settings.outputs_dir.glob("*.part")) == []
    items, total, _ = store.list_images()
    assert total == 1 and items[0].prompt == "from wsl"
    # Its source is untouched.
    assert (elsewhere / made[0].filename).exists()


def test_database_merge_preserves_db_fields_and_remaps_references(
    store, monkeypatch, tmp_path
):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)

    # Occupy id 1 in this gallery so source ids cannot accidentally remain valid.
    _make(store, "already here", 90)

    elsewhere = tmp_path / "backup" / "outputs"
    elsewhere.mkdir(parents=True)
    source_db = elsewhere.parent / "local-diffusion.db"
    other = dataclasses.replace(mine, db_path=source_db, outputs_dir=elsewhere)
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    reference = _make(storage, "source reference", 1)
    storage.set_favorite(reference.id, True)
    conditioned = _make(storage, "conditioned", 2, reference=(reference.id,))

    monkeypatch.setattr(storage, "settings", mine)
    counts = reindex.reindex(elsewhere, database=source_db)
    assert counts == {
        "indexed": 2,
        "present": 0,
        "missing": 0,
        "invalid": 0,
        "conflicts": 0,
        "unresolved_references": 0,
        "masks": 0,
        "bad_masks": 0,
        "unresolved_masks": 0,
        "outpaint_malformed": 0,
        "outpaints_recovered": 0,
        "outpaint_masks_rebuilt": 0,
    }

    items, total, _ = store.list_images(limit=10)
    assert total == 3
    by_prompt = {item.prompt: item for item in items}
    imported_reference = by_prompt["source reference"]
    imported_conditioned = by_prompt["conditioned"]
    assert imported_reference.favorite is True
    assert imported_reference.duration_ms == 1234
    assert imported_conditioned.reference == [imported_reference.id]
    assert imported_reference.id != reference.id

    # The copied PNG remains self-describing if this destination DB is rebuilt.
    with Image.open(mine.outputs_dir / conditioned.filename) as image:
        meta = json.loads(image.info["local-diffusion"])
    assert meta["reference"] == [imported_reference.id]

    # A second merge is harmless and does not duplicate rows.
    again = reindex.reindex(elsewhere, database=source_db)
    assert again["indexed"] == 0 and again["present"] == 2


def test_database_merge_can_import_a_png_without_embedded_metadata(
    store, monkeypatch, tmp_path
):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    elsewhere = tmp_path / "backup" / "outputs"
    elsewhere.mkdir(parents=True)
    source_db = elsewhere.parent / "local-diffusion.db"
    other = dataclasses.replace(mine, db_path=source_db, outputs_dir=elsewhere)
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    made = _make(storage, "database is the metadata", 5)
    Image.new("RGB", (40, 30), "red").save(elsewhere / made.filename)

    monkeypatch.setattr(storage, "settings", mine)
    counts = reindex.reindex(elsewhere, database=source_db)
    assert counts["indexed"] == 1
    imported = store.list_images()[0][0]
    assert (imported.prompt, imported.width, imported.height) == (
        "database is the metadata", 40, 30,
    )
    with Image.open(mine.outputs_dir / made.filename) as image:
        assert json.loads(image.info["local-diffusion"])["prompt"] == "database is the metadata"


def test_reindex_reads_the_pre_rename_metadata_key(tmp_path):
    path = tmp_path / "legacy.png"
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "krea",
        json.dumps(
            {
                "prompt": "legacy gallery",
                "steps": 8,
                "guidance_scale": 0,
                "seed": 7,
                "model": "krea/Krea-2-Turbo",
            }
        ),
    )
    Image.new("RGB", (32, 24)).save(path, pnginfo=info)

    meta, size, _ = reindex._read(path)

    assert size == (32, 24)
    assert meta is not None and meta["prompt"] == "legacy gallery"


def test_the_report_writes_utf8_where_its_output_is_redirected(store, monkeypatch, tmp_path):
    # `> reindex.txt` on Windows: stdout is the locale's code page, strict, and
    # a CJK name for the folder being copied in was a UnicodeEncodeError in the
    # report's header. (The per-file lines go to `out`, bound to the same
    # stream at import, which the reconfigure changes in place.)
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = tmp_path / "来源"
    source.mkdir()
    redirected = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(redirected, encoding="cp1252"))
    assert reindex.main(["--from", str(source), "--dry-run"]) == 0
    sys.stdout.flush()
    assert f"source   : {source}" in redirected.getvalue().decode("utf-8")


def test_the_recycle_bin_survives_a_reindex_and_a_database_merge_but_not_a_png_only_one(
    store, monkeypatch, tmp_path
):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    # In place: the binned row is in the index, so it is skipped, still binned,
    # and its PNG is no new file to add.
    binned = _make(store, "in the bin", 1)
    store.bin_images([binned.id])
    assert reindex.reindex(mine.outputs_dir)["indexed"] == 0
    assert store.get_image(binned.id, include_binned=True).binned_at is not None

    elsewhere = tmp_path / "backup" / "outputs"
    elsewhere.mkdir(parents=True)
    source_db = elsewhere.parent / "local-diffusion.db"
    other = dataclasses.replace(mine, db_path=source_db, outputs_dir=elsewhere)
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    gone = _make(storage, "binned elsewhere", 2)
    live = _make(storage, "live elsewhere", 3)
    storage.bin_images([gone.id])
    monkeypatch.setattr(storage, "settings", mine)

    # With the database, the bin comes across as the star does.
    assert reindex.reindex(elsewhere, database=source_db)["indexed"] == 2
    assert [r.prompt for r in store.list_images(binned=True)[0]] == ["binned elsewhere", "in the bin"]
    assert store.list_images(search="live")[1] == 1

    # From the PNGs alone it cannot: the bin is SQLite's, and the image comes back live.
    store.settings.db_path.unlink()
    assert reindex.main([]) == 0
    assert store.list_images(binned=True)[1] == 0
    assert store.list_images()[1] == 3
    assert live.filename in {r.filename for r in store.list_images()[0]}
