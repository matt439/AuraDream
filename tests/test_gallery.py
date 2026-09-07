"""Search, paging and the file/row pair that has to stay consistent."""
from __future__ import annotations

from PIL import Image

from backend.schemas import GenerateRequest


def _add(store, prompt: str, favorite: bool = False, seed: int = 1):
    record = store.save_image(
        Image.new("RGB", (16, 16)),
        GenerateRequest(prompt=prompt, width=256, height=256),
        seed=seed,
        duration_ms=100,
        model_id="test/model",
    )
    if favorite:
        store.set_favorite(record.id, True)
    return record


def test_save_image_leaves_a_png_and_no_partial(store):
    record = _add(store, "a lighthouse in fog")
    path = store.settings.outputs_dir / record.filename
    assert path.exists()
    assert list(store.settings.outputs_dir.glob("*.part")) == []
    assert store.summarise()["orphan_files"] == 0


def test_the_png_carries_the_whole_reproduction_set(store):
    # Key-set *equality*, not presence: the PNG is the only copy of these
    # outside SQLite - the database is gitignored and clear.py deletes it
    # wholesale - so adding a field to the reproduction set has to be a
    # decision about the PNG too.
    import json

    record = store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", negative_prompt="y", width=256, height=256, mu=1.15, steps=8),
        seed=5,
        duration_ms=100,
        model_id="test/model",
        job_id="job1",
        run_id="run1",
        quantization="fp8",
    )
    with Image.open(store.settings.outputs_dir / record.filename) as png:
        meta = json.loads(png.info["krea"])
    assert set(meta) == {
        "prompt", "negative_prompt", "steps", "guidance_scale", "mu", "seed",
        "model", "quantization", "job", "run", "size",
    }
    assert (meta["seed"], meta["mu"], meta["run"], meta["quantization"]) == (5, 1.15, "run1", "fp8")
    assert meta["size"] == "256x256"  # the image's, not the request's


def test_save_image_writes_a_thumbnail_beside_the_png(store):
    record = store.save_image(
        Image.new("RGB", (1024, 768)),
        GenerateRequest(prompt="big", width=1024, height=768),
        seed=1,
        duration_ms=100,
        model_id="test/model",
    )
    thumb = store.settings.outputs_dir / record.filename.replace(".png", ".webp")
    assert record.thumb_url == f"/outputs/{thumb.name}"
    with Image.open(thumb) as small:
        assert small.format == "WEBP"
        assert max(small.size) == store.THUMB_SIDE
    # The record read back from the gallery says the same, and one from before
    # thumbnails existed says nothing rather than pointing at a 404.
    assert store.list_images()[0][0].thumb_url == record.thumb_url
    thumb.unlink()
    assert store.list_images()[0][0].thumb_url is None


def test_deleting_an_image_takes_its_thumbnail(store):
    record = _add(store, "gone")
    thumb = store.settings.outputs_dir / record.filename.replace(".png", ".webp")
    assert thumb.exists()
    assert store.delete_image(record.id)
    assert not thumb.exists()


def test_favorites_survive_the_like_fallback(store):
    # An odd number of double quotes is what pushes the query off FTS: the
    # tokens go into MATCH quoted, and `26"` closes a string it never opened.
    _add(store, 'a 26" monitor on a desk', favorite=True)
    _add(store, 'a 26" monitor in a shop')
    _add(store, "something else entirely")

    items, total = store.list_images(favorites_only=True, search='26"')
    assert [i.prompt for i in items] == ['a 26" monitor on a desk']
    assert total == 1  # the badge counted every match before, favorites or not


def test_the_fallback_matches_tokens_in_any_order(store):
    # FTS ANDs tokens wherever they fall; a single LIKE over the raw string
    # demanded them contiguous and returned nothing here.
    _add(store, 'a figure, 6" tall, on a shelf')
    items, total = store.list_images(search='a 6" figure')
    assert total == 1
    assert items[0].prompt.startswith("a figure")


def test_the_fallback_does_not_treat_wildcards_as_wildcards(store):
    _add(store, 'shot at 50%" power')
    # Only matches if the % in the search is read as LIKE's "anything" wildcard.
    _add(store, 'shot at 50 percent power"')
    items, total = store.list_images(search='50%"')
    assert total == 1
    assert items[0].prompt.startswith('shot at 50%"')


def test_search_still_uses_fts_when_it_can(store):
    _add(store, "a fox walking in the snow")
    _add(store, "a fox asleep")
    items, total = store.list_images(search="fox snow")
    assert total == 1
    assert items[0].prompt.endswith("snow")


def test_paging_agrees_with_the_count(store):
    for i in range(5):
        _add(store, f"image {i}", favorite=i % 2 == 0)
    first, total = store.list_images(limit=2, offset=0, favorites_only=True)
    second, total_again = store.list_images(limit=2, offset=2, favorites_only=True)
    assert total == total_again == 3
    assert len(first) == 2 and len(second) == 1
    assert {r.id for r in first}.isdisjoint({r.id for r in second})


def test_paging_by_id_survives_an_insert_between_pages(store):
    # OFFSET paging over a table being written into: three images landing at
    # the top between two pages shifted every row down three, so the tail of
    # page one came back as the head of page two and three older images were
    # never shown. An id is stable under inserts and deletes alike.
    ids = [_add(store, f"image {i}").id for i in range(6)]
    first, _ = store.list_images(limit=2)
    assert [r.id for r in first] == [ids[5], ids[4]]
    _add(store, "landed since")
    _add(store, "and another")
    second, total = store.list_images(limit=2, before=first[-1].id)
    assert [r.id for r in second] == [ids[3], ids[2]]
    assert total == 8  # the badge counts everything the filters match
    # The anchor composes with the filters rather than replacing them.
    store.set_favorite(ids[1], True)
    store.set_favorite(ids[5], True)
    starred, _ = store.list_images(limit=5, favorites_only=True, before=ids[5])
    assert [r.id for r in starred] == [ids[1]]


def test_clear_history_keeps_favorites_and_their_files(store):
    kept = _add(store, "keep me", favorite=True)
    gone = _add(store, "clear me")
    result = store.clear_history(keep_favorites=True)

    assert result["records"] == 1 and result["files"] == 1
    assert (store.settings.outputs_dir / kept.filename).exists()
    assert not (store.settings.outputs_dir / gone.filename).exists()
    assert store.summarise()["records"] == 1
    # Thumbnails follow their PNGs both ways.
    outputs = store.settings.outputs_dir
    assert (outputs / kept.filename.replace(".png", ".webp")).exists()
    assert not (outputs / gone.filename.replace(".png", ".webp")).exists()


def test_clear_history_sweeps_orphans_but_not_partials(store):
    _add(store, "tracked")
    (store.settings.outputs_dir / "orphan.png").write_bytes(b"not really a png")
    (store.settings.outputs_dir / "orphan.webp").write_bytes(b"its thumbnail")
    # An image another process is part way through writing.
    (store.settings.outputs_dir / "inflight.png.part").write_bytes(b"half an image")

    result = store.clear_history()
    assert result["orphans"] == 1
    assert not (store.settings.outputs_dir / "orphan.webp").exists()
    assert (store.settings.outputs_dir / "inflight.png.part").exists()
