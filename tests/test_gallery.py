"""Search, paging and the file/row pair that has to stay consistent."""
from __future__ import annotations

import pytest
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
        batch=4,
    )
    with Image.open(store.settings.outputs_dir / record.filename) as png:
        meta = json.loads(png.info["auradream"])
    # `batch` is no part of the reproduction set, but like `job` and `run`
    # it is what a reindex has to restore it from: More seeds reads it.
    assert set(meta) == {
        "prompt", "negative_prompt", "steps", "guidance_scale", "true_cfg", "mu", "seed",
        "model", "quantization", "job", "run", "size", "variables", "hash",
        "reference", "lora", "lora_scale", "refine", "strength",
        "reference_sources", "refine_source", "batch", "mask", "mask_source", "mask_blend",
        "outpaint", "outpaint_fill",
    }
    assert (meta["seed"], meta["mu"], meta["run"], meta["quantization"]) == (5, 1.15, "run1", "fp8")
    assert meta["batch"] == record.batch == 4
    assert meta["size"] == "256x256"  # the image's, not the request's
    assert meta["variables"] is None  # a plain prompt
    # Absent rather than empty: a text-to-image row with no adapter says so by
    # null in all three, which is what `applied()` normalises them to.
    assert (meta["reference"], meta["lora"], meta["lora_scale"]) == (None, None, None)
    # And it is not a refine pass: no source, and so no strength either.
    assert (meta["refine"], meta["strength"]) == (None, None)
    # Nor an inpaint: its mask and what it was are null together.
    assert (meta["mask"], meta["mask_source"], meta["mask_blend"]) == (None, None, None)
    assert meta["hash"] == record.content_hash and len(meta["hash"]) == 16


def test_the_hash_is_of_the_pixels_and_finds_the_same_picture(store):
    # Two renders that came out identical hash the same whatever their seed
    # or file; a pixel's difference does not. `hash:` then finds every image
    # that is the same picture, which is what "did this axis do anything"
    # needs answered exactly rather than by eye.
    plain = Image.new("RGB", (16, 16), (10, 20, 30))
    other = Image.new("RGB", (16, 16), (10, 20, 31))
    a = store.save_image(plain, GenerateRequest(prompt="x", width=256, height=256), seed=1,
                         duration_ms=1, model_id="m")
    b = store.save_image(plain.copy(), GenerateRequest(prompt="x", width=256, height=256), seed=2,
                         duration_ms=1, model_id="m")
    c = store.save_image(other, GenerateRequest(prompt="x", width=256, height=256), seed=3,
                         duration_ms=1, model_id="m")
    assert a.content_hash == b.content_hash != c.content_hash
    with Image.open(store.settings.outputs_dir / a.filename) as png:
        assert store.content_hash(png) == a.content_hash  # the file round-trips it
    found, total, _ = store.list_images(search=f"hash:{a.content_hash}")
    assert total == 2 and {r.seed for r in found} == {1, 2}
    # A run of identical images says so in one number.
    d = store.save_image(plain, GenerateRequest(prompt="x", width=256, height=256), seed=4,
                         duration_ms=1, model_id="m", run_id="r1")
    e = store.save_image(plain, GenerateRequest(prompt="x", width=256, height=256), seed=5,
                         duration_ms=1, model_id="m", run_id="r1")
    assert (d.content_hash, e.content_hash) == (a.content_hash, a.content_hash)
    run = store.runs()[0]
    assert (run.run_id, run.images, run.distinct) == ("r1", 2, 1)


def test_id_finds_the_image_the_page_names(store):
    # "#12" is how the page names an image - "refine of #12", the Refine
    # dialog's title - so `id:12` is the search that finds it, a span finds
    # several, and a strict scope (Delete unstarred, `clear --where`) takes it
    # as it takes any other key:value term.
    a, b, c = (_add(store, f"a fox, take {n}", seed=n) for n in (1, 2, 3))
    found, total, _ = store.list_images(search=f"id:{b.id}")
    assert total == 1 and found[0].id == b.id
    found, total, _ = store.list_images(search=f"id:{a.id}-{b.id} fox")
    assert {r.id for r in found} == {a.id, b.id}
    assert [r.id for r in store.list_images(search=f"id:>{b.id}")[0]] == [c.id]
    assert store.matching_preview(f"id:{c.id}")["records"] == 1
    with pytest.raises(ValueError, match="id: 'twelve' is not a number"):
        store.list_images(search="id:twelve")


def test_a_grid_cells_variables_round_trip(store):
    # The assignment that made the cell, beside the prose it made: the sheet
    # lays a 2x2 night out from it rather than reconstructing it from words.
    record = store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="big dog on grass", width=256, height=256),
        seed=1, duration_ms=1, model_id="m", variables={"size": "big", "ground": "grass"},
    )
    assert record.variables == {"size": "big", "ground": "grass"}
    assert store.list_images()[0][0].variables == {"size": "big", "ground": "grass"}
    import json

    with Image.open(store.settings.outputs_dir / record.filename) as png:
        assert json.loads(png.info["auradream"])["variables"] == {
            "size": "big",
            "ground": "grass",
        }


def test_save_image_writes_a_thumbnail_in_its_own_directory(store):
    record = store.save_image(
        Image.new("RGB", (1024, 768)),
        GenerateRequest(prompt="big", width=1024, height=768),
        seed=1,
        duration_ms=100,
        model_id="test/model",
    )
    thumb = store.settings.outputs_dir / "thumbnails" / record.filename.replace(".png", ".webp")
    assert record.thumb_url == f"/outputs/thumbnails/{thumb.name}"
    assert not list(store.settings.outputs_dir.glob("*.webp"))
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
    thumb = store.settings.outputs_dir / "thumbnails" / record.filename.replace(".png", ".webp")
    assert thumb.exists()
    assert store.delete_image(record.id)
    assert not thumb.exists()


def test_two_deletes_of_one_image_answer_once(store):
    # A double-clicked ⌦ or a second tab: both deletes used to find the row
    # before either removed it, and both said they had deleted it - the page
    # then had no "already deleted" to go on.
    import threading

    record = _add(store, "gone twice")
    start = threading.Barrier(6)
    answers: list[bool] = []

    def delete() -> None:
        start.wait()
        answers.append(store.delete_image(record.id))

    threads = [threading.Thread(target=delete) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(answers) == [False] * 5 + [True]
    assert store.get_image(record.id) is None


def test_favorites_survive_the_like_fallback(store):
    # An odd number of double quotes is what pushes the query off FTS: the
    # tokens go into MATCH quoted, and `26"` closes a string it never opened.
    _add(store, 'a 26" monitor on a desk', favorite=True)
    _add(store, 'a 26" monitor in a shop')
    _add(store, "something else entirely")

    items, total, _ = store.list_images(favorites_only=True, search='26"')
    assert [i.prompt for i in items] == ['a 26" monitor on a desk']
    assert total == 1  # the badge counted every match before, favorites or not


def test_the_fallback_matches_tokens_in_any_order(store):
    # FTS ANDs tokens wherever they fall; a single LIKE over the raw string
    # demanded them contiguous and returned nothing here.
    _add(store, 'a figure, 6" tall, on a shelf')
    items, total, _ = store.list_images(search='a 6" figure')
    assert total == 1
    assert items[0].prompt.startswith("a figure")


def test_the_fallback_does_not_treat_wildcards_as_wildcards(store):
    _add(store, 'shot at 50%" power')
    # Only matches if the % in the search is read as LIKE's "anything" wildcard.
    _add(store, 'shot at 50 percent power"')
    items, total, _ = store.list_images(search='50%"')
    assert total == 1
    assert items[0].prompt.startswith('shot at 50%"')


def test_search_still_uses_fts_when_it_can(store):
    _add(store, "a fox walking in the snow")
    _add(store, "a fox asleep")
    items, total, _ = store.list_images(search="fox snow")
    assert total == 1
    assert items[0].prompt.endswith("snow")


def test_paging_agrees_with_the_count(store):
    for i in range(5):
        _add(store, f"image {i}", favorite=i % 2 == 0)
    first, total, _ = store.list_images(limit=2, offset=0, favorites_only=True)
    second, total_again, _ = store.list_images(limit=2, offset=2, favorites_only=True)
    assert total == total_again == 3
    assert len(first) == 2 and len(second) == 1
    assert {r.id for r in first}.isdisjoint({r.id for r in second})


def test_paging_by_id_survives_an_insert_between_pages(store):
    # OFFSET paging over a table being written into: three images landing at
    # the top between two pages shifted every row down three, so the tail of
    # page one came back as the head of page two and three older images were
    # never shown. An id is stable under inserts and deletes alike.
    ids = [_add(store, f"image {i}").id for i in range(6)]
    first, _, _ = store.list_images(limit=2)
    assert [r.id for r in first] == [ids[5], ids[4]]
    _add(store, "landed since")
    _add(store, "and another")
    second, total, _ = store.list_images(limit=2, before=first[-1].id)
    assert [r.id for r in second] == [ids[3], ids[2]]
    assert total == 8  # the badge counts everything the filters match
    # The anchor composes with the filters rather than replacing them.
    store.set_favorite(ids[1], True)
    store.set_favorite(ids[5], True)
    starred, _, _ = store.list_images(limit=5, favorites_only=True, before=ids[5])
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
    assert (outputs / "thumbnails" / kept.filename.replace(".png", ".webp")).exists()
    assert not (outputs / "thumbnails" / gone.filename.replace(".png", ".webp")).exists()


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


def test_delete_matching_needs_a_scope_and_holds_to_its_count(store):
    import pytest

    starred = _add(store, "keep me", favorite=True)
    _add(store, "clear me")
    # `favorite = 0` is a scope of its own; nothing at all is not.
    with pytest.raises(ValueError, match="filter is required"):
        store.delete_matching(None)
    with pytest.raises(store.SelectionChanged) as caught:
        store.delete_matching(None, keep_favorites=True, expected=5)
    assert (caught.value.found, caught.value.expected) == (1, 5)
    assert store.summarise()["records"] == 2

    result = store.delete_matching(None, keep_favorites=True, expected=1)
    assert result["records"] == 1 and len(result["ids"]) == 1
    assert [r.id for r in store.list_images()[0]] == [starred.id]


def test_two_bins_of_one_image_answer_once(store):
    # As two deletes do: the lookup and the write share the lock, so a
    # double press says binned once and already once.
    import threading

    record = _add(store, "binned twice")
    start = threading.Barrier(6)
    answers: list[dict] = []

    def bin_it() -> None:
        start.wait()
        answers.append(store.bin_images([record.id, 9999]))

    threads = [threading.Thread(target=bin_it) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(len(a["binned"]) for a in answers) == [0] * 5 + [1]
    assert sorted(len(a["already"]) for a in answers) == [0] + [1] * 5
    assert all(a["missing"] == [9999] for a in answers)
    # Nothing moves on disk: a restore is one UPDATE.
    assert (store.settings.outputs_dir / record.filename).exists()


def test_restore_brings_a_binned_image_back(store):
    record = _add(store, "back again", favorite=True)
    live = _add(store, "never binned")
    store.bin_images([record.id])
    assert store.restore_images([record.id, live.id, 9999]) == {
        "restored": [record.id], "not_binned": [live.id], "missing": [9999],
    }
    back = store.get_image(record.id)
    assert back.binned_at is None and back.favorite is True
    assert store.restore_images([record.id])["not_binned"] == [record.id]


def test_emptying_the_bin_deletes_its_rows_and_files_and_holds_to_its_count(store):
    outputs = store.settings.outputs_dir
    live = _add(store, "stays")
    gone = [_add(store, "goes", seed=seed) for seed in (1, 2)]
    store.bin_images([r.id for r in gone])
    summary = store.bin_summary()
    assert summary["records"] == 2 and summary["bytes"] > 0 and summary["sources_in_use"] == []

    with pytest.raises(store.SelectionChanged) as caught:
        store.empty_bin(expected=1)
    assert (caught.value.found, caught.value.expected) == (2, 1)
    assert all((outputs / r.filename).exists() for r in gone)

    result = store.empty_bin(expected=2)
    assert (result["records"], result["files"], result["ids"]) == (2, 2, [r.id for r in gone])
    for record in gone:
        assert not (outputs / record.filename).exists()
        assert not (outputs / "thumbnails" / record.filename.replace(".png", ".webp")).exists()
    assert (outputs / live.filename).exists()
    assert store.bin_summary()["records"] == 0 and store.summarise()["records"] == 1


def test_the_bin_summary_names_binned_images_a_live_one_needs(store):
    source = _add(store, "source")
    reference = _add(store, "reference", seed=2)
    unused = _add(store, "unused", seed=3)
    store.save_image(
        Image.new("RGB", (16, 16)),
        GenerateRequest(prompt="a pass", width=256, height=256, refine=source.id, strength=0.5,
                        reference=[reference.id]),
        seed=4, duration_ms=100, model_id="test/model",
    )
    store.bin_images([source.id, reference.id, unused.id])
    assert store.bin_summary()["sources_in_use"] == [source.id, reference.id]


def test_clear_history_does_not_sweep_binned_images_as_orphans(store):
    # The sweep reads every row: a live-only read would take a binned PNG as
    # untracked and delete it under its row. A binned source of a live star
    # is kept as a row, so its file has to survive the sweep.
    source = _add(store, "a source, binned")
    star = store.save_image(
        Image.new("RGB", (16, 16)),
        GenerateRequest(prompt="a starred pass", width=256, height=256, refine=source.id, strength=0.5),
        seed=2, duration_ms=100, model_id="test/model",
    )
    store.set_favorite(star.id, True)
    unkept = _add(store, "binned, unkept", seed=3)
    store.bin_images([source.id, unkept.id])
    assert store.summarise()["orphan_files"] == 0
    result = store.clear_history(keep_favorites=True)
    assert (result["records"], result["orphans"]) == (1, 0)
    assert (store.settings.outputs_dir / source.filename).exists()
    assert not (store.settings.outputs_dir / unkept.filename).exists()
    assert store.get_image(source.id, include_binned=True).binned_at is not None
