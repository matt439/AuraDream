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


def test_clear_history_keeps_favorites_and_their_files(store):
    kept = _add(store, "keep me", favorite=True)
    gone = _add(store, "clear me")
    result = store.clear_history(keep_favorites=True)

    assert result["records"] == 1 and result["files"] == 1
    assert (store.settings.outputs_dir / kept.filename).exists()
    assert not (store.settings.outputs_dir / gone.filename).exists()
    assert store.summarise()["records"] == 1


def test_clear_history_sweeps_orphans_but_not_partials(store):
    _add(store, "tracked")
    (store.settings.outputs_dir / "orphan.png").write_bytes(b"not really a png")
    # An image another process is part way through writing.
    (store.settings.outputs_dir / "inflight.png.part").write_bytes(b"half an image")

    result = store.clear_history()
    assert result["orphans"] == 1
    assert (store.settings.outputs_dir / "inflight.png.part").exists()
