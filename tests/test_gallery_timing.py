"""The model picker's gallery timing: each checkpoint's typical direct render
in this gallery's own history (`gallery_reports.gallery_timing`)."""
from __future__ import annotations

import dataclasses

from PIL import Image

from backend import gallery_reports, storage
from backend.schemas import GenerateRequest


FLUX = "black-forest-labs/FLUX.1-Krea-dev"
ANIME = "cagliostrolab/animagine-xl-4.0"


def _plant(model=FLUX, *, width=256, height=256, steps=28, duration=10_000, quantization="fp8", **update):
    request = GenerateRequest(prompt="x", width=width, height=height, steps=steps)
    return storage.save_image(
        Image.new("RGB", (width, height)), request.model_copy(update=update), seed=1,
        duration_ms=duration, model_id=model, quantization=quantization,
    )


def _set(image_id, column, value):
    with storage._session() as conn:
        conn.execute(f"UPDATE images SET {column} = ? WHERE id = ?", (value, image_id))


def test_an_empty_history_answers_nothing(store):
    assert gallery_reports.gallery_timing("fp8") == {}


def test_a_missing_database_answers_nothing_and_is_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "gone" / "auradream.db"
    monkeypatch.setattr(storage, "settings", dataclasses.replace(storage.settings, db_path=missing))
    assert gallery_reports.gallery_timing("fp8") == {}
    assert not missing.exists()


def test_refine_edit_and_non_eager_rows_are_left_out(store):
    source = _plant(duration=9_000)
    _plant(duration=90_000, refine=source.id, strength=0.5)
    edited = _plant(duration=90_000)
    _set(edited.id, "edit", '{"recipe": "kontext"}')
    compiled = _plant(duration=90_000)
    _set(compiled.id, "execution", '{"compilation":"regional-v1"}')
    _plant(duration=0)  # a reindexed row records no duration
    _plant(duration=90_000, quantization="int8")  # another quantization
    assert gallery_reports.gallery_timing("fp8") == {
        FLUX: {"median_ms": 9_000, "width": 256, "height": 256, "steps": 28, "images": 1},
    }
    assert gallery_reports.gallery_timing("int8")[FLUX]["median_ms"] == 90_000
    assert gallery_reports.gallery_timing("none") == {}


def test_the_most_used_size_and_step_count_give_the_median_and_its_count(store):
    for duration in (12_000, 30_000, 11_000, 10_000):
        _plant(width=512, height=512, steps=28, duration=duration)
    for duration in (50_000, 51_000):
        _plant(width=512, height=512, steps=50, duration=duration)
    # More images at 1024x768 in all, but spread over three step counts:
    # the size is chosen first, then the steps within it.
    for steps in (20, 24, 28, 32, 36, 40, 44):
        _plant(width=1024, height=768, steps=steps, duration=20_000)
    for duration in (3_000, 3_200, 2_900):
        _plant(ANIME, width=1024, height=1024, steps=28, duration=duration)
    timing = gallery_reports.gallery_timing("fp8")
    assert timing[FLUX] == {"median_ms": 20_000, "width": 1024, "height": 768, "steps": 44, "images": 1}
    assert timing[ANIME] == {"median_ms": 3_000, "width": 1024, "height": 1024, "steps": 28, "images": 3}
    # A slow first image moves the median no further than one place.
    _plant(ANIME, width=1024, height=1024, steps=28, duration=60_000)
    assert gallery_reports.gallery_timing("fp8")[ANIME] == {
        "median_ms": 3_200, "width": 1024, "height": 1024, "steps": 28, "images": 4,
    }


def test_the_step_count_is_the_one_used_most_at_the_chosen_size(store):
    for duration in (12_000, 30_000, 11_000):
        _plant(width=512, height=512, steps=28, duration=duration)
    _plant(width=512, height=512, steps=50, duration=50_000)
    assert gallery_reports.gallery_timing("fp8")[FLUX] == {
        "median_ms": 12_000, "width": 512, "height": 512, "steps": 28, "images": 3,
    }


def test_only_the_latest_renders_count_and_a_tie_goes_to_the_newer(store):
    _plant(width=512, height=512, duration=5_000)
    _plant(width=768, height=768, duration=7_000)
    assert gallery_reports.gallery_timing("fp8")[FLUX]["width"] == 768
    for _ in range(3):
        _plant(width=1024, height=1024, duration=9_000)
    timing = gallery_reports.gallery_timing("fp8", sample=2)
    assert timing[FLUX] == {"median_ms": 9_000, "width": 1024, "height": 1024, "steps": 28, "images": 2}
