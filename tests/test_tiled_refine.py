"""The private four-region prototype with synthetic CPU tile renderers only."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from backend import tiled_refine as tiled


# Copied from the reviewed experiment's immutable receipt, never read from
# work/, the production gallery or a downloaded image during these tests.
FROZEN = (
    ("top-left", (0, 0, 1088, 1088), (0, 0, 1120, 1120),
     "edcd1374ab338524421162ce388462ef473d9d9bae2c844e9d698b4463ece732",
     "71c4c6609f40dfa39e4631054f6e40ed7fd0aa5b1fb3729a5e1f0a24f1a255fe"),
    ("top-right", (960, 0, 2048, 1088), (928, 0, 2048, 1120),
     "701ae96c806eb5e6a8380260dfa075d3306cf544d046ee0810b73c148d049890",
     "da7f7da891ca72fe12ec15347307f0380a6cd01115742533a4139379bec1e81d"),
    ("bottom-left", (0, 960, 1088, 2048), (0, 928, 1120, 2048),
     "124012ed2e09aa64b2b56ca41e95071a44d19b840d7253d4a29146215e9daee8",
     "df49ab70ec9a6a971e4ad933a72fa50eb9b7ad95a338a55c888a4f7adce5152c"),
    ("bottom-right", (960, 960, 2048, 2048), (928, 928, 2048, 2048),
     "88adb30666c4acc406cfec7d47eb4d95572d8e708af5397032bc911454ca5b4b",
     "60131ceb740f11f0b6c86fe0b5b04ae80ccad32ab9e61087c5cdc945bbc1bbe0"),
)
COLORS = ((211, 47, 91), (53, 197, 103), (71, 89, 229), (241, 173, 29))


def rgb_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


@pytest.fixture
def source():
    image = Image.new("RGB", (2048, 2048), (23, 79, 143))
    image.putpixel((0, 0), (3, 7, 11))
    image.putpixel((2047, 2047), (239, 131, 43))
    try:
        yield image
    finally:
        image.close()


def assert_closed(image):
    with pytest.raises(ValueError, match="closed"):
        image.getpixel((0, 0))


@pytest.fixture
def owned(monkeypatch, source):
    pictures = []
    composites = []
    original = SimpleNamespace(
        tile_mask=tiled.tile_mask, region=tiled.mask_crop.region,
        crop_inputs=tiled.mask_crop.crop_inputs, paste_back=tiled.mask_crop.paste_back,
        composite=tiled.refine.composite,
    )
    original_copy = source.copy

    def capture(image):
        pictures.append(image)
        return image

    def copied():
        return capture(original_copy())

    def mask(ordinal):
        return capture(original.tile_mask(ordinal))

    def crop_inputs(*args, **kwargs):
        inputs = original.crop_inputs(*args, **kwargs)
        pictures.extend(inputs)
        return inputs

    def paste_back(*args, **kwargs):
        return capture(original.paste_back(*args, **kwargs))

    def composite(mask, current, picture, *args, **kwargs):
        result = original.composite(mask, current, picture, *args, **kwargs)
        outside = np.asarray(mask) == 0
        assert np.array_equal(np.asarray(result)[outside], np.asarray(current)[outside])
        composites.append(result)
        return capture(result)

    def closed_except(final=None):
        for picture in {id(picture): picture for picture in pictures}.values():
            if picture is not final:
                assert_closed(picture)

    monkeypatch.setattr(source, "copy", copied)
    monkeypatch.setattr(tiled, "tile_mask", mask)
    monkeypatch.setattr(tiled.mask_crop, "crop_inputs", crop_inputs)
    monkeypatch.setattr(tiled.mask_crop, "paste_back", paste_back)
    monkeypatch.setattr(tiled.refine, "composite", composite)
    return SimpleNamespace(capture=capture, pictures=pictures, composites=composites,
                           original=original, closed_except=closed_except)


def solid_renderer(call):
    for completed in range(1, 10):
        call.on_step(completed, 9)
    return Image.new("RGB", call.plan.working, COLORS[call.ordinal])


@pytest.mark.parametrize("ordinal", range(4))
def test_frozen_masks_and_crop_geometry_match_the_reviewed_experiment(source, ordinal):
    plan = tiled.build_plan(source, 12346)
    tile = plan.tiles[ordinal]
    name, box, crop_box, mask_hash, raw_hash = FROZEN[ordinal]
    assert (tile.ordinal, tile.name, tile.paint_box, tile.crop_box, tile.working, tile.seed) == (
        ordinal, name, box, crop_box, (1040, 1040), 12346 + ordinal,
    )
    with tiled.tile_mask(ordinal) as mask:
        assert mask.mode == "L" and mask.size == (2048, 2048)
        digest = hashlib.sha256(b"L 2048x2048\n" + mask.tobytes()).hexdigest()
        assert digest == tile.mask_hash == mask_hash
        assert hashlib.sha256(mask.tobytes()).hexdigest() == tile.mask_bytes_sha256 == raw_hash
        crop = tiled.mask_crop.region(mask, tiled.RECIPE.crop)
        assert crop.box == crop_box and crop.working == (1040, 1040)


def test_the_feather_is_inward_integer_alpha_with_unchanged_canvas_edges():
    with tiled.tile_mask(0) as mask:
        assert [mask.getpixel(point) for point in ((0, 0), (1024, 0), (1025, 0),
            (1087, 0), (1087, 1087), (1088, 0))] == [255, 255, 251, 3, 3, 0]
    with tiled.tile_mask(3) as mask:
        assert [mask.getpixel(point) for point in ((959, 960), (960, 960), (961, 960),
            (1023, 1023), (2047, 2047))] == [0, 3, 3, 255, 255]


def test_recipe_pins_the_reviewed_private_experiment_and_has_no_mutable_geometry():
    recipe = tiled.RECIPE
    assert recipe.name == "private-provisional-sdxl-four-region-2048-v1"
    assert (recipe.model, recipe.revision) == (
        "cagliostrolab/animagine-xl-4.0", "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96",
    )
    assert (recipe.steps, recipe.guidance_scale, recipe.strength, recipe.num_images, recipe.mu) == (28, 5., .35, 1, None)
    assert (recipe.size, recipe.overlap, recipe.feather, recipe.latent_block) == ((2048, 2048), 128, 64, 8)
    assert recipe.order == tuple(row[0] for row in FROZEN)
    assert recipe.crop == tiled.CropSpec("bbox16", 32, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        recipe.crop.pad = 64


@pytest.mark.parametrize("ordinal", [-1, 4, True, 0., "0", None])
def test_mask_ordinal_requires_one_of_the_four_exact_integer_regions(ordinal):
    with pytest.raises(ValueError):
        tiled.tile_mask(ordinal)


@pytest.mark.parametrize("seed", [0, 12346, 2**63 - 2, 2**63 - 1])
def test_plan_wraps_tile_seeds_without_changing_the_outer_image_seed(source, seed):
    plan = tiled.build_plan(source, seed)
    assert [tile.seed for tile in plan.tiles] == [(seed + ordinal) % 2**63 for ordinal in range(4)]
    assert plan.source_rgb_sha256 == rgb_hash(source)
    assert isinstance(plan.tiles, tuple) and plan.recipe == tiled.RECIPE
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.tiles[0].seed = 17
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.recipe.steps = 12


@pytest.mark.parametrize("seed", [-1, 2**63, True, 12346., "12346", None])
def test_invalid_seed_refuses_before_any_renderer_call(source, seed):
    called = []
    with pytest.raises(ValueError):
        tiled.run(source, seed, lambda call: called.append(call))
    assert called == [] and source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("mode,size", [("L", (2048, 2048)), ("RGBA", (2048, 2048)),
                                      ("RGB", (1024, 1024)), ("RGB", (2048, 2032))])
def test_invalid_source_refuses_before_any_renderer_call(mode, size):
    called = []
    with Image.new(mode, size) as source:
        before = source.tobytes()
        with pytest.raises(ValueError):
            tiled.run(source, 12346, lambda call: called.append(call))
        assert called == [] and source.tobytes() == before


@pytest.mark.parametrize("changes", [
    {"name": "another-v1"}, {"model": "another/model"}, {"revision": "changed"},
    {"prompt": "another scene"}, {"negative_prompt": "another negative"},
    {"steps": 30}, {"steps": 28.}, {"guidance_scale": 4.}, {"strength": .4},
    {"num_images": 2}, {"num_images": True}, {"mu": 1.},
    {"size": (1024, 1024)}, {"size": (2048., 2048.)}, {"overlap": 64}, {"overlap": 128.},
    {"feather": 32}, {"feather": 64.},
    {"latent_block": 16}, {"latent_block": 8.},
    {"order": ("bottom-right", "bottom-left", "top-right", "top-left")},
    {"crop": tiled.CropSpec(pad=64)}, {"crop": tiled.CropSpec(pad=32.)},
    {"crop": tiled.CropSpec(tier=True)},
])
def test_unmeasured_recipe_or_wrong_field_types_refuse_before_the_renderer(source, changes):
    called = []
    recipe = dataclasses.replace(tiled.RECIPE, **changes)
    with pytest.raises(ValueError):
        tiled.run(source, 12346, lambda call: called.append(call), recipe=recipe)
    assert called == [] and source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("bad", ["renderer", "should_cancel", "on_progress"])
def test_noncallable_handlers_refuse_before_planning_or_rendering(source, owned, bad):
    called = []
    renderer = lambda call: called.append(call)
    options = {}
    if bad == "renderer":
        renderer = None
    else:
        options[bad] = False
    with pytest.raises(ValueError):
        tiled.run(source, 12346, renderer, **options)
    assert called == [] and owned.pictures == []
    assert source.getpixel((0, 0)) == (3, 7, 11)


def test_serial_calls_match_existing_crop_paste_and_composite_arithmetic_exactly(source, owned):
    before = source.tobytes()
    expected = Image.frombytes("RGB", source.size, before)
    progress = []
    calls = []
    result = None

    def renderer(call):
        nonlocal expected
        calls.append(call)
        assert (call.name, call.ordinal, call.seed) == (FROZEN[call.ordinal][0], len(calls) - 1, 12346 + call.ordinal)
        assert call.recipe == tiled.RECIPE
        with pytest.raises(dataclasses.FrozenInstanceError):
            call.plan.seed = 1
        with owned.original.tile_mask(call.ordinal) as mask:
            crop = owned.original.region(mask, tiled.RECIPE.crop)
            inputs = owned.original.crop_inputs(expected, mask, crop)
            try:
                for actual, oracle in zip((call.image, call.mask, call.pipeline_mask), inputs, strict=True):
                    assert actual.mode == oracle.mode and actual.size == oracle.size == (1040, 1040)
                    assert actual.tobytes() == oracle.tobytes()
            finally:
                for picture in inputs:
                    picture.close()
            raw = owned.capture(solid_renderer(call))
            with owned.original.paste_back(expected, raw, crop.box) as pasted:
                next_expected = owned.original.composite(mask, expected, pasted)
            expected.close()
            expected = next_expected
        return raw

    try:
        result = tiled.run(source, 12346, renderer, on_progress=lambda *step: progress.append(step))
        assert len(calls) == len(result.tiles) == 4
        assert isinstance(result.tiles, tuple) and result.image.mode == "RGB" and result.image.size == (2048, 2048)
        assert result.image.tobytes() == expected.tobytes()
        assert result.image.getpixel((1024, 1024)) == COLORS[3]
        assert source.tobytes() == before
        assert progress == [(ordinal, completed, 9) for ordinal in range(4) for completed in range(1, 10)]
        prior = result.plan.source_rgb_sha256
        for tile in result.tiles:
            assert tile.source_rgb_sha256 == prior and tile.raw_size == (1040, 1040) and tile.callback_count == 9
            prior = tile.result_rgb_sha256
        assert prior == rgb_hash(result.image)
        owned.closed_except(result.image)
        assert result.image.getpixel((0, 0))
    finally:
        expected.close()
        if result is not None:
            result.image.close()


@pytest.mark.parametrize("site", ["callback", "progress", "cancel"])
def test_renderer_cannot_swallow_a_callback_failure_or_cancellation_and_return_an_image(source, owned, site):
    raised = []
    cancel = False
    failure = RuntimeError("progress receiver failed")

    def progress(*args):
        if site == "progress":
            raise failure

    def renderer(call):
        nonlocal cancel
        cancel = site == "cancel"
        try:
            call.on_step(0 if site == "callback" else 1, 9)
        except (ValueError, RuntimeError, tiled.Cancelled) as exc:
            raised.append(exc)
        cancel = False
        return owned.capture(Image.new("RGB", call.plan.working))

    expected = tiled.Cancelled if site == "cancel" else ValueError if site == "callback" else RuntimeError
    with pytest.raises(expected) as caught:
        tiled.run(source, 12346, renderer, on_progress=progress, should_cancel=lambda: cancel)
    assert raised
    if site != "cancel":
        assert caught.value is raised[0]
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("raw", [None, 17, "not an image", [None], (None,)])
def test_none_or_other_nonimage_results_are_refused_with_complete_cleanup(source, owned, raw):
    with pytest.raises(ValueError):
        tiled.run(source, 12346, lambda call: raw)
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("container,count", [(list, 1), (list, 2), (tuple, 1), (tuple, 2)])
def test_malformed_image_containers_release_every_transferred_fresh_member(source, owned, container, count):
    outputs = []

    def renderer(call):
        outputs.extend(owned.capture(Image.new("RGB", call.plan.working)) for _ in range(count))
        return container(outputs)

    with pytest.raises(ValueError):
        tiled.run(source, 12346, renderer)
    owned.closed_except()
    assert all(output is not source for output in outputs)
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("container", [list, tuple])
def test_malformed_containers_cannot_transfer_ownership_of_the_callers_source(source, owned, container):
    def renderer(call):
        fresh = owned.capture(Image.new("RGB", call.plan.working))
        return container([source, call.image, fresh])

    with pytest.raises(ValueError):
        tiled.run(source, 12346, renderer)
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("alias", ["source", "image", "mask", "pipeline_mask"])
def test_returned_borrowed_alias_is_refused_without_closing_the_source(source, owned, alias):
    with pytest.raises(ValueError, match="alias"):
        tiled.run(source, 12346, lambda call: source if alias == "source" else getattr(call, alias))
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("mode,size", [("L", (1040, 1040)), ("RGBA", (1040, 1040)),
                                      ("RGB", (1039, 1040)), ("RGB", (2048, 2048))])
def test_bad_result_mode_or_size_is_refused_and_the_result_is_closed(source, owned, mode, size):
    def renderer(call):
        for completed in range(1, 10):
            call.on_step(completed, 9)
        return owned.capture(Image.new(mode, size))

    with pytest.raises(ValueError):
        tiled.run(source, 12346, renderer)
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("steps", [[], [(1, 8)], [(0, 9)], [(True, 9)], [(1, 9), (1, 9)],
                                   [(1, 9), (3, 9)], [(n, 9) for n in range(1, 9)]])
def test_missing_or_malformed_progress_never_returns_a_final_result(source, owned, steps):
    def renderer(call):
        for completed, total in steps:
            call.on_step(completed, total)
        return owned.capture(Image.new("RGB", call.plan.working))

    with pytest.raises(ValueError):
        tiled.run(source, 12346, renderer)
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


def test_cancel_before_work_keeps_the_source_alive_and_never_renders(source, owned):
    calls = []
    with pytest.raises(tiled.Cancelled):
        tiled.run(source, 12346, lambda call: calls.append(call), should_cancel=lambda: True)
    assert calls == []
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


def test_cancel_inside_a_later_tile_discards_all_owned_work(source, owned):
    calls = []
    cancelled = False

    def renderer(call):
        nonlocal cancelled
        calls.append(call.ordinal)
        for completed in range(1, 10):
            cancelled = call.ordinal == 2 and completed == 3
            call.on_step(completed, 9)
        return owned.capture(Image.new("RGB", call.plan.working))

    with pytest.raises(tiled.Cancelled):
        tiled.run(source, 12346, renderer, should_cancel=lambda: cancelled)
    assert calls == [0, 1, 2]
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


def test_cancellation_after_render_closes_its_transferred_fresh_result(source, owned):
    cancelled = False
    calls = []

    def renderer(call):
        nonlocal cancelled
        calls.append(call.ordinal)
        raw = owned.capture(solid_renderer(call))
        cancelled = True
        return raw

    with pytest.raises(tiled.Cancelled):
        tiled.run(source, 12346, renderer, should_cancel=lambda: cancelled)
    assert calls == [0]
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


def test_pipeline_mask_uses_the_frozen_block_instead_of_a_mutable_global(source, owned, monkeypatch):
    monkeypatch.setattr(tiled.refine, "LATENT_BLOCK", 16)
    original = tiled.mask_crop.crop_inputs
    blocks = []

    def observed(*args, **kwargs):
        blocks.append(kwargs.get("block"))
        return original(*args, **kwargs)

    monkeypatch.setattr(tiled.mask_crop, "crop_inputs", observed)
    stopped = RuntimeError("stop after observing first inputs")

    def renderer(call):
        raise stopped

    with pytest.raises(RuntimeError) as caught:
        tiled.run(source, 12346, renderer)
    assert caught.value is stopped and blocks == [8]
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("failure_site", ["renderer", "progress", "crop_inputs", "paste_back", "composite"])
def test_later_renderer_progress_and_helper_failures_preserve_the_primary_error_and_close_owned_images(
    source, owned, monkeypatch, failure_site,
):
    failure = RuntimeError("injected later-tile failure")
    current = -1

    def renderer(call):
        nonlocal current
        current = call.ordinal
        if current == 1 and failure_site == "renderer":
            raise failure
        return owned.capture(solid_renderer(call))

    def progress(ordinal, completed, total):
        if ordinal == 1 and completed == 3 and failure_site == "progress":
            raise failure

    if failure_site in {"crop_inputs", "paste_back", "composite"}:
        module = tiled.refine if failure_site == "composite" else tiled.mask_crop
        original = getattr(module, failure_site)
        count = 0

        def broken(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise failure
            return original(*args, **kwargs)

        monkeypatch.setattr(module, failure_site, broken)
    with pytest.raises(RuntimeError) as caught:
        tiled.run(source, 12346, renderer, on_progress=progress)
    assert caught.value is failure
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


@pytest.mark.parametrize("drift", ["mask", "crop", "working"])
def test_authoritative_geometry_drift_refuses_before_a_renderer(source, owned, monkeypatch, drift):
    if drift == "mask":
        original = tiled.tile_mask

        def changed(ordinal):
            mask = original(ordinal)
            mask.putpixel((0, 0), 0)
            return mask

        monkeypatch.setattr(tiled, "tile_mask", changed)
    else:
        original = tiled.mask_crop.region

        def changed(*args, **kwargs):
            crop = original(*args, **kwargs)
            return crop._replace(box=(0, 0, 1104, 1120)) if drift == "crop" else crop._replace(working=(1024, 1024))

        monkeypatch.setattr(tiled.mask_crop, "region", changed)
    calls = []
    with pytest.raises(ValueError):
        tiled.run(source, 12346, lambda call: calls.append(call))
    assert calls == []
    owned.closed_except()
    assert source.getpixel((0, 0)) == (3, 7, 11)


def test_importing_the_unregistered_prototype_does_not_import_frameworks_or_use_the_network():
    script = r'''
import builtins, socket, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {"torch", "diffusers", "transformers"}:
        assert "backend.config" in sys.modules, "framework import preceded config"
        raise AssertionError("cold prototype import reached " + name)
    return original(name, *args, **kwargs)
def network(*args, **kwargs):
    raise AssertionError("cold prototype import used the network")
builtins.__import__ = guarded
socket.socket.connect = network
socket.socket.connect_ex = network
socket.create_connection = network
from backend import tiled_refine
assert tiled_refine.PRIVATE_PROVISIONAL.startswith("private-provisional-")
from PIL import Image
class RendererReached(Exception):
    pass
source = Image.new("RGB", (2048, 2048), (7, 17, 29))
def renderer(call):
    assert "backend.config" in sys.modules, "renderer preceded config"
    raise RendererReached
try:
    try:
        tiled_refine.run(source, 12346, renderer)
    except RendererReached:
        pass
    else:
        raise AssertionError("the renderer was not reached")
    assert source.getpixel((0, 0)) == (7, 17, 29)
    assert not any(name in sys.modules for name in ("torch", "diffusers", "transformers"))
finally:
    source.close()
'''
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
