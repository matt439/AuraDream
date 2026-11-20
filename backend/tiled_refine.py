"""Private, provisional GEN#49 four-region CPU orchestration.

An injected renderer receives the frozen tile plan and borrowed crop inputs.
The module consumes a prepared canvas and has no model or gallery access. Its fresh
RGB output transfers ownership to this module; only a complete final composite
transfers ownership to the caller. The caller's source is always borrowed.
Fresh images in a malformed flat list/tuple return also transfer ownership and
are closed on refusal; aliases of the caller's source are never adopted.

This proves serial orchestration, not pipeline or product qualification. The
2048-square input is already prepared: the reviewed experiment used a CPU
Lanczos 2x fixture, not a native 2048-square diffusion render.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, fields

import numpy as np
from PIL import Image

from . import mask_crop, refine, seeds, tiled_refine_contract
from .backends.base import Cancelled
from .record_fields import mask_hash

PRIVATE_PROVISIONAL = tiled_refine_contract.RECIPE_NAME
_SIZE = tiled_refine_contract.CANVAS_SIZE
_LAYOUT = (
    ("top-left", (0, 0, 1088, 1088)),
    ("top-right", (960, 0, 2048, 1088)),
    ("bottom-left", (0, 960, 1088, 2048)),
    ("bottom-right", (960, 960, 2048, 2048)),
)
TILE_COUNT = len(_LAYOUT)
_MASK_HASHES = (
    "edcd1374ab338524421162ce388462ef473d9d9bae2c844e9d698b4463ece732",
    "701ae96c806eb5e6a8380260dfa075d3306cf544d046ee0810b73c148d049890",
    "124012ed2e09aa64b2b56ca41e95071a44d19b840d7253d4a29146215e9daee8",
    "88adb30666c4acc406cfec7d47eb4d95572d8e708af5397032bc911454ca5b4b",
)
_CROP_BOXES = (
    (0, 0, 1120, 1120), (928, 0, 2048, 1120),
    (0, 928, 1120, 2048), (928, 928, 2048, 2048),
)
_DENOISE_STEPS = 9  # The frozen SDXL recipe: floor(28 * .35).


@dataclass(frozen=True)
class CropSpec:
    rule: str = "bbox16"
    pad: int = 32
    tier: int = 1


@dataclass(frozen=True)
class Recipe:
    name: str = PRIVATE_PROVISIONAL
    model: str = tiled_refine_contract.MODEL_ID
    revision: str = tiled_refine_contract.REVISION
    prompt: str = (
        "masterpiece, best quality, anime scenery, continuous wide landscape, "
        "a stone bridge crossing horizontally through the center over a river "
        "flowing vertically through the center, wooden cabin on the riverbank, "
        "tall pine trees with fine branches, textured stonework, snowy mountains "
        "across the horizon, morning sunlight"
    )
    negative_prompt: str = "low quality, blurry"
    steps: int = tiled_refine_contract.STEPS
    guidance_scale: float = tiled_refine_contract.GUIDANCE_SCALE
    strength: float = tiled_refine_contract.STRENGTH
    num_images: int = 1
    mu: None = None
    size: tuple[int, int] = _SIZE
    crop: CropSpec = CropSpec()
    overlap: int = 128
    feather: int = 64
    latent_block: int = 8
    order: tuple[str, ...] = tuple(name for name, _ in _LAYOUT)


RECIPE = Recipe()


@dataclass(frozen=True)
class TilePlan:
    ordinal: int
    name: str
    paint_box: tuple[int, int, int, int]
    mask_hash: str
    mask_bytes_sha256: str
    crop_box: tuple[int, int, int, int]
    working: tuple[int, int]
    seed: int


@dataclass(frozen=True)
class Plan:
    recipe: Recipe
    source_rgb_sha256: str
    tiles: tuple[TilePlan, ...]


@dataclass(frozen=True)
class TileCall:
    """Borrowed inputs, valid only during the renderer call.

    The renderer must not mutate, close or retain them. Return exactly one
    fresh RGB image at ``plan.working``; do not return a list or an input alias.
    Report every denoising step through ``on_step(1..9, 9)``. The wrapper polls
    cancellation on either side of progress delivery.
    """

    plan: TilePlan
    recipe: Recipe
    image: Image.Image
    mask: Image.Image
    pipeline_mask: Image.Image
    on_step: Callable[[int, int], None]

    @property
    def ordinal(self) -> int:
        return self.plan.ordinal

    @property
    def name(self) -> str:
        return self.plan.name

    @property
    def seed(self) -> int:
        return self.plan.seed


@dataclass(frozen=True)
class TileResult:
    plan: TilePlan
    raw_size: tuple[int, int]
    callback_count: int
    source_rgb_sha256: str
    result_rgb_sha256: str


@dataclass(frozen=True)
class PrototypeResult:
    """Complete result; the caller owns ``image`` and must close it."""

    image: Image.Image
    plan: Plan
    tiles: tuple[TileResult, ...]


Renderer = Callable[[TileCall], Image.Image]
Progress = Callable[[int, int, int], None]


def _rgb_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _validate(source: Image.Image, image_seed: int, recipe: Recipe) -> None:
    if (
        type(recipe) is not Recipe
        or any(type(getattr(recipe, field.name)) is not type(getattr(RECIPE, field.name)) for field in fields(RECIPE))
        or any(type(getattr(recipe.crop, field.name)) is not type(getattr(RECIPE.crop, field.name)) for field in fields(RECIPE.crop))
        or any(type(side) is not int for side in recipe.size)
        or any(type(name) is not str for name in recipe.order)
        or recipe != RECIPE
    ):
        raise ValueError("only the frozen private provisional four-region recipe is available")
    if not isinstance(source, Image.Image) or source.mode != "RGB" or source.size != _SIZE:
        raise ValueError("the private prototype requires an already prepared 2048x2048 RGB source")
    if type(image_seed) is not int or not 0 <= image_seed <= seeds.MAX_SEED:
        raise ValueError("the private prototype requires an exact nonnegative storable integer seed")


def tile_mask(ordinal: int) -> Image.Image:
    """Owned full-size mask, using the reviewed experiment's exact integers.

    Canvas edges stay 255. At an interior edge distance ``d`` (1 at its first
    painted pixel), the level is ``min(255, max(1, d * 255 // 64))``. Thus the
    edge is 3, then 7, not 1. Corners take the minimum of the two ramps.
    """
    if type(ordinal) is not int or not 0 <= ordinal < len(_LAYOUT):
        raise ValueError("the private prototype has exactly four ordered regions")
    _, (x1, y1, x2, y2) = _LAYOUT[ordinal]
    x, y = np.arange(x1, x2), np.arange(y1, y2)
    ax = np.full(len(x), 255, dtype=np.int32)
    ay = np.full(len(y), 255, dtype=np.int32)
    for distance in ([x - x1 + 1] if x1 else []) + ([x2 - x] if x2 < _SIZE[0] else []):
        ax = np.minimum(ax, np.maximum(1, distance * 255 // RECIPE.feather))
    for distance in ([y - y1 + 1] if y1 else []) + ([y2 - y] if y2 < _SIZE[1] else []):
        ay = np.minimum(ay, np.maximum(1, distance * 255 // RECIPE.feather))
    levels = np.zeros((_SIZE[1], _SIZE[0]), dtype=np.uint8)
    levels[y1:y2, x1:x2] = np.minimum(ay[:, None], ax[None, :]).astype(np.uint8)
    return Image.fromarray(levels)


def build_plan(source: Image.Image, image_seed: int, *, recipe: Recipe = RECIPE) -> Plan:
    """Prove the bounded input and derive immutable geometry without a renderer."""
    _validate(source, image_seed, recipe)
    tiles = []
    for ordinal, (name, box) in enumerate(_LAYOUT):
        with closing(tile_mask(ordinal)) as painted:
            crop = mask_crop.region(painted, recipe.crop)
            digest = mask_hash(painted)
            if digest != _MASK_HASHES[ordinal] or crop.box != _CROP_BOXES[ordinal] or crop.working != (1040, 1040):
                raise ValueError("the private prototype no longer matches its frozen mask/crop geometry")
            tiles.append(TilePlan(
                ordinal, name, box, digest,
                hashlib.sha256(painted.tobytes()).hexdigest(), crop.box, crop.working,
                seeds.wrap_seed(image_seed + ordinal),
            ))
    return Plan(recipe, _rgb_hash(source), tuple(tiles))


def _close_owned(images: list[Image.Image], borrowed: Image.Image) -> None:
    seen = {id(borrowed)}
    for image in reversed(images):
        if id(image) not in seen:
            seen.add(id(image))
            image.close()


def run(
    source: Image.Image,
    image_seed: int,
    renderer: Renderer,
    *,
    recipe: Recipe = RECIPE,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Progress | None = None,
    before_tile: Callable[[], None] | None = None,
) -> PrototypeResult:
    """Run all four serial overlays atomically in memory, or raise.

    On cancellation or any renderer/progress/helper error there is no final or
    partial result. A retry starts over from the original source. Tile seeds
    are fresh ``seeds.wrap_seed(image_seed + ordinal)`` values; ``image_seed``
    is the first tile seed (12346 reproduces the experiment's 12346..12349).
    """
    if not callable(renderer):
        raise ValueError("the private prototype requires an injected tile renderer")
    if should_cancel is not None and not callable(should_cancel):
        raise ValueError("should_cancel must be callable or None")
    if on_progress is not None and not callable(on_progress):
        raise ValueError("on_progress must be callable or None")
    if before_tile is not None and not callable(before_tile):
        raise ValueError("before_tile must be callable or None")
    plan = build_plan(source, image_seed, recipe=recipe)
    # Honour the config-before-Diffusers rule even if the codec dependencies
    # cease to import config first. The overlay helper's framework import is lazy.
    from . import config  # noqa: F401

    cancelled = False

    def poll() -> None:
        nonlocal cancelled
        if cancelled or (should_cancel is not None and should_cancel()):
            cancelled = True
            raise Cancelled()

    working: Image.Image | None = None
    results: list[TileResult] = []
    try:
        poll()
        working = source.copy()
        for tile in plan.tiles:
            poll()
            if before_tile is not None:
                before_tile()
            owned: list[Image.Image] = []
            try:
                painted = tile_mask(tile.ordinal)
                owned.append(painted)
                crop = mask_crop.region(painted, recipe.crop)
                if crop.box != tile.crop_box or crop.working != tile.working or mask_hash(painted) != tile.mask_hash:
                    raise ValueError("private prototype geometry changed after planning")
                source_hash = _rgb_hash(working)
                inputs = mask_crop.crop_inputs(working, painted, crop, block=recipe.latent_block)
                owned.extend(inputs)
                callbacks = 0
                callback_error: BaseException | None = None

                def on_step(completed: int, total: int, *, ordinal: int = tile.ordinal) -> None:
                    nonlocal callbacks, callback_error
                    if callback_error is not None:
                        raise callback_error
                    try:
                        poll()
                        if (
                            type(completed) is not int or type(total) is not int
                            or total != _DENOISE_STEPS or completed != callbacks + 1 or completed > total
                        ):
                            raise ValueError("the frozen tile renderer must report steps 1..9 with total 9")
                        callbacks += 1
                        if on_progress is not None:
                            on_progress(ordinal, completed, total)
                        poll()
                    except BaseException as exc:
                        callback_error = exc
                        raise

                call = TileCall(tile, recipe, *inputs, on_step)
                poll()
                raw: object = renderer(call)
                # A malformed ordinary list/tuple of returned images transfers
                # its fresh members too, so rejecting output count does not leak
                # them. Never adopt the borrowed root or current composite.
                if isinstance(raw, (list, tuple)):
                    owned.extend(
                        image for image in raw
                        if isinstance(image, Image.Image) and image is not source and image is not working
                    )
                # Reject borrowed aliases before adopting an output, particularly
                # the caller's original source even when its dimensions are wrong.
                if any(raw is image for image in (source, working, *owned)):
                    raise ValueError("the tile renderer must return a fresh owned image, not an input alias")
                if not isinstance(raw, Image.Image):
                    raise ValueError("the tile renderer must return exactly one RGB image")
                owned.append(raw)
                poll()
                if callback_error is not None:
                    raise callback_error
                if raw.mode != "RGB" or raw.size != tile.working:
                    raise ValueError("the tile renderer must return one 1040x1040 RGB image")
                if callbacks != _DENOISE_STEPS:
                    raise ValueError("the frozen tile renderer did not report all nine steps")
                pasted = mask_crop.paste_back(working, raw, crop.box)
                owned.append(pasted)
                composite = refine.composite(painted, working, pasted)
                owned.append(composite)
                poll()
                results.append(TileResult(tile, raw.size, callbacks, source_hash, _rgb_hash(composite)))
                working.close()
                working = composite
                owned.pop()  # The current composite survives until the next tile.
            finally:
                _close_owned(owned, source)
        poll()
        result = PrototypeResult(working, plan, tuple(results))
        working = None  # Ownership transfers only after every tile and final poll.
        return result
    finally:
        if working is not None:
            working.close()
