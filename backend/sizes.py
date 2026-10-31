"""The size presets: thirteen aspect ratios across four megapixel tiers.

One table, read by the CLI (`--aspect/--mp`) and by the settings that bound a
request (`max_side`, `max_pixels`), and asserted equal to the browser's copy in
`web/state.js` and the README's by `tests/test_profiles.py`. It used to live in
four places by hand; `completed_cells` keys on width and height, so a preset
that differed between two of them made `--resume` unable to continue a browser
grid.

Every side is a multiple of 16 (the VAE downsamples by 16) and every entry lands
within a few percent of its tier. Where an exact ratio conflicts with the tier,
pixel count wins with less than 1% framing error. The whole 4 MP row is within
0.4% of 2048x2048 in pixel count, so it costs the same VRAM whichever framing
is picked - if the old 2K preset ran, all of these run.

3:2 and 2:3 are the ERNIE-Image and Qwen-Image model-card aspects. Animagine XL
4.0 adds 9:7, 7:4 and 12:5 plus their portrait opposites; those six exact
recommended dimensions occupy the 1 MP row. Higher tiers preserve the framings
at roughly the row's pixel count. The widest 4 MP preset is 3168x1328: 0.6% off
12:5 and only 0.04% above 2048x2048 in pixel count.

The 1.5 MP row (1.54-1.67 MP, close to the 1 MP row's 0.98-1.08) is
the row the model cards asked for: its four non-square entries are Qwen-Image's
own README shapes, verbatim, and every entry sits inside the 1024-1280 px
(1-1.64 MP) FLUX.1 Krea [dev]'s README recommends - the 2 MP row is past both.
Qwen's square is 1328x1328 (1.76 MP), too far from the tier to sit in it;
1248x1248 is the tier's own.

The rows are in the order the preset strip shows them: square, then each
portrait framing followed by its landscape inverse, pairs from nearest-square
to widest. The strip draws two adjacent inverses as one segmented box, so
the pairing is the table's order and nothing else. `Object.keys` in the
browser and `--aspect`'s choices in the CLI both take the dict's order, so it
is asserted along with the values.

Imports nothing from the package: `generate.py` reads it before `config` is
imported, and `config` reads it to size its guard rails.
"""
from __future__ import annotations

import math

SIZES: dict[str, dict[float, tuple[int, int]]] = {
    "1:1": {1: (1024, 1024), 1.5: (1248, 1248), 2: (1440, 1440), 4: (2048, 2048)},
    "7:9": {1: (896, 1152), 1.5: (1120, 1440), 2: (1264, 1632), 4: (1808, 2320)},
    "9:7": {1: (1152, 896), 1.5: (1440, 1120), 2: (1632, 1264), 4: (2320, 1808)},
    "3:4": {1: (896, 1200), 1.5: (1104, 1472), 2: (1248, 1664), 4: (1776, 2368)},
    "4:3": {1: (1200, 896), 1.5: (1472, 1104), 2: (1664, 1248), 4: (2368, 1776)},
    "2:3": {1: (848, 1264), 1.5: (1056, 1584), 2: (1184, 1776), 4: (1680, 2496)},
    "3:2": {1: (1264, 848), 1.5: (1584, 1056), 2: (1776, 1184), 4: (2496, 1680)},
    "4:7": {1: (768, 1344), 1.5: (960, 1680), 2: (1088, 1904), 4: (1552, 2704)},
    "7:4": {1: (1344, 768), 1.5: (1680, 960), 2: (1904, 1088), 4: (2704, 1552)},
    "9:16": {1: (768, 1360), 1.5: (928, 1664), 2: (1088, 1920), 4: (1536, 2736)},
    "16:9": {1: (1360, 768), 1.5: (1664, 928), 2: (1920, 1088), 4: (2736, 1536)},
    "5:12": {1: (640, 1536), 1.5: (800, 1920), 2: (928, 2224), 4: (1328, 3168)},
    "12:5": {1: (1536, 640), 1.5: (1920, 800), 2: (2224, 928), 4: (3168, 1328)},
}
TIERS: tuple[float, ...] = (1, 1.5, 2, 4)

# How far, as a natural-log ratio of width to height, a size may sit from an
# aspect's own ratio and still be that aspect at another tier (Re-render at a
# tier, `nearest_preset`). The table's own entries sit at most 0.96% from
# theirs (2:3 and 3:2 at 4 MP); this is twice that, rounded up, and a
# hand-typed 1000x700 (4.9% off 3:2, 6.9% off 4:3) is still none of them.
ASPECT_TOLERANCE = 0.02


def aspect_ratio(aspect: str) -> float:
    """The ratio an aspect's name says - 16/9 for "16:9" - not any entry's."""
    wide, high = aspect.split(":")
    return int(wide) / int(high)


def nearest_preset(width: int, height: int, tier: float) -> tuple[int, int] | None:
    """The `tier` entry of the aspect whose ratio `width`x`height` is nearest,
    or None where none is within `ASPECT_TOLERANCE` - a size is never guessed.

    Measured against each aspect's own ratio rather than its entries at that
    tier: rounded to the 16 grid, 9:16 at 1.5 MP (928x1664) sits farther from
    9:16 at 1 MP (768x1360) than 4:7 at 1.5 MP does, so entry to entry would
    send a 1 MP 9:16 to 4:7. The names are exact, and each entry is within
    1% of its own.
    """
    if tier not in TIERS:
        raise ValueError(f"{tier} MP is not a tier; the tiers are {', '.join(f'{t:g}' for t in TIERS)} MP")
    ratio = math.log(width / height)
    aspect = min(SIZES, key=lambda name: abs(math.log(aspect_ratio(name)) - ratio))
    if abs(math.log(aspect_ratio(aspect)) - ratio) > ASPECT_TOLERANCE:
        return None
    return SIZES[aspect][tier]


def every_size() -> list[tuple[int, int]]:
    return [size for tiers in SIZES.values() for size in tiers.values()]


def largest_side() -> int:
    """The longest side any preset reaches - 3168, at 12:5 and 5:12 in 4 MP."""
    return max(max(size) for size in every_size())


def largest_pixels() -> int:
    """The most pixels any preset asks for - 3168x1328 = 4,207,104 at 12:5."""
    return max(width * height for width, height in every_size())


def largest_tier_pixels(tier: float) -> int:
    """The most pixels any aspect asks for at one megapixel tier."""
    return max(tiers[tier][0] * tiers[tier][1] for tiers in SIZES.values())


def refine_largest_side() -> int:
    """The longest side a refine pass reaches: any 2 MP preset at 2x - 4448, at 12:5."""
    return 2 * max(max(tiers[2]) for tiers in SIZES.values())


def refine_largest_pixels() -> int:
    """The most pixels a refine pass asks for: any 2 MP preset at 2x - 1776x1184 x 4."""
    return 4 * largest_tier_pixels(2)
