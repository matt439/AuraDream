"""The size presets: thirteen aspect ratios across four megapixel tiers.

One table, read by the CLI (`--aspect/--mp`) and by the settings that bound a
request (`max_side`, `max_pixels`), and asserted equal to the browser's copy in
`web/app.js` and the README's by `tests/test_profiles.py`. It used to live in
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

The rows are in the order the preset strip shows them: square, then landscape
from nearest-square to widest, then portrait the same way, so each half reads
monotonically and an aspect's inverse sits at the same position in the other
half. `Object.keys` in the browser and `--aspect`'s choices in the CLI both
take the dict's order, so it is asserted along with the values.

Imports nothing from the package: `generate.py` reads it before `config` is
imported, and `config` reads it to size its guard rails.
"""
from __future__ import annotations

SIZES: dict[str, dict[float, tuple[int, int]]] = {
    "1:1": {1: (1024, 1024), 1.5: (1248, 1248), 2: (1440, 1440), 4: (2048, 2048)},
    "9:7": {1: (1152, 896), 1.5: (1440, 1120), 2: (1632, 1264), 4: (2320, 1808)},
    "4:3": {1: (1200, 896), 1.5: (1472, 1104), 2: (1664, 1248), 4: (2368, 1776)},
    "3:2": {1: (1264, 848), 1.5: (1584, 1056), 2: (1776, 1184), 4: (2496, 1680)},
    "7:4": {1: (1344, 768), 1.5: (1680, 960), 2: (1904, 1088), 4: (2704, 1552)},
    "16:9": {1: (1360, 768), 1.5: (1664, 928), 2: (1920, 1088), 4: (2736, 1536)},
    "12:5": {1: (1536, 640), 1.5: (1920, 800), 2: (2224, 928), 4: (3168, 1328)},
    "7:9": {1: (896, 1152), 1.5: (1120, 1440), 2: (1264, 1632), 4: (1808, 2320)},
    "3:4": {1: (896, 1200), 1.5: (1104, 1472), 2: (1248, 1664), 4: (1776, 2368)},
    "2:3": {1: (848, 1264), 1.5: (1056, 1584), 2: (1184, 1776), 4: (1680, 2496)},
    "4:7": {1: (768, 1344), 1.5: (960, 1680), 2: (1088, 1904), 4: (1552, 2704)},
    "9:16": {1: (768, 1360), 1.5: (928, 1664), 2: (1088, 1920), 4: (1536, 2736)},
    "5:12": {1: (640, 1536), 1.5: (800, 1920), 2: (928, 2224), 4: (1328, 3168)},
}
TIERS: tuple[float, ...] = (1, 1.5, 2, 4)


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
