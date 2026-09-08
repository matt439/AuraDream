"""The size presets: five aspect ratios across four megapixel tiers.

One table, read by the CLI (`--aspect/--mp`) and by the settings that bound a
request (`max_side`, `max_pixels`), and asserted equal to the browser's copy in
`web/app.js` and the README's by `tests/test_profiles.py`. It used to live in
four places by hand; `completed_cells` keys on width and height, so a preset
that differed between two of them made `--resume` unable to continue a browser
grid.

Every side is a multiple of 16 (the VAE downsamples by 16) and every entry lands
within a few percent of its tier; 2 and 4 MP hit their ratios exactly. The whole
4 MP row is within 0.3% of 2048x2048 in pixel count, so it costs the same VRAM
whichever framing is picked - if the old 2K preset ran, all of these run.

The 1.5 MP row (1.54-1.63 MP, the same spread as the 1 MP row's 1.04-1.08) is
the row the model cards asked for: its four non-square entries are Qwen-Image's
own README shapes, verbatim, and every entry sits inside the 1024-1280 px
(1-1.64 MP) FLUX.1 Krea [dev]'s README recommends - the 2 MP row is past both.
Qwen's square is 1328x1328 (1.76 MP), too far from the tier to sit in it;
1248x1248 is the tier's own.

Imports nothing from the package: `generate.py` reads it before `config` is
imported, and `config` reads it to size its guard rails.
"""
from __future__ import annotations

SIZES: dict[str, dict[float, tuple[int, int]]] = {
    "1:1": {1: (1024, 1024), 1.5: (1248, 1248), 2: (1440, 1440), 4: (2048, 2048)},
    "3:4": {1: (896, 1200), 1.5: (1104, 1472), 2: (1248, 1664), 4: (1776, 2368)},
    "4:3": {1: (1200, 896), 1.5: (1472, 1104), 2: (1664, 1248), 4: (2368, 1776)},
    "16:9": {1: (1360, 768), 1.5: (1664, 928), 2: (1920, 1088), 4: (2736, 1536)},
    "9:16": {1: (768, 1360), 1.5: (928, 1664), 2: (1088, 1920), 4: (1536, 2736)},
}
TIERS: tuple[float, ...] = (1, 1.5, 2, 4)


def every_size() -> list[tuple[int, int]]:
    return [size for tiers in SIZES.values() for size in tiers.values()]


def largest_side() -> int:
    """The longest side any preset reaches - 2736, at 16:9 and 9:16 in 4 MP."""
    return max(max(size) for size in every_size())


def largest_pixels() -> int:
    """The most pixels any preset asks for - 1776x2368 = 4,205,568 at 3:4."""
    return max(width * height for width, height in every_size())
