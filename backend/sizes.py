"""The size presets: five aspect ratios across three megapixel tiers.

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

Imports nothing from the package: `generate.py` reads it before `config` is
imported, and `config` reads it to size its guard rails.
"""
from __future__ import annotations

SIZES: dict[str, dict[int, tuple[int, int]]] = {
    "1:1": {1: (1024, 1024), 2: (1440, 1440), 4: (2048, 2048)},
    "3:4": {1: (896, 1200), 2: (1248, 1664), 4: (1776, 2368)},
    "4:3": {1: (1200, 896), 2: (1664, 1248), 4: (2368, 1776)},
    "16:9": {1: (1360, 768), 2: (1920, 1088), 4: (2736, 1536)},
    "9:16": {1: (768, 1360), 2: (1088, 1920), 4: (1536, 2736)},
}
TIERS: tuple[int, ...] = (1, 2, 4)


def every_size() -> list[tuple[int, int]]:
    return [size for tiers in SIZES.values() for size in tiers.values()]


def largest_side() -> int:
    """The longest side any preset reaches - 2736, at 16:9 and 9:16 in 4 MP."""
    return max(max(size) for size in every_size())


def largest_pixels() -> int:
    """The most pixels any preset asks for - 1776x2368 = 4,205,568 at 3:4."""
    return max(width * height for width, height in every_size())
