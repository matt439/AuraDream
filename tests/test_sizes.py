"""The size presets as Re-render at a tier reads them: each record's aspect
mapped to its entry at another tier (`sizes.nearest_preset`)."""
from __future__ import annotations

import math

import pytest

from backend import sizes


def _off(width: int, height: int, aspect: str) -> float:
    return abs(math.log(width / height) - math.log(sizes.aspect_ratio(aspect)))


def test_every_preset_maps_to_its_own_aspect_at_every_tier():
    # Itself at its own tier, its aspect's entry at the others - 9:16 at 1 MP
    # included, which entry to entry is nearer 4:7's 1.5 MP entry.
    for aspect, tiers in sizes.SIZES.items():
        for width, height in tiers.values():
            for tier in sizes.TIERS:
                assert sizes.nearest_preset(width, height, tier) == tiers[tier], (aspect, width, height, tier)


def test_the_tolerance_covers_the_tables_own_rounding_and_no_more_than_twice_it():
    worst = max(_off(w, h, aspect) for aspect, tiers in sizes.SIZES.items() for w, h in tiers.values())
    assert 0.0095 < worst < sizes.ASPECT_TOLERANCE <= 2.2 * worst


@pytest.mark.parametrize("width,height,tier,expected", [
    # 1 px off a preset and a hand-typed 16-grid near one: within the tolerance.
    (1025, 1024, 2, sizes.SIZES["1:1"][2]),
    (1280, 720, 1, sizes.SIZES["16:9"][1]),
    (256, 256, 4, sizes.SIZES["1:1"][4]),
    (384, 256, 1.5, sizes.SIZES["3:2"][1.5]),
    (720, 1280, 2, sizes.SIZES["9:16"][2]),
])
def test_a_size_near_a_presets_aspect_maps_to_it(width, height, tier, expected):
    assert sizes.nearest_preset(width, height, tier) == expected


def test_a_size_between_two_aspects_is_unmatched_not_guessed():
    for tier in sizes.TIERS:
        assert sizes.nearest_preset(1000, 700, tier) is None
        assert sizes.nearest_preset(1008, 704, tier) is None
    # Past the widest preset too.
    assert sizes.nearest_preset(3200, 1024, 2) is None


def test_a_tier_that_is_none_of_the_tables_is_refused():
    with pytest.raises(ValueError, match="3 MP is not a tier"):
        sizes.nearest_preset(1024, 1024, 3)
