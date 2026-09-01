"""Axis parsing and the cartesian expansion behind a grid search."""
from __future__ import annotations

import pytest

from backend import grid
from backend.schemas import GenerateRequest

BASE = dict(prompt="a lighthouse in fog", width=256, height=256, steps=8, guidance_scale=0.0, mu=1.15)


def request(**overrides) -> GenerateRequest:
    return GenerateRequest(**{**BASE, **overrides})


# ------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("8", [8]),
        ("6,8,10", [6, 8, 10]),
        ("6 8 10", [6, 8, 10]),  # spaces are separators too, for pasted lists
        ("6-12x2", [6, 8, 10, 12]),  # both ends included
        ("6-12", [6, 7, 8, 9, 10, 11, 12]),  # a bare range steps by one
        ("6-7,20", [6, 7, 20]),  # ranges and values mix
        ("8,8,6", [8, 6]),  # a repeat is a night spent rendering it twice
        ("", []),  # empty is "leave the axis alone", not "no values"
    ],
)
def test_integer_axes(spec, expected):
    assert grid.parse_axis(spec, int, label="steps") == expected


def test_float_ranges_do_not_drift():
    # 0.1 steps accumulated naively give 0.30000000000000004, which then shows
    # up in the preview, in the record and in the filename.
    assert grid.parse_axis("0-1x0.1") == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_mu_takes_none_and_nothing_else_does():
    assert grid.parse_axis("none,1.15", allow_none=True, label="mu") == [None, 1.15]
    with pytest.raises(ValueError, match="only meaningful for mu"):
        grid.parse_axis("none", label="CFG")


@pytest.mark.parametrize("spec", ["12-6", "6-12x0", "6-12x-2", "eight", "6..12"])
def test_a_bad_axis_says_so(spec):
    with pytest.raises(ValueError):
        grid.parse_axis(spec, int, label="steps")


def test_an_axis_cannot_be_a_thousand_values():
    with pytest.raises(ValueError, match="more than"):
        grid.parse_axis("1-1000", int, label="steps")


# ----------------------------------------------------------------- expansion


def test_an_empty_grid_is_the_request_it_sits_on():
    cells = grid.expand(request(seed=42))
    assert len(cells) == 1
    assert (cells[0].request.steps, cells[0].request.guidance_scale, cells[0].request.mu) == (
        8,
        0.0,
        1.15,
    )
    assert cells[0].seed == 42


def test_seeds_are_the_outer_loop():
    # A grid that runs out of night has then finished every setting for the
    # seeds it reached, rather than a slice of each.
    cells = grid.expand(request(seed=42), steps="6,8", seeds=2)
    assert [(c.seed, c.request.steps) for c in cells] == [(42, 6), (42, 8), (43, 6), (43, 8)]


def test_the_product_covers_every_combination():
    cells = grid.expand(request(seed=1), steps="6,8", cfg="0,1", mu="none,1.15", seeds=3)
    assert len(cells) == 2 * 2 * 2 * 3
    combinations = {(c.request.steps, c.request.guidance_scale, c.request.mu) for c in cells}
    assert combinations == {
        (steps, cfg, mu) for steps in (6, 8) for cfg in (0.0, 1.0) for mu in (None, 1.15)
    }


def test_a_batch_steps_the_seed_past_itself():
    # The same rule a sweep follows: image i of a batch is seed+i, so the next
    # cell has to start past the whole of the previous one or two images in the
    # grid share a seed and the gallery cannot reproduce either.
    cells = grid.expand(request(seed=100, num_images=4), seeds=3)
    assert [c.seed for c in cells] == [100, 104, 108]


def test_every_cell_is_a_single_run():
    # `runs` would multiply the night behind the count the preview promised.
    cells = grid.expand(request(seed=1, runs=5), steps="6,8")
    assert {c.request.runs for c in cells} == {1}


def test_a_cell_outside_the_schema_is_refused_up_front():
    # Not at 3am, sixty images into the sweep.
    with pytest.raises(ValueError):
        grid.expand(request(seed=1), steps="90-120x10")


def test_the_whole_grid_has_a_ceiling():
    with pytest.raises(ValueError, match="past the"):
        grid.expand(request(seed=1), steps="1-64", cfg="0-16", seeds=8)


def test_labels_stay_ascii():
    # They go into an overnight log, and a redirected stdout on Windows encodes
    # with the locale codec.
    for cell in grid.expand(request(seed=1), steps="6,8", mu="none,1.15"):
        assert cell.label is not None
        cell.label.encode("ascii")
