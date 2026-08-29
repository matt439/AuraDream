"""Every seed a client can read back has to reproduce its own image."""
from __future__ import annotations

import json
import random

from backend.schemas import GenerateRequest
from backend.seeds import MAX_SEED, RANDOM_SEED_MAX, random_seed, wrap_seed


def _limit(field: str, kind: str) -> int:
    """Pull an `le`/`ge` bound off a pydantic field's metadata."""
    for constraint in GenerateRequest.model_fields[field].metadata:
        value = getattr(constraint, kind, None)
        if value is not None:
            return value
    raise AssertionError(f"{field} has no {kind} bound")


def test_drawn_seeds_survive_a_json_round_trip():
    # `res.json()` in the browser parses every number as an IEEE double, and the
    # UI's Reuse posts back what it read. A drawn seed that does not survive
    # that trip generates a different image than the card it was reused from.
    random.seed(0)
    for _ in range(2000):
        seed = random_seed()
        assert json.loads(json.dumps(seed)) == seed
        assert int(float(seed)) == seed


def test_draw_leaves_room_for_the_largest_sweep():
    # The last image of a 64x8 sweep is base + 511, and it has to be exact too.
    span = (_limit("runs", "le") - 1) * _limit("num_images", "le") + _limit("num_images", "le") - 1
    assert RANDOM_SEED_MAX + span <= 2**53 - 1
    assert int(float(RANDOM_SEED_MAX + span)) == RANDOM_SEED_MAX + span


def test_wrap_folds_past_the_column_bound():
    # Wrapping is what keeps the seed handed to torch and the seed written to
    # SQLite the same number; the column is a signed 64-bit INTEGER.
    assert wrap_seed(MAX_SEED) == MAX_SEED
    assert wrap_seed(MAX_SEED + 1) == 0
    assert wrap_seed(MAX_SEED + 7) == 6
    assert wrap_seed(0) == 0
    assert _limit("seed", "le") == MAX_SEED


def test_hand_entered_seeds_are_not_lowered():
    # Lowering the wrap modulus with the draw would make a hand-entered 2**60
    # wrap to something else entirely.
    assert wrap_seed(2**60) == 2**60
    assert MAX_SEED > RANDOM_SEED_MAX
