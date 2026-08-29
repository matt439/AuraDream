"""Seed arithmetic, shared by the server, the CLI and the backend.

It lives on its own for two reasons. `generate.py` has to reach it *before* it
imports `config`, whose frozen settings read the KREA_* variables at import
time. And the three places that touch a seed - the draw, the per-image offset
and the generator itself - have to agree exactly: a recorded seed that does not
reproduce its own image is worse than recording no seed at all.
"""
from __future__ import annotations

import random

# The `seed` column is a signed 64-bit INTEGER; anything past this overflows the
# INSERT. It is also the modulus every offset is folded back through.
MAX_SEED = 2**63 - 1

# The largest offset a single request can add to its base seed: `runs` steps the
# seed by `num_images` each time, and both are capped in GenerateRequest.
_MAX_SWEEP_SPAN = 64 * 8

# JSON numbers are IEEE doubles and the browser is a first-class client: a seed
# above 2**53 comes back from `res.json()` already rounded, so Reuse would send
# a *different* seed than the one on the card and quietly produce a different
# image. Drawn seeds therefore stay inside the exactly-representable range, with
# room left for the whole sweep the draw might be the base of. A hand-entered
# seed is still honoured as given - it is the user's number, and anyone typing
# 19 digits is not reading it back off a JSON response.
RANDOM_SEED_MAX = 2**53 - 1 - _MAX_SWEEP_SPAN


def random_seed() -> int:
    """A seed that survives a round trip through JSON."""
    return random.randint(0, RANDOM_SEED_MAX)


def wrap_seed(seed: int) -> int:
    """Fold a seed back into the storable range.

    Sweep and batch offsets are added to a base seed, so a base near the top of
    the range walks past it. Wrapping here rather than at the INSERT keeps the
    seed handed to the generator and the seed written to the database identical
    - wrapping only at storage would record 0 for an image drawn at 2**63.
    """
    return seed % (MAX_SEED + 1)
