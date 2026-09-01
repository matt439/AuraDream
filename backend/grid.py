"""Grid search: axis specs, and the expansion into the cells a job runs.

Shared by the server (`POST /api/grid`) and the CLI (`--grid-*`), because the
one thing that cannot drift between them is what `6-12x2` means and what order
the cells come out in. The run loop is still written twice - it is welded to a
`Job` on one side and to a terminal on the other - but the product is not.

An axis is a comma- or space-separated list of values and ranges:

    8               a single value
    6,8,10          a list
    6-12x2          6 8 10 12 - start-stop, step after the x
    none,1.15       mu only: `none` is "let the pipeline derive its own shift"

An empty axis is not an empty grid: it means "the value from the panel", so a
grid with no axes set is exactly the image Generate would have made.

Seeds are the outermost loop. A grid that gets cancelled - or that runs out of
night - has then finished every parameter combination for the seeds it reached,
which is the comparison you wanted; the alternative leaves every seed half done.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .config import settings
from .schemas import GenerateRequest
from .seeds import random_seed, wrap_seed

# Per axis, and then over the whole grid. Both are guard rails against a typo
# turning into a week of GPU time - `1-1000` is eight characters.
MAX_AXIS_VALUES = 64

# mu is the only axis where "unset" is a distinct, meaningful value.
NONE_TOKENS = {"none", "null", "off", "-", "_"}

_RANGE = re.compile(
    r"^(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<stop>\d+(?:\.\d+)?)"
    r"(?:\s*[x*/]\s*(?P<step>\d+(?:\.\d+)?))?$"
)


@dataclass(frozen=True)
class Cell:
    """One pipeline call: a fully resolved request, its base seed, a caption."""

    request: GenerateRequest
    seed: int
    label: str | None = None


def _round(value: float) -> float:
    """Kill float drift so 0.1 steps do not produce 0.30000000000000004."""
    return round(value, 6)


def _range_values(match: re.Match[str], cast: Callable[[str], Any], label: str) -> list[Any]:
    start, stop = cast(match["start"]), cast(match["stop"])
    step = cast(match["step"]) if match["step"] else cast("1")
    if step <= 0:
        raise ValueError(f"{label}: the step in {match.string!r} must be above zero")
    if stop < start:
        raise ValueError(f"{label}: {match.string!r} runs backwards")
    # The epsilon is what puts the endpoint of 0-1x0.1 in the list rather than
    # a hair outside it.
    count = int((stop - start) / step + 1e-9) + 1
    if count > MAX_AXIS_VALUES:
        raise ValueError(f"{label}: {match.string!r} is {count} values, more than {MAX_AXIS_VALUES}")
    values = [start + index * step for index in range(count)]
    return [_round(value) if isinstance(value, float) else value for value in values]


def parse_axis(
    text: str | None,
    cast: Callable[[str], Any] = float,
    *,
    allow_none: bool = False,
    label: str = "axis",
) -> list[Any]:
    """Values for one axis, in the order given, duplicates dropped.

    An empty spec returns an empty list - the caller substitutes its own base
    value, which is what makes an untouched axis mean "leave this alone".
    """
    values: list[Any] = []
    for token in re.split(r"[,;\s]+", (text or "").strip()):
        if not token:
            continue
        if token.lower() in NONE_TOKENS:
            if not allow_none:
                raise ValueError(f"{label}: {token!r} is only meaningful for mu")
            values.append(None)
            continue
        match = _RANGE.match(token)
        if match is not None:
            values.extend(_range_values(match, cast, label))
            continue
        try:
            values.append(cast(token))
        except ValueError:
            raise ValueError(f"{label}: {token!r} is not a number") from None

    # A repeated value is a night spent rendering the same image twice.
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if len(unique) > MAX_AXIS_VALUES:
        raise ValueError(f"{label}: {len(unique)} values, more than the {MAX_AXIS_VALUES} allowed")
    return unique


def axis_values(request: GenerateRequest, steps: str, cfg: str, mu: str) -> dict[str, list[Any]]:
    """The three axes resolved against the request they sit on top of."""
    return {
        "steps": parse_axis(steps, int, label="steps") or [request.steps],
        "guidance_scale": parse_axis(cfg, float, label="CFG") or [request.guidance_scale],
        "mu": parse_axis(mu, float, allow_none=True, label="mu") or [request.mu],
    }


def describe(values: dict[str, list[Any]], seeds: int) -> str:
    """`4 steps x 3 CFG x 1 mu x 2 seeds`, for a preview line or a log header."""
    parts = [f"{len(values['steps'])} steps", f"{len(values['guidance_scale'])} CFG"]
    parts.append(f"{len(values['mu'])} mu")
    parts.append(f"{seeds} seed{'' if seeds == 1 else 's'}")
    return " x ".join(parts)


def _label(steps: int, cfg: float, mu: float | None) -> str:
    """ASCII on purpose: this ends up in an overnight log, and a redirected
    stdout on Windows encodes with the locale codec, where a stray en dash is
    a UnicodeEncodeError twelve hours into the run."""
    return f"{steps} steps, CFG {cfg:g}, mu {'off' if mu is None else f'{mu:g}'}"


def expand(
    request: GenerateRequest,
    *,
    steps: str = "",
    cfg: str = "",
    mu: str = "",
    seeds: int = 1,
) -> list[Cell]:
    """The cartesian product, as cells ready to hand to a job.

    Every cell is a real `GenerateRequest` rather than a patch to apply later,
    so an out-of-range axis value fails here - with the schema's own message -
    instead of somewhere inside the denoising loop at 3am.
    """
    values = axis_values(request, steps, cfg, mu)
    total = len(values["steps"]) * len(values["guidance_scale"]) * len(values["mu"]) * seeds
    if total * request.num_images > settings.max_grid_images:
        raise ValueError(
            f"{total * request.num_images} images is past the {settings.max_grid_images} "
            "a grid may queue; narrow an axis or drop the batch size"
        )

    base = request.seed if request.seed is not None else random_seed()
    template = request.model_dump()
    cells: list[Cell] = []
    for index in range(seeds):
        # Step past the whole batch, exactly as a sweep does, so no two images
        # in the grid are drawn from the same seed. Every parameter cell of one
        # seed shares it - comparing settings on the same noise is the point.
        seed = wrap_seed(base + index * request.num_images)
        for step_count in values["steps"]:
            for scale in values["guidance_scale"]:
                for shift in values["mu"]:
                    cells.append(
                        Cell(
                            request=GenerateRequest(
                                **{
                                    **template,
                                    "steps": step_count,
                                    "guidance_scale": scale,
                                    "mu": shift,
                                    "seed": seed,
                                    "runs": 1,
                                }
                            ),
                            seed=seed,
                            label=_label(step_count, scale, shift),
                        )
                    )
    return cells
