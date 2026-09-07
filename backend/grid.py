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

The prompt is an axis too, but a numeric list is no use to it. It varies through
named placeholders instead - `{size} dog sitting on {ground}` over a block of

    size = big | small
    ground = grass | dirt

- one variable per line, values split on `|` because a comma is part of a prompt
far too often to be a separator. The names cross with each other and with the
sampling axes, so that block is four cells, not two.

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

# Where "unset" is a distinct, meaningful value: mu, where it means the shift
# the pipeline derives for itself, and a prompt variable, where it means the
# word is simply not there - `size = big | small | none` asks what the picture
# looks like when nothing is said about the size at all.
NONE_TOKENS = {"none", "null", "off", "-", "_"}

_RANGE = re.compile(
    r"^(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<stop>\d+(?:\.\d+)?)"
    r"(?:\s*[x*/]\s*(?P<step>\d+(?:\.\d+)?))?$"
)

# `{size}` in a prompt. Braces because the user's own example - a bare `x` and
# `y` - would substitute inside `box` and `sky` too, and a prompt is prose.
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# One variable per line: `size = big | small`. Split on the first `=` only, so a
# value may contain one.
_VARIABLE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<values>.*)$")

# What identifies an image for resume: the prompt pair, the settings, the seed.
CellKey = tuple[str, str | None, int, float, float | None, int]


@dataclass(frozen=True)
class Cell:
    """One pipeline call: a fully resolved request, its base seed, a caption."""

    request: GenerateRequest
    seed: int
    label: str | None = None
    # What the prompt placeholders were set to, for the caption and nothing
    # else: the resolved prompt on the request is what actually runs.
    variables: dict[str, str] | None = None


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


def parse_variables(text: str | None, *, label: str = "prompt") -> dict[str, list[str]]:
    """`size = big | small` lines into `{"size": ["big", "small"]}`.

    Values split on `|`, not on the comma the numeric axes use: `a big, fluffy
    dog` is an ordinary prompt fragment and would otherwise become two cells.
    Declaration order is kept, because it is the order the grid renders in and
    the order the labels read in.
    """
    variables: dict[str, list[str]] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _VARIABLE.match(line)
        if match is None:
            raise ValueError(f"{label}: {line!r} is not `name = one | another`")
        name = match["name"]
        if name in variables:
            raise ValueError(f"{label}: {name!r} is given twice")
        values: list[str] = []
        for token in match["values"].split("|"):
            value = " ".join(token.split())
            if value.lower() in NONE_TOKENS:
                # The deliberately empty value: the placeholder and the space
                # around it come out of the prompt entirely.
                value = ""
            elif not value:
                # A stray `|`, which is a typo rather than a cell.
                continue
            # A repeat is a night spent rendering the same image twice.
            if value not in values:
                values.append(value)
        if not values:
            raise ValueError(f"{label}: {name!r} has no values")
        if len(values) > MAX_AXIS_VALUES:
            raise ValueError(
                f"{label}: {name!r} is {len(values)} values, more than {MAX_AXIS_VALUES}"
            )
        variables[name] = values
    return variables


def _placeholders(request: GenerateRequest) -> set[str]:
    return set(PLACEHOLDER.findall(request.prompt)) | set(
        PLACEHOLDER.findall(request.negative_prompt or "")
    )


def substitute(text: str | None, assignment: dict[str, str]) -> str | None:
    """Fill this cell's values into a prompt. Unknown names are left alone."""
    if text is None or not assignment:
        return text
    filled = PLACEHOLDER.sub(lambda m: assignment.get(m.group(1), m.group(0)), text)
    if any(value == "" for value in assignment.values()):
        # A dropped value must not leave the hole it came out of: `{size} dog`
        # with no size is `dog`, not ` dog`, and a doubled space is a token the
        # encoder can do without. Only horizontal runs collapse, so a prompt
        # written over several lines keeps its shape. Punctuation that belongs
        # to the value belongs *inside* it - `mood = , at dusk | none`.
        filled = re.sub(r"[ \t]{2,}", " ", filled)
        filled = "\n".join(line.strip() for line in filled.split("\n"))
    return filled


def prompt_assignments(
    request: GenerateRequest, variables: dict[str, list[str]]
) -> list[dict[str, str]]:
    """One assignment per prompt cell, the last variable varying fastest.

    Both halves of a mismatch are errors here rather than something to notice in
    the morning: a variable with no placeholder multiplies the night by a value
    that changes nothing, and a placeholder with no variable renders the literal
    `{ground}` into every image of the run.
    """
    if not variables:
        return [{}]
    used = _placeholders(request)
    for name in variables:
        if name not in used:
            raise ValueError(f"prompt: nothing in the prompt says {{{name}}}")
    for name in sorted(used - set(variables)):
        raise ValueError(f"prompt: {{{name}}} has no values")

    assignments: list[dict[str, str]] = [{}]
    for name, values in variables.items():
        assignments = [{**one, name: value} for one in assignments for value in values]
    return assignments


def axis_values(
    request: GenerateRequest, steps: str, cfg: str, mu: str, prompt: str = ""
) -> dict[str, list[Any]]:
    """The axes resolved against the request they sit on top of.

    The prompt axis resolves to finished prompts rather than to the variables
    behind them: what the preview has to show - and what the contact sheet later
    groups by - is the text that will actually be encoded.
    """
    assignments = prompt_assignments(request, parse_variables(prompt))
    return {
        "prompt": [substitute(request.prompt, one) for one in assignments],
        "steps": parse_axis(steps, int, label="steps") or [request.steps],
        "guidance_scale": parse_axis(cfg, float, label="CFG") or [request.guidance_scale],
        "mu": parse_axis(mu, float, allow_none=True, label="mu") or [request.mu],
    }


def describe(values: dict[str, list[Any]], seeds: int) -> str:
    """`2 prompts x 4 steps x 3 CFG x 1 mu x 2 seeds`, for a preview or a log.

    The prompt count appears only when there is more than one: a sampling grid
    should read exactly as it did before the prompt became an axis.
    """
    parts = []
    if len(values.get("prompt", [])) > 1:
        parts.append(f"{len(values['prompt'])} prompts")
    parts.append(f"{len(values['steps'])} steps")
    parts.append(f"{len(values['guidance_scale'])} CFG")
    parts.append(f"{len(values['mu'])} mu")
    parts.append(f"{seeds} seed{'' if seeds == 1 else 's'}")
    return " x ".join(parts)


def _label(assignment: dict[str, str], steps: int, cfg: float, mu: float | None) -> str:
    """ASCII on purpose: this ends up in an overnight log, and a redirected
    stdout on Windows encodes with the locale codec, where a stray en dash is
    a UnicodeEncodeError twelve hours into the run. Prompt values come from a
    text box, so they are replaced rather than trusted.

    The variables lead: in a prompt grid they are what changed.
    """
    parts = [f"{name}={value or 'none'}" for name, value in assignment.items()]
    parts.append(f"{steps} steps")
    parts.append(f"CFG {cfg:g}")
    parts.append(f"mu {'off' if mu is None else f'{mu:g}'}")
    return ", ".join(parts).encode("ascii", "replace").decode("ascii")


def cell_key(cell: "Cell") -> CellKey:
    """What identifies a cell in the gallery: its prompt, its settings, its seed.

    The prompt pair is part of it because a grid can vary the prompt, and two
    cells that agree on every number are still different images when it does. An
    absent negative prompt reaches the database as NULL down one path and as ''
    down another, so both sides of the comparison normalise it to None.

    Rounded to the six decimals `parse_axis` produces, so a value that has been
    through SQLite's REAL and back still matches the axis that generated it.
    """
    request = cell.request
    return (
        request.prompt,
        request.negative_prompt or None,
        request.steps,
        round(request.guidance_scale, 6),
        None if request.mu is None else round(request.mu, 6),
        cell.seed,
    )


def prune_done(cells: list[Cell], done: set[CellKey]) -> list[Cell]:
    """Drop the cells already sitting in the gallery, keeping the order.

    Order matters as much here as in `expand`: a resumed grid picks up where the
    night stopped, so what it does first is still the earliest seed it is short.
    """
    return [cell for cell in cells if cell_key(cell) not in done]


def prompts_of(cells: list[Cell]) -> list[str]:
    """The distinct prompts a grid would render, for the resume lookup."""
    return list(dict.fromkeys(cell.request.prompt for cell in cells))


def expand(
    request: GenerateRequest,
    *,
    prompt: str = "",
    steps: str = "",
    cfg: str = "",
    mu: str = "",
    seeds: int = 1,
) -> list[Cell]:
    """The cartesian product, as cells ready to hand to a job.

    Every cell is a real `GenerateRequest` rather than a patch to apply later,
    so an out-of-range axis value fails here - with the schema's own message -
    instead of somewhere inside the denoising loop at 3am. A prompt that grew
    past the schema's 4000 characters once its values were substituted in is
    caught by that same rule.
    """
    assignments = prompt_assignments(request, parse_variables(prompt))
    values = axis_values(request, steps, cfg, mu)
    total = (
        len(assignments)
        * len(values["steps"])
        * len(values["guidance_scale"])
        * len(values["mu"])
        * seeds
    )
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
        # The prompt sits just inside the seed, for the reason the seed is
        # outermost: a night that ends early has then swept every setting for
        # the prompts it reached, which is a comparison, rather than leaving
        # every prompt part-swept.
        for assignment in assignments:
            for step_count in values["steps"]:
                for scale in values["guidance_scale"]:
                    for shift in values["mu"]:
                        cells.append(
                            Cell(
                                request=GenerateRequest(
                                    **{
                                        **template,
                                        "prompt": substitute(request.prompt, assignment),
                                        "negative_prompt": substitute(
                                            request.negative_prompt, assignment
                                        ),
                                        "steps": step_count,
                                        "guidance_scale": scale,
                                        "mu": shift,
                                        "seed": seed,
                                        "runs": 1,
                                    }
                                ),
                                seed=seed,
                                label=_label(assignment, step_count, scale, shift),
                                variables=assignment or None,
                            )
                        )
    return cells
