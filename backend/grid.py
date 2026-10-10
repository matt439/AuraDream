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

A range may be written spaced out, `6 - 12 x 2`: a space beside the `-` or
the `x` between two numbers is part of the range, not a separator.

The prompt is an axis too, but a numeric list is no use to it. It varies through
named placeholders instead - `{size} dog sitting on {ground}` over a block of

    size = big | small
    ground = grass | dirt

- one variable per line, values split on `|` because a comma is part of a prompt
far too often to be a separator. The names cross with each other and with the
sampling axes, so that block is four cells, not two.

Or the prompt axis is a list of whole prompts under short labels - the prompt
analyser's original, reordered and rewritten versions of one idea, compared on
the same seeds. Those are finished text, taken verbatim, and each records its
label as the one variable `variant`, which the contact sheet lays out as rows
whenever anything else - the seeds, usually - varies to fill the columns.

An empty axis is not an empty grid: it means "the value from the panel", so a
grid with no axes set is exactly the image Generate would have made.

Seeds are the outermost loop. A grid that gets cancelled - or that runs out of
night - has then finished every parameter combination for the seeds it reached,
which is the comparison you wanted; the alternative leaves every seed half done.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .model_selection import current_model_id
from .config import settings
from .refine import start_index, steps_run
from .schemas import GenerateRequest, ImageRecord, PromptVariant
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

# A range written with spaces in it - `2 - 5`, `6-12 x2` - is closed up before
# the split on whitespace, which otherwise cut it into tokens: `-` reached the
# none check ("only meaningful for Mu") on every box but Mu, where `0.8 - 1.2`
# quietly became three cells, one of them none, and `x2` was "not a number".
# Closed up, `0.8-1.2` is a range, which with no step counts in ones and
# never gets past 0.8: `_range_values` asks for the step.
# Only between two digits, so a `-` standing alone is still Mu's none.
_SPACED_RANGE = re.compile(r"(?<=\d)\s*([-x*/])\s*(?=\d)")

# A range in someone else's syntax, `6..12` or `6 to 12`: refused as a range,
# with ours, rather than as a token that is not a number.
_OTHER_RANGE = re.compile(r"\d\s*(?:\.{2,}|to)\s*\d", re.IGNORECASE)

# `{size}` in a prompt. Braces because the user's own example - a bare `x` and
# `y` - would substitute inside `box` and `sky` too, and a prompt is prose.
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# One variable per line: `size = big | small`. Split on the first `=` only, so a
# value may contain one.
_VARIABLE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<values>.*)$")

# The variable a prompt-variant grid records each variant's label under. The
# record, the PNG chunk and the sheet already carry variables, so a label rides
# the same path with no new field; `chooseAxes` in web/sheet.js knows the name
# (`VARIANT_AXIS`) and gives it the rows, however many seeds there are.
VARIANT = "variant"

# What identifies an image for resume: the prompt pair, the settings, the seed.
# (prompt, negative prompt, steps, CFG, mu, true CFG, references, LoRA, refine,
# seed) - the seed last, because `cell_keys` swaps it for each image of a
# batch, and the four optional elements in the order they were added, because
# `run_id` names a cell by the shortest form that carries all of its non-None
# ones.
CellKey = tuple[
    str, str | None, int, float, float | None, float | None, str | None, str | None,
    str | None, int,
]

# How many of the trailing elements before the seed are optional extensions -
# true CFG, references, LoRA, refine. `run_id` drops them from the right while
# they are None, so a grid that uses none of them is named by the six-element
# key its images have carried since before any of them existed.
_KEY_EXTENSIONS = 4


def reference_key(ids: Sequence[int] | None) -> str | None:
    """The reference images as one key element: their ids, in order, or None.

    Order matters - it is the order the encoded references are concatenated in -
    so this is not a set. None and the empty list are the same thing, which is
    what lets a text-to-image row and a row whose references were dropped meet
    at the same key.
    """
    return ",".join(str(image_id) for image_id in ids) if ids else None


def lora_key(name: str | None, scale: float | None) -> str | None:
    """The adapter and the weight it ran at as one key element, or None.

    A scale with no adapter is not a setting (`applied()` puts it back to the
    default), so the two travel together or not at all. Formatted rather than
    rounded into a tuple because the same element has to be built from a
    database REAL on the resume side and match to the digit.
    """
    return None if not name else f"{name}@{round(float(scale if scale is not None else 1.0), 6):g}"


def refine_key(source: int | None, strength: float | None) -> str | None:
    """The refine source and the strength it ran at as one key element, or None.

    Built like `lora_key` and for its reasons: a strength with no source is
    not a setting, and the same element has to come out of a request here and
    out of an INTEGER and a REAL in `storage.completed_cells`, to the digit.
    """
    return None if source is None else f"{source}@{round(float(strength), 6):g}"


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
    # Without a step a range counts in ones, so `0.8-1.2` on Mu, CFG or a
    # strength is the one value 0.8: a sweep that does not sweep, which the
    # preview's totals - one value is a factor of one - would not show either.
    if match["step"] is None and count == 1 and stop > start:
        raise ValueError(
            f"{label}: {match.string!r} needs a step, e.g. {match['start']}-{match['stop']}x0.1"
        )
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
    `label` is the name of the box the spec was typed into, which every
    message leads with: the preview prints it under the whole fold.
    """
    text = (text or "").strip()
    if _OTHER_RANGE.search(text):
        raise ValueError(f"{label}: write a range as 6-12 (or 6-12x2)")
    values: list[Any] = []
    for token in re.split(r"[,;\s]+", _SPACED_RANGE.sub(r"\1", text)):
        if not token:
            continue
        if token.lower() in NONE_TOKENS:
            if not allow_none:
                raise ValueError(f"{label}: {token!r} is only meaningful for Mu")
            values.append(None)
            continue
        match = _RANGE.match(token)
        try:
            if match is not None:
                # Inside the handler too: `6-12x0.5` on an integer axis matches
                # the range regex and then fails inside int(), and the message
                # has to name the field either way.
                values.extend(_range_values(match, cast, label))
            else:
                values.append(cast(token))
        except ValueError as exc:
            if str(exc).startswith(f"{label}:"):
                raise
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


_HSPACE = (" ", "\t")
# What a word can open after with no space of its own: a bracket, a quotation
# mark, an XML element's tag.
_OPENERS = frozenset("([{<>\"“‘«")


def substitute(text: str | None, assignment: dict[str, str]) -> str | None:
    """Fill this cell's values into a prompt. Unknown names are left alone.

    A dropped value takes its space with it and nothing else (`_close_hole`).
    """
    if text is None or not assignment:
        return text
    out = ""
    pos = 0
    for match in PLACEHOLDER.finditer(text):
        value = assignment.get(match.group(1))
        if value is None:
            continue  # not this grid's: it stays in the text between the holes
        out += text[pos:match.start()]
        pos = match.end()
        if value:
            out += value
        else:
            out, pos = _close_hole(out, text, pos)
    return out + text[pos:]


def _close_hole(out: str, text: str, pos: int) -> tuple[str, int]:
    """Close the gap an empty value left between `out` and `text[pos:]`.

    `{size} dog` with no size is `dog`, not ` dog`, and `a {size} dog` is `a
    dog`: one space goes with the word - the one after it where the word
    followed a space or opened a line, a bracket or an XML element
    (`<appearance>{hair} long_hair`), else, at a line's end, the one before.
    Nothing else moves: `red_eyes, {hair}long_hair` keeps its one space.

    This used to collapse every doubled space in the prompt and strip every
    line whenever any value was empty, so on an indented prompt - XML pasted
    from a model card - the none cell also lost every line's indentation,
    some twenty Gemma tokens the other cells kept, and the comparison was no
    longer the word alone. A placeholder alone on its line leaves the line
    empty, its indentation gone with it, so the prompt keeps its line count
    (the page's grid preview lists a multi-line prompt by line number).
    Punctuation that belongs to the value belongs *inside* it - `mood = , at
    dusk | none` - so `dog, {x}, cat` keeps both commas.
    """
    line_head = out[out.rfind("\n") + 1:]
    end = text.find("\n", pos)
    line_tail = text[pos:] if end < 0 else text[pos:end]
    if not line_head.strip(" \t") and not line_tail.strip(" \t\r"):
        # Alone on its line: the line's own whitespace was only the hole's.
        return out[:len(out) - len(line_head)], pos + len(line_tail.rstrip("\r"))
    before = out[-1:] in _HSPACE
    after = text[pos:pos + 1] in _HSPACE
    if after and (before or not line_head or line_head[-1] in _OPENERS):
        return out, pos + 1
    if before and not line_tail.strip("\r"):
        return out[:-1], pos
    return out, pos


def prompt_assignments(
    request: GenerateRequest, variables: dict[str, list[str]]
) -> list[dict[str, str]]:
    """One assignment per prompt cell, the last variable varying fastest.

    Both halves of a mismatch are errors here rather than something to notice in
    the morning: a variable with no placeholder multiplies the night by a value
    that changes nothing, and a placeholder with no variable renders the literal
    `{ground}` into every image of the run - *including* when the variables box
    is empty, the easiest way to get there: the placeholders were written and
    the box forgotten, or cleared while the prompt kept them.

    The product is counted before it is built. `MAX_AXIS_VALUES` caps values per
    variable and nothing capped the number of variables, so eighteen binary
    lines built 262,144 dicts on the event loop - on every keystroke of the
    preview - before the whole-grid ceiling could refuse them.
    """
    used = _placeholders(request)
    for name in variables:
        if name not in used:
            raise ValueError(f"prompt: nothing in the prompt says {{{name}}}")
    for name in sorted(used - set(variables)):
        raise ValueError(f"prompt: {{{name}}} has no values")
    if not variables:
        return [{}]
    count = math.prod(len(values) for values in variables.values())
    if count > settings.max_grid_images:
        raise ValueError(
            f"prompt: {count} prompts is past the {settings.max_grid_images} a grid may "
            "queue; drop a variable or some of its values"
        )

    assignments: list[dict[str, str]] = [{}]
    for name, values in variables.items():
        assignments = [{**one, name: value} for one in assignments for value in values]
    return assignments


def resolve_prompts(
    request: GenerateRequest,
    prompt: str = "",
    variants: Sequence[PromptVariant] | None = None,
) -> list[tuple[dict[str, str], str | None, str | None]]:
    """The prompt axis in render order: (assignment, prompt, negative prompt) per value.

    Either the variables substituted into the request's prompt pair, or the
    variants whole. A variant is a finished prompt, so it is taken verbatim:
    nothing is substituted into it, and a `{` in a pasted JSON prompt or a `|`
    and newlines in NewBie's XML are text, not a placeholder with no values or
    a second cell. The request's own prompt is then not rendered at all. Its
    negative prompt runs beside every variant, and a placeholder left in *that*
    is still refused: nothing fills it, so it would reach the encoder as the
    literal `{avoid}` - the same reason `prompt_assignments` refuses one.

    The schema already refuses variants beside a variable block and two with
    one label; they are checked again here for the callers that do not come
    through it, and a label that is only spaces - which `min_length` passes -
    would be a sheet row with no name.
    """
    if not variants:
        return [
            (one, substitute(request.prompt, one), substitute(request.negative_prompt, one))
            for one in prompt_assignments(request, parse_variables(prompt))
        ]
    if prompt.strip():
        raise ValueError("prompt variants are the prompt axis; leave the variables empty")
    for name in sorted(set(PLACEHOLDER.findall(request.negative_prompt or ""))):
        raise ValueError(
            f"negative prompt: {{{name}}} has no values; prompt variants take no variables"
        )
    labels = [variant.label.strip() for variant in variants]
    if not all(labels):
        raise ValueError("prompt variants: every variant needs a label")
    if len(set(labels)) != len(labels):
        raise ValueError("prompt variants: each variant needs its own label")
    return [
        ({VARIANT: label}, variant.prompt, request.negative_prompt)
        for label, variant in zip(labels, variants)
    ]


def axis_values(
    request: GenerateRequest,
    steps: str,
    cfg: str,
    mu: str,
    prompt: str = "",
    lora_scale: str = "",
    variants: Sequence[PromptVariant] | None = None,
    *,
    strength: str = "",
) -> dict[str, list[Any]]:
    """The axes resolved against the request they sit on top of.

    The prompt axis resolves to finished prompts rather than to the variables
    behind them: what the preview has to show - and what the contact sheet later
    groups by - is the text that will actually be encoded. Variants add their
    labels under `variant`, in the same order, so a caller can name each text.
    """
    resolved = resolve_prompts(request, prompt, variants)
    values: dict[str, list[Any]] = {"prompt": [text for _, text, _ in resolved]}
    if variants:
        values[VARIANT] = [one[VARIANT] for one, _, _ in resolved]
    return {**values, **_sampling_values(request, steps, cfg, mu, lora_scale, strength)}


def _sampling_values(
    request: GenerateRequest,
    steps: str,
    cfg: str,
    mu: str,
    lora_scale: str = "",
    strength: str = "",
) -> dict[str, list[Any]]:
    """The numeric axes, each falling back to the request's own value.

    Each is labelled with the box it is typed into - the grid fold's, and the
    Refine dialog's Strength - so a refusal names the box to fix.
    """
    return {
        "steps": parse_axis(steps, int, label="Steps") or [request.steps],
        "guidance_scale": parse_axis(cfg, float, label="CFG") or [request.guidance_scale],
        "mu": parse_axis(mu, float, allow_none=True, label="Mu") or [request.mu],
        # Only an axis when an adapter is selected: without one the scale is
        # not a setting at all (see `GenerateRequest.applied`), and a sweep of
        # it would be the same image several times over.
        "lora_scale": (
            parse_axis(lora_scale, float, label="LoRA weight") or [request.lora_scale]
            if request.lora
            else [request.lora_scale]
        ),
        # The same rule for the refine pass: with no source the strength is
        # not a setting, and the axis collapses to the one value.
        "strength": (
            parse_axis(strength, float, label="Strength") or [request.strength]
            if request.refine is not None
            else [request.strength]
        ),
    }


def parse_seeds(text: str | None) -> list[int]:
    """The seed axis as a *list* - `4471,8823`, `100-103` - rather than a count.

    Every other axis is a spec through `parse_axis`; seeds were the one axis
    that could only be "N consecutive from the base", so the second night of
    every two-night investigation - the two seeds that gave the composition,
    crossed with a finer sweep - could not be asked for. The values are not
    wrapped: the schema's own bounds are what turn a typo'd `-5` into an error
    rather than a valid 9.2-quintillion seed, and a cell outside them fails
    with the schema's message when the cells are built.
    """
    return parse_axis(text, int, label="Seed list")


def describe(values: dict[str, list[Any]], seeds: int, listed: bool = False) -> str:
    """`2 prompts x 4 steps x 3 CFG x 1 mu x 2 seeds`, for a preview or a log.

    The prompt count appears only when there is more than one: a sampling grid
    should read exactly as it did before the prompt became an axis. `listed`
    says the seeds were given rather than counted up from the base. Variants
    are counted as what they are, since that is what the sheet's rows say.
    """
    parts = []
    if len(values.get("prompt", [])) > 1:
        noun = "variants" if VARIANT in values else "prompts"
        parts.append(f"{len(values['prompt'])} {noun}")
    parts.append(f"{len(values['steps'])} steps")
    parts.append(f"{len(values['guidance_scale'])} CFG")
    parts.append(f"{len(values['mu'])} mu")
    # Like the prompt count: a sweep that does not touch the adapter reads
    # exactly as it did before the axis existed.
    if len(values.get("lora_scale", [])) > 1:
        parts.append(f"{len(values['lora_scale'])} LoRA")
    if len(values.get("strength", [])) > 1:
        parts.append(f"{len(values['strength'])} strength")
    parts.append(f"{seeds} {'listed ' if listed else ''}seed{'' if seeds == 1 else 's'}")
    return " x ".join(parts)


def _label(
    assignment: dict[str, str],
    steps: int,
    cfg: float,
    mu: float | None,
    lora_scale: float | None = None,
    strength: float | None = None,
) -> str:
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
    # Only where an adapter is loaded: on every other run it is a constant, and
    # a log line is read by eye.
    if lora_scale is not None:
        parts.append(f"LoRA {lora_scale:g}")
    # Likewise only on a refine pass.
    if strength is not None:
        parts.append(f"strength {strength:g}")
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
        None if request.true_cfg is None else round(request.true_cfg, 6),
        reference_key(request.reference),
        lora_key(request.lora, request.lora_scale),
        refine_key(request.refine, request.strength),
        cell.seed,
    )


def duplicate_count(
    cells: list["Cell"], applied: Callable[[GenerateRequest], GenerateRequest],
    model_id: str | None = None,
) -> int:
    """How many cells would render the very same image as an earlier cell.

    Compared as the pipeline will *run* them, not as the panel spelled them:
    a `{avoid}` variable that lives in the negative prompt alone is a legal
    three-value axis, and on a checkpoint that takes no negative prompt - or
    under a CFG that never clears the threshold - those three cells run the
    same prompt, steps, seed and generator and produce three identical PNGs
    while the preview reports three prompts and triples the estimate. Counted
    per image of a batch, like resume. Refine strengths that select the same
    integer schedule start are identical here; resume keys and run ids still
    retain the requested strength.
    """
    seen: set[CellKey] = set()
    duplicates = 0
    for cell in cells:
        ran = Cell(request=applied(cell.request), seed=cell.seed)
        for key in cell_keys(ran):
            if ran.request.refine is not None:
                start = start_index(
                    ran.request.steps, ran.request.strength, model_id or current_model_id(settings)
                )
                if start is not None:
                    key = (*key[:8], f"{ran.request.refine}@step:{start}", key[9])
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
    return duplicates


def cell_keys(cell: "Cell") -> list[CellKey]:
    """One key per image the cell will write - `seed + offset` for each of the batch.

    `completed_cells` answers with one row per *image*, under the image's own
    seed, so a cell is only done when every one of its images is there. Keying
    on the base seed alone declared a four-image cell rendered the moment its
    first image committed, and a resume then never drew the other three.
    """
    base = cell_key(cell)
    return [
        (*base[:-1], wrap_seed(cell.seed + offset)) for offset in range(cell.request.num_images)
    ]


def prune_done(cells: list[Cell], done: set[CellKey]) -> list[Cell]:
    """Drop the cells already sitting in the gallery, keeping the order.

    A cell counts as done only when every image of its batch is in `done`; a
    batch that was cut off part-way is rendered again in full, since the
    pipeline draws a batch as one call and cannot resume from image two.

    Order matters as much here as in `expand`: a resumed grid picks up where the
    night stopped, so what it does first is still the earliest seed it is short.
    """
    return [cell for cell in cells if not all(key in done for key in cell_keys(cell))]


def prompts_of(cells: list[Cell]) -> list[str]:
    """The distinct prompts a grid would render, for the resume lookup."""
    return list(dict.fromkeys(cell.request.prompt for cell in cells))


_KEEP: Any = object()


def cell_from_record(
    record: ImageRecord,
    *,
    width: int | None = None,
    height: int | None = None,
    prompt: str | None = None,
    negative_prompt: Any = _KEEP,
    variables: Any = _KEEP,
) -> Cell:
    """One gallery record as one cell: its whole reproduction set, with only
    what the caller names changed.

    The producers that run records again - Re-render (another size), Swap
    tags (another prompt or negative) - build their cells here, so what
    carries over cannot drift between them: the prompt pair, steps, CFG, the
    effective true CFG (a FLUX record from before the knob was separate
    carries the real CFG it ran), mu, the references, the adapter and its
    weight, the refine source and its strength (a refined record runs again
    as a pass of the same source), the seed and the variables - what the
    placeholders were set to, or which version of a comparison this is, the
    only place a variant's label lives. Raises pydantic's ValidationError
    where the result is no valid request (an emptied prompt, a size past the
    guards).
    """
    return Cell(
        request=GenerateRequest(
            prompt=record.prompt if prompt is None else prompt,
            negative_prompt=record.negative_prompt if negative_prompt is _KEEP else negative_prompt,
            width=record.width if width is None else width,
            height=record.height if height is None else height,
            steps=record.steps,
            guidance_scale=record.guidance_scale,
            true_cfg=record.true_cfg,
            mu=record.mu,
            reference=record.reference or [],
            lora=record.lora,
            lora_scale=record.lora_scale if record.lora_scale is not None else 1.0,
            refine=record.refine,
            strength=record.strength if record.refine is not None else None,
            seed=record.seed,
        ),
        seed=record.seed,
        variables=(record.variables if variables is _KEEP else variables) or None,
    )


def cells_from_records(
    records: Sequence[ImageRecord], *, width: int | None = None, height: int | None = None
) -> list[Cell]:
    """`cell_from_record` over `records`, in the order given - the callers
    pass them oldest first, so a run again lands in the order the originals
    did. One size for every cell where `width`/`height` are given, else each
    record's own."""
    return [cell_from_record(record, width=width, height=height) for record in records]


def denoised_steps(request: GenerateRequest) -> int:
    """The steps one image of this request denoises: all of them, or the part
    of the schedule a refine pass's strength leaves (`refine.steps_run`)."""
    if request.refine is None:
        return request.steps
    return steps_run(request.steps, request.strength, current_model_id(settings))


def progress_steps(request: GenerateRequest) -> int:
    """The steps a cell's progress counts through, before its first callback.

    A batch denoises as one call and its steps are counted once - except on a
    refine pass, which makes one call per image (`DiffusersBackend._refine`)
    and counts each.
    """
    if request.refine is None:
        return request.steps
    return denoised_steps(request) * request.num_images


def estimate_seconds(
    cells: list[Cell], per_step_ms: float | None, *,
    refine_costs: dict[int, float] | None = None,
) -> float | None:
    """Generation time from direct step costs or matching refine durations.

    One helper for the preview and the CLI so the arithmetic is not written
    twice. Per *step* times the cell's steps times its batch: a grid's cells
    disagree about step count, so an average image duration would misstate
    most of them. Refine cells use full per-image durations at the same number
    of denoised steps: their fixed encode/decode work must not be multiplied
    away or priced from direct renders. No estimate if any cell has no matching
    history. The CLI pays a load per invocation on top.
    """
    priced = _cell_ms(cells, per_step_ms, refine_costs)
    return None if priced is None else round(sum(priced) / 1000, 1)


def _cell_ms(
    cells: list[Cell], per_step_ms: float | None, refine_costs: dict[int, float] | None,
) -> list[float] | None:
    """Each cell's price in milliseconds, as `estimate_seconds` sums them;
    None if any cell has no matching history."""
    priced = []
    for cell in cells:
        request = cell.request
        if request.refine is None:
            if per_step_ms is None:
                return None
            image_ms = per_step_ms * request.steps
        else:
            image_ms = (refine_costs or {}).get(denoised_steps(request))
            if image_ms is None:
                return None
        priced.append(image_ms * request.num_images)
    return priced


def seeds_within_budget(
    cells: list[Cell], seeds: int, per_step_ms: float | None, budget_seconds: float, *,
    refine_costs: dict[int, float] | None = None,
) -> int | None:
    """How many of a grid's seeds a time budget completes, by the job's own
    rule: a cell starts while the deadline has not passed and runs to its end
    (`JobManager._generate`), so a seed whose last cell starts in time
    completes, though it ends past the deadline. Dividing the budget by a
    seed's share of the estimate said 4 of 6 for a grid that made 5. Seeds are
    the outer loop, each running the same cells; priced as `estimate_seconds`
    prices, from the job's start - the load and any job ahead spend a budget
    too, which no estimate counts. None where it has no price."""
    priced = _cell_ms(cells, per_step_ms, refine_costs)
    if priced is None or seeds < 1 or len(cells) % seeds:
        return None
    per_seed = len(cells) // seeds
    elapsed_ms = 0.0
    for index, ms in enumerate(priced):
        if elapsed_ms > budget_seconds * 1000:
            return index // per_seed
        elapsed_ms += ms
    return seeds


def run_id(cells: list[Cell], model_id: str, quantization: str) -> str:
    """A name for the run that is the same however many jobs it takes to finish.

    A job id is minted per submission, so a night that died and was resumed
    carries two of them, and the search grammar cannot join them back into one
    matrix. This is derived from the run's own definition instead - every cell
    key of the *full* grid, before resume subtracts anything, plus the size,
    the batch and the checkpoint that decide what the cells become - so the
    resubmission that continues a grid recomputes the very same id, and the
    contact sheet can ask for `run:<id>` and get the whole night.

    Order-independent on purpose: two grids that render the same cells are the
    same run however their axes were spelled.

    A cell is named by the shortest key that carries all of its optional
    elements: none of true CFG, references, a LoRA or a refine source and it
    is the six-element key images have carried since before any of them
    existed, so every run id from an earlier build - and every grid that uses
    none of them - is unchanged. Each one that is set extends the name by one element, in the
    order they were added.
    """
    first = cells[0].request

    def named(cell: Cell) -> str:
        key = cell_key(cell)
        head, extras, seed = key[:-1 - _KEY_EXTENSIONS], list(key[-1 - _KEY_EXTENSIONS : -1]), key[-1]
        while extras and extras[-1] is None:
            extras.pop()
        return json.dumps([*head, *extras, seed])

    definition = {
        "model": model_id,
        "quant": quantization,
        "size": [first.width, first.height],
        "batch": first.num_images,
        "cells": sorted(named(cell) for cell in cells),
    }
    digest = hashlib.sha1(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    return digest[:12]


def expand(
    request: GenerateRequest,
    *,
    prompt: str = "",
    steps: str = "",
    cfg: str = "",
    mu: str = "",
    lora_scale: str = "",
    strength: str = "",
    seeds: int = 1,
    seed_list: list[int] | None = None,
    variants: Sequence[PromptVariant] | None = None,
) -> list[Cell]:
    """The cartesian product, as cells ready to hand to a job.

    Every cell is a real `GenerateRequest` rather than a patch to apply later,
    so an out-of-range axis value fails here - with the schema's own message -
    instead of somewhere inside the denoising loop at 3am. A prompt that grew
    past the schema's 4000 characters once its values were substituted in is
    caught by that same rule.

    The seed axis is either `seeds` consecutive from the base - stepped by the
    batch size, as a sweep steps - or exactly `seed_list`, which wins when it
    is given. The ceiling counts whichever applies: swapping only the loop
    would have let a 64-value list queue 64 times past `max_grid_images`.

    `variants`, when given, is the prompt axis in place of `prompt`'s variables
    (see `resolve_prompts`), in the order written: the first is the first row.
    Everything downstream - the cell key, the run id, resume, the duplicate
    count - reads the resolved prompt, so a variant is an ordinary prompt cell
    to all of it and only the label is new.
    """
    prompts = resolve_prompts(request, prompt, variants)
    values = _sampling_values(request, steps, cfg, mu, lora_scale, strength)
    seed_count = len(seed_list) if seed_list else seeds
    total = (
        len(prompts)
        * len(values["steps"])
        * len(values["guidance_scale"])
        * len(values["mu"])
        * len(values["lora_scale"])
        * len(values["strength"])
        * seed_count
    )
    if total * request.num_images > settings.max_grid_images:
        raise ValueError(
            f"{total * request.num_images} images is past the {settings.max_grid_images} "
            "a grid may queue; narrow an axis or drop the batch size"
        )

    base = request.seed if request.seed is not None else random_seed()
    # Step past the whole batch, exactly as a sweep does, so no two images in
    # the grid are drawn from the same seed. A listed seed is taken as given -
    # the schema, not a wrap, decides whether it is one.
    seed_values = seed_list or [wrap_seed(base + index * request.num_images) for index in range(seeds)]
    template = request.model_dump()
    cells: list[Cell] = []
    for seed in seed_values:
        # Every parameter cell of one seed shares it - comparing settings on
        # the same noise is the point.
        # The prompt sits just inside the seed, for the reason the seed is
        # outermost: a night that ends early has then swept every setting for
        # the prompts it reached, which is a comparison, rather than leaving
        # every prompt part-swept.
        for assignment, text, negative in prompts:
            for step_count in values["steps"]:
                for scale in values["guidance_scale"]:
                    for shift in values["mu"]:
                        # Inside the rest, because it is the cheapest change of
                        # all: a `set_adapters` on the pipeline already in memory.
                        for weight in values["lora_scale"]:
                            # One value unless this is a refine pass with a
                            # strength axis.
                            for amount in values["strength"]:
                                cells.append(
                                    Cell(
                                        request=GenerateRequest(
                                            **{
                                                **template,
                                                "prompt": text,
                                                "negative_prompt": negative,
                                                "steps": step_count,
                                                "guidance_scale": scale,
                                                "true_cfg": request.true_cfg,
                                                "mu": shift,
                                                "lora_scale": weight,
                                                "strength": amount,
                                                "seed": seed,
                                                "runs": 1,
                                            }
                                        ),
                                        seed=seed,
                                        label=_label(
                                            assignment,
                                            step_count,
                                            scale,
                                            shift,
                                            weight if request.lora else None,
                                            amount if request.refine is not None else None,
                                        ),
                                        variables=assignment or None,
                                    )
                                )
                                cells[-1].request.check_refine_steps()
    return cells
