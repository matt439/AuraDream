"""Then refine: a refine pass of every image a job makes, right after it.

The pure parts. The pass itself is Mass refine's (`refine_batch.plan`),
built in the worker from each record once it is saved, so its source is a
gallery id like any pass's; what is here is what the submit checks and
prices before there is a record - a stand-in pass of a cell's request -
and the run the passes are saved under.
"""
from __future__ import annotations

import hashlib
import json

from .grid import Cell
from .refine import scaled
from .refine_batch import too_big
from .config import settings
from .schemas_generate import GenerateRequest, ThenRefine, default_strength


def resolved(spec: ThenRefine) -> ThenRefine:
    """`spec` with its strength filled as the running checkpoint's own where
    it names none: what the job runs, the run id names and the page says."""
    if spec.strength is not None:
        return spec
    return spec.model_copy(update={"strength": default_strength()})


def stand_in(request: GenerateRequest, spec: ThenRefine) -> GenerateRequest:
    """The pass a cell of `request` would get, before any image exists: the
    pass's size, strength and steps, one image, a placeholder source. For
    the checks and the estimates only - never queued, never saved; the real
    pass is `refine_batch.plan` of the saved record, which makes the same
    request with the record's own id."""
    if request.edit is not None:
        raise ValueError("Then refine cannot follow an instruction edit")
    width, height = scaled(request.width, request.height, spec.scale)
    return request.model_copy(update={
        "width": width, "height": height, "num_images": 1, "runs": 1,
        "refine": 1, "strength": spec.strength, "steps": spec.steps or request.steps,
        "reference": [], "mask": None, "outpaint": None, "upscaler": None, "crop_anchor": "center",
    })


def execution_units(cells: list[Cell], spec: ThenRefine | None) -> list[Cell]:
    """The calls a job of `cells` makes, in run order: each cell and, with
    Then refine, a stand-in pass per image after it. What the grid preview,
    the CLI and the queue's stored estimate price, so the expansion is
    written once."""
    if spec is None:
        return list(cells)
    return [
        unit
        for cell in cells
        for unit in [cell, *[Cell(request=stand_in(cell.request, spec), seed=cell.seed)] * cell.request.num_images]
    ]


def refusal(requests: list[GenerateRequest], spec: ThenRefine, unavailable: str | None) -> str | None:
    """Why these cells cannot each be followed by a pass, in words, or None.
    `unavailable` is `submission.refine_unavailable`'s answer."""
    if unavailable is not None:
        return unavailable
    seen = set()
    for request in requests:
        if request.edit is not None:
            return "Then refine cannot follow an instruction edit"
        if request.refine is not None:
            what = (
                "an outpaint" if request.outpaint is not None
                else "an inpaint" if request.mask is not None else "a refine pass"
            )
            return f"Then refine follows a direct render; this is already {what} of #{request.refine}"
        key = request.width, request.height, request.steps
        if key in seen:
            continue
        seen.add(key)
        width, height = scaled(request.width, request.height, spec.scale)
        if too_big(width, height):
            return (
                f"{request.width}×{request.height} at {spec.scale:g}× is {width}×{height}, past the "
                f"{settings.refine_max_side}-pixel side and "
                f"{settings.refine_max_pixels / 1e6:.1f} MP a refine pass takes"
            )
        probe = stand_in(request, spec)
        try:
            probe.check_budget()
        except ValueError as exc:
            return str(exc)
        try:
            probe.check_refine_steps()
        except ValueError:
            return (
                f"strength {spec.strength:g} of {probe.steps} steps leaves no step; "
                "set the pass's Steps or raise the strength"
            )
    return None


def pass_run_id(base_run_id: str, spec: ThenRefine) -> str:
    """The run the passes are saved under, separate from their bases and
    the same for every job that continues the base run, so a
    Resume or a Finish of it joins the passes already made."""
    definition = {
        "then_refine_of": base_run_id, "scale": spec.scale,
        "strength": spec.strength, "steps": spec.steps,
    }
    return hashlib.sha1(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:12]
