"""Cold saved-draft validation, separate from storage and job admission.

A preset is editable intent. Its effective request/axes are an explanation,
never saved back over that intent or handed to the queue. Nullable seeds stay
nullable; random-variable examples use only a disposable, fixed sample.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from . import grid
from .model_defaults import defaults_for
from .schemas_generate import GenerateRequest, GridRequest
from .schemas_presets import PresetPayload


class PresetValidation(BaseModel):
    """Effective base settings plus requested and normalized grid values.

    ``inactive`` names saved fields unused on at least one effective cell.
    ``values`` follows the ordinary preview's resolved requested axes, while
    ``effective_values`` explains negative/Mu/True CFG after normalization.
    ``effective`` serializes only GenerateRequest's base fields; the grid's
    editable definition remains in ``saved.grid``. Its template and nullable
    seed survive; no sampled seed or expanded cell is returned as draft state.
    """

    saved: PresetPayload
    effective: GenerateRequest
    inactive: list[str]
    values: dict[str, list[float | int | str | None]]
    effective_values: dict[str, list[float | int | str | None]] = Field(default_factory=dict)
    cells: int
    images: int
    prompt_sample: bool = False


def validate_payload(
    payload: PresetPayload,
    model_id: str,
    capabilities: Mapping[str, Any] | None = None,
    *,
    applier: Callable[[GenerateRequest], GenerateRequest] | None = None,
) -> PresetValidation:
    """Validate a draft without resolving sources, drawing seeds or loading.

    The caller holds model selection stable and may supply the current
    backend's pure ``applied`` reader for its known live capabilities. Without
    that reader, the exact checkpoint's cold profile supplies the same request
    normalization. No backend description (and therefore no CUDA read) occurs.
    """
    # Revalidate even an internal model_copy, which bypasses field validators.
    payload = PresetPayload.model_validate(payload.model_dump())
    profile = defaults_for(model_id)
    caps = {**profile, **{key: value for key, value in (capabilities or {}).items() if value is not None}}
    if not caps.get("text_to_image_supported", True):
        raise ValueError(caps.get("text_to_image_reason") or "This checkpoint cannot generate text-to-image setups.")

    def applied(request: GenerateRequest) -> GenerateRequest:
        if applier is not None:
            return applier(request)
        return request.applied(
            negative_prompt_supported=caps["negative_prompt_supported"],
            cfg_threshold=caps["cfg_threshold"],
            mu_supported=caps["mu_supported"],
            true_cfg_supported=caps["true_cfg_supported"],
        )

    # Explicitly resolve defaults from the supplied id. GenerateRequest's
    # ordinary omitted-field defaults otherwise read the current selection.
    fields = payload.model_dump(exclude={"mode", "grid"})
    for name in ("steps", "guidance_scale", "mu", "true_cfg"):
        if fields[name] is None:
            fields[name] = profile.get(name)
    if payload.mode == "grid":
        fields["runs"] = 1
        assert payload.grid is not None
        request = GridRequest(**fields, **payload.grid.model_dump(), resume=False)
    else:
        request = GenerateRequest(**fields)

    effective = applied(request)
    inactive: set[str] = set()
    true_cfg_supported = caps["true_cfg_supported"]
    if applier is not None and payload.true_cfg is not None:
        # A loaded custom pipeline may disagree with the cold profile. This
        # pure normalization probe distinguishes its separate True CFG knob
        # from an ordinary CFG value, including equal values in the draft.
        probe = request.model_copy(update={
            "negative_prompt": "saved setup capability check",
            "guidance_scale": 1.0, "true_cfg": 20.0,
        })
        true_cfg_supported = applied(probe).true_cfg == 20.0

    def note_inactive(before: GenerateRequest, after: GenerateRequest) -> None:
        if before.negative_prompt and after.negative_prompt is None:
            inactive.add("negative_prompt")
        if before.mu is not None and after.mu is None:
            inactive.add("mu")
        if payload.true_cfg is not None and (
            not true_cfg_supported or after.true_cfg is None
        ):
            inactive.add("true_cfg")
        if caps.get("cfg_ignored"):
            inactive.add("guidance_scale")

    if isinstance(request, GridRequest):
        axes = payload.grid
        assert axes is not None
        if axes.mu_axis.strip() and applied(request.model_copy(update={"mu": 0.0})).mu is None:
            raise ValueError("Mu axis is unavailable on the selected checkpoint; clear it before restoring this setup.")
        if axes.cfg_axis.strip() and caps.get("cfg_ignored"):
            raise ValueError("CFG axis is ignored on the selected checkpoint; clear it before restoring this setup.")
        seed_list = grid.parse_seeds(axes.seed_axis)
        # Only the disposable expansion sees this sentinel. Neither saved nor
        # effective state, nor the seed-axis answer, receives it.
        sample = request.model_copy(update={"seed": 0}) if request.seed is None else request
        cells = grid.expand(
            sample, prompt=axes.prompt_axis, prompt_mode=axes.prompt_mode,
            steps=axes.steps_axis, cfg=axes.cfg_axis, mu=axes.mu_axis,
            seeds=axes.seeds, seed_list=seed_list or None, size=axes.size_axis,
        )
        values = grid.axis_values(
            request, axes.steps_axis, axes.cfg_axis, axes.mu_axis,
            axes.prompt_axis, prompt_mode=axes.prompt_mode, cells=cells,
            size=axes.size_axis,
        )
        effective_cells = [applied(cell.request) for cell in cells]
        for cell, normalized in zip(cells, effective_cells):
            note_inactive(cell.request, normalized)
        values["seed"] = seed_list
        effective_values = {
            "negative_prompt": list(dict.fromkeys(cell.negative_prompt for cell in effective_cells)),
            "mu": list(dict.fromkeys(cell.mu for cell in effective_cells)),
            "true_cfg": list(dict.fromkeys(cell.true_cfg for cell in effective_cells)),
        }
        return PresetValidation(
            saved=payload, effective=effective, inactive=sorted(inactive),
            values=values, effective_values=effective_values, cells=len(cells),
            images=sum(cell.request.num_images for cell in cells),
            prompt_sample=axes.prompt_mode == "random" and payload.seed is None and not seed_list,
        )

    request.check_budget()
    note_inactive(request, effective)
    return PresetValidation(
        saved=payload, effective=effective, inactive=sorted(inactive),
        values={}, cells=request.runs, images=request.num_images * request.runs,
    )
