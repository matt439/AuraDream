"""The server's model and loading options, separate from startup settings.

Only main's exclusive switch operation writes this selection. Readers in the
loader, schemas, queue and gallery use the same snapshot; the standalone CLI
never changes it and therefore keeps its environment/command-line settings.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from sys import modules as imported_modules
from typing import Any

from .launch import CHECKPOINTS, FORMATS
from .execution import (
    INVALID_LOADER_CONFIGURATION, canonical_loader_execution, loader_configuration_reason,
    loader_unavailable_reason, require_loader_available, step_cache_configuration_reason,
    unavailable_reason,
)
from .loader_choices import OFFLOAD_CHOICES, QUANTIZATION_CHOICES
from .model_defaults import OFFLOAD_MODES, QUANT_MODES

# One assignment publishes the entire selection, including its revision. The
# initial two-field form remains useful to reset the selection in tests. The
# revision starts at this process's start in milliseconds (below 2**53), so a
# page still holding the last process's revision is refused after a restart.
_selection: tuple = (None, int(time.time() * 1000))
# Omission keeps the current intent; an explicitly empty recipe selects Eager.
UNSET_EXECUTION = object()


def current_settings(settings):
    """Read one immutable snapshot without modifying the launch configuration."""
    selected = _selection
    if len(selected) < 2:
        raise ValueError("the current model selection is incomplete")
    quant, offload = selected[2:4] if len(selected) >= 4 else (None, None)
    return replace(
        settings,
        model_id=selected[0] or settings.model_id,
        quantization=quant or settings.quantization,
        offload=offload or settings.offload,
        loader_execution=selected[4] if len(selected) >= 5 else getattr(settings, "loader_execution", None),
        loader_execution_error=selected[5] if len(selected) >= 6 else getattr(settings, "loader_execution_error", None),
    )


def current_model_id(settings) -> str:
    return _selection[0] or settings.model_id


def revision() -> int:
    return _selection[1]


def select(
    model_id: str, *, quantization: str | None = None, offload: str | None = None,
    loader_execution: object = UNSET_EXECUTION,
) -> None:
    global _selection
    selected = _selection
    if len(selected) < 2:
        raise ValueError("the current model selection is incomplete")
    quant, placement = selected[2:4] if len(selected) >= 4 else (None, None)
    updated = (model_id, selected[1] + 1, quantization or quant, offload or placement)
    if loader_execution is UNSET_EXECUTION:
        _selection = (*updated, selected[4], selected[5] if len(selected) >= 6 else None) if len(selected) >= 5 else updated
    else:
        # Only an explicitly validated choice repairs a startup diagnostic.
        _selection = (*updated, canonical_loader_execution(loader_execution), None)


def execution_configuration_reason(
    selected, *, execution: object = UNSET_EXECUTION, model_id: str | None = None,
    quantization: str | None = None, offload: str | None = None,
) -> str | None:
    """A cold finding from one frozen selection, never a later global read."""
    if execution is UNSET_EXECUTION:
        if selected.loader_execution_error is not None:
            return INVALID_LOADER_CONFIGURATION
        execution = selected.loader_execution
    return loader_configuration_reason(
        execution, model_id=model_id or selected.model_id,
        quantization=quantization or selected.quantization,
        offload=offload or selected.offload, device=selected.device,
    )


def execution_status(selected, *, execution: object = UNSET_EXECUTION) -> dict:
    actual = canonical_loader_execution(selected.loader_execution)
    draft = actual if execution is UNSET_EXECUTION else canonical_loader_execution(execution)
    configuration_reason = execution_configuration_reason(selected, execution=execution)
    reason = configuration_reason or loader_unavailable_reason(draft)
    return {
        "execution": actual.model_dump(exclude_none=True) if actual is not None else None,
        "execution_available": reason is None,
        "execution_unavailable_reason": reason,
        "execution_configuration_reason": configuration_reason,
    }


def execution_options(
    selected, *, execution: object = UNSET_EXECUTION, model_id: str | None = None,
    quantization: str | None = None, offload: str | None = None,
) -> dict[str, list[dict]]:
    """Keep every named choice visible, including incompatible combinations."""
    draft = canonical_loader_execution(selected.loader_execution if execution is UNSET_EXECUTION else execution)
    values = draft.model_dump(exclude_none=True) if draft is not None else {}
    result = {}
    for field, key, choices in (
        ("compilation", "compilation", ((None, "Eager"), ("regional-v1", "Regional compilation"))),
        ("attention_backend", "attention", ((None, "Native"), ("cudnn-v1", "cuDNN attention"))),
    ):
        options = []
        for value, label in choices:
            candidate = canonical_loader_execution({**values, field: value})
            options.append({
                "value": value,
                "label": label,
                "configuration_reason": loader_configuration_reason(
                    candidate, model_id=model_id or selected.model_id,
                    quantization=quantization or selected.quantization,
                    offload=offload or selected.offload, device=selected.device,
                ),
                "unavailable_reason": loader_unavailable_reason(candidate),
            })
        result[key] = options
    result["step_cache"] = [{
        "value": value,
        "label": label,
        "configuration_reason": step_cache_configuration_reason(
            {"step_cache": value}, loader_execution=draft,
            model_id=model_id or selected.model_id,
            quantization=quantization or selected.quantization,
            offload=offload or selected.offload, device=selected.device,
        ),
        "unavailable_reason": unavailable_reason({"step_cache": value}),
    } for value, label in ((None, "Off"), ("fbc-005-v1", "First Block Cache"))]
    return result


def format_for(settings) -> str:
    """The selected Nunchaku weight format, or the RTX 50 default."""
    model_id = current_model_id(settings)
    return next((
        name for name, _ in FORMATS
        if any(c.lite_repo(name) == model_id for c in CHECKPOINTS)
    ), "nvfp4")


def options(settings) -> dict[str, list[dict[str, str]]]:
    """Cold browser choices; Nunchaku's kernels are Linux-only."""
    return {
        "quantizations": [
            {"value": value, "label": label, "description": description}
            for value, label, description in QUANTIZATION_CHOICES
            if value != "nunchaku" or sys.platform == "linux"
        ],
        "offloads": [
            {"value": value, "label": label, "description": description}
            for value, label, description in OFFLOAD_CHOICES
        ],
        "formats": [
            {"value": value, "label": value.upper(), "description": label} for value, label in FORMATS
        ],
    }


def available(settings, quantization: str | None = None, format: str | None = None) -> list[dict[str, str]]:
    """The catalogue for a requested quantization and optional Nunchaku format.

    Keep the startup's explicit off-catalogue selection available for switching
    back. A HTTP request cannot introduce arbitrary repository or local paths.
    An unknown startup checkpoint remains tied to its explicit quantization.
    """
    quantization = quantization or current_settings(settings).quantization
    if quantization not in QUANT_MODES or (format is not None and format not in dict(FORMATS)):
        return []
    fmt = None
    if quantization == "nunchaku":
        if sys.platform != "linux":
            return []
        fmt = format or format_for(settings)
    models = []
    for checkpoint in CHECKPOINTS:
        model_id = checkpoint.lite_repo(fmt) if fmt else checkpoint.repo
        if model_id:
            models.append({"model_id": model_id, "label": checkpoint.name})
    catalogue = {c.repo for c in CHECKPOINTS} | {
        c.lite_repo(name) for name, _ in FORMATS for c in CHECKPOINTS if c.lite
    }
    if (
        settings.model_id not in catalogue
        and quantization == settings.quantization
        and settings.model_id not in {m["model_id"] for m in models}
    ):
        models.append({"model_id": settings.model_id, "label": settings.model_id})
    return models


def validate(
    settings, model_id: str, *, quantization: str | None = None,
    offload: str | None = None, format: str | None = None,
    execution: object = UNSET_EXECUTION, load: bool = True, backend: Any = None,
):
    """Resolve browser options and refuse incompatible choices before unload.

    ``backend`` is the running instance whose gate and host decide an
    edit-only target (``instruction_edit.require_model_available``); without
    one the gate counts as closed.
    """
    # Explicit unsupported load attempts stop before even resolving options.
    # An omitted recipe preserves the selected intent, including a diagnostic.
    checked = canonical_loader_execution(execution) if execution is not UNSET_EXECUTION else UNSET_EXECUTION
    if checked is not UNSET_EXECUTION and load:
        require_loader_available(checked)
    selected = current_settings(settings)
    if checked is UNSET_EXECUTION:
        if selected.loader_execution_error is not None:
            raise ValueError(INVALID_LOADER_CONFIGURATION)
        checked = canonical_loader_execution(selected.loader_execution)
        if load:
            require_loader_available(checked)
    quantization = selected.quantization if quantization is None else quantization
    offload = selected.offload if offload is None else offload
    if quantization not in QUANT_MODES:
        raise ValueError(f"Quantization must be one of: {', '.join(QUANT_MODES)}")
    if offload not in OFFLOAD_MODES:
        raise ValueError(f"Offload must be one of: {', '.join(OFFLOAD_MODES)}")
    # Before the catalogue, the Hub, the save and the unload, for Apply &
    # save as for Switch & load: an edit-only checkpoint is selected only
    # where the backend answered the literal True, by its exact id, and
    # inside the measured loader envelope.
    from .instruction_edit import is_edit_only, require_model_available

    if is_edit_only(model_id):
        target = replace(selected, model_id=model_id, quantization=quantization, offload=offload,
                         loader_execution=checked, loader_execution_error=None)
        require_model_available(
            model_id, qualified=False if backend is None else backend.instruction_edit_qualified(),
            loader=None if backend is None else backend.instruction_edit_loader(target),
        )
    reason = loader_configuration_reason(
        checked, model_id=model_id, quantization=quantization, offload=offload,
        device=selected.device,
    )
    if reason is not None:
        raise ValueError(reason)
    if format is not None and format not in dict(FORMATS):
        raise ValueError(f"Nunchaku format must be one of: {', '.join(dict(FORMATS))}")
    if quantization == "nunchaku" and sys.platform != "linux":
        raise ValueError("Nunchaku needs Linux; on Windows, start the server in WSL 2.")
    if quantization == "nunchaku":
        true_values = {"1", "ON", "YES", "TRUE"}
        if (
            os.getenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "false").upper() not in true_values
            or os.getenv("DIFFUSERS_DISABLE_REMOTE_CODE", "false").upper() in true_values
        ):
            raise ValueError(
                "Nunchaku needs DIFFUSERS_TRUST_REMOTE_KERNELS=true before server startup; "
                "the environment currently disables its remote kernels."
            )
        # Diffusers freezes this flag at import. An imported ASGI app may
        # have loaded it before the environment changed: refuse before
        # unloading working weights, without importing diffusers for a list.
        constants = imported_modules.get("diffusers.utils.constants")
        if constants is not None and not getattr(constants, "DIFFUSERS_TRUST_REMOTE_KERNELS", False):
            raise ValueError(
                "Nunchaku remote kernels were disabled when Diffusers was imported; "
                "restart with DIFFUSERS_TRUST_REMOTE_KERNELS=true or python -m backend.main."
            )
    target_format = next((
        name for name, _ in FORMATS
        if any(c.lite_repo(name) == model_id for c in CHECKPOINTS)
    ), None)
    if model_id not in {item["model_id"] for item in available(settings, quantization, format or target_format)}:
        raise ValueError("Choose a checkpoint from the catalogue for this quantization.")
    return replace(selected, model_id=model_id, quantization=quantization, offload=offload,
                   loader_execution=checked, loader_execution_error=None)
