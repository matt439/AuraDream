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

from .launch import CHECKPOINTS, FORMATS
from .execution import require_default_execution
from .loader_choices import OFFLOAD_CHOICES, QUANTIZATION_CHOICES
from .model_defaults import OFFLOAD_MODES, QUANT_MODES

# One assignment publishes the entire selection, including its revision. The
# initial two-field form remains useful to reset the selection in tests. The
# revision starts at this process's start in milliseconds (below 2**53), so a
# page still holding the last process's revision is refused after a restart.
_selection: tuple = (None, int(time.time() * 1000))


def current_settings(settings):
    """Read one immutable snapshot without modifying the launch configuration."""
    selected = _selection
    quant, offload = selected[2:] if len(selected) == 4 else (None, None)
    return replace(
        settings,
        model_id=selected[0] or settings.model_id,
        quantization=quant or settings.quantization,
        offload=offload or settings.offload,
    )


def current_model_id(settings) -> str:
    return _selection[0] or settings.model_id


def revision() -> int:
    return _selection[1]


def select(model_id: str, *, quantization: str | None = None, offload: str | None = None) -> None:
    global _selection
    selected = _selection
    quant, placement = selected[2:] if len(selected) == 4 else (None, None)
    _selection = (model_id, selected[1] + 1, quantization or quant, offload or placement)


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
    execution: object = None,
):
    """Resolve browser options and refuse incompatible choices before unload."""
    # This also covers direct/unvalidated callers, not only typed HTTP input.
    # Refuse before even resolving the current loader or browsing a catalogue.
    require_default_execution(execution)
    selected = current_settings(settings)
    quantization = selected.quantization if quantization is None else quantization
    offload = selected.offload if offload is None else offload
    if quantization not in QUANT_MODES:
        raise ValueError(f"Quantization must be one of: {', '.join(QUANT_MODES)}")
    if offload not in OFFLOAD_MODES:
        raise ValueError(f"Offload must be one of: {', '.join(OFFLOAD_MODES)}")
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
    return replace(selected, model_id=model_id, quantization=quantization, offload=offload)
