"""The server's current checkpoint, separate from immutable startup settings.

Only main's exclusive switch operation writes this selection. Readers in the
loader, schemas, queue and gallery use the same snapshot; the standalone CLI
never changes it and therefore keeps its environment/command-line settings.
"""
from __future__ import annotations

import sys

from .launch import CHECKPOINTS, FORMATS

_selection: tuple[str | None, int] = (None, 0)


def current_model_id(settings) -> str:
    return _selection[0] or settings.model_id


def revision() -> int:
    return _selection[1]


def select(model_id: str) -> None:
    global _selection
    _selection = (model_id, _selection[1] + 1)


def available(settings) -> list[dict[str, str]]:
    """The startup catalogue, within this process's quantization/runtime.

    Keep the startup's explicit off-catalogue selection available for switching
    back. A HTTP request cannot introduce arbitrary repository or local paths.
    Nunchaku retains the startup's weight format and requires Linux, as at boot.
    """
    fmt = None
    if settings.quantization == "nunchaku":
        if sys.platform != "linux":
            return []
        fmt = next((
            name for name, _ in FORMATS
            if any(c.lite_repo(name) == settings.model_id for c in CHECKPOINTS)
        ), "nvfp4")
    models = []
    for checkpoint in CHECKPOINTS:
        model_id = checkpoint.lite_repo(fmt) if fmt else checkpoint.repo
        if model_id:
            models.append({"model_id": model_id, "label": checkpoint.name})
    if settings.model_id not in {m["model_id"] for m in models}:
        models.append({"model_id": settings.model_id, "label": settings.model_id})
    return models
