"""Small helpers the backend and catalogue tests share.

`DiffusersBackend` is imported inside `loaded_backend`, so importing this
module for `catalogue_repos` alone does not pull in torch and diffusers.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from backend import launch

if TYPE_CHECKING:
    from backend.backends.diffusers_backend import DiffusersBackend


def catalogue_repos() -> list[str]:
    """Every catalogue checkpoint id, its Nunchaku Lite repos included."""
    return [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]


def loaded_backend(pipe) -> DiffusersBackend:
    """A DiffusersBackend 'loaded' with `pipe` and no weights."""
    from backend.backends.diffusers_backend import DiffusersBackend

    backend = DiffusersBackend()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend
