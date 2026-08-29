"""The contract every generation engine implements.

Keeping the API surface this small is what lets a ComfyUI-driven backend replace
the in-process diffusers one without the HTTP layer or the UI noticing.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas import GenerateRequest

# (completed_steps, total_steps) -> None
ProgressFn = Callable[[int, int], None]


class GenerationBackend(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def load(self) -> None:
        """Bring weights into memory. Safe to call when already loaded."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release weights and free VRAM."""

    @abc.abstractmethod
    def is_ready(self) -> bool: ...

    @abc.abstractmethod
    def generate(
        self,
        request: "GenerateRequest",
        seed: int,
        on_progress: ProgressFn | None = None,
    ) -> list["Image"]:
        """Run one synchronous generation. Called on a worker thread."""

    @abc.abstractmethod
    def describe(self) -> dict:
        """Backend/device metadata for the status endpoint."""
