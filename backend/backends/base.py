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
# Polled between denoising steps; True means stop.
CancelFn = Callable[[], bool]


class Cancelled(Exception):
    """Raised out of `generate()` when the caller asked it to stop.

    A diffusion step is not interruptible from the outside, so cancellation is
    cooperative: the backend polls `should_cancel` at a step boundary and unwinds
    from there. Whatever finished before that point is still valid output.
    """


class GenerationBackend(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def load(self) -> None:
        """Bring weights into memory. Safe to call when already loaded."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release weights and free VRAM. May block behind a load in progress."""

    def try_unload(self) -> bool:
        """Unload unless a load is in progress, in which case do nothing.

        For shutdown: the process is exiting, so freeing VRAM buys nothing, and
        blocking the event loop behind a multi-GB download to discover there
        was nothing to unload froze Ctrl-C for the length of the download.
        Returns whether the weights are now unloaded. The default cannot tell,
        so it unloads.
        """
        self.unload()
        return True

    @abc.abstractmethod
    def is_ready(self) -> bool: ...

    @abc.abstractmethod
    def generate(
        self,
        request: "GenerateRequest",
        seed: int,
        on_progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
    ) -> list["Image"]:
        """Run one synchronous generation. Called on a worker thread.

        Poll `should_cancel` at every step boundary and raise `Cancelled` when it
        answers True.
        """

    @abc.abstractmethod
    def describe(self) -> dict:
        """Backend/device metadata for the status endpoint."""

    def applied(self, request: "GenerateRequest") -> "GenerateRequest":
        """The request as this backend will actually run it - see
        `GenerateRequest.applied`. What the callers record beside the image and
        compare against for resume, so a backend that quietly drops a setting
        must say so here. The default applies everything as given.
        """
        return request

    def set_console_progress(self, enabled: bool) -> None:
        """Show or hide the engine's *own* console progress bar. Optional.

        Only a caller that renders progress itself on the same terminal needs
        this - the CLI, whose status line otherwise shares a line with the
        engine's bar and is overwritten by it. The server never calls it: its
        progress goes to the browser over SSE, and the bar in the log is
        independently useful. Default is a no-op, so a backend that prints
        nothing can ignore it.
        """
