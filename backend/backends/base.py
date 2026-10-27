"""The contract every generation engine implements.

Keeping the API surface this small is what lets a ComfyUI-driven backend replace
the in-process diffusers one without the HTTP layer or the UI noticing.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable

from ..execution import require_default_execution
from ..instruction_edit import require_execution_available

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas import GenerateRequest
    from .regional_compile import RegionalDiagnostics

# (completed_steps, total_steps) -> None
ProgressFn = Callable[[int, int], None]
# Polled between denoising steps; True means stop.
CancelFn = Callable[[], bool]
# (jpeg_bytes, completed_steps, total_steps) -> None: a rough picture of the
# image so far, from the latents, a few times a second at most. Optional for
# a backend to produce and for a caller to take.
PreviewFn = Callable[[bytes, int, int], None]
ExecutionDiagnosticsReader = Callable[[], "RegionalDiagnostics | None"]


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
        on_preview: PreviewFn | None = None,
    ) -> list["Image"]:
        """Run one synchronous generation. Called on a worker thread.

        Poll `should_cancel` at every step boundary and raise `Cancelled` when it
        answers True. `on_preview`, when given, may be handed a small JPEG of
        the image so far between steps; a backend that cannot draw one simply
        never calls it.
        """

    @abc.abstractmethod
    def describe(self) -> dict:
        """Metadata for `/api/status`.

        Required keys, which `main.status()` indexes directly: `model_id`,
        `quantization`, `offload`. Optional, merged when present:
        `cfg_threshold`, `negative_prompt_supported`, `true_cfg_supported`,
        `reference_supported`, `refine_supported` and `inpaint_supported` (None
        until a pipeline is loaded - its parameter list, or for a refine pass
        or an inpaint its class, is the answer), `inpaint_strength` (the
        start: a profile's own measured `inpaint_start`, else the inpaint
        class's own strength default, the library's), `lora_supported` (never None: what decides it is
        the installation, not the checkpoint), `device_name`, `vram_total_mb`,
        `vram_used_mb`. Load errors are *not*
        reported here:
        `main.ensure_loaded` records them, and that is what `ModelStatus.detail`
        serves.
        """

    def applied(self, request: "GenerateRequest") -> "GenerateRequest":
        """The request as this backend will actually run it - see
        `GenerateRequest.applied`. What the callers record beside the image and
        compare against for resume, so a backend that quietly drops a setting
        must say so here. The default applies everything as given.
        """
        qualified = self.instruction_edit_qualified()
        if getattr(request, "edit", None) is not None:
            require_execution_available(request.validate_edit(), qualified=qualified)
        require_default_execution(request.execution)
        request = request.with_execution(None)
        require_default_execution(request.execution)
        if request.edit is not None:
            # The edit branch of `GenerateRequest.applied` ignores every
            # capability but this one; it binds the edit and drops the negative.
            return request.applied(
                negative_prompt_supported=False, cfg_threshold=0.0,
                instruction_edit_supported=qualified is True,
            )
        return request

    def instruction_edit_qualified(self) -> bool:
        """Whether this backend instance may load and run an instruction edit.

        The one gate: an instance answer, never a setting, an environment
        variable or a flag. The default, and the real diffusers backend until
        its card qualification, answers the literal False; only a test or
        fake stand-in opens it. Callers compare with ``is True``.
        """
        return False

    def execution_diagnostics_reader(self) -> ExecutionDiagnosticsReader | None:
        """Optional pure regional-state reader, with no model/backend reference.

        A job binds this after loading for its active cell, then freezes the
        returned value and drops the reader. Eager/default backends return None;
        calling this starts no imports, compilation, model or device work.
        """
        return None

    def execution_diagnostics(self) -> "RegionalDiagnostics | None":
        """Current immutable observation; separate from rendering permission."""
        reader = self.execution_diagnostics_reader()
        return None if reader is None else reader()

    def lora_supported(self) -> bool:
        """Can this backend apply a LoRA adapter? Optional; default no.

        Separate from `describe()` because the HTTP layer refuses a request
        naming an adapter *at submit*, with the reason - a 64-cell night must
        not discover at cell 1 that nothing here can load one.
        """
        return False

    def check_refine_supported(self, request: "GenerateRequest") -> None:
        """Refuse a pass the loaded pipeline cannot perform, before any cell.

        A profile is only a pre-load guess for an unfamiliar checkpoint id.
        Normalization must not turn an accepted refine job into direct renders.
        """
        if getattr(request, "edit", None) is not None:
            require_execution_available(request.validate_edit(), qualified=self.instruction_edit_qualified())
        if getattr(request, "upscaler", None) is not None:
            from .. import refine

            reason = refine.upscaler_reason(request, self.describe()["model_id"])
            if reason is not None:
                raise ValueError(reason)
            if self.applied(request).upscaler != request.upscaler:
                raise ValueError("the loaded pipeline cannot preserve the requested Refine upscaler")
        if request.refine is not None and self.applied(request).refine != request.refine:
            if request.inpainting:
                through = ", which an outpaint runs through" if request.outpaint is not None else ""
                raise ValueError(
                    f"{self.describe()['model_id']} has no inpaint in this app; the loaded "
                    f"pipeline has no inpaint class{through}, or its family is not enabled yet"
                )
            raise ValueError(
                f"{self.describe()['model_id']} has no refine pass in this app; "
                "the loaded pipeline has no compatible image-to-image counterpart"
            )

    def start_sigma(self, request: "GenerateRequest") -> float | None:
        """How much noise a refine pass starts at, as a fraction. Optional.

        `request` is a pass as `applied()` runs it. The answer is read off the
        loaded pipeline's own schedule for that size, steps, mu and strength,
        or None where it cannot be read without a guess - before a load, or a
        schedule whose sigmas are no noise fraction. The default cannot read
        one: the Refine preview then says nothing of it.
        """
        return None

    def last_peak(self) -> tuple[int, int] | None:
        """(allocated, reserved) bytes at most during the last `generate()`
        that returned images. Optional.

        A result saved beside the images, never part of what made them. The
        peak is the call's, so a batch's images share it. None where this
        backend cannot say - no CUDA device, a call that raised, or the
        default.
        """
        return None

    def prompt_expansion_supported(self) -> bool:
        """Whether this backend can turn a short prompt into a reviewed one.

        Kept separate from ``generate()`` deliberately: an expansion is shown
        to the user and becomes the prompt that is recorded, rather than a
        hidden rewrite performed inside an image call.
        """
        return False

    def expand_prompt(
        self, prompt: str, width: int, height: int, should_cancel: CancelFn | None = None,
    ) -> str:
        """Expand synchronously, polling cancellation between opaque stages."""
        raise RuntimeError("this checkpoint has no prompt expander")

    def prompt_expander_status(self) -> dict | None:
        """What an expansion would wait on before it runs, for `/api/status`:
        ``{"loaded": bool, "fetch": [(name, bytes), ...]}`` - whether the
        expander is in memory already, and what a first use downloads - or
        None where there is no expander. File checks only, never the network:
        the status is polled. Optional; the default has no expander.
        """
        return None

    def lora_unavailable_reason(self) -> str | None:
        """Why not, for that refusal. None when adapters can be applied."""
        return None if self.lora_supported() else "this backend cannot apply LoRA adapters"

    def lora_mismatch(self, name: str) -> str | None:
        """Why this adapter cannot attach to the running checkpoint - modules
        or widths its denoiser has not got - or None when it can, or when
        nothing here can tell. Per adapter where `lora_unavailable_reason` is
        per installation: the listing hides what does not fit and the submit
        refuses it by name. Optional; the default has no opinion.
        """
        return None

    def free_cache(self) -> None:
        """Return cached accelerator memory without unloading. Optional.

        Called at a job's pause point, between cells: under model offload the
        components are back on the CPU by then and what the card still holds
        is the caching allocator's reserve, which this hands back so another
        program can use the GPU for twenty minutes mid-night. The weights stay
        loaded; resuming costs nothing but the next cell's own transfers.
        """

    def set_console_progress(self, enabled: bool) -> None:
        """Show or hide the engine's *own* console progress bar. Optional.

        Only a caller that renders progress itself on the same terminal needs
        this - the CLI, whose status line otherwise shares a line with the
        engine's bar and is overwritten by it. The server never calls it: its
        progress goes to the browser over SSE, and the bar in the log is
        independently useful. Default is a no-op, so a backend that prints
        nothing can ignore it.
        """
