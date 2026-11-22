"""The contract every generation engine implements.

The five abstract methods (`load`, `unload`, `is_ready`, `generate`,
`describe`) are the contract; everything else is an optional hook with a
default. Keeping the required surface that small is what lets a
ComfyUI-driven backend replace the in-process diffusers one without the HTTP
layer or the UI noticing.
"""
from __future__ import annotations

import abc
import os
import sys
from typing import TYPE_CHECKING, Any, Callable

from ..execution import require_default_execution
from ..instruction_edit import (
    LoaderChoice, edit_unavailable_reason, installed_diffusers_version, require_execution_available,
)

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..mask_crop import Crop
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


def _refine_native_observed() -> bool:
    """Read startup/current/already-imported Native facts without imports."""
    from ..config import ATTENTION_ENVIRONMENT_WARNING

    if ATTENTION_ENVIRONMENT_WARNING is not None or os.environ.get("DIFFUSERS_ATTN_BACKEND", "native") != "native":
        return False
    dispatch = sys.modules.get("diffusers.models.attention_dispatch")
    for name in ("diffusers.utils.constants", "diffusers.models.attention_dispatch"):
        if name not in sys.modules:
            continue
        module = sys.modules[name]
        if module is None or getattr(module, "DIFFUSERS_ATTN_BACKEND", None) != "native":
            return False
    if dispatch is None:
        return True  # No runtime import yet: the actual call checks again.
    try:
        active, function = dispatch._AttentionBackendRegistry.get_active_backend()
        native = dispatch.AttentionBackendName.NATIVE
        return active is native and function is dispatch._native_attention and callable(function)
    except Exception:
        return False


class Cancelled(Exception):
    """Raised out of `generate()` when the caller asked it to stop.

    A diffusion step is not interruptible from the outside, so cancellation is
    cooperative: the backend polls `should_cancel` at a step boundary and unwinds
    from there. Whatever finished before that point is still valid output.
    """


class ExpansionTooLong(Exception):
    """Raised out of `expand_prompt()` when the expander's answer ran past the
    prompt box's limit. Its message is the page's sentence (a 422 there)."""


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
        must say so here. The default refuses a nondefault execution recipe,
        clears the recipe, and binds an edit (dropping its negative); any other
        request is applied as given.
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
        variable or a flag. The default answers the literal False; the real
        diffusers backend (released after its card trial) and the test and
        fake stand-ins answer True, and the measured envelope still decides
        (``instruction_edit_refusal``). Callers compare with ``is True``.
        """
        return False

    def instruction_edit_host(self) -> tuple[str, str | None]:
        """The platform and installed diffusers version an edit would run on.

        Facts, never permission: they feed the measured envelope's check
        (``instruction_edit.loader_refusal``), which only narrows an open gate.
        """
        return sys.platform, installed_diffusers_version()

    def instruction_edit_loader(self, selected: Any) -> LoaderChoice:
        """This instance's loading facts for ``selected`` (``current_settings``)."""
        platform, diffusers_version = self.instruction_edit_host()
        return LoaderChoice.of(selected, platform=platform, diffusers_version=diffusers_version)

    def instruction_edit_refusal(self, selected: Any) -> str | None:
        """Why no instruction edit can queue here with ``selected``, or None.

        The gate, the selected checkpoint and the loader envelope, in that
        order (``edit_unavailable_reason``); a source's aspect is the caller's.
        """
        return edit_unavailable_reason(
            selected.model_id, qualified=self.instruction_edit_qualified(),
            loader=self.instruction_edit_loader(selected),
        )

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

    def refine_host(self) -> tuple[str, str | None, str | None, bool]:
        """Supplied/cached qualification facts, without a framework/device probe."""
        from .. import card_facts
        from ..config import settings
        from ..model_selection import current_settings

        selected = current_settings(settings)
        found = card_facts.observed(selected.device)
        return (sys.platform, installed_diffusers_version(),
                None if found is None else found.name, _refine_native_observed())

    def refine_refusal(self, selected: Any, request: "GenerateRequest | None" = None, *,
                       check_hardware: bool = True) -> str | None:
        """The measured Anima envelopes; other checkpoints retain their contract."""
        from .. import refine_qualification

        if request is not None and (request.refine is None or request.inpainting):
            return None
        if not refine_qualification.applies(selected.model_id):
            return None
        platform, version, gpu, native = self.refine_host()
        facts = refine_qualification.LoaderFacts.of(
            selected, platform=platform, diffusers_version=version, gpu_name=gpu, native_attention=native,
        )
        return (refine_qualification.loader_refusal(selected.model_id, facts, check_hardware=check_hardware)
                or (refine_qualification.request_refusal(selected.model_id, request) if request is not None else None))

    def tiled_refine_refusal(self, selected: Any, *, check_hardware: bool = True) -> str | None:
        """Why the measured tiled recipe cannot run on this backend."""
        from ..tiled_refine_contract import HELD

        return HELD

    def regional_prompts_refusal(self, selected: Any, *, check_hardware: bool = True) -> str | None:
        from ..spatial_conditioning import REGIONAL_UNAVAILABLE

        return REGIONAL_UNAVAILABLE

    def controlnet_qualified(self, mode: str) -> bool:
        """Whether a named ControlNet mode was qualified on this backend."""
        return False

    def controlnet_owner_refusal(self) -> str | None:
        """A ready pipeline may narrow the cold loader's capability."""
        return None

    def check_refine_supported(self, request: "GenerateRequest") -> None:
        """Refuse a pass the loaded pipeline cannot perform, before any cell.

        A profile is only a pre-load guess for an unfamiliar checkpoint id.
        Normalization must not turn an accepted refine job into direct renders.
        """
        if request.refine is not None:
            from ..config import settings
            from ..model_selection import current_settings

            reason = self.refine_refusal(current_settings(settings), request)
            if reason is not None:
                raise ValueError(reason)
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

    def last_mask_crop(self) -> "Crop | None":
        """The masked-area region the last `generate()` that returned images
        ran: its box on the source and its working size. Optional.

        A result beside the peak, never part of what made the images or of
        any key: the request's `mask_crop` spec is, and the region follows
        from it and the mask (`mask_crop.region`). None for a run without
        one, a call that raised, or the default.
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

    def prompt_expansion_refusal(self) -> str | None:
        """Why a supported expander cannot run under the current settings, in
        the page's words, answered before anything loads - or None. Optional;
        the default refuses nothing."""
        return None

    def prompt_expander_status(self) -> dict | None:
        """What an expansion would wait on before it runs, for `/api/status`:
        ``{"loaded": bool, "fetch": [(name, bytes), ...], "name": str,
        "translate": bool}`` - whether the expander is in memory already,
        what a first use downloads, the expander's name as the page says it,
        and whether its answer is translated after (ERNIE's) - or None where
        there is no expander. File checks only, never the network: the
        status is polled. Optional; the default has no expander.
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
