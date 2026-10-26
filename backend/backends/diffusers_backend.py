"""In-process backend built on HuggingFace diffusers.

Sizing note: Krea 2 is a ~13B DiT and needs ~25GB in bfloat16, which does not fit
a 16GB card. The default config quantizes the transformer and text encoders to
float8 weight-only via torchao, then leans on accelerate's model offload so only
one component is resident at a time. Peak VRAM lands near 13GB.

Nothing here is Krea-specific: the pipeline class and the set of quantizable
components are both read from the component tree's `model_index.json`, so
`LOCAL_DIFFUSION_MODEL_ID` can point at any diffusers text-to-image repo.
Illustrious's official single file is converted to a cached tree first.

`LOCAL_DIFFUSION_QUANT=nunchaku` is the other way round: the checkpoint arrives already
quantized (Nunchaku Lite, SVDQuant 4-bit weights and activations, with its own
CUDA kernels) and names its quantizer in its own `config.json`s, so this backend
passes diffusers no quantization config at all. Those kernels are published for
Linux only, which on a Windows box means running the server from WSL 2.
"""
from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import TYPE_CHECKING, Any

from .. import latent_preview, loras, refine
from ..model_selection import current_model_id, current_settings
from ..config import ATTENTION_ENVIRONMENT_WARNING, settings
from ..execution import (
    ExecutionSpec, canonical_execution, compiler_metadata, regional_candidate_reason,
    require_default_execution,
)
from ..model_defaults import INPAINT_DEFERRED, defaults_for, inpaint_family, inpaint_start
from ..seeds import wrap_seed
from . import anima_loading, loading, netayume_loading, newbie_loading, regional_compile, schedule, single_file
from .base import CancelFn, Cancelled, GenerationBackend, PreviewFn, ProgressFn

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas import GenerateRequest
    from .regional_compile import RegionalRegistration

log = logging.getLogger(__name__)

# Pipelines whose `do_classifier_free_guidance` is `guidance_scale > 0`. Of the
# 131 diffusers pipelines that take a `negative_prompt` and no `true_cfg_scale`,
# 122 gate at `> 1` (SD3, SDXL, Chroma, Lumina 2, ...) and only these guide from
# zero - so "has negative_prompt" cannot mean 0.0 on its own, as it used to.
_GUIDES_FROM_ZERO = ("Krea2Pipeline", "ZImage")

# What this backend cannot do without. `callback_on_step_end` is the only place
# the pipeline hands control back - the progress source and the mandatory
# cancellation point; `generator` is the per-image seed list every recorded seed
# depends on. A pipeline missing either (the old `callback`/`callback_steps` API,
# say) is refused at load rather than silently drawing unseeded noise under a
# recorded seed with a Cancel that never lands.
_REQUIRED_PARAMS = frozenset({"generator", "callback_on_step_end"})

# One adapter slot. diffusers can hold several and blend them, but every extra
# name has to reach `CellKey`, the embedded metadata and Reuse to be reproducible,
# and one adapter at a weight is the comparison a LoRA is bought for.
_LORA_ADAPTER = "local_diffusion"


@dataclass
class _PipelineCandidate:
    """Own unpublished or detached pipeline references until transfer/close.

    The eager loader uses this transaction too. Private preparation can attach
    its regional registration; public execution selection remains unavailable.
    """

    release: Callable[[], None]
    pipe: Any = None
    params: set[str] = field(default_factory=set)
    registration: RegionalRegistration | None = None
    execution: ExecutionSpec | None = None
    related: tuple[Any, ...] = ()
    _invalid: bool = field(default=False, init=False)

    def __enter__(self) -> _PipelineCandidate:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note(f"Discarding the pipeline candidate also failed: {type(cleanup).__name__}: {cleanup}")

    def disown(self) -> None:
        self.pipe, self.params, self.registration, self.execution, self.related = None, set(), None, None, ()

    def invalidate(self) -> None:
        # Registration can fail while backend/weights locks are held. Prevent
        # publication now; the outer owner drops references/releases afterward.
        self._invalid = True

    def close(self) -> None:
        release = self.pipe is not None or self.registration is not None or bool(self.related)
        registration = self.registration
        self.disown()
        failure: BaseException | None = None
        if registration is not None:
            try:
                registration.close()
            except BaseException as exc:
                failure = exc
        if release:
            try:
                self.release()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"Releasing the detached pipeline also failed: {type(exc).__name__}: {exc}")
        if failure is not None:
            raise failure


class DiffusersBackend(GenerationBackend):
    name = "diffusers"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._call_params: set[str] = set()
        self._regional_registration: RegionalRegistration | None = None
        self._loaded_execution: ExecutionSpec | None = None
        # The adapter currently attached to the live pipeline, as
        # (name, weight). Hot-swapped between cells rather than baked into
        # `load()`: diffusers strips and re-attaches the offload hooks around
        # the injection itself, so there is no "adapters must precede
        # enable_model_cpu_offload" rule to obey - and a load path driven from
        # the job worker is the one thing `main.ensure_loaded` exists to stop.
        self._lora: tuple[str, float] | None = None
        # The last generate()'s (allocated, reserved) peak bytes, for
        # `last_peak()`; cleared as each call begins.
        self._peak: tuple[int, int] | None = None
        # What an adapter is matched against, as (source, denoiser): the live
        # pipeline's once there is one, the checkpoint's configs on the meta
        # device before that, and None where neither could be read. The same
        # map either way - the configs are what the live one was built from -
        # so the live answer replaces the configs' once and is kept across an
        # unload of the same checkpoint. A switch invalidates it by public id.
        self._denoiser_cache: tuple[str, "loras.Denoiser | None"] | None = None
        self._denoiser_model_id: str | None = None
        # The refine pass. `_refine_class` is (pipeline class, its
        # image-to-image sibling or None), asked of diffusers once per class;
        # `_refine_pipe` is that sibling built over the live pipeline's own
        # components on the first refine, with its call's parameter names.
        # It holds the same modules the pipeline does, so it goes when the
        # pipeline goes - or an unload would free nothing.
        self._refine_class: tuple[type, type | None] | None = None
        self._refine_pipe: Any = None
        self._refine_params: set[str] = set()
        # The inpaint class, the same way: asked once per pipeline class,
        # built over the same components on the first inpaint, and dropped
        # with the pipeline.
        self._inpaint_class: tuple[type, type | None] | None = None
        self._inpaint_pipe: Any = None
        self._inpaint_params: set[str] = set()
        # diffusers draws its own tqdm bar per denoising loop. Kept on by
        # default; the CLI turns it off because it renders its own status line.
        self._console_progress = True

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        # No public loader or selection can request a nonempty recipe yet.
        self._load_prepared(None)

    def _load_prepared(self, execution: ExecutionSpec | None) -> None:
        """Private preparation path, also used by every ordinary eager load.

        A supplied recipe prepares ownership only; public generation continues
        to refuse every nonempty request and loaded execution specification.
        """
        execution = canonical_execution(execution)
        # The candidate exits after the backend/weights locks, including a
        # failed placement or late preparation. Publication disowns it once.
        with _PipelineCandidate(self._release, execution=execution) as candidate, self._lock, ExitStack() as held:
            if self._pipe is not None:
                if canonical_execution(self._loaded_execution) != execution:
                    raise RuntimeError("unload the current pipeline before preparing another execution recipe")
                return
            selected = current_settings(settings)
            checkpoint_id = selected.model_id
            if execution is not None:
                if ATTENTION_ENVIRONMENT_WARNING is not None:
                    raise ValueError("Regional preparation requires native attention without a nonnative external override.")
                reason = regional_candidate_reason(
                    execution, compiler_metadata(), model_id=checkpoint_id,
                    quantization=selected.quantization, offload=selected.offload, device=selected.device,
                )
                if reason is not None:
                    raise ValueError(reason)
            # No meta-device build beside the load (`loading.weights_loading`):
            # the adapter fit's patches every module's parameters process-wide.
            # The conversions below run inside the hold.
            held.enter_context(loading.weights_loading())
            import torch
            from diffusers.pipelines.pipeline_utils import DiffusionPipeline

            # Offload resolves its own accelerator inside diffusers, so it needs
            # CUDA whatever LOCAL_DIFFUSION_DEVICE says; the check has to see both, or
            # `LOCAL_DIFFUSION_DEVICE=cpu` sailed past it and died after the download in
            # `enable_model_cpu_offload` with a message naming neither variable.
            needs_cuda = (
                selected.offload != "none" or torch.device(selected.device).type == "cuda"
            )
            if needs_cuda and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available to PyTorch. Model offload needs an accelerator; "
                    "to run on the CPU set LOCAL_DIFFUSION_OFFLOAD=none and LOCAL_DIFFUSION_DEVICE=cpu."
                )

            # DiffusionPipeline resolves the concrete class from model_index.json,
            # so this works for Krea2Pipeline, FluxPipeline, and anything else.
            log.info("loading %s (quant=%s)", checkpoint_id, selected.quantization)
            # Single-file catalogue entries are converted once on CPU. Keep
            # the official id in settings, status and saved reproduction data.
            source_id = anima_loading.resolve_model(checkpoint_id, selected.quantization)
            source_id = netayume_loading.resolve_model(source_id, selected.quantization)
            source_id = newbie_loading.resolve_model(source_id, selected.quantization)
            source_id = single_file.resolve_model(source_id, selected.quantization)
            model_id = loading.resolve_model(source_id, quantization=selected.quantization)
            load_settings = replace(selected, model_id=model_id)
            kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
            quant = loading.quantization_config(load_settings)
            if quant is not None:
                kwargs["quantization_config"] = quant
            # What the pipeline lists but this backend never runs stays on the
            # Hub: `pe=None` is how diffusers is told, and the concrete class
            # prunes the folder from the download as well as from the load -
            # the base class, resolving the same name from the same index,
            # would download and load it (see loading.UNUSED_COMPONENTS).
            loader: Any = DiffusionPipeline
            if anima_loading.matches(checkpoint_id):
                loader = anima_loading.pipeline_class(checkpoint_id)
            elif checkpoint_id.lower() == newbie_loading.MODEL_ID.lower():
                from .newbie_pipeline import NewbiePipeline

                loader = NewbiePipeline
            class_name, unused = loading.unused_components(model_id)
            if unused:
                import diffusers

                assert class_name is not None, "excluded components require their concrete pipeline class"
                loader = getattr(diffusers, class_name, None) or DiffusionPipeline
                log.info("leaving out %s via %s", ", ".join(sorted(unused)), loader.__name__)
                kwargs.update(unused)

            # Pinned repos download only their component tree, after the
            # quantizer checks above have had a chance to refuse the pairing.
            model_id = loading.resolve_model(source_id, weights=True)
            # An override can already own loaded weights before a complete
            # pipeline exists. Keep those references in the transaction too.
            candidate.related = tuple(loading.component_overrides(checkpoint_id, quant, resolved_model_id=model_id).items())
            kwargs.update(candidate.related)
            try:
                with loading.checkpoint_files_released(quant is not None):
                    candidate.pipe = loader.from_pretrained(model_id, **kwargs)
            finally:
                # Also covers a failure entering the shard-release context.
                # The candidate owns overrides, not a retained load frame.
                kwargs.clear()
            log.info("resolved pipeline class: %s", type(candidate.pipe).__name__)

            loading.place_pipeline(candidate.pipe, selected)

            # VAE decode at 2K is the other memory spike; slice and tile it. Most
            # pipelines expose the switches themselves; FLUX.2 klein's does not,
            # but its VAE has them, so fall through to the component.
            for opt, on_vae in (("enable_vae_slicing", "enable_slicing"), ("enable_vae_tiling", "enable_tiling")):
                try:
                    fn = getattr(candidate.pipe, opt, None) or getattr(getattr(candidate.pipe, "vae", None), on_vae, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:  # noqa: BLE001 - purely opportunistic
                            log.debug("%s unavailable", opt)
                finally:
                    fn = None  # no bound-method reference in a failed load frame

            params = set(inspect.signature(candidate.pipe.__call__).parameters)
            missing = sorted(_REQUIRED_PARAMS - params)
            if missing:
                raise RuntimeError(
                    f"{type(candidate.pipe).__name__}.__call__ takes no {' or '.join(missing)}; this "
                    "backend needs the current diffusers callback and generator API, or "
                    "seeds would reproduce nothing and Cancel could never land."
                )
            self._apply_console_progress(candidate.pipe)
            schedule.warn_about_post_processing(candidate.pipe)
            candidate.params = params
            if execution is not None:
                try:
                    candidate.registration = regional_compile.register_regions(
                        getattr(candidate.pipe, "transformer", None), discard_on_failure=candidate.invalidate,
                    )
                    if type(candidate.registration) is not regional_compile.RegionalRegistration:
                        raise RuntimeError("regional preparation did not return its owned registration")
                    candidate.registration.require(getattr(candidate.pipe, "transformer", None))
                except BaseException:
                    candidate.invalidate()
                    raise
            # Publication is last, after the weights hold has exited too.
            held.close()
            log.info("model ready: %s", type(candidate.pipe).__name__)
            self._publish_candidate_locked(candidate)

    def _publish_candidate_locked(self, candidate: _PipelineCandidate) -> None:
        """Transfer a fully prepared candidate without replacing working weights."""
        if candidate._invalid:
            raise RuntimeError("cannot publish an invalidated pipeline candidate")
        if self._pipe is not None or self._regional_registration is not None or self._loaded_execution is not None:
            raise RuntimeError("cannot publish a pipeline over working weights")
        if candidate.pipe is None:
            raise RuntimeError("cannot publish an empty pipeline candidate")
        execution = canonical_execution(candidate.execution)
        regional = execution is not None and execution.compilation is not None
        if regional != (candidate.registration is not None):
            raise ValueError("a regional pipeline and its registration must transfer together")
        params = set(candidate.params)
        self._regional_registration = candidate.registration
        self._loaded_execution = execution
        self._call_params = params
        self._pipe = candidate.pipe
        candidate.disown()

    def _detach_pipeline_locked(self) -> _PipelineCandidate:
        """Drop every shared-pipeline reference together; retain compatibility caches."""
        candidate = _PipelineCandidate(
            self._release, self._pipe, self._call_params, self._regional_registration,
            self._loaded_execution,
            tuple(pipe for pipe in (self._refine_pipe, self._inpaint_pipe) if pipe is not None),
        )
        self._pipe, self._call_params = None, set()
        self._regional_registration, self._loaded_execution = None, None
        self._lora = None
        self._refine_pipe, self._refine_params = None, set()
        self._inpaint_pipe, self._inpaint_params = None, set()
        return candidate

    def _invoke_pipeline(self, runner: Any, kwargs: dict[str, Any]) -> Any:
        """One actual call boundary; eager calls never import compiler policy."""
        execution = canonical_execution(self._loaded_execution)
        registration = self._regional_registration
        if execution is None:
            if registration is not None:
                raise RuntimeError("an unrecorded regional registration cannot execute as Eager")
            return runner(**kwargs)
        if (execution.compilation != "regional-v1" or execution.attention_backend is not None
                or execution.step_cache is not None):
            raise ValueError("this private call boundary supports only the initial regional recipe")
        if ATTENTION_ENVIRONMENT_WARNING is not None:
            raise ValueError("Regional execution requires native attention without a nonnative external override.")
        if type(registration) is not regional_compile.RegionalRegistration:
            raise RuntimeError("the regional pipeline has no valid owned registration")
        if runner is not self._pipe or self._lora is not None or kwargs.get("num_images_per_prompt") != 1:
            raise ValueError("the first regional call needs its base pipeline, no LoRA and one image")
        registration.require(getattr(runner, "transformer", None))
        with regional_compile.strict_compiler_policy():
            return runner(**kwargs)

    def _require_execution(self, request: GenerateRequest) -> None:
        require_default_execution(request.execution)
        # Private prepared state cannot make an ordinary Eager request run
        # compiled code, even if it bypassed the closed selection surface.
        require_default_execution(self._loaded_execution)
        if self._regional_registration is not None:
            raise ValueError("Regional execution is not available in this build; unload this pipeline.")

    def prompt_expansion_supported(self) -> bool:
        pipe = self._pipe
        if pipe is None:
            # Enough to expose the button before a load. Once loaded, the
            # method and call signature are the authoritative answer.
            return "ernie-image" in current_model_id(settings).lower().replace("_", "-")
        return callable(getattr(pipe, "_enhance_prompt_with_pe", None)) and "use_pe" in self._call_params

    def prompt_expander_status(self) -> dict | None:
        """Whether the rewriter is registered on the live pipeline, and what a
        first Expand downloads where its weights are not on disk: the 7.7 GB
        the page should name before the press that starts it."""
        if not self.prompt_expansion_supported():
            return None
        pipe = self._pipe
        loaded = pipe is not None and getattr(pipe, "pe", None) is not None
        fetch = []
        if not loaded and not loading.prompt_expander_cached(current_model_id(settings)):
            fetch.append(("ERNIE's prompt rewriter", loading.PROMPT_EXPANDER_BYTES))
        return {"loaded": loaded, "fetch": fetch}

    def expand_prompt(
        self, prompt: str, width: int, height: int, should_cancel: CancelFn | None = None,
    ) -> str:
        """Run ERNIE's rewriter once and return the text the image will record."""
        def check() -> None:
            if should_cancel is not None and should_cancel():
                raise Cancelled("prompt expansion stopped")

        with self._lock:
            check()
            pipe = self._pipe
            if pipe is None:
                raise RuntimeError("the image model is not loaded")
            enhance = getattr(pipe, "_enhance_prompt_with_pe", None)
            if not callable(enhance) or "use_pe" not in self._call_params:
                raise RuntimeError(f"{type(pipe).__name__} has no prompt expander")
            if getattr(pipe, "pe", None) is None or getattr(pipe, "pe_tokenizer", None) is None:
                components = loading.prompt_expander_components(current_model_id(settings))
                # The synchronous download/load may finish into the cache;
                # an abandoned request must not start the GPU rewrite after it.
                check()
                pipe.register_modules(**components)
                # The offload chain was built while PE was absent. Rebuild it
                # so invoking pe.generate moves that model to the chosen card
                # and back just like the image pipeline's other components.
                loading.place_pipeline(pipe, current_settings(settings))
            try:
                check()
                expanded = enhance(prompt, pipe._execution_device, width=width, height=height)
            finally:
                free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
                if callable(free_hooks):
                    free_hooks()
            check()
            return str(expanded).strip()

    def _apply_console_progress(self, pipe: Any) -> None:
        fn = getattr(pipe, "set_progress_bar_config", None)
        if callable(fn):
            fn(disable=not self._console_progress)

    def set_console_progress(self, enabled: bool) -> None:
        # Settable before the load as well as after it, so a CLI can ask for
        # silence up front rather than racing the first generation.
        self._console_progress = enabled
        if self._pipe is not None:
            self._apply_console_progress(self._pipe)
        for runner in (self._refine_pipe, self._inpaint_pipe):
            if runner is not None:
                self._apply_console_progress(runner)

    def unload(self) -> None:
        with self._lock:
            candidate = self._detach_pipeline_locked()
        candidate.close()

    def try_unload(self) -> bool:
        """Unload unless `load()` holds the lock, without waiting for it.

        `load()` holds `_lock` across the whole of `from_pretrained`, so a
        blocking unload during a download waits the download out and then
        throws away what it built. At shutdown that wait froze the event loop;
        from the Unload button it is the slowest possible way of doing nothing.
        """
        if not self._lock.acquire(blocking=False):
            return False
        try:
            candidate = self._detach_pipeline_locked()
        finally:
            self._lock.release()
        candidate.close()
        return True

    @staticmethod
    def _release() -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def free_cache(self) -> None:
        # The same call unload makes after dropping the pipeline; here the
        # pipeline stays. Under model offload nearly everything the card holds
        # between cells is allocator reserve, so this is most of the VRAM.
        self._release()

    def is_ready(self) -> bool:
        return self._pipe is not None

    # ------------------------------------------------------------- generation

    def _mu_supported(self, pipe: Any) -> bool:
        """Can a requested mu reach this pipeline's schedule? `schedule.mu_supported`."""
        return schedule.mu_supported(pipe, self._call_params)

    def applied(self, request: "GenerateRequest") -> "GenerateRequest":
        """What this pipeline will run, from its parameter list once loaded.

        Before a load the checkpoint profile's guess stands in, which is what
        `/api/status` serves the UI in the same state - so the resume prune and
        the row a job later writes agree whether or not the weights were resident
        when the grid was previewed.
        """
        self._require_execution(request)
        if self._pipe is None:
            profile = defaults_for(current_model_id(settings))
            return request.applied(
                negative_prompt_supported=profile["negative_prompt_supported"],
                cfg_threshold=profile["cfg_threshold"],
                mu_supported=profile["mu_supported"],
                true_cfg_supported=profile["true_cfg_supported"],
                reference_supported=profile["reference_supported"],
                # Not a guess even here: whether an adapter can be loaded is a
                # fact about this installation, not about the checkpoint.
                lora_supported=self.lora_supported(),
                refine_supported=profile["refine_supported"],
                inpaint_supported=profile["inpaint_supported"],
            )
        threshold = self.cfg_threshold()
        assert threshold is not None, "a loaded pipeline supplies its guidance threshold"
        return request.applied(
            negative_prompt_supported="negative_prompt" in self._call_params,
            cfg_threshold=threshold,
            mu_supported=self._mu_supported(self._pipe),
            true_cfg_supported=self._true_cfg_supported(),
            reference_supported=self._reference_supported(),
            lora_supported=self.lora_supported(),
            refine_supported=self._refine_supported(),
            inpaint_supported=self._inpaint_supported(),
        )

    def _reference_supported(self) -> bool:
        """Does this pipeline condition on reference images?

        `Flux2KleinPipeline.__call__` takes `image` as its first parameter,
        resizes each reference to 1 MP, VAE-encodes it and concatenates the
        tokens onto the latent sequence at every step. Nothing else in the
        catalogue has the parameter, and handing one to a pipeline that has not
        got it would silently drop the whole point of the request.
        """
        return "image" in self._call_params

    def _refine_supported(self) -> bool:
        """Can this pipeline run a refine pass?

        It can where diffusers ships an image-to-image class beside the
        loaded pipeline's own (`refine.sibling`), since that class runs over
        the components already here. The class is the whole answer, so it is
        asked once per pipeline class and kept.
        """
        pipe = self._pipe
        if pipe is None:
            return False
        if self._refine_class is None or self._refine_class[0] is not type(pipe):
            self._refine_class = (type(pipe), refine.sibling(type(pipe)))
        return self._refine_class[1] is not None

    def _refiner(self, pipe: Any) -> Any:
        """The image-to-image sibling over the live pipeline's components.

        Built on the first refine and kept: a second pipeline object costs
        nothing - it registers the modules the first one holds, hooks and
        adapters included - but building it per cell would re-run a
        constructor for every image of a strength sweep. Its scheduler is the
        same object too, which is what lets `schedule.forced_mu` on the
        loaded pipeline reach a refine pass.
        """
        if self._refine_pipe is None:
            if not self._refine_supported():
                raise RuntimeError(
                    f"{type(pipe).__name__} has no image-to-image counterpart to refine with"
                )
            sibling = self._refine_class
            assert sibling is not None and sibling[1] is not None  # The capability check resolved it.
            runner = refine.build(pipe, sibling[1])
            self._apply_console_progress(runner)
            self._refine_params = set(inspect.signature(runner.__call__).parameters)
            self._refine_pipe = runner
            log.info(
                "refine pass: %s over %s's components", type(runner).__name__, type(pipe).__name__
            )
        return self._refine_pipe

    def _inpaint_supported(self) -> bool:
        """Can this pipeline run an inpaint?

        Where diffusers ships an inpaint class beside the loaded pipeline's
        (`refine.sibling(..., "Inpaint")`) *and* the checkpoint's profile
        enables it. The library exports one for families this app has not
        enabled yet (`INPAINT_DEFERRED`), and a class alone is not the
        decision: an inpaint there is refused, never run unmeasured. The
        loaded class must also be the profile's own `inpaint_family`: a
        hand-entered id the catch-all profile takes (FLUX.1 Kontext, which
        loads as `FluxKontextPipeline` beside its own inpaint class) runs a
        class no one measured, so it is refused too.
        """
        pipe = self._pipe
        if pipe is None:
            return False
        if self._inpaint_class is None or self._inpaint_class[0] is not type(pipe):
            self._inpaint_class = (type(pipe), refine.sibling(type(pipe), "Inpaint"))
        return (
            self._inpaint_class[1] is not None
            and type(pipe).__name__ not in INPAINT_DEFERRED
            and bool(defaults_for(current_model_id(settings))["inpaint_supported"])
            and type(pipe).__name__ == inpaint_family(current_model_id(settings))
        )

    def _inpainter(self, pipe: Any) -> Any:
        """The inpaint class over the live pipeline's components, as `_refiner`."""
        if self._inpaint_pipe is None:
            if not self._inpaint_supported():
                raise RuntimeError(f"{type(pipe).__name__} has no inpaint pass in this app")
            sibling = self._inpaint_class
            assert sibling is not None and sibling[1] is not None  # The capability check resolved it.
            runner = refine.build(pipe, sibling[1])
            self._apply_console_progress(runner)
            self._inpaint_params = set(inspect.signature(runner.__call__).parameters)
            self._inpaint_pipe = runner
            log.info("inpaint: %s over %s's components", type(runner).__name__, type(pipe).__name__)
        return self._inpaint_pipe

    def inpaint_strength(self) -> float | None:
        """Where an inpaint's Strength starts on the loaded pipeline, or None.

        The profile's own measured start where the class's default was
        measured as the wrong one (`model_defaults.inpaint_start`, FLUX.1's
        0.95), else the loaded inpaint class's own `strength` default - the
        library's number, not the checkpoint's (`refine.library_strength`).
        """
        if not self._inpaint_supported():
            return None
        start = inpaint_start(current_model_id(settings))
        if start is not None:
            return start
        sibling = self._inpaint_class
        assert sibling is not None and sibling[1] is not None  # Inpainting is supported above.
        return refine.library_strength(sibling[1])

    def start_sigma(self, request: "GenerateRequest") -> float | None:
        """How much noise this refine pass starts at, read off the schedule.

        `schedule.start_sigma` over the loaded pipeline and its sibling class,
        at the pass's start index (`refine.start_index`, the profile's, which
        is None for a checkpoint the table does not know). None before a
        load, without a sibling, on a scheduler that is not flow matching,
        and where reading it fails: it is the preview's advice, and a
        preview it broke would refuse the pass. An inpaint (or an outpaint, at
        its padded size) asks its own class's recipe in
        `schedule.START_RECIPES`: Z-Image's has one; SDXL's scheduler is not
        flow matching, so no answer there.
        """
        pipe = self._pipe
        if pipe is None or request.refine is None or request.strength is None:
            return None
        inpaint = request.inpainting
        steps = request.steps
        assert steps is not None, "the start-sigma request has its applied step count"
        start = refine.start_index(
            steps, request.strength, current_model_id(settings), inpaint
        )
        supported = self._inpaint_supported() if inpaint else self._refine_supported()
        if start is None or not supported:
            return None
        counterpart = self._inpaint_class if inpaint else self._refine_class
        assert counterpart is not None and counterpart[1] is not None  # Supported by the resolved class.
        sibling = counterpart[1]
        try:
            return schedule.start_sigma(pipe, sibling, request, start)
        except Exception:  # noqa: BLE001 - advice: no answer rather than a refused preview
            log.debug("no start sigma for this pass", exc_info=True)
            return None

    def lora_unavailable_reason(self) -> str | None:
        """Why an adapter cannot be applied here, or None when one can.

        Answerable before the load, because the two things that decide it are
        the environment's: peft, and a base whose linears an adapter can attach
        to. Nunchaku's are not - the checkpoint arrives quantized to 4-bit
        SVDQuant with its own kernels, and the modules PEFT would wrap are not
        `nn.Linear` at all - so it is refused the way `loading.nunchaku_preflight`
        refuses, up front and by name, rather than at the first cell.
        """
        if current_settings(settings).quantization == loading.NUNCHAKU:
            return (
                "LOCAL_DIFFUSION_QUANT=nunchaku loads a checkpoint that is already quantized to "
                "4-bit SVDQuant, whose linear layers are not the ones a LoRA attaches "
                "to; run the same checkpoint under fp8, int8 or none to use adapters"
            )
        refusal = loading.lora_refusal(current_model_id(settings))
        if refusal is not None:
            return refusal
        if not loading.importable("peft"):
            return "LoRA adapters need the peft package; `uv sync` installs it"
        if self._pipe is not None and not hasattr(self._pipe, "load_lora_weights"):
            return f"{type(self._pipe).__name__} has no LoRA loader"
        return None

    def lora_supported(self) -> bool:
        return self.lora_unavailable_reason() is None

    def _denoiser(self) -> "loras.Denoiser | None":
        """What an adapter must fit: the live denoiser, else the configs'.

        None when neither can be read - a gated repo without a token, a
        single-file checkpoint not yet converted, a load under way (the
        meta-device build waits for none: `loading.empty_weights`) - and then the fit goes
        unchecked rather than refused: the load that follows either fixes
        the answer or fails with the real reason. An unanswered question is
        asked again on the next call; an answered one is kept.
        """
        model_id = current_model_id(settings)
        if self._denoiser_model_id != model_id:
            self._denoiser_cache = None
            self._denoiser_model_id = model_id
        pipe = self._pipe
        cached = self._denoiser_cache
        if pipe is not None:
            if cached is None or cached[0] != "live":
                self._denoiser_cache = cached = ("live", self._live_denoiser(pipe))
        elif cached is None or cached[1] is None:
            self._denoiser_cache = cached = ("configs", self._configured_denoiser())
        return cached[1]

    @staticmethod
    def _live_denoiser(pipe: Any) -> "loras.Denoiser | None":
        # The pipeline class names its denoiser attribute for its LoRA loader:
        # `transformer_name` on the transformer families, `unet_name` on SDXL.
        component = getattr(pipe, "transformer_name", None) or getattr(pipe, "unet_name", None)
        if not component:
            return None
        module = getattr(pipe, component, None)
        if module is None:
            return None
        try:
            return loras.Denoiser.of(type(pipe), component, module)
        except Exception:  # noqa: BLE001 - a pipeline this cannot read is one the fit skips
            log.debug("could not map %s's %s", type(pipe).__name__, component, exc_info=True)
            return None

    @staticmethod
    def _configured_denoiser() -> "loras.Denoiser | None":
        # The same resolution `load()` makes ahead of the weights, minus the
        # one step that must never run from a listing: a single-file
        # checkpoint is read from its cached conversion or not at all.
        model_id = current_model_id(settings)
        try:
            source_id = (
                anima_loading.converted_tree(model_id)
                or netayume_loading.converted_tree(model_id)
                or single_file.converted_tree(model_id)
                or model_id
            )
            return loading.denoiser(loading.resolve_model(source_id))
        except Exception as exc:  # noqa: BLE001 - unanswered, not refused
            log.info("adapter fit unchecked until the load: %s", exc)
            return None

    def lora_mismatch(self, name: str) -> str | None:
        denoiser = self._denoiser()
        return None if denoiser is None else loras.mismatch(name, denoiser)

    def _apply_lora(self, pipe: Any, request: "GenerateRequest") -> None:
        """Bring the live pipeline to the adapter this cell asks for.

        Cheap between cells and that is the point: a weight change is a
        `set_adapters` on modules already injected, so `0-1x0.25` costs one
        load and four re-weightings rather than five reloads. The state is
        tracked here rather than read back off the pipeline because a failed
        injection must leave the object clean *and* this record of it honest -
        otherwise the next cell believes an adapter is attached that is not.
        """
        wanted = (request.lora, round(request.lora_scale, 6)) if request.lora else None
        if wanted == self._lora:
            return
        if self._lora is not None and (wanted is None or wanted[0] != self._lora[0]):
            pipe.unload_lora_weights()
            self._lora = None
        if wanted is None:
            return
        try:
            if self._lora is None:
                log.info("loading LoRA %s", wanted[0])
                pipe.load_lora_weights(str(loras.path_for(wanted[0])), adapter_name=_LORA_ADAPTER)
            pipe.set_adapters([_LORA_ADAPTER], [wanted[1]])
        except Exception:
            # Half-attached is worse than not attached: the cell fails either
            # way, and the rest of the night must not run under an adapter
            # nothing recorded.
            self._lora = None
            try:
                pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001 - already failing; do not mask it
                log.debug("could not unload the half-applied adapter", exc_info=True)
            raise
        self._lora = wanted

    def _reference_images(self, ids: list[int]) -> list["Image"]:
        """The reference PNGs as images, in the order the request gave them.

        A local import: this module imports `Image` as the *class* for typing
        only, and the opener is on the module. Decoded here and the file handle
        closed, because the pipeline holds them across the whole denoise.
        """
        from PIL import Image as PILImage

        from .. import storage

        images: list[Image] = []
        for path in storage.reference_paths(ids):
            with PILImage.open(path) as handle:
                images.append(handle.convert("RGB"))
        return images

    def _true_cfg_supported(self) -> bool:
        """Does this pipeline keep real CFG apart from its guidance value?

        FLUX.1's `guidance_scale` is an input the distilled model was trained
        on, and `true_cfg_scale` is the classic uncond pass; on every other
        pipeline here `guidance_scale` is that pass. The parameter's presence
        is the whole answer, and it is what `applied()` keys the recorded
        `true_cfg` on.
        """
        return "true_cfg_scale" in self._call_params

    def generate(
        self,
        request: "GenerateRequest",
        seed: int,
        on_progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
        on_preview: PreviewFn | None = None,
    ) -> list["Image"]:
        # Reset before anything can allocate - the LoRA swap and the Anime6B
        # pre-pass count - and read after the last GPU work. Set only on a
        # return, so a cancelled or failed cell leaves no peak for the next.
        self._peak = None
        self._require_execution(request)
        device = self._peak_device() if self._pipe is not None else None
        if device is not None:
            try:
                import torch

                torch.cuda.reset_peak_memory_stats(device)
            except Exception:  # noqa: BLE001 - a measurement never fails a cell
                device = None
        images = self._generate_images(request, seed, on_progress, should_cancel, on_preview)
        if device is not None:
            try:
                import torch

                self._peak = (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
            except Exception:  # noqa: BLE001 - a measurement never fails a cell
                self._peak = None
        return images

    def last_peak(self) -> tuple[int, int] | None:
        return self._peak

    def _peak_device(self) -> int | None:
        """The CUDA index the weights run on, by `describe()`'s rule, or None."""
        try:
            import torch

            selected = current_settings(settings)
            device = torch.device(selected.device)
            if torch.cuda.is_available() and (device.type == "cuda" or selected.offload != "none"):
                return device.index if device.type == "cuda" and device.index is not None else 0
        except Exception:  # noqa: BLE001 - a measurement never fails a cell
            pass
        return None

    def _generate_images(
        self,
        request: "GenerateRequest",
        seed: int,
        on_progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
        on_preview: PreviewFn | None = None,
    ) -> list["Image"]:
        self._require_execution(request)
        if self._pipe is None:
            # Loading happens in exactly one place - main.ensure_loaded() - which
            # holds a lock across the whole load. Loading here instead would do
            # it on the generation thread while it holds the job gate, which is
            # the second load path that rule exists to prevent.
            raise RuntimeError("model is not loaded")
        import torch

        pipe = self._pipe
        if getattr(request, "upscaler", None) is not None:
            self.check_refine_supported(request)
        # Run what will be recorded: the negative prompt where the pipeline can
        # apply it, mu where it can take one. Anything dropped here is dropped
        # from the row too, since the callers save `applied()` as well.
        if request.negative_prompt and "negative_prompt" not in self._call_params:
            log.info("this pipeline takes no negative prompt; ignoring it")
        request = self.applied(request)
        # A refine pass runs the pipeline's image-to-image sibling, built over
        # the same components; everything below hands `runner` what it would
        # have handed `pipe`. `applied()` has already dropped the source where
        # there is no sibling, so a source here means there is one.
        # An inpaint is a refine pass with a mask, through the inpaint class
        # instead; `applied()` has dropped a mask the pipeline cannot take
        # along with its source, so a mask here means there is one.
        refining = request.refine is not None
        # An outpaint is an inpaint over a padded copy of its source.
        inpainting = request.inpainting
        if inpainting:
            runner, params = self._inpainter(pipe), self._inpaint_params
        elif refining:
            runner, params = self._refiner(pipe), self._refine_params
        else:
            runner, params = pipe, self._call_params
        total = request.steps
        assert total is not None, "generation runs the request's applied step count"
        # Before the cancel check and the kwargs: a swap that fails must fail
        # the cell, not run it under whatever adapter the last cell left on.
        # On the loaded pipeline either way: the adapter goes into the
        # denoiser, which the sibling shares.
        self._apply_lora(pipe, request)

        if should_cancel is not None and should_cancel():
            raise Cancelled

        # `callback_on_step_end` is the only place the pipeline hands control back
        # mid-generation, so it doubles as the cancellation point: raising here
        # unwinds out of the denoising loop. The step is already finished by then,
        # which is why a cancel lands within one step rather than instantly.
        # The preview sits in its own try/except and switches itself off on
        # the first failure: an exception out of this callback is how Cancelled
        # ends a job, and a wrong-shaped latent must cost a picture, not a night.
        preview = {"on": on_preview is not None, "last": 0.0}
        # Where a refine batch is: its images are denoised one call each (see
        # `_refine`), so progress counts through all of them rather than
        # starting over per image. A text-to-image batch is one call.
        batch = {"index": 0, "count": request.num_images if refining else 1}

        def _step_end(_pipe, step: int, _timestep, cb_kwargs):
            # A refine pass runs the part of the schedule its strength leaves,
            # and how that rounds is the pipeline's own - SDXL and the flow
            # pipelines disagree by a step - so the count is read off the call.
            steps = (getattr(_pipe, "num_timesteps", None) or total) if refining else total
            done = batch["index"] * steps + step + 1
            if on_progress is not None:
                on_progress(done, steps * batch["count"])
            if preview["on"] and on_preview is not None and step + 1 < steps:
                now = time.monotonic()
                if now - preview["last"] >= latent_preview.INTERVAL_SECONDS:
                    try:
                        jpeg = self._preview(runner, cb_kwargs.get("latents"), request, step)
                    except Exception as exc:  # noqa: BLE001 - a preview is never worth a cell
                        log.warning("live preview switched off: %s", exc)
                        preview["on"] = False
                    else:
                        if jpeg is not None:
                            preview["last"] = now
                            on_preview(jpeg, done, steps * batch["count"])
            if should_cancel is not None and should_cancel():
                raise Cancelled
            return cb_kwargs

        # One generator per image, seeded seed+i to match the per-image seeds
        # storage records. A single generator draws the whole batch's noise in
        # one go, so image *i* depended on the batch size and re-running its
        # recorded seed alone reproduced something else. diffusers collapses a
        # one-element list back to the scalar path, so batches of 1 are
        # unchanged. Seed on CPU so results reproduce regardless of offload
        # placement.
        # `wrap_seed` here and at the storage call are the same helper on
        # purpose: image *i* of a batch whose base sits near 2**63 must be drawn
        # from the seed that gets written next to it.
        generator = [
            torch.Generator(device="cpu").manual_seed(wrap_seed(seed + offset))
            for offset in range(request.num_images)
        ]

        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": total,
            "guidance_scale": request.guidance_scale,
            "num_images_per_prompt": request.num_images,
            "generator": generator,
            "callback_on_step_end": _step_end,
        }
        # Where guidance switches on is a property of the pipeline, not a
        # constant. On FLUX `guidance_scale` is a distilled conditioning value
        # baked into the model, and a negative prompt is ignored until
        # `true_cfg_scale` rises above 1.0 - costing a second forward pass per
        # step. Krea 2 has no such parameter and guides from 0 up, where the
        # old `> 1.0` gate dropped the negative prompt and the pipeline then ran
        # the uncond pass against an empty string anyway. Below the threshold
        # neither pipeline encodes a negative prompt at all, so nothing is saved
        # by dropping it early - `applied()` drops it so that the recorded
        # settings are honest about what ran, and it has already decided by
        # this line - including what `true_cfg` is: on FLUX the request's own
        # separate scale, which the distilled `guidance_scale` no longer sets.
        # FLUX.2 klein takes no negative prompt at all (`negative_prompt_embeds`
        # only, and the base model's uncond pass runs against ""), so there the
        # question of a threshold never arises.
        # Reference conditioning, where the pipeline has it: klein resizes each
        # to 1 MP itself, encodes it and concatenates the tokens onto the latent
        # sequence at every step - which is why one reference costs 2-3x the
        # time rather than a single VAE encode. `applied()` has already dropped
        # them where there is no such parameter, so this is the row's story too.
        # On a refine pass `image` is the source instead - resized to the
        # requested size here, since SDXL's image-to-image call takes its size
        # from the image and no width or height at all - and `strength` is how
        # far down the schedule the pipeline noises it before denoising back.
        # An inpaint's source is not resized - it runs at its source's size,
        # which the mask was painted at. The class is handed the stored mask
        # widened to the latent cells it touches (`refine.pipeline_mask`), so
        # it repaints everywhere the composite below blends with the painted
        # grey levels - which it is given instead.
        painted = None
        inpaint_source: Image | None = None
        # An outpaint's picture is its source on the padded canvas, the new
        # area holding the fill (`refine.outpaint_source`); its mask is the
        # one the submit built and stored, read like any other.
        if inpainting:
            source_id, mask = request.refine, request.mask
            assert source_id is not None and mask is not None, "an inpaint has its source and stamped mask"
            if request.outpaint is not None:
                inpaint_source = refine.outpaint_source(
                    source_id, request.outpaint, request.width, request.height
                )
            else:
                inpaint_source = refine.inpaint_source(source_id, request.width, request.height)
            kwargs["image"] = inpaint_source
            painted = refine.mask_image(mask, request.width, request.height)
            block = getattr(runner, "vae_scale_factor", None)
            kwargs["mask_image"] = refine.pipeline_mask(
                painted, block if isinstance(block, int) and block > 0 else refine.LATENT_BLOCK
            )
            kwargs["strength"] = request.strength
        elif refining:
            source_id = request.refine
            assert source_id is not None, "a refine pass has its source"
            crop = {"crop_anchor": request.crop_anchor} if request.crop_anchor != "center" else {}
            if request.upscaler is not None:
                kwargs["image"] = refine.source_image(
                    source_id, request.width, request.height, upscaler=request.upscaler,
                    should_cancel=should_cancel, device=current_settings(settings).device,
                    **crop,
                )
            else:
                kwargs["image"] = refine.source_image(source_id, request.width, request.height, **crop)
            kwargs["strength"] = request.strength
        elif request.reference and self._reference_supported():
            kwargs["image"] = self._reference_images(request.reference)
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
            if self._true_cfg_supported():
                kwargs["true_cfg_scale"] = request.true_cfg
        # ERNIE-Image's prompt rewriter, where the pipeline has the switch:
        # off, whether or not the component was loaded. The prompt the row
        # records is the prompt the model saw.
        if "use_pe" in params:
            kwargs["use_pe"] = False
        if current_model_id(settings).lower() == netayume_loading.MODEL_ID.lower():
            kwargs["system_prompt"] = netayume_loading.SYSTEM_PROMPT
        # mu is the whole shift, not an addition to the pipeline's own. Where
        # the scheduler shifts dynamically, override the value it is given; where
        # it does not, hand it a pre-shifted schedule - minus the *static* shift
        # it will apply on top regardless, so the total is still e^mu. The
        # Refine preview's start sigma asks the same question (`start_sigma`).
        sigmas, mu = schedule.requested_shift(pipe, total, request.mu)
        if sigmas is not None:
            kwargs["sigmas"] = sigmas

        for key in set(kwargs) - params:
            log.debug("dropping unsupported pipeline arg %r", key)
            kwargs.pop(key)

        # no_grad, NOT inference_mode. `enable_model_cpu_offload()` moves each
        # component across devices from inside forward(), and torchao's quantized
        # tensor subclasses hit torch 2.13's storage-aliasing guard when that move
        # happens under inference_mode:
        #   RuntimeError: Attempted to set the storage of a tensor on device
        #   "cuda:0" to a storage on different device "cpu"
        # no_grad avoids the functionalization path and is otherwise equivalent
        # here. Verified identical outputs for both fp8 and int8.
        with torch.no_grad(), schedule.forced_mu(pipe, mu):
            if refining:
                images = self._refine(pipe, runner, kwargs, generator, batch, should_cancel)
            else:
                return list(self._invoke_pipeline(pipe, kwargs).images)
        if not inpainting:
            return images
        # The class decoded the whole picture, so its "kept" region is the
        # VAE's round trip of the source; put the source's own pixels back
        # wherever the mask does not reach (`refine.composite`).
        processor = getattr(runner, "image_processor", None)
        assert painted is not None and inpaint_source is not None  # Built together in the inpaint branch.
        return [
            refine.composite(painted, inpaint_source, image, processor)
            for image in images
        ]

    def _refine(
        self,
        pipe: Any,
        runner: Any,
        kwargs: dict[str, Any],
        generators: list[Any],
        batch: dict[str, int],
        should_cancel: CancelFn | None,
    ) -> list["Image"]:
        """Run a refine cell: one image-to-image call per image of the batch.

        One call each, with that image's own generator, because the batch
        path does not keep the per-image seed: before the noise is drawn the
        source goes through the VAE, whose encoding is *sampled*, and
        `FluxImg2ImgPipeline` draws that sample for the whole batch from the
        first generator alone. Image *i* of a batch would then differ from a
        solo run of its recorded seed, which is the one thing a recorded seed
        is for. A single call draws the sample and then the noise from one
        generator, on every pipeline, so the two cannot disagree. The cost is
        a text-encoder pass and a VAE encode per image rather than per batch.

        The sibling never had `enable_model_cpu_offload` called on it - the
        hooks are the modules' own, placed through the loaded pipeline - so
        its end-of-call `maybe_free_model_hooks` does nothing and the last
        component used stays on the card. The loaded pipeline's puts
        everything back, after a cancel or a failure too.
        """
        images: list[Image] = []
        try:
            for index, generator in enumerate(generators):
                if index and should_cancel is not None and should_cancel():
                    raise Cancelled
                batch["index"] = index
                result = self._invoke_pipeline(runner, {**kwargs, "num_images_per_prompt": 1, "generator": generator})
                images.extend(result.images)
        finally:
            free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
            if callable(free_hooks):
                free_hooks()
        return images

    def _preview(self, pipe: Any, latents: Any, request: "GenerateRequest", step: int) -> bytes | None:
        """A JPEG of the first image so far, or None when this pipeline cannot show one.

        Keyed on the VAE, since the latent-to-RGB map is the VAE's; unpacked
        through the pipeline's own helper, since the packing is the pipeline's.
        A pipeline with no fitted map or no unpacker is not an error, and says
        so once at DEBUG by switching the preview off through the caller.
        """
        if latents is None:
            raise RuntimeError("the pipeline hands the callback no latents")
        factors = latent_preview.factors_for(getattr(pipe, "vae", None))
        if factors is None:
            raise RuntimeError(f"no latent-to-RGB map for {type(getattr(pipe, 'vae', None)).__name__}")
        latent = latent_preview.unpack(pipe, latents, request.height, request.width)
        if latent is None:
            raise RuntimeError(f"{type(pipe).__name__} has no _unpack_latents")
        # The latents are at the sigma of the step just taken; the terminal 0 is
        # the last entry, so index step + 1 is always in range. A refine pass
        # starts part of the way down the schedule, at the index its pipeline
        # set on the scheduler; a text-to-image call starts at 0 or sets none.
        try:
            begin = getattr(pipe.scheduler, "begin_index", None) or 0
            sigma = float(pipe.scheduler.sigmas[begin + step + 1])
        except Exception:  # noqa: BLE001 - a scheduler without sigmas: draw it plain
            sigma = 0.0
        return latent_preview.render(latent, factors, sigma)

    # ----------------------------------------------------------------- status

    def cfg_threshold(self) -> float | None:
        """Guidance scale above which a negative prompt does anything.

        FLUX.1 bakes `guidance_scale` into the model and only runs an uncond pass
        when `true_cfg_scale > 1`; Krea 2 takes a negative prompt and guides from
        0 up; a pipeline with neither parameter - FLUX.2 klein - has the classic
        `guidance_scale > 1` gate (and, distilled, warns above it). None until a
        pipeline is loaded, since the parameter list is the only honest answer.
        """
        if self._pipe is None:
            return None
        if "true_cfg_scale" in self._call_params:
            return 1.0
        if "negative_prompt" in self._call_params and type(self._pipe).__name__.startswith(
            _GUIDES_FROM_ZERO
        ):
            return 0.0
        return 1.0

    def describe(self) -> dict:
        selected = current_settings(settings)
        info: dict[str, Any] = {
            "model_id": selected.model_id,
            "quantization": selected.quantization,
            "offload": selected.offload,
            "cfg_threshold": self.cfg_threshold(),
            "mu_supported": self._mu_supported(self._pipe) if self._pipe is not None else None,
            # None until loaded, like the threshold: the parameter list is the answer.
            "negative_prompt_supported": (
                "negative_prompt" in self._call_params if self._pipe is not None else None
            ),
            "true_cfg_supported": self._true_cfg_supported() if self._pipe is not None else None,
            "reference_supported": (
                self._reference_supported() if self._pipe is not None else None
            ),
            # The pipeline's class is the answer here: whether diffusers
            # ships an image-to-image sibling for it.
            "refine_supported": self._refine_supported() if self._pipe is not None else None,
            # The inpaint class and the profile's decision together.
            "inpaint_supported": self._inpaint_supported() if self._pipe is not None else None,
            # And where its Strength box can start: the class's own default.
            "inpaint_strength": self.inpaint_strength(),
            # Never None: peft and the quantization mode decide this one, and
            # both are known before a single weight is downloaded.
            "lora_supported": self.lora_supported(),
        }
        try:
            import torch

            # The card the weights are on, not GPU 0: with offload the
            # components live on `LOCAL_DIFFUSION_DEVICE` between moves, and a status
            # line reporting another card's memory hides a dropped setting.
            device = torch.device(selected.device)
            if torch.cuda.is_available() and (device.type == "cuda" or selected.offload != "none"):
                index = device.index if device.type == "cuda" and device.index is not None else 0
                free, total = torch.cuda.mem_get_info(index)
                info["device_name"] = torch.cuda.get_device_name(index)
                info["vram_total_mb"] = total // (1024 * 1024)
                info["vram_used_mb"] = (total - free) // (1024 * 1024)
        except Exception:  # noqa: BLE001 - status must never raise
            pass
        return info
