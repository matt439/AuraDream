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
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .. import latent_preview, loras
from ..config import settings
from ..model_defaults import defaults_for
from ..seeds import wrap_seed
from . import anima_loading, loading, netayume_loading, schedule, single_file
from .base import CancelFn, Cancelled, GenerationBackend, PreviewFn, ProgressFn

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas import GenerateRequest

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


class DiffusersBackend(GenerationBackend):
    name = "diffusers"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._call_params: set[str] = set()
        # The adapter currently attached to the live pipeline, as
        # (name, weight). Hot-swapped between cells rather than baked into
        # `load()`: diffusers strips and re-attaches the offload hooks around
        # the injection itself, so there is no "adapters must precede
        # enable_model_cpu_offload" rule to obey - and a load path driven from
        # the job worker is the one thing `main.ensure_loaded` exists to stop.
        self._lora: tuple[str, float] | None = None
        # What an adapter is matched against, as (source, denoiser): the live
        # pipeline's once there is one, the checkpoint's configs on the meta
        # device before that, and None where neither could be read. The same
        # map either way - the configs are what the live one was built from -
        # so the live answer replaces the configs' once and is kept across an
        # unload, the checkpoint being fixed for the life of the process.
        self._denoiser_cache: tuple[str, "loras.Denoiser | None"] | None = None
        # diffusers draws its own tqdm bar per denoising loop. Kept on by
        # default; the CLI turns it off because it renders its own status line.
        self._console_progress = True

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        with self._lock:
            if self._pipe is not None:
                return
            import torch
            from diffusers import DiffusionPipeline

            # Offload resolves its own accelerator inside diffusers, so it needs
            # CUDA whatever LOCAL_DIFFUSION_DEVICE says; the check has to see both, or
            # `LOCAL_DIFFUSION_DEVICE=cpu` sailed past it and died after the download in
            # `enable_model_cpu_offload` with a message naming neither variable.
            needs_cuda = (
                settings.offload != "none" or torch.device(settings.device).type == "cuda"
            )
            if needs_cuda and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available to PyTorch. Model offload needs an accelerator; "
                    "to run on the CPU set LOCAL_DIFFUSION_OFFLOAD=none and LOCAL_DIFFUSION_DEVICE=cpu."
                )

            # DiffusionPipeline resolves the concrete class from model_index.json,
            # so this works for Krea2Pipeline, FluxPipeline, and anything else.
            log.info("loading %s (quant=%s)", settings.model_id, settings.quantization)
            # Single-file catalogue entries are converted once on CPU. Keep
            # the official id in settings, status and saved reproduction data.
            source_id = anima_loading.resolve_model(settings.model_id, settings.quantization)
            source_id = netayume_loading.resolve_model(source_id, settings.quantization)
            source_id = single_file.resolve_model(source_id, settings.quantization)
            model_id = loading.resolve_model(source_id)
            load_settings = replace(settings, model_id=model_id)
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
            if settings.model_id.lower() == anima_loading.MODEL_ID.lower():
                from .anima_pipeline import AnimaTurboPipeline

                loader = AnimaTurboPipeline
            class_name, unused = loading.unused_components(model_id)
            if unused:
                import diffusers

                loader = getattr(diffusers, class_name, None) or DiffusionPipeline
                log.info("leaving out %s via %s", ", ".join(sorted(unused)), loader.__name__)
                kwargs.update(unused)

            # Nested repos download only their component tree, after the
            # quantizer checks above have had a chance to refuse the pairing.
            model_id = loading.resolve_model(source_id, weights=True)
            kwargs.update(loading.component_overrides(settings.model_id, quant))
            pipe = loader.from_pretrained(model_id, **kwargs)
            log.info("resolved pipeline class: %s", type(pipe).__name__)

            loading.place_pipeline(pipe, settings)

            # VAE decode at 2K is the other memory spike; slice and tile it. Most
            # pipelines expose the switches themselves; FLUX.2 klein's does not,
            # but its VAE has them, so fall through to the component.
            vae = getattr(pipe, "vae", None)
            for opt, on_vae in (("enable_vae_slicing", "enable_slicing"), ("enable_vae_tiling", "enable_tiling")):
                fn = getattr(pipe, opt, None) or getattr(vae, on_vae, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 - purely opportunistic
                        log.debug("%s unavailable", opt)

            params = set(inspect.signature(pipe.__call__).parameters)
            missing = sorted(_REQUIRED_PARAMS - params)
            if missing:
                raise RuntimeError(
                    f"{type(pipe).__name__}.__call__ takes no {' or '.join(missing)}; this "
                    "backend needs the current diffusers callback and generator API, or "
                    "seeds would reproduce nothing and Cancel could never land."
                )
            self._apply_console_progress(pipe)
            schedule.warn_about_post_processing(pipe)
            self._call_params = params
            self._pipe = pipe
            log.info("model ready: %s", type(pipe).__name__)

    def prompt_expansion_supported(self) -> bool:
        pipe = self._pipe
        if pipe is None:
            # Enough to expose the button before a load. Once loaded, the
            # method and call signature are the authoritative answer.
            return "ernie-image" in settings.model_id.lower().replace("_", "-")
        return callable(getattr(pipe, "_enhance_prompt_with_pe", None)) and "use_pe" in self._call_params

    def expand_prompt(self, prompt: str, width: int, height: int) -> str:
        """Run ERNIE's rewriter once and return the text the image will record."""
        with self._lock:
            pipe = self._pipe
            if pipe is None:
                raise RuntimeError("the image model is not loaded")
            enhance = getattr(pipe, "_enhance_prompt_with_pe", None)
            if not callable(enhance) or "use_pe" not in self._call_params:
                raise RuntimeError(f"{type(pipe).__name__} has no prompt expander")
            if getattr(pipe, "pe", None) is None or getattr(pipe, "pe_tokenizer", None) is None:
                components = loading.prompt_expander_components(settings.model_id)
                pipe.register_modules(**components)
                # The offload chain was built while PE was absent. Rebuild it
                # so invoking pe.generate moves that model to the chosen card
                # and back just like the image pipeline's other components.
                loading.place_pipeline(pipe, settings)
            try:
                expanded = enhance(prompt, pipe._execution_device, width=width, height=height)
            finally:
                free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
                if callable(free_hooks):
                    free_hooks()
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

    def unload(self) -> None:
        with self._lock:
            if self._pipe is None:
                return
            self._pipe = None
            self._call_params = set()
            # The adapter lives on the pipeline that is going away.
            self._lora = None
        self._release()

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
            loaded, self._pipe, self._call_params = self._pipe is not None, None, set()
            self._lora = None
        finally:
            self._lock.release()
        if loaded:
            self._release()
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
        if self._pipe is None:
            profile = defaults_for(settings.model_id)
            return request.applied(
                negative_prompt_supported=profile["negative_prompt_supported"],
                cfg_threshold=profile["cfg_threshold"],
                mu_supported=profile["mu_supported"],
                true_cfg_supported=profile["true_cfg_supported"],
                reference_supported=profile["reference_supported"],
                # Not a guess even here: whether an adapter can be loaded is a
                # fact about this installation, not about the checkpoint.
                lora_supported=self.lora_supported(),
            )
        return request.applied(
            negative_prompt_supported="negative_prompt" in self._call_params,
            cfg_threshold=self.cfg_threshold(),
            mu_supported=self._mu_supported(self._pipe),
            true_cfg_supported=self._true_cfg_supported(),
            reference_supported=self._reference_supported(),
            lora_supported=self.lora_supported(),
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

    def lora_unavailable_reason(self) -> str | None:
        """Why an adapter cannot be applied here, or None when one can.

        Answerable before the load, because the two things that decide it are
        the environment's: peft, and a base whose linears an adapter can attach
        to. Nunchaku's are not - the checkpoint arrives quantized to 4-bit
        SVDQuant with its own kernels, and the modules PEFT would wrap are not
        `nn.Linear` at all - so it is refused the way `loading.nunchaku_preflight`
        refuses, up front and by name, rather than at the first cell.
        """
        if settings.quantization == loading.NUNCHAKU:
            return (
                "LOCAL_DIFFUSION_QUANT=nunchaku loads a checkpoint that is already quantized to "
                "4-bit SVDQuant, whose linear layers are not the ones a LoRA attaches "
                "to; run the same checkpoint under fp8, int8 or none to use adapters"
            )
        if settings.model_id.lower() == anima_loading.MODEL_ID.lower():
            return "User LoRAs are not supported by this Anima Turbo integration"
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
        single-file checkpoint not yet converted - and then the fit goes
        unchecked rather than refused: the load that follows either fixes
        the answer or fails with the real reason. An unanswered question is
        asked again on the next call; an answered one is kept.
        """
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
        module = getattr(pipe, component, None) if component else None
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
        try:
            source_id = (
                anima_loading.converted_tree(settings.model_id)
                or netayume_loading.converted_tree(settings.model_id)
                or single_file.converted_tree(settings.model_id)
                or settings.model_id
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
        if self._pipe is None:
            # Loading happens in exactly one place - main.ensure_loaded() - which
            # holds a lock across the whole load. Loading here instead would do
            # it on the generation thread while it holds the job gate, which is
            # the second load path that rule exists to prevent.
            raise RuntimeError("model is not loaded")
        import torch

        pipe = self._pipe
        # Run what will be recorded: the negative prompt where the pipeline can
        # apply it, mu where it can take one. Anything dropped here is dropped
        # from the row too, since the callers save `applied()` as well.
        if request.negative_prompt and "negative_prompt" not in self._call_params:
            log.info("this pipeline takes no negative prompt; ignoring it")
        request = self.applied(request)
        total = request.steps
        # Before the cancel check and the kwargs: a swap that fails must fail
        # the cell, not run it under whatever adapter the last cell left on.
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

        def _step_end(_pipe, step: int, _timestep, cb_kwargs):
            if on_progress is not None:
                on_progress(step + 1, total)
            if preview["on"] and step + 1 < total:
                now = time.monotonic()
                if now - preview["last"] >= latent_preview.INTERVAL_SECONDS:
                    try:
                        jpeg = self._preview(pipe, cb_kwargs.get("latents"), request, step)
                    except Exception as exc:  # noqa: BLE001 - a preview is never worth a cell
                        log.warning("live preview switched off: %s", exc)
                        preview["on"] = False
                    else:
                        if jpeg is not None:
                            preview["last"] = now
                            on_preview(jpeg, step + 1, total)
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
            "num_inference_steps": request.steps,
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
        if request.reference and self._reference_supported():
            kwargs["image"] = self._reference_images(request.reference)
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
            if self._true_cfg_supported():
                kwargs["true_cfg_scale"] = request.true_cfg
        # ERNIE-Image's prompt rewriter, where the pipeline has the switch:
        # off, whether or not the component was loaded. The prompt the row
        # records is the prompt the model saw.
        if "use_pe" in self._call_params:
            kwargs["use_pe"] = False
        if settings.model_id.lower() == netayume_loading.MODEL_ID.lower():
            kwargs["system_prompt"] = netayume_loading.SYSTEM_PROMPT
        # mu is the whole shift, not an addition to the pipeline's own. Where
        # the scheduler shifts dynamically, override the value it is given; where
        # it does not, hand it a pre-shifted schedule - minus the *static* shift
        # it will apply on top regardless, so the total is still e^mu.
        mu = request.mu
        if mu is not None and not schedule.shifts_itself(pipe):
            import math

            kwargs["sigmas"] = schedule.shifted_sigmas(
                request.steps, mu - math.log(schedule.static_shift(pipe))
            )
            mu = None

        for key in set(kwargs) - self._call_params:
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
            result = pipe(**kwargs)
        return list(result.images)

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
        # the last entry, so index step + 1 is always in range.
        try:
            sigma = float(pipe.scheduler.sigmas[step + 1])
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
        info: dict[str, Any] = {
            "model_id": settings.model_id,
            "quantization": settings.quantization,
            "offload": settings.offload,
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
            # Never None: peft and the quantization mode decide this one, and
            # both are known before a single weight is downloaded.
            "lora_supported": self.lora_supported(),
        }
        try:
            import torch

            # The card the weights are on, not GPU 0: with offload the
            # components live on `LOCAL_DIFFUSION_DEVICE` between moves, and a status
            # line reporting another card's memory hides a dropped setting.
            device = torch.device(settings.device)
            if torch.cuda.is_available() and (device.type == "cuda" or settings.offload != "none"):
                index = device.index if device.type == "cuda" and device.index is not None else 0
                free, total = torch.cuda.mem_get_info(index)
                info["device_name"] = torch.cuda.get_device_name(index)
                info["vram_total_mb"] = total // (1024 * 1024)
                info["vram_used_mb"] = (total - free) // (1024 * 1024)
        except Exception:  # noqa: BLE001 - status must never raise
            pass
        return info
