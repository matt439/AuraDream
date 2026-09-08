"""In-process backend built on HuggingFace diffusers.

Sizing note: Krea 2 is a ~13B DiT and needs ~25GB in bfloat16, which does not fit
a 16GB card. The default config quantizes the transformer and text encoders to
float8 weight-only via torchao, then leans on accelerate's model offload so only
one component is resident at a time. Peak VRAM lands near 13GB.

Nothing here is Krea-specific: the pipeline class and the set of quantizable
components are both read from the repo's `model_index.json`, so `KREA_MODEL_ID`
can point at FLUX.1-Krea-dev or any other diffusers text-to-image repo.

`KREA_QUANT=nunchaku` is the other way round: the checkpoint arrives already
quantized (Nunchaku Lite, SVDQuant 4-bit weights and activations, with its own
CUDA kernels) and names its quantizer in its own `config.json`s, so this backend
passes diffusers no quantization config at all. Those kernels are published for
Linux only, which on a Windows box means running the server from WSL 2.
"""
from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import logging
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..config import settings
from ..model_defaults import defaults_for
from ..seeds import wrap_seed
from .base import CancelFn, Cancelled, GenerationBackend, ProgressFn

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas import GenerateRequest

log = logging.getLogger(__name__)

# torchao weight-only modes this backend applies itself while loading.
_QUANT_BACKENDS = ("fp8", "int8")
# A checkpoint quantized before upload. `transformer/config.json` carries this
# `quant_method`, and diffusers picks the quantizer up from there on its own.
_NUNCHAKU = "nunchaku"
_NUNCHAKU_METHOD = "nunchaku_lite"
_NUNCHAKU_EXAMPLE = "lite-infer/krea-2-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"

# Keys in model_index.json worth quantizing. The VAE stays in bf16: it is small
# and quantizing it shows up as visible decode artefacts.
_QUANTIZABLE_PREFIXES = ("transformer", "unet", "text_encoder")

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


def _importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _hub_file(model_id: str, filename: str) -> str:
    """Path to one file of a checkpoint, downloading only that file.

    A local directory is resolved locally: `from_pretrained` branches on
    `os.path.isdir` itself, but the two callers here run *before* it - the
    fp8/int8 component scan and the Nunchaku preflight - and a directory id
    handed to the Hub client died with a validation error naming the model id
    and not the quantization.
    """
    import os

    if os.path.isdir(model_id):
        return os.path.join(model_id, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename)


class DiffusersBackend(GenerationBackend):
    name = "diffusers"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._call_params: set[str] = set()
        # diffusers draws its own tqdm bar per denoising loop. Kept on by
        # default; the CLI turns it off because it renders its own status line.
        self._console_progress = True

    # ---------------------------------------------------------------- loading

    @staticmethod
    def _component_libraries(model_id: str) -> dict[str, str]:
        """Map component name -> owning library, from the repo's model_index.json.

        Entries look like `"text_encoder": ["transformers", "Qwen3VLModel"]`, which
        is exactly what decides whether a component needs the diffusers or the
        transformers flavour of TorchAoConfig.
        """
        with open(_hub_file(model_id, "model_index.json"), encoding="utf-8") as fh:
            index = json.load(fh)

        libraries: dict[str, str] = {}
        for name, spec in index.items():
            # Only ["library", "ClassName"] pairs are components. Krea 2 also
            # carries plain config values, and `text_encoder_select_layers`
            # matches the prefix below without being a component at all.
            if name.startswith("_") or not isinstance(spec, (list, tuple)) or len(spec) != 2:
                continue
            if not all(isinstance(part, str) for part in spec):
                continue
            if name.startswith(_QUANTIZABLE_PREFIXES) and spec[0]:
                libraries[name] = spec[0]
        return libraries

    def _quantization_config(self) -> Any:
        """Build a pipeline-level torchao config, or None for full bf16.

        diffusers and transformers each ship their own `TorchAoConfig` and each
        validates only its own class, so components need a config instance from
        their owning library, built from the same torchao `AOBaseConfig`. The
        single `quant_backend=` shorthand raises on any pipeline whose text
        encoder is a transformers model, because those signatures disagree.
        """
        if settings.quantization == _NUNCHAKU:
            self._nunchaku_preflight()
            log.info("%s is pre-quantized; loading with its own quantization config", settings.model_id)
            return None
        if settings.quantization not in _QUANT_BACKENDS:
            return None

        from diffusers import PipelineQuantizationConfig
        from diffusers import TorchAoConfig as DiffusersTorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        # Both are one byte per weight; int8 tends to be more accurate, fp8 keeps
        # more dynamic range for outlier channels.
        make = Float8WeightOnlyConfig if settings.quantization == "fp8" else Int8WeightOnlyConfig
        config_for = {
            "transformers": TransformersTorchAoConfig,
            "diffusers": DiffusersTorchAoConfig,
        }

        mapping = {}
        for component, library in self._component_libraries(settings.model_id).items():
            cls = config_for.get(library)
            if cls is None:
                log.warning("skipping %s: unknown library %r", component, library)
                continue
            self._refuse_prequantized(component)
            mapping[component] = cls(quant_type=make())

        if not mapping:
            log.warning("no quantizable components found; loading in full bf16")
            return None
        log.info("quantizing %s to %s", sorted(mapping), settings.quantization)
        return PipelineQuantizationConfig(quant_mapping=mapping)

    @staticmethod
    def _refuse_prequantized(component: str) -> None:
        """The mirror image of `_nunchaku_preflight`: torchao on top of a repo
        that is already quantized.

        `KREA_QUANT=fp8` pointed at a Nunchaku Lite re-upload downloads ~9 GB and
        then dies deep inside transformers ("quantized with BitsAndBytesConfig
        but you are passing a TorchAoConfig"). One small config file says so up
        front. Checked per quantizable component, since the text encoder and the
        transformer name their quantizers separately.
        """
        with open(_hub_file(settings.model_id, f"{component}/config.json"), encoding="utf-8") as fh:
            method = (json.load(fh).get("quantization_config") or {}).get("quant_method")
        if method:
            raise RuntimeError(
                f"{settings.model_id} is already quantized ({component}: quant_method="
                f"{method!r}); KREA_QUANT={settings.quantization} would stack torchao on top "
                "and fail after the download. Use KREA_QUANT=nunchaku (Linux) for a Nunchaku "
                "Lite repo, or point KREA_MODEL_ID at an unquantized checkpoint."
            )

    @staticmethod
    def _nunchaku_preflight() -> None:
        """Refuse before the download, with the reason, when Nunchaku cannot run.

        Each way the mode can be broken otherwise surfaces minutes into a load,
        or not at all: the `kernels` package has no Windows builds and fails
        from inside the weight loader; bitsandbytes holds these checkpoints'
        NF4 text encoders; and a repo that is not pre-quantized loads without
        complaint in full bf16 and runs out of memory at the first image.
        """
        if sys.platform != "linux":
            raise RuntimeError(
                "KREA_QUANT=nunchaku needs Linux: the Nunchaku Lite kernels are "
                "published for x86_64 Linux only. On Windows, run the server from "
                "WSL 2 (bash wsl/setup.sh, then bash wsl/serve.sh)."
            )
        missing = [name for name in ("kernels", "bitsandbytes") if not _importable(name)]
        if missing:
            raise RuntimeError(
                f"KREA_QUANT=nunchaku needs {' and '.join(missing)}; "
                "`uv sync` on Linux installs them."
            )
        with open(_hub_file(settings.model_id, "transformer/config.json"), encoding="utf-8") as fh:
            method = (json.load(fh).get("quantization_config") or {}).get("quant_method")
        if method != _NUNCHAKU_METHOD:
            raise RuntimeError(
                f"{settings.model_id} is not a Nunchaku Lite checkpoint (transformer "
                f"quant_method={method!r}); under KREA_QUANT=nunchaku it would load in "
                f"full bf16. Point KREA_MODEL_ID at a pre-quantized repo such as "
                f"{_NUNCHAKU_EXAMPLE}."
            )

    def load(self) -> None:
        with self._lock:
            if self._pipe is not None:
                return
            import torch
            from diffusers import DiffusionPipeline

            # Offload resolves its own accelerator inside diffusers, so it needs
            # CUDA whatever KREA_DEVICE says; the check has to see both, or
            # `KREA_DEVICE=cpu` sailed past it and died after the download in
            # `enable_model_cpu_offload` with a message naming neither variable.
            needs_cuda = (
                settings.offload != "none" or torch.device(settings.device).type == "cuda"
            )
            if needs_cuda and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available to PyTorch. Model offload needs an accelerator; "
                    "to run on the CPU set KREA_OFFLOAD=none and KREA_DEVICE=cpu."
                )

            # DiffusionPipeline resolves the concrete class from model_index.json,
            # so this works for Krea2Pipeline, FluxPipeline, and anything else.
            log.info("loading %s (quant=%s)", settings.model_id, settings.quantization)
            kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
            quant = self._quantization_config()
            if quant is not None:
                kwargs["quantization_config"] = quant

            pipe = DiffusionPipeline.from_pretrained(settings.model_id, **kwargs)
            log.info("resolved pipeline class: %s", type(pipe).__name__)

            # The offload hooks resolve their own accelerator when given none,
            # which is always GPU 0: `KREA_DEVICE=cuda:1` used to load onto
            # cuda:0 while describe() reported that card's name and memory, so
            # nothing on screen said the setting had been dropped. Only a CUDA
            # device is passed through - offload cannot target the CPU, and
            # the preflight above has already insisted on an accelerator.
            device = torch.device(settings.device)
            target = device if device.type == "cuda" else None
            if settings.offload == "sequential":
                pipe.enable_sequential_cpu_offload(device=target)
            elif settings.offload == "model":
                pipe.enable_model_cpu_offload(device=target)
            else:
                pipe.to(device)

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
            self._warn_about_schedule_post_processing(pipe)
            self._call_params = params
            self._pipe = pipe
            log.info("model ready: %s", type(pipe).__name__)

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

    # -------------------------------------------------------------- schedule

    @staticmethod
    def _shifted_sigmas(steps: int, mu: float) -> list[float]:
        """The schedule a shift of `mu` produces, computed here rather than by the scheduler.

            sigma' = e^mu / (e^mu + (1/sigma - 1))

        which is `FlowMatchEulerDiscreteScheduler._time_shift_exponential` over
        the default `linspace(1, 1/steps, steps)` ramp, to the byte. Only used
        for a scheduler that does *not* shift dynamically - see `_shifts_itself`
        and `_static_shift`, which decides how much of `mu` is left to apply.
        """
        import math

        out: list[float] = []
        for i in range(steps):
            sigma = 1.0 - i / steps
            out.append(math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0)))
        return out

    @staticmethod
    def _shifts_itself(pipe: Any) -> bool:
        """Does this pipeline's scheduler apply a shift of its own?

        `sigmas` is not the final schedule. Every flow-matching pipeline here
        derives a mu - 1.15 hard-coded for a distilled checkpoint, resolution
        dependent otherwise - and hands it to `set_timesteps`, which shifts
        whatever sigmas it was given whenever `use_dynamic_shifting` is set (it
        is, in both shipped `scheduler_config.json`s). The shift multiplies the
        odds sigma/(1-sigma) by e^mu, so two of them compose into a single shift
        of `mu_a + mu_b`: passing a pre-shifted schedule does not replace the
        pipeline's mu, it adds to it.
        """
        config = getattr(getattr(pipe, "scheduler", None), "config", None)
        return bool(getattr(config, "use_dynamic_shifting", False))

    @staticmethod
    def _static_shift(pipe: Any) -> float:
        """The shift a non-dynamic scheduler applies to *whatever* sigmas it gets.

        `use_dynamic_shifting: false` does not mean no shift. `set_timesteps` is
        an if/else, and the else branch runs `shift * s / (1 + (shift - 1) * s)`
        - which multiplies the odds s/(1-s) by `shift`, exactly as the dynamic
        branch multiplies them by `e^mu`. The two compose, so a pre-shifted
        schedule handed to a scheduler with `shift=3.0` (SD3, SD3.5; Lumina 2
        ships 6.0) landed at `mu + ln 3` while the row recorded `mu`. The
        instance attribute, not `config.shift`: `set_timesteps` reads
        `self.shift`, which `set_shift()` can change after construction.
        """
        scheduler = getattr(pipe, "scheduler", None)
        shift = getattr(scheduler, "shift", None)
        if shift is None:
            shift = getattr(getattr(scheduler, "config", None), "shift", 1.0)
        return float(shift)

    @staticmethod
    def _warn_about_schedule_post_processing(pipe: Any) -> None:
        """Say so once when the scheduler reshapes the schedule after the shift.

        `shift_terminal` stretches it and the karras/exponential/beta options
        replace it outright, in both branches - so a requested mu then means
        something other than "the shift that landed". None of the shipped
        checkpoints set any of them; an arbitrary repo might.
        """
        config = getattr(getattr(pipe, "scheduler", None), "config", None)
        active = [
            key
            for key in ("shift_terminal", "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas")
            if getattr(config, key, None)
        ]
        if active:
            log.warning(
                "scheduler post-processes the schedule (%s); a requested mu is not the "
                "shift that lands",
                ", ".join(active),
            )

    @contextmanager
    def _forced_mu(self, pipe: Any, mu: float | None):
        """Run the pipeline with the scheduler shifted by `mu` and nothing else.

        Krea's CLI exposes --mu as *the* shift, which is what it has to mean
        here too. Intercepting the value on its way to the scheduler is the only
        way to say that without re-deriving `calculate_shift` - and with it the
        latent sequence length, VAE scale factor and patch size - outside the
        pipeline that owns them.

        `functools.wraps` is load-bearing: `retrieve_timesteps` reads
        `inspect.signature(scheduler.set_timesteps)` to decide whether custom
        sigmas are supported at all, and only finds the real signature by
        following `__wrapped__`. Patching the instance is safe because
        generation is serialized through a single worker thread.
        """
        scheduler = getattr(pipe, "scheduler", None)
        original = getattr(scheduler, "set_timesteps", None)
        signature = inspect.signature(original) if original is not None else None
        if mu is None or signature is None or "mu" not in signature.parameters:
            if mu is not None:
                log.debug("scheduler takes no mu; ignoring mu=%s", mu)
            yield
            return

        @functools.wraps(original)
        def set_timesteps(*args: Any, **kwargs: Any):
            bound = signature.bind(*args, **kwargs)
            bound.arguments["mu"] = mu
            return original(*bound.args, **bound.kwargs)

        scheduler.set_timesteps = set_timesteps
        try:
            yield
        finally:
            # Drop the instance attribute; the class's bound method resurfaces.
            scheduler.__dict__.pop("set_timesteps", None)

    # ------------------------------------------------------------- generation

    def _mu_supported(self, pipe: Any) -> bool:
        """Can a requested mu reach this pipeline's schedule at all?

        Two routes: a scheduler that shifts for itself takes the value through
        `set_timesteps(mu=...)`, which `_forced_mu` intercepts; one that does not
        takes an explicit pre-shifted `sigmas` list, which needs the pipeline to
        accept `sigmas`. With neither, mu is dropped - and has to be recorded as
        dropped, or the row claims a shift the image never had.
        """
        scheduler = getattr(pipe, "scheduler", None)
        if self._shifts_itself(pipe):
            original = getattr(scheduler, "set_timesteps", None)
            return original is not None and "mu" in inspect.signature(original).parameters
        return "sigmas" in self._call_params

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
            )
        return request.applied(
            negative_prompt_supported="negative_prompt" in self._call_params,
            cfg_threshold=self.cfg_threshold(),
            mu_supported=self._mu_supported(self._pipe),
        )

    def generate(
        self,
        request: "GenerateRequest",
        seed: int,
        on_progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
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

        if should_cancel is not None and should_cancel():
            raise Cancelled

        # `callback_on_step_end` is the only place the pipeline hands control back
        # mid-generation, so it doubles as the cancellation point: raising here
        # unwinds out of the denoising loop. The step is already finished by then,
        # which is why a cancel lands within one step rather than instantly.
        def _step_end(_pipe, step: int, _timestep, cb_kwargs):
            if on_progress is not None:
                on_progress(step + 1, total)
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
        # this line.
        # FLUX.2 klein takes no negative prompt at all (`negative_prompt_embeds`
        # only, and the base model's uncond pass runs against ""), so there the
        # question of a threshold never arises.
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
            if "true_cfg_scale" in self._call_params:
                kwargs["true_cfg_scale"] = request.guidance_scale
        # mu is the whole shift, not an addition to the pipeline's own. Where
        # the scheduler shifts dynamically, override the value it is given; where
        # it does not, hand it a pre-shifted schedule - minus the *static* shift
        # it will apply on top regardless, so the total is still e^mu.
        mu = request.mu
        if mu is not None and not self._shifts_itself(pipe):
            import math

            kwargs["sigmas"] = self._shifted_sigmas(
                request.steps, mu - math.log(self._static_shift(pipe))
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
        with torch.no_grad(), self._forced_mu(pipe, mu):
            result = pipe(**kwargs)
        return list(result.images)

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
            # None until loaded, like the threshold: the parameter list is the answer.
            "negative_prompt_supported": (
                "negative_prompt" in self._call_params if self._pipe is not None else None
            ),
        }
        try:
            import torch

            # The card the weights are on, not GPU 0: with offload the
            # components live on `KREA_DEVICE` between moves, and a status
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
