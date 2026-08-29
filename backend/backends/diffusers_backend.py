"""In-process backend built on HuggingFace diffusers.

Sizing note: Krea 2 is a ~13B DiT and needs ~25GB in bfloat16, which does not fit
a 16GB card. The default config quantizes the transformer and text encoders to
float8 weight-only via torchao, then leans on accelerate's model offload so only
one component is resident at a time. Peak VRAM lands near 13GB.

Nothing here is Krea-specific: the pipeline class and the set of quantizable
components are both read from the repo's `model_index.json`, so `KREA_MODEL_ID`
can point at FLUX.1-Krea-dev or any other diffusers text-to-image repo.
"""
from __future__ import annotations

import inspect
import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from ..config import settings
from .base import CancelFn, Cancelled, GenerationBackend, ProgressFn

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas import GenerateRequest

log = logging.getLogger(__name__)

_QUANT_BACKENDS = ("fp8", "int8")

# Keys in model_index.json worth quantizing. The VAE stays in bf16: it is small
# and quantizing it shows up as visible decode artefacts.
_QUANTIZABLE_PREFIXES = ("transformer", "unet", "text_encoder")


class DiffusersBackend(GenerationBackend):
    name = "diffusers"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._call_params: set[str] = set()
        self._error: str | None = None

    # ---------------------------------------------------------------- loading

    @staticmethod
    def _component_libraries(model_id: str) -> dict[str, str]:
        """Map component name -> owning library, from the repo's model_index.json.

        Entries look like `"text_encoder": ["transformers", "Qwen3VLModel"]`, which
        is exactly what decides whether a component needs the diffusers or the
        transformers flavour of TorchAoConfig.
        """
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_id, "model_index.json"), encoding="utf-8") as fh:
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
            mapping[component] = cls(quant_type=make())

        if not mapping:
            log.warning("no quantizable components found; loading in full bf16")
            return None
        log.info("quantizing %s to %s", sorted(mapping), settings.quantization)
        return PipelineQuantizationConfig(quant_mapping=mapping)

    def load(self) -> None:
        with self._lock:
            if self._pipe is not None:
                return
            import torch
            from diffusers import DiffusionPipeline

            if settings.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available to PyTorch")

            # DiffusionPipeline resolves the concrete class from model_index.json,
            # so this works for Krea2Pipeline, FluxPipeline, and anything else.
            log.info("loading %s (quant=%s)", settings.model_id, settings.quantization)
            kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
            quant = self._quantization_config()
            if quant is not None:
                kwargs["quantization_config"] = quant

            pipe = DiffusionPipeline.from_pretrained(settings.model_id, **kwargs)
            log.info("resolved pipeline class: %s", type(pipe).__name__)

            if settings.offload == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif settings.offload == "model":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(settings.device)

            # VAE decode at 2K is the other memory spike; slice and tile it.
            for opt in ("enable_vae_slicing", "enable_vae_tiling"):
                fn = getattr(pipe, opt, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 - purely opportunistic
                        log.debug("%s unavailable", opt)

            self._call_params = set(inspect.signature(pipe.__call__).parameters)
            self._pipe = pipe
            self._error = None
            log.info("model ready")

    def unload(self) -> None:
        with self._lock:
            if self._pipe is None:
                return
            self._pipe = None
            self._call_params = set()
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_ready(self) -> bool:
        return self._pipe is not None

    # ------------------------------------------------------------- generation

    @staticmethod
    def _shifted_sigmas(steps: int, mu: float) -> list[float]:
        """Krea's CLI exposes --mu; the diffusers pipeline exposes `sigmas`.

        Reproduce flow-matching dynamic shifting explicitly:
            sigma' = e^mu / (e^mu + (1/sigma - 1))
        """
        import math

        out: list[float] = []
        for i in range(steps):
            sigma = 1.0 - i / steps
            out.append(math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0)))
        return out

    def generate(
        self,
        request: "GenerateRequest",
        seed: int,
        on_progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
    ) -> list["Image"]:
        if self._pipe is None:
            self.load()
        import torch

        pipe = self._pipe
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
        generator = [
            torch.Generator(device="cpu").manual_seed(seed + offset)
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
        # CFG is inactive at or below 1.0, which is where Turbo runs, and a
        # negative prompt there just costs a wasted text-encoder pass.
        if request.negative_prompt and request.guidance_scale > 1.0:
            kwargs["negative_prompt"] = request.negative_prompt
            # On FLUX, `guidance_scale` is a distilled conditioning value baked
            # into the model, not classifier-free guidance -- a negative prompt
            # is silently ignored unless true_cfg_scale rises above 1.0. Costs a
            # second forward pass per step.
            if "true_cfg_scale" in self._call_params:
                kwargs["true_cfg_scale"] = request.guidance_scale
        if request.mu is not None:
            kwargs["sigmas"] = self._shifted_sigmas(request.steps, request.mu)

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
        with torch.no_grad():
            result = pipe(**kwargs)
        return list(result.images)

    # ----------------------------------------------------------------- status

    def describe(self) -> dict:
        info: dict[str, Any] = {
            "backend": self.name,
            "model_id": settings.model_id,
            "quantization": settings.quantization,
            "offload": settings.offload,
            "detail": self._error,
            "pipeline": type(self._pipe).__name__ if self._pipe is not None else None,
        }
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                info["device_name"] = torch.cuda.get_device_name(0)
                info["vram_total_mb"] = total // (1024 * 1024)
                info["vram_used_mb"] = (total - free) // (1024 * 1024)
        except Exception:  # noqa: BLE001 - status must never raise
            pass
        return info
