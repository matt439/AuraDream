"""The measured Anima Turbo Refine envelope, independent of runtime imports.

Supplied facts decide admission. These result limits do not change the request,
its reproduction identity or any other checkpoint's ordinary Refine behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution import canonical_execution, canonical_loader_execution


@dataclass(frozen=True)
class QualifiedRefine:
    model_id: str = "circlestone-labs/Anima"
    gpu_name: str = "NVIDIA GeForce RTX 5080"
    platform: str = "win32"
    diffusers_version: str = "0.42.1"
    quantization: str = "fp8"
    offload: str = "model"
    devices: tuple[str, ...] = ("cuda", "cuda:0")
    size: tuple[int, int] = (1024, 1024)
    steps: int = 8
    guidance_scale: float = 1.0
    strengths: tuple[float, ...] = (.125, .25, .4)
    default_strength: float = .25


ANIMA_TURBO = QualifiedRefine()
NAME = "Anima Turbo Refine"


def applies(model_id: str) -> bool:
    """Recognize case variants too, so they cannot escape the exact-id hold."""
    return model_id.casefold() == ANIMA_TURBO.model_id.casefold()


@dataclass(frozen=True)
class LoaderFacts:
    quantization: str
    offload: str
    device: str
    platform: str
    diffusers_version: str | None
    gpu_name: str | None
    native_attention: bool
    loader_execution: object = None

    @classmethod
    def of(cls, selected: Any, *, platform: str, diffusers_version: str | None,
           gpu_name: str | None, native_attention: bool) -> LoaderFacts:
        return cls(selected.quantization, selected.offload, selected.device,
                   platform, diffusers_version, gpu_name, native_attention,
                   getattr(selected, "loader_execution", None))


def loader_refusal(model_id: str, facts: LoaderFacts, *, check_hardware: bool = True) -> str | None:
    """The first unmeasured supplied loading fact, without probing anything."""
    if not applies(model_id):
        return None
    envelope = ANIMA_TURBO
    if model_id != envelope.model_id:
        choice = f"select {envelope.model_id} exactly"
    elif facts.platform != envelope.platform:
        choice = "this platform; only Windows was measured"
    elif facts.quantization != envelope.quantization:
        choice = f"{facts.quantization} quantization; only fp8 was measured"
    elif facts.offload != envelope.offload:
        choice = f"{facts.offload} offload; only model offload was measured"
    elif facts.device not in envelope.devices:
        choice = "this device; only CUDA device 0 was measured"
    elif facts.diffusers_version != envelope.diffusers_version:
        choice = f"this diffusers version; only {envelope.diffusers_version} was measured"
    elif check_hardware and facts.gpu_name is None:
        choice = "the GPU has not been verified; load the checkpoint or wait for the card facts"
    elif check_hardware and facts.gpu_name != envelope.gpu_name:
        choice = "this GPU; only the RTX 5080 was measured"
    elif not facts.native_attention:
        choice = "the external attention override; only Native attention was measured"
    else:
        try:
            execution = canonical_loader_execution(facts.loader_execution)
        except (TypeError, ValueError):
            choice = "an unreadable performance configuration"
        else:
            if execution is None:
                return None
            choice = "this performance configuration; only Eager with Native attention was measured"
    return f"{NAME} is held: {choice}."


def source_refusal(model_id: str, size: tuple[int, int]) -> str | None:
    if applies(model_id) and size != ANIMA_TURBO.size:
        return f"{NAME} is measured only from a 1024×1024 source; this source is {size[0]}×{size[1]}."
    return None


def request_refusal(model_id: str, request: Any) -> str | None:
    """Check every actual expanded pass; never normalize unsupported intent."""
    if not applies(model_id) or request.refine is None:
        return None
    envelope = ANIMA_TURBO
    try:
        execution = canonical_execution(getattr(request, "execution", None))
    except (TypeError, ValueError):
        return f"{NAME} has an unreadable performance configuration."
    if execution is not None:
        return f"{NAME} is measured only with Eager, Native attention and step caching off."
    if request.inpainting or request.edit is not None or request.upscaler is not None:
        return f"{NAME} is measured only as an ordinary whole-image Refine pass."
    if (request.width, request.height) != envelope.size:
        return f"{NAME} is measured only at 1024×1024; enlargement and other output sizes are held."
    if type(request.steps) is not int or request.steps != envelope.steps:
        return f"{NAME} is measured only at 8 steps; set Steps to 8."
    if request.guidance_scale != envelope.guidance_scale or request.true_cfg is not None:
        return f"{NAME} is measured only at CFG 1; set CFG to 1."
    if request.strength not in envelope.strengths:
        return f"{NAME} is measured only at strengths 0.125, 0.25 and 0.4."
    if request.lora is not None or request.reference or request.mu is not None:
        return f"{NAME} is measured without LoRAs, references or a Mu override."
    if request.crop_anchor != "center":
        return f"{NAME} is measured with the source unchanged at its own size; use the centre crop anchor."
    return None


def suggestions(model_id: str) -> dict[str, Any] | None:
    if not applies(model_id):
        return None
    return {"steps": ANIMA_TURBO.steps, "strengths": list(ANIMA_TURBO.strengths),
            "steps_reason": f"{NAME} is measured at 8 steps; set Steps to 8 to compare the measured strengths."}
