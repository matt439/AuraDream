"""Measured Anima Refine envelopes, independent of runtime imports.

Supplied facts decide admission. These result limits do not change the request,
its reproduction identity or any other checkpoint's ordinary Refine behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .execution import canonical_execution, canonical_loader_execution


@dataclass(frozen=True)
class QualifiedRefine:
    model_id: str = "circlestone-labs/Anima"
    name: str = "Anima Turbo Refine"
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
    empty_negative: bool = False  # False after the variant's conditioning branch is qualified.


ANIMA_TURBO = QualifiedRefine()
ANIMA_AESTHETIC = QualifiedRefine(
    model_id="circlestone-labs/Anima@aesthetic-v1.1", name="Anima Aesthetic Refine",
    steps=30, guidance_scale=4., default_strength=.125, empty_negative=False,
)
ANIMA_PREVIEW = QualifiedRefine(
    model_id="Gazingstars123/Anima-2.9B", name="Anima 2.9B Preview Refine",
    steps=30, guidance_scale=4., default_strength=.125, empty_negative=False,
)
ENVELOPES = MappingProxyType({entry.model_id.casefold(): entry for entry in (
    ANIMA_TURBO, ANIMA_AESTHETIC, ANIMA_PREVIEW,
)})
NAME = ANIMA_TURBO.name  # Retained name of the first measured slice.


def envelope_for(model_id: str) -> QualifiedRefine | None:
    return ENVELOPES.get(model_id.casefold())


def applies(model_id: str) -> bool:
    """Recognize case variants too, so they cannot escape the exact-id hold."""
    return envelope_for(model_id) is not None


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
    envelope = envelope_for(model_id)
    if envelope is None:
        return None
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
    return f"{envelope.name} is held: {choice}."


def source_refusal(model_id: str, size: tuple[int, int]) -> str | None:
    envelope = envelope_for(model_id)
    if envelope is not None and size != envelope.size:
        return f"{envelope.name} is measured only from a 1024×1024 source; this source is {size[0]}×{size[1]}."
    return None


def request_refusal(model_id: str, request: Any) -> str | None:
    """Check every actual expanded pass; never normalize unsupported intent."""
    envelope = envelope_for(model_id)
    if envelope is None or request.refine is None:
        return None
    name = envelope.name
    try:
        execution = canonical_execution(getattr(request, "execution", None))
    except (TypeError, ValueError):
        return f"{name} has an unreadable performance configuration."
    if execution is not None:
        return f"{name} is measured only with Eager, Native attention and step caching off."
    if request.inpainting or request.edit is not None or request.upscaler is not None:
        return f"{name} is measured only as an ordinary whole-image Refine pass."
    if (request.width, request.height) != envelope.size:
        return f"{name} is measured only at 1024×1024; enlargement and other output sizes are held."
    if type(request.steps) is not int or request.steps != envelope.steps:
        return f"{name} is measured only at {envelope.steps} steps; set Steps to {envelope.steps}."
    negative = getattr(request, "negative_prompt", None)
    recorded_cfg = (negative and envelope.guidance_scale > 1
                    and request.true_cfg == envelope.guidance_scale)
    if request.guidance_scale != envelope.guidance_scale or (request.true_cfg is not None and not recorded_cfg):
        return f"{name} is measured only at CFG {envelope.guidance_scale:g}; set CFG to {envelope.guidance_scale:g}."
    if envelope.empty_negative and negative not in (None, ""):
        return f"{name} is measured with an empty negative prompt; clear the negative prompt."
    if request.strength not in envelope.strengths:
        return f"{name} is measured only at strengths 0.125, 0.25 and 0.4."
    if request.lora is not None or request.reference or request.mu is not None:
        return f"{name} is measured without LoRAs, references or a Mu override."
    if request.crop_anchor != "center":
        return f"{name} is measured with the source unchanged at its own size; use the centre crop anchor."
    return None


def suggestions(model_id: str) -> dict[str, Any] | None:
    envelope = envelope_for(model_id)
    if envelope is None:
        return None
    return {"steps": envelope.steps, "strengths": list(envelope.strengths),
            "steps_reason": f"{envelope.name} is measured at {envelope.steps} steps; set Steps to {envelope.steps} to compare the measured strengths."}
