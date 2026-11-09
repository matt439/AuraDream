"""FLUX 3 Image Klein and Klein Base: pinned component folders, no index.

Black Forest Labs publish these repositories for diffusers' `ModularPipeline`
only: a `modular_model_index.json` and the `transformer/`, `text_encoder/` and
`vae/` folders, beside the reference code's own root files (`model.safetensors`,
`ae.safetensors`) and two quantized `variants/`. There is no `model_index.json`
for `DiffusionPipeline` to resolve, so the app's own pipeline
(`flux3_pipeline`) loads each component from its subfolder of one pinned
snapshot. Only those three folders and the index come down: never the root
BFL files or the variants, which would more than double the download.

Two passes at the same revision, as the other pinned loaders: the configs
first, which the preflight reads (the index's classes, the transformer's
flow-map flag) before a weight is asked for, then the weights. The index's
component paths are never followed: every component loads from this
snapshot's own subfolder.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

# Resolve the app's remote-kernel policy even for direct imports.
from .. import config as _config  # noqa: F401
from .single_file import HubRepo

log = logging.getLogger(__name__)

MODEL_ID = "black-forest-labs/flux-3-image-klein"
BASE_MODEL_ID = "black-forest-labs/flux-3-image-klein-base"
# Pinned on 2026-11-09 from the Hub's API (the repositories' head then).
SOURCE: HubRepo = {"repo_id": MODEL_ID, "revision": "09561231f44e41381c461caf53d5452005a60ac9"}
BASE_SOURCE: HubRepo = {"repo_id": BASE_MODEL_ID, "revision": "a83ce283b1daa2f3d782645ae7b06c8e73af1b41"}
_SOURCES: dict[str, HubRepo] = {MODEL_ID.lower(): SOURCE, BASE_MODEL_ID.lower(): BASE_SOURCE}

INDEX = "modular_model_index.json"
COMPONENTS = ("transformer", "text_encoder", "vae")
CONFIG_PATTERNS = [INDEX, *(f"{name}/config.json" for name in COMPONENTS)]
WEIGHT_PATTERNS = [INDEX, *(f"{name}/*" for name in COMPONENTS)]

# What each component is, as the index must name it: (library, class). The
# blocks' own expected components (diffusers/modular_pipelines/flux3).
EXPECTED: dict[str, tuple[str, str]] = {
    "transformer": ("diffusers", "Flux3Transformer2DModel"),
    "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration"),
    "vae": ("diffusers", "AutoencoderKLFlux2"),
}
# The modular pipeline class each index names, and whether its transformer
# is the step-distilled flow-map one (`target_timestep`; the distilled loop
# passes it, the Base loop never does, and the transformer refuses a mismatch).
_VARIANTS: dict[str, tuple[str, bool]] = {
    MODEL_ID.lower(): ("Flux3DistilledModularPipeline", True),
    BASE_MODEL_ID.lower(): ("Flux3ModularPipeline", False),
}
# Flux3SetTimestepsStep's constant shift, configured on the scheduler: the
# repositories ship no scheduler folder.
SHIFT = 6.93
# The README's dtypes: the VAE in float32, everything else in bfloat16.
VAE_DTYPE = "float32"


def matches(model_id: str) -> bool:
    return model_id.lower() in _SOURCES


def source(model_id: str) -> HubRepo:
    return _SOURCES[model_id.lower()]


def pipeline_class(model_id: str) -> Any:
    from .flux3_pipeline import Flux3KleinBasePipeline, Flux3KleinPipeline

    key = model_id.lower()
    if key == MODEL_ID.lower():
        return Flux3KleinPipeline
    if key == BASE_MODEL_ID.lower():
        return Flux3KleinBasePipeline
    raise ValueError(f"Unknown FLUX 3 selection: {model_id}")


def _read_json(snapshot: str, name: str) -> Any:
    with open(Path(snapshot, *name.split("/")), encoding="utf-8") as stream:
        return json.load(stream)


def preflight(model_id: str, snapshot: str) -> None:
    """Refuse a snapshot whose index or transformer is not this variant's.

    Before any weight: the index names the expected modular class and
    component classes, and the transformer's `flow_map` matches the loop the
    app runs for this id. Says what differs, never a file's content.
    """
    pipeline, flow_map = _VARIANTS[model_id.lower()]
    try:
        index = _read_json(snapshot, INDEX)
        transformer = _read_json(snapshot, "transformer/config.json")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{model_id}'s pinned configs could not be read; no weights were loaded.") from exc
    if not isinstance(index, dict) or index.get("_class_name") != pipeline:
        raise RuntimeError(f"{model_id}'s index does not name {pipeline}; no weights were loaded.")
    for name, (library, cls) in EXPECTED.items():
        spec = index.get(name)
        if not (isinstance(spec, list) and len(spec) >= 2 and spec[0] == library and spec[1] == cls):
            raise RuntimeError(f"{model_id}'s index does not name {library} {cls} as {name}; no weights were loaded.")
    if not isinstance(transformer, dict) or transformer.get("_class_name") != EXPECTED["transformer"][1]:
        raise RuntimeError(f"{model_id}'s transformer is not a {EXPECTED['transformer'][1]}; no weights were loaded.")
    if bool(transformer.get("flow_map", False)) is not flow_map:
        kind = "step-distilled (flow-map)" if flow_map else "undistilled"
        raise RuntimeError(f"{model_id}'s transformer is not the {kind} one this loader runs; no weights were loaded.")


def resolve_model(model_id: str, *, weights: bool, quantization: str | None = None) -> str:
    """The pinned snapshot: its configs (checked), then its component weights."""
    if quantization == "nunchaku":
        raise RuntimeError(
            "FLUX 3 [Klein] has no Nunchaku Lite build in this app yet. "
            "Use AURADREAM_QUANT=fp8, int8 or none."
        )
    from huggingface_hub import snapshot_download

    pin = source(model_id)
    patterns = WEIGHT_PATTERNS if weights else CONFIG_PATTERNS
    log.info("resolving %s (%s)", model_id, "weights" if weights else "configs")
    snapshot = str(snapshot_download(pin["repo_id"], revision=pin["revision"], allow_patterns=patterns))
    preflight(model_id, snapshot)
    return snapshot


def component_libraries(snapshot: str) -> dict[str, str] | None:
    """The quantizable components' owning libraries, from a FLUX 3 snapshot's
    own modular index (its entries carry a third, load-spec element), or
    None where `snapshot` is not a local tree with one and no model_index."""
    if not os.path.isdir(snapshot) or os.path.isfile(os.path.join(snapshot, "model_index.json")):
        return None
    try:
        index = _read_json(snapshot, INDEX)
    except FileNotFoundError:
        return None
    libraries: dict[str, str] = {}
    for name in ("transformer", "text_encoder"):
        spec = index.get(name) if isinstance(index, dict) else None
        if isinstance(spec, list) and len(spec) >= 2 and isinstance(spec[0], str) and spec[0]:
            libraries[name] = spec[0]
    return libraries


def load_components(pipeline: Any, snapshot: str, **kwargs: Any) -> Any:
    """Each component from `snapshot`'s subfolder, quantized as a standard load
    would quantize it, then `pipeline` built over them.

    `quantization_config` is the per-component `PipelineQuantizationConfig`
    the backend built (`loading.quantization_config`), and only its mapping is
    read. Inside `loading.checkpoint_files_released`, the transformer's shards
    are released by the patched diffusers shard loader as for any load; the
    text encoder, loaded here rather than through diffusers' `load_sub_model`,
    gets the same copy-out of its unquantized tensors (`own_tensors`).
    """
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from . import loading

    dtype = kwargs.pop("dtype", torch.bfloat16)
    quant = kwargs.pop("quantization_config", None)
    if kwargs:
        raise TypeError(f"FLUX 3 loading takes no {', '.join(sorted(kwargs))}")
    mapping: dict[str, Any] = dict(getattr(quant, "quant_mapping", None) or {}) if quant is not None else {}
    unknown = set(mapping) - {"transformer", "text_encoder"}
    if unknown:
        raise ValueError(f"FLUX 3 has no quantizable {', '.join(sorted(unknown))}")

    root = Path(snapshot)
    transformer = Flux3Transformer2DModel.from_pretrained(
        snapshot, subfolder="transformer", dtype=dtype,
        quantization_config=mapping.get("transformer"), local_files_only=True,
    )
    encoder_quant = mapping.get("text_encoder")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        str(root / "text_encoder"), dtype=dtype,
        quantization_config=encoder_quant, local_files_only=True,
    )
    if encoder_quant is not None:
        # As `checkpoint_files_released`'s per-component hook does for a
        # transformers component that `load_sub_model` loaded.
        loading.own_tensors(text_encoder)
    tokenizer = AutoProcessor.from_pretrained(str(root / "text_encoder"), local_files_only=True)
    vae = AutoencoderKLFlux2.from_pretrained(
        snapshot, subfolder="vae", dtype=getattr(torch, VAE_DTYPE), local_files_only=True,
    )
    return pipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer,
        vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(shift=SHIFT),
    )
