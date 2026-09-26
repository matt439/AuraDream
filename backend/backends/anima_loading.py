"""Pinned, CPU-only conversion of the official Anima Turbo checkpoint.

Anima publishes Turbo's denoiser and text conditioner together. Its official
Base Diffusers tree supplies their configs and the shared encoder, VAE and
tokenizers, but never supplies Base denoiser or conditioner weights here.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

# Resolve the app's remote-kernel policy even for direct converter imports.
from .. import config as _config  # noqa: F401
from .single_file import cache_root

log = logging.getLogger(__name__)

MODEL_ID = "circlestone-labs/Anima"
SOURCE = {
    "repo_id": MODEL_ID,
    "filename": "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
    "revision": "f973fc41ec7545364ac9776c2440285f43ff2a30",
}
SHARED = {
    "repo_id": "circlestone-labs/Anima-Base-v1.0-Diffusers",
    "revision": "073c3a9db359c31ad0e8aa268d15775473c2176c",
}
SHARED_PATTERNS = [
    "transformer/config.json", "text_conditioner/config.json",
    "text_encoder/*", "vae/*", "tokenizer/*", "t5_tokenizer/*",
]
RECIPE = {
    "version": 1,
    "source": SOURCE,
    "shared": SHARED,
    "dtype": "bfloat16",
    "scheduler": "FlowMatchEulerDiscreteScheduler",
    "shift": 3.0,
}


def _split_checkpoint(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert Turbo's Cosmos names and retain its own learned conditioner.

The official conversion script uses an older ``net.`` prefix. Turbo v1.1's
published safetensors instead uses ``model.diffusion_model.``. Normalize only
that pinned format, then use Diffusers' Cosmos conversion for the denoiser.
    """
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers

    prefix = "model.diffusion_model."
    adapter_prefix = prefix + "llm_adapter."
    transformer, conditioner = {}, {}
    for key, value in checkpoint.items():
        if key.startswith(adapter_prefix):
            conditioner[key.removeprefix(adapter_prefix)] = value
        elif key.startswith(prefix):
            transformer["net." + key.removeprefix(prefix)] = value
        else:
            raise ValueError(f"Unexpected Anima Turbo checkpoint key: {key}")
    if not transformer or not conditioner:
        raise ValueError("Anima Turbo checkpoint requires both denoiser and text conditioner weights")
    return convert_cosmos_transformer_checkpoint_to_diffusers(transformer), conditioner


def _load_component(model_cls: Any, config: str | Path, state_dict: dict[str, Any]) -> Any:
    """Allocate on meta, then require every converted CPU tensor to fit."""
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = model_cls.from_config(model_cls.load_config(config))
    # A missing or mismatched key must fail conversion, never become random
    # weights or silently substitute the Base checkpoint's conditioner.
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model


def _convert(destination: Path) -> None:
    import torch
    from diffusers import AnimaTextConditioner, AutoencoderKLQwenImage, CosmosTransformer3DModel
    from diffusers import FlowMatchEulerDiscreteScheduler
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3Model, T5TokenizerFast

    from .anima_pipeline import AnimaTurboPipeline

    shared = Path(snapshot_download(**SHARED, allow_patterns=SHARED_PATTERNS))
    checkpoint = load_file(hf_hub_download(**SOURCE), device="cpu")
    transformer_state, conditioner_state = _split_checkpoint(checkpoint)
    del checkpoint
    transformer = _load_component(CosmosTransformer3DModel, shared / "transformer", transformer_state)
    conditioner = _load_component(AnimaTextConditioner, shared / "text_conditioner", conditioner_state)
    del transformer_state, conditioner_state
    # The shared descriptors are local pinned files, so none of these loaders
    # can follow an unpinned Hub reference or load a different Base component.
    local = {"local_files_only": True}
    pipe = AnimaTurboPipeline(
        tokenizer=AutoTokenizer.from_pretrained(shared / "tokenizer", **local),
        t5_tokenizer=T5TokenizerFast.from_pretrained(shared / "t5_tokenizer", **local),
        text_encoder=Qwen3Model.from_pretrained(shared / "text_encoder", dtype=torch.bfloat16, **local),
        text_conditioner=conditioner.to(dtype=torch.bfloat16),
        transformer=transformer.to(dtype=torch.bfloat16),
        vae=AutoencoderKLQwenImage.from_pretrained(shared / "vae", dtype=torch.bfloat16, **local),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    pipe.save_pretrained(destination, safe_serialization=True)


def _target() -> tuple[Path, Path, str]:
    recipe = json.dumps(RECIPE, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    target = cache_root() / f"anima-turbo-v1.1-{digest}"
    return target, target / "local-diffusion-conversion.json", recipe


def _converted(target: Path, marker: Path, recipe: str) -> bool:
    return (
        marker.is_file()
        and (target / "model_index.json").is_file()
        and marker.read_text(encoding="utf-8") == recipe
    )


def converted_tree(model_id: str) -> str | None:
    """Inspect an already converted tree without downloading or converting."""
    if model_id.lower() != MODEL_ID.lower():
        return None
    target, marker, recipe = _target()
    return str(target) if _converted(target, marker, recipe) else None


def resolve_model(model_id: str, quantization: str) -> str:
    """Return the cached standard component tree; retain the public id outside loading."""
    if model_id.lower() != MODEL_ID.lower():
        return model_id
    if quantization == "nunchaku":
        raise RuntimeError(
            "Anima Turbo is not a Nunchaku Lite checkpoint. "
            "Use LOCAL_DIFFUSION_QUANT=fp8, int8 or none."
        )

    from huggingface_hub.utils import WeakFileLock

    target, marker, recipe = _target()
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(root / f"{target.name}.lock"):
        if _converted(target, marker, recipe):
            return str(target)
        if target.exists():
            raise RuntimeError(f"Incomplete converted model cache; remove {target} and retry.")
        log.info("converting official Anima Turbo v1.1 on CPU once: %s", target)
        with TemporaryDirectory(prefix=f"{target.name}-", dir=root) as work:
            tree = Path(work) / "tree"
            _convert(tree)
            if not (tree / "model_index.json").is_file():
                raise RuntimeError("Anima Turbo conversion did not produce model_index.json")
            (tree / marker.name).write_text(recipe, encoding="utf-8")
            tree.rename(target)
    log.info("converted model cached: %s", target)
    return str(target)
