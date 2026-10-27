"""Pinned, CPU-only conversion of Anima checkpoints.

Anima publishes each denoiser and text conditioner together. Its official
Base Diffusers tree supplies their configs and the shared encoder, VAE and
tokenizers, but never supplies Base denoiser or conditioner weights here.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NotRequired, TypedDict

# Resolve the app's remote-kernel policy even for direct converter imports.
from .. import config as _config  # noqa: F401
from .single_file import CONVERSION_MARKER, HubFile, HubRepo, cache_root, sweep_work

log = logging.getLogger(__name__)


class _ConversionRecipe(TypedDict):
    version: int
    source: HubFile
    shared: HubRepo
    dtype: str
    scheduler: str
    shift: float
    checkpoint_prefix: NotRequired[str]
    transformer_config: NotRequired[dict[str, int]]


MODEL_ID = "circlestone-labs/Anima"
# An app selector, not another Hub repository. Keep the bare id's historical
# Turbo meaning so saved images and resume keys never change checkpoints.
AESTHETIC_MODEL_ID = MODEL_ID + "@aesthetic-v1.1"
PREVIEW_MODEL_ID = "Gazingstars123/Anima-2.9B"
SOURCE: HubFile = {
    "repo_id": MODEL_ID,
    "filename": "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
    "revision": "f973fc41ec7545364ac9776c2440285f43ff2a30",
}
AESTHETIC_SOURCE: HubFile = {
    "repo_id": MODEL_ID,
    "filename": "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors",
    "revision": "594c27fea35648b87c86a9b4d5436a6024c820b5",
}
PREVIEW_SOURCE: HubFile = {
    "repo_id": PREVIEW_MODEL_ID,
    "filename": "Anima-2.9B-preview-v1.safetensors",
    "revision": "9f9cb502dbae7a616c3cc5a530633427fe735665",
}
SHARED: HubRepo = {
    "repo_id": "circlestone-labs/Anima-Base-v1.0-Diffusers",
    "revision": "073c3a9db359c31ad0e8aa268d15775473c2176c",
}
SHARED_PATTERNS = [
    "transformer/config.json", "text_conditioner/config.json",
    "text_encoder/*", "vae/*", "tokenizer/*", "t5_tokenizer/*",
]
RECIPE: _ConversionRecipe = {
    "version": 1,
    "source": SOURCE,
    "shared": SHARED,
    "dtype": "bfloat16",
    "scheduler": "FlowMatchEulerDiscreteScheduler",
    "shift": 3.0,
}
AESTHETIC_RECIPE: _ConversionRecipe = {**RECIPE, "source": AESTHETIC_SOURCE}
PREVIEW_RECIPE: _ConversionRecipe = {
    **RECIPE,
    "source": PREVIEW_SOURCE,
    "checkpoint_prefix": "net.",
    "transformer_config": {"num_layers": 40},
}
_RECIPES: dict[str, _ConversionRecipe] = {
    MODEL_ID.lower(): RECIPE,
    AESTHETIC_MODEL_ID.lower(): AESTHETIC_RECIPE,
    PREVIEW_MODEL_ID.lower(): PREVIEW_RECIPE,
}


def matches(model_id: str) -> bool:
    return model_id.lower() in _RECIPES


def pipeline_class(model_id: str) -> Any:
    from .anima_pipeline import AnimaAestheticPipeline, AnimaPreviewPipeline, AnimaTurboPipeline

    if model_id.lower() == PREVIEW_MODEL_ID.lower():
        return AnimaPreviewPipeline
    if model_id.lower() == AESTHETIC_MODEL_ID.lower():
        return AnimaAestheticPipeline
    if model_id.lower() == MODEL_ID.lower():
        return AnimaTurboPipeline
    raise ValueError(f"Unknown Anima selection: {model_id}")


def _split_checkpoint(
    checkpoint: dict[str, Any], prefix: str = "model.diffusion_model.",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert the Cosmos names and retain the checkpoint's learned conditioner.

    The official conversion script uses an older ``net.`` prefix. Both v1.1
    safetensors files instead use ``model.diffusion_model.``. Normalize only
    that pinned format; 2.9B preview uses the original ``net.`` prefix. The
    pinned recipe chooses the format, then Diffusers maps the denoiser names.
    """
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers

    adapter_prefix = prefix + "llm_adapter."
    transformer, conditioner = {}, {}
    for key, value in checkpoint.items():
        if key.startswith(adapter_prefix):
            conditioner[key.removeprefix(adapter_prefix)] = value
        elif key.startswith(prefix):
            transformer["net." + key.removeprefix(prefix)] = value
        else:
            raise ValueError(f"Unexpected Anima checkpoint key: {key}")
    if not transformer or not conditioner:
        raise ValueError("Anima checkpoint requires both denoiser and text conditioner weights")
    return convert_cosmos_transformer_checkpoint_to_diffusers(transformer), conditioner


def _load_component(
    model_cls: Any, config: str | Path, state_dict: dict[str, Any], **config_overrides: Any,
) -> Any:
    """Allocate on meta, then require every converted CPU tensor to fit."""
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = model_cls.from_config(model_cls.load_config(config), **config_overrides)
    # A missing or mismatched key must fail conversion, never become random
    # weights or silently substitute the Base checkpoint's conditioner.
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model


def _convert(destination: Path, model_id: str = MODEL_ID) -> None:
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3Model
    # Transformers 5's lazy Fast-name fallback returns this same class.
    from transformers.models.t5.tokenization_t5 import T5Tokenizer as T5TokenizerFast

    shared = Path(snapshot_download(**SHARED, allow_patterns=SHARED_PATTERNS))
    recipe = _RECIPES[model_id.lower()]
    checkpoint = load_file(hf_hub_download(**recipe["source"]), device="cpu")
    transformer_state, conditioner_state = _split_checkpoint(
        checkpoint, recipe.get("checkpoint_prefix", "model.diffusion_model."),
    )
    del checkpoint
    # Preview expands the same-width Base transformer from 28 to 40 layers.
    # Override its pinned architecture before strict loading; all learned
    # layers still come from the selected checkpoint, including the new ones.
    transformer = _load_component(
        CosmosTransformer3DModel, shared / "transformer", transformer_state,
        **recipe.get("transformer_config", {}),
    )
    conditioner = _load_component(AnimaTextConditioner, shared / "text_conditioner", conditioner_state)
    del transformer_state, conditioner_state
    # The shared descriptors are local pinned files, so none of these loaders
    # can follow an unpinned Hub reference or load a different Base component.
    pipe = pipeline_class(model_id)(
        tokenizer=AutoTokenizer.from_pretrained(shared / "tokenizer", local_files_only=True),
        t5_tokenizer=T5TokenizerFast.from_pretrained(shared / "t5_tokenizer", local_files_only=True),
        text_encoder=Qwen3Model.from_pretrained(shared / "text_encoder", dtype=torch.bfloat16, local_files_only=True),
        text_conditioner=conditioner.to(dtype=torch.bfloat16),
        transformer=transformer.to(dtype=torch.bfloat16),
        vae=AutoencoderKLQwenImage.from_pretrained(shared / "vae", dtype=torch.bfloat16, local_files_only=True),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    pipe.save_pretrained(destination, safe_serialization=True)


def _target(model_id: str = MODEL_ID) -> tuple[Path, Path, str]:
    selected = _RECIPES[model_id.lower()]
    recipe = json.dumps(selected, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    stem = Path(selected["source"]["filename"]).stem
    target = cache_root() / f"{stem}-{digest}"
    return target, target / CONVERSION_MARKER, recipe


def _converted(target: Path, marker: Path, recipe: str) -> bool:
    return (
        marker.is_file()
        and (target / "model_index.json").is_file()
        and marker.read_text(encoding="utf-8") == recipe
    )


def converted_tree(model_id: str) -> str | None:
    """Inspect an already converted tree without downloading or converting."""
    if not matches(model_id):
        return None
    target, marker, recipe = _target(model_id)
    return str(target) if _converted(target, marker, recipe) else None


def resolve_model(model_id: str, quantization: str) -> str:
    """Return the cached standard component tree; retain the public id outside loading."""
    if not matches(model_id):
        return model_id
    if quantization == "nunchaku":
        raise RuntimeError(
            "This Anima selection is not a Nunchaku Lite checkpoint. "
            "Use AURADREAM_QUANT=fp8, int8 or none."
        )

    from huggingface_hub.utils._fixes import WeakFileLock

    target, marker, recipe = _target(model_id)
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(root / f"{target.name}.lock"):
        # Converted or not: leftovers from before a finished conversion count too.
        sweep_work(target)
        if _converted(target, marker, recipe):
            return str(target)
        if target.exists():
            raise RuntimeError(f"Incomplete converted model cache; remove {target} and retry.")
        log.info("converting %s on CPU once: %s", model_id, target)
        # A cleanup that meets an open file leaves the rest to the next load's
        # sweep rather than raising over the conversion's own error.
        with TemporaryDirectory(prefix=f"{target.name}-", dir=root, ignore_cleanup_errors=True) as work:
            tree = Path(work) / "tree"
            _convert(tree, model_id)
            if not (tree / "model_index.json").is_file():
                raise RuntimeError("Anima conversion did not produce model_index.json")
            (tree / marker.name).write_text(recipe, encoding="utf-8")
            tree.rename(target)
    log.info("converted model cached: %s", target)
    return str(target)
