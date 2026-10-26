"""Pinned CPU conversion of NetaYume v4 into the normal Lumina 2 loader.

Only the denoiser is fine-tuned upstream. The official Lumina 2 parent
supplies its standard Gemma 2 encoder, FLUX VAE, tokenizers and descriptors.
Quantization is applied later by the normal per-component loader.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .. import config as _config  # noqa: F401 - before any diffusers imports
from .single_file import HubFile, HubRepo, cache_root, sweep_work

log = logging.getLogger(__name__)

MODEL_ID = "duongve/NetaYume-Lumina-Image-2.0"
SOURCE: HubFile = {
    "repo_id": MODEL_ID,
    "filename": "Unet/v4/NetaYumev4_unet.safetensors",
    "revision": "69c29c441119d43405d8373c654b16827d3cf46e",
}
SHARED: HubRepo = {
    "repo_id": "Alpha-VLLM/Lumina-Image-2.0",
    "revision": "53504abd8178b30685b6c4c7a4cd181ff78b73e9",
}
SHARED_PATTERNS = [
    "transformer/config.json", "scheduler/scheduler_config.json",
    "text_encoder/*", "vae/*", "tokenizer/*",
]
# The author's anime instruction. Lumina2Pipeline adds its <Prompt Start>
# separator itself. This fixed part of the recipe never replaces the image
# prompt recorded in the gallery, and is sent explicitly after every load.
SYSTEM_PROMPT = "You are an assistant designed to generate anime images based on textual prompts."
RECIPE = {
    "version": 1,
    "source": SOURCE,
    "shared": SHARED,
    "dtype": "bfloat16",
    "scheduler": "FlowMatchEulerDiscreteScheduler",
    "shift": 6.0,
    "system_prompt": SYSTEM_PROMPT,
}


def _load_transformer(config: Path, checkpoint: dict[str, Any]) -> Any:
    """Use upstream's Lumina mapping, requiring every converted tensor to fit."""
    from accelerate import init_empty_weights
    from diffusers import Lumina2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_lumina2_to_diffusers

    # The pinned split file uses the original unprefixed keys. The upstream
    # converter removes norm_final.weight, an unused original Lumina tensor.
    state = convert_lumina2_to_diffusers(dict(checkpoint))
    with init_empty_weights():
        transformer = Lumina2Transformer2DModel.from_config(
            Lumina2Transformer2DModel.load_config(config),
        )
    transformer.load_state_dict(state, strict=True, assign=True)
    return transformer


def _convert(destination: Path) -> None:
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, Lumina2Pipeline
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    from transformers import Gemma2Model, GemmaTokenizerFast

    shared = Path(snapshot_download(**SHARED, allow_patterns=SHARED_PATTERNS))
    checkpoint = load_file(hf_hub_download(**SOURCE), device="cpu")
    transformer = _load_transformer(shared / "transformer", checkpoint)
    del checkpoint
    pipe = Lumina2Pipeline(
        transformer=transformer.to(dtype=torch.bfloat16),
        text_encoder=Gemma2Model.from_pretrained(shared / "text_encoder", dtype=torch.bfloat16, local_files_only=True),
        vae=AutoencoderKL.from_pretrained(shared / "vae", dtype=torch.bfloat16, local_files_only=True),
        tokenizer=GemmaTokenizerFast.from_pretrained(shared / "tokenizer", local_files_only=True),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(shared / "scheduler", local_files_only=True),
    )
    pipe.save_pretrained(destination, safe_serialization=True)


def _target() -> tuple[Path, Path, str]:
    recipe = json.dumps(RECIPE, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    target = cache_root() / f"netayume-lumina-v4-{digest}"
    return target, target / "local-diffusion-conversion.json", recipe


def _converted(target: Path, marker: Path, recipe: str) -> bool:
    return (
        marker.is_file()
        and (target / "model_index.json").is_file()
        and marker.read_text(encoding="utf-8") == recipe
    )


def converted_tree(model_id: str) -> str | None:
    """Look up an existing tree without initiating downloads or conversion."""
    if model_id.lower() != MODEL_ID.lower():
        return None
    target, marker, recipe = _target()
    return str(target) if _converted(target, marker, recipe) else None


def resolve_model(model_id: str, quantization: str) -> str:
    """Resolve only the catalogue's pinned v4 identity; keep public metadata unchanged."""
    if model_id.lower() != MODEL_ID.lower():
        return model_id
    if quantization == "nunchaku":
        raise RuntimeError(
            "NetaYume Lumina v4 is not a Nunchaku Lite checkpoint. "
            "Use LOCAL_DIFFUSION_QUANT=fp8, int8 or none."
        )
    from huggingface_hub.utils import WeakFileLock

    target, marker, recipe = _target()
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(root / f"{target.name}.lock"):
        # Converted or not: leftovers from before a finished conversion count too.
        sweep_work(target)
        if _converted(target, marker, recipe):
            return str(target)
        if target.exists():
            raise RuntimeError(f"Incomplete converted model cache; remove {target} and retry.")
        log.info("converting official NetaYume Lumina v4 on CPU once: %s", target)
        # A cleanup that meets an open file leaves the rest to the next load's
        # sweep rather than raising over the conversion's own error.
        with TemporaryDirectory(prefix=f"{target.name}-", dir=root, ignore_cleanup_errors=True) as work:
            tree = Path(work) / "tree"
            _convert(tree)
            if not (tree / "model_index.json").is_file():
                raise RuntimeError("NetaYume conversion did not produce model_index.json")
            (tree / marker.name).write_text(recipe, encoding="utf-8")
            tree.rename(target)
    log.info("converted model cached: %s", target)
    return str(target)
