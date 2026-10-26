"""Pinned, strictly converted NewBie Exp0.1 components; no remote model code."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from .. import config as _config  # noqa: F401 - before Diffusers
from .single_file import HubRepo, cache_root, sweep_work

log = logging.getLogger(__name__)
MODEL_ID = "NewBie-AI/NewBie-image-Exp0.1"
SOURCE: HubRepo = {"repo_id": MODEL_ID, "revision": "4f7825a1a7e1ed5c1882a52e3c51d1119a43e298"}
PATTERNS = [
    "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json", "text_encoder/gemma3-4b-it.safetensors",
    "text_encoder/*.json", "text_encoder/*.model",
    "clip_model/config.json", "clip_model/jina-clip-v2.safetensors", "clip_model/*.json",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
]
RECIPE = {
    "version": 1, "source": SOURCE, "dtype": "bfloat16",
    "text_encoder": "Gemma3TextModel", "text_encoder_2": "NewbieJinaTextEncoder",
    "transformer": "NewbieTransformer2DModel",
    "scheduler": "FlowMatchEulerDiscreteScheduler", "shift": 6.0,
    "prompt_prefix": "", "cfg_normalization": "whole_sample_cap",
    "gemma_max_length": 512, "jina_max_length": 2048,
}


def _load_gemma(checkpoint, config):
    from accelerate import init_empty_weights
    from transformers import Gemma3TextConfig, Gemma3TextModel

    state = {}
    for name, tensor in checkpoint.items():
        if name == "spiece_model":
            continue  # embedded tokenizer, independently loaded from the same pin
        if not name.startswith("model."):
            raise ValueError(f"Unexpected NewBie Gemma tensor: {name}")
        state[name.removeprefix("model.")] = tensor
    with init_empty_weights():
        model = Gemma3TextModel(Gemma3TextConfig.from_dict(config))
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def _convert(destination: Path) -> None:
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from .newbie_pipeline import NewbiePipeline
    from .newbie_text_encoder import load_jina_text_encoder
    from .newbie_transformer import load_original_transformer

    source = Path(snapshot_download(**SOURCE, allow_patterns=PATTERNS))
    config = json.loads((source / "text_encoder/config.json").read_text(encoding="utf-8"))
    transformer = load_original_transformer(
        load_file(source / "transformer/diffusion_pytorch_model.safetensors", device="cpu"),
    )
    gemma = _load_gemma(
        load_file(source / "text_encoder/gemma3-4b-it.safetensors", device="cpu"),
        config["text_config"],
    )
    jina = load_jina_text_encoder(
        load_file(source / "clip_model/jina-clip-v2.safetensors", device="cpu"),
    )
    pipe = NewbiePipeline(
        transformer=transformer.to(dtype=torch.bfloat16),
        text_encoder=gemma.to(dtype=torch.bfloat16),
        text_encoder_2=jina.to(dtype=torch.bfloat16),
        tokenizer=AutoTokenizer.from_pretrained(source / "text_encoder", local_files_only=True),
        tokenizer_2=AutoTokenizer.from_pretrained(source / "clip_model", local_files_only=True),
        vae=AutoencoderKL.from_pretrained(source / "vae", dtype=torch.bfloat16, local_files_only=True),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=6.0, use_dynamic_shifting=False),
    )
    pipe.save_pretrained(destination, safe_serialization=True)


def _target() -> tuple[Path, Path, str]:
    recipe = json.dumps(RECIPE, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    target = cache_root() / f"newbie-exp0.1-{digest}"
    return target, target / "local-diffusion-conversion.json", recipe


def _converted(target: Path, marker: Path, recipe: str) -> bool:
    return (
        marker.is_file() and (target / "model_index.json").is_file()
        and marker.read_text(encoding="utf-8") == recipe
    )


def converted_tree(model_id: str) -> str | None:
    if model_id.lower() != MODEL_ID.lower():
        return None
    target, marker, recipe = _target()
    return str(target) if _converted(target, marker, recipe) else None


def resolve_model(model_id: str, quantization: str) -> str:
    if model_id.lower() != MODEL_ID.lower():
        return model_id
    if quantization == "nunchaku":
        raise RuntimeError(
            "NewBie Image Exp0.1 is not a Nunchaku Lite checkpoint. "
            "Use LOCAL_DIFFUSION_QUANT=fp8, int8 or none."
        )
    from huggingface_hub.utils import WeakFileLock

    target, marker, recipe = _target()
    target.parent.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(target.parent / f"{target.name}.lock"):
        # Converted or not: leftovers from before a finished conversion count too.
        sweep_work(target)
        if _converted(target, marker, recipe):
            return str(target)
        if target.exists():
            raise RuntimeError(f"Incomplete converted model cache; remove {target} and retry.")
        log.info("converting pinned NewBie components on CPU once: %s", target)
        # A cleanup that meets an open file leaves the rest to the next load's
        # sweep rather than raising over the conversion's own error.
        with TemporaryDirectory(
            prefix=f"{target.name}-", dir=target.parent, ignore_cleanup_errors=True,
        ) as work:
            tree = Path(work) / "tree"
            _convert(tree)
            if not (tree / "model_index.json").is_file():
                raise RuntimeError("NewBie conversion did not produce model_index.json")
            (tree / marker.name).write_text(recipe, encoding="utf-8")
            tree.rename(target)
    return str(target)
