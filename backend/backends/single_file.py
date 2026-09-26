"""Convert catalogue checkpoints that have no Diffusers component tree.

Conversion is CPU-only and cached once. Normal loading still goes through
PipelineQuantizationConfig: the pipeline single-file loader does not apply
its per-component quant_mapping to the CLIP encoders.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

log = logging.getLogger(__name__)

ILLUSTRIOUS_ID = "OnomaAIResearch/Illustrious-XL-v2.0"
WAI_ID = "LyliaEngine/waiIllustriousSDXL_v170"
# Pin both weights and descriptors so a changed upstream main cannot silently
# change this model's scheduler, conversion or recorded generation settings.
ILLUSTRIOUS_SOURCE = {
    "repo_id": ILLUSTRIOUS_ID,
    "filename": "Illustrious-XL-v2.0.safetensors",
    "revision": "69459c1fe6f46db41ab31e6114f05acc0e06bcaa",
}
# This is a transport mirror of WAI0731's v17 checkpoint, not an official
# author repository. Its misleading Hub LoRA metadata is not a loader recipe.
WAI_SOURCE = {
    "repo_id": WAI_ID,
    "filename": "waiIllustriousSDXL_v170.safetensors",
    "revision": "5ef4e2da7173a160ad04aebcaa2fdcd6d20ed792",
}
WAI_SHA256 = "f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04"
SDXL_CONFIG = {
    "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "revision": "462165984030d82259a11f4367a4eed129e94a7b",
}
RECIPE = {
    "version": 1,
    "source": ILLUSTRIOUS_SOURCE,
    "config": SDXL_CONFIG,
    "dtype": "bfloat16",
    "scheduler": "EulerAncestralDiscreteScheduler",
    "prediction_type": "epsilon",
}
WAI_RECIPE = {
    **RECIPE,
    "source": WAI_SOURCE,
    "source_sha256": WAI_SHA256,
}


@dataclass(frozen=True)
class _Checkpoint:
    name: str
    slug: str
    recipe: dict


_CHECKPOINTS = {
    ILLUSTRIOUS_ID.lower(): _Checkpoint(
        "Illustrious XL v2.0-STABLE", "illustrious-xl-v2.0-stable", RECIPE,
    ),
    WAI_ID.lower(): _Checkpoint("WAI Illustrious v17", "wai-illustrious-v17", WAI_RECIPE),
}


def cache_root() -> Path:
    from huggingface_hub.constants import HF_HOME

    return Path(HF_HOME) / "local-diffusion" / "converted"


def _convert(destination: Path, model_id: str = ILLUSTRIOUS_ID) -> None:
    import diffusers
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    recipe = _CHECKPOINTS[model_id.lower()].recipe
    # Descriptors and tokenizers only: every model weight comes from the
    # selected checkpoint, never from the SDXL base model.
    config = snapshot_download(
        **recipe["config"],
        allow_patterns=["*.json", "tokenizer/*.txt", "tokenizer_2/*.txt"],
    )
    checkpoint = hf_hub_download(**recipe["source"])
    expected_hash = recipe.get("source_sha256")
    if expected_hash:
        with open(checkpoint, "rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"{model_id} checkpoint SHA-256 does not match the pinned release")
    pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
        checkpoint, config=config, local_files_only=True,
        dtype=torch.bfloat16, add_watermarker=False,
    )
    # WAI's unused final CLIP-L layer contains NaNs. Keep its weights intact:
    # SDXL takes the preceding hidden state and pools only encoder 2. The real
    # encode_prompt regression in test_wai_conditioning checks this for all
    # three quantization modes; no replacement encoder is involved.
    # Both entries recommend Euler A. Fix epsilon prediction and persist the
    # scheduler with the component tree, before ordinary quantized loading.
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, prediction_type="epsilon",
    )
    pipe.save_pretrained(destination, safe_serialization=True)


def _target(model_id: str = ILLUSTRIOUS_ID) -> tuple[Path, Path, str]:
    """The converted tree this recipe names, its marker and the recipe text."""
    entry = _CHECKPOINTS[model_id.lower()]
    recipe = json.dumps(entry.recipe, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    target = cache_root() / f"{entry.slug}-{digest}"
    return target, target / "local-diffusion-conversion.json", recipe


def _converted(target: Path, marker: Path, recipe: str) -> bool:
    return (
        marker.is_file()
        and (target / "model_index.json").is_file()
        and marker.read_text(encoding="utf-8") == recipe
    )


def converted_tree(model_id: str) -> str | None:
    """The cached tree for a single-file id, or None: never a conversion.

    For the callers that read a checkpoint's configs ahead of the load - the
    adapter fit check answers the listing at page open - and must not be the
    thing that starts a 7 GB download and a CPU conversion. The marker is
    written last, so a read without the lock sees a finished tree or none.
    """
    if model_id.lower() not in _CHECKPOINTS:
        return None
    target, marker, recipe = _target(model_id)
    return str(target) if _converted(target, marker, recipe) else None


def resolve_model(model_id: str, quantization: str) -> str:
    """Return a loadable tree, keeping the caller's public model id unchanged."""
    entry = _CHECKPOINTS.get(model_id.lower())
    if entry is None:
        return model_id
    if quantization == "nunchaku":
        raise RuntimeError(
            f"{entry.name} is not a Nunchaku Lite checkpoint. "
            "Use LOCAL_DIFFUSION_QUANT=fp8, int8 or none."
        )

    from huggingface_hub.utils import WeakFileLock

    target, marker, recipe = _target(model_id)
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    # Server and CLI processes may ask at the same time. Publish only a fully
    # saved tree; a failed conversion leaves no cache hit for the next load.
    with WeakFileLock(root / f"{target.name}.lock"):
        if _converted(target, marker, recipe):
            return str(target)
        if target.exists():
            raise RuntimeError(f"Incomplete converted model cache; remove {target} and retry.")
        log.info(
            "converting %s on CPU once (6.94 GB download plus about 7 GB cached tree): %s",
            entry.recipe["source"]["repo_id"], target,
        )
        with TemporaryDirectory(prefix=f"{target.name}-", dir=root) as work:
            tree = Path(work) / "tree"
            _convert(tree, model_id)
            if not (tree / "model_index.json").is_file():
                raise RuntimeError("Single-file conversion did not produce model_index.json")
            (tree / marker.name).write_text(recipe, encoding="utf-8")
            tree.rename(target)
    log.info("converted model cached: %s", target)
    return str(target)
