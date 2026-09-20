"""Convert catalogue checkpoints that have no Diffusers component tree.

Conversion is CPU-only and cached once. Normal loading still goes through
PipelineQuantizationConfig: the pipeline single-file loader does not apply
its per-component quant_mapping to the CLIP encoders.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

log = logging.getLogger(__name__)

ILLUSTRIOUS_ID = "OnomaAIResearch/Illustrious-XL-v2.0"
# Pin both weights and descriptors so a changed upstream main cannot silently
# change this model's scheduler, conversion or recorded generation settings.
ILLUSTRIOUS_SOURCE = {
    "repo_id": ILLUSTRIOUS_ID,
    "filename": "Illustrious-XL-v2.0.safetensors",
    "revision": "69459c1fe6f46db41ab31e6114f05acc0e06bcaa",
}
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


def cache_root() -> Path:
    from huggingface_hub.constants import HF_HOME

    return Path(HF_HOME) / "local-diffusion" / "converted"


def _convert(destination: Path) -> None:
    import diffusers
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    # Descriptors and tokenizers only: every model weight comes from Onoma's
    # checkpoint, never from the SDXL base model or a community conversion.
    config = snapshot_download(
        **SDXL_CONFIG,
        allow_patterns=["*.json", "tokenizer/*.txt", "tokenizer_2/*.txt"],
    )
    checkpoint = hf_hub_download(**ILLUSTRIOUS_SOURCE)
    pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
        checkpoint, config=config, local_files_only=True,
        dtype=torch.bfloat16, add_watermarker=False,
    )
    # The authors use Euler A, >20 steps, CFG 5-7.5 (report appendix B.2).
    # Fixed for this catalogue model, and persisted with its component tree.
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, prediction_type="epsilon",
    )
    pipe.save_pretrained(destination, safe_serialization=True)


def _target() -> tuple[Path, Path, str]:
    """The converted tree this recipe names, its marker and the recipe text."""
    recipe = json.dumps(RECIPE, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    target = cache_root() / f"illustrious-xl-v2.0-stable-{digest}"
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
    if model_id.lower() != ILLUSTRIOUS_ID.lower():
        return None
    target, marker, recipe = _target()
    return str(target) if _converted(target, marker, recipe) else None


def resolve_model(model_id: str, quantization: str) -> str:
    """Return a loadable tree, keeping the caller's public model id unchanged."""
    if model_id.lower() != ILLUSTRIOUS_ID.lower():
        return model_id
    if quantization == "nunchaku":
        raise RuntimeError(
            "Illustrious XL v2.0-STABLE is not a Nunchaku Lite checkpoint. "
            "Use LOCAL_DIFFUSION_QUANT=fp8, int8 or none."
        )

    from huggingface_hub.utils import WeakFileLock

    target, marker, recipe = _target()
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
            ILLUSTRIOUS_ID, target,
        )
        with TemporaryDirectory(prefix=f"{target.name}-", dir=root) as work:
            tree = Path(work) / "tree"
            _convert(tree)
            if not (tree / "model_index.json").is_file():
                raise RuntimeError("Single-file conversion did not produce model_index.json")
            (tree / marker.name).write_text(recipe, encoding="utf-8")
            tree.rename(target)
    log.info("converted model cached: %s", target)
    return str(target)
