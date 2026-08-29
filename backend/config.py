"""Runtime settings, overridable by environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Which checkpoint to serve. Krea 2 Turbo is the 8-step distilled variant.

    # krea/Krea-2-Turbo
    # black-forest-labs/FLUX.1-Krea-dev
    model_id: str = os.getenv("KREA_MODEL_ID", "black-forest-labs/FLUX.1-Krea-dev")

    # "fp8"  - torchao float8 weight-only on the transformer (~12GB, fits 16GB cards)
    # "int8" - torchao int8 weight-only, smaller still, slightly softer output
    # "none" - full bfloat16, needs ~25GB VRAM
    quantization: str = os.getenv("KREA_QUANT", "fp8")

    # "model"      - one component on the GPU at a time (recommended for <=16GB)
    # "sequential" - per-submodule offload, lowest VRAM, much slower
    # "none"       - keep everything resident
    offload: str = os.getenv("KREA_OFFLOAD", "model")

    device: str = os.getenv("KREA_DEVICE", "cuda")
    load_on_startup: bool = _env_bool("KREA_PRELOAD", False)

    outputs_dir: Path = ROOT / "outputs"
    db_path: Path = ROOT / "krea.db"
    web_dir: Path = ROOT / "web"

    host: str = os.getenv("KREA_HOST", "127.0.0.1")
    port: int = int(os.getenv("KREA_PORT", "7860"))

    # Guard rails so a typo in the UI cannot ask for a 16k image.
    max_side: int = 2048
    max_pixels: int = 2048 * 2048


settings = Settings()
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
