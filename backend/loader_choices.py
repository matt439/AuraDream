"""Quantization, offload and Nunchaku format choices, worded once.

The terminal menu (`launch.py`) and the browser picker (`model_selection.py`)
print the same descriptions. Only figures that hold for any catalogue
checkpoint belong here; per-model memory and timing are in docs/verified.md.
This module imports nothing, so launch.py can read it before config.
"""
from __future__ import annotations

_FOR_16_GB = "recommended for a 16 GB card"

# (value, short browser label, description).
QUANTIZATION_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("fp8", "FP8", f"8-bit float weights; {_FOR_16_GB}."),
    ("int8", "INT8", "8-bit integer weights, the same size as FP8; where compared (not FLUX.1, Krea 2, klein 9B or Qwen-Image), a little faster and usually nearer BF16."),
    ("none", "BF16", "Full bfloat16 precision; the most VRAM, and FLUX-size models may not fit 16 GB."),
    ("nunchaku", "Nunchaku 4-bit", "Pre-quantized 4-bit lite-infer builds, the smallest; Linux or WSL only."),
)
OFFLOAD_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("model", "Model offload", f"One component on the GPU at a time; {_FOR_16_GB}."),
    ("sequential", "Sequential", "Smaller pieces move to the GPU as needed; the least VRAM, but much slower."),
    ("none", "No offload", "Everything stays on the GPU; the most VRAM, 24 GB+ for FLUX-size models."),
)

QUANTIZATIONS: tuple[tuple[str, str], ...] = tuple((v, d) for v, _, d in QUANTIZATION_CHOICES)
OFFLOADS: tuple[tuple[str, str], ...] = tuple((v, d) for v, _, d in OFFLOAD_CHOICES)
FORMATS: tuple[tuple[str, str], ...] = (
    ("nvfp4", "RTX 50 (Blackwell)"),
    ("int4", "RTX 20 to 40"),
)
