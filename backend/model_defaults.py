"""Recommended generation parameters per checkpoint.

Pipeline signatures do carry defaults, but they describe the *family* rather than
the distilled variant: `Krea2Pipeline` reports 28 steps at CFG 4.5, which suits
Krea 2 Raw and is badly wrong for Turbo (8 steps, CFG 0). So key off the id.

`mu` is deliberately None outside Turbo. It is Krea's flow-shift knob, applied
here by handing the pipeline an explicit `sigmas` schedule; FLUX derives its own
resolution-dependent shift, and overriding it with Turbo's constant degrades the
result.
"""
from __future__ import annotations

from typing import Any

# Matched in order against a normalised model id.
_PROFILES: list[tuple[str, dict[str, Any]]] = [
    ("krea-2-turbo", {"steps": 8, "guidance_scale": 0.0, "mu": 1.15}),
    ("krea-2-raw", {"steps": 52, "guidance_scale": 3.5, "mu": None}),
    ("flux", {"steps": 28, "guidance_scale": 3.5, "mu": None}),
]
_FALLBACK: dict[str, Any] = {"steps": 28, "guidance_scale": 3.5, "mu": None}


def defaults_for(model_id: str) -> dict[str, Any]:
    key = model_id.lower().replace("_", "-")
    for token, profile in _PROFILES:
        if token in key:
            return dict(profile)
    return dict(_FALLBACK)
