"""Recommended generation parameters per checkpoint.

Pipeline signatures do carry defaults, but they describe the *family* rather than
the distilled variant: `Krea2Pipeline` reports 28 steps at CFG 4.5, which suits
Krea 2 Raw and is badly wrong for Turbo (8 steps, CFG 0). So key off the id.

`mu` is deliberately None outside Turbo. It is Krea's flow-shift knob, applied
by overriding the shift the pipeline would otherwise derive for itself; FLUX
derives a resolution-dependent one, and replacing it with Turbo's constant
degrades the result.

`cfg_threshold` is the guidance scale above which a negative prompt does
anything at all: FLUX bakes `guidance_scale` into the model and only runs a real
uncond pass when `true_cfg_scale > 1`, while Krea 2 guides from 0. It is a hint
for clients that have not loaded a pipeline yet - `/api/status` replaces it with
what the loaded pipeline actually exposes, and the backend gates on that.

`hints` is the same knowledge as prose, for the UI's tooltips: what steps, CFG
and mu mean *on this checkpoint*, which is not the same sentence twice. They are
format strings over the profile's own numbers, so the text cannot drift from the
values the controls are seeded with, and `hints_for` takes the live threshold the
way `/api/status` does.
"""
from __future__ import annotations

from typing import Any

# Matched in order against a normalised model id.
_PROFILES: list[tuple[str, dict[str, Any]]] = [
    (
        "krea-2-turbo",
        {
            "steps": 8,
            "guidance_scale": 0.0,
            "mu": 1.15,
            "cfg_threshold": 0.0,
            "hints": {
                "steps": (
                    "Denoising steps. Turbo is distilled to resolve in {steps}; past "
                    "about 12 the extra passes cost time without changing much."
                ),
                "guidance_scale": (
                    "Classifier-free guidance. Turbo is trained for {guidance_scale:g}. "
                    "This pipeline guides from 0 up, so anything above {cfg_threshold:g} "
                    "brings the negative prompt in and adds an uncond pass per step "
                    "(~2x slower)."
                ),
                "mu": (
                    "Flow-matching timestep shift, replacing the one the scheduler would "
                    "derive. {mu:g} is Turbo's trained value; higher spends more steps at "
                    "high noise (looser composition), lower holds detail. Blank leaves "
                    "the pipeline's own shift alone."
                ),
            },
        },
    ),
    (
        "krea-2-raw",
        {
            "steps": 52,
            "guidance_scale": 3.5,
            "mu": None,
            "cfg_threshold": 0.0,
            "hints": {
                "steps": (
                    "Denoising steps. Raw is undistilled and wants around {steps}; well "
                    "below that it stays soft."
                ),
                "guidance_scale": (
                    "Classifier-free guidance, {guidance_scale:g} recommended. Krea 2 "
                    "guides from 0 up, so any value above {cfg_threshold:g} uses the "
                    "negative prompt and costs an uncond pass per step (~2x slower)."
                ),
                "mu": (
                    "Flow-matching timestep shift. Leave blank on Raw - the pipeline "
                    "derives its own, and a value here overrides it outright."
                ),
            },
        },
    ),
    (
        "flux",
        {
            "steps": 28,
            "guidance_scale": 3.5,
            "mu": None,
            "cfg_threshold": 1.0,
            "hints": {
                "steps": (
                    "Denoising steps. FLUX is tuned around {steps}; below about 20 it "
                    "degrades, above about 40 it gains little."
                ),
                "guidance_scale": (
                    "On FLUX this is the distilled guidance value baked into the model "
                    "({guidance_scale:g} recommended), not true CFG. A real uncond pass - "
                    "and with it the negative prompt - only starts above "
                    "{cfg_threshold:g}, at roughly 2x the time per step."
                ),
                "mu": (
                    "Flow-matching timestep shift. FLUX derives its own from the image "
                    "resolution, and forcing a constant over it degrades the result: "
                    "leave this blank."
                ),
            },
        },
    ),
]
_FALLBACK: dict[str, Any] = {
    "steps": 28,
    "guidance_scale": 3.5,
    "mu": None,
    "cfg_threshold": 1.0,
    "hints": {
        "steps": (
            "Denoising steps. {steps} is a safe starting point for an unrecognised "
            "checkpoint; more steps cost proportionally more time."
        ),
        "guidance_scale": (
            "Classifier-free guidance, {guidance_scale:g} to start. Above "
            "{cfg_threshold:g} the negative prompt applies, at roughly 2x the time "
            "per step."
        ),
        "mu": (
            "Flow-matching timestep shift. Leave blank unless the checkpoint is known "
            "to want a fixed one - otherwise the pipeline derives its own."
        ),
    },
}


def _profile_for(model_id: str) -> dict[str, Any]:
    key = model_id.lower().replace("_", "-")
    for token, profile in _PROFILES:
        if token in key:
            return profile
    return _FALLBACK


def defaults_for(model_id: str) -> dict[str, Any]:
    """The numbers only - callers put these straight into a request or a field."""
    return {k: v for k, v in _profile_for(model_id).items() if k != "hints"}


def hints_for(model_id: str, cfg_threshold: float | None = None) -> dict[str, str]:
    """Tooltip text for steps/CFG/mu on this checkpoint.

    `cfg_threshold` overrides the profile's guess with what a loaded pipeline
    actually exposes, so the tooltip and the greyed-out negative prompt field
    cannot disagree about where guidance switches on.
    """
    profile = _profile_for(model_id)
    values = {k: v for k, v in profile.items() if k != "hints"}
    if cfg_threshold is not None:
        values["cfg_threshold"] = cfg_threshold
    return {key: text.format(**values) for key, text in profile["hints"].items()}
