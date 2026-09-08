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

`negative_prompt_supported` says whether the pipeline takes a negative prompt at
all. FLUX.2 [klein]'s does not - its `__call__` has no `negative_prompt`, and the
undistilled base runs its uncond pass against an empty string - so the field is
greyed out whatever the CFG, rather than promising an effect that cannot happen.
Like the threshold, it is a guess from the id until a pipeline's parameter list
replaces it.

`true_cfg_supported` says whether real CFG is a knob of its own. On FLUX.1 the
`guidance_scale` is a value the distilled model takes as an input and the
uncond pass is `true_cfg_scale`, a separate number; everywhere else here the
`guidance_scale` *is* that pass. Where it is separate the threshold above is
compared against it rather than against the guidance value, and the UI offers
a True CFG box. A guess from the id until the parameter list answers.

`hints` is the same knowledge as prose, for the UI's tooltips: what steps, CFG
and mu mean *on this checkpoint*, which is not the same sentence twice. They are
format strings over the profile's own numbers, so the text cannot drift from the
values the controls are seeded with, and `hints_for` takes the live threshold the
way `/api/status` does.
"""
from __future__ import annotations

from typing import Any

# What the server runs when nothing says otherwise. Here rather than in
# config.py because launch.py needs the same three values for its menu and
# must not import config - whose frozen settings read the environment at
# import, which is the very thing the menu runs ahead of.
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
DEFAULT_QUANT = "fp8"
DEFAULT_OFFLOAD = "model"

# The vocabularies config.py validates KREA_QUANT and KREA_OFFLOAD against, and
# launch.py builds its menus from. An unknown value used to select the heaviest
# path silently - `KREA_QUANT=fp16` loaded 25 GB of bf16 and OOMed at the first
# image, `KREA_OFFLOAD=cpu` kept everything resident - while /api/status echoed
# the typo back as if it meant something.
QUANT_MODES: tuple[str, ...] = ("fp8", "int8", "none", "nunchaku")
OFFLOAD_MODES: tuple[str, ...] = ("model", "sequential", "none")

# Matched in order against a normalised model id.
_PROFILES: list[tuple[str, dict[str, Any]]] = [
    (
        "krea-2-turbo",
        {
            "steps": 8,
            "guidance_scale": 0.0,
            "mu": 1.15,
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "true_cfg_supported": False,
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
            "negative_prompt_supported": True,
            "true_cfg_supported": False,
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
    # Both klein entries must precede "flux": the ids are FLUX.2-klein-*.
    (
        "klein-base",
        {
            "steps": 50,
            "guidance_scale": 4.0,
            "mu": None,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": False,
            "true_cfg_supported": False,
            "hints": {
                "steps": (
                    "Denoising steps. klein base is undistilled and wants around {steps}; "
                    "well below that it stays soft."
                ),
                "guidance_scale": (
                    "Real classifier-free guidance, {guidance_scale:g} recommended. Above "
                    "{cfg_threshold:g} the pipeline runs an uncond pass per step (~2x "
                    "slower) - against an empty prompt: this pipeline takes no negative "
                    "prompt."
                ),
                "mu": (
                    "Flow-matching timestep shift. klein derives its own from the image "
                    "size and the step count; a value here overrides it outright. Leave "
                    "blank."
                ),
            },
        },
    ),
    (
        "klein",
        {
            "steps": 4,
            "guidance_scale": 1.0,
            "mu": None,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": False,
            "true_cfg_supported": False,
            "hints": {
                "steps": (
                    "Denoising steps. klein is distilled to resolve in {steps}; more adds "
                    "time, not detail."
                ),
                "guidance_scale": (
                    "Ignored on the distilled klein - guidance is baked into the model, "
                    "and the pipeline warns above {cfg_threshold:g}. Leave it at "
                    "{guidance_scale:g}. There is no negative prompt either."
                ),
                "mu": (
                    "Flow-matching timestep shift. klein derives its own from the image "
                    "size and the step count; a value here overrides it outright. Leave "
                    "blank."
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
            "negative_prompt_supported": True,
            "true_cfg_supported": True,
            "hints": {
                "steps": (
                    "Denoising steps. FLUX is tuned around {steps}; below about 20 it "
                    "degrades, above about 40 it gains little."
                ),
                "guidance_scale": (
                    "On FLUX this is the distilled guidance value the model takes as an "
                    "input ({guidance_scale:g} recommended), not classifier-free "
                    "guidance. The negative prompt is driven by True CFG, the box "
                    "beside this one."
                ),
                "true_cfg": (
                    "Real classifier-free guidance, kept apart from the guidance value: "
                    "above {cfg_threshold:g} the pipeline runs a second, unconditioned "
                    "pass per step against the negative prompt (~2x slower). Blank is "
                    "off. Two ways to use it: keep {guidance_scale:g} and add a mild "
                    "1.5, or drop the guidance value to about 1 and let 3-5 here do the "
                    "work."
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
    "negative_prompt_supported": True,
    "true_cfg_supported": False,
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


# What the CFG tooltip says when the loaded pipeline disagrees with the profile
# about the negative prompt: three of the five profiles' CFG hints assert
# something about it, and a checkpoint the id table does not recognise but
# whose pipeline takes none would otherwise be greyed out "because this
# checkpoint takes no negative prompt" under a badge saying it applies above 1.
_NEGATIVE_HINTS = {
    False: (
        "Classifier-free guidance, {guidance_scale:g} recommended. This pipeline takes no "
        "negative prompt; above {cfg_threshold:g} it runs an uncond pass per step "
        "(~2x slower) against an empty one."
    ),
    True: (
        "Classifier-free guidance, {guidance_scale:g} recommended. Above {cfg_threshold:g} "
        "the negative prompt applies, at roughly 2x the time per step."
    ),
}


# The True CFG box's text where the knob does not exist: every profile but
# FLUX.1's, and any loaded pipeline whose `__call__` has no `true_cfg_scale`.
_NO_TRUE_CFG_HINT = (
    "Not a separate knob on this checkpoint: its CFG box is real classifier-free "
    "guidance already, and the negative prompt follows that."
)
# And where a pipeline the id table did not expect turns out to have one.
_TRUE_CFG_HINT = (
    "Real classifier-free guidance, kept apart from the guidance value: above "
    "{cfg_threshold:g} the pipeline runs an uncond pass per step against the negative "
    "prompt (~2x slower). Blank is off."
)


def hints_for(
    model_id: str,
    cfg_threshold: float | None = None,
    negative_prompt_supported: bool | None = None,
    true_cfg_supported: bool | None = None,
) -> dict[str, str]:
    """Tooltip text for steps/CFG/mu/true CFG on this checkpoint.

    `cfg_threshold`, `negative_prompt_supported` and `true_cfg_supported`
    override the profile's guesses with what a loaded pipeline actually
    exposes, so the tooltip and the greyed-out fields cannot disagree about
    where guidance switches on, whether a negative prompt does anything at
    all, or which box drives it.
    """
    profile = _profile_for(model_id)
    values = {k: v for k, v in profile.items() if k != "hints"}
    if cfg_threshold is not None:
        values["cfg_threshold"] = cfg_threshold
    hints = dict(profile["hints"])
    if (
        negative_prompt_supported is not None
        and negative_prompt_supported != profile["negative_prompt_supported"]
    ):
        hints["guidance_scale"] = _NEGATIVE_HINTS[negative_prompt_supported]
    separate = profile["true_cfg_supported"] if true_cfg_supported is None else true_cfg_supported
    if not separate:
        hints["true_cfg"] = _NO_TRUE_CFG_HINT
    elif "true_cfg" not in hints:
        hints["true_cfg"] = _TRUE_CFG_HINT
    return {key: text.format(**values) for key, text in hints.items()}
