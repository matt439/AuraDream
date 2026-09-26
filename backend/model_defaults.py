"""Recommended generation parameters per checkpoint.

Pipeline signatures do carry defaults, but they describe the *family* rather than
the distilled variant: `Krea2Pipeline` reports 28 steps at CFG 4.5, which suits
Krea 2 Raw and is badly wrong for Turbo (8 steps, CFG 0). So key off the id.

The table is first-match, so a distilled variant whose id also contains its
family's token (klein before flux, schnell before flux, Z-Image Turbo before
Z-Image, ERNIE-Image Turbo before ERNIE-Image) sits ahead of it.

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

`reference_supported` says whether the pipeline conditions on reference images
at all. FLUX.2 [klein]'s `__call__` takes `image` as its first parameter and
concatenates the encoded references onto the latent sequence; nothing else here
has one, so the control is greyed out for the checkpoint rather than accepting
input that would be dropped. A guess from the id until the parameter list
answers, like the two above.

`mu_supported` says whether the checkpoint uses a flow-matching scheduler whose
sigmas can be shifted. Modern SDXL exposes a custom `sigmas` argument too, but
its Euler scheduler is not flow matching; treating the argument alone as proof
would replace Animagine's noise schedule with an unrelated flow schedule.

`hints` is the same knowledge as prose, for the UI's tooltips: what steps, CFG
and mu mean *on this checkpoint*, which is not the same sentence twice. They are
format strings over the profile's own numbers, so the text cannot drift from the
values the controls are seeded with, and `hints_for` takes the live threshold the
way `/api/status` does.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import guides

# What the server runs when nothing says otherwise. Here rather than in
# config.py because launch.py needs the same three values for its menu and
# must not import config - whose frozen settings read the environment at
# import, which is the very thing the menu runs ahead of.
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
DEFAULT_QUANT = "fp8"
DEFAULT_OFFLOAD = "model"

# The vocabularies config.py validates LOCAL_DIFFUSION_QUANT and LOCAL_DIFFUSION_OFFLOAD against, and
# launch.py builds its menus from. An unknown value used to select the heaviest
# path silently - `LOCAL_DIFFUSION_QUANT=fp16` loaded 25 GB of bf16 and OOMed at the first
# image, `LOCAL_DIFFUSION_OFFLOAD=cpu` kept everything resident - while /api/status echoed
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
            "mu_supported": True,
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.KREA2_TURBO_GUIDE,
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
            "mu_supported": True,
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.KREA2_RAW_GUIDE,
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
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": False,
            "reference_supported": True,
            "true_cfg_supported": False,
            "guide": guides.KLEIN_GUIDE,
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
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": False,
            "reference_supported": True,
            "true_cfg_supported": False,
            "guide": guides.KLEIN_GUIDE,
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
    # And schnell must precede "flux": the id is FLUX.1-schnell. Its transformer
    # has no guidance input at all, so the CFG box is decorative there and only
    # True CFG does anything.
    (
        "schnell",
        {
            "steps": 4,
            "guidance_scale": 0.0,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": True,
            "guide": guides.SCHNELL_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. schnell is distilled to resolve in {steps}; more adds "
                    "time, not detail."
                ),
                "guidance_scale": (
                    "Ignored on schnell: its transformer has no guidance input, so this "
                    "number never reaches the model. Leave it at {guidance_scale:g}. The "
                    "negative prompt is driven by True CFG, the box beside this one."
                ),
                "true_cfg": (
                    "Real classifier-free guidance on a 4-step model: above "
                    "{cfg_threshold:g} the pipeline runs a second, unconditioned pass per "
                    "step against the negative prompt (~2x slower). Blank is off, which "
                    "is how schnell is meant to run."
                ),
                "mu": (
                    "Flow-matching timestep shift. Leave blank: schnell's 4-step schedule "
                    "is not shifted, and a value here overrides it outright."
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
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": True,
            "guide": guides.FLUX1_GUIDE,
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
# Animagine is an SDXL finetune with classic classifier-free guidance and an
# Euler scheduler. The repo's Opt update changed its Diffusers component tree
# and scheduler together; follow that manifest rather than silently replacing
# it with the model card's older Euler Ancestral recommendation.
_PROFILES.append(
    (
        "animagine-xl",
        {
            "steps": 28,
            "guidance_scale": 5.0,
            "mu": None,
            "mu_supported": False,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.ANIMAGINE_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. Animagine recommends 25-28; {steps} is the "
                    "model card's preferred setting."
                ),
                "guidance_scale": (
                    "Real classifier-free guidance. Animagine recommends 4-7, "
                    "with {guidance_scale:g} as its default. Above "
                    "{cfg_threshold:g} the negative prompt applies and adds an "
                    "uncond pass per step (~2x slower)."
                ),
                "mu": (
                    "Unavailable: Animagine uses SDXL's Euler noise schedule, not "
                    "flow matching, so a flow-shift Mu does not apply."
                ),
            },
        },
    )
)
_PROFILES.append(
    (
        "kohaku-xl-zeta",
        {
            "steps": 28,
            "guidance_scale": 5.0,
            "mu": None,
            "mu_supported": False,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.KOHAKU_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. Kohaku XL Zeta recommends 12-50; "
                    "{steps} is this app's starting point."
                ),
                "guidance_scale": (
                    "Real classifier-free guidance. Kohaku XL Zeta recommends "
                    "3.5-6.5; this app starts at {guidance_scale:g}. Above "
                    "{cfg_threshold:g} the negative prompt applies and adds "
                    "an uncond pass per step."
                ),
                "mu": (
                    "Unavailable: Kohaku XL Zeta uses SDXL's Euler noise "
                    "schedule, not flow matching."
                ),
            },
        },
    )
)
_PROFILES.append(
    (
        "illustrious-xl-v2.0",
        {
            "steps": 28,
            "guidance_scale": 5.0,
            "mu": None,
            "mu_supported": False,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.ILLUSTRIOUS_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. The authors use more than 20 with Euler "
                    "Ancestral, 28 in their v2.0 release example; {steps} is "
                    "this app's starting point."
                ),
                "guidance_scale": (
                    "Real classifier-free guidance. The authors use CFG 5-7.5, "
                    "6.5 in their v2.0 release example; this app starts at "
                    "{guidance_scale:g}. Above {cfg_threshold:g} the negative "
                    "prompt applies and adds an uncond pass per step."
                ),
                "mu": (
                    "Unavailable: Illustrious uses SDXL's Euler Ancestral noise "
                    "schedule, not flow matching."
                ),
            },
        },
    )
)
# Qwen-Image is Krea 2's parent and looks like FLUX.1 to the controls: the
# transformer takes no guidance input, and the uncond pass is `true_cfg_scale`
# - which the pipeline only runs when a negative prompt is present.
_PROFILES.append(
    (
        "qwen-image",
        {
            "steps": 50,
            "guidance_scale": 1.0,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": True,
            "guide": guides.QWEN_IMAGE_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. Qwen-Image is undistilled and wants around {steps}; "
                    "well below that it stays soft."
                ),
                "guidance_scale": (
                    "Ignored on Qwen-Image: its transformer has no guidance input, so this "
                    "number never reaches the model. Leave it at {guidance_scale:g}. The "
                    "negative prompt is driven by True CFG, the box beside this one."
                ),
                "true_cfg": (
                    "Real classifier-free guidance, which this checkpoint is meant to be "
                    "sampled with - about 4 - and which the pipeline runs only when a "
                    "negative prompt is given, so type one (a word will do). Above "
                    "{cfg_threshold:g} it costs a second pass per step (~2x slower); "
                    "blank is off, and unguided Qwen-Image is noticeably worse."
                ),
                "mu": (
                    "Flow-matching timestep shift. Qwen-Image derives its own from the "
                    "image size, and a value here overrides it outright; leave this blank."
                ),
            },
        },
    )
)

# Z-Image Turbo before the undistilled family entry, which carries the
# ZImagePipeline's own defaults. Both guide from zero, like Krea 2. The
# pipeline computes a resolution-dependent mu and hands it to a scheduler
# shipped with `use_dynamic_shifting: false, shift: 3.0`, which ignores it:
# the shift that lands is the constant 3, i.e. mu = ln 3. A requested mu goes
# through the backend's static-shift fallback and replaces it exactly.
_Z_IMAGE_MU_HINT = (
    "Flow-matching timestep shift. Z-Image ships a fixed shift of 3 (a mu of "
    "1.10) and its scheduler ignores the value the pipeline derives from the "
    "image size; a number here replaces that shift outright. Leave blank."
)
_PROFILES += [
    (
        "z-anime",
        {
            "steps": 40,
            "guidance_scale": 4.0,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.Z_ANIME_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. Z-Anime Base recommends 28-50; {steps} "
                    "is the author's Diffusers example."
                ),
                "guidance_scale": (
                    "Classifier-free guidance. The author recommends 3-5, with "
                    "{guidance_scale:g} in the Diffusers example. Above "
                    "{cfg_threshold:g} the negative prompt applies and adds "
                    "an uncond pass per step."
                ),
                "mu": (
                    "Leave blank for the supplied FlowMatch Euler scheduler's "
                    "fixed shift of 6 (mu 1.79). A value here replaces it. "
                    "The author's ComfyUI Euler Ancestral/beta recommendation "
                    "is a different sampling setup."
                ),
            },
        },
    ),
    (
        "z-image-turbo",
        {
            "steps": 8,
            "guidance_scale": 0.0,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.Z_IMAGE_TURBO_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. Z-Image Turbo is distilled to resolve in {steps}; "
                    "more adds time, not detail."
                ),
                "guidance_scale": (
                    "Classifier-free guidance. Turbo is trained for {guidance_scale:g}. "
                    "This pipeline guides from 0 up, so anything above {cfg_threshold:g} "
                    "brings the negative prompt in and adds an uncond pass per step "
                    "(~2x slower)."
                ),
                "mu": _Z_IMAGE_MU_HINT,
            },
        },
    ),
    (
        "z-image",
        {
            "steps": 50,
            "guidance_scale": 5.0,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.Z_IMAGE_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. The undistilled Z-Image wants around {steps}; well "
                    "below that it stays soft."
                ),
                "guidance_scale": (
                    "Classifier-free guidance, {guidance_scale:g} recommended. Z-Image "
                    "guides from 0 up, so any value above {cfg_threshold:g} uses the "
                    "negative prompt and costs an uncond pass per step (~2x slower)."
                ),
                "mu": _Z_IMAGE_MU_HINT,
            },
        },
    ),
]
# ERNIE-Image Turbo ahead of the undistilled family entry, which carries the
# ErnieImagePipeline's own defaults. The pipeline has the classic `> 1` gate
# with a negative prompt and no true CFG - klein's arrangement with the
# negative prompt back. Its scheduler is shipped static at shift 4 and the
# pipeline hands it a plain linspace, taking no `sigmas` or `mu` from outside,
# so a requested mu has no way in: the backend drops it and the row says so.
_ERNIE_MU_HINT = (
    "Flow-matching timestep shift. ERNIE-Image ships a fixed shift of 4 (a mu "
    "of 1.39) and its pipeline takes no schedule from outside, so a number "
    "here cannot be applied and is dropped from the record. Leave blank."
)
_PROFILES += [
    (
        "ernie-image-turbo",
        {
            "steps": 8,
            "guidance_scale": 1.0,
            "mu": None,
            "mu_supported": False,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.ERNIE_TURBO_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. ERNIE-Image Turbo is distilled to resolve in "
                    "{steps}; more adds time, not detail."
                ),
                "guidance_scale": (
                    "Classifier-free guidance. Turbo is distilled for {guidance_scale:g}, "
                    "where no uncond pass runs; above {cfg_threshold:g} the negative "
                    "prompt applies and costs a second pass per step (~2x slower)."
                ),
                "mu": _ERNIE_MU_HINT,
            },
        },
    ),
    (
        "ernie-image",
        {
            "steps": 50,
            "guidance_scale": 4.0,
            "mu": None,
            "mu_supported": False,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": guides.ERNIE_GUIDE,
            "hints": {
                "steps": (
                    "Denoising steps. The undistilled ERNIE-Image wants around {steps}; "
                    "well below that it stays soft."
                ),
                "guidance_scale": (
                    "Classifier-free guidance, {guidance_scale:g} recommended. Above "
                    "{cfg_threshold:g} the negative prompt applies, at roughly 2x the "
                    "time per step."
                ),
                "mu": _ERNIE_MU_HINT,
            },
        },
    ),
]
# FLUX.1 Krea [dev] is FLUX.1 [dev] to every control - the same numbers, the
# same hints - and its own checkpoint to the prompting guide: Krea's README
# recommends 1024-1280 px sides and photo language. Ahead of "flux", whose
# token its id carries too.
_FLUX_PROFILE = dict(_PROFILES[[token for token, _ in _PROFILES].index("flux")][1])
_PROFILES.insert(
    [token for token, _ in _PROFILES].index("flux"),
    ("krea-dev", {**_FLUX_PROFILE, "guide": guides.KREA_DEV_GUIDE}),
)

# Split-file models resolve a specific release rather than every checkpoint
# whose name contains a family token. In particular, "anima" is also part of
# "animagine", and Anima's Base and Aesthetic variants need different defaults.
_EXACT_PROFILES: dict[str, dict[str, Any]] = {
    "duongve/netayume-lumina-image-2.0": {
        "steps": 40,
        "guidance_scale": 5.0,
        "mu": None,
        "mu_supported": True,
        "cfg_threshold": 1.0,
        "negative_prompt_supported": True,
        "reference_supported": False,
        "true_cfg_supported": False,
        "guide": guides.NETAYUME_GUIDE,
        "hints": {
            "steps": (
                "Denoising steps. NetaYume's published v2 family guidance is "
                "40-50; {steps} is this app's starting point for the pinned v4 "
                "release, not a measured optimum."
            ),
            "guidance_scale": (
                "Classifier-free guidance. NetaYume's published v2 family "
                "guidance is 4-7; {guidance_scale:g} is this app's starting "
                "point. Above {cfg_threshold:g} the negative prompt applies "
                "and adds an uncond pass per step."
            ),
            "mu": (
                "Leave blank for the standard Lumina 2 FlowMatch Euler "
                "scheduler's fixed shift of 6 (mu 1.79). A value here "
                "replaces it. The author's ComfyUI Euler Ancestral/normal "
                "and res_multistep/linear_quadratic recommendations are "
                "different sampling setups."
            ),
        },
    },
    "circlestone-labs/anima": {
        "steps": 8,
        "guidance_scale": 1.0,
        "mu": None,
        "mu_supported": False,
        "cfg_threshold": 1.0,
        "negative_prompt_supported": True,
        "reference_supported": False,
        "true_cfg_supported": False,
        "guide": guides.ANIMA_TURBO_GUIDE,
        "hints": {
            "steps": (
                "Denoising steps. Anima Turbo is distilled for 8-12; "
                "{steps} is this app's starting point."
            ),
            "guidance_scale": (
                "Classifier-free guidance. The authors recommend "
                "{guidance_scale:g} for Turbo, where no uncond pass runs. "
                "Above {cfg_threshold:g} the negative prompt applies and "
                "adds a second pass per step."
            ),
            "mu": (
                "Unavailable: Anima Turbo's pipeline uses its fixed flow "
                "schedule and does not expose a Mu override."
            ),
        },
    },
}

_FALLBACK: dict[str, Any] = {
    "steps": 28,
    "guidance_scale": 3.5,
    "mu": None,
    "mu_supported": True,
    "cfg_threshold": 1.0,
    "negative_prompt_supported": True,
    "reference_supported": False,
    "true_cfg_supported": False,
    "guide": guides.FALLBACK_GUIDE,
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
    if key in _EXACT_PROFILES:
        return _EXACT_PROFILES[key]
    for token, profile in _PROFILES:
        if token in key:
            return profile
    return _FALLBACK


_PROSE = ("hints", "guide")


def defaults_for(model_id: str) -> dict[str, Any]:
    """The numbers only - callers put these straight into a request or a field."""
    return {k: v for k, v in _profile_for(model_id).items() if k not in _PROSE}


def guide_for(model_id: str) -> dict[str, Any]:
    """The prompting guide's data half: documented pixels/aspects, the word
    for the range, and the authors' example prompts. The prose half rides in
    `hints_for` under `prompt` and `size`, beside the other tooltips."""
    guide = _profile_for(model_id)["guide"]
    return {
        "pixels": guide["pixels"],
        "pixels_word": guide["pixels_word"],
        "aspects": list(guide["aspects"]) if guide.get("aspects") is not None else None,
        "resolutions": list(guide["resolutions"]) if guide.get("resolutions") is not None else None,
        "examples": list(guide["examples"]),
        "tag_helper": deepcopy(guide.get("tag_helper")),
    }


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


# The reference-image box. Not per profile: the sentence is the same wherever
# the capability is, and the two states are the whole of it.
_REFERENCE_HINTS = {
    False: (
        "Reference images condition the picture on one you have already made. This "
        "checkpoint's pipeline takes none - FLUX.2 [klein] is the one here that does."
    ),
    True: (
        "Up to four gallery images to condition on - same character, new pose; a "
        "sketch to finish. Each is resized to 1 MP, encoded and concatenated onto "
        "the latent sequence every step. Measured on this card, one is about free "
        "and two cost about a third more: under model offload the per-image weight "
        "transfer dominates the longer sequence."
    ),
}

# The LoRA box. Unlike every other capability here this one is the *environment*
# talking, not the checkpoint: peft has to be installed and the base weights
# have to be ones an adapter can be applied to.
_LORA_HINTS = {
    False: (
        "LoRA adapters need peft installed and a base this backend quantizes itself "
        "- a pre-quantized Nunchaku checkpoint's linears are not the layers an "
        "adapter attaches to."
    ),
    True: (
        "A LoRA adapter from the loras/ directory, at a scale. Swapped on the live "
        "pipeline per image, so a scale axis is an ordinary sweep: 0-1x0.25 shows "
        "where the style arrives and where it takes over."
    ),
}


def hints_for(
    model_id: str,
    cfg_threshold: float | None = None,
    negative_prompt_supported: bool | None = None,
    true_cfg_supported: bool | None = None,
    reference_supported: bool | None = None,
    lora_supported: bool | None = None,
) -> dict[str, str]:
    """Tooltip text for steps/CFG/mu/true CFG, the prompt and the size on this
    checkpoint.

    `cfg_threshold`, `negative_prompt_supported` and `true_cfg_supported`
    override the profile's guesses with what a loaded pipeline actually
    exposes, so the tooltip and the greyed-out fields cannot disagree about
    where guidance switches on, whether a negative prompt does anything at
    all, or which box drives it.
    """
    profile = _profile_for(model_id)
    values = {k: v for k, v in profile.items() if k not in _PROSE}
    if cfg_threshold is not None:
        values["cfg_threshold"] = cfg_threshold
    hints = dict(profile["hints"])
    hints["prompt"] = profile["guide"]["prompt"]
    hints["size"] = profile["guide"]["size"]
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
    references = (
        profile["reference_supported"] if reference_supported is None else reference_supported
    )
    hints["reference"] = _REFERENCE_HINTS[bool(references)]
    # No profile guess to fall back on: whether an adapter can be loaded is a
    # fact about this installation, which only the backend knows.
    if lora_supported is not None:
        hints["lora"] = _LORA_HINTS[bool(lora_supported)]
    return {key: text.format(**values) for key, text in hints.items()}
