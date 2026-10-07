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

`refine_supported` says whether the checkpoint can run a refine pass - a
gallery image scaled up and re-denoised part of the way. It can where
diffusers ships an image-to-image class beside the checkpoint's pipeline
(`refine.sibling`): FLUX.1, SDXL, Z-Image and Qwen-Image. A guess from the id
until a loaded pipeline's class answers; `tests/test_refine.py` holds each
guess to the installed diffusers. `refine_strength` is where the Strength box
starts there, None where there is no pass: a profile number like the steps,
because a strength is an index into the checkpoint's own schedule - 0.4 on
SDXL, where WAI's author documents 0.35-0.5; 0.1 on the shifted FLUX.1 dev
schedule, but 0.5 on schnell's unshifted four-step schedule.
`refine_measured` is the largest pass `docs/verified.md` measured fitting the
card, where it measured one - a `Measured` row: the checkpoint and
quantization it ran on and its size. The Refine dialog warns past it and does
not open past it. `defaults_for` gives the page its pixels as
`refine_tight_pixels`, None for every profile with no row - the panel's
generic 1600x1600 line - and `hints_for` the words naming the run
(`refine_tight`), since a profile can be shared: FLUX.1 dev and the lite
repos quote Krea dev's fp8 run, and say so (Qwen-Image's is NVFP4, the
only way it fits the card). A figure, never a VRAM model:
each one is a row of that file, and `tests/test_profiles.py` holds it there.

`mu_supported` says whether the checkpoint uses a flow-matching scheduler whose
sigmas can be shifted. Modern SDXL exposes a custom `sigmas` argument too, but
its Euler scheduler is not flow matching; treating the argument alone as proof
would replace Animagine's noise schedule with an unrelated flow schedule.

`mu_static` is the shift a blank Mu leaves, as a mu, where the scheduler's
shift is a fixed number: ln 6 (1.79) on Z-Anime, NetaYume and NewBie, ln 3
(1.10) on Z-Image and Z-Image Turbo - the shipped `shift` of a scheduler with
`use_dynamic_shifting: false`, which the loaders pin. None everywhere else:
FLUX and Krea 2 Raw derive theirs from the size. The page shows the number
in place of "auto" and puts a sheet's Mu none at it, rather than first; the
stored mu stays null, and the grid's `none` stays its word. The Mu tips
quote it.

`cfg_ignored` says the CFG box never reaches the model: the distilled klein
(guidance baked in; the pipeline warns above 1), schnell and Qwen-Image (no
guidance input; True CFG is their knob). Their CFG tips open "Ignored", and
a test holds the two together. The grid's CFG axis then offers no example:
a sweep of it is the same image at every value.

`hints` is the same knowledge as prose, for the UI's tooltips: what steps, CFG
and mu mean *on this checkpoint*, which is not the same sentence twice. They are
format strings over the profile's own numbers, so the text cannot drift from the
values the controls are seeded with, and `hints_for` takes the live threshold the
way `/api/status` does. One of them is a name rather than a tip: `cfg_label`,
FLUX.1's "CFG (distilled)", which the page writes on the panel's and the grid's
CFG labels where a profile gives one (plain "CFG" elsewhere).
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, NamedTuple

from . import guides
from .refine import DEFAULT_STRENGTH, leaves_a_step

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


class Measured(NamedTuple):
    """A refine pass `docs/verified.md`'s "Past 4 MP" section ran without
    spilling: the checkpoint as the row names it, the quantization it ran
    at (fp8, or NVFP4 where only Nunchaku's re-upload fits the card), and
    the pass's size."""

    checkpoint: str
    quantization: str
    width: int
    height: int


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
            "refine_supported": False,
            "refine_strength": None,
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
            "refine_supported": False,
            "refine_strength": None,
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
            "refine_supported": False,
            "refine_strength": None,
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
            "cfg_ignored": True,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": False,
            "reference_supported": True,
            "refine_supported": False,
            "refine_strength": None,
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
            "cfg_ignored": True,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "refine_supported": True,
            "refine_strength": 0.5,
            # Measured: 1536x1536 within a gigabyte of the card's top, as Krea
            # dev's same transformer is; nothing larger was run.
            "refine_measured": Measured("FLUX.1 schnell", "fp8", 1536, 1536),
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
            "refine_supported": True,
            # Measured on Krea dev: at 1.5x of a 1 MP image the shift is
            # already mu 2.0, so 0.1 starts the pass at sigma 0.47 and keeps
            # the face; 0.15 (0.62) and up repaint it.
            "refine_strength": 0.1,
            # Measured on Krea dev: 1536x1536 within a gigabyte of the card's
            # top, 2048x2048 spilling. FLUX.1 dev and the lite repos share this
            # profile, and so quote Krea dev's run, naming it.
            "refine_measured": Measured("FLUX.1 Krea dev", "fp8", 1536, 1536),
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
                # The CFG box's own name, the panel's and the grid's: the
                # authors call it CFG, so it keeps the word, qualified - two
                # boxes both called CFG read as one knob twice. A name, not a
                # tip; every other profile's box is plain CFG.
                "cfg_label": "CFG (distilled)",
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
                # Under the Refine dialog's Strength, for a pass past
                # `refine_measured`: the measured runs in verified.md,
                # "Past 4 MP".
                "refine_large": (
                    "FLUX.1's shift grows with the size, so one strength starts a larger "
                    "pass at more noise: measured on Krea dev at 2x of 1 MP, 0.1 started "
                    "at sigma 0.75 and repainted the subject; Mu 1.15 at 0.2 started at "
                    "0.46 and kept it."
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
            "refine_supported": True,
            "refine_strength": 0.4,
            # Measured: 2896x2896, the pass's own guard, at 10.3 GB of the card.
            "refine_measured": Measured("Animagine XL 4.0", "fp8", 2896, 2896),
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
            "refine_supported": True,
            "refine_strength": 0.4,
            # Measured: 2896x2896, the pass's own guard, at 10.3 GB of the card.
            "refine_measured": Measured("Kohaku XL Zeta", "fp8", 2896, 2896),
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
            "refine_supported": True,
            "refine_strength": 0.4,
            # Measured: 2896x2896, the pass's own guard, at 10.4 GB of the card.
            "refine_measured": Measured("Illustrious XL v2.0", "fp8", 2896, 2896),
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
            "cfg_ignored": True,
            "mu": None,
            "mu_supported": True,
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "refine_supported": True,
            "refine_strength": 0.3,
            # Measured under Nunchaku, the only way it fits this card:
            # 1280x1280 with true CFG at 15.0 GB of the card; nothing larger.
            "refine_measured": Measured("Qwen-Image", "NVFP4", 1280, 1280),
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
    "{mu_static:.2f}) and its scheduler ignores the value the pipeline derives from the "
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
            "mu_static": math.log(6),
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "refine_supported": True,
            # Shift 6: 0.15 starts the pass at sigma 0.51.
            "refine_strength": 0.15,
            # Measured: 1536x1536 at 13.7 GB of the card, 2048x2048 filling it.
            "refine_measured": Measured("Z-Anime", "fp8", 1536, 1536),
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
                    "fixed shift of 6 (mu {mu_static:.2f}). A value here replaces it. "
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
            "mu_static": math.log(3),
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "refine_supported": True,
            "refine_strength": 0.25,
            # Measured: 2048x2048 at 12.5 GB of the card; 8.4 MP not run, its
            # growth putting it past the card.
            "refine_measured": Measured("Z-Image Turbo", "fp8", 2048, 2048),
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
            "mu_static": math.log(3),
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "refine_supported": True,
            "refine_strength": 0.25,
            # Measured: 1536x1536 at 13.1 GB of the card, as Z-Anime; its
            # 2048x2048 filled the card, so nothing larger was run.
            "refine_measured": Measured("Z-Image", "fp8", 1536, 1536),
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
            "refine_supported": False,
            "refine_strength": None,
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
            "refine_supported": False,
            "refine_strength": None,
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
# recommends 1024-1280 px sides. Ahead of "flux", whose token its id carries
# too.
_FLUX_PROFILE = dict(_PROFILES[[token for token, _ in _PROFILES].index("flux")][1])
_PROFILES.insert(
    [token for token, _ in _PROFILES].index("flux"),
    ("krea-dev", {**_FLUX_PROFILE, "guide": guides.KREA_DEV_GUIDE}),
)

# These profiles resolve a specific release rather than every checkpoint
# whose name contains a family token. In particular, "anima" is also part of
# "animagine", and Anima's Base and Aesthetic variants need different defaults.
_EXACT_PROFILES: dict[str, dict[str, Any]] = {
    "minthy/rouwei-0.8": {
        "steps": 28,
        "guidance_scale": 6.0,
        "mu": None,
        "mu_supported": False,
        "cfg_threshold": 1.0,
        "negative_prompt_supported": True,
        "reference_supported": False,
        "refine_supported": True,
        "refine_strength": 0.4,
        "true_cfg_supported": False,
        "guide": guides.ROUWEI_GUIDE,
        "hints": {
            "steps": (
                "Denoising steps. RouWei 0.8 recommends 20-28 with Euler Ancestral; "
                "{steps} is this app's starting point."
            ),
            "guidance_scale": (
                "Real classifier-free guidance. RouWei 0.8 epsilon recommends CFG "
                "4-8; this app starts at {guidance_scale:g}. Above {cfg_threshold:g} "
                "the negative prompt applies and adds an uncond pass per step."
            ),
            "mu": (
                "Unavailable: RouWei 0.8 uses SDXL's Euler Ancestral noise schedule, "
                "not flow matching."
            ),
        },
    },
    "lyliaengine/waiillustrioussdxl-v170": {
        "steps": 28,
        "guidance_scale": 6.0,
        "mu": None,
        "mu_supported": False,
        "cfg_threshold": 1.0,
        "negative_prompt_supported": True,
        "reference_supported": False,
        "refine_supported": True,
        "refine_strength": 0.4,
        # Measured: 2560x3360 at 11.4 GB of the card - past the pass's own
        # 8.4 MP guard, so nothing the dialog offers runs tight.
        "refine_measured": Measured("WAI v17", "fp8", 2560, 3360),
        "true_cfg_supported": False,
        "guide": guides.WAI_GUIDE,
        "hints": {
            "steps": (
                "Denoising steps. WAI v17 recommends 15-30 with Euler Ancestral; "
                "{steps} is this app's starting point."
            ),
            "guidance_scale": (
                "Real classifier-free guidance. WAI v17 recommends CFG 5-7; "
                "this app starts at {guidance_scale:g}. Above {cfg_threshold:g} "
                "the negative prompt applies and adds an uncond pass per step."
            ),
            "mu": (
                "Unavailable: WAI v17 uses SDXL's Euler Ancestral noise schedule, "
                "not flow matching."
            ),
        },
    },
    "newbie-ai/newbie-image-exp0.1": {
        "steps": 28,
        "guidance_scale": 4.0,
        "mu": None,
        "mu_supported": True,
        "mu_static": math.log(6),
        "cfg_threshold": 1.0,
        "negative_prompt_supported": True,
        "reference_supported": False,
        "refine_supported": False,
        "refine_strength": None,
        "true_cfg_supported": False,
        "guide": guides.NEWBIE_GUIDE,
        "hints": {
            "steps": (
                "Denoising steps. The author's NewBie quickstart uses {steps}; "
                "this is the app's starting point, not a measured optimum."
            ),
            "guidance_scale": (
                "Classifier-free guidance. The author's sampling script uses "
                "{guidance_scale:g}; this app uses it as a starting point "
                "with the card's 28 steps. Above {cfg_threshold:g} the negative prompt "
                "applies and adds an uncond pass per step."
            ),
            "mu": (
                "Flow-matching timestep shift. Leave blank for NewBie's "
                "standard FlowMatch Euler shift 6 (mu {mu_static:.2f}); a value here replaces "
                "its shift."
            ),
        },
    },
    "duongve/netayume-lumina-image-2.0": {
        "steps": 40,
        "guidance_scale": 5.0,
        "mu": None,
        "mu_supported": True,
        "mu_static": math.log(6),
        "cfg_threshold": 1.0,
        "negative_prompt_supported": True,
        "reference_supported": False,
        "refine_supported": False,
        "refine_strength": None,
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
                "scheduler's fixed shift of 6 (mu {mu_static:.2f}). A value here "
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
        "refine_supported": False,
        "refine_strength": None,
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
    "circlestone-labs/anima@aesthetic-v1.1": {
        "steps": 30,
        "guidance_scale": 4.0,
        "mu": None,
        "mu_supported": False,
        "cfg_threshold": 1.0,
        "negative_prompt_supported": True,
        "reference_supported": False,
        "refine_supported": False,
        "refine_strength": None,
        "true_cfg_supported": False,
        "guide": guides.ANIMA_AESTHETIC_GUIDE,
        "hints": {
            "steps": (
                "Denoising steps. The Anima card recommends 30-50; "
                "{steps} is this app's starting point for Aesthetic."
            ),
            "guidance_scale": (
                "Classifier-free guidance. The Anima card recommends 4-5; "
                "{guidance_scale:g} is this app's starting point for Aesthetic. "
                "Above {cfg_threshold:g} the negative prompt applies and "
                "adds a second pass per step."
            ),
            "mu": (
                "Unavailable: Anima Aesthetic's pipeline uses its fixed flow "
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
    "refine_supported": False,
    "refine_strength": None,
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


# The prose, and the measured refine pass, whose pixels reach the defaults as
# `refine_tight_pixels` and whose words the hints as `refine_tight`.
_NOT_DEFAULTS = ("hints", "guide", "refine_measured")

# The checkpoints whose pipeline is this repo's own class - both Anima variants'
# modular blocks and NewBie's - beside which diffusers exports no
# image-to-image class, so `refine.sibling` answers None loaded or not
# (`tests/test_refine.py`). Their `refine_supported` is the answer a load
# would give, not a guess: `refine_settled` says so, and the Refine dialog
# stops telling the user that a load will decide.
_OWN_PIPELINES = frozenset({
    "circlestone-labs/anima", "circlestone-labs/anima@aesthetic-v1.1",
    "newbie-ai/newbie-image-exp0.1",
})


def defaults_for(model_id: str) -> dict[str, Any]:
    """The numbers only - callers put these straight into a request or a field.

    `refine_tight_pixels` is the measured pass's pixels where a pass was
    measured; every other profile answers None for it, which the page reads
    as "not measured". `refine_settled` is whether `refine_supported` is
    final before a load (`_OWN_PIPELINES`); the status sets it once a loaded
    pipeline has answered too (`main._merged_defaults`).
    """
    profile = _profile_for(model_id)
    measured = profile.get("refine_measured")
    return {
        **{k: v for k, v in profile.items() if k not in _NOT_DEFAULTS},
        # Every status says both: None where the shift is not a fixed number,
        # False where the CFG box reaches the model.
        "mu_static": profile.get("mu_static"),
        "cfg_ignored": bool(profile.get("cfg_ignored", False)),
        "refine_tight_pixels": measured.width * measured.height if measured else None,
        "refine_settled": model_id.lower().replace("_", "-") in _OWN_PIPELINES,
    }


def guide_for(model_id: str) -> dict[str, Any]:
    """The prompting guide's data half: documented pixels/aspects, the word
    for the range, and the authors' example prompts. The prose half rides in
    `hints_for` under `prompt` and `size`, beside the other tooltips."""
    guide = _profile_for(model_id)["guide"]
    return {
        "pixels": guide["pixels"],
        "pixels_word": guide["pixels_word"],
        "pixels_source": guide.get("pixels_source"),
        "aspects": list(guide["aspects"]) if guide.get("aspects") is not None else None,
        "resolutions": list(guide["resolutions"]) if guide.get("resolutions") is not None else None,
        "examples": list(guide["examples"]),
        # The authors' template slots the placeholder rule finds, as data:
        # the Examples menu says an example holds one and selects it on a
        # pick. Patterns, not the rule; the rest of `analysis` stays out.
        "slots": [
            entry["pattern"] for entry in guide.get("analysis", {}).get("placeholders", [])
        ],
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

# The Refine dialog's Strength badge. One sentence wherever the capability is,
# like the reference box, with the profile's own starting strength in it, and
# its own sweep for the example: what a strength does is the schedule's, which
# is why `refine_strength` is a profile key and not one constant (see "The
# refine pass" in the design notes for each value's basis, and for why one
# strength redraws more of a larger pass on FLUX.1 - which the dialog says
# under Strength there, `refine_large`, once the size gets that far).
_REFINE_HINTS = {
    False: (
        "A refine pass scales a gallery image up and re-denoises it part of the way. "
        "This app's pipeline has no compatible image-to-image counterpart - "
        "the FLUX.1, SDXL, Z-Image and Qwen-Image checkpoints have one."
    ),
    True: (
        "Strength: how much of the source is redrawn - low keeps the picture, 1 keeps "
        "nothing. It starts at {refine_strength:g} on this checkpoint. Several values, "
        "such as {refine_sweep}, run as one sweep and open as a sheet."
    ),
}


def strength_sweep(start: float, steps: int) -> list[float]:
    """Three strengths around `start` - half of it, it, and half as much again -
    on the 0.05 grid, less any that leaves no step of `steps` or passes 1.

    The Refine dialog's Compare 3 writes the same values into Strength
    (`compareStrengths` in web/refine.js, the same arithmetic in the same
    order, so the floats round alike), and the Strength tip quotes them: the
    checkpoint's own neighbourhood, where a fixed example would be SDXL's
    0.3-0.5 everywhere - each of which repaints the subject on FLUX.1.

    Half rounds up, on the quotient snapped to six places first: Z-Anime's
    0.15 makes ties of both neighbours, and in floats 0.5 * 0.15 / 0.05 is
    1.4999999999999998 and 1.5 * 0.15 / 0.05 is 4.499999999999999, which
    floored to 0.05,0.15,0.2 - a sweep of -0.10/+0.05 where 0.1,0.15,0.25 are
    the three verified.md measured. Rounding the hundredths instead does not
    fix it, and `round()` would round half to even where the page's
    `Math.round` does not.
    """
    out: list[float] = []
    for factor in (0.5, 1, 1.5):
        value = round(math.floor(round(factor * start / 0.05, 6) + 0.5) * 0.05, 2)
        if 0 < value <= 1 and leaves_a_step(steps, value) and value not in out:
            out.append(value)
    return out

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
    refine_supported: bool | None = None,
) -> dict[str, str]:
    """Tooltip text for steps/CFG/mu/true CFG, the prompt and the size on this
    checkpoint - and, on FLUX.1, the CFG box's own name (`cfg_label`).

    `cfg_threshold`, `negative_prompt_supported` and `true_cfg_supported`
    override the profile's guesses with what a loaded pipeline actually
    exposes, so the tooltip and the greyed-out fields cannot disagree about
    where guidance switches on, whether a negative prompt does anything at
    all, or which box drives it. `reference_supported` and `refine_supported`
    do the same for their boxes.
    """
    profile = _profile_for(model_id)
    values = {k: v for k, v in profile.items() if k not in _NOT_DEFAULTS}
    if cfg_threshold is not None:
        values["cfg_threshold"] = cfg_threshold
    # A checkpoint whose profile names no starting strength - one the table
    # guessed has no refine pass, or does not know - runs at the general one
    # if its pipeline turns out to have a pass, and the tooltip quotes that.
    values["refine_strength"] = values.get("refine_strength") or DEFAULT_STRENGTH
    values["refine_sweep"] = ",".join(
        f"{v:g}" for v in strength_sweep(values["refine_strength"], values["steps"])
    )
    hints = dict(profile["hints"])
    hints["prompt"] = profile["guide"]["prompt"]
    hints["size"] = profile["guide"]["size"]
    if (
        negative_prompt_supported is not None
        and negative_prompt_supported != profile["negative_prompt_supported"]
    ):
        hints["guidance_scale"] = _NEGATIVE_HINTS[negative_prompt_supported]
        # The tip now calls the box classifier-free guidance: its name must
        # not say otherwise.
        hints.pop("cfg_label", None)
    separate = profile["true_cfg_supported"] if true_cfg_supported is None else true_cfg_supported
    if not separate:
        hints["true_cfg"] = _NO_TRUE_CFG_HINT
    elif "true_cfg" not in hints:
        hints["true_cfg"] = _TRUE_CFG_HINT
    references = (
        profile["reference_supported"] if reference_supported is None else reference_supported
    )
    hints["reference"] = _REFERENCE_HINTS[bool(references)]
    refines = profile["refine_supported"] if refine_supported is None else refine_supported
    hints["refine"] = _REFINE_HINTS[bool(refines)]
    # The Refine dialog's size line past the measured pass, naming the run it
    # quotes - on FLUX.1 dev that is Krea dev's, and fp8 whatever this server
    # runs - with its size as verified.md writes it: one decimal read 2.4 MP
    # for both a 2.43 MP request and the 2.36 MP line it was past.
    measured = profile.get("refine_measured")
    if measured and refines:
        hints["refine_tight"] = (
            f"past the {measured.width}×{measured.height} "
            f"({measured.width * measured.height / 1e6:.2f} MP) measured on "
            f"{measured.checkpoint} at {measured.quantization}"
        )
    if not refines:
        hints.pop("refine_large", None)
    # No profile guess to fall back on: whether an adapter can be loaded is a
    # fact about this installation, which only the backend knows.
    if lora_supported is not None:
        hints["lora"] = _LORA_HINTS[bool(lora_supported)]
    return {key: text.format(**values) for key, text in hints.items()}
