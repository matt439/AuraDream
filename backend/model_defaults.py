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

# The prompting guides. Each profile carries one under "guide": `prompt` and
# `size` are prose for the ? badges on the Prompt box and the Size presets
# (paragraphs, so they hold no format fields - they go through `.format` with
# the other hints, and a brace would fail
# `test_every_profile_explains_its_controls`), `pixels` is the megapixel
# range the checkpoint's authors put in writing, `pixels_word` is what that
# writing amounts to - a model card's "supports 0.1-2 MP", a README's
# "recommended 1024-1280" and a report's "trained to 1328" are not the same
# promise, and the size line quotes the word - and `examples` are the
# authors' own prompts, verbatim, for the Examples menu. Every sentence is
# sourced; docs/design-notes.md ("The prompting guide") keeps the URLs.
_FLUX1_PROMPT = (
    "Plain descriptive sentences, not a tag list: subject first, then setting, "
    "style, camera and lighting - what comes first carries the most weight. "
    "30-80 words suits most images; only the first 77 tokens reach the CLIP "
    "encoder, and T5 stops reading at 512.\n"
    "Put any text to render in quotation marks. There is no negative prompt "
    "unless True CFG is on, so describe what you want rather than what you do "
    "not. English is the most precise language."
)
_FLUX1_SIZE = (
    "Black Forest Labs documents FLUX.1 for 0.1 to 2 MP at any aspect ratio, "
    "sides in multiples of 16; 1024x1024 and 1360x768 are its reference sizes. "
    "Above 2 MP is outside that range and is where users report doubled "
    "subjects and smeared detail: render at 1-2 MP and upscale."
)
_FLUX1_EXAMPLES = [
    "A cat holding a sign that says hello world",
    "a photo of a forest with mist swirling around the tree trunks. The word "
    '"FLUX" is painted over it in big, red brush strokes with visible texture',
    "A tall sharp-featured man in an oversized charcoal wool coat, standing on a "
    "wet cobblestone street at night, fashion editorial, moody street lighting",
    "At high noon on a blustery day, capture the surreal presence of a sentient "
    "tree, seemingly rooted underwater just off a tumultuous ocean shore",
]
_FLUX1_GUIDE = {
    "prompt": _FLUX1_PROMPT,
    "size": _FLUX1_SIZE,
    "pixels": (0.1, 2.0),
    "pixels_word": "documented",
    "examples": _FLUX1_EXAMPLES,
}
_KREA_DEV_GUIDE = {
    "prompt": (
        "FLUX.1 prompting - descriptive sentences, subject first, text in "
        "quotation marks, 30-80 words, no negative prompt without True CFG - on "
        "a checkpoint tuned for photographs without the oversaturated 'AI look'. "
        "It answers best to photo language: a mood, a lens, a light, a film "
        "stock, a colour grade. Krea calls it opinionated - give it room rather "
        "than over-specifying.\n"
        "Krea recommends 28-32 steps and CFG 3.5-5."
    ),
    "size": (
        "Krea's README recommends sides between 1024 and 1280 pixels (about 1 to "
        "1.6 MP), in multiples of 16; the FLUX.1 family is documented for 0.1 to "
        "2 MP. The 1 and 1.5 MP presets are inside the recommendation; 2 MP is "
        "just past it and usually holds; 4 MP is well outside - render smaller "
        "and upscale."
    ),
    "pixels": (1.0, 1.64),
    "pixels_word": "recommended",
    "examples": [
        "Moody night photography, a small store with an orange glowing sign that "
        "says 'Black Forest Labs'",
        "White owl, close up portrait, mysterious ghost like owl, low contrast, "
        "cinematic dark moody photo, slight blue color grading",
        "Photo of two people having a tea party in a lush garden",
        "A frog holding a sign that says hello world",
    ],
}
_SCHNELL_GUIDE = {
    "prompt": (
        "FLUX.1 prompting - descriptive sentences, subject first, text in "
        "quotation marks - with a shorter memory: schnell's T5 encoder stops at "
        "256 tokens, about 180 words, so keep it under that. Guidance is off on "
        "schnell, so a negative prompt does nothing unless True CFG is on; "
        "describe what you want instead."
    ),
    "size": _FLUX1_SIZE,
    "pixels": (0.1, 2.0),
    "pixels_word": "documented",
    "examples": _FLUX1_EXAMPLES,
}
_KLEIN_GUIDE = {
    "prompt": (
        "Subject, then action, then style, then context, as sentences: FLUX.2 "
        "weights what comes first. 30-80 words is BFL's sweet spot; the Qwen3 "
        "encoder reads up to 512 tokens. Put every word that must appear in the "
        "picture in quotation marks - a sign or label without one comes out as "
        "gibberish. Name a camera, lens or film stock for photorealism. "
        "Structured JSON prompts (scene, subjects, lighting, camera...) are "
        "understood too.\n"
        "There is no negative prompt on FLUX.2 at all: say what you want. With "
        "reference images, say how each is used - 'the jacket from image 1'."
    ),
    "size": (
        "Every official klein example is 1024x1024. BFL's FLUX.2 guidance is a "
        "minimum of 64x64, a maximum of 4 MP and up to 2 MP recommended, sides "
        "in multiples of 16. Reference images are resized to about 1 MP on the "
        "way in, so the output need not match them."
    ),
    "pixels": (0.1, 2.0),
    "pixels_word": "recommended",
    "examples": [
        "A cat holding a sign that says hello world",
        "An Entry of a Sushi Restaurant, The text 'OPEN' appears in red neon "
        "letters above the door",
        "Soaking wet tiger cub taking shelter under a banana leaf in the rainy "
        "jungle, close up photo",
        "Turn this cat into a dog",
    ],
}
_KREA2_PROMPT = (
    "Natural language, one cohesive paragraph. Krea's guide: long, detailed "
    "prompts give the best results - the training captions were dense - but a "
    "short one works too. Put words to render in quotation marks; text "
    "rendering was a reward in post-training. The Qwen3-VL encoder reads 512 "
    "tokens and was chosen for multilingual reach, though Krea publishes no "
    "language list.\n"
)
_KREA2_EXAMPLES = [
    "a fox walking in the snow",
    "a red fox sitting in fresh snow, golden hour, photorealistic",
    "immense rocket launch exhaust as seen from extremely close up",
    "A tiny, russet-brown harvest mouse clings to a slender diagonal branch amid "
    "vibrant green lobed leaves and small round buds. The mouse has soft "
    "textured fur, glossy black eyes, a pink nose, fine whiskers, and delicate "
    "pink paws firmly gripping the wood. In this macro photograph, an extremely "
    "shallow depth of field sharply focuses on the animal's face. The deep green "
    "background dissolves into a smooth, creamy bokeh, illuminated by soft, "
    "diffused natural lighting that highlights the intricate details of the fur "
    "and foliage.",
]
_KREA2_TURBO_GUIDE = {
    "prompt": _KREA2_PROMPT + (
        "Turbo runs guidance-free at CFG 0, where the negative prompt is unused; "
        "any CFG above 0 brings it in."
    ),
    "size": (
        "Krea documents Turbo for 1k to 2k: sides from 1024 to 2048, padded to a "
        "multiple of 16 - every preset tier, and 2048x2048 is the model "
        "card's own example. Pretraining topped out at 1024 px, so 1 MP is the "
        "size it has seen most; 4 MP is the documented ceiling."
    ),
    "pixels": (1.0, 4.2),
    "pixels_word": "documented",
    "examples": _KREA2_EXAMPLES,
}
_KREA2_RAW_GUIDE = {
    "prompt": _KREA2_PROMPT + (
        "Raw is the undistilled checkpoint Krea does not recommend for "
        "inference - train on Raw, run on Turbo - so expect variety over polish. "
        "It wants about 52 steps at CFG 3.5, and the negative prompt applies at "
        "any CFG above 0."
    ),
    "size": (
        "Krea's README says Raw was trained to generate up to 1k: 1 MP is the "
        "trained size, sides in multiples of 16. Everything above it is "
        "untrained territory - Turbo is the checkpoint with the 2k range."
    ),
    "pixels": (0.25, 1.0),
    "pixels_word": "trained",
    "examples": _KREA2_EXAMPLES,
}
_QWEN_IMAGE_GUIDE = {
    "prompt": (
        "A dense, caption-like description. The encoder is asked to 'describe "
        "the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background', so write in "
        "those terms. Qwen appends ', Ultra HD, 4K, cinematic composition.' to "
        "English prompts (Chinese: '超清，4K，电影级构图') and recommends it. "
        "Text is its strength, English and Chinese alike: put every string in "
        "quotation marks and say where it goes, and it lays out signs, labels "
        "and paragraphs. Qwen2.5-VL encoder, 512 tokens.\n"
        "Sample with True CFG 4 and a negative prompt - a single space if there "
        "is nothing to remove."
    ),
    "size": (
        "Qwen-Image was trained to 1328x1328 (1.8 MP) and its README lists "
        "1328x1328, 1664x928, 928x1664, 1472x1104, 1104x1472, 1584x1056 and "
        "1056x1584 - about 1.5-1.8 MP at every ratio, sides in multiples of 16. "
        "The 1.5 MP presets at 3:4, 4:3, 16:9 and 9:16 are those shapes exactly; "
        "1 MP sits inside its 640p-1328p training; 2 MP is just past the list "
        "and 4 MP well past it."
    ),
    "pixels": (0.4, 1.77),
    "pixels_word": "trained",
    "examples": [
        "A coffee shop entrance features a chalkboard sign reading \"Qwen Coffee "
        "😊 $2 per cup,\" with a neon light beside it displaying \"通义千问\". "
        "Next to it hangs a poster showing a beautiful Chinese woman, and beneath "
        "the poster is written \"π≈3.1415926-53589793-23846264-33832795-"
        "02384197\". Ultra HD, 4K, cinematic composition.",
        "Bookstore window display. A sign displays \"New Arrivals This Week\". "
        "Below, a shelf tag with the text \"Best-Selling Novels Here\". To the "
        "side, a colorful poster advertises \"Author Meet And Greet on Saturday\" "
        "with a central portrait of the author. There are four books on the "
        "bookshelf, namely \"The light between worlds\" \"When stars are "
        "scattered\" \"The slient patient\" \"The night circus\"",
        "a red fox sitting in fresh snow, golden hour, photorealistic, Ultra HD, "
        "4K, cinematic composition.",
    ],
}
_Z_IMAGE_PROMPT = (
    "Long, detailed natural language: the maintainers say Turbo works best with "
    "it, and expand short prompts through an LLM before generating. The "
    "training captions were plain, objective, bilingual descriptions that "
    "included any visible text, so write like one. English and Chinese both "
    "work, and text rendering in either is a strength: quote the words. Qwen3-4B "
    "encoder, 512 tokens.\n"
)
_Z_IMAGE_EXAMPLES = [
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
    "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
    "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
    "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
    "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
    "blurred colorful distant lights.",
    "一幅为名为\"造相「Z-IMAGE-TURBO」\"的项目设计的创意海报。画面巧妙地将文字概念视觉化："
    "一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。",
    "a red fox sitting in fresh snow, golden hour, photorealistic",
]
_Z_IMAGE_SIZE = (
    "Tongyi documents 512x512 to 2048x2048 of total pixel area at any aspect "
    "ratio, sides in multiples of 16, with 1024x1024 as the card's example; the "
    "technical report puts the training range at up to 1k-1.5k. So 1-2 MP is "
    "inside what it was trained on, and 4 MP is the documented ceiling but "
    "beyond it."
)
_Z_IMAGE_TURBO_GUIDE = {
    "prompt": _Z_IMAGE_PROMPT + (
        "Turbo runs without classifier-free guidance, so at CFG 0 the negative "
        "prompt is unused."
    ),
    "size": _Z_IMAGE_SIZE,
    "pixels": (0.26, 2.2),
    "pixels_word": "trained",
    "examples": _Z_IMAGE_EXAMPLES,
}
_Z_IMAGE_GUIDE = {
    "prompt": _Z_IMAGE_PROMPT + (
        "The undistilled Z-Image wants CFG 3-5 over 28-50 steps, and Tongyi "
        "strongly recommends a negative prompt with it."
    ),
    "size": _Z_IMAGE_SIZE,
    "pixels": (0.26, 2.2),
    "pixels_word": "trained",
    "examples": _Z_IMAGE_EXAMPLES,
}
_ERNIE_PROMPT = (
    "A content-rich, detailed visual description in one paragraph - subject, "
    "setting, colours, light, layout - the way the model card's own example "
    "reads. The repo's prompt enhancer, which this app leaves out, exists to "
    "expand a short prompt into exactly that, so write the expanded form "
    "yourself. Dense and layout-sensitive text is its strength, English and "
    "Chinese - posters, infographics, UI - so quote the words and say where "
    "they go. Ministral-3 encoder, up to 2048 tokens.\n"
)
_ERNIE_SIZE = (
    "Baidu lists 1024x1024, 848x1264, 1264x848, 768x1376, 1376x768, 896x1200 "
    "and 1200x896 - every one about 1 MP - and trained through 256, 512 and "
    "1024 px stages; sides in multiples of 16. 2 and 4 MP render on this card "
    "without failing, but nothing above 1 MP is documented."
)
_ERNIE_EXAMPLES = [
    "This is a photograph depicting an urban street scene. Shot at eye level, it "
    "shows a covered pedestrian or commercial street. Slightly below the center "
    "of the frame, a cyclist rides away from the camera toward the background. "
    "The overall color palette is warm, dominated by golden yellows and dark "
    "browns, evoking the atmosphere of dusk or early morning.",
    "一只黑白相间的中华田园犬",
    "Generate a webpage screenshot that briefly introduces LLMs",
]
_ERNIE_TURBO_GUIDE = {
    "prompt": _ERNIE_PROMPT + (
        "Turbo runs at guidance 1, where the negative prompt does nothing; raise "
        "CFG above 1 to bring it in."
    ),
    "size": _ERNIE_SIZE,
    "pixels": (1.0, 1.0),
    "pixels_word": "documented",
    "examples": _ERNIE_EXAMPLES,
}
_ERNIE_GUIDE = {
    "prompt": _ERNIE_PROMPT + (
        "The undistilled ERNIE-Image samples at CFG 4 over 50 steps, where the "
        "negative prompt applies."
    ),
    "size": _ERNIE_SIZE,
    "pixels": (1.0, 1.0),
    "pixels_word": "documented",
    "examples": _ERNIE_EXAMPLES,
}
_FALLBACK_GUIDE = {
    "prompt": (
        "Unrecognised checkpoint. Most current models read natural-language "
        "descriptions - subject, setting, style, light - better than tag lists, "
        "and want text to render in quotation marks. Its model card says the "
        "encoder's token limit and whether a negative prompt does anything."
    ),
    "size": (
        "Unrecognised checkpoint: 1 MP with sides in multiples of 16 is the size "
        "nearly every current model is trained around. Its model card says what "
        "else it supports."
    ),
    "pixels": None,
    "pixels_word": "documented",
    "examples": [],
}

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
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": _KREA2_TURBO_GUIDE,
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
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": _KREA2_RAW_GUIDE,
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
            "reference_supported": True,
            "true_cfg_supported": False,
            "guide": _KLEIN_GUIDE,
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
            "reference_supported": True,
            "true_cfg_supported": False,
            "guide": _KLEIN_GUIDE,
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
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": True,
            "guide": _SCHNELL_GUIDE,
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
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": True,
            "guide": _FLUX1_GUIDE,
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
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": True,
            "guide": _QWEN_IMAGE_GUIDE,
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
        "z-image-turbo",
        {
            "steps": 8,
            "guidance_scale": 0.0,
            "mu": None,
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": _Z_IMAGE_TURBO_GUIDE,
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
            "cfg_threshold": 0.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": _Z_IMAGE_GUIDE,
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
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": _ERNIE_TURBO_GUIDE,
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
            "cfg_threshold": 1.0,
            "negative_prompt_supported": True,
            "reference_supported": False,
            "true_cfg_supported": False,
            "guide": _ERNIE_GUIDE,
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
    ("krea-dev", {**_FLUX_PROFILE, "guide": _KREA_DEV_GUIDE}),
)

_FALLBACK: dict[str, Any] = {
    "steps": 28,
    "guidance_scale": 3.5,
    "mu": None,
    "cfg_threshold": 1.0,
    "negative_prompt_supported": True,
    "reference_supported": False,
    "true_cfg_supported": False,
    "guide": _FALLBACK_GUIDE,
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


_PROSE = ("hints", "guide")


def defaults_for(model_id: str) -> dict[str, Any]:
    """The numbers only - callers put these straight into a request or a field."""
    return {k: v for k, v in _profile_for(model_id).items() if k not in _PROSE}


def guide_for(model_id: str) -> dict[str, Any]:
    """The prompting guide's data half: the documented pixel range, the word
    for it, and the authors' example prompts. The prose half rides in
    `hints_for` under `prompt` and `size`, beside the other tooltips."""
    guide = _profile_for(model_id)["guide"]
    return {
        "pixels": guide["pixels"],
        "pixels_word": guide["pixels_word"],
        "examples": list(guide["examples"]),
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
