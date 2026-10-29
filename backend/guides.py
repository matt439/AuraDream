"""The prompting guides: what each checkpoint's authors put in writing.

`model_defaults` hangs one of these under each profile's "guide". `prompt`
and `size` are prose for the ? badges on the Prompt box and the Size presets
(paragraphs, so they hold no format fields - they go through `.format` with
the other hints, and a brace would fail
`test_every_profile_explains_its_controls`), `pixels` is the megapixel
range the checkpoint's authors put in writing, `pixels_word` is what that
writing amounts to - a model card's "supports 0.1-2 MP", a README's
"recommended 1024-1280" and a report's "trained to 1328" are not the same
promise, and the size line quotes the word. `"example"` is the weakest:
the span of the sizes the authors' own examples use, where they publish no
range at all (NetaYume, NewBie, WAI), which the line words as the examples'
size and never as what the checkpoint is "for" - in `pixels_source`'s words,
whose sizes they are as the size prose names them ("the author's reference
sizes" on WAI, one author and a floor that is no example). `aspects` and
`resolutions`, where the card publishes a finite table of shapes, mark the
presets that are its framings and recognise its exact sizes typed in. And
`examples` are the
authors' own prompts, verbatim, for the Examples menu; an `example_notes`
entry labels a contiguous excerpt where the full prompt cannot run here, or
says where an example parts from its own guide's advice (one too short, a
sign's text unquoted), so the menu says so before Check does.
Every sentence is
sourced from the model card, README, prompting guide or report, never
invented; docs/design-notes.md ("The prompting guide") keeps the URLs.

`analysis` is the same guide as data for `prompt_rules`, the analyser's rule
checks: one entry per checkable statement, each carrying the words it rests
on as `cite` - a verbatim piece of this guide's `prompt` prose, or of one of
its `examples` where the entry says `"source": "author"` - and a `url` of its
own or the guide's. An entry is a claim the prose already makes, never a new
one, and a guide that states nothing checkable has an empty one. It is not
part of `guide_for`: the page shows findings, not the rules behind them.
"""
from __future__ import annotations

from collections.abc import Collection

# The primary sources the `analysis` entries point at, as design-notes keeps
# them. BFL's unified guide covers the whole FLUX family (FLUX.1, Kontext,
# FLUX.2); its technical page is where "working without negative prompts"
# lives, its building page the length table.
_BFL_BASICS = "https://docs.bfl.ai/guides/prompting_unified_basics"
_BFL_BUILDING = "https://docs.bfl.ai/guides/prompting_unified_building"
_BFL_TECHNICAL = "https://docs.bfl.ai/guides/prompting_unified_technical"
_BFL_FLUX2 = "https://docs.bfl.ai/guides/prompting_guide_flux2"
# BFL's FLUX.1 Kontext pages, written around its [pro] API; the overview is
# where [dev] is "Editing mode only".
_BFL_KONTEXT = "https://docs.bfl.ai/kontext/kontext_overview"
_BFL_KONTEXT_EDITING = "https://docs.bfl.ai/kontext/kontext_image_editing"

_FLUX1_PROMPT = (
    "Plain descriptive sentences, not a tag list. Subject first, then setting, "
    "style, camera and lighting - what comes first carries the most weight. "
    "30-80 words suits most images; only the first 77 tokens reach the CLIP "
    "encoder, and T5 stops reading at 512.\n"
    "Put any text to render in quotation marks. There is no negative prompt "
    "unless True CFG is on, so describe what you want rather than what you do "
    "not."
)
_FLUX1_SIZE = (
    "Black Forest Labs documents FLUX.1 for 0.1 to 2 MP at any aspect ratio, "
    "sides in multiples of 16; 1024x1024 and 1360x768 are its reference sizes. "
    "Above 2 MP is outside that range and is where users report doubled "
    "subjects and smeared detail: render at 1-2 MP and upscale, which the "
    "Refine pass does here."
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
# BFL's samples are shorter than the 30-80 words its building page suggests,
# and its hello-world sign leaves the text unquoted. The examples stay as
# written; the menu says beside each where it parts from the guide's advice,
# so a Check of one is not the first news (`example_notes`, held by
# `test_the_authors_examples_draw_no_finding_unnoted`).
_SHORT = "Author sample: under 30-80 words"
_SHORT_SIGN = "Author sample: under 30-80 words, its sign's text unquoted"
FLUX1_GUIDE = {
    "prompt": _FLUX1_PROMPT,
    "size": _FLUX1_SIZE,
    "pixels": (0.1, 2.0),
    "pixels_word": "documented",
    "examples": _FLUX1_EXAMPLES,
    "example_notes": {
        _FLUX1_EXAMPLES[0]: _SHORT_SIGN,
        **{prompt: _SHORT for prompt in _FLUX1_EXAMPLES[1:]},
    },
    # The prose once ended "English is the most precise language"; BFL's
    # current basics page says FLUX answers other languages "with the same
    # level of quality", so it is gone from the prose, not only unchecked.
    # Each sentence carries one page's words, so a critic cite of it opens
    # the page its rule finding does (`prompt_sources`).
    "analysis": {
        "url": _BFL_BASICS,
        "style": {"expects": "prose", "cite": "Plain descriptive sentences, not a tag list"},
        "order": {
            "steps": ["subject", "setting", "style", "camera", "lighting"],
            "cite": (
                "Subject first, then setting, style, camera and lighting - what "
                "comes first carries the most weight"
            ),
            # BFL's building page: its template, and "Start with the image
            # type and subject".
            "url": _BFL_BUILDING,
        },
        "words": {"range": (30, 80), "cite": "30-80 words suits most images", "url": _BFL_BUILDING},
        "quote_text": {"cite": "Put any text to render in quotation marks."},
        # "bfl": the message may quote BFL's own replacement for a phrase its
        # table names ("no people" -> "empty", "deserted", "solitary").
        "negation": {
            "cite": "describe what you want rather than what you do not",
            "url": _BFL_TECHNICAL,
            "replacements": "bfl",
        },
    },
}
KREA_DEV_GUIDE = {
    "prompt": (
        "FLUX.1 prompting - descriptive sentences, text in quotation marks. Put "
        "the subject first, in 30-80 words. There is no negative prompt without "
        "True CFG. This checkpoint is tuned for photographs without the "
        "oversaturated 'AI look'. "
        "Camera settings - lens, framing, depth of field - are optional: BFL's "
        "guide adds them to a photograph where framing matters.\n"
        "Krea recommends 28-32 steps and CFG 3.5-5."
    ),
    "size": (
        "Krea's README recommends sides between 1024 and 1280 pixels (about 1 to "
        "1.6 MP), in multiples of 16; the FLUX.1 family is documented for 0.1 to "
        "2 MP. The 1 and 1.5 MP presets are inside the recommendation; 2 MP is "
        "just past it and usually holds; 4 MP is well outside - render smaller "
        "and upscale, which the Refine pass does here."
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
    "example_notes": {
        "Moody night photography, a small store with an orange glowing sign that "
        "says 'Black Forest Labs'": _SHORT,
        "White owl, close up portrait, mysterious ghost like owl, low contrast, "
        "cinematic dark moody photo, slight blue color grading": (
            "Author sample: a tag list, under 30-80 words"
        ),
        "Photo of two people having a tea party in a lush garden": _SHORT,
        "A frog holding a sign that says hello world": _SHORT_SIGN,
    },
    # Krea publishes no prompting rules of its own - its README is settings,
    # its posts say "opinionated" - so every entry is the FLUX.1 family's,
    # sourced to BFL. The camera sentence is BFL's building page ("Use these
    # when framing matters"); it replaced two sentences no page said - "photo
    # language" and "give it room" - which the critic then contradicted.
    # "Subject first" is that page's "Start with the image type and subject".
    # The 'AI look' is BFL's launch post's. The opening summary is three
    # sentences, one per BFL page, so each links the page its rules cite.
    "analysis": {
        "url": _BFL_BASICS,
        "style": {"expects": "prose", "cite": "descriptive sentences"},
        "order": {"steps": ["subject first"], "cite": "subject first", "url": _BFL_BUILDING},
        "words": {"range": (30, 80), "cite": "30-80 words", "url": _BFL_BUILDING},
        "quote_text": {"cite": "text in quotation marks"},
        "negation": {
            "cite": "no negative prompt without True CFG",
            "url": _BFL_TECHNICAL,
            "replacements": "bfl",
        },
    },
}
SCHNELL_GUIDE = {
    "prompt": (
        "FLUX.1 prompting - descriptive sentences, text in quotation marks. Put "
        "the subject first. schnell's T5 encoder reads up to 512 tokens in this "
        "app, as on [dev]. BFL's reference code gives schnell a shorter memory, "
        "256 tokens (about 180 words), which this app does not apply. Guidance is "
        "off on schnell, so a negative prompt does nothing unless True CFG is "
        "on; describe what you want instead."
    ),
    "size": _FLUX1_SIZE,
    "pixels": (0.1, 2.0),
    "pixels_word": "documented",
    "examples": _FLUX1_EXAMPLES,
    # No word count here (below), so only the sign's text parts from it.
    "example_notes": {_FLUX1_EXAMPLES[0]: "Author sample: its sign's text unquoted"},
    # No word limit: the "about 180 words" rests on a 256-token T5 limit this
    # app does not apply (FluxPipeline encodes 512 unless told otherwise), and
    # the encoder report measures the real one. The prose says so, and the
    # critic is not given that sentence (`prompt_sources`, `unsourced`).
    "analysis": {
        "url": _BFL_BASICS,
        "style": {"expects": "prose", "cite": "descriptive sentences"},
        "order": {"steps": ["subject first"], "cite": "subject first", "url": _BFL_BUILDING},
        "quote_text": {"cite": "text in quotation marks"},
        "negation": {
            "cite": "describe what you want instead",
            "url": _BFL_TECHNICAL,
            "replacements": "bfl",
        },
    },
}
KLEIN_GUIDE = {
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
    # The menu says "needs a reference image" beside the last itself.
    "example_notes": {
        "A cat holding a sign that says hello world": _SHORT_SIGN,
        "An Entry of a Sushi Restaurant, The text 'OPEN' appears in red neon "
        "letters above the door": _SHORT,
        "Soaking wet tiger cub taking shelter under a banana leaf in the rainy "
        "jungle, close up photo": _SHORT,
        "Turn this cat into a dog": _SHORT,
    },
    "analysis": {
        "url": _BFL_FLUX2,
        # JSON is the guide's own alternative to sentences, not a tag list.
        "style": {"expects": "prose", "accepts": ["json"], "cite": "as sentences"},
        "order": {
            "steps": ["subject", "action", "style", "context"],
            "cite": (
                "Subject, then action, then style, then context, as sentences: "
                "FLUX.2 weights what comes first."
            ),
        },
        "words": {"range": (30, 80), "cite": "30-80 words is BFL's sweet spot"},
        "quote_text": {
            "cite": (
                "Put every word that must appear in the picture in quotation marks "
                "- a sign or label without one comes out as gibberish."
            ),
        },
        "negation": {
            "cite": "There is no negative prompt on FLUX.2 at all: say what you want.",
            "replacements": "bfl",
        },
        "references": {
            "cite": "With reference images, say how each is used - 'the jacket from image 1'.",
        },
    },
}
# FLUX.1 Kontext [dev] edits a gallery image from an instruction and does not
# generate from text; until it is catalogued this guide is reached only by its
# exact id. Every sentence but the measured drift is BFL's Kontext pages'; the
# examples are BFL's own edit instructions, verbatim (the overview's straight
# apostrophes). Its `pixels` are the span of those examples' output sizes
# (1248x832 to 800x1328), which this app does not run: the size prose says
# that only square sources are qualified here, at 512x512, and that those
# sizes are no published maximum (`pixels_source` names them in its words).
_KONTEXT_DRIFT = (
    "Measured on this card, the rest of the picture still drifts: a recoloured "
    "ball also grew and moved."
)
KONTEXT_GUIDE = {
    "prompt": (
        "An instruction for the source image, not a description of a new one: say "
        "what to change, as in 'Change the car color to red'. "
        "BFL writes that Kontext keeps everything else untouched and the character "
        "consistent across multiple edits. "
        + _KONTEXT_DRIFT + "\n"
        "To change text in the picture, put the text in quotation marks: Replace "
        "'joy' with 'BFL'."
    ),
    "size": (
        "BFL's API keeps an edit as close to the input image's size as it can, "
        "rounded to multiples of 32, at about 1 MP and aspect ratios from 3:7 to "
        "7:3. That is FLUX.1 Kontext [pro]'s endpoint. The size highlights follow "
        "the output sizes of the authors' edit examples, about 1.04-1.06 MP; no "
        "published maximum is given for [dev]. Here [dev] is qualified for square "
        "sources only, which render at 512x512; other shapes are held until they "
        "are measured."
    ),
    "pixels": (1.04, 1.06),
    "pixels_word": "example",
    "pixels_source": "the authors' edit examples",
    "examples": [
        "Change the car color to red",
        "Remove the object from her face",
        "She is now taking a selfie in the streets of Freiburg, it's a lovely day out.",
        "It's now snowing, everything is covered in snow.",
        "Replace 'joy' with 'BFL'",
    ],
    "analysis": {
        "url": _BFL_KONTEXT_EDITING,
        # "The most effective way to edit text is using quotation marks
        # around the specific text you want to change".
        "quote_text": {"cite": "put the text in quotation marks"},
    },
}
_KREA2_PROMPT = (
    "Natural language. One cohesive paragraph. Krea's guide: long, detailed "
    "prompts give the best results, but a short one works too. Krea's report "
    "says the training captions were dense. Put words to render in quotation "
    "marks. Text rendering was a reward in post-training. The Qwen3-VL encoder "
    "reads 512 tokens and was chosen for multilingual reach, though Krea "
    "publishes no language list.\n"
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
# "One cohesive paragraph" is the expansion prompt's structure rule, the
# prompting page's companion; the rules' other words are docs/prompting.md.
# The two sit in sentences of their own, and the technical report's claims in
# theirs, so each sentence links one page (`prompt_sources`).
_KREA2_ANALYSIS = {
    "url": "https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md",
    "style": {"expects": "prose", "cite": "Natural language"},
    "one_paragraph": {
        "cite": "One cohesive paragraph",
        "url": "https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt",
    },
    "quote_text": {"cite": "Put words to render in quotation marks"},
}
KREA2_TURBO_GUIDE = {
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
    "analysis": _KREA2_ANALYSIS,
}
KREA2_RAW_GUIDE = {
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
    "analysis": _KREA2_ANALYSIS,
}
QWEN_IMAGE_GUIDE = {
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
    "example_notes": {
        "Bookstore window display. A sign displays \"New Arrivals This Week\". "
        "Below, a shelf tag with the text \"Best-Selling Novels Here\". To the "
        "side, a colorful poster advertises \"Author Meet And Greet on Saturday\" "
        "with a central portrait of the author. There are four books on the "
        "bookshelf, namely \"The light between worlds\" \"When stars are "
        "scattered\" \"The slient patient\" \"The night circus\"": (
            "Author sample: without the positive magic"
        ),
    },
    "analysis": {
        "url": "https://github.com/QwenLM/Qwen-Image",
        "style": {"expects": "prose", "cite": "A dense, caption-like description."},
        "quote_text": {"cite": "put every string in quotation marks and say where it goes"},
        # The README's `positive_magic`, verbatim, per language.
        "magic": {
            "en": ", Ultra HD, 4K, cinematic composition.",
            "zh": ", 超清，4K，电影级构图.",
            "cite": (
                "Qwen appends ', Ultra HD, 4K, cinematic composition.' to English "
                "prompts (Chinese: '超清，4K，电影级构图') and recommends it"
            ),
        },
        "negative_prompt": {"cite": "Sample with True CFG 4 and a negative prompt"},
    },
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
# No negative-prompt rule for the undistilled Z-Image: its card calls one
# optional, and it is not in the catalogue. "Long, detailed" is the
# maintainers' answer in the Turbo repo's discussion 8, not the card's.
_Z_IMAGE_ANALYSIS = {
    "url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
    "style": {
        "expects": "prose",
        "cite": "Long, detailed natural language",
        "url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8",
    },
    "quote_text": {"cite": "quote the words"},
}
_Z_IMAGE_SIZE = (
    "Tongyi documents 512x512 to 2048x2048 of total pixel area at any aspect "
    "ratio, sides in multiples of 16, with 1024x1024 as the card's example; the "
    "technical report puts the training range at up to 1k-1.5k. So 1-2 MP is "
    "inside what it was trained on, and 4 MP is the documented ceiling but "
    "beyond it."
)
Z_IMAGE_TURBO_GUIDE = {
    "prompt": _Z_IMAGE_PROMPT + (
        "Turbo runs without classifier-free guidance, so at CFG 0 the negative "
        "prompt is unused."
    ),
    "size": _Z_IMAGE_SIZE,
    "pixels": (0.26, 2.2),
    "pixels_word": "trained",
    "examples": _Z_IMAGE_EXAMPLES,
    "analysis": _Z_IMAGE_ANALYSIS,
}
Z_IMAGE_GUIDE = {
    "prompt": _Z_IMAGE_PROMPT + (
        "The undistilled Z-Image wants CFG 3-5 over 28-50 steps; its card calls "
        "a negative prompt optional, but powerful for removing unwanted content."
    ),
    "size": _Z_IMAGE_SIZE,
    "pixels": (0.26, 2.2),
    "pixels_word": "trained",
    "examples": _Z_IMAGE_EXAMPLES,
    "analysis": _Z_IMAGE_ANALYSIS,
}
Z_ANIME_GUIDE = {
    "prompt": (
        "Describe the character, clothing, pose, setting and lighting in natural "
        "language. The author favours descriptive sentences over tag lists. "
        "Use a negative prompt for unwanted features; Base supports it fully. "
        "Start with 40 steps and CFG 4, as in the author's Diffusers example. "
        "This is the experimental Base model, not the 4-step or 8-step variant."
    ),
    "size": (
        "The author documents approximately 512x512 through 2048x2048 at any "
        "aspect ratio. Suggested sizes include 1024x1024, 1024x1536 for detailed "
        "portraits and 1920x1088 for wide scenes. Start at 1-1.5 MP, then try "
        "2 MP; the 4 MP ceiling is an author claim, not a local quality guarantee."
    ),
    "pixels": (0.26, 4.2),
    "pixels_word": "documented",
    "examples": [
        # The card's complete "Good" example, ahead of its editable templates.
        "A young anime girl with long silver hair and golden eyes, wearing a "
        "traditional shrine maiden outfit with white haori and red hakama. "
        "She stands in a sunlit bamboo forest, cherry blossoms falling softly "
        "around her. Warm afternoon light filtering through the trees, detailed "
        "fabric shading, expressive face, calm serene expression, high quality "
        "anime illustration with fine line work.",
        "Detailed anime portrait of [character], soft rim lighting, expressive "
        "eyes with detailed reflections, fine hair strands, clean linework, "
        "professional anime illustration quality.",
        # The card's "Backgrounds & Landscapes" template, verbatim.
        "Anime [location] at [time of day], [lighting], [atmosphere], beautiful "
        "background art, wallpaper quality, highly detailed environment.",
    ],
    "analysis": {
        "url": "https://huggingface.co/SeeSee21/Z-Anime",
        "style": {
            "expects": "prose",
            "cite": "The author favours descriptive sentences over tag lists.",
        },
        # The card: "Negative prompts have full effect on Z-Anime Base and are
        # highly recommended." It names no text to use, so no suggestion.
        "negative_prompt": {
            "cite": "Use a negative prompt for unwanted features; Base supports it fully.",
        },
        # The examples are templates: each bracket is a slot to fill. The
        # patterns are read by the page's RegExp too (`guide_for`'s `slots`).
        "placeholders": [
            {
                "pattern": r"\[character\]",
                "cite": "Detailed anime portrait of [character]",
                "source": "author",
                "fill": "the character's name or description",
            },
            {
                "pattern": r"\[(?:location|time of day|lighting|atmosphere)\]",
                "cite": "Anime [location] at [time of day], [lighting], [atmosphere]",
                "source": "author",
            },
        ],
    },
}
_ERNIE_PROMPT = (
    "A content-rich, detailed visual description in one paragraph - subject, "
    "setting, colours, light, layout - the way the model card's own example "
    "reads. Use Expand to run the repo's prompt enhancer on a short prompt; "
    "its result is put back in this box for review and becomes the exact prompt "
    "the image records. Dense and layout-sensitive text is its strength, English and "
    "Chinese - posters, infographics, UI - so quote the words and say where "
    "they go. Baidu's report names Ministral-3 as the text encoder. The "
    "tokenizer config caps a prompt at 2048 tokens.\n"
)
_ERNIE_SIZE = (
    "Baidu recommends 1024x1024, 848x1264, 1264x848, 768x1376, 1376x768, 896x1200 "
    "and 1200x896 - every one about 1 MP - and trained through 256, 512 and "
    "1024 px stages; sides in multiples of 16. The 1 MP 1:1, 2:3, 3:2, 3:4 and "
    "4:3 presets are five of those sizes exactly, and 9:16 and 16:9 stand for "
    "768x1376 and 1376x768 at 16 px shorter; those seven are marked, and the "
    "size line names any other framing. 2 and 4 MP render on this card "
    "without failing, but nothing above 1 MP is documented."
)
# The card's "Recommended Parameters" table as data: the framings its seven
# sizes are (9:16 is 768x1376's, 1.79:1 against the preset's 1.77:1), and the
# sizes themselves, so Baidu's own 768x1376 typed in is recognised as one.
_ERNIE_ASPECTS = ["1:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"]
_ERNIE_RESOLUTIONS = [
    (1024, 1024),
    (848, 1264),
    (1264, 848),
    (768, 1376),
    (1376, 768),
    (896, 1200),
    (1200, 896),
]
_ERNIE_EXAMPLES = [
    "This is a photograph depicting an urban street scene. Shot at eye level, it "
    "shows a covered pedestrian or commercial street. Slightly below the center "
    "of the frame, a cyclist rides away from the camera toward the background. "
    "The overall color palette is warm, dominated by golden yellows and dark "
    "browns, evoking the atmosphere of dusk or early morning.",
    "一只黑白相间的中华田园犬",
    "Generate a webpage screenshot that briefly introduces LLMs",
]
_ERNIE_ANALYSIS = {
    "url": "https://huggingface.co/baidu/ERNIE-Image-Turbo",
    "style": {"expects": "prose", "cite": "A content-rich, detailed visual description"},
    "one_paragraph": {"cite": "in one paragraph"},
    "quote_text": {"cite": "quote the words and say where they go"},
}
ERNIE_TURBO_GUIDE = {
    "prompt": _ERNIE_PROMPT + (
        "Turbo runs at guidance 1, where the negative prompt does nothing; raise "
        "CFG above 1 to bring it in."
    ),
    "size": _ERNIE_SIZE,
    "pixels": (1.0, 1.0),
    "pixels_word": "documented",
    "aspects": _ERNIE_ASPECTS,
    "resolutions": _ERNIE_RESOLUTIONS,
    "examples": _ERNIE_EXAMPLES,
    "analysis": _ERNIE_ANALYSIS,
}
ERNIE_GUIDE = {
    "prompt": _ERNIE_PROMPT + (
        "The undistilled ERNIE-Image samples at CFG 4 over 50 steps, where the "
        "negative prompt applies."
    ),
    "size": _ERNIE_SIZE,
    "pixels": (1.0, 1.0),
    "pixels_word": "documented",
    "aspects": _ERNIE_ASPECTS,
    "resolutions": _ERNIE_RESOLUTIONS,
    "examples": _ERNIE_EXAMPLES,
    "analysis": _ERNIE_ANALYSIS,
}
_ANIMAGINE_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
    "fewer digits, cropped, worst quality, low quality, low score, bad score, "
    "average score, signature, watermark, username, blurry"
)
ANIMAGINE_GUIDE = {
    "prompt": (
        "Comma-separated Danbooru-style tags, not natural-language prose. Put "
        "1girl, 1boy or 1other first, then the character name and series, a "
        "safe or sensitive rating, and the subject, pose and scene tags; finish "
        "with: masterpiece, high score, great score, absurdres. Both CLIP "
        "tokenizers stop at 77 tokens, so put identity and essential details "
        "first. Text rendering is not supported.\n"
        "The Tags menu and Check name those steps Person count, Character, "
        "Series, Rating, Details and Quality; year, artist and meta tags go in "
        "Details.\n"
        "Autocomplete and the chips check Danbooru's live tag index up to this "
        "checkpoint's 2025-01-07 cutoff. Unknown is advisory, not blocked.\n"
        f"The model card recommends this negative prompt: {_ANIMAGINE_NEGATIVE}."
    ),
    "size": (
        "Animagine was trained at 1024x1024. Its recommended shapes are "
        "1024x1024; 1152x896, 1216x832, 1344x768 and 1536x640 landscape; "
        "and their portrait counterparts 896x1152, 832x1216, 768x1344 and "
        "640x1536 - all about 1 MP and multiples of 64. Every aspect is a 1 MP "
        "preset here; seven use those exact dimensions, while the existing "
        "3:2 pair keeps ERNIE-Image's nearby shapes. The authors warn "
        "that higher resolutions such as 1536x1536 can degrade because the "
        "training resolution was 1024x1024."
    ),
    "pixels": (0.98, 1.05),
    "pixels_word": "recommended",
    "aspects": ["1:1", "9:7", "3:2", "7:4", "12:5", "7:9", "2:3", "4:7", "5:12"],
    "resolutions": [
        (1024, 1024),
        (1152, 896),
        (1216, 832),
        (1344, 768),
        (1536, 640),
        (896, 1152),
        (832, 1216),
        (768, 1344),
        (640, 1536),
    ],
    "examples": [
        "1girl, green hair, sweater, looking at viewer, upper body, beanie, "
        "outdoors, night, turtleneck, masterpiece, high score, great score, absurdres",
        "1boy, male focus, green hair, sweater, looking at viewer, upper body, "
        "beanie, outdoors, night, turtleneck, masterpiece, high score, great score, "
        "absurdres",
        "1girl, firefly \\(honkai: star rail\\), honkai \\(series\\), honkai: star "
        "rail, safe, casual, solo, looking at viewer, outdoors, smile, reaching "
        "towards viewer, night, masterpiece, high score, great score, absurdres",
    ],
    # The card's complete named vocabulary. Character, series, subject, pose
    # and scene remain free text: the card prescribes their position but does
    # not publish a finite list, so the app must not invent one.
    "tag_helper": {
        "order": ["Person count", "Character", "Series", "Rating", "Details", "Quality"],
        "groups": [
            {"label": "Person count", "tags": ["1girl", "1boy", "1other"], "exclusive": True},
            {
                "label": "Rating",
                "tags": ["safe", "sensitive", "nsfw", "explicit"],
                "exclusive": True,
            },
            {
                "label": "Quality",
                "tags": ["masterpiece", "best quality", "low quality", "worst quality"],
            },
            {
                "label": "Score",
                "tags": [
                    "high score",
                    "great score",
                    "good score",
                    "average score",
                    "bad score",
                    "low score",
                ],
            },
        ],
        "year_tag": True,
        "vocabulary": "danbooru",
        "vocabulary_cutoff": "2025-01-07",
        "recommended_suffix": ["masterpiece", "high score", "great score", "absurdres"],
        "negative_prompt": _ANIMAGINE_NEGATIVE,
    },
    # `order` places a comma item by the helper's own groups, the suffix and
    # the year form, which are known here, or by its Danbooru category, which
    # only a lookup knows; its steps are `tag_helper.order`. The card's
    # template ends "everything else in any order", so year, artist and meta
    # tags share the Details slot.
    "analysis": {
        "url": "https://huggingface.co/cagliostrolab/animagine-xl-4.0",
        "style": {
            "expects": "tags",
            "cite": "Comma-separated Danbooru-style tags, not natural-language prose.",
        },
        "order": {
            "cite": (
                "Put 1girl, 1boy or 1other first, then the character name and series, "
                "a safe or sensitive rating, and the subject, pose and scene tags; "
                "finish with: masterpiece, high score, great score, absurdres."
            ),
            "groups": {
                "Person count": "Person count",
                "Rating": "Rating",
                "Quality": "Quality",
                "Score": "Quality",
            },
            "suffix": "Quality",
            "year": "Details",
            "categories": {
                "character": "Character",
                "copyright": "Series",
                "general": "Details",
                "artist": "Details",
                "meta": "Details",
            },
            "general": "Details",
        },
        "suffix": {"cite": "finish with: masterpiece, high score, great score, absurdres"},
        "quote_text": {"supported": False, "cite": "Text rendering is not supported."},
        "negative_prompt": {"cite": "The model card recommends this negative prompt"},
    },
}
KOHAKU_GUIDE = {
    "prompt": (
        "Kohaku XL Zeta accepts Danbooru tags, natural-language descriptions, "
        "or both; long prompts are optional. Its Epsilon-style order is person "
        "count, character, series, artist, general tags or prose, quality, "
        "date, meta tags, rating. Use spaces instead of underscores in tags.\n"
        "The Tags menu carries person count and the author's quality, rating "
        "and date tags. "
        "For the explicit rating the author uses both nsfw and explicit. "
        "Autocomplete checks Danbooru through the 2024-07-10 dataset cutoff; "
        "unknown tags remain advisory. The author specifies no fixed quality "
        "suffix or universal negative prompt. Negative prompts apply at CFG > 1.\n"
        "The card describes a 300-token training context, but this app's "
        "standard SDXL pipeline stops at 77 tokens in each supplied CLIP tokenizer. "
        "Keep prompts concise and put essential details first."
    ),
    "size": (
        "The author recommends 1024x1024 or a similar pixel count, about 1 MP. "
        "The 1 MP presets match that recommendation across aspect ratios. "
        "Larger presets are available but fall outside the recommended area; "
        "training bucket side limits are not a native megapixel guarantee."
    ),
    "pixels": (1024**2 / 1e6, 1024**2 / 1e6),
    "pixels_word": "recommended",
    # DEFAULT_TAGS from the author's TIPO-DEMO app.py; its image backend
    # defaults to Kohaku XL Zeta. The model card's JPEG samples have no prompt
    # metadata, and Hugging Face's generated library snippet is not author text.
    "examples": [
        "1girl, king halo (umamusume), umamusume,\n"
        "ningen mame, ciloranko, ogipote, misu kasumi,\n"
        "solo, leaning forward, sky,\n"
        "masterpiece, absurdres, sensitive, newest",
    ],
    "example_notes": {
        "1girl, king halo (umamusume), umamusume,\n"
        "ningen mame, ciloranko, ogipote, misu kasumi,\n"
        "solo, leaning forward, sky,\n"
        "masterpiece, absurdres, sensitive, newest": (
            "Author sample: its rating before its date, against the card's order"
        ),
    },
    "tag_helper": {
        "order": [
            "Person count",
            "Character",
            "Series",
            "Artist",
            "General tags / prose",
            "Quality",
            "Date",
            "Meta",
            "Rating",
        ],
        "groups": [
            {"label": "Person count", "tags": ["1girl", "1boy", "1other"], "exclusive": True},
            {
                "label": "Quality",
                "tags": [
                    "masterpiece",
                    "best quality",
                    "great quality",
                    "good quality",
                    "normal quality",
                    "low quality",
                    "worst quality",
                ],
                "exclusive": True,
            },
            {
                "label": "Date",
                "tags": ["newest", "recent", "mid", "early", "old"],
                "exclusive": True,
            },
            # Explicit captions carry both nsfw and explicit, so these must
            # remain separate toggles rather than mutually exclusive choices.
            {"label": "Rating", "tags": ["safe", "sensitive", "nsfw", "explicit"]},
        ],
        "vocabulary": "danbooru",
        "vocabulary_cutoff": "2024-07-10",
        "hybrid_prompt": True,
    },
    "analysis": {
        "url": (
            "https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/"
            "c14affd9483ac7603cfff3f06b05da22a06deaa6/README.md"
        ),
        "order": {
            "cite": (
                "Its Epsilon-style order is person count, character, series, artist, "
                "general tags or prose, quality, date, meta tags, rating."
            ),
            "url": "https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon-rev2#prompt-format",
            "groups": {
                "Person count": "Person count",
                "Quality": "Quality",
                "Date": "Date",
                "Rating": "Rating",
            },
            "categories": {
                "character": "Character",
                "copyright": "Series",
                "artist": "Artist",
                "general": "General tags / prose",
                "meta": "Meta",
            },
            "general": "General tags / prose",
        },
        # Delta's card keeps underscores in short emoji tags (^_^, o_o), so
        # only an underscore joining words counts.
        "underscores": {
            "cite": "Use spaces instead of underscores in tags.",
            "url": "https://huggingface.co/KBlueLeaf/Kohaku-XL-Delta#tags",
        },
        "explicit_pair": {
            "cite": "For the explicit rating the author uses both nsfw and explicit.",
            "url": "https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon-rev2#prompt-format",
        },
    },
}
# The negative prompt on both of the v0.1 model card's examples, verbatim. The
# v2.0 release example runs the short form `worst quality, bad quality`.
_ILLUSTRIOUS_NEGATIVE = (
    "worst quality, comic, multiple views, bad quality, low quality, lowres, "
    "displeasing, very displeasing, bad anatomy, bad hands, scan artifacts, "
    "monochrome, greyscale, signature, twitter username, jpeg artifacts, 2koma, "
    "4koma, guro, extra digits, fewer digits"
)
ILLUSTRIOUS_GUIDE = {
    "prompt": (
        "Illustrious XL v2.0 is an untuned base: its authors say it has no "
        "default style and was not trained with an aesthetic set, so a bare "
        "prompt renders the dataset's average look. Give it a style: a Danbooru "
        "artist tag or style tags. Add one composition tag (upper body, cowboy "
        "shot, portrait or full body); the authors warn that stacking "
        "conflicting composition tags confuses it.\n"
        "Tag lists and natural-language descriptions both work; their "
        "natural-language example ends its prose with tags (persona 3, "
        "masterpiece). The training order is person count, character, rating, "
        "general tags, artist, quality, year. The two CLIP encoders each stop at "
        "77 tokens, prose included, so keep it short: the quality and year tags "
        "at the end of that order are the first cut. Autocomplete and the chips check Danbooru's tag index through "
        "the dataset's August 2024 cutoff and label artist tags as artist.\n"
        "The rating slot takes Danbooru's own rating names - general, sensitive, "
        "questionable or explicit; the report's prompts use general. "
        "End a tag prompt with one quality tier (good quality, best quality or "
        "masterpiece) and optionally one year modifier (recent or newest). "
        "masterpiece, newest is a practical starting tail, not an official fixed "
        "suffix.\n"
        "Every example the authors publish carries a negative prompt. Their "
        f"v0.1 card runs both of its examples with: {_ILLUSTRIOUS_NEGATIVE}. "
        "Their v2.0 release example uses worst quality, bad quality. The Tags "
        "menu fills the long one. A negative prompt applies at CFG > 1."
    ),
    "size": (
        "The authors document native 1-2.25 MP generation for Illustrious v1.0+, "
        "trained through 1536x1536; v2.0 supports 512-1536 sides in multiples of "
        "32 at up to 10:1, and their v2.0 release example is 1824x1248 at 28 "
        "steps and CFG 6.5. Up to 4 MP is possible with some loss; "
        "their report also shows a direct 2048x2048 v1.0 result without upscaling. "
        "Start at 1-2.25 MP; the 4 MP preset is available but outside that native range."
    ),
    "pixels": (1.0, 2.25),
    "pixels_word": "documented",
    # Report figure 3's explicit prompt, the v0.1 card's second example and the
    # v2.0 release post's scenery prompt; the post's natural-language example
    # runs past 77 tokens, so it is quoted in the hint and not offered whole.
    # The helper uses the paper's own ordering and tables; Animagine's suffix
    # does not belong to this checkpoint. The Rating group is Danbooru's four
    # rating names - the post field the captions were built from, which the
    # tag index does not carry (they exist there only as deprecated, empty
    # placeholders), so nothing but a local group can accept them. The report
    # itself writes only `general`. A single free-form sentence remains outside
    # chip validation because this profile explicitly supports natural language.
    "examples": [
        "1boy, doctor, masterpiece, looking at viewer",
        "1girl, extremely dark, black theme, silhouette, rim lighting, black, "
        "looking at viewer, low contrast, masterpiece",
        "analogous colors, scenery, forest, masterpiece, black theme, dark, "
        "no lighting, night, black background",
    ],
    # The report's and the post's own prompts put masterpiece mid-list.
    "example_notes": {
        "1boy, doctor, masterpiece, looking at viewer": (
            "Author sample: masterpiece ahead of general tags, against the training order"
        ),
        "analogous colors, scenery, forest, masterpiece, black theme, dark, "
        "no lighting, night, black background": (
            "Author sample: masterpiece ahead of general tags, against the training order"
        ),
    },
    "tag_helper": {
        "order": [
            "Person count",
            "Character",
            "Rating",
            "General",
            "Artist",
            "Quality",
            "Year",
        ],
        "groups": [
            {"label": "Person count", "tags": ["1girl", "1boy"], "exclusive": True},
            {
                "label": "Rating",
                "tags": ["general", "sensitive", "questionable", "explicit"],
                "exclusive": True,
            },
            {
                "label": "Quality",
                "tags": [
                    "worst quality",
                    "bad quality",
                    "average quality",
                    "good quality",
                    "best quality",
                    "masterpiece",
                ],
                "exclusive": True,
            },
            {
                "label": "Year",
                "tags": ["oldest", "old", "modern", "recent", "newest"],
                "exclusive": True,
            },
        ],
        "vocabulary": "danbooru",
        # The archived v2.0 card says the dataset was updated through 2024-08;
        # use the inclusive end of that documented month as the lookup boundary.
        "vocabulary_cutoff": "2024-08-31",
        "hybrid_prompt": True,
        "negative_prompt": _ILLUSTRIOUS_NEGATIVE,
    },
    # The report's schema has no series slot, so copyright tags go with the
    # general ones; `rating_names` corrects only Animagine's `safe` - the
    # release post's sfw, nsfw and r-18 are accepted alternatives, not errors.
    "analysis": {
        "url": "https://arxiv.org/html/2409.19946v1",
        "order": {
            "cite": (
                "The training order is person count, character, rating, general tags, "
                "artist, quality, year."
            ),
            "groups": {
                "Person count": "Person count",
                "Rating": "Rating",
                "Quality": "Quality",
                "Year": "Year",
            },
            "categories": {
                "character": "Character",
                "copyright": "General",
                "general": "General",
                "artist": "Artist",
                "meta": "General",
            },
            "general": "General",
        },
        "quality_tier": {
            "tiers": ["good quality", "best quality", "masterpiece"],
            "group": "Quality",
            "years": "Year",
            "cite": (
                "End a tag prompt with one quality tier (good quality, best quality or "
                "masterpiece) and optionally one year modifier (recent or newest)."
            ),
        },
        "composition": {
            "tags": ["upper body", "cowboy shot", "portrait", "full body"],
            "cite": "the authors warn that stacking conflicting composition tags confuses it",
            "url": "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0",
        },
        "rating_names": {
            "replace": {"safe": "general"},
            "cite": (
                "The rating slot takes Danbooru's own rating names - general, "
                "sensitive, questionable or explicit"
            ),
        },
        "negative_prompt": {
            "cite": "Every example the authors publish carries a negative prompt.",
            "url": "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0",
        },
    },
}
_WAI_NEGATIVE = "bad quality,worst quality,worst detail,sketch,censor,"
_WAI_EXAMPLES = [
    # Contiguous scene excerpts from WAI0731's v17 Tensor.Art images
    # 990778028365743217 and 990778122855024231. Full sources in design-notes.
    "1girl, hatsune miku, dress, long hair, jewelry, white dress, blue eyes, "
    "necklace, bubble, solo, underwater, hair between eyes, looking at viewer, "
    "bangs, blue hair, bow",
    "1girl, horns, solo, japanese clothes, red eyes, long hair, white hair, "
    "sash, kimono, armor, obi, looking at viewer, oni horns, floral print, scarf",
]
WAI_GUIDE = {
    "prompt": (
        "WAI Illustrious v17 is an anime finetune with its own style. Use tags "
        "for the subject and scene; the family release notes also describe "
        "natural-language support. Finish with the author's quality recipe: "
        "masterpiece, best quality, amazing quality; add your subject before it. "
        "Examples contains scene excerpts from the author's v17 gallery. "
        "Both SDXL CLIP tokenizers "
        "stop at 77 tokens, so keep essential details near the beginning.\n"
        "The author's rating tags are general, sensitive, nsfw and explicit; "
        "the author suggests adding nsfw to the negative prompt when filtering "
        "sexual content. Autocomplete checks Danbooru's live index; WAI publishes "
        "no vocabulary cutoff, so a recognized tag is advisory rather than "
        "evidence that v17 learned it.\n"
        # Quoted and closed: the recipe ends in a comma, and without the full
        # stop the next sentence - the app's, not the card's - ran into it.
        f"The v17 card recommends this negative prompt: '{_WAI_NEGATIVE}'. "
        "The card's weighted-prompt syntax is not implemented by this "
        "text-to-image pipeline."
    ),
    "size": (
        "The v17 author recommends original dimensions larger than 1024x1024 "
        "and uses 1024x1344 in the examples. The size highlights follow those "
        "documented reference sizes, about 1.05-1.38 MP; no published maximum "
        "is given. A size above that reference span is not a claim of unsupported "
        "resolution. The author's examples then use a separate 1.5x high-resolution "
        "refinement pass at 20 steps and denoising strength 0.35-0.5 with "
        "R-ESRGAN 4x+ Anime6B. Generate renders directly at the requested size. "
        "This app's Refine pass defaults to Lanczos and offers opt-in R-ESRGAN "
        "4x+ Anime6B, followed by a Lanczos resize to the target size. It runs "
        "the part of Steps the strength leaves, rather than a separate hires "
        "step count. The GAN option still needs local card quality, memory and "
        "timing verification; it is not a claim of matching the author's images."
    ),
    "pixels": (1024**2 / 1e6, 1024 * 1344 / 1e6),
    # Reference sizes, not a range: the size line says so (sizes.js). One
    # author's, and the floor is the size they say to exceed rather than an
    # example, so the line calls them what the prose does.
    "pixels_word": "example",
    "pixels_source": "the author's reference sizes",
    # The author's gallery supplies scenes; the card's recipe remains in Tags.
    "examples": _WAI_EXAMPLES,
    "example_notes": {
        prompt: "Author scene excerpt; quality suffix omitted"
        for prompt in _WAI_EXAMPLES
    },
    "tag_helper": {
        "order": ["Subject and scene", "Quality"],
        "groups": [
            {
                "label": "Rating",
                "tags": ["general", "sensitive", "nsfw", "explicit"],
                "exclusive": True,
            },
            {
                "label": "Quality",
                "tags": ["masterpiece", "best quality", "amazing quality"],
            },
        ],
        "vocabulary": "danbooru",
        "vocabulary_cutoff": None,
        "hybrid_prompt": True,
        "recommended_suffix": ["masterpiece", "best quality", "amazing quality"],
        "negative_prompt": _WAI_NEGATIVE,
    },
    # Two slots: everything the author calls subject and scene, then the
    # recipe. Where a rating tag goes is not written, so it is subject. The
    # cites stay clear of the Examples-menu sentence, which is the
    # app's menu talking (`prompt_sources`, `unsourced`).
    "analysis": {
        "url": "https://tensor.art/models/990778216270553015/WAI-illustrious-v17",
        "order": {
            "cite": (
                "Finish with the author's quality recipe: masterpiece, best quality, "
                "amazing quality; add your subject before it."
            ),
            "groups": {"Quality": "Quality", "Rating": "Subject and scene"},
            "suffix": "Quality",
            "categories": {
                "character": "Subject and scene",
                "copyright": "Subject and scene",
                "general": "Subject and scene",
                "artist": "Subject and scene",
                "meta": "Subject and scene",
            },
            "general": "Subject and scene",
        },
        "suffix": {
            "cite": (
                "Finish with the author's quality recipe: masterpiece, best quality, "
                "amazing quality"
            ),
        },
        "subject": {"cite": "add your subject before it"},
        "rating_negative": {
            "ratings": ["general", "sensitive"],
            "cite": (
                "the author suggests adding nsfw to the negative prompt when filtering "
                "sexual content"
            ),
        },
        "negative_prompt": {"cite": "The v17 card recommends this negative prompt"},
    },
}

_NOOBAI_PREFIX = "masterpiece, best quality, newest, absurdres, highres, safe,"
_NOOBAI_NEGATIVE = (
    "nsfw, worst quality, old, early, low quality, lowres, signature, username, "
    "logo, bad hands, mutated hands, mammal, anthro, furry, ambiguous form, "
    "feral, semi-anthro"
)
NOOBAI_GUIDE = {
    "prompt": (
        "NoobAI-XL V-pred 1.0 was trained with native Danbooru and e621 tags "
        "and natural-language captions. The card recommends the prefix "
        f"'{_NOOBAI_PREFIX}'. Examples includes the card's complete Diffusers prompt. "
        "Its separate caption template lists person count, character, series, "
        "artists, special tags, general tags, then other tags. The author's "
        "Diffusers example uses artist: before artist names.\n"
        "Quality tags are masterpiece, best quality, good quality, normal quality "
        "and worst quality. Aesthetic tags include very awa and worst aesthetic. "
        "Date tags use year followed by four digits, or old, early, mid, recent "
        "and newest; newest covers the years 2021-2024. "
        f"The card recommends this negative prompt: '{_NOOBAI_NEGATIVE}'.\n"
        "Both SDXL CLIP tokenizers stop at 77 tokens, including special tokens. "
        "Autocomplete checks Danbooru through the card's approximate 2024-10-23 "
        "cutoff; it does not check e621, and a match does not prove the model "
        "learned a tag."
    ),
    "size": (
        "The author recommends about 1024x1024 total area, roughly 1 MP: "
        "768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832 and "
        "1344x768, with 832x1216 highlighted. The app's 2:3 and 3:2 presets "
        "have slightly different dimensions; type the card's sizes for an exact "
        "match. Direct generation has been checked at 1024x1024 and Refine "
        "from that size to 1536x1536 at strength 0.4. Larger direct renders "
        "remain unmeasured here."
    ),
    "pixels": (1.0, 1024**2 / 1e6),
    "pixels_word": "recommended",
    "aspects": ["1:1", "7:9", "9:7", "2:3", "3:2", "4:7", "7:4"],
    "resolutions": [
        (768, 1344), (832, 1216), (896, 1152), (1024, 1024),
        (1152, 896), (1216, 832), (1344, 768),
    ],
    # The pinned card's complete Diffusers prompt: 76 tokens including the
    # special tokens in each published CLIP tokenizer (77 available).
    "examples": [
        "masterpiece, best quality,artist:john_kafka,artist:nixeu,artist:quasarcake, "
        "chromatic aberration, film grain, horror \\(theme\\), limited palette, "
        "x-shaped pupils, high contrast, color contrast, cold colors, arlecchino "
        "\\(genshin impact\\), black theme,  gritty, graphite \\(medium\\)",
    ],
    "tag_helper": {
        # The prefix and caption template are distinct author recommendations,
        # not one unambiguous total order. Show the template through the critic
        # sources, without inventing placement for its "other tags" bucket.
        "order": [],
        "groups": [
            {"label": "Person count", "tags": ["1girl", "1boy", "1other"]},
            {"label": "Rating", "tags": ["safe"]},
            {
                "label": "Quality",
                "tags": [
                    "masterpiece", "best quality", "good quality",
                    "normal quality", "worst quality",
                ],
            },
            {"label": "Aesthetic", "tags": ["very awa", "worst aesthetic"]},
            # "Date tags" in the prose above, as on the card; "Year" is the
            # helper's own heading for the year-NNNN form beside it.
            {"label": "Date", "tags": ["old", "early", "mid", "recent", "newest"]},
            {"label": "Resolution", "tags": ["absurdres", "highres"]},
        ],
        "year_tag": True,
        "vocabulary": "danbooru",
        "vocabulary_cutoff": "2024-10-23",
        "artist_prefix": "artist:",
        "hybrid_prompt": True,
        "negative_prompt": _NOOBAI_NEGATIVE,
    },
    "analysis": {
        "url": "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0",
        "negative_prompt": {"cite": "The card recommends this negative prompt"},
    },
}

_ROUWEI_EXAMPLES = [
    # The full scene chunks from the pinned author's examples_epsilon/28.png
    # and 33.png, without their separate style/quality chunk and BREAK.
    "Three girls Koyoi with white fox ears and red hakama, Kunoichi-chan with "
    "wolf ears and blue floral kimono, and Mutsuki with cat ears and purple "
    "kimonostand barefoot in shallow water at night, holding glowing paper "
    "lanterns. The scene features soft bokeh lighting, water reflections, and "
    "a soft-focus foliage background.",
    "1girl, senko \\(sewayaki kitsune no senko-san\\), fox girl, fox ears, fox "
    "tail, blonde hair, yellow eyes, hair ornament, miko, apron, red hakama, "
    "white socks, geta, tilted head,\nindoors, kitchen, table, holding a plate "
    "with a slice of strawberry cake",
]
ROUWEI_GUIDE = {
    "prompt": (
        "RouWei 0.8 epsilon is an Illustrious retrain for anime and other illustration "
        "styles. Use Danbooru tags, natural language, or both; the author recommends "
        "combining tags with descriptive phrases. Write comma-separated tags with "
        "spaces instead of underscores.\n"
        "The only positive quality tags are masterpiece and best quality; both are "
        "optional. The negative quality tags are low quality and worst quality; the "
        "author says to retain low quality. Do not use lowres: the author removed "
        "that meta tag. Keep the negative prompt short; the author's recipe is "
        "worst quality, low quality, watermark. Examples contains the author's "
        "scene excerpts without the separate style and quality chunk.\n"
        "Put by followed by a space before each artist name. The author asks for "
        "artist tags in a separate CLIP chunk, or at least at the very end of the "
        "prompt. This app does not implement separate chunks: BREAK is literal "
        "text, and both SDXL CLIP tokenizers stop at 77 tokens. Keep the complete "
        "prompt within that limit so the artists at its end are read.\n"
        "Autocomplete checks Danbooru through the author's end-of-April-2025 "
        "dataset cutoff; a match is advisory, not proof the checkpoint learned it."
    ),
    "size": (
        "The author recommends about 1-1.5 MP for direct generation, any aspect "
        "ratio with dimensions divisible by 64. The app's presets are shared "
        "between checkpoints; check custom dimensions against that multiple. "
        "The author's separate hires pass uses a 1.5x latent upscale at strength "
        "0.6, or a GAN upscaler at strength 0.3-0.55. This app's Refine pass "
        "starts at 0.4 and defaults to Lanczos, with opt-in R-ESRGAN 4x+ "
        "Anime6B followed by a Lanczos resize to the target size. The author "
        "does not name a particular GAN; this option is not the latent recipe "
        "and still needs local card quality, memory and timing verification."
    ),
    "pixels": (1.0, 1.5),
    "pixels_word": "recommended",
    "examples": _ROUWEI_EXAMPLES,
    "example_notes": {
        prompt: "Author scene excerpt; separate style and quality chunk omitted"
        for prompt in _ROUWEI_EXAMPLES
    },
    "tag_helper": {
        "order": [],
        "groups": [
            {"label": "Quality", "tags": ["masterpiece", "best quality"]},
            {
                "label": "Style",
                "tags": [
                    "2.5d", "anime screencap", "bold line", "sketch", "cgi",
                    "digital painting", "flat colors", "smooth shading", "minimalistic",
                    "ink style", "oil style", "pastel style",
                ],
            },
            {
                "label": "Color",
                "tags": [
                    "low brightness", "high brightness", "low saturation", "high saturation",
                    "low gamma", "high gamma", "sharp colors", "soft colors", "hdr", "sdr",
                ],
            },
        ],
        "vocabulary": "danbooru",
        "vocabulary_cutoff": "2025-04-30",
        "artist_prefix": "by ",
        "hybrid_prompt": True,
        "negative_prompt": "worst quality, low quality, watermark",
    },
    "analysis": {
        "url": "https://huggingface.co/Minthy/RouWei-0.8",
        "underscores": {"cite": "Write comma-separated tags with spaces instead of underscores."},
        "artist_prefix": {
            "prefix": "by ",
            "cite": "Put by followed by a space before each artist name.",
        },
        "negative_prompt": {"cite": "the author's recipe is worst quality, low quality, watermark"},
    },
}

# The card's tag order, verbatim, and its sections as the Tags menu and
# Check name them.
_ANIMA_ORDER = (
    "[quality/meta/year/safety tags] [1girl/1boy/1other etc] [character] "
    "[series] [artist] [general tags]"
)
_ANIMA_STEPS = ["Quality/meta/year/safety", "Person count", "Character", "Series", "Artist", "General"]
# The family's spelling and order, then (after the variant's own sentences)
# the order, the Tags menu and the vocabulary. Every sentence about the
# model is the card's own words or one already in the family's prose.
_ANIMA_SPELLING = (
    "Use tags, natural language, or both. Write tags in lowercase with "
    "spaces instead of underscores, except score tags, and put @ before "
    "an artist name.\n"
    "For natural language, describe the image in at least two sentences. "
    "Use normal capitalization for character and series names, and describe "
    "each character's appearance when naming more than one. "
)
_ANIMA_TAGS = (
    f"The card's tag order: {_ANIMA_ORDER}. Within each tag section, the tags "
    "can be in arbitrary order. You can mix tags and natural language in "
    "arbitrary order.\n"
    "The Tags menu and Check name those sections Quality/meta/year/safety, "
    "Person count, Character, Series, Artist and General; Check leaves a "
    "sentence where it is written. Autocomplete and the chips check "
    "Danbooru's tag index; the card names no cutoff. When using a tag that "
    "is different between Danbooru and Gelbooru, prefer the Gelbooru version."
)


def _anima_helper(*, score: bool) -> dict:
    """The card's named tags, by its own sections: the quality tags (and,
    where the variant's advice allows them, the score tags), the time
    period and meta tags, the safety tags and the person count its order
    names. Character, series, artist and general tags stay free text."""
    groups = [
        {"label": "Person count", "tags": ["1girl", "1boy", "1other"], "exclusive": True},
        {
            "label": "Quality",
            "tags": [
                "masterpiece", "best quality", "good quality", "normal quality",
                "low quality", "worst quality",
            ],
        },
    ]
    if score:
        # "score_9, score_8, ..., score_1" on the card.
        groups.append({"label": "Score", "tags": [f"score_{n}" for n in range(9, 0, -1)]})
    groups += [
        {"label": "Period", "tags": ["newest", "recent", "mid", "early", "old"]},
        {
            "label": "Meta",
            "tags": ["highres", "absurdres", "anime screenshot", "jpeg artifacts", "official art"],
        },
        {"label": "Safety", "tags": ["safe", "sensitive", "nsfw", "explicit"], "exclusive": True},
    ]
    return {
        "order": list(_ANIMA_STEPS),
        "groups": groups,
        "year_tag": True,
        "vocabulary": "danbooru",
        # The card names none; the lookup asks the live index.
        "vocabulary_cutoff": None,
        "artist_prefix": "@",
        "hybrid_prompt": True,
    }


def _anima_order(groups: Collection[str]) -> dict:
    """`analysis.order` over a variant's helper groups. Danbooru's meta
    category goes with the general tags: the card's full tag example writes
    twitter username, a Danbooru meta tag, among them, and the meta tags it
    names first are the Meta group's, which place by the group. An @ name
    takes the Artist slot by its prefix (`prompt_order._place`), and a
    sentence stands anywhere (`free_sentences`). The score tags take the
    first slot whether or not the menu offers them: Aesthetic's leaves them
    out, but the card's order and its full tag example still place them."""
    first = _ANIMA_STEPS[0]
    scores = [] if "Score" in groups else [f"score_{n}" for n in range(9, 0, -1)]
    return {
        "cite": _ANIMA_ORDER,
        "groups": {label: ("Person count" if label == "Person count" else first) for label in groups},
        "year": first,
        "categories": {
            "character": "Character",
            "copyright": "Series",
            "artist": "Artist",
            "general": "General",
            "meta": "General",
        },
        "general": "General",
        "free_sentences": True,
        **({"tags": {first: scores}} if scores else {}),
    }


_ANIMA_TURBO_HELPER = {
    **_anima_helper(score=True),
    # Offered, never checked: the card's "Recommended positive prefix".
    "recommended_prefix": ["masterpiece", "best quality", "score_7", "safe"],
}
_ANIMA_AESTHETIC_HELPER = _anima_helper(score=False)

_ANIMA_ANALYSIS = {
    "url": "https://huggingface.co/circlestone-labs/Anima",
    "underscores": {
        "keep": r"score_\d+(?:_up)?",
        "cite": "with spaces instead of underscores, except score tags",
    },
    "lowercase": {"cite": "Write tags in lowercase"},
    "artist_prefix": {"cite": "put @ before an artist name"},
    "sentences": {
        "min": 2,
        "cite": "For natural language, describe the image in at least two sentences.",
    },
}

ANIMA_TURBO_GUIDE = {
    "prompt": (
        "Anima Turbo makes anime and other illustrations. Its distillation "
        "adds a consistent default style, with less variety than the base. "
        + _ANIMA_SPELLING
        + "At Turbo's recommended CFG 1, negative prompts have no effect.\n"
        'Recommended positive prefix: "masterpiece, best quality, score_7, safe, ". '
        "The Tags menu offers it as Start with.\n"
        + _ANIMA_TAGS
    ),
    "size": (
        "The authors document 512x512 to 1536x1536 pixels for Anima, roughly "
        "0.26-2.36 MP. The 1, 1.5 and 2 MP presets are inside this range; the "
        "4 MP preset is outside it. This is the documented range, not a measurement "
        "on this card."
    ),
    "pixels": (512**2 / 1e6, 1536**2 / 1e6),
    "pixels_word": "documented",
    # The model card's dragon caption, both of its sentences, without its
    # optional dataset/title prefix. Keep the excerpt verbatim, and whole: the
    # card asks for at least two sentences, and its first alone drew the
    # analyser's own "1 sentence" note on the authors' example.
    "examples": [
        "Digital painting of a fiery dragon with glowing yellow eyes, black "
        "horns, and a long, sinuous tail, perched on a glowing, molten rock formation. "
        "The background is a gradient of dark purple to orange."
    ],
    "tag_helper": _ANIMA_TURBO_HELPER,
    # The card's tag order is checked; its positive prefix is offered (the
    # Tags menu's Start with), never a finding.
    "analysis": {
        **_ANIMA_ANALYSIS,
        "order": _anima_order({group["label"] for group in _ANIMA_TURBO_HELPER["groups"]}),
    },
}

# The variants share the family's size range, example, spelling rules and
# order. Aesthetic has its own quality advice and is not distilled like
# Turbo: its Tags menu leaves out the score tags its section advises
# against, and it has no prefix to start with, since the card's carries
# score_7.
ANIMA_AESTHETIC_GUIDE = {
    **ANIMA_TURBO_GUIDE,
    "prompt": (
        "Anima Aesthetic makes anime and other illustrations. It is fine-tuned "
        "for consistency and a higher-quality default art style. "
        + _ANIMA_SPELLING
        + "Aesthetic does not need positive quality tags; masterpiece and best "
        "quality may stay. The author recommends avoiding score_* tags in both "
        "the positive and negative prompts. "
        "A negative prompt applies at CFG > 1.\n"
        + _ANIMA_TAGS
    ),
    "tag_helper": _ANIMA_AESTHETIC_HELPER,
    "analysis": {
        **_ANIMA_ANALYSIS,
        "order": _anima_order({group["label"] for group in _ANIMA_AESTHETIC_HELPER["groups"]}),
    },
}

# The preview's author follows the Anima vocabulary, but lists prompting
# ingredients rather than a fixed sequence. Keep that vocabulary without
# importing Turbo's score prefix, checked order or two-sentence minimum.
ANIMA_PREVIEW_GUIDE = {
    "prompt": (
        "Anima 2.9B preview v1 makes anime and illustrations from tags, natural "
        "language, or both. Describe the scene and background in detail; the author "
        "warns that short prompts often produce simple backgrounds. Put @ before "
        "an artist name, follow character names with series tags, and describe "
        "each character's appearance in multi-character scenes. The training "
        "captions have no score tags, though the author says they may still be used.\n"
        "Following the parent Anima guide, write tags in lowercase with spaces "
        "instead of underscores, except score tags. The Tags menu uses the parent "
        "Anima vocabulary; autocomplete checks Danbooru's live index. This app "
        "checks no fixed tag order. The example is inherited from the parent "
        "Anima card. A negative prompt applies at CFG > 1. This app uses fixed "
        "FlowMatch Euler; the author's alternative samplers and schedules are "
        "different setups."
    ),
    "size": (
        "The author recommends 812x1216 and 1152x1536, about 0.99-1.77 MP, "
        "and calls 1536x1536 (2.36 MP) iffy. Size highlighting follows the first "
        "two sizes. The listed 812-pixel side is off this app's 16-pixel grid; "
        "use a nearby 1 MP preset. These are author recommendations, not "
        "measurements on this card."
    ),
    "pixels": (812 * 1216 / 1e6, 1152 * 1536 / 1e6),
    "pixels_word": "recommended",
    "examples": list(ANIMA_TURBO_GUIDE["examples"]),
    "example_notes": {
        example: "Parent Anima author example; the preview card publishes no scene prompt."
        for example in ANIMA_TURBO_GUIDE["examples"]
    },
    "tag_helper": {**_anima_helper(score=False), "order": []},
    "analysis": {
        "url": "https://huggingface.co/Gazingstars123/Anima-2.9B",
        "underscores": {
            "keep": r"score_\d+(?:_up)?",
            "cite": "write tags in lowercase with spaces instead of underscores, except score tags",
            "url": "https://huggingface.co/circlestone-labs/Anima",
        },
        "lowercase": {
            "cite": "write tags in lowercase",
            "url": "https://huggingface.co/circlestone-labs/Anima",
        },
        "artist_prefix": {"cite": "Put @ before an artist name"},
    },
}

NETAYUME_GUIDE = {
    "prompt": (
        "NetaYume Lumina v4 is an anime finetune of Neta Lumina. Its author's "
        "published v2 family guide recommends Danbooru tags; the Neta Lumina "
        "base also supports natural-language descriptions. The Examples menu uses the image "
        "prompt from the author's v2 workflow.\n"
        "The author's anime system instruction is applied automatically: "
        "write only the image's tags or prose here, without a system prefix "
        "or <Prompt Start> marker. The standard Lumina 2 pipeline uses a "
        "256-token total sequence, including the system instruction. A "
        "negative prompt applies at CFG > 1. This app uses standard "
        "FlowMatch Euler; the author's ComfyUI sampler recommendations "
        "are different setups."
    ),
    "size": (
        "The inherited Neta Lumina guide gives a 1024x1024 example, and "
        "NetaYume's published v2 workflow uses 1024x1536. These documented "
        "examples span about 1.05-1.57 MP; the size highlighting follows "
        "that example span. Start at 1 MP. The NetaYume card publishes no "
        "v4 resolution range: 1.57 MP is not a supported maximum or a "
        "quality limit, and larger sizes have not been measured here."
    ),
    "pixels": (1024**2 / 1e6, 1024 * 1536 / 1e6),
    # Two example sizes, which the prose above says are no bound: the Neta
    # Lumina guide's and NetaYume's own workflow's, so more than one author.
    "pixels_word": "example",
    "pixels_source": "the authors' example sizes",
    # The v2 workflow's positive image prompt, verbatim after its system
    # instruction and <Prompt Start> marker. The loader applies the system
    # instruction itself, so copying that prefix here would duplicate it.
    "examples": [
        "Gawr Gura, 1girl, solo, anime style, vibrant colors, blue eyes, blush, "
        "standing, holding a sign, sign says \"I love you\", english text, "
        "front-facing, full body, soft lighting, humorous mood, masterpiece, "
        "best quality, amazing quality, indoors"
    ],
    # The instruction itself is `netayume_loading.SYSTEM_PROMPT`, which the
    # rule reads from there rather than from a copy.
    "analysis": {
        "url": (
            "https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/"
            "69c29c441119d43405d8373c654b16827d3cf46e/README.md"
        ),
        "system_prefix": {"cite": "without a system prefix or <Prompt Start> marker"},
    },
}

NEWBIE_GUIDE = {
    "prompt": (
        "NewBie image Exp0.1 is an experimental anime model. It accepts "
        "natural language, tags, or XML. The authors recommend XML for "
        "scenes with multiple characters to keep each character's attributes "
        "together. Use a separate <character_1>, <character_2> block with "
        "name, gender, appearance, clothing, expression, action and position; "
        "put the shared scene in <general_tags>. The Examples menu includes "
        "the author's two-character XML prompt. Keep $character_1$ and "
        "$character_2$ for unnamed characters, or replace them with names. "
        "Gemma reads up to 512 tokens "
        "and Jina CLIP up to 2048 in this app; keep the main description "
        "within Gemma's shorter limit."
    ),
    "size": (
        "The author's quickstart uses 1024x1024, about 1.05 MP. Start at "
        "the 1 MP square preset; the size highlighting follows that "
        "documented example area. The card publishes no minimum or maximum "
        "resolution range: this example is not a supported maximum or a "
        "quality limit."
    ),
    "pixels": (1024**2 / 1e6, 1024**2 / 1e6),
    # One example size, which the prose above says is no bound: the
    # author's quickstart.
    "pixels_word": "example",
    "pixels_source": "the author's example",
    # Model card's XML example, with indentation normalized. These are XML
    # fields, not a flat comma-separated tag list: do not add a tag helper.
    "examples": [
        "<character_1>\n"
        "<n>$character_1$</n>\n"
        "<gender>1girl</gender>\n"
        "<appearance>chibi, red_eyes, blue_hair, long_hair, hair_between_eyes, "
        "head_tilt, tareme, closed_mouth</appearance>\n"
        "<clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, "
        "short_sleeves, red_neckerchief, bow, blue_skirt, miniskirt, "
        "pleated_skirt, blue_hat, mini_hat, thighhighs, grey_thighhighs, "
        "black_shoes, mary_janes</clothing>\n"
        "<expression>happy, smile</expression>\n"
        "<action>standing, holding, holding_briefcase</action>\n"
        "<position>center_left</position>\n"
        "</character_1>\n\n"
        "<character_2>\n"
        "<n>$character_2$</n>\n"
        "<gender>1girl</gender>\n"
        "<appearance>chibi, red_eyes, pink_hair, long_hair, very_long_hair, "
        "multi-tied_hair, open_mouth</appearance>\n"
        "<clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, "
        "short_sleeves, red_neckerchief, bow, red_skirt, miniskirt, "
        "pleated_skirt, hair_bow, multiple_hair_bows, white_bow, ribbon_trim, "
        "ribbon-trimmed_bow, white_thighhighs, black_shoes, mary_janes, "
        "bow_legwear, bare_arms</clothing>\n"
        "<expression>happy, smile</expression>\n"
        "<action>standing, holding, holding_briefcase, waving</action>\n"
        "<position>center_right</position>\n"
        "</character_2>\n\n"
        "<general_tags>\n"
        "<count>2girls, multiple_girls</count>\n"
        "<style>anime_style, digital_art</style>\n"
        "<background>white_background, simple_background</background>\n"
        "<atmosphere>cheerful</atmosphere>\n"
        "<quality>high_resolution, detailed</quality>\n"
        "<objects>briefcase</objects>\n"
        "<other>alternate_costume</other>\n"
        "</general_tags>",
    ],
    # The prose says "name"; the card's example writes <n>, so either counts.
    "analysis": {
        "url": (
            "https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/tree/"
            "4f7825a1a7e1ed5c1882a52e3c51d1119a43e298"
        ),
        "order": {
            "steps": ["one <character_N> block per character", "<general_tags>"],
            "cite": (
                "Use a separate <character_1>, <character_2> block with name, gender, "
                "appearance, clothing, expression, action and position; put the "
                "shared scene in <general_tags>."
            ),
        },
        "xml": {
            "names": ["n", "name"],
            "fields": ["gender", "appearance", "clothing", "expression", "action", "position"],
            "scene": "general_tags",
            "cite": (
                "Use a separate <character_1>, <character_2> block with name, gender, "
                "appearance, clothing, expression, action and position; put the "
                "shared scene in <general_tags>."
            ),
        },
        "xml_multi": {
            "cite": (
                "The authors recommend XML for scenes with multiple characters to "
                "keep each character's attributes together."
            ),
        },
    },
}

FALLBACK_GUIDE = {
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
    # The prose above is the app's, not an author's: nothing for a rule to
    # cite, and the critic is told these are general sentences, not an
    # author's (`critic.build_context`).
    "analysis": {},
}
