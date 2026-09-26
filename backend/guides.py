"""The prompting guides: what each checkpoint's authors put in writing.

`model_defaults` hangs one of these under each profile's "guide". `prompt`
and `size` are prose for the ? badges on the Prompt box and the Size presets
(paragraphs, so they hold no format fields - they go through `.format` with
the other hints, and a brace would fail
`test_every_profile_explains_its_controls`), `pixels` is the megapixel
range the checkpoint's authors put in writing, `pixels_word` is what that
writing amounts to - a model card's "supports 0.1-2 MP", a README's
"recommended 1024-1280" and a report's "trained to 1328" are not the same
promise, and the size line quotes the word - and `examples` are the
authors' own prompts, verbatim, for the Examples menu. Every sentence is
sourced from the model card, README, prompting guide or report, never
invented; docs/design-notes.md ("The prompting guide") keeps the URLs.
"""
from __future__ import annotations

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
FLUX1_GUIDE = {
    "prompt": _FLUX1_PROMPT,
    "size": _FLUX1_SIZE,
    "pixels": (0.1, 2.0),
    "pixels_word": "documented",
    "examples": _FLUX1_EXAMPLES,
}
KREA_DEV_GUIDE = {
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
SCHNELL_GUIDE = {
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
Z_IMAGE_TURBO_GUIDE = {
    "prompt": _Z_IMAGE_PROMPT + (
        "Turbo runs without classifier-free guidance, so at CFG 0 the negative "
        "prompt is unused."
    ),
    "size": _Z_IMAGE_SIZE,
    "pixels": (0.26, 2.2),
    "pixels_word": "trained",
    "examples": _Z_IMAGE_EXAMPLES,
}
Z_IMAGE_GUIDE = {
    "prompt": _Z_IMAGE_PROMPT + (
        "The undistilled Z-Image wants CFG 3-5 over 28-50 steps, and Tongyi "
        "strongly recommends a negative prompt with it."
    ),
    "size": _Z_IMAGE_SIZE,
    "pixels": (0.26, 2.2),
    "pixels_word": "trained",
    "examples": _Z_IMAGE_EXAMPLES,
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
        "Detailed anime portrait of [character], soft rim lighting, expressive "
        "eyes with detailed reflections, fine hair strands, clean linework, "
        "professional anime illustration quality.",
    ],
}
_ERNIE_PROMPT = (
    "A content-rich, detailed visual description in one paragraph - subject, "
    "setting, colours, light, layout - the way the model card's own example "
    "reads. Use Expand to run the repo's prompt enhancer on a short prompt; "
    "its result is put back in this box for review and becomes the exact prompt "
    "the image records. Dense and layout-sensitive text is its strength, English and "
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
ERNIE_TURBO_GUIDE = {
    "prompt": _ERNIE_PROMPT + (
        "Turbo runs at guidance 1, where the negative prompt does nothing; raise "
        "CFG above 1 to bring it in."
    ),
    "size": _ERNIE_SIZE,
    "pixels": (1.0, 1.0),
    "pixels_word": "documented",
    "examples": _ERNIE_EXAMPLES,
}
ERNIE_GUIDE = {
    "prompt": _ERNIE_PROMPT + (
        "The undistilled ERNIE-Image samples at CFG 4 over 50 steps, where the "
        "negative prompt applies."
    ),
    "size": _ERNIE_SIZE,
    "pixels": (1.0, 1.0),
    "pixels_word": "documented",
    "examples": _ERNIE_EXAMPLES,
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
}
KOHAKU_GUIDE = {
    "prompt": (
        "Kohaku XL Zeta accepts Danbooru tags, natural-language descriptions, "
        "or both; long prompts are optional. Its Epsilon-style order is person "
        "count, character, series, artist, general tags or prose, quality, "
        "date, meta tags, rating. Use spaces instead of underscores in tags.\n"
        "The Tags menu carries the author's quality, rating and date tags. "
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
        "prompt renders the dataset's average look. Give it a style - a Danbooru "
        "artist tag (autocomplete labels them artist) or style tags - and one "
        "composition tag (upper body, cowboy shot, portrait or full body); the "
        "authors warn that stacking conflicting composition tags confuses it.\n"
        "Tag lists and natural-language descriptions both work; their "
        "natural-language example ends its prose with tags (persona 3, "
        "masterpiece). The training order is person count, character, rating, "
        "general tags, artist, quality, year. The two CLIP encoders each stop at "
        "77 tokens, prose included, so keep it short or put the quality tags "
        "early. Autocomplete and the chips check Danbooru's tag index through "
        "the dataset's August 2024 cutoff.\n"
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
}
ANIMA_TURBO_GUIDE = {
    "prompt": (
        "Anima Turbo makes anime and other illustrations. Its distillation "
        "adds a consistent default style, with less variety than the base. "
        "Use tags, natural language, or both. Write tags in lowercase with "
        "spaces instead of underscores, except score tags, and put @ before "
        "an artist name.\n"
        "For natural language, describe the image in at least two sentences. "
        "Use normal capitalization for character and series names, and describe "
        "each character's appearance when naming more than one. "
        "At Turbo's recommended CFG 1, negative prompts have no effect."
    ),
    "size": (
        "The authors document 512x512 to 1536x1536 pixels for Anima, roughly "
        "0.26-2.36 MP. The 1 and 2 MP presets are inside this range; the 4 MP "
        "preset is outside it. This is the documented range, not a measurement "
        "on this card."
    ),
    "pixels": (512**2 / 1e6, 1536**2 / 1e6),
    "pixels_word": "documented",
    # First sentence of the model card's dragon caption, without its optional
    # dataset/title prefix. Keep the excerpt verbatim.
    "examples": [
        "Digital painting of a fiery dragon with glowing yellow eyes, black "
        "horns, and a long, sinuous tail, perched on a glowing, molten rock formation."
    ],
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
    "pixels_word": "documented",
    # The v2 workflow's positive image prompt, verbatim after its system
    # instruction and <Prompt Start> marker. The loader applies the system
    # instruction itself, so copying that prefix here would duplicate it.
    "examples": [
        "Gawr Gura, 1girl, solo, anime style, vibrant colors, blue eyes, blush, "
        "standing, holding a sign, sign says \"I love you\", english text, "
        "front-facing, full body, soft lighting, humorous mood, masterpiece, "
        "best quality, amazing quality, indoors"
    ],
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
}
