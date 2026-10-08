"""What the prompt critic may cite: the guide's own sentences and the authors' words.

The critic (`critic.py`) is a 4B language model, and left to itself it gives
the advice it has seen most - "add 8k, masterpiece" to a checkpoint whose
authors forbid meta tags, quotation marks around a negation, a camera where
the prompt already names one. What makes it usable is that it judges against a
fixed, numbered list and cites one entry per issue, the ids being an enum in
the answer's schema. This module is that list, per guide constant:

- **G sentences** - the guide's `prompt` prose split into sentences, the same
  text the ? badge shows, so a citation can never name advice the page does not.
  Their link is the page of the `analysis` entry whose cite they carry - the
  page that entry's rule finding links - so a rule finding and a critic cite
  of the same words open the same page; the prose gives each page's words a
  sentence of their own, which a test holds. A sentence no entry cites links
  to the page `sentence_urls` names for it (Krea [dev]'s settings are Krea's
  README, Illustrious's v0.1 negative prompt its v0.1 card, ERNIE's 2048
  tokens its tokenizer's config), else to the guide's `analysis["url"]`. A
  sentence the prose carries without an author behind it (`unsourced`) - the
  app describing its own menus, buttons and what it does not implement, or
  its own judgement - is left out: the page may say it, the critic may not
  raise an issue from it.
- **A excerpts** - short, rule-bearing passages the checkpoint's authors put
  in writing: a model card's tag order, BFL's negation table, the system
  prompts the authors wrote for their own prompt rewriters. Verbatim, trimmed
  of markdown emphasis; where the original is Chinese, `text` is this app's
  translation and says so in `note`, with the original kept in `original`.
  Nothing whole - no model card, no rewriter prompt entire - and nothing from
  NewBie's captioning instruction beyond its XML field rules.

An excerpt that was not written for this exact checkpoint says whose it is:
FLUX.2's guide was written for [pro] and [max] and applies to klein by family,
NetaYume inherits its parent Neta Lumina's prompt book, and FLUX.1 Krea [dev]
has no prompting rules of Krea's at all - only Black Forest Labs' family guide,
which is what `caveat` tells the critic.

Guides are matched by object identity through `model_defaults`, the same path
the ? badge takes, so a checkpoint cannot be critiqued against one guide while
the page shows another.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import guides
from .model_defaults import _profile_for


@dataclass(frozen=True)
class Excerpt:
    # Stable across releases, unlike the A1..An the critic sees, which is
    # positional per checkpoint.
    id: str
    text: str
    url: str
    # Whose words these are when not this checkpoint's own authors', or that
    # `text` is a translation. Sent to the critic beside the text.
    note: str | None = None
    # The authors' own words when `text` is this app's translation of them.
    original: str | None = None
    # An order the authors give - a tag sequence, a rewriter's sections - on a
    # checkpoint whose guide states none for the app to check. The page shows
    # these under Order as information (`PromptAnalysis.author_orders`).
    order: bool = False


@dataclass(frozen=True)
class Grounding:
    """One guide constant's sources."""

    excerpts: tuple[Excerpt, ...] = ()
    # Told to the critic ahead of the sources, where how far they reach needs
    # saying: Krea [dev] has only BFL's family guide, NetaYume's system prefix
    # is the app's job, and so on.
    caveat: str | None = None
    # G sentences, verbatim, the critic may not cite - no author page says
    # them, or they do not hold in this app: the ? badge carries them, the
    # critic is not given them.
    unsourced: tuple[str, ...] = ()
    # (sentence, url) for a G sentence no `analysis` entry cites whose page -
    # the authors' page, or their file that sets a limit - is not the guide's
    # `analysis["url"]`.
    sentence_urls: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GuideSentence:
    text: str
    url: str | None


@dataclass(frozen=True)
class Sources:
    """A checkpoint's grounding, resolved through its profile."""

    model_id: str
    guide_name: str
    guide: Mapping[str, Any]
    # The citable G sentences, in the prose's order.
    sentences: tuple[GuideSentence, ...]
    # The guide's `analysis["url"]`: the link of a G sentence that neither an
    # entry's cite nor `sentence_urls` gives another. None only for the
    # fallback, which has no authors.
    url: str | None
    excerpts: tuple[Excerpt, ...]
    caveat: str | None


# A sentence ends at . ! or ? followed by whitespace, and a paragraph at a
# newline. Dots inside FLUX.1 or v2.0 have no whitespace after them, and none
# of the guides writes "e.g." - which is what a test holds.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")


def _sentence_spans(prose: str) -> list[tuple[int, int]]:
    """Where in the prose each G sentence lies, as (start, end)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for end, following in [
        *((match.start(), match.end()) for match in _SENTENCE_BREAK.finditer(prose)),
        (len(prose), len(prose)),
    ]:
        part = prose[start:end]
        if part.strip():
            lead = len(part) - len(part.lstrip())
            spans.append((start + lead, start + len(part.rstrip())))
        start = following
    return spans


def guide_sentences(prose: str) -> list[str]:
    """The guide's prompt prose as the numbered G sentences, in order."""
    return [prose[start:end] for start, end in _sentence_spans(prose)]


def _cited_pages(guide: Mapping[str, Any]) -> list[tuple[int, int, str | None]]:
    """(start, end, url) of every `analysis` entry's cite in the prose, with
    the url its rule finding links: the entry's own, else the guide's."""
    analysis = guide.get("analysis") or {}
    entries = [
        entry
        for value in analysis.values()
        for entry in (value if isinstance(value, list) else [value])
        if isinstance(entry, dict) and entry.get("cite") and entry.get("source") != "author"
    ]
    return [
        (match.start(), match.end(), entry.get("url") or analysis.get("url"))
        for entry in entries
        for match in re.finditer(re.escape(entry["cite"]), guide["prompt"])
    ]


def _page(cited: list[tuple[int, int, str | None]], start: int, end: int) -> str | None:
    """The page of the entries cited in the sentence at start..end - one page
    per sentence, which a test holds, else the first by where its words begin
    there - or None where no cite touches it."""
    touching = [
        (max(first, start), url) for first, last, url in cited if first < end and last > start
    ]
    return min(touching, key=lambda item: item[0])[1] if touching else None


# --- Black Forest Labs: the FLUX family pages --------------------------------

_BFL_BASICS = "https://docs.bfl.ai/guides/prompting_unified_basics"
_BFL_BUILDING = "https://docs.bfl.ai/guides/prompting_unified_building"
_BFL_TECHNICAL = "https://docs.bfl.ai/guides/prompting_unified_technical"
_BFL_FLUX2 = "https://docs.bfl.ai/guides/prompting_guide_flux2"
_FLUX2_UPSAMPLING = (
    "https://github.com/black-forest-labs/flux2/blob/main/src/flux2/system_messages.py"
)
_BFL_FAMILY = "BFL's guide for the whole FLUX family"

def _bfl_replacements(note: str) -> Excerpt:
    return Excerpt(
        "bfl.replacements",
        '"no people" → "empty", "deserted", "solitary"; "no text" → "clean surfaces", '
        '"unmarked", "blank"; "not dark" → "brightly lit", "sun-drenched"',
        _BFL_TECHNICAL,
        note + "; three rows of its table of common replacements",
    )


def _bfl_family(note: str) -> tuple[Excerpt, ...]:
    return (
        Excerpt(
            "bfl.image_type",
            "Start with the image type and subject.",
            _BFL_BUILDING,
            note,
        ),
        Excerpt(
            "bfl.template",
            "[SUBJECT], [LOCATION], [STYLE], [CAMERA SETTINGS], [LIGHTING], [COLORS], "
            "[EFFECT], [ADDITIONAL ELEMENTS] … This template is a useful starting "
            "structure, not a strict formula.",
            _BFL_BASICS,
            note,
        ),
        Excerpt(
            "bfl.pattern",
            "For FLUX, the most reliable pattern is usually: 1. Start with a clear "
            "subject 2. Add the main action or state 3. Add mood, context, and visual "
            "direction only when they improve the image",
            _BFL_BUILDING,
            note,
        ),
        Excerpt(
            "bfl.subject_first",
            "Prompt order matters here too. If FLUX keeps pulling too far back, make "
            "the subject clear first and move environmental details later in the "
            "sentence.",
            _BFL_BUILDING,
            note,
        ),
        Excerpt(
            "bfl.style_early",
            "If style is central to the result, mention it early and keep it concrete.",
            _BFL_BUILDING,
            note,
        ),
        Excerpt(
            "bfl.natural",
            "FLUX works best when your prompt reads like a clear description of the "
            "image you want to generate.",
            _BFL_BASICS,
            note,
        ),
        Excerpt(
            "bfl.filler",
            "Start short. Add only what changes the image. More words do not "
            "automatically mean better results.",
            _BFL_BUILDING,
            note,
        ),
        Excerpt(
            "bfl.quotes",
            "When you want FLUX to generate specific text inside an image, place the "
            "exact wording in quotation marks.",
            _BFL_BASICS,
            note,
        ),
        Excerpt(
            "bfl.negation",
            "Most FLUX models do not support negative prompts. Even when they can "
            "process them, AI models generally struggle with negation — writing \"a "
            "person without glasses\" causes the model to focus on \"glasses\" and often "
            "generate exactly what you were trying to avoid.",
            _BFL_TECHNICAL,
            note,
        ),
        _bfl_replacements(note),
        Excerpt(
            "bfl.front_load",
            "Front-load the positive description — word order signals priority",
            _BFL_TECHNICAL,
            note,
        ),
    )


_FLUX2_NOTE = "written for FLUX.2 [pro] and [max]; klein by family"
_FLUX2_UPSAMPLING_NOTE = (
    "BFL's FLUX.2 prompt-upsampling instruction, which the official CLI also runs for klein"
)
_KLEIN = (
    Excerpt(
        "flux2.framework",
        "Use this framework for consistent results: Subject + Action + Style + Context",
        _BFL_FLUX2,
        _FLUX2_NOTE,
    ),
    Excerpt(
        "flux2.word_order",
        "Word order matters - FLUX.2 pays more attention to what comes first. Put your "
        "most important elements at the beginning:",
        _BFL_FLUX2,
        _FLUX2_NOTE,
    ),
    Excerpt(
        "flux2.priority",
        "Priority order: Main subject → Key action → Critical style → Essential "
        "context → Secondary details",
        _BFL_FLUX2,
        _FLUX2_NOTE,
    ),
    Excerpt(
        "flux2.no_negative",
        "FLUX.2 does not support negative prompts. Focus on describing what you want, "
        "not what you don't want.",
        _BFL_FLUX2,
        _FLUX2_NOTE,
    ),
    Excerpt(
        "flux2.length",
        "Medium (30-80 words): Usually ideal for most projects",
        _BFL_FLUX2,
        _FLUX2_NOTE,
    ),
    Excerpt(
        "flux2.klein",
        "On FLUX.2 [klein], what you write is what you get — be descriptive.",
        _BFL_TECHNICAL,
    ),
    Excerpt(
        "flux2.text",
        "Put ALL text in quotation marks, matching the prompt's language. Always "
        "provide explicit quoted text for objects that would contain text in reality "
        "(signs, labels, screens, etc.) - without it, the model generates gibberish.",
        _FLUX2_UPSAMPLING,
        _FLUX2_UPSAMPLING_NOTE,
    ),
    Excerpt(
        "flux2.details",
        "Add concrete visual specifics - form, scale, textures, materials, lighting "
        "(quality, direction, color), shadows, spatial relationships, and environmental "
        "context.",
        _FLUX2_UPSAMPLING,
        _FLUX2_UPSAMPLING_NOTE,
    ),
    _bfl_replacements(_BFL_FAMILY),
)

# --- Krea 2 ------------------------------------------------------------------

_KREA2_GUIDE_URL = "https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md"
_KREA2_EXPANSION = "https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt"
_KREA2_EXPANSION_NOTE = "Krea's system prompt for an LLM that expands prompts"
_KREA2 = (
    Excerpt(
        "krea2.natural",
        "We recommend users to use natural language prompts to generate images.",
        _KREA2_GUIDE_URL,
    ),
    Excerpt(
        "krea2.long",
        "Long detailed prompts yield best results, but the model is capable of "
        "generating high quality images with minimal prompt engineering.",
        _KREA2_GUIDE_URL,
    ),
    Excerpt(
        "krea2.quotes",
        "For text rendering, we recommend putting quotes around the words to be rendered.",
        _KREA2_GUIDE_URL,
    ),
    Excerpt(
        "krea2.faithful",
        "Faithfulness First: Preserve all original subjects, actions, colors, and "
        "spatial relationships. Do not add new objects, props, characters, or animals "
        "unless the user clearly implies them.",
        _KREA2_EXPANSION,
        _KREA2_EXPANSION_NOTE,
    ),
    Excerpt(
        "krea2.structure",
        "Practical T2I Structure: Write a prompt that a text-to-image model can parse "
        "cleanly. Group subjects with their own attributes and actions. Use grounded "
        "phrasing for poses, interactions, and spatial layout.",
        _KREA2_EXPANSION,
        _KREA2_EXPANSION_NOTE,
    ),
    Excerpt(
        "krea2.text",
        "Text Rendering: If the user requests visible text, quotes, labels, or "
        "typography, specify the exact text clearly and wrap requested words in quotes.",
        _KREA2_EXPANSION,
        _KREA2_EXPANSION_NOTE,
    ),
    Excerpt(
        "krea2.overspecification",
        "Avoid Over-Specification: Do not invent highly specific clothing, colors, "
        "materials, or scene details unless the input supports them.",
        _KREA2_EXPANSION,
        _KREA2_EXPANSION_NOTE,
    ),
    Excerpt(
        "krea2.paragraph",
        "Structure: Write one cohesive paragraph after the thinking block. No bullets, "
        "JSON, or markdown.",
        _KREA2_EXPANSION,
        _KREA2_EXPANSION_NOTE,
    ),
    Excerpt(
        "krea2.existing_detail",
        "Respect Existing Detail: If the user's prompt is already detailed, lightly "
        "polish and finalize rather than heavily expanding — preserve their phrasing "
        "and direction.",
        _KREA2_EXPANSION,
        _KREA2_EXPANSION_NOTE,
    ),
)

# --- Qwen-Image --------------------------------------------------------------

_QWEN_UTILS = (
    "https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py"
)
_QWEN_UTILS_2512 = (
    "https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils_2512.py"
)
_QWEN_NOTE = "Qwen's official prompt-enhancement instruction"
_QWEN = (
    Excerpt(
        "qwen.refine",
        "Refine descriptions of subject characteristics, visual style, spatial "
        "relationships, and shot composition;",
        _QWEN_UTILS,
        _QWEN_NOTE,
    ),
    Excerpt(
        "qwen.text",
        "If the input requires rendering text in the image, enclose specific text in "
        "quotation marks, specify its position (e.g., top-left corner, bottom-right "
        "corner) and style. This text should remain unaltered and not translated;",
        _QWEN_UTILS,
        _QWEN_NOTE,
    ),
    Excerpt(
        "qwen.style",
        "Match the Prompt to a precise, niche style aligned with the user’s intent. If "
        "unspecified, choose the most appropriate style (e.g., realistic photography "
        "style);",
        _QWEN_UTILS,
        _QWEN_NOTE,
    ),
    Excerpt(
        "qwen.length",
        "Please ensure that the Rewritten Prompt is less than 200 words.",
        _QWEN_UTILS,
        _QWEN_NOTE,
    ),
    Excerpt(
        "qwen.no_negation",
        "The rewritten prompt should not contain any negation words. For example, if "
        "the user input is \"no chopsticks\", chopsticks should not appear in the "
        "rewritten prompt.",
        _QWEN_UTILS,
        "app translation of rule 8 of Qwen's Chinese prompt-enhancement instruction",
        "改写之后的prompt中不应该出现任何否定词。如：用户输入为“不要有筷子”，"
        "则改写之后的prompt中不应该出现筷子。",
    ),
    Excerpt(
        "qwen.portrait_flow",
        "To ensure clarity, a logical flow is recommended for portrait descriptions. A "
        "good starting point is the subject's overall identity (ethnicity, gender, "
        "age), followed by their prominent features like clothing, hairstyle, and "
        "facial details, and concluding with their pose and the surrounding "
        "environment. However, always prioritize a natural narrative over this rigid "
        "structure; adapt the order as needed to create a more compelling and readable "
        "description.",
        _QWEN_UTILS_2512,
        "written for Qwen-Image-2512, the same family",
        order=True,
    ),
)

# --- Z-Image -----------------------------------------------------------------

_Z_IMAGE_PE = "https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py"


def _z_image(note: str) -> tuple[Excerpt, ...]:
    return (
        Excerpt(
            "zimage.core",
            "First, analyse and lock the core elements of the user's prompt that must "
            "not change: subject, quantity, action, state, and any specified IP names, "
            "colours and text. These are the foundation you must preserve absolutely.",
            _Z_IMAGE_PE,
            note,
            "首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，"
            "以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。",
        ),
        Excerpt(
            "zimage.detail",
            "Then, once the core picture is set (whether it came directly from the user "
            "or from your reasoning), you add professional aesthetic and realistic "
            "detail to it. This includes a clear composition, the lighting and "
            "atmosphere, material textures, a colour scheme, and a space built in layers.",
            _Z_IMAGE_PE,
            note,
            "然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的"
            "美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，"
            "并构建富有层次感的空间。",
        ),
        Excerpt(
            "zimage.text",
            "Transcribe every piece of text that should appear in the final image word "
            "for word, and enclose it in English double quotes (\"\") as an explicit "
            "generation instruction.",
            _Z_IMAGE_PE,
            note,
            "你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用"
            "英文双引号（\"\"）括起来，以此作为明确的生成指令。",
        ),
        Excerpt(
            "zimage.objective",
            "Your final description must be objective and concrete; metaphors and "
            "emotional rhetoric are strictly forbidden, and it never contains meta-tags "
            "or drawing instructions such as \"8K\" or \"masterpiece\".",
            _Z_IMAGE_PE,
            note,
            "你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含\"8K\"、"
            "\"杰作\"等元标签或绘制指令。",
        ),
    )


_Z_IMAGE_TURBO_NOTE = "app translation of the Z-Image Turbo Space's prompt-enhancer instruction"
_Z_IMAGE_BASE_NOTE = (
    "app translation of the Z-Image Turbo Space's prompt-enhancer instruction; same family"
)

# --- ERNIE-Image -------------------------------------------------------------

_ERNIE_PE = "https://github.com/baidu/ERNIE-Image/blob/main/src/pe_prompt.txt"
_ERNIE_NOTE = "app translation of Baidu's prompt-enhancer instruction"
_ERNIE = (
    Excerpt(
        "ernie.visible",
        "Aim at the picture to be generated and describe what should actually be "
        "visible in it, without subjective evaluation.",
        _ERNIE_PE,
        _ERNIE_NOTE,
        "以最终要生成出来的画面为目标，描述其中实际应当可见的内容，不进行主观评价。",
    ),
    Excerpt(
        "ernie.sections",
        "1. Image type 2. Subject and scene description 3. Visual details and style "
        "4. Entity recognition 5. Text transcription rules 6. Content instantiation "
        "rules 7. Interface and additional elements (if applicable)",
        _ERNIE_PE,
        _ERNIE_NOTE + "; its section headings, in their order",
        "1. 图片类型说明 2. 主体与场景描述 3. 视觉细节与风格 4. 实体识别 5. 文字抄录规则 "
        "6. 内容实例化规则 7. 界面与附加元素（如适用）",
        order=True,
    ),
    Excerpt(
        "ernie.image_type",
        "State the image type explicitly, such as photo, screenshot, illustration, hand "
        "drawing, poster, advertisement or interface screenshot.",
        _ERNIE_PE,
        _ERNIE_NOTE,
        "明确指出图片类型，如：照片、截图、插画、手绘图、海报、广告图、界面截图等。",
    ),
    Excerpt(
        "ernie.text",
        "Transcribe word for word in the original language, wrapped in English single "
        "quotes. The text must be described in full; omitting any of it is not allowed.",
        _ERNIE_PE,
        _ERNIE_NOTE,
        "按原文语种逐字抄录，用英文单引号包裹。 … 文字内容需要完整描述出来，不允许省略任何文字内容",
    ),
    Excerpt(
        "ernie.character",
        "If the user's prompt contains a specific subject, such as a particular anime "
        "character, the description must include that character's name and must not "
        "over-describe the character's features, to avoid errors.",
        _ERNIE_PE,
        _ERNIE_NOTE,
        "如果用户 prompt 中包含某个特定主体，例如二次元的某个角色，生成的描述中要有这个角色的"
        "名字，并且不要过度描述角色的特征，防止出错。",
    ),
    Excerpt(
        "ernie.slots",
        "Do not keep abstract slots from the user's request as they are; expand them "
        "as far as possible into concrete content that can be seen, drawn and rendered.",
        _ERNIE_PE,
        _ERNIE_NOTE,
        "不要把用户需求里的抽象槽位原样保留下来；应尽量把它们展开成可见、可画、可渲染的具体内容。",
    ),
)

# --- Z-Anime -----------------------------------------------------------------

_Z_ANIME_CARD = "https://huggingface.co/SeeSee21/Z-Anime"
_Z_ANIME = (
    Excerpt("zanime.natural", "Natural language works best — not tag lists.", _Z_ANIME_CARD),
    Excerpt(
        "zanime.good",
        "A young anime girl with long silver hair and golden eyes, wearing a "
        "traditional shrine maiden outfit with white haori and red hakama.",
        _Z_ANIME_CARD,
        "the start of the card's example under Good",
    ),
    Excerpt(
        "zanime.avoid",
        "anime girl, silver hair, shrine maiden, bamboo, cherry blossom, warm light",
        _Z_ANIME_CARD,
        "the card's example under Avoid",
    ),
    Excerpt(
        "zanime.template",
        "Anime [location] at [time of day], [lighting], [atmosphere], beautiful "
        "background art, wallpaper quality, highly detailed environment.",
        _Z_ANIME_CARD,
        "a card template; the brackets are slots to fill",
    ),
    Excerpt(
        "zanime.negative",
        "Negative prompts have full effect on Z-Anime Base and are highly recommended.",
        _Z_ANIME_CARD,
    ),
)

# --- The tag checkpoints -----------------------------------------------------

_ANIMAGINE_CARD = "https://huggingface.co/cagliostrolab/animagine-xl-4.0"
_ANIMAGINE = (
    Excerpt(
        "animagine.template",
        "The model was trained with tag-based captions and the tag-ordering method. Use "
        "this structured template: 1girl/1boy/1other, character name, from which "
        "series, rating, everything else in any order and end with quality enhancement",
        _ANIMAGINE_CARD,
    ),
    Excerpt(
        "animagine.quality",
        "Add these tags at the end of your prompt: masterpiece, high score, great "
        "score, absurdres",
        _ANIMAGINE_CARD,
    ),
    Excerpt(
        "animagine.tags_only",
        "Prompt Format: Limited to tag-based text prompts; natural language input may "
        "not be effective",
        _ANIMAGINE_CARD,
    ),
)

_KOHAKU_ZETA = "https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta"
_KOHAKU = (
    Excerpt(
        "kohaku.format",
        "<1girl/1boy/1other/...>, <character>, <series>, <artists>, <general tags>, "
        "<quality tags>, <year tags>, <meta tags>, <rating tags>",
        "https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon-rev2#prompt-format",
        "Kohaku XL Epsilon rev2's prompt format, which the Zeta card refers to",
    ),
    Excerpt(
        "kohaku.zeta",
        'As same as Kohaku XL Epsilon or Delta, but you can replace "general tags" '
        'with "natural language caption". You can also put both together.',
        _KOHAKU_ZETA,
    ),
    Excerpt(
        "kohaku.explicit",
        "Explicit: nsfw, explicit",
        _KOHAKU_ZETA,
        "the card's rating tags for explicit content",
    ),
    Excerpt(
        "kohaku.underscores",
        "Remember to remove all the underscore in tags. (Underscores in short tags are "
        "not be removed, which are very likely part of emoji tags.)",
        "https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon",
        "the Kohaku XL Epsilon card",
    ),
)

_ILLUSTRIOUS_REPORT = "https://arxiv.org/html/2409.19946v1"
_ILLUSTRIOUS = (
    Excerpt(
        "illustrious.order",
        "Tags were separated and reordered following a specific schema: person count "
        "||| character names ||| rating ||| general tags ||| artist ||| score range "
        "based rating ||| year modifier",
        _ILLUSTRIOUS_REPORT,
        "the Illustrious report on its training captions, section 3.1.2",
    ),
    Excerpt(
        "illustrious.captions",
        "Starting from Illustrious v2.0, Multi Level Captions has been introduced. … "
        "Multi-Captioning involves assigning multiple captions to a single image likes "
        "natural language and tags.",
        _ILLUSTRIOUS_REPORT,
        "the Illustrious report, appendix A.5",
    ),
)

_WAI_V17 = "https://civarchive.com/models/827184/wai-nsfw-illustrious-sdxl?modelVersionId=2883731"
_WAI_NOTE = "author WAI0731's v17 description, archived copy"
_WAI = (
    Excerpt(
        "wai.recipe",
        ",masterpiece,best quality,amazing quality,",
        _WAI_V17,
        "the v17 description's recommended positive prompt",
    ),
    Excerpt(
        "wai.too_many",
        "Please do not add too many quality and aesthetic-related tags, nor overly long "
        "negative prompts, as this will actually reduce image quality and make it more "
        "blurry.",
        _WAI_V17,
        _WAI_NOTE,
    ),
    Excerpt(
        "wai.ratings",
        "There are four safety rating tags: general, sensitive, nsfw,explicit",
        _WAI_V17,
        _WAI_NOTE,
    ),
)

_ANIMA_CARD = "https://huggingface.co/circlestone-labs/Anima"
_ANIMA = (
    Excerpt(
        "anima.order",
        "[quality/meta/year/safety tags] [1girl/1boy/1other etc] [character] [series] "
        "[artist] [general tags]",
        _ANIMA_CARD,
        "the card's tag order",
        order=True,
    ),
    Excerpt(
        "anima.order_within",
        "Within each tag section, the tags can be in arbitrary order.",
        _ANIMA_CARD,
    ),
    Excerpt(
        "anima.prefix",
        'Recommended positive prefix: "masterpiece, best quality, score_7, safe, "',
        _ANIMA_CARD,
    ),
    Excerpt(
        "anima.spelling",
        "Use lowercase for tags, and spaces instead of underscores. Score tags are the "
        "only tags that use underscores.",
        _ANIMA_CARD,
    ),
    Excerpt(
        "anima.artist",
        'Prefix artist with @. E.g. "@big chungus". You must put @ in front of the '
        "artist. The effect will be very weak if you don't.",
        _ANIMA_CARD,
    ),
    Excerpt(
        "anima.sentences",
        "If using pure natural langauge, more descriptive is better. Aim for at least 2 "
        "sentences. Extremely short prompts can give unexpected results.",
        _ANIMA_CARD,
    ),
    Excerpt(
        "anima.mix",
        "You can mix tags and natural language in arbitrary order.",
        _ANIMA_CARD,
    ),
    Excerpt(
        "anima.characters",
        "Name a character, then describe their basic appearance. … This is extra "
        "important when prompting for multiple characters. If you just list off "
        "character names with no description of appearance, the model can get confused.",
        _ANIMA_CARD,
    ),
)

# The family prefix recommends score_7, which Aesthetic's own advice rejects.
# Keep the remaining family guidance and the variant's explicit exception.
_ANIMA_AESTHETIC = tuple(excerpt for excerpt in _ANIMA if excerpt.id != "anima.prefix") + (
    Excerpt(
        "anima.aesthetic_quality",
        "You don't need to use quality tags in the positive at all",
        _ANIMA_CARD,
        "Aesthetic Version Prompting",
    ),
    Excerpt(
        "anima.aesthetic_scores",
        "not using score_* tags in both the positive and negative prompt",
        _ANIMA_CARD,
        "the author's recommendation under Aesthetic Version Prompting",
    ),
)

_NETA_BOOK = "https://neta.art/blog/neta_lumina_prompt_book/"
_NETA_NOTE = "inherited from the parent Neta Lumina's prompt book"
_NETAYUME = (
    Excerpt(
        "neta.formula",
        "Good prompts are often highly structured. Use what you need: character‑only or "
        "environment‑only prompts are fine.",
        _NETA_BOOK,
        _NETA_NOTE,
    ),
    Excerpt(
        "neta.sequence",
        "1. Character trigger words (e.g., 1girl, 1boy, 2girls, character name, etc.) "
        "2. Picture style prompt words 3. Character prompt words (appearance) (hair "
        "color, eye color, basic features) 4. Character costume prompt (specific costume "
        "description) 5. Character expression and action prompts (expression, posture, "
        "action) 6. Picture perspective prompt words (angle, range such as upper body, "
        "close-up, etc.) 7. Special effects prompts (lighting, special effects) 8. Scene "
        "atmosphere prompt (environment, atmosphere) 9. Quality tips (best quality)",
        _NETA_BOOK,
        _NETA_NOTE + "; its standard sequence for the tags",
        order=True,
    ),
    Excerpt(
        "neta.hybrid",
        "Neta Lumina excels at complex prose. Build skeleton with tags first, then "
        "flesh out details in natural language.",
        _NETA_BOOK,
        _NETA_NOTE,
    ),
    Excerpt(
        "neta.prose_order",
        "1. Composition aspect 2. Light and shadow processing 3. Characteristics and "
        "Clothing 4. Scene details 5. Artistic style",
        _NETA_BOOK,
        _NETA_NOTE + "; its order for the natural-language part",
        order=True,
    ),
    Excerpt(
        "neta.no_duplicates",
        "Ensure that Tags complement and do not duplicate natural language",
        _NETA_BOOK,
        _NETA_NOTE,
    ),
    Excerpt(
        "neta.specific",
        "Vague: beautiful girl; Specific: A graceful girl with flowing silver hair that "
        "catches the moonlight",
        _NETA_BOOK,
        _NETA_NOTE + "; its first best practice, concrete and vivid descriptions",
    ),
)

_NEWBIE_CARD = "https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1"
_NEWBIE_XML = (
    "https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/blob/"
    "4f7825a1a7e1ed5c1882a52e3c51d1119a43e298/XML%20Prompt.json"
)
_NEWBIE_XML_NOTE = "the authors' captioning instruction that made the training captions"
_NEWBIE = (
    Excerpt(
        "newbie.xml",
        "We reformatted the dataset text into an XML structured format for our "
        "experiments. Empirically, this improved attention binding and "
        "attribute/element disentanglement, and also led to faster convergence.",
        _NEWBIE_CARD,
    ),
    Excerpt(
        "newbie.inputs",
        "Besides that, It also supports natural language and tags inputs.",
        _NEWBIE_CARD,
    ),
    Excerpt(
        "newbie.multiple",
        "In multi character scenes, using XML structured prompt typically leads to more "
        "accurate image generation results.",
        _NEWBIE_CARD,
    ),
    Excerpt(
        "newbie.character_fields",
        "<n>, <gender>, <appearance>, <clothing>, <body_type>, <expression>, <action>, "
        "<interaction>, <position>",
        _NEWBIE_XML,
        "each <character_N> block's fields in the order of " + _NEWBIE_XML_NOTE,
    ),
    Excerpt(
        "newbie.anonymous_names",
        "Character names are temporarily unknown, please use placeholders like "
        "$character_1$, $character_2$, etc.",
        _NEWBIE_XML,
        _NEWBIE_XML_NOTE + "; these identifiers may remain for unnamed characters",
    ),
    Excerpt(
        "newbie.general_fields",
        "<count>, <artists>, <style>, <background>, <environment>, <perspective>, "
        "<atmosphere>, <lighting>, <quality>, <objects>, <other>",
        _NEWBIE_XML,
        "the <general_tags> fields in the order of " + _NEWBIE_XML_NOTE,
    ),
    Excerpt(
        "newbie.gender",
        "ALWAYS include gender tags (1girl, 1boy, etc.) for each character in their "
        "specific gender section",
        _NEWBIE_XML,
        _NEWBIE_XML_NOTE,
    ),
    Excerpt(
        "newbie.position",
        "Add detailed information about character positions in the image - this is "
        "EXTREMELY important",
        _NEWBIE_XML,
        _NEWBIE_XML_NOTE,
    ),
    Excerpt(
        "newbie.tag_format",
        "Use professional prompt word format for tags, not natural language",
        _NEWBIE_XML,
        _NEWBIE_XML_NOTE,
    ),
    Excerpt(
        "newbie.omit",
        "If any XML attribute is not applicable or not visible in the image, you can "
        "omit that tag entirely.",
        _NEWBIE_XML,
        _NEWBIE_XML_NOTE,
    ),
)

_KREA_DEV_README = "https://github.com/krea-ai/flux-krea"
# How BFL's own pages read their order, told to the critic on every FLUX.1
# checkpoint: its steps open with the image type, its template names camera
# settings, and it calls the template a starting structure. A 4B critic left
# to "subject first" called Krea's own "Moody night photography, a small
# store..." out of order and told the same prompt to put its style first.
_FLUX1_CAVEAT = (
    "BFL's own steps start with the image type and subject, so an opening that "
    "names the image type or medium ('Photo of', 'Moody night photography') still "
    "puts the subject first. Camera settings are part of BFL's template, not a fault."
)
_KREA_DEV_CAMERA = (
    "Camera settings - lens, framing, depth of field - are optional: BFL's guide adds "
    "them to a photograph where framing matters."
)
# BFL's launch post: the checkpoint "overcomes the oversaturated 'AI look'".
_KREA_DEV_POST = "https://bfl.ai/blog/flux-1-krea-dev"
_KREA2_README = "https://github.com/krea-ai/krea-2"
_KREA2_REPORT = "https://www.krea.ai/blog/krea-2-technical-report"
# What the technical report says - dense captions, a text-rendering reward,
# an encoder chosen for "stronger multilingual generalization" - in the prose
# both Krea 2 guides share.
_KREA2_REPORTED = tuple(
    (sentence, _KREA2_REPORT)
    for sentence in (
        "Krea's report says the training captions were dense.",
        "Text rendering was a reward in post-training.",
        "The Qwen3-VL encoder reads 512 tokens and was chosen for multilingual reach, "
        "though Krea publishes no language list.",
    )
)
_Z_IMAGE_CARD = "https://huggingface.co/Tongyi-MAI/Z-Image"
# The report names the encoder; each repo's tokenizer config sets the 2048
# its pipeline truncates at.
_ERNIE_REPORT = (
    "Baidu's report names Ministral-3 as the text encoder.",
    "https://arxiv.org/abs/2605.25347",
)
_ERNIE_TOKENS = "The tokenizer config caps a prompt at 2048 tokens."
_ILLUSTRIOUS_V01 = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0"
# The authors' v2.0 release post, read through its archive: Civitai refuses
# this region.
_ILLUSTRIOUS_RELEASE = "https://civarchive.com/models/1369089?modelVersionId=1546777"
_ILLUSTRIOUS_CARD_NEGATIVE = (
    "Their v0.1 card runs both of its examples with: "
    + guides.ILLUSTRIOUS_GUIDE["tag_helper"]["negative_prompt"] + "."
)
_ERNIE_EXPAND = (
    "Use Expand to run the repo's prompt enhancer on a short prompt; its result is put "
    "back in this box for review and becomes the exact prompt the image records."
)

# Keyed by the constant's name in guides.py; `test_every_guide_constant_is_grounded`
# holds the two in step. `unsourced` lists the prose's sentences that are this
# app talking - its menus, autocomplete, Expand, the sampler it runs, what it
# does not implement, a starting point or reading of its own - which the ?
# badge needs and no author wrote, and schnell's reference limit, which does
# not hold here.
# How the pipeline reads the prompt - a token limit, where guidance brings the
# negative prompt in, the system instruction it adds - stays citable: it is
# what runs, and `test_no_citable_sentence_misstates_a_token_limit` holds the
# limits to prompt_encoding. No rule entry cites an unsourced sentence.
GROUNDING: dict[str, Grounding] = {
    "NOOBAI_GUIDE": Grounding(
        (
            Excerpt(
                "noobai.prefix",
                "masterpiece, best quality, newest, absurdres, highres, safe,",
                "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0",
                note="the recommended positive prefix, not a complete scene",
            ),
            Excerpt(
                "noobai.order",
                "<1girl/1boy/1other/...>, <character>, <series>, <artists>, "
                "<special tags>, <general tags>, <other tags>",
                "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0",
                note="the separate caption template; the card also recommends a quality prefix",
                order=True,
            ),
        ),
        caveat=(
            "This is NoobAI-XL V-pred 1.0. The card gives a positive prefix and a "
            "separate caption template; do not invent a combined total tag order. "
            "Its quality, aesthetic and date tags are its own, not Illustrious's "
            "or Animagine's. The artist: prefix occurs in its Diffusers example; "
            "the card does not require it for every artist. Keep the 77-token "
            "CLIP limit; this app does not implement prompt chunking."
        ),
        unsourced=(
            "Examples includes the card's complete Diffusers prompt.",
            "Autocomplete checks Danbooru through the card's approximate 2024-10-23 "
            "cutoff; it does not check e621, and a match does not prove the model learned a tag.",
        ),
        sentence_urls=(
            (
                "Both SDXL CLIP tokenizers stop at 77 tokens, including special tokens.",
                "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/"
                "66aa55e3469c27c29a89813cd35dd95fb7485fa1/tokenizer/tokenizer_config.json",
            ),
        ),
    ),
    "ROUWEI_GUIDE": Grounding(
        (
            Excerpt(
                "rouwei.quality",
                "All except low quality in negative can be ommited.",
                "https://huggingface.co/Minthy/RouWei-0.8",
                note="the four quality tags; positive quality tags are optional",
            ),
            Excerpt(
                "rouwei.artist",
                "Used with by",
                "https://huggingface.co/Minthy/RouWei-0.8",
                note="the artist-name prefix, including its trailing space",
            ),
            Excerpt(
                "rouwei.artist_order",
                "or at least put them in the very end.",
                "https://huggingface.co/Minthy/RouWei-0.8",
                note=(
                    "artist tags: the author's fallback when separate CLIP chunks are "
                    "unavailable, as in this app"
                ),
                order=True,
            ),
        ),
        caveat=(
            "This is RouWei 0.8 epsilon, not its v-prediction release. Positive quality "
            "tags are optional. Do not invent a full tag order or import another "
            "checkpoint's quality, rating or date vocabulary. This app has one 77-token "
            "CLIP sequence: never recommend BREAK or separate chunks; the author's "
            "fallback is to put artist tags at the end, within the token limit."
        ),
        unsourced=(
            "Examples contains the author's scene excerpts without the separate style and quality chunk.",
            "This app does not implement separate chunks: BREAK is literal text, and both "
            "SDXL CLIP tokenizers stop at 77 tokens.",
            "Keep the complete prompt within that limit so the artists at its end are read.",
            "Autocomplete checks Danbooru through the author's end-of-April-2025 dataset "
            "cutoff; a match is advisory, not proof the checkpoint learned it.",
        ),
    ),
    "FLUX1_GUIDE": Grounding(_bfl_family(_BFL_FAMILY), caveat=_FLUX1_CAVEAT),
    "KREA_DEV_GUIDE": Grounding(
        _bfl_family(_BFL_FAMILY + "; Krea published no prompting rules of its own"),
        caveat=(
            "Krea published no prompting rules for FLUX.1 Krea [dev] - its README gives "
            "settings only. The prompting in the G sentences and the A excerpts is Black "
            "Forest Labs' guide for the whole FLUX family, which this checkpoint belongs to. "
            + _FLUX1_CAVEAT
        ),
        # The settings sentence is Krea's README's, the camera sentence BFL's
        # building page, the 'AI look' BFL's launch post; the rest links the
        # page of the rule entries it carries. (Two sentences with no author
        # behind them - "photo language", "give it room" - were taken out of
        # the prose after review, not hidden here.)
        sentence_urls=(
            (
                "This checkpoint is tuned for photographs without the oversaturated "
                "'AI look'.",
                _KREA_DEV_POST,
            ),
            (_KREA_DEV_CAMERA, _BFL_BUILDING),
            ("Krea recommends 28-32 steps and CFG 3.5-5.", _KREA_DEV_README),
        ),
    ),
    "SCHNELL_GUIDE": Grounding(
        _bfl_family(_BFL_FAMILY),
        caveat=_FLUX1_CAVEAT,
        # BFL's reference limit, which this app does not apply: a critic
        # citing it asked for 180 words where T5 reads 512.
        unsourced=(
            "BFL's reference code gives schnell a shorter memory, 256 tokens (about 180 "
            "words), which this app does not apply.",
        ),
    ),
    "KLEIN_GUIDE": Grounding(
        _KLEIN,
        caveat=(
            "BFL's FLUX.2 prompting guide was written for FLUX.2 [pro] and [max]; it "
            "applies to klein by family. klein takes no negative prompt at all."
        ),
    ),
    # The settings are the README's ("run with 8 steps and CFG disabled",
    # "TRAIN on Raw and RUN on Turbo").
    "KREA2_TURBO_GUIDE": Grounding(
        _KREA2,
        sentence_urls=_KREA2_REPORTED + (
            (
                "Turbo runs guidance-free at CFG 0, where the negative prompt is unused; "
                "any CFG above 0 brings it in.",
                _KREA2_README,
            ),
        ),
    ),
    "KREA2_RAW_GUIDE": Grounding(
        _KREA2,
        sentence_urls=_KREA2_REPORTED + (
            (
                "Raw is the undistilled checkpoint Krea does not recommend for inference - "
                "train on Raw, run on Turbo - so expect variety over polish.",
                _KREA2_README,
            ),
            (
                "It wants about 52 steps at CFG 3.5, and the negative prompt applies at "
                "any CFG above 0.",
                _KREA2_README,
            ),
        ),
    ),
    "QWEN_IMAGE_GUIDE": Grounding(_QWEN),
    "Z_IMAGE_TURBO_GUIDE": Grounding(_z_image(_Z_IMAGE_TURBO_NOTE)),
    "Z_IMAGE_GUIDE": Grounding(
        _z_image(_Z_IMAGE_BASE_NOTE),
        sentence_urls=(
            (
                "The undistilled Z-Image wants CFG 3-5 over 28-50 steps; its card calls a "
                "negative prompt optional, but powerful for removing unwanted content.",
                _Z_IMAGE_CARD,
            ),
        ),
    ),
    "Z_ANIME_GUIDE": Grounding(
        _Z_ANIME,
        caveat=(
            "The card's advice to put important details early is written for its "
            "4-step variant, not this Base model, and is not a source here."
        ),
    ),
    "ERNIE_TURBO_GUIDE": Grounding(
        _ERNIE,
        unsourced=(_ERNIE_EXPAND,),
        sentence_urls=(
            _ERNIE_REPORT,
            (
                _ERNIE_TOKENS,
                "https://huggingface.co/baidu/ERNIE-Image-Turbo/blob/main/tokenizer/"
                "tokenizer_config.json",
            ),
        ),
    ),
    # The undistilled model's settings are its own card's.
    "ERNIE_GUIDE": Grounding(
        _ERNIE,
        unsourced=(_ERNIE_EXPAND,),
        sentence_urls=(
            _ERNIE_REPORT,
            (
                _ERNIE_TOKENS,
                "https://huggingface.co/baidu/ERNIE-Image/blob/main/tokenizer/"
                "tokenizer_config.json",
            ),
            (
                "The undistilled ERNIE-Image samples at CFG 4 over 50 steps, where the "
                "negative prompt applies.",
                "https://huggingface.co/baidu/ERNIE-Image",
            ),
        ),
    ),
    "ANIMAGINE_GUIDE": Grounding(
        _ANIMAGINE,
        unsourced=(
            "Autocomplete and the chips check Danbooru's live tag index up to this "
            "checkpoint's 2025-01-07 cutoff.",
            "Unknown is advisory, not blocked.",
            # The app naming its own steps for the card's sequence, whose
            # template ends "everything else in any order".
            "The Tags menu and Check name those steps Person count, Character, Series, "
            "Rating, Details and Quality; year, artist and meta tags go in Details.",
        ),
    ),
    "KOHAKU_GUIDE": Grounding(
        _KOHAKU,
        # The last is the app's advice: the card's context is 300 tokens.
        unsourced=(
            "The Tags menu carries person count and the author's quality, rating and date tags.",
            "Autocomplete checks Danbooru through the 2024-07-10 dataset cutoff; unknown "
            "tags remain advisory.",
            "Keep prompts concise and put essential details first.",
        ),
    ),
    "ILLUSTRIOUS_GUIDE": Grounding(
        _ILLUSTRIOUS,
        # The style advice is the app's reading of "no default style" - the
        # authors' own examples name no artist, and no rule asks for one; the
        # tail is the app's starting point, "not an official fixed suffix" in
        # its own words.
        unsourced=(
            "Give it a style: a Danbooru artist tag or style tags.",
            "Autocomplete and the chips check Danbooru's tag index through the dataset's "
            "August 2024 cutoff and label artist tags as artist.",
            "masterpiece, newest is a practical starting tail, not an official fixed suffix.",
            "The Tags menu fills the long one.",
        ),
        # The report is the rules' page; what the v0.1 card and the v2.0
        # release post say links to them.
        sentence_urls=(
            (
                "Illustrious XL v2.0 is an untuned base: its authors say it has no default "
                "style and was not trained with an aesthetic set, so a bare prompt renders "
                "the dataset's average look.",
                _ILLUSTRIOUS_RELEASE,
            ),
            (
                "Tag lists and natural-language descriptions both work; their "
                "natural-language example ends its prose with tags (persona 3, masterpiece).",
                _ILLUSTRIOUS_RELEASE,
            ),
            (_ILLUSTRIOUS_CARD_NEGATIVE, _ILLUSTRIOUS_V01),
            ("Their v2.0 release example uses worst quality, bad quality.", _ILLUSTRIOUS_RELEASE),
        ),
    ),
    "WAI_GUIDE": Grounding(
        _WAI,
        caveat="The author documents no tag order for WAI v17 beyond the quality recipe.",
        unsourced=(
            "Examples contains scene excerpts from the author's v17 gallery.",
            "Autocomplete checks Danbooru's live index; WAI publishes no vocabulary cutoff, "
            "so a recognized tag is advisory rather than evidence that v17 learned it.",
            "The card's weighted-prompt syntax is not implemented by this text-to-image "
            "pipeline.",
        ),
    ),
    "ANIMA_TURBO_GUIDE": Grounding(
        _ANIMA,
        caveat=(
            "The Anima card covers the whole family; this app runs Anima Turbo v1.1, "
            "where the recommended CFG 1 runs no negative prompt."
        ),
    ),
    "ANIMA_AESTHETIC_GUIDE": Grounding(
        _ANIMA_AESTHETIC,
        caveat=(
            "The Anima card covers the whole family; this app runs Anima Aesthetic "
            "v1.1. Its Aesthetic Version Prompting advice takes precedence over "
            "the family's quality-tag recipe: positive quality tags are optional, "
            "masterpiece and best quality may stay, and score_* tags are discouraged "
            "in both prompts. Turbo's distillation and CFG 1 advice do not apply."
        ),
    ),
    "NETAYUME_GUIDE": Grounding(
        _NETAYUME,
        caveat=(
            "The A excerpts come from the parent Neta Lumina's prompt book; NetaYume's "
            "own card gives only system prompts. This app adds the system instruction "
            "itself, so never suggest adding a system prefix or <Prompt Start>."
        ),
        # The system instruction sentence stays: it is how the pipeline reads
        # the prompt, and the system-prefix rule cites it.
        unsourced=(
            "The Examples menu uses the image prompt from the author's v2 workflow.",
            "This app uses standard FlowMatch Euler; the author's ComfyUI sampler "
            "recommendations are different setups.",
        ),
    ),
    "NEWBIE_GUIDE": Grounding(
        _NEWBIE,
        caveat=(
            "The XML example's $character_N$ identifiers are valid anonymous "
            "character labels, used literally in the author's example. They do "
            "not need to be replaced with names."
        ),
        unsourced=("The Examples menu includes the author's two-character XML prompt.",),
        # The app's pipeline truncates where the author's own training code's
        # `encode_prompt` does.
        sentence_urls=(
            (
                "Keep $character_1$ and $character_2$ for unnamed characters, or "
                "replace them with names.",
                _NEWBIE_XML,
            ),
            (
                "Gemma reads up to 512 tokens and Jina CLIP up to 2048 in this app; keep "
                "the main description within Gemma's shorter limit.",
                "https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/"
                "709e9a482fb2b861ad6aa50dcafb0198cad66c1d/finetune.py",
            ),
        ),
    ),
    # Its prose is the app's general advice, cited as that (`build_context`
    # names no authors when `url` is None); its opening is a label.
    "FALLBACK_GUIDE": Grounding(
        caveat=(
            "This checkpoint is not in the catalogue: no author sources are known for "
            "it, so only the general guide sentences apply."
        ),
        unsourced=("Unrecognised checkpoint.",),
    ),
}


def guide_constants() -> dict[str, Mapping[str, Any]]:
    """guides.py's guide dicts by constant name."""
    return {
        name: value
        for name, value in vars(guides).items()
        if name.endswith("_GUIDE") and isinstance(value, dict) and "prompt" in value
    }


def guide_name(guide: Mapping[str, Any]) -> str:
    """The constant a profile's guide is, by identity - never by equal content,
    since two guides may share their prose and differ in their sources."""
    for name, value in guide_constants().items():
        if value is guide:
            return name
    raise KeyError("the guide is not one of guides.py's constants")


def sources_for(model_id: str) -> Sources:
    """Everything the critic may cite for this checkpoint."""
    guide = _profile_for(model_id)["guide"]
    name = guide_name(guide)
    grounding = GROUNDING[name]
    url = guide.get("analysis", {}).get("url")
    own = dict(grounding.sentence_urls)
    prose = guide["prompt"]
    cited = _cited_pages(guide)
    return Sources(
        model_id=model_id,
        guide_name=name,
        guide=guide,
        sentences=tuple(
            GuideSentence(sentence, own.get(sentence) or _page(cited, start, end) or url)
            for start, end in _sentence_spans(prose)
            if (sentence := prose[start:end]) not in grounding.unsourced
        ),
        url=url,
        excerpts=grounding.excerpts,
        caveat=grounding.caveat,
    )


def author_orders(model_id: str) -> list[dict[str, str | None]]:
    """The orders the authors' own texts give, on a checkpoint whose guide
    states none this app checks: the `order` excerpts, as text, url and note.

    Empty where a documented order exists (`prompt_order.documented_order`):
    that order is the one shown. Check's Order section
    (`PromptAnalysis.author_orders`) and the Tags menu
    (`PromptTagHelper.author_orders`) both read this, so the two never
    disagree about what the authors give.
    """
    from .prompt_order import documented_order  # lazy, as model_defaults imports this module lazily

    if documented_order(model_id) is not None:
        return []
    guide = _profile_for(model_id)["guide"]
    return [
        {"text": excerpt.text, "url": excerpt.url, "note": excerpt.note}
        for excerpt in GROUNDING[guide_name(guide)].excerpts
        if excerpt.order
    ]
