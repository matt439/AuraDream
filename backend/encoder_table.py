"""What each checkpoint's pipeline feeds its text encoders: the table.

Pure data and lookup - no tokenizer, no threads, no I/O. `prompt_encoding`
measures a prompt against these records with the checkpoint's own tokenizers.

**Every template string is the pipeline's.** Where diffusers keeps one on the
pipeline class it is copied here and held equal to the installed source by
`tests/test_prompt_encoding.py`, so an upgrade that changes it fails a test
rather than the analyser's answers.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .backends import anima_loading, loading, netayume_loading, newbie_loading, single_file

Field = Literal["prompt", "negative_prompt"]

# --- the pipelines' own templates and lengths ---------------------------------
# Copied from the installed pipelines; tests/test_prompt_encoding.py reads the
# source files and fails when one of these stops matching.

# Krea2Pipeline and QwenImagePipeline wrap the prompt in the same system turn.
QWEN_SYSTEM_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
QWEN_ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
# Krea 2 truncates prefix + prompt to `max_sequence_length + start_idx -
# num_suffix_tokens` and appends the suffix after the padding, so the suffix
# is never the part that is cut.
KREA2_PREFIX_TOKENS = 34
KREA2_SUFFIX_TOKENS = 5
KREA2_SEQUENCE = 512
# Qwen-Image truncates the whole templated string to `tokenizer_max_length +
# start_idx`, drops the first `start_idx` hidden states and keeps the next
# `max_sequence_length` - the suffix sits inside that window and goes first.
QWEN_IMAGE_TOKENIZER_MAX = 1024
QWEN_IMAGE_PREFIX_TOKENS = 34
QWEN_IMAGE_SEQUENCE = 512
# Z-Image and klein truncate the rendered chat template as a whole.
Z_IMAGE_SEQUENCE = 512
KLEIN_SEQUENCE = 512
FLUX_T5_SEQUENCE = 512
# Lumina 2 joins its system prompt to the prompt, never to the negative one.
LUMINA_SEPARATOR = " <Prompt Start> "
LUMINA_SEQUENCE = 256
# backend/backends/anima_pipeline.py hands this to the modular encoders, which
# truncate the Qwen3 tokens and the T5 ids to it independently.
ANIMA_SEQUENCE = 512

_Z_ANIME_ID = "SeeSee21/Z-Anime"
_PIPELINES = "diffusers/pipelines/"


@dataclass(frozen=True)
class TokenizerSource:
    """Where one tokenizer lives: a Hub repo or local directory, the subfolder
    inside it, and the revision where the loader pins one."""

    repo: str
    subfolder: str
    revision: str | None = None
    # A catalogue id whose converted tree, once a load has built it, holds a
    # copy of this tokenizer under `tree_subfolder` - read first, so a
    # checkpoint that has run here once needs no network for its count.
    tree: str | None = None
    tree_subfolder: str | None = None

    @property
    def label(self) -> str:
        return f"{self.repo}/{self.subfolder}"


@dataclass(frozen=True)
class EncoderSpec:
    """One text encoder as its pipeline feeds it."""

    name: str
    source: TokenizerSource
    # The pipeline's truncation length over the whole string it builds,
    # special tokens included. None is the tokenizer's own `model_max_length`,
    # which is what the CLIP and ERNIE pipelines pass.
    max_length: int | None
    # The pipeline behaviour the budget rests on, and the file it is in.
    statement: str
    file: str
    # Text the pipeline puts around the prompt inside the truncated string.
    prefix: str = ""
    suffix: str = ""
    # Render the tokenizer's own chat template instead, with this switch.
    chat_thinking: bool | None = None
    # Output positions kept after truncation, as (dropped prefix, kept count):
    # Qwen-Image's `[drop_idx:]` and `[:max_sequence_length]`.
    window: tuple[int, int] | None = None
    # The negative prompt: the same template, bare (Lumina 2's system prompt
    # joins the prompt only), or never encoded (klein takes none).
    negative: Literal["same", "bare", "none"] = "same"
    # What the reading is, beside its budget. A bare negative has none of
    # the template a role like NetaYume's describes, so it takes no role.
    role: str | None = None
    # FLUX.1's CLIP-L feeds one pooled vector while T5 carries the sequence,
    # so its cut is advice rather than lost text.
    pooled_only: bool = False

    @property
    def cite(self) -> str:
        return f"{self.statement} ({self.file})"

    def role_for(self, field: Field) -> str | None:
        return None if field == "negative_prompt" and self.negative == "bare" else self.role


# --- the table ----------------------------------------------------------------


def _own(model_id: str, subfolder: str) -> TokenizerSource:
    # An ordinary Diffusers repo, lite-infer's re-uploads included, carries
    # its tokenizers in its own subfolders.
    return TokenizerSource(model_id, subfolder)


def _pinned(
    pin: dict[str, str], subfolder: str, tree: str, tree_subfolder: str | None = None
) -> TokenizerSource:
    # A loader's pinned source, whose converted tree keeps a copy once built.
    return TokenizerSource(
        pin["repo_id"], subfolder, pin["revision"], tree, tree_subfolder or subfolder
    )


def _clip_pair(tokenizer: TokenizerSource, tokenizer_2: TokenizerSource) -> list[EncoderSpec]:
    statement = (
        "StableDiffusionXLPipeline.encode_prompt truncates each CLIP tokenizer to its "
        "77 tokens, the negative prompt without a warning"
    )
    file = _PIPELINES + "stable_diffusion_xl/pipeline_stable_diffusion_xl.py"
    return [
        EncoderSpec("CLIP-L", tokenizer, None, statement, file),
        EncoderSpec("CLIP-G", tokenizer_2, None, statement, file),
    ]


def _sdxl(model_id: str) -> list[EncoderSpec]:
    return _clip_pair(_own(model_id, "tokenizer"), _own(model_id, "tokenizer_2"))


def _noobai(model_id: str) -> list[EncoderSpec]:
    # The ordinary Diffusers tree and its tokenizers share the loader's pin.
    return _clip_pair(
        TokenizerSource(loading.NOOB_MODEL_ID, "tokenizer", loading.NOOB_REVISION),
        TokenizerSource(loading.NOOB_MODEL_ID, "tokenizer_2", loading.NOOB_REVISION),
    )


def _sdxl_single_file(model_id: str) -> list[EncoderSpec]:
    # The single-file checkpoints' conversion takes its
    # tokenizers from the pinned SDXL base descriptors.
    config = single_file.SDXL_CONFIG
    return _clip_pair(
        _pinned(config, "tokenizer", model_id), _pinned(config, "tokenizer_2", model_id)
    )


def _flux1(model_id: str) -> list[EncoderSpec]:
    file = _PIPELINES + "flux/pipeline_flux.py"
    return [
        EncoderSpec(
            "CLIP-L",
            _own(model_id, "tokenizer"),
            None,
            "FluxPipeline truncates CLIP to its tokenizer's 77 tokens and keeps only the "
            "pooled output",
            file,
            role="one pooled vector, not the token sequence - T5-XXL carries the words",
            pooled_only=True,
        ),
        EncoderSpec(
            "T5-XXL",
            _own(model_id, "tokenizer_2"),
            FLUX_T5_SEQUENCE,
            f"FluxPipeline truncates T5 at max_sequence_length={FLUX_T5_SEQUENCE}, </s> included",
            file,
        ),
    ]


def _krea2(model_id: str) -> list[EncoderSpec]:
    length = KREA2_SEQUENCE + KREA2_PREFIX_TOKENS - KREA2_SUFFIX_TOKENS
    return [
        EncoderSpec(
            "Qwen3-VL-4B",
            _own(model_id, "tokenizer"),
            length,
            f"Krea2Pipeline truncates its system prefix and the prompt to {length} tokens, "
            "then appends the assistant suffix",
            _PIPELINES + "krea2/pipeline_krea2.py",
            prefix=QWEN_SYSTEM_PREFIX,
        ),
    ]


def _qwen_image(model_id: str) -> list[EncoderSpec]:
    return [
        EncoderSpec(
            "Qwen2.5-VL-7B",
            _own(model_id, "tokenizer"),
            QWEN_IMAGE_TOKENIZER_MAX + QWEN_IMAGE_PREFIX_TOKENS,
            f"QwenImagePipeline wraps the prompt in its chat template, drops the "
            f"{QWEN_IMAGE_PREFIX_TOKENS} system tokens and keeps {QWEN_IMAGE_SEQUENCE}",
            _PIPELINES + "qwenimage/pipeline_qwenimage.py",
            prefix=QWEN_SYSTEM_PREFIX,
            suffix=QWEN_ASSISTANT_SUFFIX,
            window=(QWEN_IMAGE_PREFIX_TOKENS, QWEN_IMAGE_SEQUENCE),
        ),
    ]


def _z_image_encoder(source: TokenizerSource) -> EncoderSpec:
    return EncoderSpec(
        "Qwen3-4B",
        source,
        Z_IMAGE_SEQUENCE,
        "ZImagePipeline renders the tokenizer's chat template (enable_thinking=True) and "
        f"truncates the whole string to {Z_IMAGE_SEQUENCE} tokens",
        _PIPELINES + "z_image/pipeline_z_image.py",
        chat_thinking=True,
    )


def _z_image(model_id: str) -> list[EncoderSpec]:
    return [_z_image_encoder(_own(model_id, "tokenizer"))]


def _z_anime(model_id: str) -> list[EncoderSpec]:
    # The nested tree's own tokenizer, at the revision the loader pins; the
    # encoder weights come from Z-Image, the tokenizer does not.
    revision, folder = loading.NESTED_MODELS[_Z_ANIME_ID]
    return [_z_image_encoder(TokenizerSource(_Z_ANIME_ID, f"{folder}/tokenizer", revision))]


def _klein(model_id: str) -> list[EncoderSpec]:
    return [
        EncoderSpec(
            "Qwen3-8B" if "9b" in _key(model_id) else "Qwen3-4B",
            _own(model_id, "tokenizer"),
            KLEIN_SEQUENCE,
            "Flux2KleinPipeline renders the tokenizer's chat template (enable_thinking=False) "
            f"and truncates the whole string to {KLEIN_SEQUENCE} tokens",
            _PIPELINES + "flux2/pipeline_flux2_klein.py",
            chat_thinking=False,
            negative="none",
        ),
    ]


def _ernie(model_id: str) -> list[EncoderSpec]:
    return [
        EncoderSpec(
            "Ministral-3B",
            _own(model_id, "tokenizer"),
            None,
            "ErnieImagePipeline.encode_prompt truncates at the tokenizer's model_max_length, "
            "<s> included",
            _PIPELINES + "ernie_image/pipeline_ernie_image.py",
        ),
    ]


def _netayume(model_id: str) -> list[EncoderSpec]:
    return [
        EncoderSpec(
            "Gemma 2 2B",
            _pinned(netayume_loading.SHARED, "tokenizer", netayume_loading.MODEL_ID),
            LUMINA_SEQUENCE,
            f"Lumina2Pipeline puts the system prompt and '{LUMINA_SEPARATOR.strip()}' before "
            f"the prompt, not the negative prompt, and truncates to {LUMINA_SEQUENCE} tokens "
            "with <bos>",
            _PIPELINES + "lumina2/pipeline_lumina2.py",
            prefix=netayume_loading.SYSTEM_PROMPT + LUMINA_SEPARATOR,
            negative="bare",
            role=f"shares its {LUMINA_SEQUENCE} tokens with the anime system instruction",
        ),
    ]


def _newbie(model_id: str) -> list[EncoderSpec]:
    source, recipe = newbie_loading.SOURCE, newbie_loading.RECIPE
    gemma, jina = recipe["gemma_max_length"], recipe["jina_max_length"]
    file = "backend/backends/newbie_pipeline.py"
    return [
        EncoderSpec(
            "Gemma 3 4B",
            _pinned(source, "text_encoder", newbie_loading.MODEL_ID, "tokenizer"),
            gemma,
            f"NewbiePipeline truncates Gemma 3 to {gemma} tokens, <bos> included",
            file,
        ),
        EncoderSpec(
            "Jina CLIP v2",
            _pinned(source, "clip_model", newbie_loading.MODEL_ID, "tokenizer_2"),
            jina,
            f"NewbiePipeline truncates Jina to {jina} tokens, <s> and </s> included, and pools "
            "their masked mean",
            file,
            role="one pooled vector: the mean over what it reads",
        ),
    ]


def _anima(model_id: str) -> list[EncoderSpec]:
    shared = anima_loading.SHARED
    file = "diffusers/modular_pipelines/anima/encoders.py"
    return [
        EncoderSpec(
            "Qwen3-0.6B",
            _pinned(shared, "tokenizer", model_id),
            ANIMA_SEQUENCE,
            f"Anima's text encoder step truncates the Qwen3 tokens to {ANIMA_SEQUENCE}, the "
            "length backend/backends/anima_pipeline.py passes",
            file,
        ),
        EncoderSpec(
            "T5 ids",
            _pinned(shared, "t5_tokenizer", model_id),
            ANIMA_SEQUENCE,
            f"the same step truncates the T5 token ids to {ANIMA_SEQUENCE}, </s> included",
            file,
            role="token ids only: they index the conditioner's learned queries; no T5 encoder runs",
        ),
    ]


def _key(model_id: str) -> str:
    # model_defaults' normalisation, so an id matches here where it matches there.
    return model_id.lower().replace("_", "-")


# Pinned releases first, by exact id: their tokenizers come from the pinned
# sources the loaders convert from, not from the public id. Never a broad
# "anima" token, which Animagine also contains.
_EXACT: dict[str, Callable[[str], list[EncoderSpec]]] = {
    _key(loading.NOOB_MODEL_ID): _noobai,
    _key(single_file.ROUWEI_ID): _sdxl_single_file,
    _key(single_file.ILLUSTRIOUS_ID): _sdxl_single_file,
    _key(single_file.WAI_ID): _sdxl_single_file,
    _key(_Z_ANIME_ID): _z_anime,
    _key(anima_loading.MODEL_ID): _anima,
    _key(anima_loading.AESTHETIC_MODEL_ID): _anima,
    _key(anima_loading.PREVIEW_MODEL_ID): _anima,
    _key(netayume_loading.MODEL_ID): _netayume,
    _key(newbie_loading.MODEL_ID): _newbie,
}
# Then first-match on a token, as in model_defaults._PROFILES. The tokens are
# narrower than the profiles': "flux.1" rather than "flux", so FLUX.2 klein
# never reaches FLUX.1's CLIP and T5, and "krea-2", which FLUX.1-Krea-dev
# does not contain.
_FAMILIES: list[tuple[str, Callable[[str], list[EncoderSpec]]]] = [
    ("krea-2", _krea2),
    ("klein", _klein),
    ("flux.1", _flux1),
    ("qwen-image", _qwen_image),
    ("z-image", _z_image),
    ("ernie-image", _ernie),
    ("animagine-xl", _sdxl),
    ("kohaku-xl", _sdxl),
]


def encoders_for(model_id: str) -> list[EncoderSpec] | None:
    """The text encoders this checkpoint's pipeline feeds, or None when the
    table does not know it - an unknown budget is reported as unchecked."""
    key = _key(model_id)
    build = _EXACT.get(key)
    if build is None:
        build = next((family for token, family in _FAMILIES if token in key), None)
    return None if build is None else build(model_id)
