"""FLUX 3 [Klein]'s prompt upsampler, as an explicit Expand.

Black Forest Labs' `flux-3-image-klein-pup` is a Qwen3.5-9B fine-tune (9.4B
parameters, 18.8 GB in bf16, Apache-2.0) that expands a short request into
the full scene caption FLUX 3 [Klein] is trained on. The cards ask for it
before every image; diffusers' modular blocks run it inside the image call.
This app never does that: the prompt the row records is the prompt the model
saw. Expand runs it once, on request, and writes its caption into the prompt
box for review - ERNIE's rewriter's seam (`DiffusersBackend.expand_prompt`).

What is followed, and from where:

- The repository is pinned (`SOURCE`), configs first and refused before the
  weights when they name another model (`preflight`).
- The text-to-image system prompt is diffusers' own copy
  (`diffusers.modular_pipelines.flux3.prompt_upsampler.SYSTEM_PROMPTS["t2i"]`,
  which says it is the `system_prompts.json` shipped beside the weights;
  identical at this pin, checked 2026-11-09). Only `t2i`: Expand has no
  reference images or layout boxes to give it.
- The messages are built as diffusers' `upsample_prompt` builds them (content
  as lists of parts, `enable_thinking=False`, at most 3584 new tokens).
- The caption is stopped as soon as it runs past the app's 4000-character
  prompt limit (`PROMPT_LIMIT`, `GenerateRequest.prompt`'s): diffusers' 3584
  tokens are 10,000 characters and more, which the box could never take, so
  an over-long caption ends there and is refused in words
  (`ExpansionTooLong`), never written whole and thrown away after. The
  encoder's own window (8192 tokens) is wider than the box; the box decides.
- The decoding is the model card's: "greedy, `top_p` 0.95, `top_k` 20,
  presence penalty 1.5 for text-to-image" - BFL's own `flux3/pup.py` passes
  the same to vLLM (`temperature` 0, `presence_penalty` 1.5 for t2i).
  diffusers' step decodes greedily with no presence penalty, so the two
  differ; this follows the card. Under greedy decoding `top_p` and `top_k`
  never change the chosen token, so only the penalty is applied
  (`PresencePenalty`, vLLM's meaning: once per token already generated).
- The weights are quantized to one byte per weight on load, through the same
  `PipelineQuantizationConfig(quant_mapping=...)` rule as the image pipeline
  (`loading.quant_mapping_config`): the server's fp8 or int8, else int8. In
  bf16 it could not fit a 16 GB card at all.
- It joins the image pipeline's model offload as its first component
  (`Flux3KleinPipeline.model_cpu_offload_seq`), as ERNIE's rewriter does, so
  it is on the card only while it writes - never beside the transformer.
  Without offload every component stays on the card, so it is refused there
  (`refusal`).

None of its fit, peak memory, time or output is measured on the card.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

# Resolve the app's remote-kernel policy even for direct imports.
from .. import config as _config  # noqa: F401
from .base import CancelFn, Cancelled, ExpansionTooLong
from .single_file import HubRepo

if TYPE_CHECKING:
    from ..config import Settings

log = logging.getLogger(__name__)

MODEL_ID = "black-forest-labs/flux-3-image-klein-pup"
# Pinned on 2026-11-09 from the Hub's API (the repository's head then).
SOURCE: HubRepo = {"repo_id": MODEL_ID, "revision": "a57899549f8f6f64c6c4d88db30827d496ff76d0"}
WEIGHTS = "model.safetensors"
CONFIG_PATTERNS = ["*.json", "chat_template.jinja"]
WEIGHT_PATTERNS = [*CONFIG_PATTERNS, WEIGHTS]
# What a first Expand downloads at the pin, from the tree API: the weights
# (18,819,722,392 bytes) and the tokenizer, configs and chat template beside
# them - never the README.
BYTES = 18_839_738_489
EXPECTED_CLASS = "Qwen3_5ForConditionalGeneration"
# The names the page says, and the pipeline's components it registers as
# (diffusers' own `ComponentSpec` names for the step).
NAME = "FLUX 3's prompt upsampler"
COMPONENT = "prompt_upsampler"
PROCESSOR = "prompt_upsampler_processor"
MODE = "t2i"
# The card's decoding for text-to-image (see the module docstring).
PRESENCE_PENALTY = 1.5
# The one-byte scheme when the image model is not one of them.
FALLBACK_QUANT = "int8"
# The longest prompt the app takes (`GenerateRequest.prompt`'s max_length):
# a caption past it is stopped there and refused (`too_long`).
PROMPT_LIMIT = 4000


def system_prompt() -> str:
    """The t2i system prompt, diffusers' copy of the shipped one."""
    from diffusers.modular_pipelines.flux3.prompt_upsampler import SYSTEM_PROMPTS

    return SYSTEM_PROMPTS[MODE]


def max_new_tokens() -> int:
    from diffusers.modular_pipelines.flux3.prompt_upsampler import MAX_NEW_TOKENS

    return MAX_NEW_TOKENS


def too_long() -> str:
    """The refusal for a caption past `PROMPT_LIMIT`, in the page's words."""
    return (
        f"{NAME} wrote past {PROMPT_LIMIT:,} characters, the most a prompt takes here, and was "
        "stopped there. Its caption grows with the scene asked for: ask for fewer subjects, or "
        "less quoted text, and Expand again, or write the caption yourself"
    )


def quantization(settings: Settings) -> str:
    """The server's one-byte scheme, or int8 where it runs another."""
    from .loading import QUANT_BACKENDS

    return settings.quantization if settings.quantization in QUANT_BACKENDS else FALLBACK_QUANT


def refusal(settings: Settings) -> str | None:
    """Why it cannot run under these settings, in the page's words, or None.

    Without offload every component of the image pipeline stays on the card,
    so a 9.4 GB upsampler would sit beside the transformer. Answered from
    the settings alone, before anything loads or downloads.
    """
    if settings.offload == "none":
        return (
            "FLUX 3's prompt upsampler runs only with model or sequential offload: with "
            "offload none it would stay on the card beside the image model. Choose Model "
            "offload in Models, then Expand"
        )
    return None


def cached() -> bool:
    """Whether the pinned weights are on disk already. File checks only - no
    network - so the status poll can ask it every few seconds."""
    from huggingface_hub import try_to_load_from_cache

    try:
        found = try_to_load_from_cache(MODEL_ID, WEIGHTS, revision=SOURCE["revision"])
    except (OSError, ValueError):
        return False
    return isinstance(found, str) and os.path.isfile(found)


def preflight(snapshot: str) -> None:
    """Refuse a snapshot whose config names another model, before weights."""
    import json

    try:
        with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{MODEL_ID}'s pinned config could not be read; no weights were loaded.") from exc
    if not isinstance(config, dict) or config.get("architectures") != [EXPECTED_CLASS]:
        raise RuntimeError(f"{MODEL_ID}'s config does not name {EXPECTED_CLASS}; no weights were loaded.")


def resolve(*, weights: bool) -> str:
    """The pinned snapshot: its configs (checked), then its weights."""
    from huggingface_hub import snapshot_download

    patterns = WEIGHT_PATTERNS if weights else CONFIG_PATTERNS
    log.info("resolving %s (%s)", MODEL_ID, "weights" if weights else "configs")
    snapshot = str(snapshot_download(SOURCE["repo_id"], revision=SOURCE["revision"], allow_patterns=patterns))
    preflight(snapshot)
    return snapshot


def load(settings: Settings, should_cancel: CancelFn | None = None) -> dict[str, Any]:
    """The upsampler and its processor, quantized, as pipeline components.

    Loaded on the CPU; the pipeline's offload moves it to the card for the
    call alone. Stops between the download and the load when asked.
    """
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from . import loading

    def check() -> None:
        if should_cancel is not None and should_cancel():
            raise Cancelled("prompt expansion stopped")

    resolve(weights=False)
    check()
    snapshot = resolve(weights=True)
    check()
    quant = loading.quant_mapping_config(quantization(settings), {COMPONENT: "transformers"})
    mapping = dict(getattr(quant, "quant_mapping", None) or {})
    log.info("loading %s at %s", NAME, SOURCE["revision"][:8])
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.bfloat16, quantization_config=mapping.get(COMPONENT), local_files_only=True,
    ).eval()
    # A quantizing load leaves its unquantized tensors (norms, embeddings)
    # mapped from the 18.8 GB file; copy them out, as every quantized
    # transformers component is (`loading.checkpoint_files_released`).
    loading.own_tensors(model)
    return {COMPONENT: model, PROCESSOR: processor}


def messages(prompt: str) -> list[dict[str, Any]]:
    """The chat, as diffusers' `upsample_prompt` builds a t2i one: content as
    lists of parts, the shipped system prompt, then the request."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def _processors(
    prompt_length: int, should_cancel: CancelFn | None, decode: Any = None,
) -> tuple[Any, Any]:
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    class PresencePenalty(LogitsProcessor):
        """vLLM's presence penalty: `penalty` off the logit of every token
        the answer has generated so far, once however often (never the
        prompt's)."""

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            generated = input_ids[:, prompt_length:]
            if generated.shape[1] == 0:
                return scores
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, generated.to(scores.device), True)
            return scores - PRESENCE_PENALTY * seen.to(scores.dtype)  # pyright: ignore[reportReturnType]

    class StopWhenAsked(StoppingCriteria):
        """Ends the generation at the next token once Stop is pressed."""

        def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> torch.BoolTensor:
            stop = should_cancel is not None and should_cancel()
            return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)  # pyright: ignore[reportReturnType]

    class StopPastLimit(StoppingCriteria):
        """Ends the generation once the caption so far is past the prompt
        limit: the rest could never reach the box (`too_long`)."""

        def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> torch.BoolTensor:
            stop = decode is not None and len(decode(input_ids[0, prompt_length:]).strip()) > PROMPT_LIMIT
            return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)  # pyright: ignore[reportReturnType]

    return LogitsProcessorList([PresencePenalty()]), StoppingCriteriaList([StopWhenAsked(), StopPastLimit()])


def upsample(model: Any, processor: Any, prompt: str, device: Any, should_cancel: CancelFn | None = None) -> str:
    """One caption for `prompt`: greedy, with the card's presence penalty,
    polled for Stop at every token. Raises `Cancelled` when stopped, and
    `ExpansionTooLong` once the caption runs past `PROMPT_LIMIT`."""
    import torch

    inputs = processor.apply_chat_template(
        messages(prompt), add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False,
    ).to(device)
    prompt_length = int(inputs["input_ids"].shape[1])
    def decode(ids: Any) -> str:
        return str(processor.decode(ids, skip_special_tokens=True))

    logits, stopping = _processors(prompt_length, should_cancel, decode)
    # no_grad, never inference_mode, as around every pipeline call: offload
    # moves torchao tensors across devices inside forward().
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens(), do_sample=False,
            logits_processor=logits, stopping_criteria=stopping,
        )
    if should_cancel is not None and should_cancel():
        raise Cancelled("prompt expansion stopped")
    caption = decode(generated[0, prompt_length:]).strip()
    if not caption:
        raise RuntimeError("FLUX 3's prompt upsampler returned an empty caption")
    if len(caption) > PROMPT_LIMIT:
        raise ExpansionTooLong(too_long())
    return caption
