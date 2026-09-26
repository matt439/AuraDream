"""SDXL ignores WAI v17's non-finite final CLIP-L layer in both prompts.

The actual SDXL encoder path and tiny random CLIPs run on CPU, without model
downloads. Keep this contract checked when Diffusers/Transformers change:
preserving the checkpoint is safe only while conditioning uses the preceding
hidden state and the second encoder supplies the pooled embeddings.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

# Import config before Diffusers, as in the application's model-loading path.
from backend import config as _config  # noqa: F401

import pytest
import torch
from diffusers import StableDiffusionXLPipeline
from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection


class TinyTokenizer:
    model_max_length = 4

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        return SimpleNamespace(input_ids=torch.tensor([
            [0, 3 if prompt == "positive" else 5, 1, 2]
            for prompt in prompts
        ]))


def _encoder_config(**kwargs):
    return CLIPTextConfig(
        vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=12, num_attention_heads=2,
        max_position_embeddings=4, bos_token_id=0, eos_token_id=2,
        projection_dim=8, **kwargs,
    )


def _encode(first, second, tokenizer):
    # Only tokenization is a stand-in. Calling the installed pipeline method
    # directly avoids constructing an unrelated VAE, denoiser and scheduler.
    pipe = SimpleNamespace(
        text_encoder=first, text_encoder_2=second,
        tokenizer=tokenizer, tokenizer_2=tokenizer,
        config=SimpleNamespace(force_zeros_for_empty_prompt=True),
    )
    return StableDiffusionXLPipeline.encode_prompt(
        pipe, prompt="positive", negative_prompt="negative",
        device=torch.device("cpu"), do_classifier_free_guidance=True,
    )


@pytest.mark.parametrize("quant", ["none", "fp8", "int8"])
def test_unused_clip_l_nans_preserve_both_sdxl_conditioning_paths(quant):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        first = CLIPTextModel(_encoder_config()).eval().to(torch.bfloat16)
        second = CLIPTextModelWithProjection(_encoder_config()).eval().to(torch.bfloat16)
    broken = copy.deepcopy(first)
    tokenizer = TinyTokenizer()

    with torch.no_grad():
        # WAI's affected layer is index 11. Poison every matrix there to make
        # accidentally consuming its output unambiguously fail this test.
        for parameter in broken.encoder.layers[11].parameters():
            if parameter.ndim == 2:
                parameter.fill_(float("nan"))

    if quant != "none":
        quant_config = Float8WeightOnlyConfig() if quant == "fp8" else Int8WeightOnlyConfig()
        for encoder in (first, broken, second):
            quantize_(encoder, quant_config)

    with torch.no_grad():
        expected = _encode(first, second, tokenizer)
        actual = _encode(broken, second, tokenizer)
        for clean, with_nans in zip(expected, actual, strict=True):
            assert torch.isfinite(with_nans).all()
            assert torch.equal(clean, with_nans)

        positive, negative, pooled_positive, pooled_negative = actual
        assert not torch.equal(positive, negative)
        assert not torch.equal(pooled_positive, pooled_negative)

        for prompt, pooled in (("positive", pooled_positive), ("negative", pooled_negative)):
            ids = tokenizer(prompt).input_ids
            first_outputs = broken(ids, output_hidden_states=True)
            assert torch.isfinite(first_outputs.hidden_states[-2]).all()
            assert not torch.isfinite(first_outputs.pooler_output).all()
            # Both positive and negative pooling must come from encoder 2;
            # encoder 1's last hidden state and pool remain deliberately NaN.
            assert torch.equal(pooled, second(ids).text_embeds)
