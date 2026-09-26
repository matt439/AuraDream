"""CPU inference contracts with the real flow scheduler and tiny components."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin

from backend.backends import Cancelled, schedule
from backend.backends.newbie_pipeline import NewbiePipeline


class _Tokenizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        values = [[2, 3, 0] if prompt == "sunny garden" else [5, 0, 0] for prompt in prompts]
        ids = torch.tensor(values)
        return SimpleNamespace(input_ids=ids, attention_mask=ids.ne(0).long())


class _CPUModel(ModelMixin):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _Gemma(_CPUModel):
    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
        assert output_hidden_states and not use_cache
        values = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(hidden_states=(values - 10, values, values + 10))


class _Jina(_CPUModel):
    def forward(self, input_ids, attention_mask):
        pooled = (input_ids * attention_mask).sum(-1, keepdim=True).float()
        return SimpleNamespace(pooler_output=pooled)


class _Transformer(_CPUModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=3)
        self.calls = []
        self.fixed_predictions = None

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask, pooled_projections, return_dict):
        condition = (encoder_hidden_states.squeeze(-1) * encoder_attention_mask).sum(-1)
        prediction = hidden_states * 0.05 + (condition + pooled_projections[:, 0]).view(-1, 1, 1, 1) / 100
        if self.fixed_predictions is not None:
            prediction = self.fixed_predictions[len(self.calls)].expand_as(hidden_states)
        self.calls.append(dict(
            initial=hidden_states.clone(), prediction=prediction.clone(), timestep=timestep.clone(),
            embeddings=encoder_hidden_states.clone(), mask=encoder_attention_mask.clone(),
            pooled=pooled_projections.clone(),
        ))
        return (prediction,)


class _VAE(_CPUModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            block_out_channels=[1, 1, 1, 1], scaling_factor=0.5, shift_factor=0.25,
        )
        self.decoded = []

    def decode(self, latents, return_dict):
        self.decoded.append(latents.clone())
        return (latents.repeat_interleave(8, -1).repeat_interleave(8, -2),)


@pytest.fixture
def pipe():
    pipeline = NewbiePipeline(
        transformer=_Transformer(), text_encoder=_Gemma(), text_encoder_2=_Jina(),
        tokenizer=_Tokenizer(), tokenizer_2=_Tokenizer(), vae=_VAE(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3),
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def _generator(seed=17):
    return torch.Generator(device="cpu").manual_seed(seed)


def _render(pipe, **changes):
    args = dict(prompt="sunny garden", negative_prompt="rain", height=32, width=32,
                num_inference_steps=3, guidance_scale=1.0, generator=_generator(), output_type="latent")
    args.update(changes)
    return pipe(**args).images


def test_dual_conditioning_uses_penultimate_state_masks_and_pooled_vectors(pipe):
    encoded, mask, pooled = pipe._encode(["sunny garden", "rain"], torch.device("cpu"), 2)
    torch.testing.assert_close(encoded.squeeze(-1), torch.tensor([[2., 3., 0.], [2., 3., 0.], [5., 0., 0.], [5., 0., 0.]]))
    torch.testing.assert_close(mask, torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]]))
    torch.testing.assert_close(pooled, torch.tensor([[5.], [5.], [5.], [5.]]))
    assert pipe.tokenizer.prompts == pipe.tokenizer_2.prompts == ["sunny garden", "rain"]


def test_xml_example_can_reach_both_encoders_without_jina_truncation(pipe):
    # The pinned author's XML example is 486 Gemma tokens but 539 Jina tokens.
    # Giving both towers Gemma's 512-token budget silently lost the XML tail.
    class Tokenizer:
        def __init__(self, count):
            self.count = count

        def __call__(self, prompts, max_length, **kwargs):
            ids = torch.ones(len(prompts), min(self.count, max_length), dtype=torch.long)
            return SimpleNamespace(input_ids=ids, attention_mask=ids)

    pipe.tokenizer = Tokenizer(486)
    pipe.tokenizer_2 = Tokenizer(539)
    encoded, mask, pooled = pipe._encode(["author XML"], torch.device("cpu"), 1)
    assert encoded.shape[1] == mask.shape[1] == 486
    assert pooled.item() == 539


@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_seeded_batch_matches_individual_images_and_repeats(pipe, guidance):
    seeds = [17, 18, 19]
    images = _render(pipe, guidance_scale=guidance, num_images_per_prompt=3,
                     generator=[_generator(seed) for seed in seeds])
    assert not torch.equal(images[0], images[1])
    for seed, image in zip(seeds, images):
        torch.testing.assert_close(image, _render(pipe, guidance_scale=guidance, generator=_generator(seed))[0])
    torch.testing.assert_close(_render(pipe, guidance_scale=guidance), images[:1])


@pytest.mark.parametrize("guidance", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("shrink", [False, True])
def test_cfg_threshold_whole_sample_capped_norm_and_velocity_sign(pipe, guidance, shrink):
    conditional = torch.arange(1, 49).float().reshape(1, 3, 4, 4) / 10
    # The shrink case distinguishes a capped norm from erroneously enlarging
    # a weak prediction; reversal distinguishes global from per-row norms.
    unconditional = conditional * 1.75 if shrink else conditional.flip(-1) * 2
    pipe.transformer.fixed_predictions = [conditional, unconditional]
    result = _render(pipe, num_inference_steps=1, guidance_scale=guidance)
    expected_prompts = ["sunny garden", "rain"] if guidance > 1 else ["sunny garden"]
    assert pipe.tokenizer.prompts == pipe.tokenizer_2.prompts == expected_prompts
    assert len(pipe.transformer.calls) == len(expected_prompts)
    prediction = conditional
    if guidance > 1:
        prediction = unconditional + guidance * (conditional - unconditional)
        ratio = min(conditional.flatten().norm().item() / prediction.flatten().norm().item(), 1.0)
        prediction = prediction * ratio
    # A sigma 1 -> 0 step consumes NEGATIVE NextDiT velocity, so the latent
    # gains the model prediction. Checking only the transformer call misses it.
    torch.testing.assert_close(result, pipe.transformer.calls[0]["initial"] + prediction)


def test_mu_lands_on_real_scheduler_and_nextdit_time_is_reversed(pipe):
    steps, mu = 4, 0.8
    sigmas = schedule.shifted_sigmas(steps, mu - math.log(schedule.static_shift(pipe)))
    _render(pipe, num_inference_steps=steps, sigmas=sigmas)
    torch.testing.assert_close(pipe.scheduler.sigmas, torch.tensor(schedule.shifted_sigmas(steps, mu) + [0.0]))
    for call, sigma in zip(pipe.transformer.calls, pipe.scheduler.sigmas):
        torch.testing.assert_close(call["timestep"], (1 - sigma).reshape(1))


def test_cancellation_offloads_and_next_call_recovers(pipe, monkeypatch):
    pipe.enable_model_cpu_offload(device="cpu")
    hooks = list(pipe._all_hooks)
    cleanup = pipe.maybe_free_model_hooks
    cleaned = []
    seen = []

    def release():
        cleaned.append(True)
        cleanup()

    monkeypatch.setattr(pipe, "maybe_free_model_hooks", release)

    def cancel(pipeline, step, timestep, values):
        assert pipeline is pipe and pipe.scheduler.step_index == step + 1
        assert values["latents"].shape == (1, 3, 4, 4)
        seen.append(step)
        if step == 1:
            raise Cancelled
        return values

    with pytest.raises(Cancelled):
        _render(pipe, callback_on_step_end=cancel, output_type="pil")
    assert seen == [0, 1] and cleaned == [True]
    assert pipe.vae.decoded == []
    assert len(pipe._all_hooks) == len(hooks) == 4
    assert all(a is not b for a, b in zip(hooks, pipe._all_hooks))
    images = _render(pipe, output_type="pil")
    assert len(images) == 1 and images[0].size == (32, 32)
    assert seen == [0, 1] and cleaned == [True, True]
    assert len(pipe.vae.decoded) == 1


def test_callback_latent_update_reaches_decode_scaling(pipe):
    def replace(pipeline, step, timestep, values):
        return {"latents": torch.full_like(values["latents"], 0.4)}

    images = _render(pipe, callback_on_step_end=replace, output_type="pil")
    torch.testing.assert_close(pipe.vae.decoded[0], torch.full((1, 3, 4, 4), 1.05))
    assert images[0].mode == "RGB"


def test_bad_generator_count_is_refused_before_denoising(pipe):
    with pytest.raises(ValueError):
        _render(pipe, num_images_per_prompt=2, generator=[_generator()])
    assert pipe.transformer.calls == []


def test_real_tiny_pipeline_saves_loads_and_reproduces_without_remote_code(tmp_path):
    from diffusers import AutoencoderKL
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        Gemma3TextConfig, Gemma3TextModel, JinaEmbeddingsV3Config, PreTrainedTokenizerFast,
    )

    from backend.backends.newbie_text_encoder import NewbieJinaTextEncoder
    from backend.backends.newbie_transformer import NewbieTransformer2DModel

    tokenizer_object = Tokenizer(WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "sunny": 2, "garden": 3, "rain": 4, "[BOS]": 5, "[EOS]": 6},
        unk_token="[UNK]",
    ))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer_object.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 5), ("[EOS]", 6)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
    )
    torch.manual_seed(818)
    pipeline = NewbiePipeline(
        tokenizer=tokenizer, tokenizer_2=tokenizer,
        text_encoder=Gemma3TextModel(Gemma3TextConfig(
            vocab_size=19, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=1, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=32, query_pre_attn_scalar=8, pad_token_id=0,
        )).eval(),
        text_encoder_2=NewbieJinaTextEncoder(JinaEmbeddingsV3Config(
            vocab_size=19, hidden_size=6, intermediate_size=12, num_hidden_layers=1,
            num_attention_heads=1, max_position_embeddings=32, pad_token_id=0,
        )).eval(),
        transformer=NewbieTransformer2DModel(
            sample_size=4, in_channels=4, hidden_size=24, num_layers=1,
            num_refiner_layers=1, num_attention_heads=4, num_kv_heads=2,
            multiple_of=8, axes_dim_rope=(2, 2, 2), axes_lens=(8, 8, 8),
            cap_feat_dim=8, pooled_projection_dim=6,
        ).eval(),
        vae=AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(4, 4, 4, 4), norm_num_groups=1,
            mid_block_add_attention=False, shift_factor=0.1159,
        ).eval(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3),
    )
    pipeline.set_progress_bar_config(disable=True)
    expected = _render(pipeline, guidance_scale=2, num_inference_steps=2)
    pipeline.save_pretrained(tmp_path, safe_serialization=True)
    restored = NewbiePipeline.from_pretrained(tmp_path, local_files_only=True)
    restored.set_progress_bar_config(disable=True)
    actual = _render(restored, guidance_scale=2, num_inference_steps=2)
    torch.testing.assert_close(actual, expected)
    assert type(restored.transformer) is NewbieTransformer2DModel
    assert type(restored.text_encoder_2) is NewbieJinaTextEncoder
    assert actual.device.type == "cpu"
