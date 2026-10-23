"""Run the upstream Anima blocks on CPU with tiny, deterministic components.

Only learned models and tokenizers are stand-ins: prompt batching, latent noise,
CFG, timesteps, scheduler steps and image postprocessing all use Diffusers.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from backend.backends import Cancelled
from backend.backends.anima_pipeline import AnimaAestheticPipeline, AnimaPreviewPipeline, AnimaTurboPipeline
from backend.latent_preview import unpack
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin


class _Tokenizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        # Distinct positive and negative embeddings let tests observe whether
        # the negative pass is really used in the denoising arithmetic.
        values = [[2 if prompt == "sunny garden" else 5] for prompt in prompts]
        ids = torch.tensor(values, dtype=torch.long, device="cpu")
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


class _CPUModel(ModelMixin):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, device="cpu"))


class _TextEncoder(_CPUModel):
    def forward(self, input_ids, attention_mask, output_hidden_states):
        return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))


class _Conditioner(_CPUModel):
    def forward(self, source_hidden_states, target_input_ids,
                target_attention_mask, source_attention_mask):
        # Include both encoder paths without requiring either model's weights.
        return (source_hidden_states + target_input_ids.unsqueeze(-1)) / 100


class _Transformer(_CPUModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=3)
        self.calls = []

    def forward(self, hidden_states, timestep, padding_mask,
                encoder_hidden_states, return_dict):
        condition = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        prediction = hidden_states * 0.1 + condition
        self.calls.append((hidden_states.clone(), prediction.clone()))
        return (prediction,)


class _VAE(_CPUModel):
    temperal_downsample = [False, True, True]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            z_dim=3, latents_mean=[0.0] * 3, latents_std=[1.0] * 3,
        )
        self.decode_calls = 0

    def decode(self, latents, return_dict):
        self.decode_calls += 1
        return (latents.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1),)


@pytest.fixture(params=[AnimaTurboPipeline, AnimaAestheticPipeline, AnimaPreviewPipeline])
def pipe(request):
    pipeline = request.param(
        tokenizer=_Tokenizer(), t5_tokenizer=_Tokenizer(),
        text_encoder=_TextEncoder(), text_conditioner=_Conditioner(),
        transformer=_Transformer(), vae=_VAE(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _render(pipe, **changes):
    args = dict(
        prompt="sunny garden", negative_prompt="rain",
        width=32, height=32, num_inference_steps=3,
        generator=_generator(17), guidance_scale=1.0,
    )
    args.update(changes)
    return pipe(**args).images


@pytest.mark.parametrize("guidance", [1.0, 4.0])
def test_batch_images_reproduce_individually_with_cpu_generators(pipe, guidance):
    seeds = [17, 18, 19]
    images = _render(
        pipe, generator=[_generator(seed) for seed in seeds],
        num_images_per_prompt=len(seeds), guidance_scale=guidance,
    )
    assert len(images) == len(seeds)
    assert len({image.tobytes() for image in images}) == len(seeds)
    for seed, image in zip(seeds, images):
        rerendered = _render(pipe, generator=_generator(seed), guidance_scale=guidance)[0]
        assert image.mode == "RGB"
        assert image.size == (32, 32)
        assert image.tobytes() == rerendered.tobytes()
    assert pipe.device == torch.device("cpu")


@pytest.mark.parametrize("guidance", [0.0, 1.0, 4.0])
def test_guidance_uses_negative_pass_only_above_one(pipe, guidance):
    finished = []

    def record(pipeline, step, timestep, values):
        finished.append(values["latents"].clone())
        return values

    _render(pipe, num_inference_steps=1, guidance_scale=guidance,
            callback_on_step_end=record)
    expected_prompts = ["sunny garden", "rain"] if guidance > 1 else ["sunny garden"]
    assert pipe.tokenizer.prompts == expected_prompts
    assert pipe.t5_tokenizer.prompts == expected_prompts
    calls = pipe.transformer.calls
    assert len(calls) == len(expected_prompts)
    initial, positive = calls[0]
    prediction = positive
    if guidance > 1:
        _, negative = calls[1]
        assert not torch.equal(positive, negative)
        prediction = negative + guidance * (positive - negative)
    # A single real FlowMatch step goes from sigma=1 to zero. Checking its
    # output catches an ignored or mis-scaled negative pass, not just a call.
    torch.testing.assert_close(finished[0], initial - prediction)
    assert pipe.vae.decode_calls == 1


def test_callback_observes_completed_steps_and_cancel_skips_decode(pipe, monkeypatch, caplog):
    # Diffusers disables propagation by default, so opt in during this test
    # to ensure caplog would catch its misleading cancellation tracebacks.
    monkeypatch.setattr(logging.getLogger("diffusers"), "propagate", True)
    caplog.set_level(logging.ERROR, logger="diffusers")
    # Exercise the real Accelerate hook lifecycle on an explicit CPU device;
    # model weights never touch an accelerator, even on GPU-equipped hosts.
    pipe.enable_model_cpu_offload(device="cpu")
    previous_hooks = list(pipe._all_hooks)
    seen = []
    freed = []
    original_cleanup = pipe.maybe_free_model_hooks

    def cleanup():
        freed.append(True)
        original_cleanup()

    monkeypatch.setattr(pipe, "maybe_free_model_hooks", cleanup)

    def cancel(pipeline, step, timestep, values):
        assert pipeline is pipe
        assert pipeline.scheduler.step_index == step + 1
        assert values["latents"].shape == (1, 3, 1, 4, 4)
        preview = unpack(pipeline, values["latents"], height=32, width=32)
        assert preview.shape == (3, 4, 4)
        torch.testing.assert_close(preview, values["latents"][0, :, 0])
        assert timestep == pipeline.scheduler.timesteps[step]
        seen.append(step)
        if step == 1:
            raise Cancelled
        return values

    with pytest.raises(Cancelled):
        _render(pipe, callback_on_step_end=cancel)
    assert not any(
        record.levelno >= logging.ERROR and "Error in block" in record.getMessage()
        for record in caplog.records
    )
    assert seen == [0, 1]
    assert len(pipe.transformer.calls) == 2
    assert pipe.vae.decode_calls == 0
    assert pipe._step.callback is None
    assert freed == [True]
    assert len(pipe._all_hooks) == len(previous_hooks) == 4
    assert all(before is not after for before, after in zip(previous_hooks, pipe._all_hooks))

    # A subsequent call must neither inherit the cancelled callback nor a
    # partially advanced scheduler/guider state from the previous attempt.
    images = _render(pipe)
    assert len(images) == 1
    assert seen == [0, 1]
    assert pipe.vae.decode_calls == 1
    assert freed == [True, True]


def test_guided_calls_start_with_fresh_guider_after_unguided_call(pipe):
    _render(pipe, guidance_scale=0.0)
    unguided_guider = pipe.guider
    first = _render(pipe, guidance_scale=1.5)
    guided_guider = pipe.guider
    second = _render(pipe, guidance_scale=1.5)
    assert pipe.guider is not guided_guider
    assert guided_guider is not unguided_guider
    assert first[0].tobytes() == second[0].tobytes()
    assert pipe.tokenizer.prompts == [
        "sunny garden", "sunny garden", "rain", "sunny garden", "rain",
    ]


def test_generator_count_must_match_batch(pipe):
    with pytest.raises(ValueError, match="list of generators of length 1"):
        _render(pipe, generator=[_generator(17)], num_images_per_prompt=2)
    assert pipe.transformer.calls == []
    assert pipe.vae.decode_calls == 0
    assert pipe._step.callback is None
