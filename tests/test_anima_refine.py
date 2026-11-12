"""The private Anima image-to-image pass on CPU (GEN#48 stage 1).

Tiny stand-ins replace the learned models and tokenizers only; the encoder
step, schedule slicing, noise mixing, guidance, denoising loop and decode are
the installed Diffusers Anima blocks, on a real FlowMatch scheduler. Nothing
here is a card measurement or a release: the public Refine stays held.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from backend import refine
from backend.backends import Cancelled
from backend.backends.anima_pipeline import AnimaAestheticPipeline, AnimaPreviewPipeline, AnimaTurboPipeline
from backend.backends.anima_refine import (
    IMG2IMG,
    AnimaAestheticImg2ImgPipeline,
    AnimaPreviewImg2ImgPipeline,
    AnimaTurboImg2ImgPipeline,
    start_index,
    steps_run,
)
from backend.model_defaults import defaults_for
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines.anima.before_denoise import AnimaImg2ImgSetTimestepsStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.utils.torch_utils import randn_tensor
from tests.test_anima_pipeline import _Conditioner, _TextEncoder, _Tokenizer, _Transformer, _VAE

ANIMA_IDS = ("circlestone-labs/Anima", "circlestone-labs/Anima@aesthetic-v1.1", "Gazingstars123/Anima-2.9B")
MEAN = [0.25, -0.5, 0.125]
STD = [2.0, 0.5, 4.0]
SDXL_ID = "cagliostrolab/animagine-xl-4.0"  # refine's non-mu (SDXL) rounding


class _Posterior:
    """Mode and sample differ, so a sampled encode cannot pass for the mode."""

    def __init__(self, mode):
        self._mode = mode

    def mode(self):
        return self._mode

    def sample(self, generator=None):
        return self._mode + 100 + torch.randn(self._mode.shape, generator=generator)


class _EncodingVAE(_VAE):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(z_dim=3, latents_mean=list(MEAN), latents_std=list(STD))
        self.encoded = []

    def encode(self, image):
        self.encoded.append(image.clone())
        return SimpleNamespace(latent_dist=_Posterior(image[..., ::8, ::8]))


def _text_pipe(cls):
    pipe = cls(
        tokenizer=_Tokenizer(), t5_tokenizer=_Tokenizer(),
        text_encoder=_TextEncoder(), text_conditioner=_Conditioner(),
        transformer=_Transformer(), vae=_EncodingVAE(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


@pytest.fixture(params=[AnimaTurboPipeline, AnimaAestheticPipeline, AnimaPreviewPipeline])
def pipes(request):
    text = _text_pipe(request.param)
    runner = refine.build(text, IMG2IMG[request.param])
    runner.set_progress_bar_config(disable=True)
    return text, runner


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _source(colour=(200, 40, 90)):
    image = Image.new("RGB", (32, 32), colour)
    image.putpixel((3, 5), (0, 255, 0))
    return image


def _refine(runner, **changes):
    args = dict(
        prompt="sunny garden", negative_prompt="rain", image=_source(),
        strength=0.5, width=32, height=32, num_inference_steps=4,
        generator=_generator(17), guidance_scale=1.0,
    )
    args.update(changes)
    return runner(**args).images


def _block_slice(steps, strength):
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    components = SimpleNamespace(scheduler=scheduler, _execution_device=torch.device("cpu"))
    state = PipelineState(values={"num_inference_steps": steps, "strength": strength})
    _, state = AnimaImg2ImgSetTimestepsStep()(components, state)
    return scheduler.begin_index, state.get("num_inference_steps"), float(scheduler.sigmas[scheduler.begin_index])


# --- The public Refine stays held -------------------------------------------


@pytest.mark.parametrize("text_cls", [AnimaTurboPipeline, AnimaAestheticPipeline, AnimaPreviewPipeline])
def test_public_resolver_still_refuses_every_anima_class(text_cls):
    assert refine.sibling(text_cls) is None
    assert refine.sibling(text_cls, "Inpaint") is None
    assert refine.sibling(IMG2IMG[text_cls]) is None


@pytest.mark.parametrize("model_id", ANIMA_IDS)
def test_anima_profiles_keep_refine_unsupported(model_id):
    profile = defaults_for(model_id)
    assert profile["refine_supported"] is False
    assert profile["inpaint_supported"] is False
    assert refine.start_index(8, 0.4, model_id) is None


@pytest.mark.parametrize(
    "cls, steps, cfg",
    [(AnimaTurboImg2ImgPipeline, 8, 1.0), (AnimaAestheticImg2ImgPipeline, 30, 4.0),
     (AnimaPreviewImg2ImgPipeline, 30, 4.0)],
)
def test_call_takes_what_a_refine_pass_hands_it_with_family_defaults(cls, steps, cfg):
    import inspect

    params = inspect.signature(cls.__call__).parameters
    own = set(inspect.signature(next(k for k, v in IMG2IMG.items() if v is cls).__call__).parameters)
    assert refine.REQUIRED <= set(params)
    assert not (refine.CARRIED & own) - set(params)
    assert params["num_inference_steps"].default == steps
    assert params["guidance_scale"].default == cfg
    assert params["strength"].default is inspect.Parameter.empty


# --- Rounding and the schedule ------------------------------------------------


@pytest.mark.parametrize(
    "steps, strength, begin, run, sigma",
    [(8, 0.4, 4, 4, 0.75), (8, 1.0, 0, 8, 1.0), (30, 0.4, 18, 12, 0.666667),
     (30, 0.1, 27, 3, 0.25), (8, 0.125, 7, 1, 0.3)],
)
def test_installed_block_matches_stage_zero_probe(steps, strength, begin, run, sigma):
    assert _block_slice(steps, strength)[:2] == (begin, run)
    assert _block_slice(steps, strength)[2] == pytest.approx(sigma, abs=1e-6)
    assert (start_index(steps, strength), steps_run(steps, strength)) == (begin, run)


def test_rounding_matches_the_installed_block_on_fractional_boundaries():
    strengths = [k / 100 for k in range(1, 101)] + [0.29, 0.3, 0.7, 1 / 3, 2 / 3, 0.35, 0.45, 0.55]
    for steps in (1, 3, 4, 7, 8, 10, 12, 28, 30, 50):
        for strength in strengths:
            begin, run, _ = _block_slice(steps, strength)
            if run == 0:
                continue  # the empty slice the call refuses
            assert start_index(steps, strength) == begin, (steps, strength)
            assert steps_run(steps, strength) == run, (steps, strength)


def test_anima_rounds_as_flow_not_as_its_mu_flag_says():
    # `refine.start_index` keys the rounding on `mu_supported`, which Anima's
    # fixed shift leaves false; SDXL's branch would run three steps of eight.
    assert steps_run(8, 0.4) == 4
    assert defaults_for(SDXL_ID)["mu_supported"] is False
    assert 8 - refine.start_index(8, 0.4, SDXL_ID) == 3


def test_start_sigma_reads_a_copy_and_matches_the_first_noised_step(pipes):
    _, runner = pipes
    live = runner.scheduler
    live.set_timesteps(5)
    before = (live.timesteps.clone(), live.sigmas.clone(), live.begin_index, live.shift)
    expected = runner.start_sigma(8, 0.4)
    assert expected == pytest.approx(0.75)
    assert torch.equal(live.timesteps, before[0]) and torch.equal(live.sigmas, before[1])
    assert (live.begin_index, live.shift) == before[2:]

    seen = []

    def record(pipeline, step, timestep, values):
        if step == 0:
            scheduler = pipeline.scheduler
            seen.append(float(scheduler.sigmas[scheduler.begin_index]))
            seen.append(float(timestep) / scheduler.config.num_train_timesteps)
        return values

    _refine(runner, num_inference_steps=8, strength=0.4, callback_on_step_end=record)
    assert seen[0] == pytest.approx(expected)
    assert seen[1] == pytest.approx(expected)


def test_start_sigma_keeps_a_runtime_shift_set_on_the_live_scheduler(pipes):
    # Every fixture's config says shift 3; a live `set_shift` must still reach
    # the copy, or the preview names shift 3's sigma for a shift-5 pass.
    _, runner = pipes
    runner.scheduler.set_shift(5.0)
    expected = runner.start_sigma(8, 0.4)
    assert expected == pytest.approx(5 * 0.5 / (1 + 4 * 0.5))
    assert runner.scheduler.shift == 5.0

    seen = []

    def record(pipeline, step, timestep, values):
        if step == 0:
            scheduler = pipeline.scheduler
            seen.append(float(scheduler.sigmas[scheduler.begin_index]))
        return values

    _refine(runner, num_inference_steps=8, strength=0.4, callback_on_step_end=record)
    assert seen == [pytest.approx(expected)]


def test_start_sigma_refuses_what_the_call_refuses(pipes):
    _, runner = pipes
    live = runner.scheduler
    live.set_timesteps(5)
    before = live.sigmas.clone()
    for strength in (0.0, -0.25, 1.5, 1e-20):
        with pytest.raises(ValueError):
            runner.start_sigma(8, strength)
    assert torch.equal(live.sigmas, before)


def test_strength_that_leaves_no_step_is_refused_before_any_component(pipes):
    _, runner = pipes
    # 1e-20 is positive but vanishes in `steps - steps * strength`.
    for strength in (0.0, -0.25, 1.5, 1e-20):
        with pytest.raises(ValueError):
            _refine(runner, num_inference_steps=4, strength=strength)
    assert runner.transformer.calls == []
    assert runner.vae.encoded == []
    assert runner._step.callback is None


# --- The source, its encoding and the noise ----------------------------------


def test_noised_source_is_posterior_mode_normalised_then_scaled(pipes):
    _, runner = pipes
    source = _source()
    _refine(runner, image=source, num_inference_steps=8, strength=0.4, generator=_generator(5))
    (encoded,) = runner.vae.encoded
    processed = runner.image_processor.preprocess(source, height=32, width=32)
    torch.testing.assert_close(encoded, processed.unsqueeze(2))
    mean = torch.tensor(MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1, 1)
    normalised = (processed.unsqueeze(2)[..., ::8, ::8] - mean) / std
    # Posterior mode draws nothing, so the generator's first draw is the noise.
    noise = randn_tensor(normalised.shape, generator=_generator(5), dtype=torch.float32)
    sigma = runner.start_sigma(8, 0.4)
    first_input = runner.transformer.calls[0][0]
    torch.testing.assert_close(first_input, sigma * noise + (1 - sigma) * normalised)


def test_source_changes_the_result(pipes):
    _, runner = pipes
    red = _refine(runner, image=_source((200, 40, 90)))[0]
    blue = _refine(runner, image=_source((20, 40, 220)))[0]
    assert red.tobytes() != blue.tobytes()
    assert red.size == blue.size == (32, 32)


def test_one_image_per_call_reproduces_its_seed(pipes):
    _, runner = pipes
    first = _refine(runner, generator=_generator(17))[0]
    again = _refine(runner, generator=_generator(17))[0]
    listed = _refine(runner, generator=[_generator(17)])[0]
    other = _refine(runner, generator=_generator(18))[0]
    assert first.tobytes() == again.tobytes() == listed.tobytes()
    assert first.tobytes() != other.tobytes()


@pytest.mark.parametrize("guidance", [0.0, 1.0, 4.0])
def test_negative_conditioning_only_above_cfg_one(pipes, guidance):
    _, runner = pipes
    finished = []

    def record(pipeline, step, timestep, values):
        scheduler = pipeline.scheduler
        begin = scheduler.begin_index
        finished.append((values["latents"].clone(), scheduler.sigmas[begin], scheduler.sigmas[begin + 1]))
        return values

    _refine(runner, guidance_scale=guidance, callback_on_step_end=record)
    expected = ["sunny garden", "rain"] if guidance > 1 else ["sunny garden"]
    assert runner.tokenizer.prompts == expected
    assert runner.t5_tokenizer.prompts == expected
    calls = runner.transformer.calls
    assert len(calls) == len(expected) * steps_run(4, 0.5)
    initial, positive = calls[0]
    prediction = positive
    if guidance > 1:
        _, negative = calls[1]
        assert not torch.equal(positive, negative)
        prediction = negative + guidance * (positive - negative)
    # The first real Euler step from the sliced start: an ignored or
    # mis-scaled negative pass changes this, not just the call count.
    latents, sigma, following = finished[0]
    torch.testing.assert_close(latents, initial + (following - sigma) * prediction)


# --- Shared components, no casts or downloads ---------------------------------


def test_sibling_shares_every_component_without_casts_or_downloads(monkeypatch):
    import huggingface_hub
    from diffusers import DiffusionPipeline

    def refuse(*args, **kwargs):
        raise AssertionError("no download, move or cast is allowed")

    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(huggingface_hub, name, refuse)
    text = _text_pipe(AnimaTurboPipeline)
    dtypes = {name: [p.dtype for p in module.parameters()]
              for name, module in text.components.items() if isinstance(module, torch.nn.Module)}
    monkeypatch.setattr(torch.nn.Module, "to", refuse)
    monkeypatch.setattr(DiffusionPipeline, "to", refuse)
    runner = refine.build(text, AnimaTurboImg2ImgPipeline)
    runner.set_progress_bar_config(disable=True)
    assert set(runner.components) == set(text.components)
    for name, component in text.components.items():
        assert runner.components[name] is component, name
    _refine(runner)
    for name, module in text.components.items():
        if isinstance(module, torch.nn.Module):
            assert [p.dtype for p in module.parameters()] == dtypes[name]
            assert all(p.device == torch.device("cpu") for p in module.parameters())


def test_direct_samples_keep_their_pixels_and_noise_after_a_pass():
    text = _text_pipe(AnimaTurboPipeline)
    args = dict(prompt="sunny garden", negative_prompt="rain", width=32, height=32,
                num_inference_steps=3, guidance_scale=1.0)
    before = text(generator=_generator(17), **args).images[0]
    runner = refine.build(text, AnimaTurboImg2ImgPipeline)
    runner.set_progress_bar_config(disable=True)
    _refine(runner)
    after = text(generator=_generator(17), **args).images[0]
    assert before.tobytes() == after.tobytes()
    # The text pass's first denoiser input is still the generator's first draw.
    text.transformer.calls.clear()
    text(generator=_generator(17), **args)
    first = text.transformer.calls[0][0]
    torch.testing.assert_close(first, randn_tensor(first.shape, generator=_generator(17), dtype=torch.float32))


# --- Progress and cleanup ------------------------------------------------------


def test_callback_sees_the_sliced_count_before_its_first_step(pipes):
    _, runner = pipes
    seen = []

    def record(pipeline, step, timestep, values):
        assert pipeline is runner
        assert pipeline.num_timesteps == steps_run(8, 0.4)
        scheduler = pipeline.scheduler
        assert scheduler.step_index == scheduler.begin_index + step + 1
        assert timestep == scheduler.timesteps[scheduler.begin_index + step]
        seen.append(step)
        return values

    images = _refine(runner, num_inference_steps=8, strength=0.4, callback_on_step_end=record)
    assert seen == [0, 1, 2, 3]
    assert len(images) == 1 and images[0].mode == "RGB"
    assert runner.vae.decode_calls == 1
    assert runner._step.callback is None


def _count_cleanup(monkeypatch, runner):
    freed = []
    original = runner.maybe_free_model_hooks

    def cleanup():
        freed.append(True)
        original()

    monkeypatch.setattr(runner, "maybe_free_model_hooks", cleanup)
    return freed


def test_cleanup_after_success(pipes, monkeypatch):
    _, runner = pipes
    freed = _count_cleanup(monkeypatch, runner)
    _refine(runner, callback_on_step_end=lambda *a: a[3])
    assert freed == [True]
    assert runner._step.callback is None


def test_cleanup_after_an_exception(pipes, monkeypatch):
    _, runner = pipes
    freed = _count_cleanup(monkeypatch, runner)

    def fail(*args, **kwargs):
        raise RuntimeError("denoiser failed")

    monkeypatch.setattr(runner.transformer, "forward", fail)
    with pytest.raises(RuntimeError, match="denoiser failed"):
        _refine(runner, callback_on_step_end=lambda *a: a[3])
    assert freed == [True]
    assert runner._step.callback is None
    assert runner.vae.decode_calls == 0


def test_cancel_unwinds_without_decode_or_error_log_and_offload_is_restored(pipes, monkeypatch, caplog):
    text, runner = pipes
    monkeypatch.setattr(logging.getLogger("diffusers"), "propagate", True)
    caplog.set_level(logging.ERROR, logger="diffusers")
    # The loaded owner's real Accelerate hooks, on an explicit CPU device.
    text.enable_model_cpu_offload(device="cpu")
    owner_hooks = list(text._all_hooks)
    freed = _count_cleanup(monkeypatch, runner)
    seen = []

    def cancel(pipeline, step, timestep, values):
        seen.append(step)
        if step == 1:
            raise Cancelled
        return values

    with pytest.raises(Cancelled):
        _refine(runner, num_inference_steps=8, strength=0.5, callback_on_step_end=cancel)
    assert not any(
        record.levelno >= logging.ERROR and "Error in block" in record.getMessage()
        for record in caplog.records
    )
    assert seen == [0, 1]
    assert runner.vae.decode_calls == 0
    assert runner._step.callback is None
    assert freed == [True]
    # The backend's `_refine` frees the owner's hooks after every pass.
    text.maybe_free_model_hooks()
    assert len(text._all_hooks) == len(owner_hooks) == 4
    assert all(before is not after for before, after in zip(owner_hooks, text._all_hooks))

    # The next pass inherits neither the callback nor the half-run schedule.
    images = _refine(runner, num_inference_steps=8, strength=0.5)
    assert len(images) == 1
    assert seen == [0, 1]
    assert runner.num_timesteps == 4
    assert runner.vae.decode_calls == 1
