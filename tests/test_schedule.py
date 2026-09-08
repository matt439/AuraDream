"""What matters is the schedule the *scheduler* ends up with, not the list the
backend computes: `sigmas` is an input to the shift, not the shift's result."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from backend.backends.diffusers_backend import DiffusersBackend
from backend.schemas import GenerateRequest

STEPS = 8


class StubPipeline:
    """The timestep half of Krea2Pipeline/FluxPipeline, and nothing else.

    Both build the default ramp, derive a mu of their own and hand both to
    `retrieve_timesteps`, which is where the shift is actually applied.
    """

    def __init__(self, scheduler, own_mu: float = 1.15):
        self.scheduler = scheduler
        self.own_mu = own_mu
        self.sigmas_arg = None
        self.generators = None
        self.steps_run = 0

    def __call__(
        self,
        prompt,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        callback_on_step_end,
        sigmas=None,
        negative_prompt=None,
    ):
        from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps

        self.sigmas_arg = sigmas
        self.generators = generator
        ramp = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        retrieve_timesteps(self.scheduler, num_inference_steps, None, sigmas=ramp, mu=self.own_mu)
        # The denoising loop's one hand-back, as the real pipelines do it - so
        # the backend's progress, cancel and per-image generator code is run by
        # a test rather than only read.
        for step in range(num_inference_steps):
            self.steps_run += 1
            callback_on_step_end(self, step, None, {})
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


def _backend(scheduler, own_mu=1.15):
    backend = DiffusersBackend()
    pipe = StubPipeline(scheduler, own_mu)
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend, pipe


def _request(**kwargs):
    base = dict(prompt="a lighthouse in fog", width=256, height=256, steps=STEPS)
    return GenerateRequest(**{**base, **kwargs})


def _applied(scheduler) -> list[float]:
    # set_timesteps appends the terminal 0; the schedule proper is what precedes
    # it. The scheduler works in float32, hence the tolerance on every compare.
    return [float(s) for s in scheduler.sigmas[:-1]]


def _schedule(mu: float):
    return pytest.approx(DiffusersBackend._shifted_sigmas(STEPS, mu), abs=1e-6)


def test_requested_mu_is_the_shift_that_lands():
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, _ = _backend(scheduler, own_mu=1.15)
    backend.generate(_request(mu=1.15), seed=1)

    assert _applied(scheduler) == _schedule(1.15)
    # The double-shifted schedule this used to produce, for contrast.
    assert _applied(scheduler) != _schedule(2.30)


@pytest.mark.parametrize("mu", [0.0, 0.5, 3.0])
def test_a_mu_other_than_the_pipelines_own_is_honoured_exactly(mu):
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, pipe = _backend(scheduler, own_mu=1.15)
    backend.generate(_request(mu=mu), seed=1)

    assert pipe.sigmas_arg is None  # the pipeline's own ramp, shifted once
    assert _applied(scheduler) == _schedule(mu)


def test_no_mu_leaves_the_pipeline_alone():
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, pipe = _backend(scheduler, own_mu=0.7)
    backend.generate(_request(mu=None), seed=1)

    assert pipe.sigmas_arg is None
    assert _applied(scheduler) == _schedule(0.7)


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_a_scheduler_that_does_not_shift_dynamically_still_lands_on_mu(shift):
    # `use_dynamic_shifting=False` is not "no shift": the else branch applies
    # `shift` statically to whatever sigmas it is handed, and shifts compose.
    # At shift=1.0 that is a no-op, which is exactly what hid the bug; SD3
    # ships 3.0. Asserted on the scheduler's sigmas, never on the list the
    # backend computed - which is the input to the shift, not its result.
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False, shift=shift)
    backend, pipe = _backend(scheduler, own_mu=1.15)
    backend.generate(_request(mu=1.15), seed=1)

    assert pipe.sigmas_arg is not None  # the explicit-schedule route was taken
    assert _applied(scheduler) == _schedule(1.15)
    assert _applied(scheduler) != _schedule(1.15 + math.log(shift)) or shift == 1.0


def test_one_generator_per_image_seeded_seed_plus_i():
    # The headline batch invariant: image i of a batch is drawn from the seed
    # that gets written next to it, wrapped by the same helper both save paths
    # use. A base near MAX_SEED pins the list-not-scalar shape, the offset and
    # the wrap in one assertion.
    from backend.seeds import MAX_SEED, wrap_seed

    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, pipe = _backend(scheduler)
    backend.generate(_request(num_images=4), seed=MAX_SEED - 1)
    assert [g.initial_seed() for g in pipe.generators] == [
        wrap_seed(MAX_SEED - 1 + i) for i in range(4)
    ]


def test_progress_is_reported_and_a_cancel_lands_within_a_step():
    from backend.backends import Cancelled

    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, pipe = _backend(scheduler)
    seen = []
    backend.generate(_request(), seed=1, on_progress=lambda step, total: seen.append((step, total)))
    assert seen == [(i + 1, STEPS) for i in range(STEPS)]

    backend, pipe = _backend(scheduler)
    with pytest.raises(Cancelled):
        backend.generate(_request(), seed=1, should_cancel=lambda: pipe.steps_run >= 3)
    # Polled at the step boundary: the third step finished, the fourth never ran.
    assert pipe.steps_run == 3


class PreviewingPipeline(StubPipeline):
    """FLUX's shape for the preview: packed latents in the callback, a 4-argument
    static `_unpack_latents`, and a VAE the factor table knows."""

    vae_scale_factor = 8

    class _Vae:
        class config:
            latent_channels = 16

    # The table is keyed on the class *name*, which is what the stand-in shares.
    _Vae.__name__ = "AutoencoderKL"
    vae = _Vae()

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

        return FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, sigmas=None,
                 negative_prompt=None):
        import torch
        from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps

        ramp = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        retrieve_timesteps(self.scheduler, num_inference_steps, None, sigmas=ramp, mu=self.own_mu)
        # (batch, patches, channels * 4): a 256x256 image is 16x16 patches of 64.
        latents = torch.randn(1, (height // 16) * (width // 16), 64, generator=torch.Generator().manual_seed(0))
        for step in range(num_inference_steps):
            self.steps_run += 1
            callback_on_step_end(self, step, None, {"latents": latents})
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


def test_a_preview_is_drawn_from_the_callbacks_latents(monkeypatch):
    from backend import latent_preview

    monkeypatch.setattr(latent_preview, "INTERVAL_SECONDS", 0.0)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, pipe = _backend(scheduler)
    frames = []
    backend.generate(_request(), seed=1, on_preview=lambda jpeg, step, total: frames.append((step, total, jpeg)))
    # Nothing at all from the plain stub, whose callback carries no latents:
    # the preview switched itself off and the generation ran to the end.
    assert frames == [] and pipe.steps_run == STEPS

    backend, pipe = DiffusersBackend(), PreviewingPipeline(scheduler)
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    backend.generate(_request(), seed=1, on_preview=lambda jpeg, step, total: frames.append((step, total, jpeg)))
    assert frames, "no preview from a pipeline that hands the callback latents"
    steps, totals, jpegs = zip(*frames)
    # Never after the last step - the real image follows it - and never
    # while the latent is still noise (the first steps of a shifted schedule).
    assert max(steps) < STEPS and min(steps) > 1 and set(totals) == {STEPS}
    assert all(j[:3] == b"\xff\xd8\xff" for j in jpegs)  # JPEG magic
    with Image.open(__import__("io").BytesIO(jpegs[-1])) as frame:
        assert frame.size == (32 * latent_preview.UPSCALE, 32 * latent_preview.UPSCALE)


def test_a_broken_preview_costs_a_picture_not_the_cell(monkeypatch):
    from backend import latent_preview

    monkeypatch.setattr(latent_preview, "INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(latent_preview, "render", lambda *a, **k: 1 / 0)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, pipe = DiffusersBackend(), PreviewingPipeline(scheduler)
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    frames = []
    images = backend.generate(_request(), seed=1, on_preview=lambda *a: frames.append(a))
    # The exception out of the callback is exactly how Cancelled ends a job;
    # a preview's must not.
    assert len(images) == 1 and pipe.steps_run == STEPS and frames == []


def test_applied_reports_what_the_pipeline_will_run():
    # A negative prompt at or below the pipeline's threshold is dropped, above
    # it kept, and mu reaches the scheduler either way.
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backend, _ = _backend(scheduler)
    threshold = backend.cfg_threshold()
    asked = _request(negative_prompt="blurry", guidance_scale=threshold, mu=1.15)
    ran = backend.applied(asked)
    assert (ran.negative_prompt, ran.mu) == (None, 1.15)
    above = _request(negative_prompt="blurry", guidance_scale=threshold + 0.5)
    assert backend.applied(above).negative_prompt == "blurry"
    # Unloaded, the checkpoint profile's guess stands in.
    assert DiffusersBackend().applied(asked).prompt == asked.prompt


def test_the_formula_matches_the_schedulers_own():
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    ramp = np.linspace(1.0, 1 / STEPS, STEPS)
    theirs = scheduler.time_shift(1.15, 1.0, ramp)
    ours = DiffusersBackend._shifted_sigmas(STEPS, 1.15)
    assert list(theirs) == pytest.approx(ours)
    # And that the shifts compose additively, which is why passing a pre-shifted
    # schedule to a shifting scheduler produced a mu of 2.30.
    assert list(scheduler.time_shift(1.15, 1.0, np.array(ours))) == pytest.approx(
        DiffusersBackend._shifted_sigmas(STEPS, 2.30)
    )
    assert math.isclose(1.15 + 1.15, 2.30)
