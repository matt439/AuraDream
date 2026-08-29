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
        ramp = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        retrieve_timesteps(self.scheduler, num_inference_steps, None, sigmas=ramp, mu=self.own_mu)
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


def test_a_scheduler_that_does_not_shift_gets_an_explicit_schedule():
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False, shift=1.0)
    backend, pipe = _backend(scheduler, own_mu=1.15)
    backend.generate(_request(mu=1.15), seed=1)

    assert pipe.sigmas_arg == _schedule(1.15)
    assert _applied(scheduler) == _schedule(1.15)


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
