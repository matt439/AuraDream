"""The live preview's map for SDXL's VAE and its non-flow sigma rule (`GEN#6`).

The map itself was fitted on the CPU from a real VAE
(`tools/fit_latent_preview.py`; docs/verified.md has the record) and cannot be
refitted here; what is held is that it is the one `factors_for` picks for
SDXL's VAE, and that a latent from SDXL's Euler schedulers reaches `render` in
the flow form the gain, threshold and blur were written for, with the
schedulers' own `add_noise` as the judge of what their latents are.
"""
from __future__ import annotations

import inspect
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image

from backend import latent_preview
from backend.backends.diffusers_backend import DiffusersBackend
from backend.schemas import GenerateRequest

STEPS = 28
# SDXL's shipped scheduler settings (Animagine, Kohaku, NoobAI's base config).
_SDXL = dict(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
             timestep_spacing="leading", steps_offset=1)


def _vae(channels: int):
    # Keyed on the class *name*, which the stand-in shares with diffusers'.
    vae_type = type("AutoencoderKL", (), {})
    vae = vae_type()
    vae.config = SimpleNamespace(latent_channels=channels)
    return vae


def test_sdxl_has_a_four_channel_map_and_flux_keeps_its_own():
    factors = latent_preview.factors_for(_vae(4))
    assert factors is latent_preview.FACTORS["AutoencoderKL/4"]
    assert len(factors.matrix) == 4 and all(len(row) == 3 for row in factors.matrix)
    assert factors.downsample == 8
    # Three colours from four channels: a map that loses one would draw grey.
    assert np.linalg.matrix_rank(np.array(factors.matrix)) == 3
    assert latent_preview.factors_for(_vae(16)) is latent_preview.FACTORS["AutoencoderKL/16"]


def test_a_flow_scheduler_and_one_unknown_pass_unchanged():
    latent = torch.ones(4, 2, 2)
    for scheduler in (FlowMatchEulerDiscreteScheduler(), SimpleNamespace(sigmas=[1.0, 0.0]), None):
        out, sigma = latent_preview.flow_view(scheduler, latent, 0.4)
        assert out is latent and sigma == 0.4


@pytest.mark.parametrize("scheduler_type", [
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
])
def test_a_non_flow_latent_becomes_the_flow_form(scheduler_type):
    """Whatever the scheduler holds at a sigma (its own `add_noise`), the view
    is exactly `(1 - s) x_0 + s eps`, the form `render`'s gain undoes."""
    scheduler = scheduler_type(**_SDXL)
    scheduler.set_timesteps(STEPS)
    generator = torch.Generator().manual_seed(0)
    x0 = torch.randn(1, 4, 8, 8, generator=generator)
    eps = torch.randn(1, 4, 8, 8, generator=generator)
    for index in (0, 5, 14, STEPS - 1):
        timestep = scheduler.timesteps[index:index + 1]
        sigma = float(scheduler.sigmas[index])
        noisy = scheduler.add_noise(x0, eps, timestep)
        view, s = latent_preview.flow_view(scheduler, noisy, sigma)
        assert s == pytest.approx(sigma / (1 + sigma))
        assert view == pytest.approx((1 - s) * x0 + s * eps, abs=1e-4)


def test_the_flow_threshold_is_sigma_nine():
    """Flow's 0.9 is sigma 9: the zero-terminal-SNR schedule NoobAI runs
    draws nothing for its first steps, and the terminal 0 is the plain latent."""
    for sigma, drawn in ((9.5, False), (8.9, True), (4096.0, False), (0.5, True)):
        s = latent_preview.flow_view(EulerDiscreteScheduler(**_SDXL), torch.zeros(1), sigma)[1]
        assert (s <= latent_preview.MAX_SIGMA) is drawn, sigma
    scheduler = EulerDiscreteScheduler(**_SDXL, prediction_type="v_prediction",
                                       rescale_betas_zero_snr=True)
    scheduler.set_timesteps(STEPS)
    drawn = [latent_preview.flow_view(scheduler, torch.zeros(1), float(s))[1] <= latent_preview.MAX_SIGMA
             for s in scheduler.sigmas]
    assert drawn[:4] == [False] * 4 and all(drawn[6:])
    scheduler = EulerDiscreteScheduler(**_SDXL)
    latent = torch.ones(1)
    out, s = latent_preview.flow_view(scheduler, latent, 0.0)
    assert s == 0.0 and torch.equal(out, latent)


class SdxlPipeline:
    """StableDiffusionXLPipeline's shape for the preview: unpacked 4-channel
    latents in the callback, no `_unpack_latents`, an Euler scheduler holding
    `x_0 + sigma * eps` after each step."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.vae = _vae(4)
        self.steps_run = 0

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, negative_prompt=None):
        self.scheduler.set_timesteps(num_inference_steps)
        seed = torch.Generator().manual_seed(0)
        x0 = torch.randn(1, 4, height // 8, width // 8, generator=seed)
        eps = torch.randn(1, 4, height // 8, width // 8, generator=seed)
        for step in range(num_inference_steps):
            self.steps_run += 1
            sigma = self.scheduler.sigmas[step + 1]
            callback_on_step_end(self, step, None, {"latents": x0 + sigma * eps})
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


def _sdxl_backend(scheduler):
    backend, pipe = DiffusersBackend(), SdxlPipeline(scheduler)
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend, pipe


def test_an_sdxl_render_draws_frames_once_its_noise_allows(monkeypatch):
    monkeypatch.setattr(latent_preview, "INTERVAL_SECONDS", 0.0)
    seen: list[float] = []
    real_render = latent_preview.render

    def render(latent, factors, sigma=0.0):
        seen.append(sigma)
        return real_render(latent, factors, sigma)

    monkeypatch.setattr(latent_preview, "render", render)
    backend, pipe = _sdxl_backend(EulerDiscreteScheduler(**_SDXL))
    frames = []
    request = GenerateRequest(prompt="a lighthouse in fog", width=256, height=256, steps=STEPS)
    images = backend.generate(request, seed=1, on_preview=lambda jpeg, step, total: frames.append((step, jpeg)))
    assert len(images) == 1 and pipe.steps_run == STEPS
    # Every sigma `render` was handed is the flow one, below 1.
    assert seen and all(0.0 <= s < 1.0 for s in seen)
    steps = [step for step, _ in frames]
    # Never the last step (the image follows it); the first step lands at
    # sigma 8.85, just inside flow's 0.9.
    assert steps == list(range(1, STEPS))
    with Image.open(io.BytesIO(frames[-1][1])) as frame:
        assert frame.size == (32 * latent_preview.UPSCALE, 32 * latent_preview.UPSCALE)


def test_an_sdxl_refine_reads_its_sigma_where_the_pass_is(monkeypatch):
    """The img2img pipeline sets `begin_index`; the Euler sigma read there is
    turned into its flow value."""
    seen: list[float] = []
    monkeypatch.setattr(latent_preview, "unpack", lambda pipe, latents, height, width: latents)
    monkeypatch.setattr(latent_preview, "render", lambda latent, factors, sigma: seen.append(sigma) or b"jpeg")
    scheduler = EulerDiscreteScheduler(**_SDXL)
    scheduler.set_timesteps(STEPS)
    scheduler.set_begin_index(17)
    pipe = SimpleNamespace(scheduler=scheduler, vae=_vae(4))
    request = GenerateRequest(prompt="x", width=256, height=256, steps=STEPS)
    assert DiffusersBackend()._preview(pipe, torch.zeros(4, 32, 32), request, step=2) == b"jpeg"
    sigma = float(scheduler.sigmas[17 + 2 + 1])
    assert seen == [pytest.approx(sigma / (1 + sigma))]
