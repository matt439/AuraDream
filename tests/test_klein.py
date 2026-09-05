"""FLUX.2 [klein] through the same loader: the profile, the negative prompt, the VAE.

klein's ids contain "flux", so without their own entries the profile matcher
would seed 28 steps at CFG 3.5 for a model distilled to 4 steps that ignores
guidance. Its pipeline also takes no `negative_prompt` and puts the VAE tiling
switches on the VAE rather than on the pipeline.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest
from PIL import Image

from backend.backends import diffusers_backend as module
from backend.backends.diffusers_backend import DiffusersBackend
from backend.model_defaults import defaults_for, hints_for
from backend.schemas import GenerateRequest

KLEIN = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_BASE = "black-forest-labs/FLUX.2-klein-base-4B"
KLEIN_LITE = "lite-infer/flux.2-klein-9b-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"


@pytest.mark.parametrize(
    "model_id, steps, cfg",
    [(KLEIN, 4, 1.0), (KLEIN_LITE, 4, 1.0), (KLEIN_BASE, 50, 4.0)],
)
def test_klein_profiles_win_over_flux(model_id, steps, cfg):
    profile = defaults_for(model_id)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (steps, cfg, None)
    assert profile["negative_prompt_supported"] is False
    hints = hints_for(model_id)
    assert set(hints) == {"steps", "guidance_scale", "mu"}
    assert all(text and "{" not in text for text in hints.values())


def test_flux_one_is_unchanged():
    profile = defaults_for("black-forest-labs/FLUX.1-Krea-dev")
    assert (profile["steps"], profile["guidance_scale"]) == (28, 3.5)
    assert profile["negative_prompt_supported"] is True


class KleinLikePipeline:
    """A `__call__` with no `negative_prompt`, and a VAE that owns the tiling switches."""

    def __init__(self):
        self.calls: list[dict] = []
        self.vae = type("Vae", (), {"tiled": False, "sliced": False})()
        self.vae.enable_tiling = lambda: setattr(self.vae, "tiled", True)
        self.vae.enable_slicing = lambda: setattr(self.vae, "sliced", True)

    def to(self, device):
        return self

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
    ):
        self.calls.append(dict(prompt=prompt, guidance_scale=guidance_scale))
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


def test_a_negative_prompt_is_never_sent_to_a_pipeline_without_one():
    backend = DiffusersBackend()
    pipe = KleinLikePipeline()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    described = backend.describe()
    assert described["negative_prompt_supported"] is False
    # No true_cfg_scale and no negative_prompt: the classic `guidance_scale > 1`
    # gate, which is also where the distilled model starts warning.
    assert described["cfg_threshold"] == 1.0

    request = GenerateRequest(
        prompt="a lighthouse in fog", negative_prompt="blurry", width=256, height=256,
        steps=4, guidance_scale=4.0,
    )
    images = backend.generate(request, seed=1)
    assert len(images) == 1
    assert "negative_prompt" not in pipe.calls[0]


def test_load_reaches_for_the_vaes_own_tiling_switches(monkeypatch):
    import diffusers

    pipe = KleinLikePipeline()
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", lambda *a, **k: pipe)
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(module.settings, quantization="none", offload="none", device="cpu"),
    )
    backend = DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert pipe.vae.tiled and pipe.vae.sliced
    assert backend.describe()["negative_prompt_supported"] is False
