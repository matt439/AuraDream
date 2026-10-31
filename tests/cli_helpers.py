"""What the command-line tests share: the local `cli` fixture, its execution
environment isolation, and the gallery images and masks the passes start from.

`test_cli.py`, `test_cli_passes.py` and `test_cli_via_server.py` import these
by name, the autouse fixture included, so each file keeps the same isolation.
"""
from __future__ import annotations

import dataclasses
import os
import signal

import pytest

import backend.backends as backends
from backend import config, generate

from tests.fakes import FakeBackend


@pytest.fixture(autouse=True)
def isolated_execution_environment(monkeypatch):
    # apply_env writes directly; record original absence before each test so
    # new explicit flags cannot leak into another local or remote invocation.
    for name in ("AURADREAM_COMPILATION", "AURADREAM_ATTENTION", "AURADREAM_LOADER_EXECUTION_ERROR"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cli(store, monkeypatch):
    fake = FakeBackend()
    # _project() imports get_backend and settings at call time, so these are
    # what main() sees: the fake, and the throwaway gallery's paths rather than
    # the real auradream.db in the checkout.
    monkeypatch.setattr(backends, "get_backend", lambda name="diffusers": fake)
    monkeypatch.setattr(config, "settings", store.settings)
    # No server is probed: the developer's own may well be up on 7860, and a
    # suite that refuses to run beside it is a suite nobody runs.
    monkeypatch.setattr(generate, "_server_holds_gpu", lambda host, port, timeout=2.0: None)
    handler = signal.getsignal(signal.SIGINT)
    yield fake
    # main() installs a SIGINT handler and never removes it.
    signal.signal(signal.SIGINT, handler)


SIZE = ["--width", "256", "--height", "256"]


def project_claiming(module, model_id):
    """`_project()` with the settings claiming another checkpoint.

    The reference check reads the profile for `settings.model_id`, and the
    profile is what says whether the pipeline takes an `image` at all.
    """
    original = module._project

    def patched():
        project = original()
        project.settings = dataclasses.replace(project.settings, model_id=model_id)
        return project

    return patched


def to_refine(store):
    """A gallery image to refine: a 2:3 portrait at this checkpoint's id."""
    from PIL import Image

    from backend.schemas import GenerateRequest

    return store.save_image(
        Image.new("RGB", (256, 384)),
        GenerateRequest(prompt="a fox", width=256, height=384),
        seed=5, duration_ms=10, model_id=store.settings.model_id,
    )


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


def mask_file(tmp_path, size=(256, 384), box=(0, 0, 128, 64), name="mask.png"):
    from PIL import Image

    image = Image.new("L", size, 0)
    image.paste(255, box)
    path = tmp_path / name
    image.convert("RGB").save(path)  # any mode: kept as one channel
    return path, image


def rewrite_record(path, change) -> None:
    import json

    from PIL import Image, PngImagePlugin

    with Image.open(path) as image:
        image.load()
        meta = json.loads(image.info["auradream"])
        change(meta)
        info = PngImagePlugin.PngInfo()
        info.add_text("auradream", json.dumps(meta))
        image.save(path, format="PNG", pnginfo=info)
