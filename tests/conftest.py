"""A throwaway gallery - its own database file and outputs directory - and
the app on the fake backend (`fake`, `client`) for the HTTP tests."""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys

import pytest
from fastapi.testclient import TestClient

# The suite runs on config.py's defaults, whatever the shell exports. `settings`
# reads the environment once, when `backend.config` is first imported - by the
# imports below - so this has to happen here, not in a fixture: `wsl/env.sh`
# exports Nunchaku and a lite-infer checkpoint, and tests written against the
# default checkpoint and fp8 failed under it. Where the gallery and the adapters
# live stays the shell's: the tests redirect those themselves, and one that
# forgot should reach this environment's gallery, not the one under /mnt/c.
assert "backend.config" not in sys.modules, "backend.config was imported before the tests' environment was set"
_KEPT = {"LOCAL_DIFFUSION_OUTPUTS_DIR", "LOCAL_DIFFUSION_DB_PATH", "LOCAL_DIFFUSION_LORAS_DIR"}
for _name in [name for name in os.environ if name.startswith("LOCAL_DIFFUSION_") and name not in _KEPT]:
    del os.environ[_name]

from backend import main, prompt_encoding, storage  # noqa: E402
from backend.jobs import JobManager  # noqa: E402
from tests.api_helpers import ORIGIN  # noqa: E402
from tests.fakes import FakeBackend, FakeCritic, FakeTranslator, fake_tokenizer  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    # `settings` is frozen, so point the module at a copy rather than mutating it.
    monkeypatch.setattr(
        storage,
        "settings",
        dataclasses.replace(storage.settings, db_path=tmp_path / "local-diffusion.db", outputs_dir=outputs),
    )
    storage.init_db()
    return storage


@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    """Nothing here downloads. `load()` reads a checkpoint's model_index.json
    ahead of the weights, so a test that stubs `from_pretrained` must not be
    let through to the Hub for that one file either; a test that needs an
    index replaces this stub with its own."""
    from backend.backends import loading

    def refuse(model_id, filename):
        raise RuntimeError(f"the tests do not reach the Hub ({model_id}/{filename})")

    monkeypatch.setattr(loading, "hub_file", refuse)
    # Single-file conversion has its own downloads. Tests exercising that
    # path replace these too; a missed stub must never pull real weights.
    import huggingface_hub

    def refuse_download(*args, **kwargs):
        raise RuntimeError("the tests do not download model files")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse_download)


@pytest.fixture
def fake(store, monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(
        main.settings, db_path=store.settings.db_path,
    ))
    monkeypatch.setattr(main, "_startup_preload", None)
    monkeypatch.setattr(main.model_selection, "_selection", (None, 0))
    backend = FakeBackend()
    translator = FakeTranslator()
    backend.translator = translator
    critic = FakeCritic()
    backend.critic = critic
    monkeypatch.setattr(main, "backend", backend)
    monkeypatch.setattr(main, "prompt_translator", translator)
    monkeypatch.setattr(main, "prompt_critic", critic)
    monkeypatch.setattr(main, "_load_state", {"state": "unloaded", "detail": None})
    monkeypatch.setattr(main, "_load_lock", asyncio.Lock())
    # A fresh manager per test: shutting the app down drains its one worker
    # thread for good. Wired the way main.py wires it, so the single load path
    # the whole app relies on is the one under test.
    monkeypatch.setattr(main, "jobs", JobManager(backend, main.ensure_loaded))
    # The analyser's tokenizers without the Hub, and a cache that starts and
    # ends empty: it is module state, and another test file reads it too.
    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", fake_tokenizer)
    prompt_encoding.unload()
    yield backend
    prompt_encoding.unload()


@pytest.fixture
def client(fake):
    # base_url has to be a host the app answers to - which is the point of the
    # middleware, and why TestClient's default "testserver" would 400.
    with TestClient(main.app, base_url=ORIGIN) as c:
        yield c
