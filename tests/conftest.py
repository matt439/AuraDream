"""A throwaway gallery - its own database file and outputs directory - and
the app on the fake backend (`fake`, `client`) for the HTTP tests."""
from __future__ import annotations

import asyncio
import dataclasses

import pytest
from fastapi.testclient import TestClient

from backend import main, prompt_encoding, storage
from backend.jobs import JobManager
from tests.api_helpers import ORIGIN
from tests.fakes import FakeBackend, FakeCritic, FakeTranslator, fake_tokenizer


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
