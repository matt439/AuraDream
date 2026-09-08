"""A throwaway gallery: its own database file and its own outputs directory."""
from __future__ import annotations

import dataclasses

import pytest

from backend import storage


@pytest.fixture
def store(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    # `settings` is frozen, so point the module at a copy rather than mutating it.
    monkeypatch.setattr(
        storage,
        "settings",
        dataclasses.replace(storage.settings, db_path=tmp_path / "krea.db", outputs_dir=outputs),
    )
    storage.init_db()
    return storage


@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    """Nothing here downloads. `load()` reads a checkpoint's model_index.json
    ahead of the weights, so a test that stubs `from_pretrained` must not be
    let through to the Hub for that one file either; a test that needs an
    index replaces this stub with its own."""
    from backend.backends import diffusers_backend as module

    def refuse(model_id, filename):
        raise RuntimeError(f"the tests do not reach the Hub ({model_id}/{filename})")

    monkeypatch.setattr(module, "_hub_file", refuse)
