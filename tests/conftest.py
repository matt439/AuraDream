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
