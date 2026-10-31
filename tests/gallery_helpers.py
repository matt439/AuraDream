"""What the gallery read tests share: a guard that a query stays cold, and a
snapshot of every row and PNG to show a read changed neither.

Imported by name (the fixture too) by the search tests and More seeds'
headline tests.
"""
from __future__ import annotations

import hashlib

import pytest

from backend import main


@pytest.fixture
def cold_queries_only(client, monkeypatch):
    """Fail a gallery read that loads the model, generates or queues.

    Yields the attempts list, which a test may also assert empty itself;
    after the test, the queue must still be empty.
    """
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("model-or-queue")
        pytest.fail("a gallery query entered model loading, generation or queue submission")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.backend, "load", forbidden)
    monkeypatch.setattr(main.backend, "generate", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    monkeypatch.setattr(main.jobs, "submit", forbidden)
    yield attempts
    assert attempts == []
    assert client.get("/api/jobs").json() == []


def gallery_state(store):
    """Every images row, in id order, and each PNG's sha256 by file name."""
    with store._session() as connection:
        rows = [tuple(row) for row in connection.execute("SELECT * FROM images ORDER BY id").fetchall()]
    pngs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in store.settings.outputs_dir.glob("*.png")}
    return rows, pngs
