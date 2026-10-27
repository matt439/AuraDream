"""Shared helpers for the bundle and gallery-lock tests: save, export, a destination gallery, snapshots."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import subprocess
import sys
from contextlib import closing

from PIL import Image

from backend import bundles, storage
from backend.schemas import GenerateRequest


def _save(prompt="source", seed=1, **kwargs):
    variables = kwargs.pop("variables", None)
    return storage.save_image(
        Image.new("RGB", (256, 256), (seed % 256, 24, 80)),
        GenerateRequest(prompt=prompt, width=256, height=256, steps=28, **kwargs),
        seed=seed, duration_ms=1234 + seed, model_id="test/model",
        job_id="job-one", run_id="run-one", quantization="fp8", variables=variables,
    )


def _export():
    return bundles.export_bundle(expected=bundles.preview()["records"])


def _destination(store, monkeypatch, tmp_path):
    settings = dataclasses.replace(
        store.settings, db_path=tmp_path / "destination.db", outputs_dir=tmp_path / "destination",
    )
    monkeypatch.setattr(storage, "settings", settings)
    storage.init_db()
    return settings


def _snapshot(store):
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    files = {
        path.relative_to(store.settings.outputs_dir).as_posix(): path.read_bytes()
        for path in store.settings.outputs_dir.rglob("*") if path.is_file()
    }
    return rows, files


def _metadata(path):
    with Image.open(path) as image:
        return json.loads(image.info["auradream"])


def _locked_database(staged):
    raise sqlite3.OperationalError("database is locked")


# The child reserves the gallery, says so, and keeps it until its stdin has a
# line (or closes) - so the test, not a sleep, decides when it lets go.
_HOLDING_CHILD = """
from pathlib import Path
import sys
from backend.gallery_lock import hold
with hold(Path(sys.argv[1])):
    print('held', flush=True)
    sys.stdin.readline()
"""


def _holding_child(db_path):
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDING_CHILD, str(db_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert child.stdout.readline().strip() == "held"
    return child
