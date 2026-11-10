"""Known memory failures, through the real load and cell reporting paths.

All exceptions are constructed on the CPU; no allocation failure is induced.
"""
from __future__ import annotations

import errno
import logging

import pytest
import torch

from backend import main
from backend.error_text import generation_error, short_error
from tests.api_helpers import run_one, wait_for_status


def _failure(kind):
    if kind == "gpu":
        return torch.OutOfMemoryError("synthetic allocator detail " * 40)
    if kind == "ram":
        return MemoryError("synthetic RAM detail")
    if kind == "enomem":
        return OSError(errno.ENOMEM, "synthetic allocation detail")
    error = OSError("synthetic paging-file detail")
    error.winerror = 1455
    return error


@pytest.mark.parametrize("kind", ["gpu", "ram", "enomem", "pagefile"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_load_status_job_and_log_keep_the_right_memory_words(client, fake, monkeypatch, caplog, kind, wrapped):
    original = _failure(kind)
    error = OSError("a loader wrapper") if wrapped else original
    if wrapped:
        error.__cause__ = original

    def load():
        raise error

    caplog.set_level(logging.DEBUG, logger="auradream")
    fake.loaded = False
    monkeypatch.setattr(fake, "load", load)
    assert client.post("/api/model/load").status_code == 200
    status = wait_for_status(client, state="error")
    expected = generation_error(error, loading=True)
    assert status["detail"] == expected
    assert "checkpoint" in expected and "Models" in expected
    assert "synthetic" not in expected and "OSError" not in expected
    if kind == "gpu":
        assert "graphics card" in expected and "Sequential" in expected
        assert "smaller image" not in expected  # Size cannot fix a checkpoint load.
    else:
        assert "RAM or paging-file" in expected and "lighter checkpoint" in expected
    assert (status["detail_kind"], status["detail_url"]) == (None, None)
    _, events = run_one(client, steps=1)
    done = events[-1]
    assert (done["status"], done["error"], done["started_at"], done["failures"]) == (
        "error", expected, None, 0,
    )
    traces = [record for record in caplog.records if record.message == "load traceback"]
    assert traces and traces[0].exc_info[1] is error
    assert str(original) in caplog.text
    monkeypatch.setattr(fake, "load", lambda: setattr(fake, "loaded", True))
    client.post("/api/model/load")
    ready = wait_for_status(client, state="ready")
    assert (ready["detail"], ready["detail_kind"], ready["detail_url"]) == (None, None, None)


@pytest.mark.parametrize("failed_calls, expected_status", [({1}, "done"), ({1, 2, 3}, "error")])
def test_render_memory_errors_preserve_skip_and_stop_rules(client, fake, monkeypatch, caplog,
                                                          failed_calls, expected_status):
    original_generate = fake.generate
    raw = "synthetic private allocator detail " * 40
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls in failed_calls:
            raise torch.OutOfMemoryError(raw)
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(fake, "generate", generate)
    caplog.set_level(logging.DEBUG, logger="backend.jobs")
    _, events = run_one(client, steps=1, runs=3)
    done = events[-1]
    assert (done["status"], done["failures"], len(done["images"])) == (
        expected_status, len(failed_calls), 3 - len(failed_calls),
    )
    assert "graphics card ran out of memory" in done["error"]
    assert "smaller image" in done["error"] and "Sequential" in done["error"]
    assert "synthetic" not in done["error"]
    assert len(done["failed_cells"]) == len(failed_calls)
    assert all("graphics card" in cell["error"] for cell in done["failed_cells"])
    assert done["stopped_by"] == ("failures" if expected_status == "error" else None)
    assert any(record.exc_info and isinstance(record.exc_info[1], torch.OutOfMemoryError)
               for record in caplog.records)
    assert raw in caplog.text


def test_memory_classification_is_typed_and_cycle_bounded():
    class DerivedOOM(torch.OutOfMemoryError):
        pass

    assert "graphics card" in generation_error(DerivedOOM("details"))
    error = OSError("outer")
    error.__cause__ = MemoryError("inner")
    error.__cause__.__context__ = error
    assert "RAM or paging-file" in generation_error(error, loading=True)
    assert generation_error(error) == short_error(error)
    for unknown in [RuntimeError("CUDA out of memory"), OSError(errno.ENOSPC, "disk full"),
                    type("OutOfMemoryError", (Exception,), {})("other library")]:
        assert generation_error(unknown, loading=True) == short_error(unknown)
