"""A full disk under the CLI's own save loop (GEN#57).

Before the hold, `generate.py` waited only on `GalleryBusy`: `ENOSPC` or
SQLite's "database or disk is full" from `storage.save_image` reached the
outer handler, which threw the render away and ended the run with exit 1,
every later cell unrendered. Now the image is held in memory and the save
is tried again until there is room or a Ctrl-C; every other save error
still ends the run as it did.
"""
from __future__ import annotations

import errno
import sqlite3

import pytest

from backend import generate, storage

from tests.cli_helpers import (  # noqa: F401  (fixtures by name)
    SIZE,
    cli,
    isolated_execution_environment,
)

HELD = "the disk is full: this image is held in memory"


def _enospc() -> OSError:
    return OSError(errno.ENOSPC, "No space left on device")


def _sqlite_full() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("database or disk is full")


FULL = pytest.mark.parametrize("full", [_enospc, _sqlite_full], ids=["enospc", "sqlite-full"])


@pytest.fixture(autouse=True)
def no_poll_wait(monkeypatch):
    monkeypatch.setattr(generate, "DISK_FULL_POLL", 0.0, raising=False)


def _refusing(monkeypatch, error, calls, *, when):
    """`save_image` raising `error()` on each call whose 1-based number
    `when` accepts; `calls` counts them all."""
    real_save = storage.save_image

    def save(*args, **kwargs):
        calls.append(len(calls) + 1)
        if when(calls[-1]):
            raise error()
        return real_save(*args, **kwargs)

    monkeypatch.setattr(storage, "save_image", save)


@FULL
def test_a_full_disk_holds_the_image_until_there_is_room(cli, store, monkeypatch, capsys, full):
    # The second cell's save is refused three times, then the disk has room.
    calls: list[int] = []
    _refusing(monkeypatch, full, calls, when=lambda n: 2 <= n <= 4)
    assert generate.main(["a fox", "--seed", "7", "--runs", "3", *SIZE]) == 0
    # No cell rendered twice, none skipped, and the held image saved.
    assert len(cli.requests) == 3
    assert sorted(r.seed for r in store.list_images()[0]) == [7, 8, 9]
    assert len(calls) == 6
    err = capsys.readouterr().err
    # Said once however long the wait, and never as a failure.
    assert err.count(HELD) == 1
    assert "Ctrl-C stops without it" in err
    assert "failed" not in err and "OSError" not in err and "OperationalError" not in err


@FULL
def test_ctrl_c_ends_a_full_disk_wait_with_the_reason(cli, store, monkeypatch, capsys, full):
    # The disk never gets room; the user's first Ctrl-C arrives mid-wait.
    import signal

    calls: list[int] = []
    real_save = storage.save_image

    def save(*args, **kwargs):
        calls.append(len(calls) + 1)
        if calls[-1] == 1:
            return real_save(*args, **kwargs)
        if calls[-1] == 4:
            # main()'s own handler, called as the signal would call it.
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        raise full()

    monkeypatch.setattr(storage, "save_image", save)
    code = generate.main(["a fox", "--seed", "7", "--runs", "3", *SIZE])
    # A cancel's code, the first image kept, the held one gone, no third render.
    assert code == 130
    assert [r.seed for r in store.list_images()[0]] == [7]
    assert len(cli.requests) == 2
    captured = capsys.readouterr()
    assert captured.err.count(HELD) == 1
    assert "stopped with the disk full: run 2/3's image was not saved; --resume continues it" in captured.err
    assert "Forced stop" not in captured.err
    assert "1/3 image(s)" in captured.out


def test_a_batch_keeps_what_it_saved_before_the_refused_image(cli, store, monkeypatch, capsys):
    calls: list[int] = []
    _refusing(monkeypatch, _enospc, calls, when=lambda n: n in (2, 3))
    assert generate.main(["a fox", "--seed", "7", "--batch", "3", *SIZE, "-q"]) == 0
    assert len(cli.requests) == 1
    assert sorted(r.seed for r in store.list_images()[0]) == [7, 8, 9]
    assert capsys.readouterr().err.count(HELD) == 1


def test_a_then_refine_passes_save_is_held_the_same_way(cli, store, monkeypatch, capsys):
    # The base saves; its pass's save is refused twice, then lands.
    calls: list[int] = []
    _refusing(monkeypatch, _enospc, calls, when=lambda n: n in (2, 3))
    flags = ["--then-refine-scale", "1.5", "--then-refine-strength", "0.5"]
    assert generate.main(["a fox", "--seed", "7", "--steps", "8", *SIZE, *flags, "-q"]) == 0
    assert [r.refine is not None for r in cli.requests] == [False, True]
    rows = store.list_images()[0]
    assert sorted(r.refine is not None for r in rows) == [False, True]
    assert capsys.readouterr().err.count(HELD) == 1


def test_any_other_save_error_still_ends_the_run(cli, store, monkeypatch, capsys):
    calls: list[int] = []
    _refusing(
        monkeypatch, lambda: PermissionError(errno.EACCES, "Permission denied"), calls,
        when=lambda n: n == 2,
    )
    assert generate.main(["a fox", "--seed", "7", "--runs", "3", *SIZE, "-q"]) == 1
    assert len(calls) == 2 and len(cli.requests) == 2
    err = capsys.readouterr().err
    assert "PermissionError" in err and HELD not in err


def test_a_real_wait_polls_at_its_interval(cli, store, monkeypatch):
    # The wait sleeps in short slices so a Ctrl-C is heard inside it.
    slept: list[float] = []
    monkeypatch.setattr(generate, "DISK_FULL_POLL", 1.0)
    monkeypatch.setattr(generate.time, "sleep", slept.append)
    calls: list[int] = []
    _refusing(monkeypatch, _enospc, calls, when=lambda n: n == 1)
    assert generate.main(["a fox", "--seed", "7", *SIZE, "-q"]) == 0
    assert slept and max(slept) <= 0.25 and sum(slept) == pytest.approx(1.0)
