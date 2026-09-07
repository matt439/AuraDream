"""The headless path, end to end with a backend that draws nothing.

It is the other writer into the same gallery, and CLAUDE.md's reason for its
duplicated sweep loop ends with a promise - a cell in the browser and a cell
in the terminal cannot drift apart - kept until now by shared code alone.
"""
from __future__ import annotations

import signal

import pytest

import backend.backends as backends
from backend import config, generate

from tests.test_api import FakeBackend


@pytest.fixture
def cli(store, monkeypatch):
    fake = FakeBackend()
    # _project() imports get_backend and settings at call time, so these are
    # what main() sees: the fake, and the throwaway gallery's paths rather than
    # the real krea.db in the checkout.
    monkeypatch.setattr(backends, "get_backend", lambda name="diffusers": fake)
    monkeypatch.setattr(config, "settings", store.settings)
    handler = signal.getsignal(signal.SIGINT)
    yield fake
    # main() installs a SIGINT handler and never removes it.
    signal.signal(signal.SIGINT, handler)


SIZE = ["--width", "256", "--height", "256"]


def test_a_dry_run_loads_and_writes_nothing(cli, store, capsys):
    assert generate.main(["a fox", "--dry-run", "--seed", "7", "--runs", "3", *SIZE]) == 0
    assert cli.loads == 0 and cli.requests == []
    assert store.list_images()[1] == 0
    out, err = capsys.readouterr()
    assert "seed     : 7" in out and "run      : " in out
    assert "Dry run" in err  # stderr, so `--quiet > paths.txt` stays paths


def test_a_dry_run_with_resume_creates_no_database(cli, store):
    # The resume lookup connects, and connecting writes the schema into any
    # file it opens; a dry run against a gallery that does not exist yet used
    # to leave a 40 KB krea.db behind.
    store.settings.db_path.unlink()
    code = generate.main(["a fox", "--dry-run", "--resume", "--seed", "1", "--grid-steps", "6,8", *SIZE])
    assert code == 0
    assert not store.settings.db_path.exists()


def test_a_cli_sweep_writes_the_seeds_the_server_would(cli, store):
    # --runs 3 --batch 2 --seed 7: cells at 7, 9, 11 and images 7..12 - the
    # same six the HTTP path's test asserts for the same request.
    assert generate.main(["a fox", "--seed", "7", "--runs", "3", "--batch", "2", *SIZE, "-q"]) == 0
    assert cli.seeds == [7, 9, 11]
    items, total = store.list_images()
    assert sorted(r.seed for r in items) == [7, 8, 9, 10, 11, 12]
    assert len({r.job_id for r in items}) == 1
    assert len({r.run_id for r in items}) == 1 and items[0].run_id


def test_a_failed_cell_is_skipped_and_the_exit_code_says_so(cli, store):
    cli.fail_on = {2}
    code = generate.main(["a fox", "--seed", "1", "--grid-steps", "6,8,10", *SIZE, "-q"])
    assert code == 1
    assert [r.steps for r in cli.requests] == [6, 8, 10]
    assert sorted(r.steps for r in store.list_images()[0]) == [6, 10]


def test_resume_skips_what_the_gallery_holds(cli, store):
    assert generate.main(["a fox", "--seed", "3", "--grid-steps", "6", *SIZE, "-q"]) == 0
    code = generate.main(
        ["a fox", "--seed", "3", "--grid-steps", "6,8", "--resume", *SIZE, "-q"]
    )
    assert code == 0
    assert [r.steps for r in cli.requests] == [6, 8]


def test_mu_and_no_mu_are_exclusive(cli):
    with pytest.raises(SystemExit) as exc:
        generate.main(["a fox", "--mu", "1.15", "--no-mu"])
    assert exc.value.code == 2


def test_a_bad_axis_prints_one_line(cli, capsys):
    assert generate.main(["a fox", "--grid-steps", "90-120x10", *SIZE]) == 2
    err = capsys.readouterr().err.strip()
    assert err.startswith("steps:") and "\n" not in err


def test_durations_have_an_hours_branch():
    assert generate._duration(45) == "45s"
    assert generate._duration(150) == "2m30s"
    assert generate._duration(4 * 3600 + 12 * 60 + 30) == "4h12m"
