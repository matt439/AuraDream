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
    # No server is probed: the developer's own may well be up on 7860, and a
    # suite that refuses to run beside it is a suite nobody runs.
    monkeypatch.setattr(generate, "_server_holds_gpu", lambda host, port, timeout=2.0: None)
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


def test_a_seed_list_runs_the_grid_at_exactly_those_seeds(cli, store, capsys):
    code = generate.main(["a fox", "--seed-list", "4471,8823", "--grid-steps", "6,8", *SIZE, "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "seeds    : 4471, 8823  (listed)" in out and "2 listed seeds" in out
    assert generate.main(["a fox", "--seed-list", "4471,8823", "--grid-steps", "6", *SIZE, "-q"]) == 0
    assert cli.seeds == [4471, 8823]
    # Resume needs no --seed when the seeds are listed.
    assert generate.main(["a fox", "--seed-list", "4471,8823", "--grid-steps", "6", "--resume", *SIZE, "-q"]) == 0
    assert cli.seeds == [4471, 8823]  # nothing ran the second time
    with pytest.raises(SystemExit):
        generate.main(["a fox", "--seed-list", "1,2", "--grid-seeds", "3"])


def test_from_image_reads_the_pngs_own_record(cli, store, capsys, monkeypatch):
    # The PNG's metadata was write-only: starting from an image you already
    # have cost a browser trip and six numbers retyped by hand.
    monkeypatch.delenv("KREA_MODEL_ID", raising=False)
    assert generate.main([
        "a fox in fog", "-n", "blurry", "--seed", "7", "--steps", "6", "--cfg", "0.5",
        "--mu", "1.3", *SIZE, "-q",
    ]) == 0
    path = capsys.readouterr().out.strip().split("  seed ")[0]
    assert path.endswith(".png")

    assert generate.main(["--from-image", path, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "prompt   : a fox in fog" in out
    assert "negative : blurry" in out
    assert "size     : 256x256" in out
    assert "sampling : 6 steps, CFG 0.5, mu 1.3" in out
    assert "seed     : 7" in out
    assert f"from     : {path}" in out and "job " in out
    # The checkpoint travels too - a Turbo PNG must not render on FLUX while
    # claiming to reproduce - which is why the PNG is read before the flags
    # reach the environment.
    import os

    assert os.environ["KREA_MODEL_ID"] == store.settings.model_id
    # An explicit flag still wins over the record.
    assert generate.main(["--from-image", path, "--steps", "9", "--aspect", "16:9", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "sampling : 9 steps" in out and "size     : 1360x768" in out


def test_from_image_refuses_a_file_without_the_record(cli, tmp_path):
    from PIL import Image

    plain = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(plain)
    with pytest.raises(SystemExit) as exc:
        generate.main(["--from-image", str(plain), "--dry-run"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        generate.main(["--dry-run"])  # neither a prompt nor an image
    assert exc.value.code == 2


def test_a_server_holding_the_gpu_is_refused_with_its_own_code(cli, store, monkeypatch, capsys):
    # A second copy of the weights beside a server's resident copy is an OOM
    # after a multi-minute load. Advisory, after the dry-run guard, exit 4.
    monkeypatch.setattr(
        generate, "_server_holds_gpu", lambda host, port, timeout=2.0: "krea/Krea-2-Turbo (ready)"
    )
    assert generate.main(["a fox", *SIZE, "--dry-run"]) == 0  # a dry run touches nothing
    assert generate.main(["a fox", *SIZE, "-q"]) == 4
    assert "Krea-2-Turbo" in capsys.readouterr().err and cli.requests == []
    assert generate.main(["a fox", *SIZE, "-q", "--ignore-server"]) == 0
    assert len(cli.requests) == 1


def test_the_probe_tells_no_server_from_a_busy_one(monkeypatch):
    import urllib.error

    import urllib.request

    def refused(url, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    assert generate._server_holds_gpu("127.0.0.1", 7860) is None

    def slow(url, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", slow)
    assert "did not answer" in generate._server_holds_gpu("127.0.0.1", 7860)

    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    import json

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda url, timeout: Reply(json.dumps({"state": "unloaded", "model_id": "m"}).encode()),
    )
    assert generate._server_holds_gpu("0.0.0.0", 7860) is None
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda url, timeout: Reply(json.dumps({"state": "loading", "model_id": "m"}).encode()),
    )
    assert generate._server_holds_gpu("127.0.0.1", 7860) == "m (loading)"


def test_a_budget_stops_the_cli_between_cells(cli, store, capsys):
    cli.step_seconds = 0.02
    code = generate.main(
        ["a fox", "--seed", "1", "--steps", "5", "--grid-steps", "5,6,7,8", "--budget", str(0.05 / 60), *SIZE]
    )
    assert code == 130
    err = capsys.readouterr().err
    assert "time budget" in err and "--resume" in err
    assert 1 <= len(cli.requests) < 4
    assert store.list_images()[1] == len(cli.requests)


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
