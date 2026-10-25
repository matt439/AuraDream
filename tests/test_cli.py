"""The headless path, end to end with a backend that draws nothing.

It is the other writer into the same gallery, and the design notes' reason for
its duplicated sweep loop ends with a promise - a cell in the browser and a cell
in the terminal cannot drift apart - kept until now by shared code alone.
"""
from __future__ import annotations

import dataclasses
import io
import signal
import sys

import pytest

import backend.backends as backends
from backend import config, gallery_lock, generate, storage

from tests.fakes import FakeBackend


@pytest.fixture
def cli(store, monkeypatch):
    fake = FakeBackend()
    # _project() imports get_backend and settings at call time, so these are
    # what main() sees: the fake, and the throwaway gallery's paths rather than
    # the real local-diffusion.db in the checkout.
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


def test_random_prompt_cli_draws_per_image_and_saves_the_same_choices_as_grid(cli, store, capsys):
    from backend import grid
    from backend.schemas import GenerateRequest

    flags = [
        "a {size} fox", "--grid-prompt", "size = small | large | none", "--grid-prompt-mode", "random",
        "--grid-steps", "6,8", "--grid-seeds", "2", "--batch", "2", "--seed", "42", *SIZE,
    ]
    assert generate.main([*flags, "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "output   : 8 image(s) - 8 run(s) x 1" in output
    assert "one choice per image" in output and "run separately" in output
    assert cli.requests == []
    assert generate.main([*flags, "-q"]) == 0
    expected = grid.expand(
        GenerateRequest(prompt="a {size} fox", width=256, height=256, seed=42, num_images=2),
        prompt="size = small | large | none", prompt_mode="random", steps="6,8", seeds=2,
    )
    assert [(request.prompt, seed) for request, seed in zip(cli.requests, cli.seeds)] == [
        (cell.request.prompt, cell.seed) for cell in expected
    ]
    rows = sorted(store.list_images()[0], key=lambda row: row.id)
    assert [(row.seed, row.variables, row.batch) for row in rows] == [
        (cell.seed, cell.variables, 1) for cell in expected
    ]
    assert generate.main([*flags, "--resume", "-q"]) == 0
    assert len(cli.requests) == 8


def test_random_prompt_mode_requires_a_variable_block_in_the_cli(cli, capsys):
    assert generate.main(["a fox", "--grid-prompt-mode", "random", "--dry-run", *SIZE]) == 2
    assert "needs prompt variables" in capsys.readouterr().err
    assert cli.requests == []


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    """An adapter directory with two files, so a flag can win over a PNG."""
    from backend import loras

    directory = tmp_path / "loras"
    directory.mkdir()
    for name in ("style.safetensors", "other.safetensors"):
        (directory / name).write_bytes(b"weights")
    monkeypatch.setattr(
        loras, "settings", dataclasses.replace(loras.settings, loras_dir=directory)
    )
    return directory


def test_a_dry_run_loads_and_writes_nothing(cli, store, capsys):
    assert generate.main(["a fox", "--dry-run", "--seed", "7", "--runs", "3", *SIZE]) == 0
    assert cli.loads == 0 and cli.requests == []
    assert store.list_images()[1] == 0
    out, err = capsys.readouterr()
    assert "seed     : 7" in out and "run      : " in out
    assert "Dry run" in err  # stderr, so `--quiet > paths.txt` stays paths


def test_the_header_writes_utf8_where_its_output_is_redirected(cli, store, monkeypatch):
    # `> night.log` on Windows: stdout is the locale's code page, strict, and a
    # CJK or emoji prompt was a UnicodeEncodeError at the header, before
    # anything ran.
    redirected = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(redirected, encoding="cp1252"))
    assert generate.main(["一只狐狸 🦊", "--dry-run", "--seed", "7", *SIZE]) == 0
    sys.stdout.flush()
    assert "prompt   : 一只狐狸 🦊" in redirected.getvalue().decode("utf-8")


class _Nul(io.BytesIO):
    """`> NUL` on Windows: a character device, so isatty() answers True."""

    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize("name", ["generate", "stats", "clear", "reindex"])
def test_every_cli_sets_up_its_stdout_the_same_way(name, monkeypatch):
    # Four copies of one block at the top of main(), held to one behaviour
    # here; `--help` runs it and touches nothing else.
    import importlib

    main = importlib.import_module(f"backend.{name}").main

    def helped(stream):
        monkeypatch.setattr(sys, "stdout", stream)
        with pytest.raises(SystemExit) as stopped:
            main(["--help"])
        assert stopped.value.code == 0
        return sys.stdout

    # pythonw has no stdout, and printing to None is a no-op: the setup must
    # not be what fails a scheduled `clear --yes`.
    helped(None)
    # A StringIO holds text, with no encoding to change.
    assert "usage:" in helped(io.StringIO()).getvalue()
    # Redirected to a file or a pipe: UTF-8, whatever the code page.
    assert helped(io.TextIOWrapper(io.BytesIO(), encoding="cp1252")).encoding == "utf-8"
    # `> NUL` answers isatty() and still encodes with the code page, so on
    # Windows it is UTF-8 too - the console itself is UTF-8 already. Elsewhere
    # a terminal keeps its own encoding.
    nul = helped(io.TextIOWrapper(_Nul(), encoding="cp1252"))
    assert nul.encoding == ("utf-8" if sys.platform == "win32" else "cp1252")


def test_a_dry_run_with_resume_creates_no_database(cli, store):
    # The resume lookup connects, and connecting writes the schema into any
    # file it opens; a dry run against a gallery that does not exist yet used
    # to leave a 40 KB local-diffusion.db behind.
    store.settings.db_path.unlink()
    code = generate.main(["a fox", "--dry-run", "--resume", "--seed", "1", "--grid-steps", "6,8", *SIZE])
    assert code == 0
    assert not store.settings.db_path.exists()


def test_a_dry_run_with_resume_does_not_move_existing_thumbnails(cli, store, capsys):
    assert generate.main(["a fox", "--seed", "3", "--grid-steps", "6", *SIZE, "-q"]) == 0
    record = store.list_images()[0][0]
    png = store.settings.outputs_dir / record.filename
    legacy = png.with_suffix(".webp")
    thumb = png.parent / "thumbnails" / legacy.name
    original = thumb.read_bytes()
    thumb.rename(legacy)
    thumb.parent.rmdir()
    capsys.readouterr()

    assert generate.main([
        "a fox", "--seed", "3", "--grid-steps", "6,8", "--resume", "--dry-run", *SIZE,
    ]) == 0

    assert "1 cell(s) already rendered, skipping" in capsys.readouterr().out
    assert not thumb.parent.exists()
    assert legacy.read_bytes() == original
    assert cli.loads == 1 and len(cli.requests) == 1


def test_a_cli_sweep_writes_the_seeds_the_server_would(cli, store):
    # --runs 3 --batch 2 --seed 7: cells at 7, 9, 11 and images 7..12 - the
    # same six the HTTP path's test asserts for the same request.
    assert generate.main(["a fox", "--seed", "7", "--runs", "3", "--batch", "2", *SIZE, "-q"]) == 0
    assert cli.seeds == [7, 9, 11]
    items, total, _ = store.list_images()
    assert sorted(r.seed for r in items) == [7, 8, 9, 10, 11, 12]
    assert len({r.job_id for r in items}) == 1
    assert len({r.run_id for r in items}) == 1 and items[0].run_id


def test_the_cli_saves_its_calls_peak_on_every_image_of_the_batch(cli, store):
    # As the server's worker does: the call's peak whole on each image.
    cli.peak = (3 * 1024**3, 4 * 1024**3)
    assert generate.main(["a fox", "--seed", "7", "--batch", "2", *SIZE, "-q"]) == 0
    items = store.list_images()[0]
    assert len(items) == 2
    assert {(r.vram_peak_allocated, r.vram_peak_reserved) for r in items} == {cli.peak}


def test_then_refine_follows_each_image_with_the_servers_pass(cli, store):
    from backend import grid, refine_batch, then_refine
    from backend.model_defaults import defaults_for
    from backend.model_selection import current_model_id
    from backend.schemas import ThenRefine

    flags = ["--then-refine-scale", "1.5", "--then-refine-strength", "0.5"]
    assert generate.main(["a fox", "--seed", "7", "--runs", "2", "--steps", "8", *SIZE, *flags, "-q"]) == 0
    # Each cell, then its pass, as the server's worker runs them.
    assert [r.refine is not None for r in cli.requests] == [False, True, False, True]
    rows = sorted(store.list_images()[0], key=lambda r: r.id)
    bases = [r for r in rows if r.refine is None]
    passes = {r.refine: r for r in rows if r.refine is not None}
    assert set(passes) == {b.id for b in bases}
    spec = ThenRefine(scale=1.5, strength=0.5)
    running = current_model_id(config.settings)
    for base in bases:
        made = passes[base.id]
        assert made.run_id == then_refine.pass_run_id(base.run_id, spec) and made.job_id == base.job_id
        planned = refine_batch.plan(
            base, running=running, profile=defaults_for(running), scale=1.5, strength=0.5, steps=None,
            include_foreign=False, skip_passes=False,
        )
        assert grid.cell_key(planned.cell) == grid.cell_key(grid.cell_from_record(made))


def test_then_refine_flags_are_refused_in_words(cli, store, capsys):
    for argv, said in (
        (["--then-refine-strength", "0.5"], "need --then-refine-scale"),
        (["--then-refine-scale", "3"], "from 1 to 2"),
        (["--then-refine-scale", "1.5", "--refine", "1"], "are alternatives"),
    ):
        with pytest.raises(SystemExit):
            generate.main(["a fox", *SIZE, *argv])
        assert said in capsys.readouterr().err
    assert generate.main(["a fox", "--steps", "4", *SIZE, "--then-refine-scale", "1.5",
                          "--then-refine-strength", "0.1", "-q"]) == 2
    assert "leaves no step" in capsys.readouterr().err
    assert cli.requests == []


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


def _held_gallery(store, monkeypatch, seconds):
    """Another process's long hold - a browser bundle export - beginning as
    the first image is saved, past a shortened SESSION_WAIT."""
    import threading

    from backend import gallery_lock

    monkeypatch.setattr(gallery_lock, "SESSION_WAIT", 0.2)
    started = threading.Event()

    def hold():
        with gallery_lock.hold(store.settings.db_path):
            started.set()
            threading.Event().wait(seconds)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert started.wait(5)
    return holder


def test_a_save_refused_by_a_busy_gallery_waits_rather_than_ending_the_night(
    cli, store, monkeypatch, capsys
):
    # The server's job worker retries a GalleryBusy save; the CLI lost the
    # rendered image and every cell after it, returning 1 with nothing saved.
    real_save = storage.save_image
    holders = []

    def save_after_a_hold(*args, **kwargs):
        if not holders:
            holders.append(_held_gallery(store, monkeypatch, 0.8))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(storage, "save_image", save_after_a_hold)
    assert generate.main(["a fox", "--seed", "7", "--runs", "3", *SIZE]) == 0
    assert store.list_images()[1] == 3
    err = capsys.readouterr().err
    assert err.count("gallery busy in another process; still waiting to save") == 1
    holders[0].join(5)


def test_resume_against_a_busy_gallery_says_so_without_a_traceback(cli, store, monkeypatch, capsys):
    holder = _held_gallery(store, monkeypatch, 0.8)
    code = generate.main(["a fox", "--seed", "3", "--grid-steps", "6,8", "--resume", *SIZE, "-q"])
    assert code == 1 and cli.requests == []
    assert "--resume: another Local Diffusion process is using the gallery" in capsys.readouterr().err
    holder.join(5)


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
    monkeypatch.delenv("LOCAL_DIFFUSION_MODEL_ID", raising=False)
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

    assert os.environ["LOCAL_DIFFUSION_MODEL_ID"] == store.settings.model_id
    # An explicit flag still wins over the record.
    assert generate.main(["--from-image", path, "--steps", "9", "--aspect", "16:9", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "sampling : 9 steps" in out and "size     : 1360x768" in out


def test_true_cfg_is_a_flag_where_the_checkpoint_keeps_it_apart(cli, store, monkeypatch, capsys):
    from backend import config

    flux = dataclasses.replace(store.settings, model_id="black-forest-labs/FLUX.1-Krea-dev")
    monkeypatch.setattr(config, "settings", flux)
    monkeypatch.setattr(storage, "settings", flux)
    # The fake stands in for FluxPipeline's shape, as the profile predicts it.
    cli.cfg_threshold, cli.true_cfg_supported = 1.0, True
    # On FLUX.1 the negative prompt needs --true-cfg; --cfg is the distilled
    # guidance value and has no say.
    assert generate.main(["a fox", "-n", "blurry", "--cfg", "3.5", "--dry-run", *SIZE]) == 0
    out = capsys.readouterr().out
    assert "[ignored: True CFG is at or below 1 - set --true-cfg]" in out
    assert generate.main(["a fox", "-n", "blurry", "--true-cfg", "1.5", "--dry-run", *SIZE]) == 0
    out = capsys.readouterr().out
    assert "negative : blurry\n" in out and "true CFG 1.5" in out
    # Rendered, the record carries the real scale and --from-image reads it back.
    assert generate.main(["a fox", "-n", "blurry", "--true-cfg", "1.5", "--seed", "3", *SIZE, "-q"]) == 0
    path = capsys.readouterr().out.strip().split("  seed ")[0]
    assert cli.requests[-1].true_cfg == 1.5 and cli.requests[-1].negative_prompt == "blurry"
    assert store.list_images()[0][0].true_cfg == 1.5
    assert generate.main(["--from-image", path, "--dry-run"]) == 0
    assert "true CFG 1.5" in capsys.readouterr().out

    # Where --cfg is real CFG already the flag is ignored and says so.
    turbo = dataclasses.replace(store.settings, model_id="krea/Krea-2-Turbo")
    monkeypatch.setattr(config, "settings", turbo)
    monkeypatch.setattr(storage, "settings", turbo)
    cli.cfg_threshold, cli.true_cfg_supported = 0.0, False
    assert generate.main(["a fox", "--true-cfg", "3", "--dry-run", *SIZE]) == 0
    assert "true cfg : 3  [ignored" in capsys.readouterr().out


def test_a_dry_run_warns_about_cells_that_render_the_same_image(cli, store, capsys):
    # A negative-prompt variable under a CFG that never clears the threshold:
    # the cells differ on paper and not in pixels, and the warning is the
    # cheapest place to find out.
    cli.cfg_threshold = 1.0
    assert generate.main([
        "a fox", "-n", "{avoid}", "--grid-prompt", "avoid = blurry | dark | noisy",
        "--cfg", "0.5", "--seed", "1", "--dry-run", *SIZE,
    ]) == 0
    assert "WARNING  : 2 of 3 image(s) would be identical" in capsys.readouterr().out


def test_a_mu_the_checkpoint_cannot_take_is_reported_as_ignored(cli, store, capsys):
    # applied() drops it and the row records none, so the header must not
    # print it as though it ran - as a dropped negative prompt is marked.
    assert generate.main(["a fox", "--mu", "1.3", "--dry-run", *SIZE]) == 0
    out = capsys.readouterr().out
    assert "mu 1.3\n" in out and "takes no Mu" not in out
    cli.mu_supported = False
    assert generate.main(["a fox", "--mu", "1.3", "--dry-run", *SIZE]) == 0
    assert "mu 1.3  [ignored: this checkpoint takes no Mu]" in capsys.readouterr().out
    assert generate.main(["a fox", "--dry-run", *SIZE]) == 0
    assert "takes no Mu" not in capsys.readouterr().out
    # A Mu sweep there: every cell is the same image, and the axis and the
    # warning both name Mu as the reason.
    assert generate.main(["a fox", "--grid-mu", "1,1.5,2", "--seed", "1", "--dry-run", *SIZE]) == 0
    out = capsys.readouterr().out
    assert "1, 1.5, 2  [ignored: this checkpoint takes no Mu]" in out
    assert "2 of 3 image(s) would be identical" in out and "a Mu where it takes none" in out


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


def test_from_image_refuses_a_record_that_is_not_one(cli, tmp_path, capsys):
    # Through png_settings, which the page's Load settings from PNG shares: a
    # record that is JSON but no object is an argument error, not a traceback.
    from PIL import Image, PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("local-diffusion", "[1, 2]")
    odd = tmp_path / "odd.png"
    Image.new("RGB", (8, 8)).save(odd, pnginfo=info)
    with pytest.raises(SystemExit) as exc:
        generate.main(["--from-image", str(odd), "--dry-run"])
    assert exc.value.code == 2
    assert "is not a settings record" in capsys.readouterr().err


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
    assert err.startswith("Steps: 110 is above 100") and "\n" not in err


def test_a_batch_past_the_budget_names_the_flag_to_use(cli, capsys):
    # The command line's words: --runs for a plain sweep, a smaller --batch
    # on a grid, which refuses --runs.
    big = ["--width", "1440", "--height", "1440", "--batch", "12"]
    assert generate.main(["a fox", *big]) == 2
    assert "use --runs instead of --batch" in capsys.readouterr().err
    assert generate.main(["a fox", *big, "--grid-steps", "6,8"]) == 2
    assert "lower --batch: a grid runs its seeds one batch at a time" in capsys.readouterr().err


def test_durations_have_an_hours_branch():
    assert generate._duration(45) == "45s"
    assert generate._duration(150) == "2m30s"
    assert generate._duration(4 * 3600 + 12 * 60 + 30) == "4h12m"


# --------------------------------------------------------- conditioning flags


def test_a_reference_is_refused_before_the_load(cli, store, capsys):
    # The same refusal the server makes at submit: a night that renders every
    # cell as plain text-to-image because the pipeline ignores `image` looks
    # like a success, and by morning it is one.
    code = generate.main(["a fox", "--reference", "1", *SIZE])
    assert code == 2 and cli.loads == 0
    assert "takes no reference images" in capsys.readouterr().err


def test_a_reference_that_is_not_in_the_gallery_is_named(cli, store, monkeypatch, capsys):
    monkeypatch.setattr(
        generate, "_project", _project_claiming(generate, "black-forest-labs/FLUX.2-klein-4B")
    )
    code = generate.main(["a fox", "--reference", "4242", *SIZE])
    assert code == 2 and cli.loads == 0
    assert "no gallery image with id 4242" in capsys.readouterr().err


def test_an_adapter_that_is_not_there_is_refused_before_the_load(cli, store, capsys, tmp_path):
    from backend import loras

    code = generate.main(["a fox", "--lora", "missing.safetensors", *SIZE])
    assert code == 2 and cli.loads == 0
    err = capsys.readouterr().err
    assert "no LoRA named" in err and str(loras.settings.loras_dir) in err


def test_an_adapter_for_another_checkpoint_is_refused_before_the_load(cli, store, capsys, lora_dir):
    cli.lora_mismatches = {"style.safetensors": "style.safetensors attaches to 494 modules X has not got"}
    code = generate.main(["a fox", "--lora", "style.safetensors", *SIZE])
    assert code == 2 and cli.loads == 0
    assert "494 modules X has not got" in capsys.readouterr().err
    assert generate.main(["a fox", "--lora", "other.safetensors", "--dry-run", *SIZE]) == 0


def test_a_weight_sweep_needs_an_adapter(cli, store):
    with pytest.raises(SystemExit) as raised:
        generate.main(["a fox", "--grid-lora-scale", "0,1", *SIZE])
    assert raised.value.code == 2


def test_from_image_carries_the_conditioning_it_recorded(cli, store, monkeypatch, capsys, lora_dir):
    from PIL import Image

    from backend.schemas import GenerateRequest

    def keep(**extra):
        return store.save_image(
            Image.new("RGB", (256, 256)),
            GenerateRequest(prompt="a fox", width=256, height=256, **extra),
            seed=11, duration_ms=10, model_id="black-forest-labs/FLUX.2-klein-4B",
        )

    source = keep()  # the image the second one was conditioned on
    record = keep(reference=[source.id], lora="style.safetensors", lora_scale=0.4)
    monkeypatch.setattr(
        generate, "_project", _project_claiming(generate, "black-forest-labs/FLUX.2-klein-4B")
    )
    path = str(store.settings.outputs_dir / record.filename)
    # A dry run, so nothing loads: what is being checked is that the flags the
    # PNG fills in are the whole reproduction set, adapter and references too.
    assert generate.main(["--from-image", path, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"reference: {source.id}" in out
    assert "lora     : style.safetensors @ 0.4" in out

    # An explicit flag still wins over the file - and the recorded weight goes
    # with the recorded adapter, since it is that adapter's number.
    assert generate.main(["--from-image", path, "--dry-run", "--lora", "other.safetensors"]) == 0
    assert "lora     : other.safetensors @ 1" in capsys.readouterr().out


def test_from_image_carries_the_variables_while_the_prompt_is_the_images(cli, store, capsys):
    # A comparison version's name lives in its variables alone, and a `{var}`
    # grid's assignment is what its sheet lays out: a reproduction that
    # dropped them filed the new image as an unnamed prompt.
    import json
    from pathlib import Path

    from PIL import Image

    from backend import grid
    from backend.schemas import GenerateRequest

    def made(*argv):
        """What each new image recorded, on its row and in its PNG's chunk."""
        assert generate.main([*argv, "-q"]) == 0
        paths = [line.split("  seed ")[0] for line in capsys.readouterr().out.splitlines()]
        rows = {r.filename: r for r in store.list_images()[0]}
        recorded = []
        for path in paths:
            with Image.open(path) as png:
                chunk = json.loads(png.info["local-diffusion"])["variables"]
            row = rows[Path(path).name]
            assert chunk == row.variables  # the PNG's own record, for a reindex
            recorded.append((row.prompt, row.variables))
        return paths, recorded

    (small, _), recorded = made(
        "a {size} cat", "-n", "blurry", "--grid-prompt", "size = small | large", "--seed", "5", *SIZE
    )
    assert recorded[0] == ("a small cat", {"size": "small"})
    assert made("--from-image", small, "--width", "512", "--height", "512")[1] == [
        ("a small cat", {"size": "small"})
    ]
    # Retyped, the pair is still the image's; changed, either half is another
    # text, which a label would misname - a variable may fill the negative.
    assert made("--from-image", small, "a small cat", "-n", "blurry")[1] == [
        ("a small cat", {"size": "small"})
    ]
    assert made("--from-image", small, "a large cat")[1] == [("a large cat", None)]
    assert made("--from-image", small, "-n", "noisy")[1] == [("a small cat", None)]
    # Every cell of a sweep carries them, and names them in the log as the
    # grid's own cells did.
    assert made("--from-image", small, "--grid-steps", "6,8")[1] == [
        ("a small cat", {"size": "small"})
    ] * 2
    assert generate.main(["--from-image", small, "--grid-steps", "6,8"]) == 0
    out = capsys.readouterr().out
    assert "size=small, 6 steps" in out and "size=small, 8 steps" in out

    # A comparison's version is taken verbatim, as the grid took it: its
    # `{word}` is text, not a placeholder with no values.
    version = store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="a sign reading {word}", width=256, height=256),
        seed=9, duration_ms=10, model_id=store.settings.model_id,
        variables={grid.VARIANT: "rewrite"},
    )
    path = str(store.settings.outputs_dir / version.filename)
    assert made("--from-image", path, "--grid-steps", "6,8")[1] == [
        ("a sign reading {word}", {grid.VARIANT: "rewrite"})
    ] * 2
    assert generate.main(["--from-image", path, "--grid-steps", "6,8", "--dry-run"]) == 0
    assert "prompt   : a sign reading {word}" in capsys.readouterr().out
    # `--grid-prompt` is a prompt axis of its own, and names its own cells.
    assert made("--from-image", path, "--grid-prompt", "word = open | shut")[1] == [
        ("a sign reading open", {"word": "open"}),
        ("a sign reading shut", {"word": "shut"}),
    ]
    # A blank one is no axis, and ends nothing.
    assert made("--from-image", path, "--grid-prompt", "")[1] == [
        ("a sign reading {word}", {grid.VARIANT: "rewrite"})
    ]

    # A user's own grid may name its variable `variant` too. `none` records an
    # empty value and this one is past a label's 40 characters: no label a
    # comparison could have recorded, so each is an assignment, carried as one.
    long = "an extremely fluffy, long-haired orange tabby"
    (bare, spelled), _ = made(
        "a {variant} cat", "--grid-prompt", f"variant = none | {long}", "--seed", "5", *SIZE
    )
    assert made("--from-image", bare)[1] == [("a cat", {grid.VARIANT: ""})]
    assert made("--from-image", spelled)[1] == [(f"a {long} cat", {grid.VARIANT: long})]


def _project_claiming(module, model_id):
    """`_project()` with the settings claiming another checkpoint.

    The reference check reads the profile for `settings.model_id`, and the
    profile is what says whether the pipeline takes an `image` at all.
    """
    original = module._project

    def patched():
        project = original()
        project.settings = dataclasses.replace(project.settings, model_id=model_id)
        return project

    return patched


# ------------------------------------------------------------ the refine pass


def _to_refine(store):
    """A gallery image to refine: a 2:3 portrait at this checkpoint's id."""
    from PIL import Image

    from backend.schemas import GenerateRequest

    return store.save_image(
        Image.new("RGB", (256, 384)),
        GenerateRequest(prompt="a fox", width=256, height=384),
        seed=5, duration_ms=10, model_id=store.settings.model_id,
    )


def test_a_refine_renders_at_the_sources_size_times_the_scale(cli, store, capsys):
    from backend.schemas import default_strength

    source = _to_refine(store)
    assert generate.main(["a fox", "--refine", str(source.id), "--steps", "20", "--seed", "7"]) == 0
    ran = cli.requests[-1]
    # No size given: the source's own, times the default 1.5. No strength
    # given: the checkpoint's own starting one, and the header says how
    # many of the twenty steps that leaves to run.
    start = default_strength()
    assert (ran.refine, ran.strength, ran.width, ran.height) == (source.id, start, 384, 576)
    out = capsys.readouterr().out
    assert (
        f"refine   : image {source.id} at strength {start:g}  "
        f"(about {int(20 * start)} of 20 steps; source 256x384)"
    ) in out
    newest = store.list_images()[0][0]
    assert (newest.refine, newest.strength, newest.width, newest.height) == (source.id, start, 384, 576)

    assert generate.main(
        ["a fox", "--refine", str(source.id), "--scale", "2", "--strength", "0.3", "--steps", "10", "-q"]
    ) == 0
    ran = cli.requests[-1]
    assert (ran.strength, ran.width, ran.height) == (0.3, 512, 768)
    # A size given outright is the size: the source is cropped to it.
    assert generate.main(["a fox", "--refine", str(source.id), "--steps", "20", "-q", *SIZE]) == 0
    assert (cli.requests[-1].width, cli.requests[-1].height) == (256, 256)


def test_refine_flags_need_a_source_and_one_way_of_sizing(cli, store):
    source = _to_refine(store)
    for argv in (
        ["a fox", "--strength", "0.3", *SIZE],
        ["a fox", "--scale", "2"],
        ["a fox", "--grid-strength", "0.2,0.4", *SIZE],
        ["a fox", "--refine", str(source.id), "--scale", "2", *SIZE],
        ["a fox", "--refine", str(source.id), "--scale", "0"],
        ["a fox", "--refine", str(source.id), "--scale", "nan"],
        ["a fox", "--refine", str(source.id), "--scale", "inf"],
        ["a fox", "--refine", str(source.id), "--scale=-inf"],
        ["a fox", "--refine", str(source.id), "--scale", "1e308"],
    ):
        with pytest.raises(SystemExit) as raised:
            generate.main(argv)
        assert raised.value.code == 2, argv
    assert cli.requests == []


def test_a_refine_is_refused_before_the_load(cli, store, monkeypatch, capsys):
    source = _to_refine(store)
    # A source the gallery has not got, named - with or without a size.
    assert generate.main(["a fox", "--refine", "4242"]) == 2
    assert "no gallery image with id 4242" in capsys.readouterr().err
    assert generate.main(["a fox", "--refine", "4242", *SIZE]) == 2
    assert "no gallery image with id 4242" in capsys.readouterr().err
    # A strength that leaves no step of a distilled checkpoint's few.
    assert generate.main(["a fox", "--refine", str(source.id), "--steps", "4", "--strength", "0.2"]) == 2
    assert "less than one step" in capsys.readouterr().err
    # And a checkpoint diffusers has no image-to-image pipeline for: the same
    # refusal the server makes at submit, from the profile, ahead of the load.
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, "krea/Krea-2-Turbo"))
    assert generate.main(["a fox", "--refine", str(source.id)]) == 2
    assert "has no refine pass" in capsys.readouterr().err
    assert cli.loads == 0 and cli.requests == []


def test_from_image_reproduces_a_refine_pass(cli, store, capsys):
    source = _to_refine(store)
    assert generate.main(
        ["a fox, sharper", "--refine", str(source.id), "--strength", "0.3", "--steps", "10", "--seed", "7", "-q"]
    ) == 0
    path = capsys.readouterr().out.split("  seed ")[0]
    # The PNG carries the source, the strength and its own size, so nothing
    # is derived again: the same pass, cell for cell.
    assert generate.main(["--from-image", path, "-q"]) == 0
    first, again = cli.requests[-2:]
    assert (again.refine, again.strength, again.width, again.height, again.steps) == (
        first.refine, first.strength, first.width, first.height, first.steps
    )
    assert cli.seeds[-2:] == [7, 7]
    # A flag still wins over the file, and --scale over the file's size.
    assert generate.main(["--from-image", path, "--strength", "0.6", "--scale", "2", "-q"]) == 0
    ran = cli.requests[-1]
    assert (ran.refine, ran.strength, ran.width, ran.height) == (source.id, 0.6, 512, 768)


def test_a_strength_sweep_from_the_terminal(cli, store, capsys):
    source = _to_refine(store)
    argv = ["a fox", "--refine", str(source.id), "--steps", "20", "--seed", "1",
            "--grid-strength", "0.2-0.6x0.2"]
    assert generate.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "strength: 0.2, 0.4, 0.6" in out and "3 strength" in out
    assert generate.main([*argv, "-q"]) == 0
    assert [r.strength for r in cli.requests] == [0.2, 0.4, 0.6]
    assert {r.strength for r in store.list_images(search=f"refine:{source.id}")[0]} == {0.2, 0.4, 0.6}
    # Resumed, all three are in the gallery under the keys the cells carry.
    assert generate.main([*argv, "--resume", "-q"]) == 0
    assert len(cli.requests) == 3


@pytest.mark.parametrize("axes", [
    ["--steps", "8", "--grid-strength", "0.3,0.5"],
    ["--steps", "4", "--grid-steps", "20,28"],
])
def test_refine_cli_checks_actual_cells_after_axes_replace_base(cli, store, axes):
    source = _to_refine(store)
    argv = ["a fox", "--refine", str(source.id), "--strength", "0.1", *axes]
    assert generate.main([*argv, "--dry-run"]) == 0
    assert cli.loads == 0
    assert generate.main([*argv, "-q"]) == 0
    assert len(cli.requests) == 2


def test_cli_rechecks_refine_after_loading_before_any_cell(cli, store, monkeypatch, capsys):
    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, "black-forest-labs/FLUX.1-Kontext-dev"))
    load = cli.load

    def unsupported_load():
        load()
        cli.refine_supported = False

    monkeypatch.setattr(cli, "load", unsupported_load)
    assert generate.main(["a fox", "--refine", str(source.id),
                          "--grid-strength", "0.2,0.4", "--seed", "1", "-q"]) == 2
    assert "has no refine pass" in capsys.readouterr().err
    assert cli.loads == 1 and not cli.loaded and cli.requests == []
    assert store.list_images()[1] == 1


# ------------------------------------------------------------- the inpaint

ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


def _mask_file(tmp_path, size=(256, 384), box=(0, 0, 128, 64), name="mask.png"):
    from PIL import Image

    image = Image.new("L", size, 0)
    image.paste(255, box)
    path = tmp_path / name
    image.convert("RGB").save(path)  # any mode: kept as one channel
    return path, image


def test_an_inpaint_from_the_terminal_stores_its_mask_and_records_it(cli, store, monkeypatch, tmp_path, capsys):
    source = _to_refine(store)
    path, image = _mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    argv = ["a red scarf", "--refine", str(source.id), "--mask", str(path), "--strength", "0.6", "--seed", "4"]
    # A dry run says what would run and keeps nothing.
    assert generate.main([*argv, "--dry-run"]) == 0
    assert f"inpaint  : under mask {digest[:16]}" in capsys.readouterr().out
    assert store.mask_size(digest) is None and cli.loads == 0
    assert generate.main([*argv, "-q"]) == 0
    ran = cli.requests[-1]
    # At the source's own size, never scaled, under the mask's hash.
    assert (ran.refine, ran.mask, ran.strength, ran.width, ran.height) == (source.id, digest, 0.6, 256, 384)
    assert store.mask_path(digest).is_file()
    newest = store.list_images()[0][0]
    assert (newest.refine, newest.mask) == (source.id, digest)


def test_mask_flags_are_refused_where_they_cannot_run(cli, store, monkeypatch, tmp_path, capsys):
    source = _to_refine(store)
    path, _ = _mask_file(tmp_path)
    black, _ = _mask_file(tmp_path, box=(0, 0, 0, 0), name="black.png")
    wrong, _ = _mask_file(tmp_path, size=(256, 256), name="square.png")
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    for argv in (
        ["a fox", "--mask", str(path), *SIZE],
        ["a fox", "--refine", str(source.id), "--mask", str(path), "--scale", "2"],
        ["a fox", "--refine", str(source.id), "--mask", str(black)],
        ["a fox", "--refine", str(source.id), "--mask", str(tmp_path / "nowhere.png")],
    ):
        with pytest.raises(SystemExit) as raised:
            generate.main(argv)
        assert raised.value.code == 2, argv
    capsys.readouterr()
    # A mask painted on another size, said in words, before the load.
    assert generate.main(["a fox", "--refine", str(source.id), "--mask", str(wrong)]) == 2
    assert "is 256x256, but source image" in capsys.readouterr().err
    # And a checkpoint whose inpaint is not enabled, from its profile.
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, "black-forest-labs/FLUX.2-klein-4B"))
    assert generate.main(["a fox", "--refine", str(source.id), "--mask", str(path)]) == 2
    assert "has no inpaint in this app" in capsys.readouterr().err
    assert cli.loads == 0 and cli.requests == []


def test_from_image_reproduces_an_inpaint_over_its_mask(cli, store, monkeypatch, tmp_path, capsys):
    source = _to_refine(store)
    path, image = _mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    # The mask file lost since: the PNG's own copy is the same picture.
    store.mask_path(digest).unlink()
    assert generate.main(["--from-image", made, "-q"]) == 0
    first, again = cli.requests[-2:]
    assert (again.refine, again.mask, again.strength, again.width, again.height) == (
        first.refine, first.mask, first.strength, first.width, first.height
    )
    assert store.mask_path(digest).is_file()
    # A --refine of another image takes no mask from this one.
    assert generate.main(["--from-image", made, "--refine", str(source.id), "-q"]) == 0
    assert cli.requests[-1].mask is None


def test_an_inpaint_sweep_runs_the_cells_the_server_would(cli, store, monkeypatch, tmp_path):
    from backend import grid
    from backend.schemas import GenerateRequest

    source = _to_refine(store)
    path, image = _mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--steps", "20",
                          "--seed", "3", "--grid-strength", "0.4,0.6", "-q"]) == 0
    # The server's expansion of the same request: one cell per strength,
    # every one over the mask, at the same seeds.
    request = GenerateRequest(prompt="a red scarf", width=256, height=384, steps=20,
                              refine=source.id, strength=0.4, mask=digest, seed=3)
    cells = grid.expand(request, strength="0.4,0.6", seeds=1)
    assert [(r.strength, r.mask) for r in cli.requests] == [(c.request.strength, c.request.mask) for c in cells]
    assert cli.seeds == [c.seed for c in cells]



def test_a_cli_cell_whose_mask_went_fails_in_words(cli, store, monkeypatch, tmp_path, capsys):
    source = _to_refine(store)
    path, image = _mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    saved = store.save_image

    def save_then_lose_the_mask(*args, **kwargs):
        # Gone once the first cell's image is saved, before the second began,
        # as the server's worker test has it.
        try:
            return saved(*args, **kwargs)
        finally:
            (store.masks_dir() / f"{digest}.png").unlink(missing_ok=True)

    monkeypatch.setattr(store, "save_image", save_then_lose_the_mask)
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--steps", "20",
                          "--seed", "3", "--grid-strength", "0.4,0.6", "-q"]) == 1
    assert "missing from the masks directory" in capsys.readouterr().err
    assert [r.strength for r in cli.requests] == [0.4]


def test_a_mask_file_with_alpha_or_16_bit_levels_is_read_as_painted(cli, store, monkeypatch, tmp_path):
    import numpy as np
    from PIL import Image

    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    # White strokes on a transparent layer whose hidden colour is white too,
    # as an editor exports them: only the stroke is masked.
    layer = Image.new("RGBA", (256, 384), (255, 255, 255, 0))
    layer.paste((255, 255, 255, 255), (0, 0, 128, 64))
    layer.save(tmp_path / "layer.png")
    # And a 16-bit greyscale export: scaled to 8 bits, not clipped.
    levels = np.zeros((384, 256), dtype=np.uint16)
    levels[:64, :128] = 65535
    levels[64:, :] = 100  # a near-black "keep", which clipping would make 100/255
    Image.fromarray(levels).save(tmp_path / "deep.png")
    with Image.open(tmp_path / "deep.png") as deep:
        assert deep.mode.startswith("I")
    wanted = Image.new("L", (256, 384), 0)
    wanted.paste(255, (0, 0, 128, 64))
    for name in ("layer.png", "deep.png"):
        assert generate.main(["a fox", "--refine", str(source.id), "--mask", str(tmp_path / name),
                              "--strength", "0.6", "-q"]) == 0, name
        assert cli.requests[-1].mask == store.mask_hash(wanted), name


def test_a_mask_in_a_mode_no_mask_is_in_is_refused_by_name(cli, store, monkeypatch, tmp_path, capsys):
    from PIL import Image

    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    Image.new("CMYK", (256, 384), (0, 0, 0, 255)).save(tmp_path / "print.tif")
    with pytest.raises(SystemExit) as raised:
        generate.main(["a fox", "--refine", str(source.id), "--mask", str(tmp_path / "print.tif")])
    assert raised.value.code == 2
    assert "not one in mode CMYK" in capsys.readouterr().err


def test_a_mask_without_a_strength_starts_at_the_inpaint_classs_default(cli, store, monkeypatch, tmp_path):
    import diffusers

    from backend import model_selection, refine

    source = _to_refine(store)
    path, _ = _mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    model_selection.select(ANIMAGINE)
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "-q"]) == 0
    wanted = refine.library_strength(diffusers.StableDiffusionXLInpaintPipeline)
    assert cli.requests[-1].strength == wanted == 0.9999


def test_from_image_drops_a_source_this_gallery_holds_another_picture_under(
    cli, store, monkeypatch, tmp_path, capsys
):
    source = _to_refine(store)
    path, _ = _mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--strength", "0.6",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    # The id now names another picture of the same size - a PNG from another
    # gallery, or ids renumbered since.
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--from-image", made, "-q"]) == 0
    ran = cli.requests[-1]
    assert (ran.refine, ran.mask) == (None, None)
    assert f"source image {source.id} is not the image the PNG was made from" in capsys.readouterr().err
    # Asked for as an inpaint all the same, it is refused rather than run over the wrong picture.
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "--strength", "0.5", "-q"])
    # Named on the command line, the source is the user's own choice.
    assert generate.main(["--from-image", made, "--refine", str(source.id), "-q"]) == 0
    assert cli.requests[-1].refine == source.id


# ------------------------------------------------------------- the outpaint


def test_an_outpaint_from_the_terminal_builds_stores_and_records_its_mask(cli, store, monkeypatch, capsys):
    from backend.schemas import Outpaint

    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    argv = ["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--band", "16", "--seed", "4"]
    digest = store.outpaint_mask_hash(304, 384, Outpaint(left=32, right=16, band=16))
    # A dry run says what would run and stores nothing.
    assert generate.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "outpaint : +32/+0/+16/+0, band 16 px, fill edge-box3-r16; canvas 304x384" in out
    assert "size     : 304x384" in out
    assert store.mask_size(digest) is None and cli.loads == 0
    assert generate.main([*argv, "-q"]) == 0
    ran = cli.requests[-1]
    assert (ran.refine, ran.width, ran.height, ran.mask) == (source.id, 304, 384, digest)
    assert ran.outpaint.key() == "edge-box3-r16:32,0,16,0@16"
    assert store.mask_path(digest).is_file()
    newest = store.list_images()[0][0]
    assert (newest.outpaint.key(), newest.mask) == ("edge-box3-r16:32,0,16,0@16", digest)
    # The CLI holds no mask in `mask_holds`: it runs its cells itself.
    with store._session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM mask_holds").fetchone()[0] == 0


def test_an_outpaint_preset_from_the_terminal(cli, store, monkeypatch, capsys):
    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    # 256 x 384 to 16:9: the width to ceil(384 * 16 / 9 / 16) * 16 = 688.
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "16:9", "-q"]) == 0
    ran = cli.requests[-1]
    assert ran.outpaint.key() == "edge-box3-r16:208,0,224,0@32" and (ran.width, ran.height) == (688, 384)
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "+25%", "-q"]) == 0
    assert cli.requests[-1].outpaint.key() == "edge-box3-r16:32,48,32,48@32"
    capsys.readouterr()
    # A preset that would change nothing is held in its words.
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "2:3"]) == 2
    assert f"--outpaint 2:3: #{source.id} is already 2:3" in capsys.readouterr().err


def test_outpaint_flags_are_refused_where_they_cannot_run(cli, store, monkeypatch, tmp_path, capsys):
    source = _to_refine(store)
    path, _ = _mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    refine = ["--refine", str(source.id)]
    for argv, words in (
        (["a fox", "--outpaint", "32,0,0,0", *SIZE], "--outpaint needs --refine"),
        (["a fox", *refine, "--outpaint", "50,0,0,0"], "50 is not a multiple of 16: 48 or 64"),
        (["a fox", *refine, "--outpaint", "0,0,0,0"], "an outpaint extends at least one side"),
        (["a fox", *refine, "--outpaint", "wide"], "neither four sides"),
        (["a fox", *refine, "--outpaint", "32,0,0,0", *SIZE], "--outpaint and --width/--height/--aspect/--mp"),
        (["a fox", *refine, "--outpaint", "32,0,0,0", "--scale", "2"], "--outpaint and --scale"),
        (["a fox", *refine, "--outpaint", "32,0,0,0", "--mask", str(path)], "an outpaint builds its own mask"),
        (["a fox", *refine, "--band", "16"], "--band needs --outpaint"),
    ):
        with pytest.raises(SystemExit) as raised:
            generate.main(argv)
        assert raised.value.code == 2, argv
        assert words in capsys.readouterr().err, argv
    # A band off its grid, and one leaving nothing kept.
    assert generate.main(["a fox", *refine, "--outpaint", "32,0,0,0", "--band", "12"]) == 2
    assert "12 is not a multiple of 8" in capsys.readouterr().err
    assert generate.main(["a fox", *refine, "--outpaint", "32,0,32,0", "--band", "128"]) == 2
    assert "on the left and right sides leaves 0 px of" in capsys.readouterr().err
    # And a checkpoint whose inpaint is not enabled, in the outpaint's words.
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, "black-forest-labs/FLUX.2-klein-4B"))
    assert generate.main(["a fox", *refine, "--outpaint", "32,0,0,0"]) == 2
    assert "which an outpaint runs through" in capsys.readouterr().err
    assert cli.loads == 0 and cli.requests == []


def test_an_off_grid_source_is_refused_by_name_before_any_request(cli, store, monkeypatch, capsys):
    from PIL import Image

    from backend.schemas import GenerateRequest

    source = store.save_image(
        Image.new("RGB", (264, 264)), GenerateRequest(prompt="x", width=256, height=256),
        seed=1, duration_ms=1, model_id=store.settings.model_id,
    )
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a fox", "--refine", str(source.id), "--outpaint", "16,0,0,0"]) == 2
    assert f"#{source.id} is 264 x 264, off the 16-px grid" in capsys.readouterr().err


def test_from_image_reproduces_an_outpaint_by_its_spec(cli, store, monkeypatch, capsys):
    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--band", "16",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    first = cli.requests[-1]
    # The mask file lost since: built again, never carried as a hash.
    store.mask_path(first.mask).unlink()
    assert generate.main(["--from-image", made, "-q"]) == 0
    again = cli.requests[-1]
    assert (again.outpaint, again.mask, again.width, again.height, again.strength) == (
        first.outpaint, first.mask, first.width, first.height, first.strength
    )
    assert store.mask_path(first.mask).is_file()
    # The user's own size flags are refused beside the carried outpaint.
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "--width", "512", "--height", "512"])
    assert "--outpaint and --width/--height/--aspect/--mp" in capsys.readouterr().err


def test_from_image_of_an_outpaint_takes_a_band_of_its_own(cli, store, monkeypatch, capsys):
    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    first = cli.requests[-1]
    # Another band builds another mask - what was asked, never refused as
    # a mask this build builds differently.
    assert generate.main(["--from-image", made, "--band", "16", "-q"]) == 0
    again = cli.requests[-1]
    assert again.outpaint.band == 16 and first.outpaint.band != 16
    assert (again.outpaint.left, again.outpaint.top, again.outpaint.right, again.outpaint.bottom) == (
        first.outpaint.left, first.outpaint.top, first.outpaint.right, first.outpaint.bottom
    )
    assert again.mask != first.mask
    assert again.mask == store.outpaint_mask_hash(again.width, again.height, again.outpaint)


def _rewrite_record(path, change) -> None:
    import json

    from PIL import Image, PngImagePlugin

    with Image.open(path) as image:
        image.load()
        meta = json.loads(image.info["local-diffusion"])
        change(meta)
        info = PngImagePlugin.PngInfo()
        info.add_text("local-diffusion", json.dumps(meta))
        image.save(path, format="PNG", pnginfo=info)


def test_from_image_refuses_a_mask_this_build_would_build_differently(cli, store, monkeypatch, capsys):
    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    _rewrite_record(made, lambda meta: meta.update(mask="e" * 64))
    assert generate.main(["--from-image", made, "-q"]) == 2
    assert "builds an outpaint's mask differently" in capsys.readouterr().err
    # An unknown fill is refused by name.
    _rewrite_record(made, lambda meta: meta.update(outpaint_fill="edge-box5-r8"))
    assert generate.main(["--from-image", made, "-q"]) == 2
    assert "this build has no outpaint fill 'edge-box5-r8'" in capsys.readouterr().err


def test_from_image_drops_an_outpaint_with_a_source_that_is_another_picture(cli, store, monkeypatch, capsys):
    source = _to_refine(store)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--seed", "4",
                          "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--from-image", made, "-q"]) == 0
    ran = cli.requests[-1]
    # A plain render at the PNG's own size, never an outpaint over the wrong picture.
    assert (ran.refine, ran.mask, ran.outpaint, ran.width) == (None, None, None, 304)
    assert "running as a plain generation instead of an outpaint" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "--band", "8", "-q"])
    assert "cannot be run again as an outpaint" in capsys.readouterr().err


def test_a_cli_cell_whose_outpaint_mask_went_fails_in_words(cli, store, monkeypatch, capsys):
    from backend.schemas import Outpaint

    source = _to_refine(store)
    digest = store.outpaint_mask_hash(304, 384, Outpaint(left=32, right=16))
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, ANIMAGINE))
    saved = store.save_image

    def save_then_lose_the_mask(*args, **kwargs):
        try:
            return saved(*args, **kwargs)
        finally:
            (store.masks_dir() / f"{digest}.png").unlink(missing_ok=True)

    monkeypatch.setattr(store, "save_image", save_then_lose_the_mask)
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--steps", "20",
                          "--seed", "3", "--grid-strength", "0.4,0.6", "-q"]) == 1
    assert "the outpaint's mask has gone from this gallery" in capsys.readouterr().err


# ----------------------------------------------------- script the running server


@pytest.fixture
def via_server(client, fake, monkeypatch):
    import os
    from types import SimpleNamespace

    for name in ("LOCAL_DIFFUSION_MODEL_ID", "LOCAL_DIFFUSION_QUANT", "LOCAL_DIFFUSION_OFFLOAD"):
        # Record the original absence too: apply_env may set a flag directly
        # during this invocation, after the fixture cleared its environment.
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_DIFFUSION_HOST", "127.0.0.1")
    monkeypatch.setenv("LOCAL_DIFFUSION_PORT", "7860")
    calls = []

    def request(base, path, *, body=None, data=None, headers=None, timeout=30):
        assert base == "http://127.0.0.1:7860"
        method = "POST" if body is not None or data is not None else "GET"
        calls.append((method, path, body, headers or {}))
        response = client.request(method, path, json=body, content=data, headers=headers)
        if response.status_code >= 400:
            raise generate._ServerError(str(response.json()["detail"]), response.status_code)
        return response.json()

    def no_local_backend(*args):
        pytest.fail("--via-server tried to construct/probe a local generation backend")

    monkeypatch.setattr(generate, "_server_json", request)
    monkeypatch.setattr(generate, "_SERVER_POLL_SECONDS", 0.01)
    monkeypatch.setattr(generate, "_project", no_local_backend)
    monkeypatch.setattr(generate, "_server_holds_gpu", no_local_backend)
    return SimpleNamespace(client=client, backend=fake, calls=calls, request=request)


def test_via_server_queues_the_plain_body_and_prints_each_url_seed_once(via_server, capsys):
    from backend import main

    argv = ["a fox 🦊", "--via-server", *SIZE, "--batch", "2", "--runs", "3", "--seed", "41",
            "--steps", "7", "--cfg", "2.5", "--true-cfg", "1.5", "--no-mu", "-n", "rain", "-q"]
    assert generate.main(argv) == 0
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert len(lines) == 6 and all(line.startswith("http://127.0.0.1:7860/outputs/") for line in lines)
    assert [int(line.rsplit("  seed ", 1)[1]) for line in lines] == list(range(41, 47))
    assert "Queued job" in out.err and "Queued job" not in out.out
    submitted = next(call for call in via_server.calls if call[1] == "/api/generate")
    body, headers = submitted[2:]
    assert {key: body[key] for key in ("prompt", "steps", "guidance_scale", "true_cfg", "mu", "num_images", "runs", "seed")} == {
        "prompt": "a fox 🦊", "steps": 7, "guidance_scale": 2.5, "true_cfg": 1.5,
        "mu": None, "num_images": 2, "runs": 3, "seed": 41,
    }
    assert headers["X-Local-Diffusion-Model-Revision"] == str(main.model_selection.revision())
    assert via_server.backend.seeds == [41, 43, 45]


def test_via_server_uses_the_selected_servers_profile_when_flags_are_omitted(via_server, capsys):
    from backend import main

    main.model_selection.select("cagliostrolab/animagine-xl-4.0")
    expected = via_server.client.get("/api/status").json()["defaults"]
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert not {"steps", "guidance_scale", "true_cfg", "mu", "seed", "strength"} & body.keys()
    ran = via_server.backend.requests[0]
    assert (ran.steps, ran.guidance_scale) == (expected["steps"], expected["guidance_scale"])


def test_via_server_grid_body_uses_the_same_axes_and_seed_order(via_server, capsys):
    assert generate.main(["a {animal}", "--via-server", *SIZE, "--grid-prompt", "animal = fox | wolf",
                          "--grid-steps", "6,8", "--grid-cfg", "1.5,2", "--seed-list", "11,99",
                          "--batch", "2", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/grid")
    assert (body["prompt_axis"], body["steps_axis"], body["cfg_axis"], body["seed_axis"], body["runs"]) == (
        "animal = fox | wolf", "6,8", "1.5,2", "11,99", 1,
    )
    assert via_server.backend.seeds == [seed for seed in (11, 99) for _ in range(8)]
    assert [request.prompt for request in via_server.backend.requests[:8]] == ["a fox"] * 4 + ["a wolf"] * 4


def test_via_server_random_prompt_mode_matches_the_direct_expansion(via_server, capsys):
    from backend import grid

    assert generate.main([
        "a {size} fox", "--via-server", *SIZE, "--grid-prompt", "size = small | large | none",
        "--grid-prompt-mode", "random", "--grid-steps", "6,8", "--seed-list", "42,80", "--batch", "2", "-q",
    ]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/grid")
    assert body["prompt_mode"] == "random"
    assert via_server.backend.seeds == [42, 42, 43, 43, 80, 80, 81, 81]
    expected = grid.parse_variables(body["prompt_axis"])
    for request, seed in zip(via_server.backend.requests, via_server.backend.seeds):
        assert request.num_images == 1
        assert request.prompt == grid.substitute("a {size} fox", grid.random_assignment(expected, seed))


def test_via_server_resume_and_plain_repeat_budget_use_grid_semantics(via_server, capsys):
    argv = ["a fox", "--via-server", *SIZE, "--runs", "3", "--seed", "91", "-q"]
    assert generate.main([*argv, "--budget", "10"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/grid")
    assert (body["seeds"], body["runs"], body["budget_minutes"]) == (3, 1, 10)
    capsys.readouterr()
    assert generate.main([*argv, "--resume"]) == 0
    assert "Nothing to do" in capsys.readouterr().err
    assert len(via_server.backend.requests) == 3


@pytest.mark.parametrize("flags,word", [
    (["--dry-run"], "makes no network requests"),
    (["--ignore-server"], "are alternatives"),
    (["--resume"], "needs --seed"),
    (["--grid-steps", "6,8", "--runs", "2"], "a grid steps its own seeds"),
    (["--grid-lora-scale", "0,1"], "needs --lora"),
    (["--grid-strength", "0.4,0.6"], "needs --refine"),
])
def test_via_server_refuses_incompatible_modes_before_network(via_server, capsys, flags, word):
    with pytest.raises(SystemExit) as raised:
        generate.main(["a fox", "--via-server", *SIZE, *flags])
    assert raised.value.code == 2 and word in capsys.readouterr().err
    assert via_server.calls == []


@pytest.mark.parametrize("flag,value", [("--model", "a/different-model"), ("--quant", "none"), ("--offload", "none")])
def test_via_server_model_flags_assert_the_server_selection_without_switching(via_server, capsys, flag, value):
    assert generate.main(["a fox", "--via-server", *SIZE, flag, value]) == 2
    assert f"{flag} asks for {value}" in capsys.readouterr().err
    assert [(call[0], call[1]) for call in via_server.calls] == [("GET", "/api/status")]
    assert via_server.backend.requests == []


def test_via_server_matching_model_flags_and_environment_are_accepted(via_server, monkeypatch):
    status = via_server.client.get("/api/status").json()
    monkeypatch.setenv("LOCAL_DIFFUSION_QUANT", status["quantization"])
    assert generate.main(["a fox", "--via-server", *SIZE, "--model", status["model_id"],
                          "--offload", status["offload"], "-q"]) == 0


def test_via_server_from_image_keeps_sampling_and_explicit_overrides(via_server, store, capsys):
    from tests.api_helpers import run_one

    _, events = run_one(via_server.client, prompt="a fox", width=256, height=256, steps=17,
                        guidance_scale=2.5, seed=72, mu=None)
    image = events[-1]["images"][0]
    path = store.settings.outputs_dir / image["filename"]
    _rewrite_record(path, lambda meta: meta.update(quantization="none"))
    via_server.calls.clear()
    assert generate.main(["a wolf", "--via-server", "--from-image", str(path), "--seed", "99", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["prompt"], body["seed"], body["steps"], body["guidance_scale"], body["mu"]) == ("a wolf", 99, 17, 2.5, None)
    assert "the PNG was made with quant=none" in capsys.readouterr().err


def test_via_server_refuses_carried_grid_labels_instead_of_silently_losing_them(via_server, store, capsys):
    from tests.api_helpers import stream_job

    accepted = via_server.client.post("/api/grid", json={"prompt": "a {animal}", "prompt_axis": "animal = fox | wolf",
                                                       "width": 256, "height": 256, "seed": 1}).json()
    image = stream_job(via_server.client, accepted["job_id"])[-1]["images"][0]
    with pytest.raises(SystemExit) as raised:
        generate.main(["--via-server", "--from-image", str(store.settings.outputs_dir / image["filename"])])
    assert raised.value.code == 2 and "cannot carry a PNG's grid/comparison labels" in capsys.readouterr().err
    assert via_server.calls == []


def test_via_server_refine_and_mask_use_the_server_gallery_and_upload_route(via_server, store, tmp_path, capsys):
    source = _to_refine(store)
    path, _ = _mask_file(tmp_path)
    assert generate.main(["a fox", "--via-server", "--refine", str(source.id), "--scale", "2", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["refine"], body["width"], body["height"]) == (source.id, 512, 768)
    via_server.calls.clear()
    assert generate.main(["a scarf", "--via-server", "--refine", str(source.id), "--mask", str(path),
                          "--strength", "0.6", "-q"]) == 0
    assert any(call[1] == f"/api/masks?source={source.id}" for call in via_server.calls)
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["width"], body["height"], body["strength"]) == (256, 384, 0.6)
    assert len(body["mask"]) == 64 and via_server.backend.requests[-1].mask == body["mask"]
    # The PNG's embedded mask can restore a swept server copy without the
    # CLI writing to any local masks directory itself.
    image = store.list_images()[0][0]
    store.mask_path(body["mask"]).unlink()
    assert generate.main(["--via-server", "--from-image", str(store.settings.outputs_dir / image.filename), "-q"]) == 0
    assert store.mask_path(body["mask"]).is_file()


def test_via_server_outpaint_uses_server_presets_and_keeps_its_spec(via_server, store):
    source = _to_refine(store)
    assert generate.main(["a wider view", "--via-server", "--refine", str(source.id), "--outpaint", "16:9",
                          "--band", "16", "--seed", "3", "-q"]) == 0
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert (body["width"], body["height"]) == (688, 384)
    assert body["outpaint"] == {"left": 208, "top": 0, "right": 224, "bottom": 0, "band": 16, "fill": "edge-box3-r16"}
    assert "mask" not in body
    image = store.list_images()[0][0]
    assert generate.main(["--via-server", "--from-image", str(store.settings.outputs_dir / image.filename), "-q"]) == 0
    assert via_server.backend.requests[-1].outpaint == via_server.backend.requests[-2].outpaint


def test_via_server_from_image_verifies_carried_references_against_the_server_gallery(via_server, store, capsys):
    from tests.api_helpers import run_one

    source = _to_refine(store)
    _, events = run_one(via_server.client, prompt="a reference portrait", width=256, height=256,
                        reference=[source.id], seed=10)
    path = store.settings.outputs_dir / events[-1]["images"][0]["filename"]
    assert generate.main(["--via-server", "--from-image", str(path), "-q"]) == 0
    assert via_server.backend.requests[-1].reference == [source.id]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    count = len(via_server.backend.requests)
    assert generate.main(["--via-server", "--from-image", str(path), "-q"]) == 2
    assert "not verified as the PNG's reference" in capsys.readouterr().err
    assert len(via_server.backend.requests) == count
    # An explicit source selection is the user's own choice of server ids.
    assert generate.main(["--via-server", "--from-image", str(path), "--reference", str(source.id), "-q"]) == 0


def test_via_server_from_image_drops_a_mismatched_refine_source_as_the_local_cli_does(via_server, store, capsys):
    from tests.api_helpers import run_one

    source = _to_refine(store)
    _, events = run_one(via_server.client, prompt="a refined portrait", width=256, height=384,
                        refine=source.id, strength=0.6, seed=10)
    path = store.settings.outputs_dir / events[-1]["images"][0]["filename"]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--via-server", "--from-image", str(path), "-q"]) == 0
    assert via_server.backend.requests[-1].refine is None
    assert "running as a plain generation" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        generate.main(["--via-server", "--from-image", str(path), "--scale", "2", "-q"])
    assert "name the source with --refine" in capsys.readouterr().err


def test_via_server_reports_job_failures_and_budget_cancellation(via_server, capsys):
    via_server.backend.fail_on = {1}
    assert generate.main(["a fox", "--via-server", *SIZE, "--runs", "2", "-q"]) == 1
    assert "failed" in capsys.readouterr().err
    via_server.backend.fail_on = set()
    assert generate.main(["a fox", "--via-server", *SIZE, "--runs", "3", "--budget", "0.000001", "-q"]) == 130
    assert "stopped by budget" in capsys.readouterr().err


def test_via_server_ctrl_c_cancels_only_its_owned_job(via_server, monkeypatch, capsys):
    via_server.backend.step_seconds = 0.01
    request = via_server.request
    interrupted = []

    def ctrl_c(base, path, **kwargs):
        if path.startswith("/api/jobs/") and not interrupted:
            interrupted.append(1)
            raise KeyboardInterrupt
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate, "_server_json", ctrl_c)
    assert generate.main(["a fox", "--via-server", *SIZE, "--steps", "30", "--runs", "3", "-q"]) == 130
    cancels = [call for call in via_server.calls if call[1].endswith("/cancel")]
    assert len(cancels) == 1 and "Cancelling job" in capsys.readouterr().err


def test_via_server_does_not_retry_or_fall_back_after_an_unconfirmed_submission(via_server, monkeypatch, capsys):
    request = via_server.request
    attempted = []

    def dropped_answer(base, path, **kwargs):
        if path == "/api/generate":
            attempted.append(path)
            raise generate._ServerError("connection closed")
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate, "_server_json", dropped_answer)
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 1
    assert attempted == ["/api/generate"]
    assert "Submission was not confirmed" in capsys.readouterr().err


def test_via_server_a_lost_owned_job_is_a_runtime_failure_and_is_never_resubmitted(via_server, monkeypatch, capsys):
    request = via_server.request

    def restarted(base, path, **kwargs):
        if path.startswith("/api/jobs/"):
            raise generate._ServerError("unknown job", 404)
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate, "_server_json", restarted)
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 1
    assert "unknown job (job " in capsys.readouterr().err
    assert len([call for call in via_server.calls if call[1] == "/api/generate"]) == 1


def test_via_server_refuses_a_stale_model_revision_before_generation(via_server, monkeypatch, capsys):
    from backend import main

    request = via_server.request

    def switched(base, path, **kwargs):
        if path == "/api/generate":
            main.model_selection.select("cagliostrolab/animagine-xl-4.0")
        return request(base, path, **kwargs)

    monkeypatch.setattr(generate, "_server_json", switched)
    assert generate.main(["a fox", "--via-server", *SIZE, "-q"]) == 2
    assert "checkpoint changed" in capsys.readouterr().err
    assert via_server.backend.requests == []


def test_server_json_sends_json_and_revision_without_retrying_the_transport(monkeypatch):
    import json
    import urllib.request

    calls = []

    def respond(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(b'{"job_id":"one"}')

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    assert generate._server_json("http://127.0.0.1:9999", "/api/generate", body={"prompt": "狐狸 🦊"},
                                 headers={"X-Local-Diffusion-Model-Revision": "9"}) == {"job_id": "one"}
    [(request, timeout)] = calls
    assert request.get_method() == "POST" and json.loads(request.data) == {"prompt": "狐狸 🦊"}
    assert request.get_header("X-local-diffusion-model-revision") == "9"
    assert request.get_header("Content-type") == "application/json" and timeout == 30


def test_server_json_reports_http_validation_and_network_errors_without_tracebacks(monkeypatch):
    import json
    import urllib.error
    import urllib.request

    def invalid(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 422, "Unprocessable", {},
                                     io.BytesIO(json.dumps({"detail": [{"loc": ["body", "width"], "msg": "invalid size"}]}).encode()))

    monkeypatch.setattr(urllib.request, "urlopen", invalid)
    with pytest.raises(generate._ServerError, match="width: invalid size") as error:
        generate._server_json("http://127.0.0.1:9999", "/api/generate", body={"width": 5})
    assert error.value.status == 422

    def offline(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("no listener"))

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(generate._ServerError, match="no usable answer") as error:
        generate._server_json("http://127.0.0.1:9999", "/api/status")
    assert error.value.status == 0


@pytest.mark.parametrize("model", ["LyliaEngine/waiIllustriousSDXL_v170", "Minthy/RouWei-0.8"])
def test_gan_cli_is_opt_in_and_from_image_can_carry_override_or_drop_it(cli, store, monkeypatch, capsys, model):
    from backend import model_selection, refine

    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    model_selection.select(model)
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, model))
    source = _to_refine(store)
    argv = ["a fox", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--seed", "7"]
    assert generate.main([*argv, "--dry-run"]) == 0
    assert cli.loads == 0 and "upscaler : R-ESRGAN 4x+ Anime6B" in capsys.readouterr().out
    assert generate.main([*argv, "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0].strip()
    assert cli.requests[-1].upscaler == refine.ANIME6B
    assert generate.main(["--from-image", made, "-q"]) == 0
    assert cli.requests[-1].upscaler == refine.ANIME6B
    capsys.readouterr()
    assert generate.main(["--from-image", made, "--upscaler", "lanczos", "-q"]) == 0
    assert cli.requests[-1].upscaler is None
    capsys.readouterr()
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--from-image", made, "-q"]) == 0
    assert cli.requests[-1].refine is None and cli.requests[-1].upscaler is None


def test_gan_cli_refuses_wrong_model_and_oversized_source_before_load(cli, store, monkeypatch, capsys):
    from backend import model_selection, refine

    source = _to_refine(store)
    assert generate.main(["a fox", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--dry-run"]) == 2
    assert "only for WAI" in capsys.readouterr().err and cli.loads == 0
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    model_selection.select("LyliaEngine/waiIllustriousSDXL_v170")
    monkeypatch.setattr(generate, "_project", _project_claiming(generate, "LyliaEngine/waiIllustriousSDXL_v170"))
    monkeypatch.setattr(refine, "check_upscaler_source", lambda image_id: refine.check_upscaler_size(4097, 256))
    assert generate.main(["a fox", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--dry-run"]) == 2
    assert "4.19 MP" in capsys.readouterr().err and cli.loads == 0 and not cli.requests


def test_gan_via_server_carries_the_choice_and_refuses_before_mask_upload(via_server, store, monkeypatch, tmp_path, capsys):
    from backend import model_selection, refine

    model_selection.select("LyliaEngine/waiIllustriousSDXL_v170")
    source = _to_refine(store)
    argv = ["a fox", "--via-server", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--seed", "7", "-q"]
    assert generate.main(argv) == 0
    assert via_server.backend.requests[-1].upscaler == refine.ANIME6B
    body = next(call[2] for call in via_server.calls if call[1] == "/api/generate")
    assert body["upscaler"] == refine.ANIME6B
    mask, _ = _mask_file(tmp_path)
    calls = len(via_server.calls)
    assert generate.main([*argv, "--mask", str(mask)]) == 2
    assert "ordinary Refine" in capsys.readouterr().err
    assert not any(call[1].startswith("/api/masks") or call[1] == "/api/generate" for call in via_server.calls[calls:])
