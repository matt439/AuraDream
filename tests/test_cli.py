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
from backend import config, generate, storage

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
