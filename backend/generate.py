"""Generate images from the terminal - no server, no browser.

    uv run python -m backend.generate "a lighthouse in fog"
    uv run python -m backend.generate "..." --aspect 16:9 --mp 4 --runs 12
    uv run python -m backend.generate "..." --width 1024 --height 1024 --batch 4 --seed 42
    uv run python -m backend.generate "..." --model krea/Krea-2-Turbo --quant int8

This is the web app minus the web: the same backend, the same per-image seed
maths, the same PNG metadata and the same gallery database, so an image made
here is identical to the one the browser would have made from the same settings
and shows up in the gallery beside it.

Steps, CFG and mu default to the checkpoint's profile (`model_defaults.py`) the
same way the UI seeds its controls, so the only required argument is the prompt.

Ctrl-C stops at the end of the current step and keeps everything already
written - the same cooperative cancel the Cancel button uses. A second Ctrl-C
gives up on waiting for the step.

Running the file directly (`python backend/generate.py`, or an editor's Run
button) works too - see the sys.path bootstrap below.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

# Executed as a plain script there is no parent package, so `from . import ...`
# raises ImportError. Put the project root on sys.path and import absolutely,
# which works under both invocations.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Imported up here rather than in _project(): neither pulls in config, so both
# are safe before apply_env() has had its say about the KREA_* variables.
# launch.py owns --model/--quant/--offload for the server and this CLI alike.
from backend.launch import add_model_arguments, apply_env  # noqa: E402
from backend.seeds import random_seed, wrap_seed  # noqa: E402

# The UI's preset table (web/app.js), kept in step with it. Aspect and megapixel
# tier are independent: every side is a multiple of 16 for the VAE, and the whole
# 4 MP row costs what 2048x2048 costs.
SIZES: dict[str, dict[int, tuple[int, int]]] = {
    "1:1": {1: (1024, 1024), 2: (1440, 1440), 4: (2048, 2048)},
    "3:4": {1: (896, 1200), 2: (1248, 1664), 4: (1776, 2368)},
    "4:3": {1: (1200, 896), 2: (1664, 1248), 4: (2368, 1776)},
    "16:9": {1: (1360, 768), 2: (1920, 1088), 4: (2736, 1536)},
    "9:16": {1: (768, 1360), 2: (1088, 1920), 4: (1536, 2736)},
}
TIERS = (1, 2, 4)


def _build_parser() -> argparse.ArgumentParser:
    sizes = "  ".join(
        f"{aspect} {tier}MP={SIZES[aspect][tier][0]}x{SIZES[aspect][tier][1]}"
        for aspect in SIZES
        for tier in TIERS
    )
    parser = argparse.ArgumentParser(
        prog="python -m backend.generate",
        description="Generate images headlessly. Saves to outputs/ and the gallery database.",
        epilog="sizes: " + sizes,
    )
    parser.add_argument("prompt", help="what to draw")
    parser.add_argument(
        "-n",
        "--negative",
        help="negative prompt; ignored at or below the checkpoint's CFG threshold "
        "(1 on FLUX, 0 on Krea 2)",
    )

    size = parser.add_argument_group("size")
    size.add_argument("--aspect", choices=list(SIZES), help="aspect preset (default 1:1)")
    size.add_argument("--mp", type=int, choices=TIERS, help="megapixel tier (default 1)")
    size.add_argument("--width", type=int, help="explicit width, multiple of 16")
    size.add_argument("--height", type=int, help="explicit height, multiple of 16")

    sampling = parser.add_argument_group("sampling (defaults come from the checkpoint profile)")
    sampling.add_argument("--steps", type=int, help="denoising steps")
    sampling.add_argument("--cfg", type=float, metavar="SCALE", help="guidance scale")
    # Exclusive: `--mu 1.15 --no-mu` used to parse and silently run with no shift.
    shift = sampling.add_mutually_exclusive_group()
    shift.add_argument("--mu", type=float, help="flow-matching shift; Turbo only")
    shift.add_argument("--no-mu", action="store_true", help="let the pipeline pick its own shift")
    sampling.add_argument("--seed", type=int, help="base seed; random when omitted")

    sweep = parser.add_argument_group(
        "grid search (overnight sweeps; each axis takes 8, or 6,8,10, or 6-12x2)"
    )
    sweep.add_argument(
        "--grid-prompt",
        metavar="NAME=A|B",
        action="append",
        default=[],
        help="values for a {NAME} placeholder in the prompt, split on `|`; "
        "repeat the flag for a second variable, and they cross like any other axes",
    )
    sweep.add_argument("--grid-steps", metavar="SPEC", default="", help="step counts to sweep")
    sweep.add_argument("--grid-cfg", metavar="SPEC", default="", help="guidance scales to sweep")
    sweep.add_argument(
        "--grid-mu", metavar="SPEC", default="", help="shifts to sweep; `none` for no shift"
    )
    sweep.add_argument(
        "--grid-seeds",
        type=int,
        default=1,
        metavar="N",
        help="consecutive seeds, each running the whole grid (default 1)",
    )
    sweep.add_argument(
        "--resume",
        action="store_true",
        help="skip cells this prompt has already been rendered at, so a night "
        "that died costs the cell it died on rather than the whole run",
    )

    volume = parser.add_argument_group("volume")
    volume.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="N",
        help="images per pipeline call, all resident in VRAM at once (default 1)",
    )
    volume.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="sequential passes, one image's worth of VRAM each (default 1)",
    )

    add_model_arguments(parser)

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="print only `<path>  seed <n>` per saved image on stdout; everything else goes to stderr",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved settings and exit without loading the model",
    )
    return parser


def _resolve_size(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[int, int]:
    if args.width is not None or args.height is not None:
        if args.aspect is not None or args.mp is not None:
            parser.error("--width/--height and --aspect/--mp are alternatives; pick one")
        if args.width is None or args.height is None:
            parser.error("--width and --height must be given together")
        return args.width, args.height
    return SIZES[args.aspect or "1:1"][args.mp or 1]


def _project() -> SimpleNamespace:
    """Import the project, reporting the wrong interpreter clearly."""
    try:
        from backend import grid, storage
        from backend.backends import Cancelled, get_backend
        from backend.config import settings
        from backend.model_defaults import defaults_for
        from backend.schemas import GenerateRequest, error_detail
    except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
        # Almost always `py generate.py` or a bare `python generate.py`, which
        # picks the system interpreter rather than the project's .venv.
        sys.exit(
            f"Missing dependency: {exc.name}\n\n"
            f"Running under: {sys.executable}\n"
            "That interpreter is not the project environment. Use:\n\n"
            '    uv run python -m backend.generate "your prompt"\n'
        )
    return SimpleNamespace(
        grid=grid,
        storage=storage,
        Cancelled=Cancelled,
        get_backend=get_backend,
        settings=settings,
        defaults_for=defaults_for,
        GenerateRequest=GenerateRequest,
        error_detail=error_detail,
    )


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    width, height = _resolve_size(args, parser)
    apply_env(args)
    p = _project()

    profile = p.defaults_for(p.settings.model_id)
    mu = None if args.no_mu else (args.mu if args.mu is not None else profile["mu"])
    seed = args.seed if args.seed is not None else random_seed()

    try:
        request = p.GenerateRequest(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=width,
            height=height,
            steps=args.steps if args.steps is not None else profile["steps"],
            guidance_scale=args.cfg if args.cfg is not None else profile["guidance_scale"],
            mu=mu,
            seed=seed,
            num_images=args.batch,
            runs=args.runs,
        )
        request.check_budget()
    except ValueError as exc:
        print(p.error_detail(exc), file=sys.stderr)
        return 2

    # A grid and a plain sweep are the same list of pipeline calls; only the axes
    # differ, and with none set `expand` yields exactly the runs loop this used to
    # write by hand - `seed + index * batch`, wrapped by the same helper. Sharing
    # it with the server is the point: a cell here and a cell there must be the
    # same image.
    # One flag per variable on the command line, one line per variable in the
    # browser: the same block reaches the same parser either way.
    prompt_axis = "\n".join(args.grid_prompt)
    gridding = (
        bool(prompt_axis or args.grid_steps or args.grid_cfg or args.grid_mu)
        or args.grid_seeds > 1
    )
    if gridding and args.runs != 1:
        parser.error("--runs and --grid-* are alternatives; a grid steps its own seeds")
    if args.resume and args.seed is None:
        # A fresh random base matches none of the night it is meant to continue.
        parser.error("--resume needs --seed: the seed the run it continues was started from")
    try:
        cells = p.grid.expand(
            request,
            prompt=prompt_axis,
            steps=args.grid_steps,
            cfg=args.grid_cfg,
            mu=args.grid_mu,
            seeds=args.grid_seeds if gridding else args.runs,
        )
    except ValueError as exc:
        # A cell outside the schema raises a ValidationError; same one-line
        # form the browser gets, not pydantic's own several-line repr.
        print(p.error_detail(exc), file=sys.stderr)
        return 2

    # The run's name is the whole grid's, taken before resume subtracts anything,
    # so a night continued from the terminal joins the one it continues - in the
    # gallery's `run:` filter and on the contact sheet.
    run_id = p.grid.run_id(cells, p.settings.model_id, p.settings.quantization)

    # Built before the load, and cheap to: unloaded, `applied()` answers from
    # the checkpoint profile, which is what decides the resume comparison and
    # the negative-prompt note below.
    backend = p.get_backend("diffusers")

    # Resume before the model loads: an already-finished grid should cost the
    # database lookup and nothing else. Compared as the pipeline would run each
    # cell, since that is what the rows were written from.
    skipped = 0
    if args.resume:
        p.storage.init_db()
        done = p.storage.completed_cells(
            p.grid.prompts_of(cells),
            request.width,
            request.height,
            p.settings.model_id,
            p.settings.quantization,
        )
        before = len(cells)
        cells = [
            cell
            for cell in cells
            if p.grid.prune_done(
                [p.grid.Cell(request=backend.applied(cell.request), seed=cell.seed)], done
            )
        ]
        skipped = before - len(cells)

    total_images = len(cells) * request.num_images
    if not args.quiet:
        print(
            f"model    : {p.settings.model_id} (quant={p.settings.quantization}, "
            f"offload={p.settings.offload})"
        )
        print(f"prompt   : {request.prompt}")
        if request.negative_prompt:
            # Where guidance switches on is a property of the checkpoint, not a
            # constant: FLUX bakes guidance_scale into the model and needs true
            # CFG above 1, Krea 2 guides from 0 up. Counted over the cells, not
            # the base CFG: a `--grid-cfg` that crosses the threshold applies
            # it to some cells and not others.
            applying = sum(1 for cell in cells if backend.applied(cell.request).negative_prompt)
            if not profile["negative_prompt_supported"]:
                note = "  [ignored: this checkpoint takes no negative prompt]"
            elif applying == len(cells):
                note = ""
            elif applying == 0:
                note = f"  [ignored: CFG is at or below {profile['cfg_threshold']:g}]"
            else:
                note = (
                    f"  [applies to {applying} of {len(cells)} cells - the ones above "
                    f"CFG {profile['cfg_threshold']:g}]"
                )
            print(f"negative : {request.negative_prompt}{note}")
        print(
            f"size     : {request.width}x{request.height} "
            f"({request.width * request.height / 1e6:.1f} MP)"
        )
        print(
            f"sampling : {request.steps} steps, CFG {request.guidance_scale}, "
            f"mu {'-' if request.mu is None else request.mu}"
        )
        print(f"seed     : {seed}" + ("" if args.seed is not None else "  (random)"))
        print(f"run      : {run_id}")
        if gridding:
            values = p.grid.axis_values(
                request, args.grid_steps, args.grid_cfg, args.grid_mu, prompt_axis
            )
            if len(values["prompt"]) > 1:
                # The substituted prompts, not the variables that made them:
                # what went wrong in a prompt grid is visible in the result.
                print(f"  prompts: {len(values['prompt'])}")
                for text in values["prompt"][:4]:
                    print(f"           {text}")
                if len(values["prompt"]) > 4:
                    print(f"           ... and {len(values['prompt']) - 4} more")
            for axis, shown in (("steps", "steps"), ("guidance_scale", "CFG"), ("mu", "mu")):
                listed = ", ".join("off" if v is None else f"{v:g}" for v in values[axis])
                print(f"  {shown:<7}: {listed}")
            print(f"  seeds  : {args.grid_seeds} from {seed}")
            print(f"grid     : {p.grid.describe(values, args.grid_seeds)} = {len(cells)} cell(s)")
        if skipped:
            print(f"resume   : {skipped} cell(s) already rendered, skipping")
        print(
            f"output   : {total_images} image(s) - {len(cells)} run(s) x "
            f"{request.num_images} -> {p.settings.outputs_dir}"
        )
        print()

    # To stderr: `--quiet > paths.txt` promises stdout holds nothing but paths.
    if args.dry_run:
        print("Dry run - nothing generated.", file=sys.stderr)
        return 0

    if not cells:
        print("Nothing to do: every cell is already in the gallery.", file=sys.stderr)
        return 0

    p.storage.init_db()
    # Every image of this invocation carries it, so a CLI grid groups in the
    # gallery exactly as a browser one does.
    job_id = uuid.uuid4().hex[:12]
    # This process renders its own status line; the engine's bar would share the
    # line with it and win.
    backend.set_console_progress(False)

    stop = False
    loading = True

    def _on_sigint(_signum, _frame) -> None:
        nonlocal stop
        # During the load there is no step to finish and nothing worth keeping:
        # the first Ctrl-C ends it outright rather than promising to stop "at
        # the end of this step" and then waiting out the download.
        if stop or loading:
            raise KeyboardInterrupt
        stop = True
        print("\ncancelling at the end of this step (Ctrl-C again to force)...", file=sys.stderr)

    signal.signal(signal.SIGINT, _on_sigint)

    if not args.quiet:
        print("loading weights (minutes on a cold cache)...", flush=True)
    load_started = time.perf_counter()
    try:
        backend.load()
    except KeyboardInterrupt:
        print("\nInterrupted during load; nothing generated.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not crash
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    loading = False
    if not args.quiet:
        info = backend.describe()
        device = info.get("device_name") or p.settings.device
        print(f"ready in {_duration(time.perf_counter() - load_started)} on {device}\n")

    run_index = 0
    saved = 0
    failed = 0
    consecutive = 0
    sweep_started = time.perf_counter()
    # A redrawn status line is for a terminal. Redirected to a log - which is
    # what an overnight sweep does - it is thousands of lines of noise, so there
    # the run announces itself once and says nothing more until it lands.
    live = sys.stdout.isatty() and not args.quiet

    def on_progress(step: int, total: int) -> None:
        if live:
            print(f"\r  run {run_index}/{len(cells)}  step {step}/{total}", end="", flush=True)

    def should_cancel() -> bool:
        return stop

    try:
        for index, cell in enumerate(cells):
            if stop:
                break
            run_seed = cell.seed
            run_index = index + 1
            # Every cell names its settings in a log - which is what an overnight
            # grid writes to - so a night that ends badly still says what it got
            # through and at which values.
            if not args.quiet and not live:
                varied = f"  {cell.label}" if cell.label else ""
                print(f"  run {run_index}/{len(cells)}  seed {run_seed}{varied}", flush=True)

            started = time.perf_counter()
            try:
                images = backend.generate(
                    cell.request,
                    seed=run_seed,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                )
            except p.Cancelled:
                if not args.quiet:
                    label = f"  run {run_index}/{len(cells)} cancelled"
                    print(("\r" + label + " " * 16) if live else label)
                break
            except Exception as exc:  # noqa: BLE001 - one cell's failure, not the night's
                # The same rule as the server's job loop: skip the cell and go
                # on, up to a few in a row or a third of the run, so a transient
                # OOM at 2am costs one image rather than every cell after it.
                failed += 1
                consecutive += 1
                if live:
                    print("\r" + " " * 40 + "\r", end="")
                print(
                    f"  run {run_index}/{len(cells)} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if consecutive >= 3 or failed > max(3, len(cells) // 3):
                    raise
                continue
            consecutive = 0
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if live:
                print("\r" + " " * 40 + "\r", end="")
            # What ran, not what was asked - the same rule the server follows.
            ran = backend.applied(cell.request)

            for offset, image in enumerate(images):
                # Each image in a batch gets its own reproducible seed, wrapped
                # with the same helper the generator uses so the recorded seed
                # and the seed actually drawn cannot disagree.
                record = p.storage.save_image(
                    image,
                    ran,
                    seed=wrap_seed(run_seed + offset),
                    duration_ms=elapsed_ms // max(len(images), 1),
                    model_id=p.settings.model_id,
                    job_id=job_id,
                    run_id=run_id,
                    quantization=p.settings.quantization,
                )
                saved += 1
                print(
                    f"{p.settings.outputs_dir / record.filename}  seed {record.seed}", flush=True
                )
    except KeyboardInterrupt:
        print("\nForced stop.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not crash
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        backend.unload()

    if not args.quiet:
        print(
            f"\n{saved}/{total_images} image(s) in "
            f"{_duration(time.perf_counter() - sweep_started)}."
            + (f" {failed} cell(s) failed and were skipped." if failed else "")
        )
    if failed:
        return 1
    return 130 if saved < total_images else 0


if __name__ == "__main__":
    raise SystemExit(main())
