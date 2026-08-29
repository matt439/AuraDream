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
import os
import random
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Executed as a plain script there is no parent package, so `from . import ...`
# raises ImportError. Put the project root on sys.path and import absolutely,
# which works under both invocations.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MAX_SEED = 2**63 - 1

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
        "-n", "--negative", help="negative prompt; needs CFG above 1 to have any effect"
    )

    size = parser.add_argument_group("size")
    size.add_argument("--aspect", choices=list(SIZES), help="aspect preset (default 1:1)")
    size.add_argument("--mp", type=int, choices=TIERS, help="megapixel tier (default 1)")
    size.add_argument("--width", type=int, help="explicit width, multiple of 16")
    size.add_argument("--height", type=int, help="explicit height, multiple of 16")

    sampling = parser.add_argument_group("sampling (defaults come from the checkpoint profile)")
    sampling.add_argument("--steps", type=int, help="denoising steps")
    sampling.add_argument("--cfg", type=float, metavar="SCALE", help="guidance scale")
    sampling.add_argument("--mu", type=float, help="flow-matching shift; Turbo only")
    sampling.add_argument(
        "--no-mu", action="store_true", help="let the pipeline pick its own shift"
    )
    sampling.add_argument("--seed", type=int, help="base seed; random when omitted")

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

    model = parser.add_argument_group("model (equivalent to the KREA_* variables)")
    model.add_argument("--model", metavar="ID", help="checkpoint id, e.g. krea/Krea-2-Turbo")
    model.add_argument("--quant", choices=("fp8", "int8", "none"), help="weight quantization")
    model.add_argument(
        "--offload", choices=("model", "sequential", "none"), help="offload strategy"
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="only print saved filenames")
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


def _apply_env(args: argparse.Namespace) -> None:
    """Push model overrides into the environment before config is imported.

    `settings` is a frozen dataclass whose fields are read from KREA_* at import
    time, so --model and friends have to land in os.environ first.
    """
    for value, name in (
        (args.model, "KREA_MODEL_ID"),
        (args.quant, "KREA_QUANT"),
        (args.offload, "KREA_OFFLOAD"),
    ):
        if value is not None:
            os.environ[name] = value


def _project() -> SimpleNamespace:
    """Import the project, reporting the wrong interpreter clearly."""
    try:
        from backend import storage
        from backend.backends import Cancelled, get_backend
        from backend.config import settings
        from backend.model_defaults import defaults_for
        from backend.schemas import GenerateRequest
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
        storage=storage,
        Cancelled=Cancelled,
        get_backend=get_backend,
        settings=settings,
        defaults_for=defaults_for,
        GenerateRequest=GenerateRequest,
    )


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    width, height = _resolve_size(args, parser)
    _apply_env(args)
    p = _project()

    profile = p.defaults_for(p.settings.model_id)
    mu = None if args.no_mu else (args.mu if args.mu is not None else profile["mu"])
    seed = args.seed if args.seed is not None else random.randint(0, _MAX_SEED)

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
        # Pydantic's own repr is several lines of type/url noise per field.
        errors = getattr(exc, "errors", None)
        if callable(errors):
            for error in errors():
                field = ".".join(str(part) for part in error["loc"]) or "request"
                print(f"{field}: {error['msg']}", file=sys.stderr)
        else:
            print(exc, file=sys.stderr)
        return 2

    total_images = request.runs * request.num_images
    if not args.quiet:
        print(
            f"model    : {p.settings.model_id} (quant={p.settings.quantization}, "
            f"offload={p.settings.offload})"
        )
        print(f"prompt   : {request.prompt}")
        if request.negative_prompt:
            note = "" if request.guidance_scale > 1.0 else "  [ignored: CFG is at or below 1]"
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
        print(
            f"output   : {total_images} image(s) - {request.runs} run(s) x "
            f"{request.num_images} -> {p.settings.outputs_dir}"
        )
        print()

    if args.dry_run:
        print("Dry run - nothing generated.")
        return 0

    p.storage.init_db()
    backend = p.get_backend("diffusers")
    # This process renders its own status line; the engine's bar would share the
    # line with it and win.
    backend.set_console_progress(False)

    stop = False

    def _on_sigint(_signum, _frame) -> None:
        nonlocal stop
        if stop:
            raise KeyboardInterrupt  # Second Ctrl-C: stop waiting for the step.
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

    if not args.quiet:
        info = backend.describe()
        device = info.get("device_name") or p.settings.device
        print(f"ready in {_duration(time.perf_counter() - load_started)} on {device}\n")

    run_index = 0
    saved = 0
    sweep_started = time.perf_counter()
    # A redrawn status line is for a terminal. Redirected to a log - which is
    # what an overnight sweep does - it is thousands of lines of noise, so there
    # the run announces itself once and says nothing more until it lands.
    live = sys.stdout.isatty() and not args.quiet

    def on_progress(step: int, total: int) -> None:
        if live:
            print(f"\r  run {run_index}/{request.runs}  step {step}/{total}", end="", flush=True)

    def should_cancel() -> bool:
        return stop

    try:
        for index in range(request.runs):
            if stop:
                break
            # Step the seed past the whole of the previous batch so every image in
            # a sweep keeps a seed that reproduces it on its own. Wrap rather than
            # overflow the signed 64-bit seed column.
            run_seed = (seed + index * request.num_images) % (_MAX_SEED + 1)
            run_index = index + 1
            if not args.quiet and not live:
                print(f"  run {run_index}/{request.runs}  seed {run_seed}", flush=True)

            started = time.perf_counter()
            try:
                images = backend.generate(
                    request, seed=run_seed, on_progress=on_progress, should_cancel=should_cancel
                )
            except p.Cancelled:
                if not args.quiet:
                    label = f"  run {run_index}/{request.runs} cancelled"
                    print(("\r" + label + " " * 16) if live else label)
                break
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if live:
                print("\r" + " " * 40 + "\r", end="")

            for offset, image in enumerate(images):
                # Each image in a batch gets its own reproducible seed.
                record = p.storage.save_image(
                    image,
                    request,
                    seed=run_seed + offset,
                    duration_ms=elapsed_ms // max(len(images), 1),
                    model_id=p.settings.model_id,
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
        )
    return 130 if saved < total_images else 0


if __name__ == "__main__":
    raise SystemExit(main())
