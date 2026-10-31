"""Generate images from the terminal, directly or through the running server.

    uv run python -m backend.generate "a lighthouse in fog"
    uv run python -m backend.generate "..." --aspect 16:9 --mp 4 --runs 12
    uv run python -m backend.generate "..." --width 1024 --height 1024 --batch 4 --seed 42
    uv run python -m backend.generate "..." --model krea/Krea-2-Turbo --quant int8
    uv run python -m backend.generate --from-image outputs/x.png --refine 412 --scale 1.5
    uv run python -m backend.generate "a red scarf" --refine 412 --mask scarf-mask.png

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
import dataclasses
import math
import os
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

# Imported up here rather than in _project(): none of these pulls in config, so
# all are safe before apply_env() has had its say about AURADREAM_* variables.
# The argument helpers are module globals here, by name, so patching
# `generate._image_settings` or `generate._mask_file` still reaches main().
from backend.generate_args import (  # noqa: E402
    _apply_image_settings,
    _build_parser,
    _cli_execution_intent,
    _grid_flags,
    _image_settings,
    _image_variables,
    _mask_file,
    _outpaint_sides,
    _resolve_size,
    _then_refine_flags,
)
from backend.generate_remote import _via_server  # noqa: E402
from backend.launch import apply_env, warn_legacy_env  # noqa: E402
from backend.seeds import random_seed, wrap_seed  # noqa: E402


def _check_carried_source(args: argparse.Namespace, p: SimpleNamespace, parser: argparse.ArgumentParser) -> bool:
    """Drop a refine source --from-image carried that is not, in this
    gallery, the picture the PNG was made from; answers whether it did.

    The id is the gallery's that wrote the PNG - another one's, for a file
    from WSL's gallery or a backup - so it can name any picture here, and a
    same-size one passes every later check. An inpaint composites that
    source's own pixels back, so the result would visibly be someone else's
    picture. Held as Load settings from PNG holds references
    (`api_png._verified_references`): the record's `refine_source` hash and
    size against the row's. A record without one (older files) carries as
    before, and so does a row with no hash of its pixels to compare; a
    source the command line names is the user's own choice.
    """
    known = args.carried_source
    if known is None or args.refine is None:
        return False
    record = (
        p.storage.get_image(args.refine, include_binned=True) if p.settings.db_path.exists() else None
    )
    if record is None or not record.content_hash or (
        (record.content_hash, record.width, record.height)
        == (known.get("hash"), known.get("width"), known.get("height"))
    ):
        # No row under the id is refused in words below, as it always was.
        return False
    said = "is not the image"
    what = (
        "an outpaint" if args.carried_outpaint
        else "an inpaint" if args.mask_hash is not None else "a refine pass"
    )
    named_outpaint = args.outpaint_given or args.band_given
    if (
        (args.strength is not None and not args.carried_strength) or args.scale is not None
        or args.mask is not None or named_outpaint
        or (args.upscaler is not None and not args.carried_upscaler)
        or (args.crop_anchor not in (None, "center") and not args.carried_crop_anchor)
    ):
        parser.error(
            f"--from-image: source image {args.refine} {said} the PNG was made from, so it cannot "
            f"be run again as {what}; name the source with --refine"
        )
    print(
        f"note: source image {args.refine} {said} the PNG was made from; "
        f"running as a plain generation instead of {what}",
        file=sys.stderr,
    )
    args.refine = None
    args.crop_anchor = None
    args.mask_hash = None
    # An outpaint goes with its source, or the run would be a plain render
    # at the canvas's size.
    if args.carried_outpaint:
        args.outpaint = None
        args.outpaint_fill = None
        args.carried_outpaint = False
        args.carried_outpaint_mask = None
        args.band = None
    if args.carried_strength:
        args.strength = None
    if args.carried_upscaler:
        args.upscaler = None
    return True


def _mask_from_record(path: str, digest: str):
    """The mask a --from-image PNG carries in its own `auradream-mask`
    chunk (or Local Diffusion's legacy one), where it is the one its record
    names, else None."""
    import base64
    import binascii
    import io

    from PIL import Image

    from backend.metadata import embedded_mask
    from backend.storage import mask_hash

    try:
        with Image.open(path) as image:
            raw = embedded_mask(image.info)
        if raw is None:
            return None
        with Image.open(io.BytesIO(base64.b64decode(raw, validate=True))) as embedded:
            embedded.load()
            if embedded.mode != "L" or mask_hash(embedded) != digest:
                return None
            return embedded.copy()
    except (OSError, ValueError, binascii.Error):
        return None


def _server_holds_gpu(host: str, port: int, timeout: float = 2.0) -> str | None:
    """The checkpoint a running server has on the GPU, or None when none does.

    The CLI builds its own backend and loads a second copy of the weights into
    the same card, with nothing to say a server already has ~8 GB of them
    resident - so a sweep started beside one spent minutes loading and OOMed at
    the first step. Probing `/api/status` rather than the port: the model is
    deliberately not loaded at start-up, so a listener alone is the ordinary
    harmless state. Refuses only when the state is ready or loading; a timeout
    is a server too busy to answer, which is the case to refuse. Advisory:
    one-directional, and blind across the WSL/Windows boundary.
    """
    import json
    import urllib.error
    import urllib.request

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with urllib.request.urlopen(f"http://{probe_host}:{port}/api/status", timeout=timeout) as resp:
            status = json.load(resp)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return None  # no server at all
        return "a server that did not answer in time"
    except (OSError, ValueError):
        return "a server that did not answer in time"
    if status.get("state") in ("ready", "loading"):
        return f"{status.get('model_id', 'a checkpoint')} ({status['state']})"
    return None


def _then_refine_passes(p, backend, records, spec, pass_run, job_id, on_progress, should_cancel, streak,
                        *, quiet: bool, live: bool):
    """A pass of each record a cell just saved, as the server's worker runs
    them (`JobManager._passes`): Mass refine's of that record, saved under
    the pass run; a failure keeps its base, and three in a row end the run.
    (records made, passes failed, streak)."""
    from backend.refine_batch import QUEUED, plan

    made, failed = [], 0
    for record in records:
        if should_cancel():
            break
        outcome = plan(
            record, running=p.settings.model_id, profile=p.defaults_for(p.settings.model_id),
            scale=spec.scale, strength=spec.strength, steps=spec.steps, include_foreign=False,
            skip_passes=False,
        )
        if not quiet and not live:
            print(f"  pass of #{record.id}", flush=True)
        started = time.perf_counter()
        try:
            if outcome.status != QUEUED:
                raise ValueError(outcome.detail or outcome.status)
            cell = outcome.cell
            images = backend.generate(
                cell.request, seed=cell.seed, on_progress=on_progress, should_cancel=should_cancel,
            )
        except p.Cancelled:
            break
        except Exception as exc:  # noqa: BLE001 - one pass's failure, not the night's
            failed += 1
            streak += 1
            if live:
                print("\r" + " " * 40 + "\r", end="")
            print(f"  pass of #{record.id} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if streak >= 3:
                raise
            continue
        streak = 0
        if live:
            print("\r" + " " * 40 + "\r", end="")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        ran = backend.applied(cell.request)
        peak = backend.last_peak()
        for image in images:
            saved = p.storage.save_image(
                image, ran, seed=cell.seed, duration_ms=elapsed_ms // max(len(images), 1),
                model_id=p.settings.model_id, job_id=job_id, run_id=pass_run,
                quantization=p.settings.quantization, variables=cell.variables, batch=1, peak=peak,
            )
            made.append(saved)
            print(f"{p.settings.outputs_dir / saved.filename}  seed {saved.seed}  pass of #{record.id}", flush=True)
    return made, failed, streak


def _project() -> SimpleNamespace:
    """Import the project, reporting the wrong interpreter clearly."""
    try:
        from backend import gallery_lock, gallery_reports, grid, loras, outpaint, refine, storage, submission_rules
        from backend.backends import Cancelled, get_backend
        from backend.config import settings
        from backend.model_defaults import defaults_for
        from backend.schemas import GenerateRequest, Outpaint, PromptVariant, error_detail
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
        gallery_lock=gallery_lock,
        gallery_reports=gallery_reports,
        grid=grid,
        loras=loras,
        outpaint=outpaint,
        refine=refine,
        storage=storage,
        submission_rules=submission_rules,
        Cancelled=Cancelled,
        get_backend=get_backend,
        settings=settings,
        defaults_for=defaults_for,
        GenerateRequest=GenerateRequest,
        Outpaint=Outpaint,
        PromptVariant=PromptVariant,
        error_detail=error_detail,
    )


def _mask_unusable(p: SimpleNamespace, request, mask_image, *, store: bool) -> str | None:
    """Why the inpaint's mask cannot run over its source, or None - the
    server's submit check (`submission_rules.mask_unusable`), after keeping the
    mask where this run has its picture (--mask, or the PNG's own copy) and
    is no dry run. A dry run holds the picture's size to the source and the
    request instead, and writes nothing."""
    if mask_image is not None and not store:
        source = p.storage.get_image(request.refine)
        if source is not None and mask_image.size != (source.width, source.height):
            return (
                f"mask {request.mask[:16]} is {mask_image.width}x{mask_image.height}, but source image "
                f"{request.refine} is {source.width}x{source.height}; paint the mask on that image again"
            )
        if mask_image.size != (request.width, request.height):
            return (
                f"an inpaint runs at its source's size: the mask is {mask_image.width}x"
                f"{mask_image.height}, not {request.width}x{request.height}"
            )
        return None
    if mask_image is not None:
        p.storage.store_mask(mask_image)
    elif not store:
        # A dry run of a recorded mask found nowhere: the real run says so.
        try:
            p.storage.mask_path(request.mask)
        except ValueError as exc:
            return str(exc)
        return None
    from backend.submission_rules import mask_unusable

    return mask_unusable(request)


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _resolved_cli_execution(args, selected_loader: object, source: dict | None):
    """Bind composer/replay identity before sources or grid identities."""
    from backend.execution import resolve_execution

    cache = getattr(args, "step_cache", None)
    if getattr(args, "execution_explicit", False):
        resolve_execution(selected_loader, getattr(args, "execution_recipe", None))
    if source is not None:
        recorded = source.get("execution")
        if cache is not None:
            return resolve_execution(selected_loader, recorded, step_cache=None if cache == "off" else cache)
        return resolve_execution(selected_loader, recorded)
    if cache is not None:
        return resolve_execution(selected_loader, step_cache=None if cache == "off" else cache)
    return resolve_execution(selected_loader)


def main(argv: list[str] | None = None) -> int:
    # Redirected to an overnight log, stdout on Windows encodes with the
    # locale's code page, where a CJK or emoji prompt was a UnicodeEncodeError
    # at the header, before anything ran. So does `> NUL`, which answers
    # isatty(); the Windows console is UTF-8 already. Elsewhere a terminal
    # keeps its own encoding. A StringIO has none to change, and under pythonw
    # there is no stdout.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure) and (sys.platform == "win32" or not sys.stdout.isatty()):
        reconfigure(encoding="utf-8", errors="backslashreplace")
    warn_legacy_env()  # a pre-rename variable left in a shell: named, not read
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.execution_explicit = _cli_execution_intent(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.reference_given = bool(args.reference)
    # A mask hash carried from --from-image's record, where --mask names none;
    # the record's refine source's identity, and whether its strength carried.
    args.mask_hash = None
    args.carried_source = None
    args.carried_strength = False
    args.carried_upscaler = False
    args.carried_crop_anchor = False
    # An outpaint the command line names, or one --from-image carried: its
    # fill (this build's for one named here), and the record's mask hash,
    # kept aside to hold the rebuilt mask to.
    args.outpaint_given = args.outpaint is not None
    args.band_given = args.band is not None
    args.outpaint_fill = None
    args.carried_outpaint = False
    args.carried_outpaint_mask = None
    # Whether the command line itself names a size, before a PNG fills one in:
    # --scale is an alternative to these, but not to the size of --from-image.
    size_flags = any(v is not None for v in (args.width, args.height, args.aspect, args.mp))
    # The PNG first, so its checkpoint can reach AURADREAM_MODEL_ID before the
    # settings freeze on import.
    source = _image_settings(args.from_image, parser) if args.from_image else None
    if source is not None:
        _apply_image_settings(args, source)
    if not args.prompt:
        parser.error("a prompt is required (or --from-image with a gallery PNG)")
    if args.refine is None:
        # Neither is a setting without a source, and a run that quietly
        # ignored one would be a plain render under a refine's flags.
        if args.strength is not None:
            parser.error("--strength needs --refine: there is no source to re-denoise")
        if args.scale is not None:
            parser.error("--scale needs --refine: it multiplies the source's size")
        if args.mask is not None:
            parser.error("--mask needs --refine: a mask covers the image it was painted on")
        if args.upscaler is not None:
            parser.error("--upscaler needs --refine: there is no source to upscale")
        if args.crop_anchor not in (None, "center"):
            parser.error("--crop-anchor needs --refine: there is no source to crop")
    if args.outpaint is None and args.band is not None:
        parser.error("--band needs --outpaint: the band is how far an outpaint's mask reaches into the original")
    outpaint_sides = None
    if args.outpaint is not None:
        if args.refine is None:
            parser.error("--outpaint needs --refine: an outpaint extends a gallery image")
        if args.mask is not None:
            parser.error("--outpaint and --mask are alternatives: an outpaint builds its own mask")
        if args.scale is not None:
            parser.error("--outpaint and --scale are alternatives: an outpaint runs at its source's size plus its padding")
        # The user's own size flags, never the ones a PNG filled in: the
        # canvas is always the source's size plus the padding.
        if size_flags:
            parser.error(
                "--outpaint and --width/--height/--aspect/--mp are alternatives: an outpaint runs at "
                "its source's size plus its padding"
            )
        outpaint_sides = _outpaint_sides(args.outpaint, parser)
    inpainting = args.mask is not None or args.mask_hash is not None
    if args.crop_anchor not in (None, "center") and (inpainting or outpaint_sides is not None):
        parser.error("--crop-anchor is for an ordinary --refine pass, without --mask or --outpaint")
    if inpainting and args.scale is not None:
        parser.error("--scale and --mask are alternatives: an inpaint runs at its source's own size")
    if args.scale is not None:
        if size_flags:
            parser.error("--scale and --width/--height/--aspect/--mp are alternatives; pick one")
        if not math.isfinite(args.scale) or args.scale <= 0:
            parser.error("--scale must be finite and above zero")
    # A size from the flags or from the PNG; a refine pass with neither takes
    # the source's own, scaled, once the gallery can be read.
    sized = any(v is not None for v in (args.width, args.height, args.aspect, args.mp))
    carried = _image_variables(args, source) if source is not None else None
    mask_image = _mask_file(args.mask, parser) if args.mask is not None else None
    width, height = _resolve_size(args, parser)
    args.then_refine = _then_refine_flags(args, parser)
    apply_env(args)
    if args.via_server:
        return _via_server(
            args, parser, source, carried, mask_image, width, height, sized, outpaint_sides,
            check_carried_source=_check_carried_source, mask_from_record=_mask_from_record,
            resolved_execution=_resolved_cli_execution,
        )
    # Before the project, the backend or the gallery: this command renders
    # from text, which an edit-only checkpoint never does, whatever its gate.
    from backend.instruction_edit import command_line_refusal
    from backend.model_defaults import DEFAULT_MODEL_ID

    refusal = command_line_refusal(os.environ.get("AURADREAM_MODEL_ID") or DEFAULT_MODEL_ID)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    p = _project()
    from backend.execution import require_configured_loader_available

    try:
        require_configured_loader_available(p.settings)
        effective_execution = _resolved_cli_execution(args, getattr(p.settings, "loader_execution", None), source)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if _check_carried_source(args, p, parser):
        inpainting = False
        outpaint_sides = None
    outpainting = outpaint_sides is not None

    # A checkpoint with no refine pass is refused first, by the profile and
    # before anything is built from the flags: a strength that also happens
    # to leave no step of that checkpoint's few would otherwise be the error
    # reported, about a pass that could never have run.
    if (inpainting or outpainting) and not p.defaults_for(p.settings.model_id)["inpaint_supported"]:
        # The inpaint class's question, as `applied()` asks it with a mask
        # or an outpaint.
        through = ", which an outpaint runs through" if outpainting else ""
        print(
            f"{p.settings.model_id} has no inpaint in this app - its pipeline has no inpaint class "
            f"in the installed diffusers{through}, or its family's is not enabled yet. The FLUX.1, SDXL "
            "and Z-Image checkpoints have one.",
            file=sys.stderr,
        )
        return 2
    if (
        not inpainting and not outpainting and args.refine is not None
        and not p.defaults_for(p.settings.model_id)["refine_supported"]
    ):
        print(
            f"{p.settings.model_id} has no refine pass in this app - its integration has no "
            "compatible image-to-image counterpart. The FLUX.1, SDXL, Z-Image and Qwen-Image checkpoints "
            "have one.",
            file=sys.stderr,
        )
        return 2

    refined_from = None
    outpaint_spec = None
    if args.upscaler not in (None, "lanczos"):
        reason = p.refine.upscaler_reason(SimpleNamespace(
            upscaler=args.upscaler, refine=args.refine, mask=args.mask or args.mask_hash,
            outpaint=outpaint_sides,
        ), p.settings.model_id)
        if reason:
            print(reason, file=sys.stderr)
            return 2
    if args.refine is not None and (args.scale is not None or not sized or inpainting or outpainting):
        # Only against a gallery that exists, like the reference check below:
        # a --dry-run promises to create nothing, the database included.
        # Binned too: the source check below then says it is in the recycle
        # bin, where this would have said there is no such image.
        refined_from = (
            p.storage.get_image(args.refine, include_binned=True)
            if p.settings.db_path.exists() else None
        )
        if refined_from is None:
            print(
                f"no gallery image with id {args.refine} in {p.settings.db_path} to refine",
                file=sys.stderr,
            )
            return 2
        if outpainting:
            # The canvas is the source's size plus the padding, whatever size
            # a PNG carried; an off-grid source is named before any request.
            off_grid = p.outpaint.source_off_grid(
                refined_from.width, refined_from.height, f"#{refined_from.id}"
            )
            if off_grid is not None:
                print(off_grid, file=sys.stderr)
                return 2
            if isinstance(outpaint_sides, str):
                try:
                    chosen = p.outpaint.preset(
                        refined_from.width, refined_from.height, outpaint_sides,
                        max_side=p.settings.refine_max_side, max_pixels=p.settings.refine_max_pixels,
                        name=f"#{refined_from.id}",
                    )
                except ValueError as exc:
                    parser.error(f"--outpaint: {exc}")
                if chosen["held"] is not None:
                    print(f"--outpaint {outpaint_sides}: {chosen['held']}", file=sys.stderr)
                    return 2
                outpaint_sides = (chosen["left"], chosen["top"], chosen["right"], chosen["bottom"])
            fill = args.outpaint_fill or p.outpaint.FILL
            if fill not in p.outpaint.FILLS:
                print(p.outpaint.unknown_fill(fill), file=sys.stderr)
                return 2
            left, top, right, bottom = outpaint_sides
            try:
                outpaint_spec = p.Outpaint(
                    fill=fill, left=left, top=top, right=right, bottom=bottom,
                    band=args.band if args.band is not None else p.outpaint.BAND,
                )
            except ValueError as exc:
                print(f"--outpaint: {p.error_detail(exc)}", file=sys.stderr)
                return 2
            width, height = refined_from.width + left + right, refined_from.height + top + bottom
        elif inpainting:
            # The mask was painted on the source's pixels: its own size, never
            # scaled. A size given that differs is refused with the mask below.
            if not sized:
                width, height = refined_from.width, refined_from.height
        else:
            try:
                width, height = p.refine.scaled(
                    refined_from.width, refined_from.height, args.scale or p.refine.DEFAULT_SCALE
                )
            except (ValueError, OverflowError):
                parser.error("--scale produces a size outside the supported range")

    # The mask's name. Kept (`storage.store_mask`, the upload route's own
    # writer) only once the run is past its checks and is no dry run.
    mask = None
    if mask_image is not None:
        mask = p.storage.mask_hash(mask_image)
    elif args.mask_hash is not None:
        mask = args.mask_hash
        # Not held here (a PNG from another gallery, or a mask since swept):
        # the PNG's own copy is the same picture, by the hash.
        mask_image = _mask_from_record(args.from_image, mask)

    profile = p.defaults_for(p.settings.model_id)
    mu = None if args.no_mu else (args.mu if args.mu is not None else profile["mu"])
    seed = args.seed if args.seed is not None else random_seed()

    try:
        # The same parser the axes use, so `--reference 12,15` and `12-14` mean
        # what they do everywhere else; the schema's own bound refuses a fifth.
        references = p.grid.parse_axis(args.reference, int, label="reference")
    except ValueError as exc:
        print(p.error_detail(exc), file=sys.stderr)
        return 2

    # A grid and a plain sweep are the same list of pipeline calls; only the axes
    # differ, and with none set `expand` yields exactly the runs loop this used to
    # write by hand - `seed + index * batch`, wrapped by the same helper. Sharing
    # it with the server is the point: a cell here and a cell there must be the
    # same image.
    prompt_axis, gridding = _grid_flags(args, parser)

    try:
        request = p.GenerateRequest(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=width,
            height=height,
            steps=args.steps if args.steps is not None else profile["steps"],
            guidance_scale=args.cfg if args.cfg is not None else profile["guidance_scale"],
            true_cfg=args.true_cfg,
            mu=mu,
            reference=references,
            lora=args.lora,
            lora_scale=args.lora_scale,
            refine=args.refine,
            crop_anchor=args.crop_anchor or "center",
            upscaler=None if args.upscaler == "lanczos" else args.upscaler,
            # None is "not given": the checkpoint's own starting strength.
            strength=args.strength,
            mask=mask,
            outpaint=outpaint_spec,
            seed=seed,
            num_images=args.batch,
            runs=args.runs,
            execution=effective_execution,
        )
        # A batch past the budget is told the flag to use instead, which on a
        # grid - whose seeds are its runs - is a smaller --batch.
        budget_request = request.model_copy(update={"num_images": 1}) if args.grid_prompt_mode == "random" else request
        if not args.grid_size.strip():
            budget_request.check_budget(grid=gridding, cli=True)
    except ValueError as exc:
        print(p.error_detail(exc), file=sys.stderr)
        return 2

    # A comparison's version, reproduced, is a one-variant prompt axis: taken
    # verbatim as the grid took it - a `{name}` in it is text, not a
    # placeholder with no values - and recorded under its label by the same
    # path. A `{var}` grid's assignment is already substituted into the prompt
    # it made, so it rides each cell as it is - and so does a `variant` that is
    # no label a comparison could have recorded: a user's own grid may name its
    # variable that, and `none`'s empty value or a long one fails the schema's
    # label (the prompt cannot, having passed the request's same bounds).
    variants = None
    if carried is not None and set(carried) == {p.grid.VARIANT}:
        try:
            variants = [p.PromptVariant(label=carried[p.grid.VARIANT], prompt=request.prompt)]
        except ValueError:
            variants = None
    try:
        seed_list = p.grid.parse_seeds(args.seed_list)
        cells = p.grid.expand(
            request,
            prompt=prompt_axis,
            steps=args.grid_steps,
            cfg=args.grid_cfg,
            mu=args.grid_mu,
            lora_scale=args.grid_lora_scale,
            strength=args.grid_strength,
            seeds=args.grid_seeds if gridding else args.runs,
            seed_list=seed_list or None,
            variants=variants,
            prompt_mode=args.grid_prompt_mode,
            size=args.grid_size,
            cli=True,
        )
        if carried is not None and variants is None:
            # Leading the log line too, as they do in a grid's own cells.
            named = ", ".join(f"{name}={value or 'none'}" for name, value in carried.items())
            named = named.encode("ascii", "replace").decode("ascii")
            cells = [
                dataclasses.replace(cell, variables=carried, label=f"{named}, {cell.label}")
                for cell in cells
            ]
    except ValueError as exc:
        # A cell outside the schema raises a ValidationError; same one-line
        # form the browser gets, not pydantic's own several-line repr.
        print(p.error_detail(exc), file=sys.stderr)
        return 2

    # The run's name is the whole grid's, taken before resume subtracts anything,
    # so a night continued from the terminal joins the one it continues - in the
    # gallery's `run:` filter and on the contact sheet.
    run_id = p.grid.run_id(cells, p.settings.model_id, p.settings.quantization)
    axis_cells = cells  # Full sampled prompt list, before Resume subtracts cells.

    # Built before the load, and cheap to: unloaded, `applied()` answers from
    # the checkpoint profile, which is what decides the resume comparison and
    # the negative-prompt note below.
    backend = p.get_backend("diffusers")

    # References, adapters and a refine source are all-or-nothing inputs: a
    # night that renders every cell as plain text-to-image because the pipeline
    # ignores `image`, because the adapter file is not there, or because the
    # checkpoint has no refine pass, looks like a success. Refused here for the
    # same reason the server refuses at submit - before the load, not at cell 1.
    # (Whether the checkpoint has a refine pass at all was settled above,
    # ahead of the request.)
    if args.refine is not None:
        if not p.settings.db_path.exists():
            print(
                f"no gallery database at {p.settings.db_path}, so there is no image "
                f"{args.refine} to refine",
                file=sys.stderr,
            )
            return 2
        try:
            p.storage.reference_paths([args.refine], "source image")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if request.upscaler is not None:
            try:
                p.refine.check_upscaler_source(args.refine)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
    if mask is not None:
        for group in p.submission_rules.cells_by_size(cells).values():
            reason = _mask_unusable(p, group[0].request, mask_image, store=not args.dry_run)
            if reason is not None:
                print(reason, file=sys.stderr)
                return 2
    if request.outpaint is not None:
        # The server's submit check, and - for one --from-image carried - the
        # mask this build builds held to the one the image was made under.
        from backend.submission_rules import outpaint_unusable

        for group in p.submission_rules.cells_by_size(cells).values():
            reason = outpaint_unusable(group[0].request)
            if reason is not None:
                print(reason, file=sys.stderr)
                return 2
        if args.carried_outpaint_mask is not None and p.storage.outpaint_mask_hash(
            request.width, request.height, request.outpaint
        ) != args.carried_outpaint_mask:
            print(
                "--from-image: this build builds an outpaint's mask differently from the one that "
                "made this image",
                file=sys.stderr,
            )
            return 2
    if references and not profile["reference_supported"]:
        print(
            f"{p.settings.model_id} takes no reference images - its pipeline has no `image` "
            "parameter. FLUX.2 [klein] is the checkpoint here that conditions on them.",
            file=sys.stderr,
        )
        return 2
    if references:
        # Only against a gallery that exists, like the resume lookup: a
        # --dry-run promises to create nothing, the database included.
        if not p.settings.db_path.exists():
            print(
                f"no gallery database at {p.settings.db_path}, so there is no image "
                f"{references[0]} to condition on",
                file=sys.stderr,
            )
            return 2
        try:
            p.storage.reference_paths(references)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.lora:
        reason = backend.lora_unavailable_reason()
        if reason is not None:
            print(reason, file=sys.stderr)
            return 2
        try:
            p.loras.path_for(args.lora)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        reason = backend.lora_mismatch(args.lora)
        if reason is not None:
            print(reason, file=sys.stderr)
            return 2

    # Resume before the model loads: an already-finished grid should cost the
    # database lookup and nothing else. Compared as the pipeline would run each
    # cell, since that is what the rows were written from.
    skipped = 0
    # Only against a gallery that exists: `completed_cells` connects, and a
    # connection writes the whole schema into any file it opens - which a
    # --dry-run promises not to do. No database means nothing is done yet.
    if args.resume and p.settings.db_path.exists():
        # The lookup's connection checks the schema. Leave output setup and
        # thumbnail migration to init_db() below, after the dry-run guard.
        try:
            done = set()
            for (actual_width, actual_height), group in p.submission_rules.cells_by_size(cells).items():
                done.update(p.storage.completed_cells(
                    p.grid.prompts_of(group), actual_width, actual_height,
                    p.settings.model_id, p.settings.quantization,
                ))
        except p.gallery_lock.GalleryBusy as exc:
            # Nothing is rendered yet, so nothing is lost: say so, not a traceback.
            print(f"--resume: {exc}", file=sys.stderr)
            return 1
        before = len(cells)
        cells = [
            cell
            for cell in cells
            if p.grid.prune_done(
                [p.grid.Cell(request=backend.applied(cell.request), seed=cell.seed)], done
            )
        ]
        skipped = before - len(cells)

    total_images = sum(cell.request.num_images for cell in cells)
    # Then refine: the same checks the server's submit makes, and the run the
    # passes are saved under; a pass of each image follows its cell below.
    spec = pass_run = None
    if args.then_refine is not None and cells:
        from backend import then_refine
        from backend.schemas import ThenRefine

        spec = then_refine.resolved(ThenRefine(**args.then_refine))
        unavailable = None if p.defaults_for(p.settings.model_id).get("refine_supported", True) else (
            f"{p.settings.model_id} has no refine pass in this app"
        )
        reason = then_refine.refusal([cell.request for cell in cells], spec, unavailable)
        if reason is not None:
            print(f"--then-refine-scale: {reason}", file=sys.stderr)
            return 2
        pass_run = then_refine.pass_run_id(run_id, spec)
        total_images *= 2
    if not args.quiet:
        if source is not None:
            made_with = source.get("quantization")
            print(
                f"from     : {args.from_image}"
                + (f"  (job {source['job']}" if source.get("job") else "")
                + (f", run {source['run']})" if source.get("run") else (")" if source.get("job") else ""))
            )
            if made_with and made_with != p.settings.quantization:
                print(
                    f"           made with quant={made_with}; this run uses "
                    f"{p.settings.quantization}, so the pixels will differ"
                )
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
            # Which knob decides: on FLUX.1 the real CFG is --true-cfg and the
            # CFG axis has no say; everywhere else --cfg is the real CFG.
            knob = "True CFG" if profile["true_cfg_supported"] else "CFG"
            if not profile["negative_prompt_supported"]:
                note = "  [ignored: this checkpoint takes no negative prompt]"
            elif applying == len(cells):
                note = ""
            elif applying == 0:
                note = (
                    f"  [ignored: {knob} is at or below {profile['cfg_threshold']:g}"
                    + (" - set --true-cfg]" if knob == "True CFG" else "]")
                )
            else:
                note = (
                    f"  [applies to {applying} of {len(cells)} cells - the ones above "
                    f"{knob} {profile['cfg_threshold']:g}]"
                )
            print(f"negative : {request.negative_prompt}{note}")
        if args.true_cfg is not None and not profile["true_cfg_supported"]:
            print(
                f"true cfg : {args.true_cfg:g}  [ignored: on this checkpoint --cfg is real "
                "CFG already]"
            )
        actual_sizes = list(p.submission_rules.cells_by_size(axis_cells))
        print("size     : " + ", ".join(f"{w}x{h}" for w, h in actual_sizes))
        # A Mu this checkpoint cannot take is dropped by `applied()` and the
        # row records none, so the report says so rather than show it as run.
        # Asked with a Mu of its own, since the base request may carry none.
        mu_ignored = backend.applied(request.model_copy(update={"mu": 1.0})).mu is None
        print(
            f"sampling : {request.steps} steps, CFG {request.guidance_scale}, "
            f"mu {'-' if request.mu is None else request.mu}"
            + (
                f", true CFG {args.true_cfg:g}"
                if args.true_cfg is not None and profile["true_cfg_supported"]
                else ""
            )
            + (
                "  [ignored: this checkpoint takes no Mu]"
                if request.mu is not None and mu_ignored
                else ""
            )
        )
        if references:
            print(
                f"reference: {', '.join(str(i) for i in references)}  "
                "(encoded onto the latent sequence at every step)"
            )
        if args.lora:
            print(f"lora     : {args.lora} @ {args.lora_scale:g}")
        if request.refine is not None:
            # The steps that run, since `sampling` above states the schedule's
            # length; the source's own size when the size was derived from it.
            ran = p.grid.denoised_steps(request)
            if request.outpaint is not None:
                print(
                    f"outpaint : {request.outpaint.sides()}, band {request.outpaint.band} px, fill "
                    f"{request.outpaint.fill}; canvas {request.width}x{request.height}, under a mask "
                    "built for it, composited over the padded source"
                )
            elif request.mask is not None:
                print(f"inpaint  : under mask {request.mask[:16]}, composited over the source")
            print(
                f"refine   : image {request.refine} at strength {request.strength:g}  "
                f"(about {ran} of {request.steps} steps"
                + (
                    f"; source {refined_from.width}x{refined_from.height})"
                    if refined_from is not None
                    else ")"
                )
            )
            if request.crop_anchor != "center":
                print(f"crop     : {request.crop_anchor}")
            if request.upscaler is not None:
                print(f"upscaler : {p.refine.ANIME6B_LABEL} (verified ~18 MB first use; 4x then target fit)")
        if seed_list:
            print(f"seeds    : {', '.join(str(s) for s in seed_list)}  (listed)")
        else:
            print(f"seed     : {seed}" + ("" if args.seed is not None else "  (random)"))
        print(f"run      : {run_id}")
        if gridding:
            values = p.grid.axis_values(
                request,
                args.grid_steps,
                args.grid_cfg,
                args.grid_mu,
                prompt_axis,
                args.grid_lora_scale,
                variants,
                strength=args.grid_strength,
                prompt_mode=args.grid_prompt_mode,
                cells=axis_cells,
                size=args.grid_size,
            )
            if len(values["prompt"]) > 1:
                # The substituted prompts, not the variables that made them:
                # what went wrong in a prompt grid is visible in the result.
                print(f"  prompts: {len(values['prompt'])}")
                for text in values["prompt"][:4]:
                    print(f"           {text}")
                if len(values["prompt"]) > 4:
                    print(f"           ... and {len(values['prompt']) - 4} more")
            axes = [("steps", "steps"), ("guidance_scale", "CFG"), ("mu", "mu")]
            if len(values["lora_scale"]) > 1:
                axes.append(("lora_scale", "LoRA"))
            if len(values["strength"]) > 1:
                axes.append(("strength", "strength"))
            for axis, shown in axes:
                listed = ", ".join("off" if v is None else f"{v:g}" for v in values[axis])
                if axis == "mu" and mu_ignored and any(v is not None for v in values[axis]):
                    listed += "  [ignored: this checkpoint takes no Mu]"
                print(f"  {shown:<7}: {listed}")
            if not seed_list:
                print(f"  seeds  : {args.grid_seeds} from {seed}")
            count = len(seed_list) or args.grid_seeds
            summary = p.grid.describe(
                values, count, listed=bool(seed_list),
                prompt_mode=args.grid_prompt_mode, batch=request.num_images,
            )
            print(
                f"grid     : {summary}"
                f" = {len(cells)} cell(s)"
            )
        if skipped:
            print(f"resume   : {skipped} cell(s) already rendered, skipping")
        # Cells that would render the very same image as an earlier one, as
        # the pipeline will run them - the same count the browser's preview
        # shows, and the reason to fix the grid before the night, not after.
        duplicates = p.grid.duplicate_count(cells, backend.applied)
        if duplicates:
            print(
                f"WARNING  : {duplicates} of {total_images} image(s) would be identical to "
                "an earlier one - an axis this checkpoint ignores (a negative-prompt "
                "variable, a CFG at or below the threshold, a Mu where it takes none), "
                "or strengths selecting the same denoising start"
            )
        print(
            f"output   : {total_images} image(s) - {len(cells)} run(s) x "
            f"{1 if args.grid_prompt_mode == 'random' else request.num_images} -> {p.settings.outputs_dir}"
        )
        # The same estimate the browser's preview shows, from recent images at
        # this size on this checkpoint - the one number that decides whether a
        # night's grid is a night's grid. Only from an existing gallery: the
        # lookup would otherwise write the schema into a new file, which a
        # --dry-run promises not to do.
        if p.settings.db_path.exists():
            priced = cells if spec is None else then_refine.execution_units(cells, spec)
            prices, need = p.gallery_reports.execution_estimates(priced, p.settings.model_id)
            seconds = p.grid.estimate_seconds(priced, None, by_size=prices)
            if seconds is not None:
                print(f"estimate : about {_duration(seconds)} of generation, plus the load")
            # And the disk, which is the other thing a night can run out of:
            # a full volume dies at the first save, and per the cell rule
            # takes every remaining cell with it.
            if need is not None:
                import shutil

                try:
                    free = shutil.disk_usage(p.settings.outputs_dir).free
                except OSError:
                    free = None
                line = f"disk     : about {p.gallery_reports.human_bytes(need)} of PNGs"
                if free is not None:
                    line += f"; {p.gallery_reports.human_bytes(free)} free"
                    if need > free:
                        line += "  [NOT ENOUGH]"
                print(line)
        print()

    # To stderr: `--quiet > paths.txt` promises stdout holds nothing but paths.
    if args.dry_run:
        print("Dry run - nothing generated.", file=sys.stderr)
        return 0

    if not cells:
        print("Nothing to do: every cell is already in the gallery.", file=sys.stderr)
        return 0

    # After the dry run: a dry run promises to touch nothing, the network
    # included. Its own exit code, since the README pins the other four.
    if not args.ignore_server:
        holder = _server_holds_gpu(p.settings.host, p.settings.port)
        if holder is not None:
            print(
                f"A server at {p.settings.host}:{p.settings.port} holds {holder} on the GPU; "
                "loading a second copy here would run out of memory. Unload it (the Unload "
                "button, or POST /api/model/unload), stop it, use --via-server to queue on it, or pass --ignore-server.",
                file=sys.stderr,
            )
            return 4

    p.storage.init_db()
    # An outpaint's mask, built and stored before the load (never under
    # --dry-run, which returned above): the cells then read it like any
    # stored mask. No `mask_holds` rows - this process runs them straight
    # after, and the store restarts the settle window that keeps a sweep off.
    if any(cell.request.outpaint is not None for cell in cells):
        try:
            cells = p.storage.stamp_outpaints(cells)
        except p.gallery_lock.GalleryBusy as exc:
            print(f"--outpaint: the outpaint's mask could not be stored: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"--outpaint: the outpaint's mask could not be written: {exc.strerror or exc}", file=sys.stderr)
            return 2
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
    try:
        for cell in cells:
            backend.check_refine_supported(cell.request)
        if spec is not None:
            backend.check_refine_supported(then_refine.stand_in(cells[0].request, spec))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        backend.unload()
        return 2
    if not args.quiet:
        info = backend.describe()
        device = info.get("device_name") or p.settings.device
        print(f"ready in {_duration(time.perf_counter() - load_started)} on {device}\n")

    run_index = 0
    saved = 0
    failed = 0
    consecutive = 0
    # Then refine's passes failed in a row, apart from the cells', as the
    # server's worker counts them.
    pass_streak = 0
    out_of_time = False
    # The pixel hashes of what landed: the summary says how many *different*
    # pictures a night made, which is the number an ignored axis shows up in.
    distinct: set[str] = set()
    deadline = load_started + args.budget * 60 if args.budget else None
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
            # "Start no cell after": the same rule the server's loop follows.
            if deadline is not None and time.perf_counter() > deadline:
                out_of_time = True
                print(
                    f"  time budget of {args.budget:g} min reached before run "
                    f"{index + 1}/{len(cells)}; --resume continues it",
                    file=sys.stderr,
                )
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
                # As the server's worker does: a mask lost since the checks
                # fails this cell in words, before the backend reads it.
                if cell.request.refine is not None and cell.request.mask is not None:
                    try:
                        p.storage.mask_path(cell.request.mask)
                    except ValueError as exc:
                        if cell.request.outpaint is None:
                            raise
                        raise ValueError(
                            "the outpaint's mask has gone from this gallery; run it again"
                        ) from exc
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
            # The call's peak, whole on every image of the batch, as the server saves it.
            peak = backend.last_peak()

            written = []
            for offset, image in enumerate(images):
                # Each image in a batch gets its own reproducible seed, wrapped
                # with the same helper the generator uses so the recorded seed
                # and the seed actually drawn cannot disagree.
                waiting = False
                while True:
                    try:
                        record = p.storage.save_image(
                            image,
                            ran,
                            seed=wrap_seed(run_seed + offset),
                            duration_ms=elapsed_ms // max(len(images), 1),
                            model_id=p.settings.model_id,
                            job_id=job_id,
                            run_id=run_id,
                            quantization=p.settings.quantization,
                            variables=cell.variables,
                            batch=cell.request.num_images,
                            peak=peak,
                        )
                        break
                    except p.gallery_lock.GalleryBusy:
                        # Another process has held the gallery past one
                        # SESSION_WAIT (a browser bundle export, a reindex).
                        # The render is in memory: wait again rather than
                        # lose it and every cell after it, as the server's
                        # job worker does. A second Ctrl-C still ends it.
                        if not waiting:
                            print("  gallery busy in another process; still waiting to save",
                                  file=sys.stderr, flush=True)
                            waiting = True
                saved += 1
                written.append(record)
                if record.content_hash:
                    distinct.add(record.content_hash)
                print(
                    f"{p.settings.outputs_dir / record.filename}  seed {record.seed}", flush=True
                )
            if spec is not None:
                made, failed_passes, pass_streak = _then_refine_passes(
                    p, backend, written, spec, pass_run, job_id, on_progress, should_cancel, pass_streak,
                    quiet=args.quiet, live=live,
                )
                saved += len(made)
                failed += failed_passes
                distinct.update(r.content_hash for r in made if r.content_hash)
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
            + (f" Only {len(distinct)} distinct." if saved > 1 and len(distinct) < saved else "")
            + (f" {failed} cell(s) failed and were skipped." if failed else "")
            + (" Stopped at the time budget." if out_of_time else "")
        )
    if failed:
        return 1
    # A budget stop is a cancel by the clock: the same code, since the same
    # thing is true - fewer images than asked, and a resume finishes them.
    return 130 if saved < total_images else 0


if __name__ == "__main__":
    raise SystemExit(main())
