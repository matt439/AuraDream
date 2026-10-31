"""The command line's arguments: the parser, a --from-image PNG's settings
filled into them, and the checks that need only the flags.

Split from `generate.py`, which imports these names back, so its `main()`
(and the tests that patch `generate._image_settings` or `_mask_file`) still
reach them as module globals there. Nothing here imports `config`: all of it
runs before `apply_env` has had its say about AURADREAM_* variables.
"""
from __future__ import annotations

import argparse
import math

# launch.py owns --model/--quant/--offload for the server and this CLI alike;
# sizes.py owns the preset table the browser's copy is asserted against.
from backend.launch import add_model_arguments
from backend.png_settings import image_settings
from backend.sizes import SIZES, TIERS


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
    parser.add_argument("prompt", nargs="?", help="what to draw (or take it from --from-image)")
    parser.add_argument(
        "-n",
        "--negative",
        help="negative prompt; ignored at or below the checkpoint's CFG threshold "
        "(1 on FLUX, 0 on Krea 2)",
    )
    parser.add_argument(
        "--from-image",
        metavar="PNG",
        help="start from the settings a gallery PNG carries in its metadata - prompt, "
        "negative prompt, size, steps, CFG, true CFG, mu, seed, checkpoint, references, "
        "adapter, refine source and strength, and grid variables; any flag given "
        "explicitly still wins, and the variables carry only while the prompt pair is "
        "the image's own",
    )

    size = parser.add_argument_group("size")
    size.add_argument("--aspect", choices=list(SIZES), help="aspect preset (default 1:1)")
    size.add_argument(
        "--mp", type=float, choices=TIERS, help="megapixel tier: 1, 1.5, 2 or 4 (default 1)"
    )
    size.add_argument("--width", type=int, help="explicit width, multiple of 16")
    size.add_argument("--height", type=int, help="explicit height, multiple of 16")

    sampling = parser.add_argument_group("sampling (defaults come from the checkpoint profile)")
    sampling.add_argument("--steps", type=int, help="denoising steps")
    sampling.add_argument("--cfg", type=float, metavar="SCALE", help="guidance scale")
    sampling.add_argument(
        "--true-cfg",
        type=float,
        metavar="SCALE",
        help="real classifier-free guidance on FLUX.1, where --cfg is the distilled "
        "guidance value; above 1 the negative prompt runs an uncond pass (~2x slower). "
        "Ignored where --cfg is real CFG already (Krea 2, klein)",
    )
    # Exclusive: `--mu 1.15 --no-mu` used to parse and silently run with no shift.
    shift = sampling.add_mutually_exclusive_group()
    shift.add_argument(
        "--mu",
        type=float,
        help="flow-matching shift, on checkpoints whose scheduler takes one",
    )
    shift.add_argument("--no-mu", action="store_true", help="let the pipeline pick its own shift")
    sampling.add_argument("--seed", type=int, help="base seed; random when omitted")
    sampling.add_argument(
        "--step-cache", choices=("off", "fbc-005-v1"),
        help="per-render cache recipe (off by default; nonempty recipes remain unavailable)",
    )

    conditioning = parser.add_argument_group("conditioning")
    conditioning.add_argument(
        "--reference",
        metavar="IDS",
        default="",
        help="gallery image ids to condition on - 12 or 12,15 - on a checkpoint whose "
        "pipeline takes them (FLUX.2 klein). Each is resized to 1 MP and encoded onto "
        "the latent sequence, so one reference costs roughly 2-3x the time",
    )
    conditioning.add_argument(
        "--lora",
        metavar="FILE",
        help="a .safetensors adapter in the loras directory, by file name "
        "(AURADREAM_LORAS_DIR moves it)",
    )
    conditioning.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        metavar="W",
        help="the weight that adapter is applied at (default 1.0)",
    )
    conditioning.add_argument(
        "--refine",
        type=int,
        metavar="ID",
        help="refine pass: scale this gallery image up and re-denoise it part of the way "
        "under the prompt and settings given, on a checkpoint diffusers has an "
        "image-to-image pipeline for (FLUX.1, SDXL, Z-Image, Qwen-Image). With no size "
        "given the size is the source's times --scale; a size of another shape crops "
        "the source rather than stretching it",
    )
    conditioning.add_argument(
        "--strength",
        type=float,
        metavar="S",
        help="how much of the source a refine pass redraws, above 0 up to 1: low keeps "
        "the picture, 1 keeps nothing. Defaults to the checkpoint's own starting "
        "strength (0.4 on SDXL, 0.1 on FLUX.1 dev, 0.5 on FLUX.1 schnell); an inpaint "
        "(--mask) starts where the Inpaint dialog does instead: the checkpoint's own measured "
        "start where it has one (0.95 on FLUX.1), else its inpaint class's default (0.9999 on "
        "SDXL, 1 on Z-Image). About steps x S of the steps run",
    )
    conditioning.add_argument(
        "--crop-anchor",
        choices=("center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"),
        help="with ordinary --refine: which part of a differently shaped source the crop keeps (default center)",
    )
    conditioning.add_argument(
        "--then-refine-scale", type=float, metavar="F",
        help="then refine: after each cell, a refine pass of every image it made at F times "
        "its size (1 to 2), as Mass refine makes one - the base's seed and prompt, Lanczos - "
        "saved under a run of its own. Not with --refine, --mask or --outpaint",
    )
    conditioning.add_argument(
        "--then-refine-strength", type=float, metavar="S",
        help="with --then-refine-scale: the passes' strength (default the checkpoint's own)",
    )
    conditioning.add_argument(
        "--then-refine-steps", type=int, metavar="N",
        help="with --then-refine-scale: the passes' steps (default each cell's own)",
    )
    conditioning.add_argument(
        "--upscaler", choices=("lanczos", "realesrgan-x4plus-anime6b"),
        help="with ordinary --refine on WAI v17 or RouWei: opt into R-ESRGAN 4x+ Anime6B "
        "(one verified ~18 MB download), then resize to the target; default lanczos. "
        "GAN sources are limited to 4.19 MP / 4096px per side; card behavior is unmeasured",
    )
    conditioning.add_argument(
        "--mask",
        metavar="FILE",
        help="with --refine: inpaint - repaint only where this picture is white and keep the "
        "source where it is black (grey blends), on a checkpoint whose inpaint is enabled here "
        "(the FLUX.1, SDXL and Z-Image ones). A PNG at the source's own size, which is the size the "
        "inpaint runs at; kept in the gallery's masks by its hash, which the image records",
    )
    conditioning.add_argument(
        "--outpaint",
        metavar="L,T,R,B|PRESET",
        help="with --refine: outpaint - extend the source's canvas by these four sides (each a "
        "multiple of 16, in pixels: left, top, right, bottom) and paint the new area, as an inpaint "
        "over the source on the larger canvas, its new area filled from the source's edge, under a "
        "mask the run builds itself (the new border and a band into the original). Or a preset: "
        "+25%%, or an aspect of --aspect's (16:9, 2:3, ...) to extend to. The size is the source's "
        "plus the padding; on a checkpoint whose inpaint is enabled here (the FLUX.1, SDXL and Z-Image ones)",
    )
    conditioning.add_argument(
        "--band",
        type=int,
        metavar="PX",
        help="with --outpaint: how far into the original the mask's feathered band reaches, a "
        "multiple of 8 from 0 to 256 (default 32, which joined best of 0/32/64/128 by eye on the card on WAI v17, "
        "Animagine XL 4.0 and Z-Anime)",
    )
    conditioning.add_argument(
        "--scale",
        type=float,
        metavar="F",
        help="with --refine: render at the source's own size times this, each side "
        "rounded to 16 (default 1.5), instead of at a size given by --width/--height "
        "or --aspect/--mp. A pass may go past the 4 MP presets, to any 2 MP preset "
        "at 2x (4448 on the long side, 8.4 MP)",
    )

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
    sweep.add_argument(
        "--grid-prompt-mode", choices=["cross", "random"], default="cross",
        help="cross all prompt values (default), or draw one choice per image seed; "
        "random images run separately even with --batch",
    )
    sweep.add_argument("--grid-steps", metavar="SPEC", default="", help="step counts to sweep")
    sweep.add_argument("--grid-size", metavar="WxH,...", default="", help="paired canvas sizes, e.g. 1024x1024,768x1024")
    sweep.add_argument("--grid-cfg", metavar="SPEC", default="", help="guidance scales to sweep")
    sweep.add_argument(
        "--grid-mu",
        metavar="SPEC",
        default="",
        help="shifts to sweep; `none` for the pipeline's own shift",
    )
    # A count from the base, or exactly these: the two cannot both be meant.
    which_seeds = sweep.add_mutually_exclusive_group()
    which_seeds.add_argument(
        "--grid-seeds",
        type=int,
        default=1,
        metavar="N",
        help="consecutive seeds counting up from --seed, each running the whole grid (default 1)",
    )
    which_seeds.add_argument(
        "--seed-list",
        metavar="SPEC",
        default="",
        help="run the whole grid at exactly these seeds - 4471,8823 or 100-103 - "
        "instead of counting up from --seed",
    )
    sweep.add_argument(
        "--grid-lora-scale",
        metavar="SPEC",
        default="",
        help="LoRA weights to sweep - 0-1x0.25 - needing --lora; a per-call setting, "
        "so a cell costs a re-weighting rather than a reload",
    )
    sweep.add_argument(
        "--grid-strength",
        metavar="SPEC",
        default="",
        help="refine strengths to sweep - 0.2-0.5x0.1 - needing --refine: the sheet that "
        "shows where the picture stops being the source's",
    )
    sweep.add_argument(
        "--resume",
        action="store_true",
        help="skip cells this prompt has already been rendered at, so a night "
        "that died costs the cell it died on rather than the whole run",
    )
    sweep.add_argument(
        "--budget",
        type=float,
        metavar="MINUTES",
        help="start no cell after this many minutes (the load included); over-provision "
        "the grid and take what the night reaches - the rest is a --resume",
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
    parser.add_argument(
        "--ignore-server",
        action="store_true",
        help="load the weights even if a server on AURADREAM_HOST:AURADREAM_PORT already holds them on the GPU",
    )
    parser.add_argument(
        "--via-server", action="store_true",
        help="queue on the configured running server and wait; print its PNG URLs and seeds, without loading local weights",
    )
    return parser


def _image_settings(path: str, parser: argparse.ArgumentParser) -> dict:
    """The AuraDream record a gallery PNG carries, as a dict.

    `png_settings.image_settings`, shared with the page's Load settings from
    PNG; its refusals - a JPEG, another tool's PNG, a record that is not
    JSON - are argument errors here, not tracebacks.
    """
    try:
        from backend.execution import require_default_execution

        meta = image_settings(path)
        # A file's recipe is part of its reproduction set. Until this build
        # can run it, refuse before apply_env, source reads or local loading.
        require_default_execution(meta.get("execution"))
        return meta
    except ValueError as exc:
        parser.error(f"--from-image: {exc}")


def _apply_image_settings(args: argparse.Namespace, meta: dict) -> None:
    """Fill the arguments the flags left unset from the PNG's own record.

    Before `apply_env`, so the checkpoint the image was made on can reach
    AURADREAM_MODEL_ID: a Turbo PNG rendered on FLUX would claim to reproduce an
    image it cannot. Quantization is *not* carried - a Nunchaku PNG read on
    Windows would otherwise ask for a mode this machine cannot run - and
    neither are the job and run ids: they are displayed, never reused.
    """
    if meta.get("edit") is not None:
        raise ValueError("Instruction-edit settings must be reopened from the gallery image; CLI editing is unavailable.")
    if args.prompt is None:
        args.prompt = meta.get("prompt")
    if args.negative is None:
        args.negative = meta.get("negative_prompt")
    if args.steps is None:
        args.steps = meta.get("steps")
    if args.cfg is None:
        args.cfg = meta.get("guidance_scale")
    if args.true_cfg is None and meta.get("true_cfg") is not None:
        args.true_cfg = meta["true_cfg"]
    # An explicit null in the record means the image ran with no shift of
    # ours; --mu/--no-mu on the command line still win.
    if args.mu is None and not args.no_mu and "mu" in meta:
        if meta["mu"] is None:
            args.no_mu = True
        else:
            args.mu = meta["mu"]
    if args.seed is None:
        args.seed = meta.get("seed")
    # The rest of the reproduction set. Unlike quantization these carry: the
    # reference ids are this gallery's own, and an adapter that is missing is
    # refused by name below rather than silently producing another picture.
    if not args.reference and meta.get("reference"):
        args.reference = ",".join(str(i) for i in meta["reference"])
    if args.lora is None and meta.get("lora"):
        args.lora = meta["lora"]
        if meta.get("lora_scale") is not None:
            args.lora_scale = meta["lora_scale"]
    # A refined image reproduces as a refine pass of the same source: the id
    # is this gallery's own, like a reference, and the strength is the
    # image's unless the command line names another.
    if args.refine is None and meta.get("refine") is not None:
        args.refine = meta["refine"]
        # What the PNG says that source was, held to this gallery's row
        # under the id once the gallery can be read (`_check_carried_source`).
        args.carried_source = meta.get("refine_source") if isinstance(meta.get("refine_source"), dict) else None
        if args.strength is None and meta.get("strength") is not None:
            args.strength = meta["strength"]
            args.carried_strength = True
        if args.upscaler is None and meta.get("upscaler") is not None:
            args.upscaler = meta["upscaler"]
            args.carried_upscaler = True
        if args.crop_anchor is None and meta.get("crop_anchor") is not None:
            args.crop_anchor = meta["crop_anchor"]
            args.carried_crop_anchor = True
        # An inpaint reproduces over the same mask: by the hash its record
        # names, which `_mask_from_record` finds in the gallery or in the
        # PNG's own copy. Only with the source it covers - a --refine of
        # another image takes no mask from this one.
        # An outpaint reproduces over the same padding, band and fill, its
        # mask built again - never carried as a mask hash, which the request
        # would then refuse beside the outpaint. The record's mask is kept
        # aside only to hold the rebuilt one to.
        carried_outpaint = _carried_outpaint(meta)
        if args.mask is None and args.outpaint is None and carried_outpaint is not None:
            left, top, right, bottom, band, fill = carried_outpaint
            args.outpaint = f"{left},{top},{right},{bottom}"
            args.outpaint_fill = fill
            args.carried_outpaint = True
            # The record's mask stands for this rebuild only when the band
            # is the record's too: a --band of its own builds another mask,
            # which is what was asked for, not a sign of another build.
            if args.band is None:
                args.band = band
                args.carried_outpaint_mask = meta.get("mask") if isinstance(meta.get("mask"), str) else None
        elif args.mask is None and args.outpaint is None and isinstance(meta.get("mask"), str):
            args.mask_hash = meta["mask"]
    if args.model is None and meta.get("model"):
        args.model = meta["model"]
    # The file's size only when none of the four size arguments was given.
    size = meta.get("size")
    if size and all(v is None for v in (args.width, args.height, args.aspect, args.mp)):
        width, _, height = size.partition("x")
        if width.isdigit() and height.isdigit():
            args.width, args.height = int(width), int(height)


def _carried_outpaint(meta: dict) -> tuple[int, int, int, int, int, str] | None:
    """A PNG record's outpaint as (left, top, right, bottom, band, fill),
    where it carries a well-formed one beside its mask, else None. The fill
    is kept whatever it is: an unknown one is refused by name below."""
    value, fill = meta.get("outpaint"), meta.get("outpaint_fill")
    if not isinstance(value, dict) or not isinstance(fill, str) or meta.get("mask") is None:
        return None
    left, top, right, bottom, band = (value.get(name) for name in ("left", "top", "right", "bottom", "band"))
    if (
        type(left) is not int or left < 0
        or type(top) is not int or top < 0
        or type(right) is not int or right < 0
        or type(bottom) is not int or bottom < 0
        or type(band) is not int or band < 0
    ):
        return None
    return left, top, right, bottom, band, fill


def _outpaint_sides(text: str, parser: argparse.ArgumentParser) -> tuple[int, int, int, int] | str:
    """--outpaint's four sides, or the preset it names (resolved once the
    source's size is read). A side off the grid is refused with its
    neighbours, never rounded silently."""
    from backend.sizes import SIZES

    if text in SIZES or text == "+25%":
        return text
    parts = text.split(",")
    if len(parts) != 4 or not all(part.strip().isdigit() for part in parts):
        parser.error(
            f"--outpaint: {text!r} is neither four sides L,T,R,B in pixels nor a preset "
            f"(+25%, {', '.join(SIZES)})"
        )
    sides = tuple(int(part) for part in parts)
    for side in sides:
        if side % 16:
            below = side - side % 16
            parser.error(f"--outpaint: {side} is not a multiple of 16: {below} or {below + 16}")
    if not any(sides):
        parser.error("--outpaint: an outpaint extends at least one side")
    return sides[0], sides[1], sides[2], sides[3]


def _mask_file(path: str, parser: argparse.ArgumentParser):
    """The picture --mask names, in the canonical form a mask is kept in:
    one `L` channel, 0 to 255, white repainted. A path is fine here - the
    CLI is the user's own process - and the hash is what is recorded."""
    from PIL import Image, UnidentifiedImageError

    # `refine` is a leaf: importing it does not freeze the settings, which
    # `apply_env` has not written yet.
    from backend import refine

    try:
        with Image.open(path) as image:
            image.load()
            # 16-bit levels scaled, transparency laid on black: what was
            # painted, where Pillow's `convert("L")` clips the one and drops
            # the other.
            mask = refine.canonical_mask(image)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        parser.error(f"--mask: cannot read {path}: {exc}")
    except ValueError as exc:
        parser.error(f"--mask: {path}: {exc}")
    if mask.getextrema()[1] == 0:
        parser.error("--mask: nothing is masked - the picture is black everywhere, so nothing would change")
    return mask


def _image_variables(args: argparse.Namespace, meta: dict) -> dict[str, str] | None:
    """The PNG's variables, while they still describe what this run renders.

    What the placeholders were set to, or which version of a comparison the
    image is: the only place a version's name lives, and what the sheet lays
    out, so they carry like the rest of the reproduction set - as
    `/api/rerender` carries them. But they name the prompt pair they were
    made with, and a variable may have filled either half: a command line
    that changes either one ends them, since a label on another text names a
    version it is not. Compared as text, after `_apply_image_settings`, so
    retyping the image's own prompt keeps them. `--grid-prompt` is a prompt
    axis of its own, whose cells carry what it assigns - when it holds one: a
    blank block is no axis, as `parse_variables` reads it.
    """
    variables = meta.get("variables")
    axis = any(line.strip() for line in args.grid_prompt)
    if not isinstance(variables, dict) or not variables or axis:
        return None
    ran = (meta.get("prompt"), meta.get("negative_prompt") or None)
    return variables if (args.prompt, args.negative or None) == ran else None


def _resolve_size(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[int, int]:
    if args.width is not None or args.height is not None:
        if args.aspect is not None or args.mp is not None:
            parser.error("--width/--height and --aspect/--mp are alternatives; pick one")
        if args.width is None or args.height is None:
            parser.error("--width and --height must be given together")
        return args.width, args.height
    return SIZES[args.aspect or "1:1"][args.mp or 1]


def _then_refine_flags(args, parser) -> dict | None:
    """--then-refine-*, as the request's `then_refine`, or None."""
    if args.then_refine_scale is None:
        if args.then_refine_strength is not None or args.then_refine_steps is not None:
            parser.error("--then-refine-strength and --then-refine-steps need --then-refine-scale")
        return None
    for flag, value in (("--refine", args.refine), ("--mask", args.mask), ("--outpaint", args.outpaint)):
        if value is not None:
            parser.error(f"--then-refine-scale and {flag} are alternatives: a pass follows a direct render")
    if not 1.0 <= args.then_refine_scale <= 2.0:
        parser.error("--then-refine-scale is from 1 to 2")
    strength = args.then_refine_strength
    if strength is not None and not (math.isfinite(strength) and 0.0 < strength <= 1.0):
        parser.error("--then-refine-strength is a fraction above 0, up to 1")
    if args.then_refine_steps is not None and not 1 <= args.then_refine_steps <= 100:
        parser.error("--then-refine-steps is from 1 to 100")
    return {"scale": args.then_refine_scale, "strength": strength, "steps": args.then_refine_steps}


def _grid_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, bool]:
    """The grid's prompt axis, and whether the flags ask for a grid at all,
    once the --grid-* combinations that cannot run are refused. The local
    run and --via-server both call it, so the two refuse the same lines."""
    # One flag per variable on the command line, one line per variable in the
    # browser: the same block reaches the same parser either way.
    prompt_axis = "\n".join(args.grid_prompt)
    if args.grid_seeds < 1:
        parser.error("--grid-seeds must be at least 1")
    gridding = (
        bool(
            prompt_axis
            or args.grid_size
            or args.grid_steps
            or args.grid_cfg
            or args.grid_mu
            or args.grid_lora_scale
            or args.grid_strength
            or args.seed_list
            or args.grid_prompt_mode != "cross"
        )
        or args.grid_seeds > 1
    )
    if args.grid_prompt_mode == "random" and not prompt_axis.strip():
        parser.error("--grid-prompt-mode random needs --grid-prompt variables")
    if args.grid_lora_scale and not args.lora:
        # Without an adapter the scale is not a setting at all, so the sweep
        # would be one image rendered N times under N different labels.
        parser.error("--grid-lora-scale needs --lora: with no adapter the weight does nothing")
    if args.grid_strength and args.refine is None:
        # The same for a strength with nothing to re-denoise.
        parser.error("--grid-strength needs --refine: with no source the strength does nothing")
    if gridding and args.runs != 1:
        parser.error("--runs and --grid-* are alternatives; a grid steps its own seeds")
    if args.resume and args.seed is None and not args.seed_list:
        # A fresh random base matches none of the night it is meant to continue.
        parser.error("--resume needs --seed: the seed the run it continues was started from")
    return prompt_axis, gridding


def _cli_execution_intent(args) -> bool:
    """Refuse before local image/project reads; retain explicit Eager intent."""
    import os

    from backend.execution import (
        ExecutionSpec, INVALID_LOADER_CONFIGURATION, loader_execution_from_choices,
        require_default_execution,
    )

    compilation = getattr(args, "compilation", None)
    attention = getattr(args, "attention", None)
    # This internal diagnostic is invalid even when its value is empty.
    # Generation flags do not repair a saved startup profile or its marker.
    if "AURADREAM_LOADER_EXECUTION_ERROR" in os.environ:
        raise ValueError(INVALID_LOADER_CONFIGURATION)
    spec = loader_execution_from_choices(
        compilation if compilation is not None else os.getenv("AURADREAM_COMPILATION"),
        attention if attention is not None else os.getenv("AURADREAM_ATTENTION"),
    )
    cache = getattr(args, "step_cache", None)
    if cache not in (None, "off"):
        values = {} if spec is None else spec.model_dump(exclude_none=True)
        spec = ExecutionSpec(**{**values, "step_cache": cache})
    require_default_execution(spec)
    args.execution_recipe = spec
    return (
        compilation is not None or attention is not None
        or "AURADREAM_COMPILATION" in os.environ or "AURADREAM_ATTENTION" in os.environ
    )
