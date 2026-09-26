"""Choose the checkpoint, quantization and offload before the settings freeze.

`settings` (backend/config.py) is a frozen dataclass whose fields are read from
the LOCAL_DIFFUSION_* variables at import time, so anything that wants a say in them - a
command-line flag, or the menu the server shows when it is started bare - has
to write os.environ *before* backend.config is imported. This module is that
step. It imports nothing from the package but model_defaults, and both entry
points (`backend.main`, `backend.generate`) run it before anything else.

The rule for the server: started with **no arguments** on a terminal, it asks.
Any argument at all - `--model`, `--quant`, `--offload`, or `--defaults` - means
the caller has decided, and whatever they did not name comes from the
environment exactly as before. A service file, a script, or `wsl/serve.sh`
(which forwards its arguments) is therefore never blocked on a prompt, and
neither is a run whose stdin is not a terminal.

The menu's defaults are the environment's values, so `wsl/env.sh`, which sets
LOCAL_DIFFUSION_QUANT=nunchaku and a lite-infer checkpoint, still gets what it asked for
on a bare Enter.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from backend.model_defaults import DEFAULT_MODEL_ID, DEFAULT_OFFLOAD, DEFAULT_QUANT, defaults_for

QUANTIZATIONS: tuple[tuple[str, str], ...] = (
    ("fp8", "torchao float8 weight-only; ~13 GB peak, fits a 16 GB card"),
    ("int8", "torchao int8 weight-only; smaller, slightly softer"),
    ("none", "full bfloat16; ~25 GB VRAM for Krea 2"),
    ("nunchaku", "Nunchaku Lite 4-bit, a pre-quantized lite-infer repo; Linux only"),
)
OFFLOADS: tuple[tuple[str, str], ...] = (
    ("model", "one component on the GPU at a time; right for 16 GB"),
    ("sequential", "per-submodule; lowest VRAM, much slower"),
    ("none", "everything resident; 24 GB+, and Nunchaku still spills at 16"),
)
FORMATS: tuple[tuple[str, str], ...] = (
    ("nvfp4", "RTX 50 (Blackwell)"),
    ("int4", "RTX 20 to 40"),
)


@dataclass(frozen=True)
class Checkpoint:
    name: str
    repo: str
    gate: str
    # The Nunchaku Lite re-upload, with `{fmt}` standing for nvfp4 or int4.
    # None where lite-infer has not published one.
    lite: str | None = None

    def lite_repo(self, fmt: str) -> str | None:
        return self.lite.format(fmt=fmt) if self.lite else None

    def summary(self) -> str:
        profile = defaults_for(self.repo)
        return f"{profile['steps']} steps, CFG {profile['guidance_scale']:g}"


_LITE = "lite-infer/{stem}-nunchaku-lite-{{fmt}}_r32-bnb4-text-encoder"

CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        "FLUX.1 Krea [dev]",
        "black-forest-labs/FLUX.1-Krea-dev",
        "click-through gate",
        _LITE.format(stem="flux.1-krea-dev"),
    ),
    Checkpoint(
        "Krea 2 Turbo", "krea/Krea-2-Turbo", "manual approval", _LITE.format(stem="krea-2-turbo")
    ),
    Checkpoint("Krea 2 Raw", "krea/Krea-2-Raw", "manual approval"),
    # Anime-specific SDXL finetune. The repo's Diffusers component tree was
    # replaced with the Opt UNet in its 2025-02-13 update, so the ordinary
    # from_pretrained path loads the optimized checkpoint without needing the
    # duplicate single-file checkpoint at the repo root.
    Checkpoint(
        "Animagine XL 4.0 Opt",
        "cagliostrolab/animagine-xl-4.0",
        "open",
    ),
    Checkpoint(
        "Kohaku XL Zeta",
        "KBlueLeaf/Kohaku-XL-Zeta",
        "open; anime, tags or natural-language prompts",
    ),
    Checkpoint(
        "Illustrious XL v2.0-STABLE",
        "OnomaAIResearch/Illustrious-XL-v2.0",
        "open; single file, converted locally on first load",
    ),
    Checkpoint(
        "WAI Illustrious v17",
        "LyliaEngine/waiIllustriousSDXL_v170",
        "open; community mirror, single file converted locally on first load",
    ),
    Checkpoint(
        "Z-Anime Base",
        "SeeSee21/Z-Anime",
        "open; experimental anime model, natural-language prompts",
    ),
    Checkpoint(
        "Anima Turbo v1.1",
        "circlestone-labs/Anima",
        "open; official split files",
    ),
    Checkpoint(
        "NetaYume Lumina v4",
        "duongve/NetaYume-Lumina-Image-2.0",
        "open; split denoiser, converted locally on first load",
    ),
    Checkpoint(
        "NewBie image Exp0.1",
        "NewBie-AI/NewBie-image-Exp0.1",
        "open; experimental anime model, XML or natural-language prompts",
    ),
    # The same pipeline as FLUX.1 Krea; dev is the base the LoRA ecosystem
    # targets, schnell its 4-step Apache-licensed sibling.
    Checkpoint(
        "FLUX.1 [dev]",
        "black-forest-labs/FLUX.1-dev",
        "click-through gate",
        _LITE.format(stem="flux.1-dev"),
    ),
    Checkpoint(
        "FLUX.1 [schnell]",
        "black-forest-labs/FLUX.1-schnell",
        "open",
        _LITE.format(stem="flux.1-schnell"),
    ),
    Checkpoint(
        "FLUX.2 [klein] 4B",
        "black-forest-labs/FLUX.2-klein-4B",
        "open",
        _LITE.format(stem="flux.2-klein-4b"),
    ),
    Checkpoint(
        "FLUX.2 [klein] 9B",
        "black-forest-labs/FLUX.2-klein-9B",
        "open",
        _LITE.format(stem="flux.2-klein-9b"),
    ),
    Checkpoint("FLUX.2 [klein] base 4B", "black-forest-labs/FLUX.2-klein-base-4B", "open"),
    Checkpoint("FLUX.2 [klein] base 9B", "black-forest-labs/FLUX.2-klein-base-9B", "open"),
    # 6B, 8 steps, guides from zero like Krea 2; the FLUX VAE, so the live
    # preview's matrix applies.
    Checkpoint(
        "Z-Image Turbo",
        "Tongyi-MAI/Z-Image-Turbo",
        "open",
        _LITE.format(stem="z-image-turbo"),
    ),
    # 20B, Krea 2's parent; 50 steps with true CFG. Only the 4-bit re-upload
    # fits this card, and only just: ~12 GiB peak and about 80 s per 1 MP image.
    Checkpoint(
        "Qwen-Image",
        "Qwen/Qwen-Image",
        "open",
        _LITE.format(stem="Qwen-Image"),
    ),
    # 8B single-stream DiT from Baidu, distilled like Z-Image Turbo: 8 steps
    # at guidance 1.0 under the classic `> 1` gate. Its repo carries a 7.7 GB
    # prompt rewriter ordinary loading leaves out; the UI's explicit Expand
    # action loads it on first use. FLUX.2's VAE, so no preview map yet.
    # lite-infer's name has no rank suffix, hence not `_LITE`.
    Checkpoint(
        "ERNIE-Image Turbo",
        "baidu/ERNIE-Image-Turbo",
        "open",
        "lite-infer/ERNIE-Image-Turbo-nunchaku-lite-{fmt}-bnb4-text-encoder",
    ),
)

# What config.py falls back to, from the one module both may import - importing
# config itself is the one thing this module must not do.
_DEFAULT_MODEL = DEFAULT_MODEL_ID
_DEFAULT_QUANT = DEFAULT_QUANT
_DEFAULT_OFFLOAD = DEFAULT_OFFLOAD


@dataclass(frozen=True)
class Choice:
    model_id: str
    quantization: str
    offload: str

    def environ(self) -> dict[str, str]:
        return {
            "LOCAL_DIFFUSION_MODEL_ID": self.model_id,
            "LOCAL_DIFFUSION_QUANT": self.quantization,
            "LOCAL_DIFFUSION_OFFLOAD": self.offload,
        }

    def command(self, prog: str = "uv run python -m backend.main") -> str:
        return (
            f"{prog} --model {self.model_id} --quant {self.quantization} "
            f"--offload {self.offload}"
        )


# --- command line ------------------------------------------------------------


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """The --model/--quant/--offload trio, shared by the server and the CLI."""
    group = parser.add_argument_group("model (equivalent to LOCAL_DIFFUSION_* variables)")
    group.add_argument("--model", metavar="ID", help="checkpoint id, e.g. krea/Krea-2-Turbo")
    group.add_argument(
        "--quant",
        choices=[key for key, _ in QUANTIZATIONS],
        help="weight quantization; nunchaku = a pre-quantized Nunchaku Lite repo, Linux only",
    )
    group.add_argument(
        "--offload", choices=[key for key, _ in OFFLOADS], help="offload strategy"
    )


def apply_env(args: argparse.Namespace) -> None:
    """Push the flags into the environment, where config.py will read them."""
    for value, name in (
        (args.model, "LOCAL_DIFFUSION_MODEL_ID"),
        (args.quant, "LOCAL_DIFFUSION_QUANT"),
        (args.offload, "LOCAL_DIFFUSION_OFFLOAD"),
    ):
        if value is not None:
            os.environ[name] = value


def configure(
    argv: list[str] | None = None,
    *,
    ask: Callable[[str], str] = input,
    out: TextIO | None = None,
    interactive: bool | None = None,
) -> Choice | None:
    """Settle the LOCAL_DIFFUSION_* model variables for a server start.

    Returns the menu's choice when the menu ran, else None. `argv` defaults to
    the process's own; `interactive` to whether stdin and stdout are terminals.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="python -m backend.main",
        description="Serve the generator. With no arguments on a terminal, asks which "
        "checkpoint, quantization and offload to use; any argument skips the menu "
        "and what it does not name comes from LOCAL_DIFFUSION_* variables.",
    )
    add_model_arguments(parser)
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="start with the environment's settings without asking",
    )
    args = parser.parse_args(argv)
    if argv:
        apply_env(args)
        return None
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        return None
    choice = prompt(ask=ask, out=out or sys.stdout)
    os.environ.update(choice.environ())
    return choice


# --- the menu ----------------------------------------------------------------


def _pick(
    title: str,
    options: list[tuple[str, str]],
    default: int,
    *,
    ask: Callable[[str], str],
    out: TextIO,
    free: str | None = None,
) -> str:
    """Number a list, take a number or an option's own key, Enter for the default.

    `free`, when given, names a trailing option that accepts typed text - the
    return value is then either an option key or whatever was typed. An answer
    containing "/" is taken as typed text straight away, so a repo id can be
    pasted at the checkpoint prompt without choosing "another" first.
    """
    out.write(f"\n{title}\n")
    width = max(len(key) for key, _ in options)
    for index, (key, note) in enumerate(options, 1):
        marker = "*" if index == default + 1 else " "
        out.write(f" {marker}{index:>2}  {key:<{width}}  {note}\n")
    if free:
        out.write(f"  {len(options) + 1:>2}  {free}\n")
    out.flush()
    count = len(options) + (1 if free else 0)
    while True:
        try:
            answer = ask(f"> [{default + 1}] ").strip()
        except EOFError:
            answer = ""
        if not answer:
            return options[default][0]
        if free and "/" in answer:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        if free and answer == str(count):
            while True:
                try:
                    typed = ask("repo id: ").strip()
                except EOFError:
                    # Stdin is gone - a dropped SSH session, a closed console.
                    # Take the default, as the outer prompt does; looping here
                    # spun at 100% CPU with no server ever started.
                    return options[default][0]
                if typed:
                    return typed
        for key, _ in options:
            if answer.lower() == key.lower():
                return key
        out.write(f"  pick 1-{count}\n")
        out.flush()


def _index(options: list[tuple[str, str]], key: str | None, fallback: str) -> int:
    keys = [k for k, _ in options]
    if key in keys:
        return keys.index(key)
    return keys.index(fallback)


def _match(model_id: str) -> tuple[Checkpoint | None, str | None]:
    """The catalogue entry an id names, and the 4-bit format if it is a lite repo."""
    for checkpoint in CHECKPOINTS:
        if model_id == checkpoint.repo:
            return checkpoint, None
        for fmt, _ in FORMATS:
            if model_id == checkpoint.lite_repo(fmt):
                return checkpoint, fmt
    return None, None


def prompt(
    *,
    ask: Callable[[str], str] = input,
    out: TextIO | None = None,
    env: dict[str, str] | None = None,
    linux: bool | None = None,
) -> Choice:
    """Run the menu and return what was chosen. Ctrl-C exits with status 130."""
    out = out or sys.stdout
    env = os.environ if env is None else env
    linux = sys.platform == "linux" if linux is None else linux
    try:
        return _prompt(ask, out, env, linux)
    except KeyboardInterrupt:
        out.write("\n")
        raise SystemExit(130) from None


def _prompt(ask: Callable[[str], str], out: TextIO, env: dict[str, str], linux: bool) -> Choice:
    current_model = env.get("LOCAL_DIFFUSION_MODEL_ID", _DEFAULT_MODEL)
    current_checkpoint, current_fmt = _match(current_model)

    out.write(
        "Local Diffusion. Enter keeps the marked default; any argument "
        "(--model, --quant, --offload, --defaults) skips this menu.\n"
    )

    # Quantization first: it decides which family of repos the checkpoint list
    # shows. Nunchaku Lite's kernels ship for Linux only, so off Linux the
    # option is not offered rather than offered and refused at load.
    quants = [(k, note) for k, note in QUANTIZATIONS if linux or k != "nunchaku"]
    quant = _pick(
        "Quantization",
        quants,
        _index(quants, env.get("LOCAL_DIFFUSION_QUANT"), _DEFAULT_QUANT),
        ask=ask,
        out=out,
    )

    fmt = None
    if quant == "nunchaku":
        formats = list(FORMATS)
        fmt = _pick(
            "4-bit format",
            formats,
            _index(formats, current_fmt, "nvfp4"),
            ask=ask,
            out=out,
        )

    # The checkpoint list: official repos, or the lite-infer re-uploads for
    # nunchaku, where only the families with a published build appear.
    entries: list[tuple[str, str]] = []
    repos: dict[str, str] = {}
    for checkpoint in CHECKPOINTS:
        repo = checkpoint.lite_repo(fmt) if fmt else checkpoint.repo
        if repo is None:
            continue
        gate = "not gated" if fmt else checkpoint.gate
        entries.append((checkpoint.name, f"{checkpoint.summary():<18}  {gate:<18}  {repo}"))
        repos[checkpoint.name] = repo
    default = 0
    if current_checkpoint and current_checkpoint.name in repos:
        default = [name for name, _ in entries].index(current_checkpoint.name)
    elif not fmt and current_model:
        # An id the catalogue does not know - one typed at the "another repo
        # id" prompt last time, a local directory, a community re-upload. It
        # gets a real row so that Enter keeps it, as the header promises;
        # without one three bare Enters silently started FLUX instead.
        name = "current LOCAL_DIFFUSION_MODEL_ID"
        profile = defaults_for(current_model)
        summary = f"{profile['steps']} steps, CFG {profile['guidance_scale']:g}"
        entries.append((name, f"{summary:<18}  {'unrecognised id':<18}  {current_model}"))
        repos[name] = current_model
        default = len(entries) - 1
    answer = _pick(
        "Checkpoint",
        entries,
        default,
        ask=ask,
        out=out,
        free="another repo id",
    )
    model_id = repos.get(answer, answer)

    offloads = list(OFFLOADS)
    offload = _pick(
        "Offload",
        offloads,
        _index(offloads, env.get("LOCAL_DIFFUSION_OFFLOAD"), _DEFAULT_OFFLOAD),
        ask=ask,
        out=out,
    )

    choice = Choice(model_id=model_id, quantization=quant, offload=offload)
    out.write(
        f"\nStarting {model_id} with {quant} quantization, offload {offload}.\n"
        f"The same start without the menu:\n  {choice.command()}\n\n"
    )
    out.flush()
    return choice
