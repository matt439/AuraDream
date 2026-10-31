"""Runtime settings, overridable by environment variables."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .legacy_names import default_db_path
from .execution import ExecutionSpec, loader_execution_from_choices
from .startup_preferences import LOADER_EXECUTION_ERROR, LOADER_EXECUTION_ERROR_VARIABLE
from .model_defaults import (
    DEFAULT_MODEL_ID,
    DEFAULT_OFFLOAD,
    DEFAULT_QUANT,
    OFFLOAD_MODES,
    QUANT_MODES,
)
from .seeds import MAX_BATCH
from .sizes import (
    largest_pixels,
    largest_side,
    largest_tier_pixels,
    refine_largest_pixels,
    refine_largest_side,
)

# Diffusers 0.40 snapshots os.getenv(name, "native") verbatim at import:
# absence uses native, but an explicitly empty value is not that default.
# Observe before its import without interpreting, clearing or accepting it.
ATTENTION_ENVIRONMENT_WARNING: str | None = (
    "DIFFUSERS_ATTN_BACKEND differs from Diffusers' native default. "
    "This external setting is not recorded as an AuraDream execution recipe."
    if os.getenv("DIFFUSERS_ATTN_BACKEND", "native") != "native" else None
)
if ATTENTION_ENVIRONMENT_WARNING is not None:
    logging.getLogger(__name__).warning("%s", ATTENTION_ENVIRONMENT_WARNING)

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str) -> Path | None:
    # Absolute, against the directory the app was started in: a relative
    # path would otherwise mean a different file to whatever resolves it
    # later from somewhere else (the critic starts its server in the
    # server's own directory). Not resolved: a link keeps the name it was
    # given, which the critic labels its answers with - a Hugging Face
    # snapshot's file links to a blob named by its hash.
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().absolute() if raw else None


def _loader_execution_environment() -> tuple[ExecutionSpec | None, str | None]:
    """Snapshot app-owned recipe names without framework/device/cache work."""
    compilation = os.getenv("AURADREAM_COMPILATION", "eager")
    attention = os.getenv("AURADREAM_ATTENTION", "native")
    try:
        spec = loader_execution_from_choices(compilation, attention)
    except (TypeError, ValueError):
        return None, LOADER_EXECUTION_ERROR
    return spec, LOADER_EXECUTION_ERROR if LOADER_EXECUTION_ERROR_VARIABLE in os.environ else None


_STARTUP_LOADER_EXECUTION, _STARTUP_LOADER_EXECUTION_ERROR = _loader_execution_environment()


@dataclass(frozen=True)
class Settings:
    # Which checkpoint to serve: default `model_defaults.DEFAULT_MODEL_ID`; the
    # catalogue is `launch.CHECKPOINTS`.
    model_id: str = os.getenv("AURADREAM_MODEL_ID", DEFAULT_MODEL_ID)

    # "fp8"  - torchao float8 weight-only on the transformer (~12GB, fits 16GB cards)
    # "int8" - torchao int8 weight-only, the same size as fp8; on the sixteen
    #          checkpoints compared on the card (not FLUX.1, Krea 2, klein 9B
    #          or Qwen-Image), 3-10% faster a step and nearer bf16 on 14
    #          (docs/verified.md)
    # "none" - full bfloat16, needs ~25GB VRAM
    # "nunchaku" - a checkpoint that is already quantized (Nunchaku Lite, SVDQuant
    #          4-bit, e.g. lite-infer/krea-2-turbo-nunchaku-lite-nvfp4_r32-bnb4-
    #          text-encoder). Nothing is quantized here: the repo's own configs
    #          name the quantizer. Its CUDA kernels ship for Linux only, so on a
    #          Windows machine this means WSL 2 - see wsl/ and the README.
    quantization: str = os.getenv("AURADREAM_QUANT", DEFAULT_QUANT)

    # "model"      - one component on the GPU at a time (recommended for <=16GB)
    # "sequential" - per-submodule offload, lowest VRAM, much slower
    # "none"       - keep everything resident
    offload: str = os.getenv("AURADREAM_OFFLOAD", DEFAULT_OFFLOAD)

    device: str = os.getenv("AURADREAM_DEVICE", "cuda")
    load_on_startup: bool = _env_bool("AURADREAM_PRELOAD", False)
    loader_execution: ExecutionSpec | None = _STARTUP_LOADER_EXECUTION
    loader_execution_error: str | None = _STARTUP_LOADER_EXECUTION_ERROR

    # Where the images and their index live. Overridable so a server run from
    # WSL 2 against a checkout on the Windows drive can keep them on the Linux
    # filesystem: /mnt/c is a 9P bridge, slow for writes and unsafe for SQLite's
    # locks. The web directory stays code - it is read once, and small.
    outputs_dir: Path = Path(os.getenv("AURADREAM_OUTPUTS_DIR", ROOT / "outputs")).expanduser()
    # auradream.db, or the gallery an install made as local-diffusion.db while
    # that is the only one here (`legacy_names`: it is used, never renamed).
    db_path: Path = Path(os.getenv("AURADREAM_DB_PATH", default_db_path(ROOT))).expanduser()
    # Saved archives may live on another drive without moving the gallery.
    # None keeps the legacy bundles/ beside whichever database is in use.
    bundles_dir: Path | None = _env_path("AURADREAM_BUNDLES_DIR")
    web_dir: Path = ROOT / "web"
    # Where LoRA adapters are read from - one directory of *.safetensors, listed
    # by `GET /api/loras` and chosen by *name*. A path taken from the request
    # and joined onto a directory is an arbitrary-file read reachable from
    # /api/generate, so the request never carries one.
    loras_dir: Path = Path(os.getenv("AURADREAM_LORAS_DIR", ROOT / "loras")).expanduser()

    host: str = os.getenv("AURADREAM_HOST", "127.0.0.1")
    port: int = int(os.getenv("AURADREAM_PORT", "7860"))
    # The server's log level. DEBUG is where the tracebacks and the dropped
    # pipeline arguments go; INFO is enough to follow a night.
    log_level: str = os.getenv("AURADREAM_LOG_LEVEL", "INFO").upper()

    # Host headers the API answers to, and with the port appended, the origins
    # allowed to write to it. Binding to loopback stops another machine's TCP,
    # not a browser on this one aimed at 127.0.0.1 by a hostile DNS record -
    # once rebound the attacker's page is same-origin and CORS stops applying.
    # Deliberately not derived from `host`: with AURADREAM_HOST=0.0.0.0 the browser
    # sends the machine's LAN address or name, never 0.0.0.0, so that case needs
    # naming here (AURADREAM_ALLOWED_HOSTS=my-box,192.168.1.20). There
    # is no login: whoever reaches a non-loopback bind controls the server,
    # which `main.run` warns about at start-up.
    allowed_hosts: tuple[str, ...] = tuple(
        part.strip()
        for part in os.getenv("AURADREAM_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
        if part.strip()
    )

    # Guard rails so a typo in the UI cannot ask for a 16k image. Both are the
    # largest preset the UI offers, read off the table in `sizes.py` rather
    # than by what the GPU can take: 3168 on the long side and
    # 3168x1328 = 4,207,104 pixels (12:5 at 4 MP). They used to be rounder
    # numbers derived from the table in prose and then typed in by hand, which
    # is how a table change would have left them describing nothing.
    max_side: int = largest_side()
    max_pixels: int = largest_pixels()
    # A refine pass goes further. The presets stop at 4 MP because a direct
    # render past a checkpoint's range composes badly, and a pass takes its
    # composition from the source instead - which is the reason to run one.
    # Its own guard is any 2 MP preset at 2x: 4448 on the long side and
    # 1776x1184 x 4 = 8,411,136 pixels. Still a typo guard, not a VRAM model:
    # measured 2026-10-04 (verified.md), WAI at 8.6 MP peaked at 5.5 GiB while
    # Z-Anime at 4.2 MP already filled the card, so what fits is the
    # checkpoint's and the page says so per size.
    refine_max_side: int = refine_largest_side()
    refine_max_pixels: int = refine_largest_pixels()
    # The pixels one *batch* may hold at once - `num_images` forward passes in a
    # single call, all resident together. A typo guard, not a VRAM model: cover
    # both two 4 MP images and a full batch (`MAX_BATCH`) at every 1 MP preset
    # aspect. The old two-4-MP-only default was 8,411,136 pixels and rejected
    # Batch 8 at the slightly-over-1-MP 3:4 and 4:3 presets despite measured
    # VRAM headroom; the cap went from 8 to 12 once eight 1 MP images were seen
    # to hold 9.6 GB of the 16. Runs remain the way to make many large images.
    max_batch_pixels: int = int(
        os.getenv(
            "AURADREAM_MAX_BATCH_PIXELS",
            str(max(2 * largest_pixels(), MAX_BATCH * largest_tier_pixels(1))),
        )
    )

    # The most images one grid may queue. A grid is meant to be left running
    # overnight, so this is generous - 1024 at ~30s is nine hours - but a
    # four-axis product grows fast enough that a typo needs a ceiling.
    max_grid_images: int = int(os.getenv("AURADREAM_MAX_GRID_IMAGES", "1024"))

    # The local prompt critic (`critic.py`): a pinned llama.cpp `llama-server`,
    # started per critique over a pinned Qwen3-4B GGUF, both fetched and
    # checked on first use. Off hides the Critique action everywhere.
    critic: bool = _env_bool("AURADREAM_CRITIC", True)
    # Files already on disk in place of either download: a llama-server of
    # your own build (a platform with no pinned one, a newer release), a .gguf
    # you converted. Neither is held to the pins - naming one is choosing it.
    critic_server: Path | None = _env_path("AURADREAM_CRITIC_SERVER")
    critic_model: Path | None = _env_path("AURADREAM_CRITIC_MODEL")


settings = Settings()


def _check_vocabulary(name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Refuse a AURADREAM_* value outside its vocabulary, at import, by name.

    The menu and the argparse `choices` are safe; the environment variables are
    the documented interface and three start paths skip the menu. An unknown
    value used to pick the heaviest branch silently and fail at the first image.
    """
    if value not in allowed:
        raise SystemExit(f"{name}={value!r} is not one of: {', '.join(allowed)}")


_check_vocabulary("AURADREAM_QUANT", settings.quantization, QUANT_MODES)
_check_vocabulary("AURADREAM_OFFLOAD", settings.offload, OFFLOAD_MODES)

# The outputs directory is created by `storage.ensure_outputs_dir()` - from
# `init_db()`, the server's mount and the first write - not here: an import-time
# mkdir defeated `--dry-run`'s "nothing changed", and on an unmounted
# AURADREAM_OUTPUTS_DIR created the mount point as a plain directory.

# Nunchaku Lite's kernels come from a Hub repo that diffusers does not list as a
# trusted publisher, so it refuses to download and execute them unless told to.
# Choosing AURADREAM_QUANT=nunchaku is that choice. diffusers reads the flag once, at
# import, and this module is imported before diffusers everywhere in the app -
# the server, the CLI, the tests - which is why it is set here rather than in
# the backend. setdefault, so an explicit "false" in the environment still wins.
if settings.quantization == "nunchaku":
    os.environ.setdefault("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")
