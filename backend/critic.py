"""The local prompt critic: a small language model that reads a prompt against
its checkpoint's sources and says what to change.

Layer 1 (`prompt_rules`, `prompt_encoding`) checks what rules can check. What
they cannot - whether the subject comes first, whether "shot on film maybe" is
a camera, what a better sentence would be - is this module's, and it is only
as good as what holds it down. Tried on this card before it was built, a 4B
model without reasoning called a camera missing next to "shot on film" and
suggested quoting a negation; with reasoning on and a numbered source list it
caught subject order, unquoted text and negations. So:

- **Grounded.** The system message carries the guide's sentences and the
  authors' excerpts (`prompt_sources`), numbered; each issue's `cite_id` is an
  enum of those numbers in the answer's JSON schema, and an id that resolves to
  nothing is dropped rather than shown uncited.
- **Checked.** A quote the model claims is in the prompt is looked for there,
  and nulled when it is not; a part of the prompt it names must be the
  prompt's own words, and its text to render words the prompt asks to
  letter; issues that restate a rule finding, and order issues that ask for
  different things first, are left out; the "reordered" variant must be the
  prompt's own phrases - its tags, clauses, sentences or XML blocks - moved
  whole, with any text to render unchanged, and nothing else; a rewrite
  wrapped whole in quotation marks is unwrapped, one that letters words the
  prompt does not have is dropped, and one that quotes description is
  flagged. What survives is still the model's suggestion: the route puts the
  rewrite back through layer 1, because a critic reintroduces what it was
  asked to remove, and the page shows it to be read before it is used.
- **Pinned.** Official llama.cpp `llama-server` b11206 and Qwen3-4B Q8_0 at a
  fixed revision, each checked against its sha256 before first use. The binary
  is a download like the weights, so it is treated like them: `.part`, verify,
  extract beside the target, rename. A failed install is retried, never used.
- **A child process per critique.** Started on loopback and a free port,
  asked one question, stopped in `finally` - terminate, wait, kill, the whole
  process tree - so its ~5.5 GiB of VRAM is free before `jobs.run_when_idle`
  lets an image job in. Server load is under two seconds; a resident server
  would hold the card against the image pipeline for nothing. Its CORS
  allows every origin, so a fresh random API key per start - handed over in
  its environment, never on a command line other local users can read - is
  what keeps a web page in this browser from using it, and the critic
  refuses a server that answers without it, or a port some other process
  listens on.

`prepare()` downloads (outside the GPU gate, since it takes minutes on a first
use); `critique()` runs (inside it). `supported()` and `ready()` are cheap
enough for `/api/status`.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import weakref
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Literal

from .config import settings
from .prompt_rules import QUOTED, rendered_spans
from .prompt_sources import Sources, sources_for
from .schemas import CritiqueIssue, GenerateRequest, PromptAnalysis, PromptCritique, PromptFinding

log = logging.getLogger(__name__)

# --- The pinned runtime ------------------------------------------------------

RELEASE = "b11206"
_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/" + RELEASE + "/{name}"


@dataclass(frozen=True)
class Archive:
    name: str
    size: int
    # The GitHub release API's `digest` for the asset, verified 2026-09-27.
    sha256: str

    @property
    def url(self) -> str:
        return _RELEASE_URL.format(name=self.name)


@dataclass(frozen=True)
class Build:
    # Extracted into one directory, in order: the CUDA runtime archive is
    # packed to sit beside the binaries (their rpath is $ORIGIN on Linux, and
    # Windows searches the executable's directory first).
    archives: tuple[Archive, ...]
    server: str
    # What the binaries take from the system rather than the tree, beyond the
    # C and C++ runtimes: the newest glibc symbol version they name, and
    # shared libraries by soname with the apt package that ships each.
    # `preflight` holds a system to them before anything is downloaded.
    glibc: tuple[int, int] | None = None
    libraries: tuple[tuple[str, str], ...] = ()


# Keyed by the platform part of the release's asset names. CUDA 13.4 builds:
# the default architecture list includes sm_120, so Blackwell runs native code.
BUILDS: dict[str, Build] = {
    "win-cuda-13.4-x64": Build(
        (
            Archive(
                "llama-b11206-bin-win-cuda-13.4-x64.zip",
                152343617,
                "1d775af6af5d034c602f780a3f0dabc9223ad740c0b8ed63fe2d21eeb67f0b86",
            ),
            Archive(
                "cudart-llama-bin-win-cuda-13.4-x64.zip",
                423535356,
                "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
            ),
        ),
        "llama-server.exe",
    ),
    # WSL 2. The tarballs each hold one top-level directory, which the install
    # flattens so the runtime libraries land beside the binaries.
    "ubuntu-cuda-13.4-x64": Build(
        (
            Archive(
                "llama-b11206-bin-ubuntu-cuda-13.4-x64.tar.gz",
                151688270,
                "1dc7adc08fc181e30acf0cff919c7510fda7343b352fd05235d3fde51d4bf9d2",
            ),
            Archive(
                "cudart-llama-b11206-bin-ubuntu-cuda-13.4-x64.tar.gz",
                440236702,
                "486c9d19496d67f293d9bcf6c8e166a95b0cef825b13662652f18e60ff342557",
            ),
        ),
        "llama-server",
        # Read off the installed tree's ELF headers under WSL 2 on 2026-09-28:
        # symbols up to GLIBC_2.38 (and GCC 13's GLIBCXX_3.4.32, not checked
        # here), libgomp for the CPU backends and OpenSSL 3 for the server's
        # HTTP. A fresh WSL Ubuntu has no libgomp1; Ubuntu 22.04 and Debian 12
        # cannot run the build at all.
        glibc=(2, 38),
        libraries=(
            ("libgomp.so.1", "libgomp1"),
            ("libssl.so.3", "libssl3"),
            ("libcrypto.so.3", "libssl3"),
        ),
    ),
}

# Qwen's own GGUF of Qwen3-4B, the post-trained hybrid-reasoning model, at the
# revision checked 2026-09-27 (Apache-2.0). Q8_0 with the shipped settings
# (reasoning on, SAMPLING, an 8k context), per docs/verified.md: 8.1-16.0 s
# per critique on Windows, median 10.5, 19.3 s for a tag prompt, and about
# 5.5 GiB of VRAM over the desktop's. The research's 8.8-9.0 s and ~5.2 GB
# were greedy decoding, which thinking mode is not run with.
MODEL = {
    "repo_id": "Qwen/Qwen3-4B-GGUF",
    "filename": "Qwen3-4B-Q8_0.gguf",
    "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
}
MODEL_SIZE = 4280404704
MODEL_SHA256 = "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300"
MODEL_LABEL = "Qwen3-4B Q8_0"

_INSTALL_MARKER = "local-diffusion-install.json"

# --- The server and the request ----------------------------------------------

CONTEXT = 8192
# Qwen3 thinks for about a thousand tokens on these prompts. The budget ends
# the thought with room left in `max_tokens` for the JSON after it; without
# one, a long think runs the answer off the end and nothing parses.
REASONING_BUDGET = 2048
# The least a critique may generate, thought included: the card's critiques
# generated 705-2,511 tokens on FLUX.1 Krea [dev] prompts and 2,568 on a tag
# prompt (docs/verified.md, "Prompt analyser"). A long prompt is given more
# (`max_tokens`).
MIN_TOKENS = 4096
# An answer's cost besides its copies of the prompt: six issues, their fixes
# and the JSON around them. A short prompt's answer measured under 450.
ANSWER_OVERHEAD = 800
# Qwen3's recommended sampling for thinking mode. Greedy decoding is what its
# card warns against there: it loops.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
# A cold 4 GB read from disk, then the load; `--fit` may put layers on the CPU
# when the image pipeline is holding VRAM, which is what the long request
# timeout is for (13 tok/s on the CPU against ~150 on the card).
HEALTH_TIMEOUT = 180.0
REQUEST_TIMEOUT = 600.0
STOP_TIMEOUT = 10.0
_HEALTH_POLL = 0.25
# A backstop, not the mechanism: a server this process could not stop (it was
# killed outright) drops the model and its VRAM after this long idle.
_IDLE_SLEEP = 120

PARTS = ("subject", "action", "setting", "style", "lighting", "camera", "text_to_render")
ISSUE_KINDS = ("order", "missing", "vague", "conflict", "other")
MAX_ISSUES = 6
MAX_TEXT = 4000
# What the system message may spend on the authors' examples and the rule
# findings. The whole message stays under 1.5k tokens (Qwen3's tokenizer
# averages over four characters a token on it): guide sentences and excerpts
# are the point, and everything else is trimmed around them.
SYSTEM_BUDGET_CHARS = 6000
_MAX_EXAMPLES = 3
_EXAMPLE_CHARS = 280
_MAX_FINDINGS = 8
_FINDING_CHARS = 160


class CriticError(RuntimeError):
    """The critic ran and failed: a download that did not verify, a server
    that did not start, an answer that did not parse."""


class CriticUnavailable(CriticError):
    """The critic cannot run here: switched off, no build for this platform,
    or not prepared yet."""


# --- Where things live -------------------------------------------------------


def platform_key() -> str | None:
    """The pinned build for this machine, or None where there is none."""
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return None
    if sys.platform == "win32":
        return "win-cuda-13.4-x64"
    if sys.platform.startswith("linux"):
        return "ubuntu-cuda-13.4-x64"
    return None


def glibc_version() -> tuple[int, int] | None:
    """This process's glibc, or None where the C library is not glibc (musl,
    Windows). A libc call, no file read: cheap enough for `/api/status`."""
    try:
        name, version = os.confstr("CS_GNU_LIBC_VERSION").split()
        major, minor = version.split(".")[:2]
        return (int(major), int(minor)) if name == "glibc" else None
    except (AttributeError, ValueError, OSError):
        return None


def _old_glibc(key: str) -> str | None:
    """Why this system's C library cannot run the pinned build, or None."""
    need = BUILDS[key].glibc
    if need is None:
        return None
    have = glibc_version()
    if have is not None and have >= need:
        return None
    found = f"glibc {have[0]}.{have[1]}" if have is not None else "no glibc"
    return (
        f"llama.cpp's pinned {key} build needs glibc {need[0]}.{need[1]} or newer "
        f"(Ubuntu 24.04, Debian 13) and this system has {found}; set "
        "LOCAL_DIFFUSION_CRITIC_SERVER to a llama-server built for it"
    )


def pinned_build_refusal() -> str | None:
    """Why this platform's pinned build cannot run here, or None: the
    critique route's 422 where `supported()` is False although a build is
    published for the platform."""
    key = platform_key()
    return _old_glibc(key) if key is not None else None


_LIBRARY_PROBE = """\
import ctypes, sys
for name in sys.argv[1:]:
    try:
        ctypes.CDLL(name)
    except OSError:
        print(name)
"""


def missing_libraries(sonames: tuple[str, ...]) -> list[str]:
    """Those of `sonames` the dynamic loader cannot find.

    Asked of the loader itself - LD_LIBRARY_PATH, its cache and the default
    directories, as it will search them for llama-server - in a child, so
    nothing is loaded into this process (a second OpenMP runtime beside
    torch's). A probe that cannot run checks nothing: the server's own exit
    says what it lacks.
    """
    try:
        probe = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _LIBRARY_PROBE, *sonames],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not check llama.cpp's system libraries: %s", exc)
        return []
    return [name for name in probe.stdout.split() if name in sonames]


def preflight(key: str) -> None:
    """Refuse a pinned build this system cannot run, before anything of it or
    the model is downloaded: a first use is 5 GB, and a missing library would
    otherwise surface only as the server's exit 127 after it."""
    refusal = _old_glibc(key)
    if refusal is not None:
        raise CriticUnavailable(refusal)
    needs = BUILDS[key].libraries
    missing = missing_libraries(tuple(name for name, _ in needs)) if needs else []
    if missing:
        packages = dict.fromkeys(package for name, package in needs if name in missing)
        raise CriticUnavailable(
            f"llama.cpp's pinned {key} build needs {', '.join(missing)}, which this "
            f"system does not have: sudo apt install {' '.join(packages)}"
        )


def install_root() -> Path:
    from huggingface_hub.constants import HF_HOME

    return Path(HF_HOME) / "local-diffusion" / "llama.cpp"


def _recipe(key: str) -> str:
    build = BUILDS[key]
    recipe = {
        "release": RELEASE,
        "platform": key,
        "archives": [[archive.name, archive.sha256] for archive in build.archives],
        "server": build.server,
    }
    return json.dumps(recipe, sort_keys=True, indent=2) + "\n"


def _target(key: str) -> Path:
    return install_root() / f"{RELEASE}-{key}"


def installed_server(key: str) -> Path | None:
    """The installed server binary, or None: never a download.

    The marker is written last, inside the tree before its rename, so a tree
    without it is a failure from elsewhere and does not count - nor does one
    that does not decode (re-saved as UTF-16, say), which `install_server`
    then replaces rather than failing on, as `/api/status` would.
    """
    target = _target(key)
    marker = target / _INSTALL_MARKER
    server = target / BUILDS[key].server
    try:
        if marker.read_text(encoding="utf-8") != _recipe(key):
            return None
    except (OSError, ValueError):
        return None
    return server if server.is_file() else None


# --- Downloading and installing ----------------------------------------------


def _fetch(url: str, destination: Path, size: int) -> None:
    """Stream one release asset to disk. The seam the tests replace."""
    request = urllib.request.Request(url, headers={"User-Agent": "local-diffusion"})
    step = 64 << 20
    done = reported = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done - reported >= step:
                reported = done
                log.info(
                    "downloading %s: %d of %d MB",
                    destination.name.removesuffix(".part"), done >> 20, size >> 20,
                )


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _verify_archive(path: Path, archive: Archive) -> None:
    size = path.stat().st_size
    if size != archive.size:
        raise CriticError(f"{archive.name} is {size} bytes, not the pinned {archive.size}")
    digest = _sha256(path)
    if digest != archive.sha256:
        raise CriticError(f"{archive.name} has sha256 {digest}, not the pinned {archive.sha256}")


def _download(archive: Archive, downloads: Path) -> Path:
    """One verified archive on disk, reusing a verified earlier download."""
    final = downloads / archive.name
    if final.is_file():
        try:
            _verify_archive(final, archive)
            return final
        except CriticError:
            final.unlink()
    part = final.with_name(final.name + ".part")
    part.unlink(missing_ok=True)
    log.info("downloading %s (%d MB)", archive.url, archive.size >> 20)
    try:
        _fetch(archive.url, part, archive.size)
        _verify_archive(part, archive)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(final)
    return final


def _check_member(name: str, root: Path) -> None:
    """Refuse a zip member that would land outside `root`.

    zipfile's own extraction silently rewrites `../x` and `C:/x` into the
    target instead - a tampered archive that has already passed its checksum
    is a pinned-release problem to stop at, not to repair.
    """
    path = PurePosixPath(name.replace("\\", "/"))
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z]:", name)
        or not (root / path).resolve().is_relative_to(root.resolve())
    ):
        raise CriticError(f"archive member {name!r} would land outside the install directory")


def _single_root(directory: Path) -> Path:
    """The one top-level directory a tarball packs everything under, if so."""
    entries = list(directory.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink():
        return entries[0]
    return directory


def _merge(source: Path, tree: Path) -> None:
    for entry in source.iterdir():
        destination = tree / entry.name
        if entry.is_dir() and not entry.is_symlink() and destination.is_dir():
            _merge(entry, destination)
        else:
            os.replace(entry, destination)


def _extract(archive: Path, scratch: Path, tree: Path) -> None:
    scratch.mkdir()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _check_member(member.filename, scratch)
            bundle.extractall(scratch)
    else:
        # The "data" filter refuses absolute names, `..`, links out of the
        # target and device files, rather than extracting them.
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(scratch, filter="data")
        except tarfile.FilterError as exc:
            raise CriticError(f"{archive.name}: {exc}") from exc
    _merge(_single_root(scratch), tree)


def _sweep_work(target: Path) -> None:
    """Remove what interrupted installs left beside `target`.

    Each extracts into a `<target>-*` work directory that removes itself on
    the way out - but not after a hard kill, nor on Windows when the
    interpreter exits with a file in it still open on the prepare thread, and
    each such attempt left up to 744 MB. Only an install holding the lock
    extracts, so under it every one is stale.
    """
    builds = {_target(key).name for key in BUILDS}
    for stale in target.parent.glob(f"{target.name}-*"):
        if stale.is_dir() and not stale.is_symlink() and stale.name not in builds:
            log.warning("removing an interrupted llama.cpp install: %s", stale)
            shutil.rmtree(stale, ignore_errors=True)


def install_server(key: str) -> Path:
    """Download, verify and install the pinned build once; return the binary."""
    from huggingface_hub.utils import WeakFileLock

    build = BUILDS[key]
    target = _target(key)
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    # The server and a CLI process may both ask. A tree is published whole or
    # not at all; the lock holds through the extraction as well, so a work
    # directory the sweep finds under it is never another install's in use.
    with WeakFileLock(root / f"{target.name}.lock"):
        # Installed or not: leftovers from before a finished install count too.
        _sweep_work(target)
        server = installed_server(key)
        if server is not None:
            return server
        if target.exists():
            log.warning("removing an unfinished llama.cpp install: %s", target)
            shutil.rmtree(target)
        downloads = root / "downloads"
        downloads.mkdir(exist_ok=True)
        archives = [_download(archive, downloads) for archive in build.archives]
        with tempfile.TemporaryDirectory(prefix=f"{target.name}-", dir=root) as work:
            tree = Path(work) / "tree"
            tree.mkdir()
            for index, archive in enumerate(archives):
                _extract(archive, Path(work) / f"archive-{index}", tree)
            binary = tree / build.server
            if not binary.is_file():
                raise CriticError(f"the {RELEASE} {key} archives hold no {build.server}")
            if sys.platform != "win32":
                binary.chmod(binary.stat().st_mode | 0o111)
            (tree / _INSTALL_MARKER).write_text(_recipe(key), encoding="utf-8")
            tree.rename(target)
        for archive in archives:
            archive.unlink(missing_ok=True)
    log.info("llama.cpp %s installed: %s", RELEASE, target)
    return target / build.server


def _model_cached() -> Path | None:
    """The pinned GGUF if the Hub cache already holds it; no network."""
    from huggingface_hub import try_to_load_from_cache

    try:
        cached = try_to_load_from_cache(
            MODEL["repo_id"], MODEL["filename"], revision=MODEL["revision"]
        )
    except (OSError, ValueError):
        return None
    return Path(cached) if isinstance(cached, str) and Path(cached).is_file() else None


def verify_model(path: Path) -> None:
    """Hold the GGUF to the pinned size and sha256, hashing it only once.

    Four gigabytes take seconds to hash, which every critique would otherwise
    pay; a stamp of the path, size and mtime under the install root records
    that this file was checked.
    """
    stat = path.stat()
    if stat.st_size != MODEL_SIZE:
        raise CriticError(f"{path} is {stat.st_size} bytes, not the pinned {MODEL_SIZE}")
    stamp_file = install_root() / "verified" / f"{MODEL_SHA256}.json"
    stamp = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    try:
        if json.loads(stamp_file.read_text(encoding="utf-8")) == stamp:
            return
    except (OSError, ValueError):
        pass
    digest = _sha256(path)
    if digest != MODEL_SHA256:
        # Named by the file that holds the bytes: in a Hub cache with symlinks
        # `path` is a snapshot's link to a blob.
        raise CriticError(f"{path.resolve()} has sha256 {digest}, not the pinned {MODEL_SHA256}")
    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")


def download_model() -> Path:
    import huggingface_hub

    if _model_cached() is None:
        log.info(
            "downloading the prompt critic's model %s/%s (%.1f GB)",
            MODEL["repo_id"], MODEL["filename"], MODEL_SIZE / 1e9,
        )
    path = Path(huggingface_hub.hf_hub_download(**MODEL))
    try:
        verify_model(path)
    except CriticError as exc:
        # Fetched again, once. Not by deleting the file and asking: a cache
        # with symlinks relinks the blob it still holds, without a download.
        # Forced, and with the snapshot's entry gone first - in a cache
        # without symlinks that entry *is* the file, and a forced download
        # replaces only the blob beside it.
        log.warning("%s; downloading it again", exc)
        path.unlink(missing_ok=True)
        try:
            path = Path(huggingface_hub.hf_hub_download(**MODEL, force_download=True))
        except Exception as fetch:
            # The Hub's own words for a failed forced download are "the above
            # error", which is its cause.
            reason = fetch.__cause__ or fetch
            raise CriticError(
                f"{MODEL['filename']} in the Hub cache is off its pin and could not be "
                f"downloaded again: {type(reason).__name__}: {reason}"
            ) from fetch
        try:
            verify_model(path)
        except CriticError as again:
            # Not "delete it and retry": a retry has just fetched these bytes.
            raise CriticError(
                f"{again}, even freshly downloaded at the pinned revision: the pin "
                "or the download path is wrong"
            ) from again
    return path


# --- Running the server ------------------------------------------------------


def server_command(server: Path, model: Path, port: int) -> list[str]:
    """The llama-server command line; flags as documented by b11206's --help.

    Both paths absolute: `_spawn` starts the server in its own directory, where
    a relative `-m` - or, on Linux, a relative executable - means another file.
    No `--api-key`: a command line is readable by every local user (under WSL,
    `/proc/<pid>/cmdline`), so the key goes in the environment (`_server_env`).
    """
    return [
        str(server.resolve()),
        "-m", str(model.resolve()),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-ui",
        # Never a network fetch of its own: the model is a local file.
        "--offline",
        "-c", str(CONTEXT),
        "-np", "1",
        "--jinja",
        # Places GPU layers by the VRAM actually free, rather than failing when
        # the pipeline holds some. b11206 adjusts only arguments left unset, so
        # the context stays at `-c` and layers that do not fit go to the CPU.
        "--fit", "on",
        "--reasoning-budget", str(REASONING_BUDGET),
        "--sleep-idle-seconds", str(_IDLE_SLEEP),
    ]


def _server_env(server: Path, api_key: str) -> dict[str, str]:
    """The server's environment: this process's, less every `LLAMA_*` - a
    user's settings for their own servers (`LLAMA_ARG_TOOLS` would switch on
    its shell and file tools) - plus this start's key as `LLAMA_API_KEY`, which
    b11206 reads in place of `--api-key`."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_")}
    env["LLAMA_API_KEY"] = api_key
    if sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(server.parent), env.get("LD_LIBRARY_PATH")])
        )
    return env


_WIN32: Any = None


def _win32() -> Any:
    """kernel32's job object and process calls, ntdll's resume and iphlpapi's
    TCP table, typed once."""
    global _WIN32
    if _WIN32 is None:
        import ctypes
        from ctypes import wintypes

        class Limits(ctypes.Structure):
            # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, its basic part flattened.
            _fields_ = [
                ("per_process_time", ctypes.c_int64),
                ("per_job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t),
                ("process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority_class", wintypes.DWORD),
                ("scheduling_class", wintypes.DWORD),
                ("io_counters", ctypes.c_uint64 * 6),
                ("process_memory_limit", ctypes.c_size_t),
                ("job_memory_limit", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        class ProcessIds(ctypes.Structure):
            # JOBOBJECT_BASIC_PROCESS_ID_LIST, with room for a wrapper's tree.
            _fields_ = [
                ("assigned", wintypes.DWORD),
                ("listed", wintypes.DWORD),
                ("ids", ctypes.c_size_t * 64),
            ]

        class Row4(ctypes.Structure):
            # MIB_TCPROW_OWNER_PID; the port is in network order.
            _fields_ = [
                ("state", wintypes.DWORD),
                ("address", ctypes.c_ubyte * 4),
                ("port", ctypes.c_ubyte * 4),
                ("remote_address", wintypes.DWORD),
                ("remote_port", wintypes.DWORD),
                ("pid", wintypes.DWORD),
            ]

        class Row6(ctypes.Structure):
            # MIB_TCP6ROW_OWNER_PID.
            _fields_ = [
                ("address", ctypes.c_ubyte * 16),
                ("scope", wintypes.DWORD),
                ("port", ctypes.c_ubyte * 4),
                ("remote_address", ctypes.c_ubyte * 16),
                ("remote_scope", wintypes.DWORD),
                ("remote_port", wintypes.DWORD),
                ("state", wintypes.DWORD),
                ("pid", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll")
        iphlpapi = ctypes.WinDLL("iphlpapi")
        handle, info = wintypes.HANDLE, ctypes.c_int
        for name, argtypes, restype in (
            ("CreateJobObjectW", (ctypes.c_void_p, wintypes.LPCWSTR), handle),
            (
                "SetInformationJobObject",
                (handle, info, ctypes.c_void_p, wintypes.DWORD),
                wintypes.BOOL,
            ),
            ("AssignProcessToJobObject", (handle, handle), wintypes.BOOL),
            ("TerminateJobObject", (handle, wintypes.UINT), wintypes.BOOL),
            (
                "QueryInformationJobObject",
                (handle, info, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)),
                wintypes.BOOL,
            ),
            ("CloseHandle", (handle,), wintypes.BOOL),
            ("OpenProcess", (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD), handle),
            ("IsProcessInJob", (handle, handle, ctypes.POINTER(wintypes.BOOL)), wintypes.BOOL),
            ("WaitForSingleObject", (handle, wintypes.DWORD), wintypes.DWORD),
        ):
            function = getattr(kernel32, name)
            function.argtypes, function.restype = argtypes, restype
        # Undocumented, but ntdll's since NT: resumes every thread of a
        # process from the process handle, which is all Popen keeps.
        ntdll.NtResumeProcess.argtypes = (handle,)
        ntdll.NtResumeProcess.restype = ctypes.c_long
        iphlpapi.GetExtendedTcpTable.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL, wintypes.ULONG,
            ctypes.c_int, wintypes.ULONG,
        )
        iphlpapi.GetExtendedTcpTable.restype = wintypes.DWORD
        _WIN32 = SimpleNamespace(
            ctypes=ctypes, wintypes=wintypes, kernel32=kernel32, ntdll=ntdll,
            iphlpapi=iphlpapi, Limits=Limits, ProcessIds=ProcessIds,
            rows={socket.AF_INET: Row4, socket.AF_INET6: Row6},
        )
    return _WIN32


def _new_job(process: subprocess.Popen) -> int | None:
    """A job object holding `process`, which kills everything in it when its
    last handle closes; None, and the child alone, if Windows refuses one."""
    win = _win32()
    job = win.kernel32.CreateJobObjectW(None, None)
    if not job:
        log.warning("no job object for llama-server (error %d)", win.ctypes.get_last_error())
        return None
    limits = win.Limits(flags=0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not (
        win.kernel32.SetInformationJobObject(
            job, 9, win.ctypes.byref(limits), win.ctypes.sizeof(limits)
        )
        and win.kernel32.AssignProcessToJobObject(job, int(process._handle))
    ):
        log.warning(
            "llama-server is not in a job object (error %d); only it is stopped, not "
            "anything it starts", win.ctypes.get_last_error(),
        )
        win.kernel32.CloseHandle(job)
        return None
    return job


def _job_pids(job: int) -> set[int]:
    """The processes still in `job`: empty once every one has exited."""
    win = _win32()
    listed = win.ProcessIds()
    # ERROR_MORE_DATA past 64 processes still fills the first 64.
    win.kernel32.QueryInformationJobObject(
        job, 3, win.ctypes.byref(listed), win.ctypes.sizeof(listed), None
    )
    return {int(pid) for pid in listed.ids[: min(listed.listed, len(listed.ids))]}


def _waitable(pid: int, job: int) -> int | None:
    """A handle to wait on process `pid` of `job` with; None if it is gone,
    or if the pid is no longer that job's."""
    win = _win32()
    # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
    handle = win.kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        return None
    inside = win.wintypes.BOOL()
    if win.kernel32.IsProcessInJob(handle, job, win.ctypes.byref(inside)) and inside:
        return handle
    win.kernel32.CloseHandle(handle)
    return None


def _release(job: int | None, held: dict[int, int]) -> None:
    """Close what a `_ServerProcess` holds on Windows; the job last, whose
    closing kills whatever is still in it."""
    win = _win32()
    for handle in held.values():
        win.kernel32.CloseHandle(handle)
    held.clear()
    if job is not None:
        win.kernel32.CloseHandle(job)


def _session_pids(session: int) -> set[int]:
    """The live processes in a session, by /proc: a POSIX child started in a
    session of its own and everything it starts, whatever process group each
    is in. Empty without /proc.

    A zombie holds nothing and is left out - but only once it is one whole:
    /proc shows a process by its first thread, which can be a zombie while
    the others, still exiting, hold the process's files and sockets."""
    pids = set()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return pids
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as stat:
                # state, ppid, pgrp, session after the parenthesised name.
                fields = stat.read().rpartition(")")[2].split()
            if len(fields) < 4 or fields[3] != str(session):
                continue
            if fields[0] in ("Z", "X") and len(os.listdir(f"/proc/{entry}/task")) < 2:
                continue
        except OSError:
            continue
        pids.add(int(entry))
    return pids


class _ServerProcess(subprocess.Popen):
    """llama-server and whatever it starts, as one tree.

    `LOCAL_DIFFUSION_CRITIC_SERVER` may name a wrapper - a .bat, a venv
    trampoline, a shell script that does not `exec` - whose server is a
    grandchild, which stopping the direct child alone left listening, and
    holding the card, after the gate opened. So `terminate`, `kill` and
    `wait` here reach the whole tree, the same tree `pids()` names for the
    listener check. On Windows it is a job object, joined before the child
    runs - it starts suspended and is resumed once inside - which also kills
    the tree when its handle closes, as a hard kill of this app closes it.
    On POSIX it is a session of the child's own, whose id is the child's
    pid; on Linux every process in it is signalled and waited for, whatever
    group it has moved to, and elsewhere the child's process group. Nothing
    ends that session with the app: an app killed outright, or by its
    terminal's hang-up (which no longer reaches the server), leaves it to
    `--sleep-idle-seconds` to free the card.
    """

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self._job: int | None = None
        # Windows: a handle to each process seen in the job, to wait on.
        self._held: dict[int, int] = {}
        if sys.platform != "win32":
            super().__init__(command, start_new_session=True, **kwargs)
            return
        flags = kwargs.pop("creationflags", 0) | 0x4  # CREATE_SUSPENDED
        super().__init__(command, creationflags=flags, **kwargs)
        try:
            self._job = _new_job(self)
            status = _win32().ntdll.NtResumeProcess(int(self._handle))
            if status < 0:
                raise OSError(f"could not resume the process (NTSTATUS {status & 0xFFFFFFFF:#010x})")
        except BaseException:
            # Never a suspended child left behind: Popen's own kill is
            # TerminateProcess, and closing the job kills it again.
            super().kill()
            _release(self._job, self._held)
            self._job = None
            raise
        weakref.finalize(self, _release, self._job, self._held)

    def pids(self) -> set[int]:
        """The tree's process ids, the child's own included."""
        if self._job is not None:
            return {self.pid} | _job_pids(self._job)
        if sys.platform.startswith("linux"):
            return {self.pid} | _session_pids(self.pid)
        return {self.pid}

    def _hold(self) -> set[int]:
        """The processes still in the job, each with a handle to wait on
        taken when first seen. A process leaves the job's list before its
        handles - the log, the listening socket - are closed; its own handle
        is signalled only after they are."""
        pids = _job_pids(self._job)
        for pid in pids - self._held.keys():
            handle = _waitable(pid, self._job)
            if handle is not None:
                self._held[pid] = handle
        return pids

    def _signal_tree(self, number: int) -> None:
        if sys.platform == "win32":
            if self._job is not None:
                self._hold()
                if _win32().kernel32.TerminateJobObject(self._job, 1):
                    return
            super().kill()
            return
        try:
            os.killpg(self.pid, number)
        except OSError:
            # Nothing left in the group, or nothing this user may signal.
            super().send_signal(number)
        # The rest of the session: a wrapper may have put its server in a
        # group of its own (`set -m`), which `pids()` still counts.
        for pid in _session_pids(self.pid) - {self.pid}:
            try:
                os.kill(pid, number)
            except OSError:
                pass

    def terminate(self) -> None:
        # On Windows both end the job at once, as Popen's own terminate is a
        # kill there too.
        self._signal_tree(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_tree(getattr(signal, "SIGKILL", signal.SIGTERM))

    def _others_alive(self) -> bool:
        if self._job is not None:
            return bool(self._hold())
        if sys.platform == "win32":
            return False
        if sys.platform.startswith("linux"):
            return bool(_session_pids(self.pid))
        try:
            os.killpg(self.pid, 0)
        except OSError:
            return False
        return True

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the child, then for the rest of its tree - whose exit is
        what frees the card when the child is only a wrapper - and on
        Windows until each process has closed its handles, not only left
        the job's list."""
        deadline = None if timeout is None else time.monotonic() + timeout
        code = super().wait(timeout)
        while self._others_alive():
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.05)
        win = _win32() if self._held else None
        for pid, handle in list(self._held.items()):
            left = 0xFFFFFFFF if deadline is None else (deadline - time.monotonic()) * 1000
            if win.kernel32.WaitForSingleObject(handle, max(0, int(left))) == 0x102:
                raise subprocess.TimeoutExpired(self.args, timeout)  # WAIT_TIMEOUT
            win.kernel32.CloseHandle(handle)
            del self._held[pid]
        return code


def _spawn(command: list[str], log_file: Any, api_key: str) -> subprocess.Popen:
    """Start the server as a `_ServerProcess`. The seam the tests replace."""
    server = Path(command[0])
    return _ServerProcess(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=server.parent,
        env=_server_env(server, api_key),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


# A proxy from the environment must never see loopback traffic - it would be
# handed the prompt and the key, or answer for a server it cannot reach.
_LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(
    method: str, url: str, api_key: str | None, body: Any | None, timeout: float
) -> tuple[int, Any]:
    """One HTTP exchange with the server: (status, decoded body). The seam the
    tests replace."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _LOOPBACK.open(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, raw.decode("utf-8", "replace")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# Where a listener takes connections to 127.0.0.1: bound there, to every
# address, or (dual-stack) to IPv6's.
_REACHES_LOOPBACK = {
    socket.AF_INET: frozenset({"127.0.0.1", "0.0.0.0"}),
    socket.AF_INET6: frozenset({"::", "::ffff:127.0.0.1"}),
}


def _windows_listeners(port: int) -> set[int] | None:
    """The pids listening where 127.0.0.1:`port` goes, from the TCP tables."""
    win = _win32()
    ctypes = win.ctypes
    owners: set[int] = set()
    for family, row_type in win.rows.items():
        size = ctypes.c_uint32(0)
        table = None
        for _ in range(4):
            # TCP_TABLE_OWNER_PID_LISTENER; ERROR_INSUFFICIENT_BUFFER says how
            # much the table needs, which may grow between calls.
            code = win.iphlpapi.GetExtendedTcpTable(table, ctypes.byref(size), False, family, 3, 0)
            if code == 0 and table is not None:
                break
            if code not in (0, 122):
                return None
            table = ctypes.create_string_buffer(size.value)
        else:
            return None
        count = ctypes.c_uint32.from_buffer(table).value
        for row in (row_type * count).from_buffer(table, 4):
            address = socket.inet_ntop(family, bytes(row.address))
            if row.port[0] << 8 | row.port[1] == port and address in _REACHES_LOOPBACK[family]:
                owners.add(int(row.pid))
    return owners


def _linux_listeners(port: int) -> set[str] | None:
    """The socket inodes listening where 127.0.0.1:`port` goes, from /proc."""
    inodes: set[str] = set()
    for family, name in ((socket.AF_INET, "tcp"), (socket.AF_INET6, "tcp6")):
        try:
            with open(f"/proc/net/{name}", encoding="ascii") as table:
                lines = table.read().splitlines()[1:]
        except OSError:
            if family == socket.AF_INET:
                return None
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # TCP_LISTEN
                continue
            address, _, hex_port = fields[1].partition(":")
            # Each 32-bit word printed as the host reads it.
            raw = b"".join(
                int(address[i : i + 8], 16).to_bytes(4, sys.byteorder)
                for i in range(0, len(address), 8)
            )
            reaches = socket.inet_ntop(family, raw) in _REACHES_LOOPBACK[family]
            if int(hex_port, 16) == port and reaches:
                inodes.add(fields[9])
    return inodes


def _socket_inodes(pids: set[int]) -> set[str]:
    inodes: set[str] = set()
    for pid in pids:
        try:
            descriptors = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                link = os.readlink(f"/proc/{pid}/fd/{descriptor}")
            except OSError:
                continue
            if link.startswith("socket:["):
                inodes.add(link[len("socket:[") : -1])
    return inodes


def _listener_is_ours(port: int, pids: set[int]) -> bool | None:
    """Whether what listens where a connection to 127.0.0.1:`port` goes is
    the server's own process tree (`pids`): True; False when anything else
    listens there, or nothing can be seen to; None where this platform gives
    no way to tell. The seam the tests replace.

    Windows names each listener's pid; Linux names its socket inode, which
    only the tree's own descriptors can show - another user's are unreadable,
    and theirs is exactly the listener to refuse.
    """
    try:
        if sys.platform == "win32":
            owners = _windows_listeners(port)
            return None if owners is None else bool(owners) and owners <= pids
        if sys.platform.startswith("linux"):
            inodes = _linux_listeners(port)
            return None if inodes is None else bool(inodes) and inodes <= _socket_inodes(pids)
    except (OSError, ValueError) as exc:
        log.warning("could not see who listens on port %d: %s", port, exc)
    return None


def _tail(path: Path, lines: int = 12) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            source.seek(max(0, source.tell() - 4096))
            text = source.read().decode("utf-8", "replace")
    except OSError:
        return "(no server output)"
    return " | ".join(line.strip() for line in text.splitlines()[-lines:] if line.strip())


def _error_message(payload: Any) -> str:
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            return str(error["message"])
    return str(payload)[:300]


# The Ubuntu build's libraries from outside its tree (its RUNPATH is $ORIGIN),
# by the Ubuntu package that installs each.
_APT_PACKAGES = {"libgomp.so.1": "libgomp1", "libssl.so.3": "libssl3", "libcrypto.so.3": "libssl3"}


def _exited(code: int, log_path: Path) -> str:
    tail = _tail(log_path)
    missing = re.search(r"([\w.+-]+\.so[\w.]*): cannot open shared object file", tail)
    if code == 127 and missing:
        library = missing.group(1)
        package = _APT_PACKAGES.get(library)
        if package is not None:
            return (
                f"llama-server could not start: {library} is missing; install it with "
                f"`sudo apt install {package}` (under WSL, inside the distribution)"
            )
        return f"llama-server could not start: {library} is missing: {tail}"
    return f"llama-server exited with code {code} before it was ready: {tail}"


def _wait_until_healthy(process: subprocess.Popen, base: str, log_path: Path) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while True:
        code = process.poll()
        if code is not None:
            raise CriticError(_exited(code, log_path))
        try:
            # /health is public and answers 503 while the model loads.
            status, _ = _request("GET", base + "/health", None, None, 2.0)
        except (OSError, http.client.HTTPException):
            status = None
        # Alive as well as answering. That alone does not prove the answer
        # is ours: b11206 binds with SO_REUSEADDR, so a stranger's server that
        # took the port in the gap after `_free_port` need not make ours exit.
        # Nor does `_run`'s keyless /v1/models 401: it refuses a server that
        # ignores its key, but a stranger can answer 401 too. The guard is
        # `_run`'s check that the port's listener is the server's own process
        # tree, before anything carries the key - where the platform can tell.
        if status == 200 and process.poll() is None:
            return
        if time.monotonic() >= deadline:
            raise CriticError(
                f"llama-server was not ready after {HEALTH_TIMEOUT:g} s: {_tail(log_path)}"
            )
        time.sleep(_HEALTH_POLL)


def _stop(process: subprocess.Popen) -> None:
    """Terminate, wait, kill - the whole tree of a `_ServerProcess`, even when
    the child itself has exited, since a wrapper's server outlives it. A kill
    not waited out either is an error: the card may still be taken."""
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise CriticError(
                f"llama-server (pid {process.pid}) had not exited {STOP_TIMEOUT:g} s after "
                "it was killed; it may still hold GPU memory"
            ) from exc


def _remove_log(path: Path, attempts: int = 5) -> None:
    """Delete the server's log, retried over a few hundred milliseconds as
    `storage._rename_with_retry` renames: on Windows a scanner or an indexer
    may hold a just-closed file (WinError 32). A log still held after that -
    by one of those, or a process outside the tree - is only logged: it is
    not worth the answer it would replace."""
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            if attempt == attempts - 1:
                log.warning("could not remove llama-server's log %s: %s", path, exc)
                return
            time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            log.warning("could not remove llama-server's log %s: %s", path, exc)
            return


# --- The question ------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    text: str
    url: str | None
    source: Literal["guide", "author"]


@dataclass(frozen=True)
class CriticContext:
    """Everything one critique asks with, and resolves its answer against."""

    sources: Sources
    citations: dict[str, Citation]
    # Whether the app checks an order for the checkpoint (`analysis.order`);
    # only then may `reordered` be anything but null.
    ordered: bool
    # How the prompt reads (`analysis.style`): what a piece of it is, for
    # telling a reorder from a rewrite.
    style: str
    # The rule ids layer 1 reported, where in the prompt each one points, and
    # whether the guide has a word range the rules measured: an issue
    # restating any of them is left out.
    reported: frozenset[str]
    reported_spans: tuple[tuple[str, int, int], ...]
    words_measured: bool
    system: str
    user: str
    schema: dict[str, Any]
    # The prompt's tokens, and both messages': `estimate_tokens` until the
    # server has counted them (`_counted`).
    prompt_tokens: int
    input_tokens: int

    @property
    def copies(self) -> int:
        """How often the answer repeats the prompt: about once across
        `parts`, once in `rewrite`, and once more in `reordered`."""
        return 3 if self.ordered else 2


# Qwen2's pre-tokenizer, which Qwen3 keeps, in `re` (a letter is [^\W\d_]):
# a word with the one mark or space before it, a digit, a run of punctuation,
# whitespace. Its merges never cross these pieces, so each is a token at least.
_PIECES = re.compile(
    r"'(?i:[sdmt]|ll|ve|re)|((?:[^\r\n\w]|_)?)([^\W\d_]+)|\d"
    r"| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Tokens per letter past ASCII, by script: Qwen3-4B's tokenizer on ordinary
# text in each, with room over its mean. Other letters take 1.5.
_SCRIPT_RATES = (
    (0x00C0, 0x024F, 1.2),  # Latin-1 and Latin Extended
    (0x0370, 0x03FF, 1.2),  # Greek
    (0x0400, 0x052F, 0.7),  # Cyrillic
    (0x0530, 0x058F, 1.35),  # Armenian
    (0x0590, 0x05FF, 0.7),  # Hebrew
    (0x0600, 0x06FF, 0.85),  # Arabic
    (0x0750, 0x077F, 0.85),  # Arabic Supplement
    (0x0900, 0x0DFF, 2.0),  # Devanagari to Sinhala, split around their vowel signs
    (0x0E00, 0x0EFF, 1.0),  # Thai, Lao
    (0x10A0, 0x10FF, 1.35),  # Georgian
    (0x1100, 0x11FF, 1.3),  # Hangul Jamo
    (0x1E00, 0x1EFF, 1.2),  # Latin Extended Additional
    (0x1F00, 0x1FFF, 1.2),  # Greek Extended
    (0x3040, 0x309F, 1.1),  # Hiragana
    (0x30A0, 0x30FF, 0.8),  # Katakana
    (0x3130, 0x318F, 1.3),  # Hangul Compatibility Jamo
    (0x3400, 0x4DBF, 1.0),  # CJK Extension A
    (0x4E00, 0x9FFF, 0.8),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF, 1.3),  # Hangul Syllables
    (0xF900, 0xFAFF, 1.0),  # CJK Compatibility Ideographs
)


def _letter_rate(char: str) -> float:
    code = ord(char)
    for first, last, rate in _SCRIPT_RATES:
        if first <= code <= last:
            return rate
    return 1.5


def _mark_rate(char: str) -> float:
    """Punctuation a third of a token; a symbol one, an emoji past the Basic
    Multilingual Plane a quarter more; a joiner or a combining accent two,
    since they seldom merge."""
    if char < "\x80":
        return 1 / 3
    code = ord(char)
    if 0x0300 <= code <= 0x036F or code in (0x200C, 0x200D):
        return 2.0
    return 1.25 if code > 0xFFFF else 1.0


# A mixed-case run's humps: a capital run, or a word with its one capital.
_HUMPS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")


def _ascii_word_cost(letters: str) -> float:
    """Under a third of a token per letter in a word, more in capitals; a
    mixed-case run - an identifier, a brand, base64 - a token at least per
    hump, since its humps seldom merge across a change of case."""
    if len(letters) > 1 and letters.isupper():
        return 0.45 * len(letters)
    if letters.islower() or letters.isupper() or letters[1:].islower():
        return 0.3 * len(letters)
    return sum(
        max(1.0, (0.45 if len(hump) > 1 and hump.isupper() else 0.3) * len(hump))
        for hump in _HUMPS.findall(letters)
    )


def estimate_tokens(text: str) -> int:
    """Qwen3's token count for `text`, estimated before a server runs to
    count it (`_counted`).

    Piece by piece as its pre-tokenizer splits the text (`_PIECES`, after
    the NFC its normalizer applies), each a token at least - so a digit is
    one, and so is each run of letters between a hex digest's digits - at a
    rate per character for the rest. Measured against Qwen3-4B's tokenizer
    (its GGUF's own vocabulary) on the guides, the system messages, the
    design notes, ordinary text in two dozen scripts and on hex, base64,
    URLs, JSON, XML and emoji with joiners, it is long by a quarter to two
    fifths on English prose, about a fifth on tags and simplified Chinese,
    and up to half on Cyrillic, Arabic and Thai - three fifths at most on
    ordinary text (Vietnamese), which `_ESTIMATE_FLOOR` relies on; a list of
    camel-case identifiers or brands, whole words in the vocabulary, comes
    out two and a half times over. It is short on what it cannot see,
    vocabulary: by a fifth on Turkish, Polish or Icelandic words, romanised
    names and base64, a tenth on traditional Chinese, a quarter on kanji
    names, two fifths on Cantonese and half on rare Han characters. So it is
    a bound in neither direction: scaled down, it refuses before a server
    starts only what cannot fit (`check_fits`), and it sizes a question only
    for a server that will not count it.
    """
    text = unicodedata.normalize("NFC", text)
    total = 0.0
    for match in _PIECES.finditer(text):
        piece, mark, letters = match.group(0), match.group(1), match.group(2)
        if letters:
            plain = "".join(char for char in letters if char < "\x80")
            cost = _ascii_word_cost(plain) if plain else 0.0
            cost += sum(_letter_rate(char) for char in letters if char >= "\x80")
            if mark and mark != " ":
                cost += _mark_rate(mark)
        elif piece.isdigit():
            cost = 1.0
        else:
            cost = sum(_mark_rate(char) for char in piece if char != " ")
        # A space seldom merges with what follows it past ASCII: two more
        # tokens before kana or Han, one before a symbol.
        if piece[:1] == " " and piece[1:2] >= "\x80":
            if not piece[1].isalpha():
                cost += 1.0
            elif 0x3040 <= ord(piece[1]) <= 0x9FFF:
                cost += 2.0
        total += max(1.0, cost)
    return math.ceil(total)


def max_tokens(context: CriticContext) -> int:
    """What one critique may generate: its thought, then an answer that
    repeats the prompt `copies` times, then the issues - so the thought keeps
    its whole `REASONING_BUDGET` wherever the window has room for it. Sized
    by the server's own count of the prompt (`_counted`) once it runs, and by
    the estimate only for a server that cannot count.

    Not cut to the window the input leaves: b11206 stops at the window's end
    itself (context shift is off by default; finish_reason "length"), and a
    count that errs long would stop an answer the window could hold.
    """
    need = REASONING_BUDGET + context.copies * context.prompt_tokens + ANSWER_OVERHEAD
    return min(CONTEXT, max(MIN_TOKENS, need))


# The estimate's longest overcount of ordinary text is three fifths
# (`estimate_tokens`); scaled by this, a question it refuses cannot fit
# however that text tokenizes.
_ESTIMATE_FLOOR = 0.6


def check_fits(context: CriticContext, *, estimated: bool = False) -> None:
    """Refuse a question whose answer cannot fit the window even with no
    thought at all - 4000 characters of Chinese, say, which the answer would
    repeat to well past 8192 tokens.

    Twice. Before any server starts, on the estimate (`estimated`) scaled
    down to what it can promise, so what surely cannot fit is refused
    without a server's start (3 s on the card, half a minute under WSL with
    the model on /mnt/c). Then on the server's own count (`_counted`),
    before the question is sent: what is admitted fits, but for an answer
    longer than `copies` of the prompt and `ANSWER_OVERHEAD`, and what does
    not is refused rather than left to run to the window's end.
    """
    need = context.input_tokens + context.copies * context.prompt_tokens + ANSWER_OVERHEAD
    if (need * _ESTIMATE_FLOOR if estimated else need) > CONTEXT:
        raise CriticError(
            f"the prompt is too long for the critic: its answer repeats the prompt "
            f"{context.copies} times, which with the sources does not fit its "
            f"{CONTEXT}-token window; shorten the prompt or the negative prompt"
        )


_COUNT_TIMEOUT = 30.0


def _count(base: str, api_key: str, text: str, *, special: bool) -> int | None:
    """The server's own token count of `text`, or None if it gives none."""
    status, payload = _request(
        "POST", base + "/tokenize", api_key,
        {"content": text, "add_special": special, "parse_special": special}, _COUNT_TIMEOUT,
    )
    tokens = payload.get("tokens") if status == 200 and isinstance(payload, Mapping) else None
    return len(tokens) if isinstance(tokens, list) else None


def _counted(
    base: str, api_key: str, body: dict[str, Any], context: CriticContext, prompt: str
) -> CriticContext | None:
    """`context` counted by the running server's tokenizer - the pinned
    GGUF's own vocabulary - in place of the estimate: the messages as its
    chat template renders them, which is exactly what it reads, and the
    prompt the answer repeats, as text it writes. None when the server does
    not say (a server of one's own without b11206's `/apply-template` or
    `/tokenize`), and the estimate stands. Both endpoints take the key, so
    this runs after the listener check."""
    status, rendered = _request(
        "POST", base + "/apply-template", api_key,
        {"messages": body["messages"], "chat_template_kwargs": body["chat_template_kwargs"]},
        _COUNT_TIMEOUT,
    )
    template = rendered.get("prompt") if status == 200 and isinstance(rendered, Mapping) else None
    if not isinstance(template, str):
        return None
    input_tokens = _count(base, api_key, template, special=True)
    prompt_tokens = _count(base, api_key, prompt, special=False)
    if input_tokens is None or prompt_tokens is None:
        return None
    return replace(context, input_tokens=input_tokens, prompt_tokens=prompt_tokens)


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def response_schema(cite_ids: list[str], ordered: bool) -> dict[str, Any]:
    """The answer's shape, enforced by llama-server's grammar sampling.

    `cite_id` is an enum of the ids the message numbered, so a citation cannot
    be invented, only misapplied. Properties generate in the order listed:
    the parts first, as the model's own reading of the prompt.
    """
    nullable = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "parts": {
                "type": "object",
                "properties": {name: nullable for name in PARTS},
                "required": list(PARTS),
                "additionalProperties": False,
            },
            "issues": {
                "type": "array",
                "maxItems": MAX_ISSUES,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": list(ISSUE_KINDS)},
                        "problem": {"type": "string"},
                        "quote": nullable,
                        "cite_id": {"enum": cite_ids},
                        "fix": {"type": "string"},
                    },
                    "required": ["kind", "problem", "quote", "cite_id", "fix"],
                    "additionalProperties": False,
                },
            },
            "rewrite": {"type": "string"},
            # Forced null where no order is documented: nothing to permute
            # into, and nothing for the model to spend tokens inventing.
            "reordered": nullable if ordered else {"type": "null"},
        },
        "required": ["parts", "issues", "rewrite", "reordered"],
        "additionalProperties": False,
    }


_NOT_ABOUT_WORDS = frozenset(("encoder.cut", "encoder.template_tail"))


def _finding_line(finding: PromptFinding) -> str:
    return _one_line(f"- {finding.rule} ({finding.severity}): {finding.message}", _FINDING_CHARS)


def build_context(
    model_id: str,
    request: GenerateRequest,
    applied: GenerateRequest,
    analysis: PromptAnalysis,
) -> CriticContext:
    """The system and user messages, the schema and the citation table."""
    sources = sources_for(model_id)
    citations: dict[str, Citation] = {}
    for index, sentence in enumerate(sources.sentences, 1):
        citations[f"G{index}"] = Citation(sentence.text, sentence.url, "guide")
    for index, excerpt in enumerate(sources.excerpts, 1):
        citations[f"A{index}"] = Citation(excerpt.text, excerpt.url, "author")
    order = analysis.order
    ordered = order is not None
    negative = applied.negative_prompt or None
    quotes = "quot" in sources.guide["prompt"].lower()

    # The rewrite padded short prompts with scenery and mood in nearly every
    # run on the card, against the sources' own "Start short", and wrapped
    # whole prompts in quotation marks on "text in quotation marks".
    rewrite_rule = (
        "- rewrite: the prompt revised to obey the sources. Keep the user's intent: the "
        "same subjects, count, medium and any text to render. Add no object, scenery, "
        "mood or text the prompt lacks, and never pad it to reach a word range: a range "
        "is the guide's band for most images, not a minimum. Use the form the sources "
        "ask for (sentences, tags or XML) and fix the rule findings too."
    )
    if negative is None:
        rewrite_rule += (
            " No negative prompt runs, so describe what should be seen, never what should not."
        )
    if quotes:
        rewrite_rule += (
            " Put quotation marks only around words the image must show as lettering - "
            "never around the prompt or a description."
        )
    if ordered:
        order_rule = (
            "- reordered: the prompt's own phrases moved whole into the documented order "
            "below and nothing else - add, drop, change or split no word, and move no word "
            "from one phrase to another."
        )
    else:
        # Not "no order is documented": Anima's card, Neta Lumina's prompt
        # book and ERNIE's enhancer all give one in the excerpts, which an
        # order issue may cite. What is missing is the app's own check.
        order_rule = (
            "- reordered: null; this app checks no order for this checkpoint. An order "
            "a source below states may still be an issue that cites it."
        )
    # The fallback has no authors (`sources.url` is None): its sentences are
    # the app's general guidance, and its caveat says so.
    whose = ", from the checkpoint's authors" if sources.url else ""
    lines = [
        f"You review a text-to-image prompt for the checkpoint {model_id}. Judge it only "
        f"against the numbered sources below{whose}; general habits and advice written "
        "for other models do not count.",
        "Answer in the JSON schema given:",
        "- parts: the prompt's own words for each part, null where it has none.",
        f"- issues: at most {MAX_ISSUES}, most important first, and none that no source "
        "supports. Each cites exactly one source id. quote: the prompt's exact words the "
        "issue is about, or null. Do not repeat the rule findings listed at the end.",
        rewrite_rule,
        order_rule,
    ]
    if sources.caveat:
        lines += ["", sources.caveat]
    lines += ["", "GUIDE"]
    lines += [
        f"G{index}: {sentence.text}" for index, sentence in enumerate(sources.sentences, 1)
    ]
    if sources.excerpts:
        lines += ["", "AUTHORS"]
        # A note shared by a run of excerpts is written out once; the rest
        # point back to it, which is most of FLUX.2's and Krea [dev]'s list.
        first_with: dict[str, int] = {}
        for index, excerpt in enumerate(sources.excerpts, 1):
            note = ""
            if excerpt.note in first_with:
                note = f" (as A{first_with[excerpt.note]})"
            elif excerpt.note:
                first_with[excerpt.note] = index
                note = f" ({excerpt.note})"
            lines.append(f"A{index}: {excerpt.text}{note}")
    if order is not None:
        lines += [
            "",
            "DOCUMENTED ORDER: " + " > ".join(order.steps)
            + f' - "{_one_line(order.cite, 200)}"',
        ]
    examples = [example for example in sources.guide.get("examples", []) if example.strip()]
    if examples:
        # Their form only: one rewrite on the card ended in the sign example's
        # "Black Forest Labs", another in the owl example's colour grading.
        lines += ["", "AUTHORS' EXAMPLE PROMPTS (their form only; never copy their subjects or text)"]
        lines += [f"- {_one_line(example, _EXAMPLE_CHARS)}" for example in examples[:_MAX_EXAMPLES]]
    if analysis.findings:
        lines += ["", "RULE FINDINGS (already reported)"]
        lines += [_finding_line(finding) for finding in analysis.findings[:_MAX_FINDINGS]]

    user = f"Prompt:\n{request.prompt}"
    if negative is not None:
        user += f"\n\nNegative prompt (it runs):\n{negative}"
    system = "\n".join(lines)
    return CriticContext(
        sources=sources,
        citations=citations,
        ordered=ordered,
        style=analysis.style,
        reported=frozenset(finding.rule for finding in analysis.findings),
        # Findings about the words at their span; not a cut, whose span is
        # everything past where an encoder stops reading.
        reported_spans=tuple(
            (finding.rule, finding.start, finding.end)
            for finding in analysis.findings
            if finding.field == "prompt" and finding.start is not None
            and finding.end is not None and finding.rule not in _NOT_ABOUT_WORDS
        ),
        words_measured="words" in (sources.guide.get("analysis") or {}),
        system=system,
        user=user,
        schema=response_schema(list(citations), ordered),
        prompt_tokens=estimate_tokens(request.prompt),
        input_tokens=estimate_tokens(system) + estimate_tokens(user),
    )


def request_body(context: CriticContext) -> dict[str, Any]:
    """The chat completion request, in the shape measured against b11206."""
    return {
        "messages": [
            {"role": "system", "content": context.system},
            {"role": "user", "content": context.user},
        ],
        **SAMPLING,
        "max_tokens": max_tokens(context),
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "critique", "schema": context.schema},
        },
    }


# --- Reading the answer ------------------------------------------------------

_QUOTE_TRIM = " \t\r\n\"'“”‘’„«»「」『』`.,;:!?"

# Text to render is what `prompt_rules.QUOTED` reads as quoted - straight and
# curly quotes, „…“, «…», CJK corner brackets, single quotes that do not touch
# a word - the one definition the rules use too. A reorder carries each quoted
# stretch character for character; a „…“ or «…» split across two pieces once
# passed here, when this module kept a narrower copy.
#
# Where one of a prompt's pieces ends. A tag list's items; prose's clauses and
# sentences (a stop followed by space, so FLUX.1 and 3.5 stay whole). Chinese
# punctuation counts as its English counterpart.
_TAG_BREAK = re.compile(r"[,，、\n]")
_PROSE_BREAK = re.compile(r"[,;，、；\n]|[.!?](?=\s|$)|[。！？]")
_PIECE_TRIM = " \t\r\n.,;:!?。，、；：！？"
# NewBie's top-level blocks, <character_1>...</character_1> and <general_tags>.
_XML_BLOCK = re.compile(r"<([A-Za-z_][\w.-]*)>.*?</\1>", re.DOTALL)
_HELD = re.compile(r"\x00(\d+)\x00")


def _pieces(text: str, style: str) -> list[str]:
    """The phrases a reorder may move, normalised, in order.

    A piece is a tag, a clause or sentence, an XML block - moved whole or not
    at all - so no attribute changes owner: in "a red dress and a blue hat"
    the colours stay where they are, which a count of words cannot see. Quoted
    text never splits, whatever punctuation it holds. Case, runs of whitespace
    and the punctuation at a piece's ends are not part of it.
    """
    held: list[str] = []
    # NUL marks what is held; a prompt's own would name nothing.
    text = text.replace("\x00", "")

    def hold(part: str, *, alone: bool = False) -> str:
        held.append(part)
        mark = f"\x00{len(held) - 1}\x00"
        # A block is a piece by itself: set off by breaks on both sides.
        return f"\n{mark}\n" if alone else mark

    if style == "xml":
        text = _XML_BLOCK.sub(
            lambda match: hold(
                re.sub(r">\s*<", "><", " ".join(match.group().split())), alone=True
            ),
            text,
        )
    text = QUOTED.sub(lambda match: hold(match.group()), text)
    breaks = _TAG_BREAK if style in ("tags", "xml") else _PROSE_BREAK
    pieces = []
    for raw in breaks.split(text):
        piece = _HELD.sub(lambda match: held[int(match.group(1))], raw)
        piece = " ".join(piece.split()).strip(_PIECE_TRIM).casefold()
        if piece:
            pieces.append(piece)
    return pieces


def is_reorder(candidate: str, prompt: str, style: str) -> bool:
    """Whether `candidate` is `prompt` with its pieces moved and nothing else -
    the variant that isolates order, so a grid row of anything more would lie."""
    if Counter(QUOTED.findall(candidate)) != Counter(QUOTED.findall(prompt)):
        return False
    if style == "json":
        # An object's keys have no order as values, so equal as values and
        # different as written is exactly a reorder; a list's order is content.
        try:
            theirs, ours = json.loads(candidate), json.loads(prompt)
        except (ValueError, RecursionError):
            return False
        return theirs == ours and json.dumps(theirs) != json.dumps(ours)
    theirs, ours = _pieces(candidate, style), _pieces(prompt, style)
    return Counter(theirs) == Counter(ours) and theirs != ours


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Where `quote` occurs in `text`, ignoring case and runs of whitespace.

    The model's quote is a claim; the span is found here or not at all, and
    what the page selects is the prompt's own characters, not the model's.
    """
    for candidate in (quote, quote.strip(_QUOTE_TRIM)):
        words = candidate.split()
        if not words:
            continue
        match = re.search(r"\s+".join(map(re.escape, words)), text, re.IGNORECASE)
        if match:
            return match.start(), match.end()
    return None


def _clip(value: Any) -> str:
    return value.strip()[:MAX_TEXT] if isinstance(value, str) else ""


# The answer's top-level keys (`response_schema`). An object with none of them
# is some other object - an issue, or braces in the thinking - not the answer.
_ANSWER_KEYS = ("parts", "issues", "rewrite", "reordered")
_THINK_END = "</think>"


def _answer_json(content: str) -> dict[str, Any]:
    """The answer object in the reply's content.

    llama-server splits the thinking off into `reasoning_content`; a template
    that left it inline ends it with </think>. So the answer is looked for
    after each </think>, the last first, and then in the whole content - never
    only after the last one: the answer quotes the prompt back, and a prompt
    pasted from another model's output can hold a </think> of its own. What
    follows one of those is a piece of the answer, and its first object - an
    issue - lacks the answer's keys, so it is passed over rather than read as
    an answer with nothing in it. The first failure is the one reported.
    """
    pieces = content.split(_THINK_END)
    failure: tuple[str, Exception | None] | None = None
    for i in reversed(range(len(pieces))):
        text = _THINK_END.join(pieces[i:])
        start = text.find("{")
        if start < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            failure = failure or (f"the critic's answer was not valid JSON: {exc}", exc)
            continue
        if isinstance(value, dict) and any(key in value for key in _ANSWER_KEYS):
            return value
        failure = failure or ("the critic's answer was not the JSON object it was asked for", None)
    message, cause = failure or ("the critic's answer held no JSON object", None)
    raise CriticError(message) from cause


def _issue(raw: Any, context: CriticContext, prompt: str) -> CritiqueIssue | None:
    if not isinstance(raw, Mapping):
        return None
    cite_id = str(raw.get("cite_id") or "").strip().upper()
    citation = context.citations.get(cite_id)
    problem = _clip(raw.get("problem"))
    if citation is None or not problem:
        return None
    kind = raw.get("kind") if raw.get("kind") in ISSUE_KINDS else "other"
    quote = start = end = None
    if isinstance(raw.get("quote"), str):
        span = locate(raw["quote"], prompt)
        if span is not None:
            start, end = span
            quote = prompt[start:end]
    return CritiqueIssue(
        kind=kind,
        problem=problem,
        fix=_clip(raw.get("fix")),
        cite_id=cite_id,
        cite=citation.text,
        url=citation.url,
        source=citation.source,
        quote=quote,
        start=start,
        end=end,
    )


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def _prompt_text(value: Any) -> str | None:
    """A whole prompt from the answer, or None. Over the request's own limit
    it is dropped rather than cut: a grid variant that stops mid-word is worse
    than no variant."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > MAX_TEXT:
        return None
    return text


def _flat(text: str) -> str:
    return " ".join(text.split()).casefold()


# A clause's end, which lettering does not run past.
_CLAUSE_END = re.compile(r"[,.;:!?\n，。；：！？]")


def _lettered(words: str, prompt: str, spans: list[tuple[int, int]]) -> bool:
    """Whether `words` are text the prompt asks to render: inside one of its
    `rendered_spans`, or starting in one and running on within its clause -
    the rules end unquoted lettering at a guess ("says fish and chips"), and
    the critic may read further. Never the prompt's description."""
    key = _flat(words.strip(_QUOTE_TRIM))
    if not key:
        return False
    # Whole words: "pen" is not in "OPEN".
    if any(
        f" {key} " in f" {_flat(prompt[start:end].strip(_QUOTE_TRIM))} " for start, end in spans
    ):
        return True
    at = locate(words, prompt)
    return at is not None and any(
        start <= at[0] < end and not _CLAUSE_END.search(prompt, end, at[1])
        for start, end in spans
    )


# What a part says when the prompt has no such part.
_NO_PART = frozenset(("none", "null", "n/a", "na", "nothing", "not specified", "unspecified", "-"))
_PART_BREAK = re.compile(r"[,;，；、\n]")


def _part(name: str, value: Any, prompt: str, spans: list[tuple[int, int]]) -> str | None:
    """A part of the prompt as the model named it, if it is the prompt's own.

    Every comma piece must be in the prompt (`locate`); text to render must
    also be lettering the prompt asks for. On the card the model put the
    whole prompt under text to render, "none" under camera, and lighting the
    prompt never mentioned.
    """
    text = _clip(value)
    if not text or text.strip(_QUOTE_TRIM).casefold() in _NO_PART:
        return None
    pieces = [piece for piece in _PART_BREAK.split(text) if piece.strip(_QUOTE_TRIM)]
    if not pieces or not all(locate(piece, prompt) for piece in pieces):
        return None
    if name == "text_to_render" and not all(_lettered(piece, prompt, spans) for piece in pieces):
        return None
    return text


# What an issue reads like when it restates a rule finding the critic was
# told of - which it did in nearly every answer on the card - keyed by the
# rule, with the name the page gives it. `guide.words` counts whenever the
# guide has a range: the rules measured the length, and the critic miscounts.
_RESTATES: dict[str, tuple[str, re.Pattern[str]]] = {
    "guide.words": (
        "word count",
        re.compile(
            r"(?i)\b\d+\s*(?:-|–|to)\s*\d+\s*words?\b|\b\d+\s*words?\b|\bword[- ]?count\b"
            r"|\btoo (?:short|long|brief)\b|\b(?:prompt|word) length\b|\blength (?:of|is)\b"
        ),
    ),
    "guide.negation": (
        "negation", re.compile(r"(?i)\bnegat|\babsence\b|\bwhat (?:not|you do not|you don't)\b"),
    ),
    "guide.quote_text": ("quotation marks", re.compile(r"(?i)\bquot")),
    "guide.order": ("tag order", re.compile(r"(?i)\border\b|\bfirst\b|\bbefore\b|\bafter\b")),
    "encoder.weighting": (
        "weighting", re.compile(r"(?i)\bweight|\bemphasis\b|\bparenthes|\bbracket"),
    ),
    "heuristic.quality_boilerplate": (
        "quality boilerplate", re.compile(r"(?i)\bquality (?:tag|term|word|boilerplate)|\bmasterpiece\b"),
    ),
}


# How far past a finding's words an issue's quote may reach and still be
# about those words: "with no people," for "no people".
_QUOTE_SLACK = 8


def _restated(issue: CritiqueIssue, context: CriticContext) -> str | None:
    text = f"{issue.problem} {issue.fix}"
    for rule, (name, pattern) in _RESTATES.items():
        told = rule in context.reported or (rule == "guide.words" and context.words_measured)
        if told and pattern.search(text):
            return name
    # Or it quotes the very words a finding points at - the card's critic
    # raised "no people" / "Replace with 'empty'..." beside the negation
    # finding on the same words, in words none of the patterns name.
    if issue.start is not None and issue.end is not None:
        for rule, start, end in context.reported_spans:
            if start - _QUOTE_SLACK <= issue.start and issue.end <= end + _QUOTE_SLACK:
                return _RESTATES[rule][0] if rule in _RESTATES else rule.split(".", 1)[-1]
    return None


# What an order issue asks to come first. Two in one answer that ask for
# different things - "Move subject to the beginning" beside "Place style
# description first", both seen on Krea's own example - cannot both be right.
# Read from the fix, the instruction ("Place the subject first, then the
# style"), and only then from a problem that says what is not first; a
# problem describing the prompt ("the style is placed before the subject")
# names parts in the prompt's order, not the one wanted.
_PART = (
    r"(subject|style|setting|location|environment|scene|lighting|light|camera|action"
    r"|context|image type|medium|mood|quality)"
)
_WANTS = (
    re.compile(rf"(?i)\b(?:start|begin|lead|open)\w*\s+with\s+(?:the\s+|a\s+|an\s+)?{_PART}\b"),
    re.compile(
        rf"(?i)\b{_PART}\b(?:\s+[\w'-]+){{0,3}}?\s+(?:first|early|before"
        r"|(?:at|to) the (?:beginning|start|front))\b"
    ),
)
_NOT_FIRST = re.compile(
    rf"(?i)\b{_PART}\b(?:\s+is)?\s+not\s+(?:first|front-loaded|placed first"
    r"|at the (?:beginning|start|front))\b"
)
# BFL's first step is "the image type and subject": an image type or medium
# first is the subject's place, not a rival to it.
_SAME_PART = {
    "location": "setting", "environment": "setting", "scene": "setting", "light": "lighting",
    "image type": "subject", "medium": "subject",
}


def _wants_first(issue: CritiqueIssue) -> str | None:
    if issue.kind != "order":
        return None
    # The prompt's own words, quoted in the fix, name no part.
    fix, problem = QUOTED.sub(" ", issue.fix), QUOTED.sub(" ", issue.problem)
    match = next(
        (found for pattern in _WANTS if (found := pattern.search(fix))), None
    ) or _NOT_FIRST.search(problem)
    if match is None:
        return None
    part = match.group(1).lower()
    return _SAME_PART.get(part, part)


# A rewrite's outer quotation marks, and the marks that close each opener.
_WRAPS = {'"': '"', "“": "”", "„": "“”", "«": "»", "「": "」", "『": "』", "'": "'", "‘": "’"}


def _unwrapped(text: str) -> str | None:
    """`text` without quotation marks around the whole of it, or None when it
    has none: the marks inside must still open before they close, so
    `"OPEN" sign, "CLOSED"` is two quotations and not one wrap, while the
    apostrophe in `'A dog's tail'` is no mark at all."""
    opener = text[:1]
    closers = _WRAPS.get(opener)
    if closers is None or len(text) < 3 or text[-1] not in closers:
        return None
    inner = text[1:-1]
    depth = 0
    for index, char in enumerate(inner):
        if char != opener and char not in closers:
            continue
        before = inner[index - 1] if index else " "
        after = inner[index + 1] if index + 1 < len(inner) else " "
        if before.isalnum() and after.isalnum():
            continue  # an apostrophe inside a word
        # A straight mark opens after a space, a curly one by its shape.
        opening = (before.isspace() or before in "([{") if opener in closers else char == opener
        depth += 1 if opening else -1
        if depth < 0:
            return None
    return inner.strip() if depth == 0 else None


# Words that carry no content of their own, for telling a rewrite's new
# content from its new wording.
_FILLER = frozenset(
    "a an the of with and or in on at to from by for into onto over under is are was "
    "were be been its their his her this that these those as it while which".split()
)
_CONTENT_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_CJK_TEXT = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
# New content words past which a rewrite is said to add content: a handful
# is rewording ("no people" -> "empty"), more is scenery and mood.
_ADDED_WORDS = 6


def _stem(word: str) -> str:
    word = word.casefold().replace("’", "'")
    return re.sub(r"(?:'s|ing|ed|es|s|ly)$", "", word) if len(word) > 4 else word


def _added_words(rewrite: str, prompt: str) -> list[str]:
    """The rewrite's content words the prompt has in no form."""
    # A closing single quote is not part of the word: 'Open Late'.
    have = {_stem(word.rstrip("'’")) for word in _CONTENT_WORD.findall(prompt)}
    added: list[str] = []
    for word in _CONTENT_WORD.findall(rewrite):
        word = word.rstrip("'’")
        key = word.casefold()
        if len(key) < 3 or key in _FILLER or _stem(word) in have or key in added:
            continue
        added.append(key)
    return added


def _check_rewrite(
    rewrite: str, prompt: str, spans: list[tuple[int, int]]
) -> tuple[str | None, list[PromptFinding], list[str]]:
    """The rewrite's quotation marks held to what the prompt asks to render.

    On FLUX a quotation is lettering, and the rules skip quoted text, so
    nothing after this would notice: four of 19 rewrites on the card were the
    whole prompt in quotes, one ended in the example prompts' "Black Forest
    Labs". A wrap comes off; lettering the prompt does not have drops the
    rewrite; the prompt's own description in quotes is flagged on the rewrite.
    """
    notes: list[str] = []
    inner = _unwrapped(rewrite)
    if inner is not None:
        notes.append(
            "The rewrite came back wrapped in quotation marks, which ask for lettering; "
            "they were taken off."
        )
        if not inner or _same(inner, prompt):
            return None, [], notes
        rewrite = inner
    findings: list[PromptFinding] = []
    for match in QUOTED.finditer(rewrite):
        words = match.group()[1:-1]
        if not words.strip() or _lettered(words, prompt, spans):
            continue
        if locate(words, prompt) is None:
            notes.append(
                f'The rewrite was left out: it puts "{_one_line(words, 60)}" in quotation '
                "marks - lettering the prompt does not ask for."
            )
            return None, [], notes
        findings.append(
            PromptFinding(
                rule="heuristic.rewrite_quote",
                severity="warn",
                message=(
                    f'"{_one_line(words, 60)}" is in quotation marks here but is not text '
                    "the prompt asks to render; quoted words are drawn as lettering."
                ),
                source="heuristic",
                start=match.start(),
                end=match.end(),
            )
        )
    # Told not to, the 4B critic still padded nearly every short prompt with
    # scenery and mood on the card: said, so the rewrite is read as new
    # content, and a comparison against it as testing more than wording.
    # Chinese and Japanese put no spaces between words, so they are not
    # counted this way.
    if not _CJK_TEXT.search(prompt):
        added = _added_words(rewrite, prompt)
        if len(added) >= _ADDED_WORDS:
            shown = ", ".join(added[:8]) + ("…" if len(added) > 8 else "")
            findings.append(
                PromptFinding(
                    rule="heuristic.rewrite_adds",
                    severity="info",
                    message=(
                        f"The rewrite adds {len(added)} words the prompt does not have "
                        f"({shown}): new content as well as new wording, so a comparison "
                        "against it tests both."
                    ),
                    source="heuristic",
                )
            )
    return rewrite, findings, notes


def validate(
    answer: str | Mapping[str, Any],
    context: CriticContext,
    prompt: str,
    *,
    model: str,
    seconds: float,
) -> PromptCritique:
    """The model's answer as a `PromptCritique`: every citation resolved,
    every quote and part found in the prompt, and what cannot stand left out
    and said so (`dropped`, `set_aside`). The rewrite is still the model's
    suggestion; the route lints it with layer 1."""
    data = _answer_json(answer) if isinstance(answer, str) else dict(answer)
    spans = rendered_spans(prompt)
    raw_parts = data.get("parts") if isinstance(data.get("parts"), Mapping) else {}
    parts = {name: _part(name, raw_parts.get(name), prompt, spans) for name in PARTS}

    issues: list[CritiqueIssue] = []
    dropped = 0
    set_aside: list[str] = []
    restated: list[str] = []
    raw_issues = data.get("issues") if isinstance(data.get("issues"), list) else []
    for raw in raw_issues[:MAX_ISSUES]:
        issue = _issue(raw, context, prompt)
        if issue is None:
            dropped += 1
        elif (rule := _restated(issue, context)) is not None:
            restated.append(rule)
        else:
            issues.append(issue)
    if restated:
        count = len(restated)
        set_aside.append(
            f"{count} issue{'' if count == 1 else 's'} left out: "
            f"{'it repeated' if count == 1 else 'they repeated'} the rule findings above "
            f"({', '.join(dict.fromkeys(restated))})."
        )
    wants = {index: want for index, issue in enumerate(issues) if (want := _wants_first(issue))}
    if len(set(wants.values())) > 1:
        issues = [issue for index, issue in enumerate(issues) if index not in wants]
        set_aside.append(
            f"{len(wants)} order issues left out: they asked for different things first "
            f"({', '.join(dict.fromkeys(wants.values()))})."
        )

    rewrite = _prompt_text(data.get("rewrite"))
    rewrite_findings: list[PromptFinding] = []
    if rewrite is not None and _same(rewrite, prompt):
        rewrite = None
    if rewrite is not None:
        rewrite, rewrite_findings, notes = _check_rewrite(rewrite, prompt, spans)
        set_aside += notes
    reordered = _prompt_text(data.get("reordered")) if context.ordered else None
    if reordered is not None and not is_reorder(reordered, prompt, context.style):
        if not _same(reordered, prompt):
            set_aside.append(
                "The critic's reordered version was left out: it changed words, not only "
                "their order; a reordering of your own can go under Compare."
            )
        reordered = None
    return PromptCritique(
        model=model,
        seconds=round(seconds, 2),
        parts=parts,
        issues=issues,
        rewrite=rewrite,
        reordered=reordered,
        rewrite_findings=rewrite_findings,
        dropped=dropped,
        set_aside=set_aside,
    )


def _content(response: Any) -> tuple[str, str | None]:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CriticError("llama-server returned no answer") from exc
    if not isinstance(content, str):
        raise CriticError("llama-server returned no answer text")
    return content, choice.get("finish_reason")


# --- The critic --------------------------------------------------------------


class PromptCritic:
    """Prepare once, critique on demand, never leave a server running."""

    def __init__(self) -> None:
        self._prepare_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._prepared: tuple[Path, Path] | None = None

    def supported(self) -> bool:
        """Switched on, and a server to run: one of the user's, or a pinned
        build this system's glibc can run. A missing library is not checked
        here - one `apt install` away, it is `prepare()`'s to name."""
        if not settings.critic:
            return False
        if settings.critic_server is not None:
            return True
        key = platform_key()
        return key is not None and _old_glibc(key) is None

    def ready(self) -> bool:
        """The server and the model are both on disk - no network, no hashing."""
        return self.supported() and self._server_ready() and self._model_ready()

    def _server_ready(self) -> bool:
        if settings.critic_server is not None:
            return settings.critic_server.is_file()
        key = platform_key()
        return key is not None and installed_server(key) is not None

    def _model_ready(self) -> bool:
        if settings.critic_model is not None:
            return settings.critic_model.is_file()
        return _model_cached() is not None

    def prepare(self) -> None:
        """Download, verify and install what is missing. Idempotent, and no GPU:
        the route runs it before taking the generation gate."""
        if not settings.critic:
            raise CriticUnavailable("the prompt critic is switched off (LOCAL_DIFFUSION_CRITIC)")
        with self._prepare_lock:
            self._prepared = (self._server_path(), self._model_path())

    def _server_path(self) -> Path:
        if settings.critic_server is not None:
            if not settings.critic_server.is_file():
                raise CriticUnavailable(
                    f"LOCAL_DIFFUSION_CRITIC_SERVER={settings.critic_server} is not a file"
                )
            return settings.critic_server
        key = platform_key()
        if key is None:
            raise CriticUnavailable(
                "llama.cpp publishes no pinned build for this platform here; "
                "set LOCAL_DIFFUSION_CRITIC_SERVER to a llama-server of your own"
            )
        # Ahead of both downloads: `prepare()` fetches the model after this.
        preflight(key)
        return install_server(key)

    def _model_path(self) -> Path:
        if settings.critic_model is not None:
            if not settings.critic_model.is_file():
                raise CriticUnavailable(
                    f"LOCAL_DIFFUSION_CRITIC_MODEL={settings.critic_model} is not a file"
                )
            return settings.critic_model
        return download_model()

    def _label(self, model: Path) -> str:
        return model.stem if settings.critic_model is not None else MODEL_LABEL

    def critique(
        self,
        model_id: str,
        request: GenerateRequest,
        applied: GenerateRequest,
        analysis: PromptAnalysis,
    ) -> PromptCritique:
        """Ask the critic about `request.prompt` on `model_id`. Blocking, and on
        the GPU: run it through `jobs.run_when_idle`, after `prepare()`.

        The rewrite's layer-1 lint is the route's, once the gate is open,
        since layer 1 can fetch a tokenizer; `rewrite_findings` here holds
        only `validate`'s own checks on it."""
        prepared = self._prepared
        if prepared is None or not all(path.is_file() for path in prepared):
            raise CriticUnavailable("the prompt critic is not prepared; prepare() fetches it first")
        server, model = prepared
        context = build_context(model_id, request, applied, analysis)
        check_fits(context, estimated=True)
        with self._run_lock:
            started = time.perf_counter()
            response = self._run(server, model, context, request.prompt)
            seconds = time.perf_counter() - started
        timings = response.get("timings") if isinstance(response, Mapping) else None
        if isinstance(timings, Mapping):
            log.info(
                "critique in %.1f s: %s prompt tokens, %s generated at %.0f tok/s",
                seconds, timings.get("prompt_n"), timings.get("predicted_n"),
                timings.get("predicted_per_second") or 0.0,
            )
        content, finish = _content(response)
        try:
            return validate(
                content, context, request.prompt, model=self._label(model), seconds=seconds
            )
        except CriticError as exc:
            if finish == "length":
                raise CriticError(
                    "the critic ran out of tokens before it finished its answer; its answer "
                    f"repeats the prompt {context.copies} times, so a shorter prompt leaves "
                    "it more room"
                ) from exc
            raise

    def _run(self, server: Path, model: Path, context: CriticContext, prompt: str) -> Any:
        api_key = secrets.token_urlsafe(32)
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        handle, name = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
        log_path = Path(name)
        try:
            with os.fdopen(handle, "wb") as log_file:
                try:
                    process = _spawn(server_command(server, model, port), log_file, api_key)
                except OSError as exc:
                    where = (
                        f"LOCAL_DIFFUSION_CRITIC_SERVER={server}"
                        if settings.critic_server is not None else str(server)
                    )
                    raise CriticError(
                        f"llama-server could not be started from {where}: {exc}"
                    ) from exc
                with self._process_lock:
                    self._process = process
                try:
                    _wait_until_healthy(process, base, log_path)
                    try:
                        status, _ = _request("GET", base + "/v1/models", None, None, 5.0)
                        if status != 401:
                            raise CriticError(
                                f"llama-server answered {status} without its API key; a "
                                "server any web page could reach is not one to send prompts to"
                            )
                        # Last before the key goes out: the gap after
                        # `_free_port` is on the command line for any local
                        # user to race, and a stranger can answer 401 too.
                        owned = _listener_is_ours(port, process.pids())
                        if owned is False:
                            raise CriticError(
                                f"what listens on port {port} is not the llama-server this app "
                                "started; neither the prompt nor its key was sent"
                            )
                        if owned is None:
                            log.debug("port %d's listener unchecked here; its key guards it", port)
                        body = request_body(context)
                        counted = _counted(base, api_key, body, context, prompt)
                        if counted is None:
                            log.warning(
                                "llama-server did not count the question's tokens; "
                                "sized by the estimate"
                            )
                        else:
                            check_fits(counted)
                            body["max_tokens"] = max_tokens(counted)
                        status, payload = _request(
                            "POST", base + "/v1/chat/completions", api_key, body, REQUEST_TIMEOUT
                        )
                    except (OSError, http.client.HTTPException) as exc:
                        raise CriticError(
                            f"llama-server did not answer: {exc}; {_tail(log_path)}"
                        ) from exc
                    if status != 200:
                        raise CriticError(
                            f"llama-server answered {status}: {_error_message(payload)}"
                        )
                    return payload
                finally:
                    # Always, and before the gate opens: an image job admitted
                    # next must not find 5.5 GiB of the card still taken.
                    try:
                        _stop(process)
                    finally:
                        with self._process_lock:
                            if self._process is process:
                                self._process = None
        finally:
            _remove_log(log_path)

    def unload(self) -> None:
        """Stop a server still running - the app is shutting down under it."""
        with self._process_lock:
            process = self._process
        if process is not None:
            _stop(process)


prompt_critic = PromptCritic()
