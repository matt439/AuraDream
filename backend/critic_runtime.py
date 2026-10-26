"""Pinned critic runtime: platform checks, atomic installation and verified model downloads."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypedDict

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
# 5.5 GiB of VRAM over the desktop's. The research's 8.8-9.0 s and ~5.0 GiB
# were greedy decoding, which thinking mode is not run with.
class _ModelSource(TypedDict):
    repo_id: str
    filename: str
    revision: str


MODEL: _ModelSource = {
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
        answer = os.confstr("CS_GNU_LIBC_VERSION")
        if answer is None:
            return None
        name, version = answer.split()
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


# What `install_server` makes inside its work directory: the tree it
# publishes, and a scratch directory per archive (`_extract`).
_WORK_ENTRY = re.compile(r"tree|archive-\d+")


def _is_work(path: Path, target: Path) -> bool:
    """Whether `path` is exactly one of `target`'s install work directories.

    Its name is the target's, a hyphen and the eight characters `tempfile`
    adds; it is a real directory, not a link or junction; and it holds
    nothing but the `tree` and `archive-<n>` an install extracts into - never
    another build's tree, nor a user's copy named like one
    (`b11206-win-cuda-13.4-x64-backup01` is a name `tempfile` could make, and
    a copy of an install holds the server itself).
    """
    if not re.fullmatch(re.escape(target.name) + r"-[a-z0-9_]{8}", path.name):
        return False
    if path.name in {_target(key).name for key in BUILDS}:
        return False
    if path.is_symlink() or path.is_junction() or not path.is_dir():
        return False
    try:
        return all(_WORK_ENTRY.fullmatch(entry.name) for entry in path.iterdir())
    except OSError:
        return False


def _sweep_work(target: Path) -> None:
    """Remove what interrupted installs left beside `target`.

    Each extracts into a work directory (`_is_work`) that removes itself on
    the way out - but not after a hard kill, nor on Windows when the
    interpreter exits with a file in it still open on the prepare thread, and
    each such attempt left up to 744 MB. Only an install holding the lock
    extracts, so under it every one is stale. Best-effort: nothing here fails
    an install, and what cannot go yet is tried again on the next.
    """
    try:
        found = [path for path in target.parent.iterdir() if _is_work(path, target)]
    except OSError:
        return
    for stale in found:
        shutil.rmtree(stale, ignore_errors=True)
        if stale.exists():
            log.warning("could not remove an interrupted llama.cpp install yet: %s", stale)
        else:
            log.warning("removed an interrupted llama.cpp install: %s", stale)


def install_server(key: str) -> Path:
    """Download, verify and install the pinned build once; return the binary."""
    from huggingface_hub.utils._fixes import WeakFileLock

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
