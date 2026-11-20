"""Bind an INP#13 run to its controller, Python stack, device and cached editor."""

from __future__ import annotations

import csv
import hashlib
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from backend import build_info
from backend.backends.loading import require_kontext_pin
from backend.instruction_edit import KONTEXT_V1, QUALIFIED_ENVELOPE
from backend.instruction_edit_preflight import CONFIG_PATHS

ROOT = Path(__file__).resolve().parents[1]


def _controller_files() -> dict[str, str]:
    paths = {ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / "tools/card_guard.py",
             ROOT / "tools/anima_refine_card.py", ROOT / "tools/anima_refine_pack.py"}
    tools = ROOT / "tools"
    for pattern in ("kontext*", "browser*.mjs", "card/kontext*"):
        for path in tools.glob(pattern):
            if path.is_file():
                paths.add(path)
            elif path.is_dir():
                paths.update(child for child in path.rglob("*")
                             if child.is_file() and "__pycache__" not in child.parts)
    return _file_hashes(paths, ROOT)


def _file_hashes(paths: set[Path], root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(paths):
        with path.open("rb") as stream:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def _gpu_zero() -> dict[str, str]:
    fields = ("index", "name", "uuid", "pci.bus_id", "driver_version", "memory.total")
    query = ",".join(fields)
    output = subprocess.run(
        ["nvidia-smi", "-i", "0", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(fields) or not all(rows[0]):
        raise ValueError("GPU 0 did not return a complete device and driver identity")
    return dict(zip(fields, rows[0], strict=True))


def capture_environment(mode: str) -> dict[str, Any]:
    """Read local evidence before the first server launch, without loading torch or querying the Hub."""
    if mode not in ("fake", "real"):
        raise ValueError("expected fake or real trial mode")
    packages: dict[str, str | None] = {}
    holds: list[str] = []
    for name in ("torch", "torchao", "diffusers", "transformers", "accelerate", "pillow", "huggingface-hub"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
            if mode == "real":
                holds.append(f"missing installed package {name}")
    result: dict[str, Any] = {
        "mode": mode, "git_head": build_info.read_commit(ROOT), "python": platform.python_version(),
        "platform": platform.platform(), "packages": packages, "controller_sha256": _controller_files(),
        "model_id": KONTEXT_V1.model_id, "revision": KONTEXT_V1.revision, "holds": holds,
    }
    if mode == "fake":
        return result

    from huggingface_hub.constants import HF_HUB_CACHE

    if sys.platform != QUALIFIED_ENVELOPE.platform:
        holds.append("real trial requires Windows")
    if packages["diffusers"] != QUALIFIED_ENVELOPE.diffusers_version or packages["pillow"] != "13.0.0":
        holds.append("real trial requires diffusers 0.42.1 and Pillow 13.0.0")
    try:
        device = _gpu_zero()
        result["gpu"] = device
        if device["index"] != "0" or device["name"] != "NVIDIA GeForce RTX 5080":
            holds.append("physical GPU 0 must be the RTX 5080")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0", device["uuid"]):
            holds.append("CUDA device selection does not identify physical GPU 0")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        holds.append(f"cannot identify GPU 0 and its driver: {exc}")
    cache_name = "models--" + KONTEXT_V1.model_id.replace("/", "--")
    snapshot = Path(HF_HUB_CACHE) / cache_name / "snapshots" / KONTEXT_V1.revision
    try:
        require_kontext_pin(str(snapshot))
        result["cached_config_sha256"] = _file_hashes({snapshot / name for name in CONFIG_PATHS}, snapshot)
    except (OSError, ValueError) as exc:
        holds.append(f"cached revision failed the config pin check: {exc}")
    return result
