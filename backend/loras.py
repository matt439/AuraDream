"""The LoRA adapter directory: what is in it, and the file behind a name.

One owner, because three callers need the same answer and each would otherwise
join a string onto a path of its own: the endpoint that lists the adapters for
the UI, the submit check that refuses a name this machine has not got, and the
backend that hands a file to `load_lora_weights`. A request carries a *name*
(`GenerateRequest._bare_name` refuses anything with a separator in it) and this
is the only place one becomes a path - resolved under the directory and then
checked to still be inside it, so a name that got past the schema cannot reach
another directory through a symlink or a Windows alias.
"""
from __future__ import annotations

from pathlib import Path

from .config import settings

# What `load_lora_weights` reads. `.bin`/`.pt` LoRAs exist and are pickles:
# loading one executes whatever it contains, which is not a thing to do to a
# file that merely turned up in a directory.
SUFFIXES = (".safetensors",)


def available() -> list[dict]:
    """Every adapter in the directory, by name and size, in name order.

    A missing directory is the ordinary state - most installs have no adapters
    - and reads as an empty list rather than an error.
    """
    directory = settings.loras_dir
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            try:
                size = path.stat().st_size
            except OSError:  # pragma: no cover - vanished between listing and stat
                continue
            found.append({"name": path.name, "bytes": size})
    return found


def path_for(name: str) -> Path:
    """The file for an adapter name, or a ValueError naming the directory."""
    directory = settings.loras_dir.resolve()
    path = (directory / name).resolve()
    if directory not in path.parents or path.suffix.lower() not in SUFFIXES or not path.is_file():
        raise ValueError(
            f"no LoRA named {name!r} in {settings.loras_dir}; "
            f"put a .safetensors adapter there (KREA_LORAS_DIR moves the directory)"
        )
    return path
