"""Scratch Hugging Face caches for the checkpoint delete's tests
(`test_model_cache_delete.py`, `test_api_model_storage_delete.py`).

`scratch_caches` points `HF_HOME` and `HF_HUB_CACHE` at a scratch folder for
the whole process (the environment and `huggingface_hub.constants`, which
`legacy_names.hf_home_dir`, `single_file.cache_root` and the scan read at
call time), so the real loaders' converted targets, locks and work
directories resolve there too, never to this machine's cache. Beside it a
second scratch "real" home holds a sentinel tree no plan may name or change.
The recipes are the real `backend/model-downloads.json`: the files planted
are the catalogue's own names at its recorded commits, a few bytes each.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from backend import model_info

LAYOUTS = ["copy", "symlink"]


def symlinks_work(where: Path) -> bool:
    probe, target = where / "probe-link", where / "probe-target"
    target.write_bytes(b"")
    try:
        os.symlink(target, probe)
    except (OSError, NotImplementedError):
        return False
    finally:
        target.unlink()
    probe.unlink()
    return True


def link_folder(link: Path, target: Path) -> None:
    """A directory link: a junction on Windows (no privilege needed), a
    symlink elsewhere."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def tree_state(root: Path) -> dict[str, tuple]:
    """Every entry under `root`, byte for byte: a file's bytes and mtime, a
    link's target, a folder."""
    state: dict[str, tuple] = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        here = Path(directory)
        for name in [*dirs, *files]:
            path = here / name
            key = path.relative_to(root).as_posix()
            if path.is_symlink() or path.is_junction():
                state[key] = ("link", os.readlink(path))
            elif path.is_dir():
                state[key] = ("dir",)
            else:
                state[key] = ("file", path.read_bytes(), os.lstat(path).st_mtime_ns)
        dirs[:] = [name for name in dirs if not ((here / name).is_symlink() or (here / name).is_junction())]
    return state


class Caches:
    def __init__(self, root: Path, layout: str, real_home: str) -> None:
        self.layout = layout
        self.home = root / "hf-home"
        self.hub = self.home / "hub"
        self.converted = self.home / "auradream" / "converted"
        self.real_home = Path(real_home)
        self.sentinel = root / "real-hf-home"
        self.hub.mkdir(parents=True)
        self.converted.mkdir(parents=True)

    # ----------------------------------------------------------------- Hub

    def folder(self, repo: str) -> Path:
        return self.hub / ("models--" + repo.replace("/", "--"))

    def put(self, repo: str, commit: str, name: str, content: bytes | None = None, *, hub: Path | None = None) -> Path:
        """One snapshot file, a plain copy or a link to its blob by layout."""
        folder = (hub or self.hub) / ("models--" + repo.replace("/", "--"))
        content = content if content is not None else f"{repo}:{name}".encode()
        entry = folder / "snapshots" / commit / name
        entry.parent.mkdir(parents=True, exist_ok=True)
        if self.layout == "copy":
            entry.write_bytes(content)
        else:
            blob = folder / "blobs" / hashlib.sha256(content).hexdigest()
            blob.parent.mkdir(parents=True, exist_ok=True)
            if not blob.exists():
                blob.write_bytes(content)
            if os.path.lexists(entry):  # a file two recipes select, planted twice
                entry.unlink()
            os.symlink(os.path.relpath(blob, entry.parent), entry)
        return entry

    def ref(self, repo: str, commit: str, name: str = "main") -> None:
        ref = self.folder(repo) / "refs" / name
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(commit)

    def recipe(self, model: str) -> list[tuple[str, str, list[str]]]:
        """(repository, commit, selected files) for each of `model`'s sources."""
        references = model_info.download_references()
        found = []
        for choice in references["models"][model]["sources"]:
            source = references["sources"][choice["source"]]
            found.append((source["repo_id"], source.get("commit") or source["revision"], list(choice["files"] or [])))
        return found

    def plant(self, model: str, *, only: str | None = None) -> None:
        """Every selected file of `model`'s recipe (or of its source in
        repository `only`) at its recorded commit."""
        for repo, commit, names in self.recipe(model):
            if only is not None and repo != only:
                continue
            for name in names:
                self.put(repo, commit, name)

    def own_file(self, model: str) -> tuple[str, str, str]:
        """The first selected file of `model`'s first source: its own."""
        repo, commit, names = self.recipe(model)[0]
        return repo, commit, names[0]

    # ----------------------------------------------------------- converted

    def convert(self, model: str, *, extra: bytes = b"weights") -> Path:
        """`model`'s converted tree where the real loader resolves it."""
        conversion = model_info._conversion_target(model)
        assert conversion is not None, model
        target, marker, recipe = conversion
        assert target.parent == self.converted, (target, self.converted)
        (target / "unet").mkdir(parents=True)
        (target / "model_index.json").write_text("{}")
        (target / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(extra)
        marker.write_text(recipe, encoding="utf-8")
        return target

    def older(self, name: str, marker: str | None) -> Path:
        tree = self.converted / name
        tree.mkdir(parents=True)
        (tree / "model_index.json").write_text("{}")
        if marker is not None:
            from backend.backends.single_file import CONVERSION_MARKER

            (tree / CONVERSION_MARKER).write_text(marker, encoding="utf-8")
        return tree

    # ------------------------------------------------------------ the rest

    def plant_sentinel(self) -> None:
        """A second "real" home with the same shapes: what a plan resolving
        through the real HF_HOME would name. It must never change."""
        for repo in ("krea/Krea-2-Turbo", "circlestone-labs/Anima", "stabilityai/stable-diffusion-xl-base-1.0"):
            self.put(repo, "9" * 40, "model_index.json", b"sentinel", hub=self.sentinel / "hub")
        tree = self.sentinel / "auradream" / "converted" / "anima-turbo-v1.1-sentinel"
        tree.mkdir(parents=True)
        (tree / "model_index.json").write_bytes(b"sentinel")

    def state(self) -> dict[str, tuple]:
        return {**{f"home/{key}": value for key, value in tree_state(self.home).items()},
                **{f"real/{key}": value for key, value in tree_state(self.sentinel).items()}}


def scratch_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str) -> Caches:
    """Point the process at scratch caches and plant the sentinel home."""
    import huggingface_hub.constants as constants

    if layout == "symlink" and not symlinks_work(tmp_path):
        pytest.skip("this platform will not create symlinks for the test (Windows without the privilege)")
    caches = Caches(tmp_path, layout, constants.HF_HOME)
    monkeypatch.setenv("HF_HOME", str(caches.home))
    monkeypatch.setenv("HF_HUB_CACHE", str(caches.hub))
    monkeypatch.setattr(constants, "HF_HOME", str(caches.home))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(caches.hub))
    caches.plant_sentinel()
    # No fetch another test left running counts here.
    from backend import prompt_encoding

    monkeypatch.setattr(prompt_encoding, "_running", set())
    return caches
