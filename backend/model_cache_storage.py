"""On-demand, read-only cache accounting against retained download references.

No Hub call, model import or delete plan. Unreferenced means only that the
catalogue's retained file lists do not name it; other apps/helpers can need it.
"""
from __future__ import annotations

import os
import shutil
import threading
from collections import defaultdict
from pathlib import Path

from . import model_info
from .legacy_names import hf_home_dir

_scan_lock = threading.Lock()


class ScanBusy(Exception):
    pass


def _catalogue_model_ids() -> list[str]:
    return list(dict.fromkeys(
        model for entry in model_info.CHECKPOINTS
        for model in [entry.repo, *(entry.lite_repo(fmt) for fmt, _ in model_info.FORMATS if entry.lite)]
    ))


def free_bytes(path: Path) -> int | None:
    """Free space on the containing drive, even before a cache exists."""
    try:
        while not path.exists() and path != path.parent:
            path = path.parent
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _identity(path: Path) -> tuple:
    # Follow both snapshot-to-blob and repo-to-shared-store symlinks. This
    # measures file bytes, not allocated clusters or hard-link deduplication.
    return (str(path.resolve(strict=True)),)


def _tree_files(root: Path, warnings: list[str]) -> dict[Path, int]:
    """Logical file bytes inside one conversion tree, never follow directories."""
    result = {}
    if not root.exists():
        return result
    if root.is_symlink() or root.is_junction():
        warnings.append("A converted-cache directory link was skipped.")
        return result
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda _: warnings.append("Some converted-cache folders could not be read.")):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink() and not (Path(directory) / name).is_junction()]
        for name in files:
            path = Path(directory) / name
            try:
                if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
                    warnings.append("A converted-cache link outside its tree was skipped.")
                    continue
                result[path] = path.stat().st_size
            except OSError:
                warnings.append("Some converted-cache files changed or could not be read.")
    return result


def snapshot(cache_dir: Path | None = None) -> dict:
    """Inspect once; the worker owns the reservation even if its caller leaves."""
    if not _scan_lock.acquire(blocking=False):
        raise ScanBusy
    try:
        return _snapshot(cache_dir)
    finally:
        _scan_lock.release()


def _snapshot(cache_dir: Path | None = None) -> dict:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.errors import CacheNotFound

    cache_dir = Path(cache_dir or HF_HUB_CACHE)
    references = model_info.download_references()
    catalogue = _catalogue_model_ids()
    warnings: list[str] = []
    try:
        info = scan_cache_dir(cache_dir)
    except CacheNotFound:
        info = None
    warnings.extend("A repository was skipped because its cache is incomplete or unreadable." for _ in (info.warnings if info else []))

    # (repo, exact retained commit, relative file), with None meaning the
    # selected-file map is unknown. Keep that revision conservatively claimed.
    owners: dict[tuple[str, str, str | None], set[str]] = defaultdict(set)
    known: dict[str, bool] = {}
    recorded: dict[str, set[str]] = defaultdict(set)  # repo -> retained commits
    repo_models: dict[str, set[str]] = defaultdict(set)
    for model in catalogue:
        recipe = references["models"].get(model)
        if recipe is None:
            known[model] = False
            continue
        known[model] = True
        for choice in recipe["sources"]:
            source = references["sources"][choice["source"]]
            commit = source.get("commit") or source["revision"]
            names = choice["files"]
            if names is None:
                known[model] = False
            for name in names if names is not None else [None]:
                owners[(source["repo_id"], commit, name)].add(model)
            recorded[source["repo_id"]].add(commit)
            repo_models[source["repo_id"]].add(model)

    sizes: dict[tuple, int] = {}
    model_blobs: dict[str, set[tuple]] = defaultdict(set)
    blob_models: dict[tuple, set[str]] = defaultdict(set)
    # A catalogue repository cached only at commits no reference records (a
    # download from before the recorded commit, or after a later upstream
    # one): its files are that checkpoint's, which ones a load selects is
    # unknown, as Models' own cache column says (`model_info._hub_source_state`).
    other_blobs: dict[str, set[tuple]] = defaultdict(set)
    unreferenced_blobs: set[tuple] = set()
    repos = []
    for repo in sorted(info.repos if info else [], key=lambda item: (item.repo_type, item.repo_id)):
        revisions = []
        # Beside a recorded revision, another one is an old download: unreferenced.
        other_revision = repo.repo_type == "model" and repo.repo_id in recorded and not any(
            revision.commit_hash in recorded[repo.repo_id] for revision in repo.revisions)
        for revision in sorted(repo.revisions, key=lambda item: item.commit_hash):
            revision_models: set[str] = set()
            revision_blobs: set[tuple] = set()
            revision_unreferenced: set[tuple] = set()
            revision_other = sorted(repo_models[repo.repo_id]) if other_revision else []
            for file in revision.files:
                try:
                    identity = _identity(file.blob_path)
                except OSError:
                    warnings.append("Some Hub-cache files changed or could not be read during the scan.")
                    continue
                sizes[identity] = file.size_on_disk
                revision_blobs.add(identity)
                name = file.file_path.relative_to(revision.snapshot_path).as_posix()
                selected = set()
                if repo.repo_type == "model":
                    selected = owners.get((repo.repo_id, revision.commit_hash, name), set()) | owners.get((repo.repo_id, revision.commit_hash, None), set())
                if selected:
                    revision_models.update(selected)
                    blob_models[identity].update(selected)
                    for model in selected:
                        model_blobs[model].add(identity)
                elif revision_other:
                    for model in revision_other:
                        other_blobs[model].add(identity)
                else:
                    revision_unreferenced.add(identity)
                    unreferenced_blobs.add(identity)
            revisions.append({
                "revision": revision.commit_hash, "refs": sorted(revision.refs),
                "cached_bytes": sum(sizes[key] for key in revision_blobs),
                "catalogue_models": sorted(revision_models),
                "other_revision_of": revision_other,
                "unreferenced_bytes": sum(sizes[key] for key in revision_unreferenced),
            })
        repos.append({"repo_id": repo.repo_id, "repo_type": repo.repo_type, "cached_bytes": repo.size_on_disk, "revisions": revisions})

    converted_root = hf_home_dir("converted")
    converted_files = _tree_files(converted_root, warnings)
    converted_targets = {}
    for model in catalogue:
        recipe = references["models"].get(model)
        if recipe and recipe["kind"] == "converted":
            target = model_info._conversion_target(model)
            if target is not None:
                converted_targets[model] = target[0]
    converted_models: dict[str, int] = {}
    claimed_converted: set[Path] = set()
    for model, target in converted_targets.items():
        files = {path: size for path, size in converted_files.items() if path.is_relative_to(target)}
        converted_models[model] = sum(files.values())
        claimed_converted.update(files)

    models = [{
        "model_id": model, "reference_complete": known[model],
        "cached_bytes": sum(sizes[key] for key in model_blobs[model]) if known[model] and not other_blobs[model] else None,
        "revision_bytes": sum(sizes[key] for key in model_blobs[model]),
        "other_revision_bytes": sum(sizes[key] for key in other_blobs[model]),
        "shared_bytes": sum(sizes[key] for key in model_blobs[model] if len(blob_models[key]) > 1),
        "exclusive_bytes": sum(sizes[key] for key in model_blobs[model] if len(blob_models[key]) == 1),
        "converted_bytes": converted_models.get(model, 0),
        "converted_directory": converted_targets[model].name if model in converted_targets else None,
    } for model in catalogue]
    # A blob not selected in one revision may still serve another revision.
    unreferenced_only = unreferenced_blobs - set(blob_models) - {key for keys in other_blobs.values() for key in keys}
    return {
        "cache_dir": str(cache_dir), "free_bytes": free_bytes(cache_dir),
        "converted_dir": str(converted_root), "converted_free_bytes": free_bytes(converted_root),
        "hub_bytes": info.size_on_disk if info else 0,
        "incomplete_bytes": info.incomplete_size_on_disk if info else 0,
        "converted_bytes": sum(converted_files.values()),
        "converted_unreferenced_bytes": sum(size for path, size in converted_files.items() if path not in claimed_converted),
        "unreferenced_bytes": sum(sizes[key] for key in unreferenced_only),
        "store_only_bytes": max(0, (info.size_on_disk if info else 0) - sum(sizes.values())),
        "models": models, "repositories": repos,
        "warnings": sorted(set(warnings)),
        "note": "Read-only point-in-time file-byte inventory, not disk allocation or a deletion plan. Model rows overlap where files are shared; do not add them. Unknown selected-file maps keep cached_bytes null, as does a catalogue repository cached only at commits no reference records: its bytes are other_revision_bytes, the checkpoint's, while an old revision beside a recorded one stays unreferenced. Unreferenced means absent from the catalogue's retained download references, not unused: helpers and other apps may need it. Each revision's unreferenced_bytes is local to that revision and can share blobs with another; the overall unreferenced_bytes excludes every blob a catalogue selection references. Corrupted repositories can be omitted; directory metadata, Xet chunks and files outside the Hub and conversion caches are not included.",
    }
