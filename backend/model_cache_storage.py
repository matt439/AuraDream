"""On-demand, read-only cache accounting against retained download references.

No Hub call or model import. Unreferenced means only that the catalogue's
retained file lists do not name it; other apps/helpers can need it. The scan
also keeps a per-file index (`CacheIndex`) the answer never shows: what
`model_cache_delete.plan` reads to decide what a checkpoint's delete would
take, from the same scan the inventory's numbers come from.
"""
from __future__ import annotations

import os
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import model_info
from .legacy_names import hf_home_dir

_scan_lock = threading.Lock()


class ScanBusy(Exception):
    pass


@dataclass(frozen=True)
class IndexedFile:
    """One snapshot entry of a scanned Hub repository."""

    repo_id: str
    repo_type: str
    revision: str
    # The file's path inside the snapshot, posix.
    name: str
    # The first hop: the snapshot entry itself (a link on a symlink cache, the
    # file on a plain-copy one) and what that entry names directly (the
    # repository's `blobs/<etag>`, or the entry again). A shared Xet store is
    # a further hop, never followed here.
    entry: Path
    blob: Path
    identity: tuple
    size: int
    # Catalogue ids whose recipe selects it at this commit (an unknown file
    # map claims every file of the revision).
    owners: frozenset[str]
    # Its revision's `other_revision_of`.
    other_of: tuple[str, ...]


@dataclass(frozen=True)
class IndexedRepo:
    repo_id: str
    repo_type: str
    path: Path
    # commit -> its refs
    revisions: dict[str, frozenset[str]]


@dataclass
class CacheIndex:
    """What one scan saw, file by file, for the delete planner; never answered."""

    hub: Path
    converted_root: Path
    files: list[IndexedFile] = field(default_factory=list)
    repos: dict[tuple[str, str], IndexedRepo] = field(default_factory=dict)
    # repo id -> the catalogue ids whose recipe names it
    recipe_repos: dict[str, frozenset[str]] = field(default_factory=dict)
    # resolved identity -> logical bytes
    sizes: dict[tuple, int] = field(default_factory=dict)
    converted_files: dict[Path, int] = field(default_factory=dict)
    converted_targets: dict[str, Path] = field(default_factory=dict)
    # model id -> its answer row
    rows: dict[str, dict] = field(default_factory=dict)


def _catalogue_model_ids() -> list[str]:
    return list(dict.fromkeys(
        model for entry in model_info.CHECKPOINTS
        for model in [entry.repo, *(entry.lite_repo(fmt) for fmt, _ in model_info.FORMATS if entry.lite)]
    ))


def _lite_aliases(catalogue: list[str]) -> dict[str, str]:
    """lite-infer's renamed repository -> the lite id the app writes for it
    (`launch.written_id`), for written ids in `catalogue`. An alias has no
    recipe (`tools/model_downloads.py --check`), so its cached revisions are
    the written id's other revisions, never a row of their own."""
    return {
        alias: written for entry in model_info.CHECKPOINTS if entry.lite for fmt, _ in model_info.FORMATS
        if (written := entry.lite_repo(fmt)) in catalogue
        and (alias := entry.lite_alias(fmt)) is not None and alias != written
    }


def _row_names(model: str) -> dict:
    """The catalogue entry a storage row belongs to: its official id, its
    name and, for a Nunchaku lite id, the 4-bit format (None otherwise)."""
    entry = model_info.catalogue_entry(model)
    if entry is None:
        return {"entry": model, "label": None, "variant": None}
    return {"entry": entry.repo, "label": entry.name, "variant": entry.lite_format(model)}


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


def hub_cache_dir() -> Path:
    """The Hub cache this process reads (`HF_HUB_CACHE`), at call time; the
    fake server points it at its scratch cache."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def snapshot(cache_dir: Path | None = None) -> dict:
    """Inspect once; the worker owns the reservation even if its caller leaves."""
    if not _scan_lock.acquire(blocking=False):
        raise ScanBusy
    try:
        return _snapshot(cache_dir)
    finally:
        _scan_lock.release()


def _snapshot(cache_dir: Path | None = None) -> dict:
    return _scan(cache_dir)[0]


def _scan(cache_dir: Path | None = None) -> tuple[dict, CacheIndex]:
    """The inventory's answer and the index behind it; the caller holds
    `_scan_lock`."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    cache_dir = Path(cache_dir or hub_cache_dir())
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
    aliases = _lite_aliases(catalogue)
    converted_root = hf_home_dir("converted")
    index = CacheIndex(hub=cache_dir, converted_root=converted_root, sizes=sizes,
                       recipe_repos={repo: frozenset(models) for repo, models in repo_models.items()})
    repos = []
    for repo in sorted(info.repos if info else [], key=lambda item: (item.repo_type, item.repo_id)):
        index.repos[(repo.repo_type, repo.repo_id)] = IndexedRepo(
            repo.repo_id, repo.repo_type, Path(repo.repo_path),
            {revision.commit_hash: frozenset(revision.refs) for revision in repo.revisions})
        revisions = []
        # Beside a recorded revision, another one is an old download: unreferenced.
        other_revision = repo.repo_type == "model" and repo.repo_id in recorded and not any(
            revision.commit_hash in recorded[repo.repo_id] for revision in repo.revisions)
        # lite-infer's renamed repository: every revision is the written
        # lite id's other revision (no recipe records the alias's files).
        alias_of = aliases.get(repo.repo_id) if repo.repo_type == "model" else None
        for revision in sorted(repo.revisions, key=lambda item: item.commit_hash):
            revision_models: set[str] = set()
            revision_blobs: set[tuple] = set()
            revision_unreferenced: set[tuple] = set()
            if alias_of is not None:
                revision_other = [alias_of]
            else:
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
                index.files.append(IndexedFile(
                    repo.repo_id, repo.repo_type, revision.commit_hash, name,
                    Path(file.file_path), Path(file.blob_path), identity, file.size_on_disk,
                    frozenset(selected), tuple(revision_other)))
            revisions.append({
                "revision": revision.commit_hash, "refs": sorted(revision.refs),
                "cached_bytes": sum(sizes[key] for key in revision_blobs),
                "catalogue_models": sorted(revision_models),
                "other_revision_of": revision_other,
                "unreferenced_bytes": sum(sizes[key] for key in revision_unreferenced),
            })
        repos.append({"repo_id": repo.repo_id, "repo_type": repo.repo_type, "cached_bytes": repo.size_on_disk, "revisions": revisions})

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
        "model_id": model, **_row_names(model), "reference_complete": known[model],
        "cached_bytes": sum(sizes[key] for key in model_blobs[model]) if known[model] and not other_blobs[model] else None,
        "revision_bytes": sum(sizes[key] for key in model_blobs[model]),
        "other_revision_bytes": sum(sizes[key] for key in other_blobs[model]),
        "shared_bytes": sum(sizes[key] for key in model_blobs[model] if len(blob_models[key]) > 1),
        "exclusive_bytes": sum(sizes[key] for key in model_blobs[model] if len(blob_models[key]) == 1),
        "converted_bytes": converted_models.get(model, 0),
        "converted_directory": converted_targets[model].name if model in converted_targets else None,
    } for model in catalogue]
    index.converted_files = converted_files
    index.converted_targets = converted_targets
    index.rows = {row["model_id"]: row for row in models}
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
        "note": "Read-only point-in-time file-byte inventory, not disk allocation or a deletion plan. Model rows overlap where files are shared; do not add them. Unknown selected-file maps keep cached_bytes null, as does a catalogue repository cached only at commits no reference records: its bytes are other_revision_bytes, the checkpoint's, while an old revision beside a recorded one stays unreferenced. A renamed lite-infer repository's bytes are the other_revision_bytes of the lite id the app writes; entry, label and variant name the catalogue entry a row belongs to. Unreferenced means absent from the catalogue's retained download references, not unused: helpers and other apps may need it. Each revision's unreferenced_bytes is local to that revision and can share blobs with another; the overall unreferenced_bytes excludes every blob a catalogue selection references. Corrupted repositories can be omitted; directory metadata, Xet chunks and files outside the Hub and conversion caches are not included.",
    }, index
