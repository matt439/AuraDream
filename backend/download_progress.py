"""Small, local-only samples for a load's browser status; never scan the cache."""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from . import model_info
from .model_cache_storage import free_bytes


def sample(model_id: str) -> dict:
    from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE

    return _sample(model_id, str(HF_HUB_CACHE), str(HF_HOME), int(time.monotonic() // 2), model_info.cache_revision())


@lru_cache(maxsize=64)
def _sample(model_id: str, cache_dir: str, converted_home: str, tick: int, revision: int) -> dict:
    """Stat required files and partial files in only the chosen repositories.

    Partial blob names cannot be mapped back to a particular model file, so
    their repo-scoped tally is explicitly approximate. Old snapshots and
    completed unrelated files do not join a known model's progress.
    """
    from huggingface_hub import try_to_load_from_cache

    references = model_info.download_references()
    recipe = references["models"].get(model_id)
    total = model_info.download_reference_bytes(model_id)
    downloaded = 0
    complete = bool(recipe)
    present = False
    seen: set[Path] = set()
    partial_repos: set[Path] = set()
    uncertain = False
    for choice in recipe["sources"] if recipe else []:
        source = references["sources"][choice["source"]]
        repo_dir = Path(cache_dir) / ("models--" + source["repo_id"].replace("/", "--"))
        partial_repos.add(repo_dir)
        names = choice["files"]
        if names is None:
            # We know this is a source, but no selected-file map proves its
            # total. Count its blob bytes as a cache tally, never a percent.
            uncertain = True
            complete = False
            try:
                for path in (repo_dir / "blobs").iterdir():
                    if path.is_file() and not path.name.endswith(".incomplete"):
                        resolved = path.resolve()
                        if resolved not in seen:
                            downloaded += path.stat().st_size
                            seen.add(resolved)
                            present = True
            except OSError:
                pass
            continue
        for name in names:
            if not model_info._relative_file(name):
                uncertain = True
                complete = False
                continue
            try:
                cached = try_to_load_from_cache(source["repo_id"], name, revision=source["revision"], cache_dir=cache_dir)
                if not isinstance(cached, str) or not Path(cached).is_file():
                    complete = False
                    continue
                path = Path(cached)
                # A mutable main pointing elsewhere invalidates the retained
                # total, rather than presenting an exact stale denominator.
                snapshot = path.parents[len(Path(name).parts) - 1]
                if snapshot.name != source["commit"]:
                    uncertain = True
                resolved = path.resolve()
                amount = path.stat().st_size
                if resolved not in seen:
                    downloaded += amount
                    seen.add(resolved)
                present = True
                if amount != source["files"][name]:
                    complete = False
            except (OSError, ValueError):
                complete = False
    for repo in partial_repos:
        try:
            for path in (repo / "blobs").glob("*.incomplete"):
                resolved = path.resolve()
                if path.is_file() and resolved not in seen:
                    amount = path.stat().st_size
                    downloaded += amount
                    seen.add(resolved)
                    present = present or amount > 0
        except OSError:
            pass
    if uncertain:
        total = None
    # A converted tree can be ready while its original download was removed.
    cached_state, _ = model_info._cached_state(model_id, cache_dir, converted_home, tick, revision)
    freshly_complete = complete and not uncertain and all(
        model_info._hub_source_state(references["sources"][choice["source"]], choice["files"], cache_dir) == "complete"
        for choice in recipe["sources"]
    ) if recipe else False
    state = "complete" if cached_state == "complete" or freshly_complete else "partial" if present else "unknown" if uncertain or recipe is None else "absent"
    return {
        "downloaded_bytes": min(downloaded, total) if total is not None else downloaded,
        "total_bytes": total, "cache_state": state,
        "cache_free_bytes": free_bytes(Path(cache_dir)), "approximate": True,
        "note": "Approximate local cache tally, not a network transfer meter. Partial files belong to the selected source repositories and can include another app's download. The total is the catalogue's retained download reference; it is unknown when that reference is incomplete or a cached mutable revision changed.",
    }
