"""Deleting one whole catalogue checkpoint from this server's caches: the
planner, read-only (`GEN#72` stage 1), and the delete's worker (stages 2 and
3), which `POST /api/models/storage/delete` runs on a daemon thread.

docs/maintainer/plan-model-delete.md is the design. The unit is one catalogue
entry and its group: the entry's own id, its Nunchaku lite ids and any
cached lite-infer alias of them. `plan()` reads the inventory's own scan
(`model_cache_storage._scan` and its per-file index) under the same scan
reservation and answers, without changing anything on disk:

- a repository the group alone names (no other entry's recipe does): the
  whole `models--org--name` folder - every revision, unselected files,
  `refs/`, `.no_exist/` - and each lite repository the same way;
- a repository other entries' recipes also name: file by file (stage 4),
  the group's selected files at its recorded commits, a file another entry
  selects too kept while that sharer is present (the last-holder rule) and
  taken with the last holder otherwise; a revision the delete empties takes
  its refs and `.no_exist/<commit>` with it;
- the entry's converted tree and its work directories, and older converted
  trees whose marker names the entry's own source repository and file.

What the server cannot prove safe is `held` (`refusal_codes.MODEL_DELETE_HELD`):
a shared Xet store, a link out of its repository, an unrecorded revision of a
repository several entries name, an older converted tree two entries' source
repository could own, a path outside the configured roots. Every path is
checked inside the Hub cache or the converted root before anything is
planned on it, and answers name repositories, revisions and names relative
to those roots only, never an absolute path.

`delete()` re-plans under the same reservation, refuses a different pin, and
renames every part before it removes any: each converted tree under its
loaders' own lock into exactly `single_file._is_work`'s shape
(`<target>-<8>/tree`), so it stops being a hit at once, then each
repository the group owns outright (`HUB_DELETE`, stage 3) out of the Hub
cache into its sibling trash (`model_cache_storage.trash_dir`,
`<8 hex>/<folder>`). The files inside shared repositories (stage 4) are
checked as planned next, then unlinked: each emptied revision's refs first,
then its snapshot entries, then the blobs no surviving entry resolves to,
then emptied folders bottom-up and the revision's `.no_exist/<commit>`. A
refused rename, a check that fails or a first unlink that fails puts every
rename back: nothing changed. After the first unlink a failure stops the
file-level step at once, and what stays is `remaining`: files missing from a
pinned revision, never a ref to a missing snapshot. Then each target's work
directories and each trash folder are removed; what an open file kept is
swept by the next load or delete and named by the inventory. A refusal
before anything changed is `DeleteRefused` with its code; a failure after
something changed is the outcome's `remaining`.
"""
from __future__ import annotations

import errno
import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import stat
import string
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import keep_awake, model_cache_storage, model_info
from .refusal_codes import (
    MODEL_CACHE_CHANGED,
    MODEL_DELETE_BUSY,
    MODEL_DELETE_HELD,
    MODEL_DOWNLOADING,
    MODEL_FILES_IN_USE,
    MODEL_SELECTED,
)

log = logging.getLogger(__name__)

# Deleting files inside a repository other entries' recipes also name, file
# by file, is the plan's stage 4, built: the parts it takes carry
# `shared_repository: true`. Off, a plan with such a removal is held.
SHARED_REPOSITORY_DELETE = True
# huggingface_hub 1.33's cache-wide shared Xet store marks `<hub>/blobs`
# with this file: its payloads serve several repositories (held, never planned).
SHARED_STORE_MARKER = ".huggingface-shared-blobs"
# An older converted tree's marker is read only this far.
_MARKER_BYTES = 64 * 1024
# Removing a repository the group owns outright from the Hub cache (the
# trash rename) is the plan's stage 3, built: off, `delete()` refuses a plan
# with a Hub or lite part, with nothing changed. A removal inside a shared
# repository is held by the planner while `SHARED_REPOSITORY_DELETE` is off.
HUB_DELETE = True
# How long a delete waits for a converted tree's lock. A conversion holds it
# for minutes and a load only while it sweeps and checks, so a wait past this
# is one running: `model_delete_busy`.
LOCK_TIMEOUT = 2.0
# ERNIE's prompt rewriter (Expand) inside its own repository's snapshot:
# `backends.loading.PROMPT_EXPANDER_WEIGHTS`, which imports torch, so held
# equal by tests/test_model_cache_delete_hub.py. A plan taking it says so.
PROMPT_EXPANDER_WEIGHTS = "pe/model.safetensors"
# What a work directory's suffix is drawn from: `_is_work` accepts these.
_WORK_ALPHABET = string.ascii_lowercase + string.digits
# The keep-awake hold the worker takes for its whole run.
DELETING = "deleting a checkpoint"


class NotOnDisk(LookupError):
    """No catalogue entry by that id, or nothing of it is cached: one answer."""


@dataclass(frozen=True)
class PlanContext:
    """The server's state a plan reads, by catalogue entry id (`entry.repo`):
    the selected checkpoint, the loaded one, the one saved to load on the
    next start, and Check's tokenizer fetches still in flight
    (`prompt_encoding.pending_sources`). Each counts as a present sharer.
    An id outside the catalogue stays itself (`entry_of`): `Plan.names`
    reads it by its repository."""

    selected: str | None = None
    loaded: str | None = None
    saved: str | None = None
    pending: frozenset[Any] = frozenset()

    def present(self) -> set[str]:
        return {entry for entry in (self.selected, self.loaded, self.saved) if entry is not None}


@dataclass(frozen=True)
class Removal:
    """One file or link a delete would remove; `path` is never answered."""

    kind: str
    root: str  # "hub" or "converted"
    relative: str
    path: Path
    size: int
    mtime_ns: int


@dataclass
class Part:
    kind: str  # "hub", "lite", "converted", "converted-old"
    repo: str | None = None
    revision: str | None = None
    name: str | None = None
    whole: bool = False
    shared_repository: bool = False
    bytes: int = 0
    files: int = 0
    # A converted part's current target, never answered: its lock is the one
    # the loaders take, and its work shape (`<target>-<8>/tree`) the one a
    # removal renames into - an older tree's too, so any leftover is swept
    # by the next load or delete of the entry.
    target: Path | None = None
    # What a delete would act on first: whole folders (a repository, a
    # converted tree, a work directory) or, inside a shared repository, the
    # snapshot entries and the blobs no surviving entry resolves to.
    units: list[Path] = field(default_factory=list)
    # Inside a shared repository (stage 4), the same paths by role: the
    # revision's snapshot entries, the repository blobs no surviving entry
    # resolves to and, when the delete empties the revision (`emptied`),
    # the refs naming it, unlinked first so it is never a ref to a missing
    # snapshot.
    entries: list[Path] = field(default_factory=list)
    blobs: list[Path] = field(default_factory=list)
    refs: list[Path] = field(default_factory=list)
    emptied: bool = False

    def answer(self) -> dict:
        answer: dict = {"kind": self.kind}
        if self.repo is not None:
            answer["repo"] = self.repo
        if self.revision is not None:
            answer["revision"] = self.revision
        if self.name is not None:
            answer["name"] = self.name
        answer.update(whole=self.whole, shared_repository=self.shared_repository,
                      bytes=self.bytes, files=self.files)
        return answer


@dataclass
class Plan:
    entry: str
    label: str
    # The group's catalogue ids (the entry's own, its lite ids): the gallery
    # records images under these.
    model_ids: tuple[str, ...]
    context: PlanContext
    parts: list[Part] = field(default_factory=list)
    kept_shared: list[dict] = field(default_factory=list)
    also_taken_shared: list[dict] = field(default_factory=list)
    held: dict | None = None
    downloading: list[dict] = field(default_factory=list)
    # Every repository the group's recipes name (shared ones included) and
    # the group's lite aliases: Check's fetch from any of them holds a delete.
    repositories: frozenset[str] = frozenset()
    removals: list[Removal] = field(default_factory=list)
    pin: str = ""
    # The roots the plan's scan read: the delete re-asserts them against the
    # configured ones before its first rename.
    hub: Path | None = None
    converted_root: Path | None = None

    @property
    def frees_bytes(self) -> int:
        return sum(part.bytes for part in self.parts)

    @property
    def removes_expander(self) -> bool:
        """Whether the delete takes a cached prompt rewriter (ERNIE's Expand)
        with the entry's own repository."""
        return any(removal.root == "hub" and removal.relative.split("/")[1:2] == ["snapshots"]
                   and "/".join(removal.relative.split("/")[3:]) == PROMPT_EXPANDER_WEIGHTS
                   for removal in self.removals)

    def fetching(self, sources: Iterable[Any]) -> bool:
        """Whether a tokenizer fetch in flight reads a repository or a
        converted tree this plan touches."""
        return any(getattr(source, "repo", None) in self.repositories
                   or getattr(source, "tree", None) in self.model_ids for source in sources)

    def names(self, model_id: str | None) -> bool:
        """Whether a `PlanContext` id (selected, loaded, saved) is what this
        plan would delete: its own entry, or an id outside the catalogue
        whose repository the plan names. A selected `Tongyi-MAI/Z-Image` is
        Z-Anime's recipe repository, owned and taken whole; a case variant
        of a catalogue id (`AURADREAM_MODEL_ID=black-forest-labs/flux.1-krea-dev`)
        is no catalogue entry but the same folder on NTFS, so the comparison
        ignores case, as the Hub's ids do. Another catalogue entry is never
        this plan's: a sharer's files are kept by the last-holder rule."""
        if model_id is None:
            return False
        if model_id == self.entry:
            return True
        if model_info.catalogue_entry(model_id) is not None:
            return False
        repo = model_id.split("@", 1)[0].casefold()
        return any(repo == named.casefold() for named in self.repositories)

    def answer(self) -> dict:
        return {
            "entry": self.entry, "label": self.label,
            "parts": [part.answer() for part in self.parts],
            "kept_shared": self.kept_shared, "also_taken_shared": self.also_taken_shared,
            "frees_bytes": self.frees_bytes, "held": self.held, "downloading": self.downloading,
            "removes_expander": self.removes_expander,
            "selected": self.names(self.context.selected),
            "loaded": self.names(self.context.loaded),
            "load_on_startup": self.names(self.context.saved),
            "pin": self.pin,
        }


def entry_of(model_id: str) -> str:
    """The catalogue entry id a model id belongs to (a lite id or alias to
    its entry's), or the id itself where no entry names it."""
    entry = model_info.catalogue_entry(model_id)
    return entry.repo if entry is not None else model_id


def _label(entry_id: str) -> str:
    entry = model_info.catalogue_entry(entry_id)
    return entry.name if entry is not None else entry_id


def _repo_folder(hub: Path, repo_id: str) -> Path:
    return hub / ("models--" + repo_id.replace("/", "--"))


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _lexically_inside(path: Path, root: Path) -> bool:
    """Inside `root` by name alone: never touches a path that is not. Paths
    on two drives (Windows' `commonpath` raises) are not inside."""
    path_text = os.path.normcase(os.path.abspath(path))
    root_text = os.path.normcase(os.path.abspath(root))
    try:
        return path_text != root_text and os.path.commonpath([path_text, root_text]) == root_text
    except ValueError:
        return False


def _inside(path: Path, root: Path) -> bool:
    """`path` itself (a link is removed, never what it names) lies under
    `root`, by name and through every real directory above it."""
    if not _lexically_inside(path, root):
        return False
    try:
        parent = path.parent.resolve(strict=True)
        real_root = root.resolve(strict=True)
    except OSError:
        return False
    return parent == real_root or parent.is_relative_to(real_root)


class _Collector:
    def __init__(self, hub: Path, converted: Path) -> None:
        self.roots = {"hub": hub, "converted": converted}
        self.removals: list[Removal] = []
        self.held: list[tuple[int, str]] = []
        # Whether a hold was found in something of the entry's on disk. A
        # hold that is not (a converted target outside the root, never looked
        # at) leaves an entry with nothing cached the plan's one 404.
        self.found = False

    def hold(self, priority: int, detail: str, *, on_disk: bool = True) -> None:
        self.held.append((priority, detail))
        self.found = self.found or on_disk

    def item(self, kind: str, root: str, path: Path) -> Removal | None:
        """One file or link, measured without following it."""
        try:
            info = os.lstat(path)
        except OSError:
            self.hold(3, "Some of its files changed or could not be read; refresh and try again.")
            return None
        size = info.st_size if stat.S_ISREG(info.st_mode) else 0
        relative = Path(os.path.relpath(path, self.roots[root])).as_posix()
        removal = Removal(kind, root, relative, path, size, info.st_mtime_ns)
        self.removals.append(removal)
        return removal

    def tree(self, kind: str, root: str, unit: Path) -> tuple[int, list[Removal]]:
        """Every file and link under one folder a delete would remove whole:
        (logical bytes, the items). A link whose target leaves the folder is
        held: what it names is someone else's."""
        found: list[Removal] = []
        failed: list[OSError] = []
        real_unit = unit.resolve()
        for directory, dirs, files in os.walk(unit, followlinks=False, onerror=failed.append):
            here = Path(directory)
            links = [name for name in dirs if _is_link(here / name)]
            dirs[:] = [name for name in dirs if name not in links]
            for name in [*files, *links]:
                path = here / name
                if _is_link(path):
                    try:
                        target = Path(os.path.realpath(path))
                    except OSError:
                        target = None
                    if target is None or not (target == real_unit or target.is_relative_to(real_unit)):
                        self.hold(0, "A link inside its files points outside them, as a shared Xet store's "
                                     "do; deleting it is not built yet.")
                        continue
                removal = self.item(kind, root, path)
                if removal is not None:
                    found.append(removal)
        if failed:
            self.hold(3, "Some of its folders could not be read; refresh and try again.")
        return sum(removal.size for removal in found), found


def plan(model_id: str, context: PlanContext | None = None, cache_dir: Path | None = None) -> Plan:
    """The plan for `model_id`'s catalogue entry, from one scan under the
    scan reservation. Raises `NotOnDisk` for an id no entry names or an
    entry with nothing cached, `model_cache_storage.ScanBusy` while another
    scan (or a delete) holds the reservation."""
    entry = model_info.catalogue_entry(model_id)
    if entry is None or entry.repo != model_id:
        # Exact entry ids only: a lite id is its entry's sub-line, not a unit.
        raise NotOnDisk(model_id)
    if not model_cache_storage._scan_lock.acquire(blocking=False):
        raise model_cache_storage.ScanBusy
    try:
        return _plan(entry.repo, context or PlanContext(), cache_dir)
    finally:
        model_cache_storage._scan_lock.release()


def _plan(entry_id: str, context: PlanContext, cache_dir: Path | None,
          index: model_cache_storage.CacheIndex | None = None) -> Plan:
    """`plan()`'s work, on `index` (scanned now without one); the caller
    holds `model_cache_storage._scan_lock`."""
    from .backends.single_file import CONVERSION_MARKER, _is_work

    entry = model_info.catalogue_entry(entry_id)
    assert entry is not None
    if index is None:
        _, index = model_cache_storage._scan(cache_dir)
    hub, converted_root = index.hub, index.converted_root
    lite_ids = [repo for fmt, _ in model_info.FORMATS if entry.lite and (repo := entry.lite_repo(fmt))]
    aliases = {alias for fmt, _ in model_info.FORMATS
               if entry.lite and (alias := entry.lite_alias(fmt)) and alias not in lite_ids}
    group = {entry.repo, *lite_ids}
    result = Plan(entry.repo, entry.name, tuple(sorted(group)), context)
    result.hub, result.converted_root = hub, converted_root
    collect = _Collector(hub, converted_root)

    named = {repo for repo, models in index.recipe_repos.items() if models & group}
    result.repositories = frozenset(named | aliases | set(lite_ids))
    owned = {repo for repo in result.repositories if not index.recipe_repos.get(repo, frozenset()) - group}
    shared = result.repositories - owned

    touched: set[str] = set()
    # Repositories the group alone names: whole.
    for repo in sorted(owned):
        folder = _repo_folder(hub, repo)
        if not os.path.lexists(folder):
            continue
        if not _inside(folder, hub) or _is_link(folder) or not folder.is_dir():
            collect.hold(1, f"{repo}'s cache folder is a link or lies outside the Hub cache; "
                            "deleting it is not built.")
            continue
        kind = "lite" if entry.lite_format(repo) is not None else "hub"
        size, items = collect.tree(kind, "hub", folder)
        snapshot_entries = sum(1 for item in items if item.relative.split("/")[1:2] == ["snapshots"])
        result.parts.append(Part(kind, repo=repo, whole=True, bytes=size, files=snapshot_entries, units=[folder]))
        touched.add(repo)

    # Repositories other entries' recipes name too: file by file.
    present = context.present() | _present_entries(index)
    pending_repos = {getattr(source, "repo", None) for source in context.pending}
    taken: dict[Path, tuple[model_cache_storage.IndexedFile, frozenset[str]]] = {}
    kept: dict[Path, tuple[model_cache_storage.IndexedFile, frozenset[str]]] = {}
    for file in index.files:
        if file.repo_type != "model" or file.repo_id not in shared:
            continue
        if file.owners:
            owners = {entry_of(model) for model in file.owners}
            if entry.repo not in owners:
                continue  # another entry's file in a repository both name
            sharers = frozenset(owners - {entry.repo})
            if sharers and (sharers & present or file.repo_id in pending_repos):
                # Named by the sharers that keep it (all of them for a fetch).
                kept[file.entry] = (file, frozenset(sharers & present) or sharers)
            else:
                taken[file.entry] = (file, sharers)
        elif file.other_of:
            others = {entry_of(model) for model in file.other_of}
            if entry.repo in others and others != {entry.repo}:
                collect.hold(2, f"{file.repo_id} is cached at a revision no download record names, which "
                                f"{', '.join(sorted(_label(other) for other in others - {entry.repo}))} "
                                "could also own; deleting it is not built.")
    survivors = [file for file in index.files if file.entry not in taken]
    surviving_blobs = {file.blob for file in survivors}
    counted: set[Path] = set()
    parts: dict[tuple[str, str], Part] = {}
    shares: dict[frozenset[str], list[int]] = defaultdict(list)
    for path, (file, sharers) in sorted(taken.items(), key=lambda item: str(item[0])):
        folder = _repo_folder(hub, file.repo_id)
        if _is_link(folder) or not _inside(folder, hub):
            collect.hold(1, f"{file.repo_id}'s cache folder is a link or lies outside the Hub cache; "
                            "deleting from it is not built.")
            continue
        if not (_inside(file.entry, folder) and _inside(file.blob, folder)) or (
                file.blob != file.entry and _is_link(file.blob)):
            collect.hold(0, f"A file of {file.repo_id} resolves outside its repository, as a shared Xet "
                            "store's do; deleting it is not built yet.")
            continue
        part = parts.get((file.repo_id, file.revision))
        if part is None:
            kind = "lite" if entry.lite_format(file.repo_id) is not None else "hub"
            part = parts[(file.repo_id, file.revision)] = Part(
                kind, repo=file.repo_id, revision=file.revision, shared_repository=True)
        part.files += 1
        part.units.append(path)
        part.entries.append(path)
        collect.item(part.kind, "hub", path)
        freed = 0
        if file.blob != file.entry and file.blob not in surviving_blobs and file.blob not in counted:
            part.units.append(file.blob)
            part.blobs.append(file.blob)
            collect.item(part.kind, "hub", file.blob)
        if file.blob not in surviving_blobs and file.blob not in counted:
            freed = file.size
            counted.add(file.blob)
        part.bytes += freed
        if sharers:
            shares[sharers].append(freed)
        touched.add(file.repo_id)
    # A revision the delete empties loses its refs first (they are removals
    # the pin covers): detached, never a ref to a missing snapshot, which the
    # scan calls a corrupted repository.
    for (repo, revision), part in parts.items():
        if any(file.repo_type == "model" and file.repo_id == repo and file.revision == revision
               and file.entry not in taken for file in index.files):
            continue
        part.emptied = True
        indexed = index.repos.get(("model", repo))
        folder = _repo_folder(hub, repo)
        for name in sorted(indexed.revisions.get(revision, ()) if indexed is not None else ()):
            ref = folder / "refs" / name
            if _is_link(ref) or not _inside(ref, folder):
                collect.hold(1, f"A ref of {repo} is a link or lies outside its folder; deleting it is not built.")
                continue
            part.refs.append(ref)
            part.units.append(ref)
            collect.item(part.kind, "hub", ref)
    result.parts.extend(parts[key] for key in sorted(parts))
    result.also_taken_shared = [
        {"bytes": sum(sizes), "with": sorted(_label(sharer) for sharer in sharers)}
        for sharers, sizes in sorted(shares.items(), key=lambda item: sorted(item[0]))]
    kept_by: dict[frozenset[str], set[tuple]] = defaultdict(set)
    for file, sharers in kept.values():
        kept_by[sharers].add(file.identity)
    result.kept_shared = [
        {"bytes": sum(index.sizes.get(identity, 0) for identity in identities),
         "with": sorted(_label(sharer) for sharer in sharers)}
        for sharers, identities in sorted(kept_by.items(), key=lambda item: sorted(item[0]))]

    # The converted copy, its work directories, and older recipes of it.
    targets = {model: index.converted_targets[model] for model in sorted(group) if model in index.converted_targets}
    for target in targets.values():
        # By name first: a target resolved through another HF_HOME is never
        # even looked at.
        if not _lexically_inside(target, converted_root):
            # Unlooked at, it says nothing of what the entry has on disk.
            collect.hold(1, "Its converted copy's folder lies outside this server's converted cache; "
                            "deleting it is not built.", on_disk=False)
            continue
        units = []
        if os.path.lexists(target):
            if _is_link(target) or not target.is_dir():
                collect.hold(1, "Its converted copy is a link, not a folder; deleting it is not built.")
                continue
            units.append(target)
        units.extend(path for path in _work_directories(converted_root, target, _is_work))
        if not units:
            continue
        if not _inside(target, converted_root):
            collect.hold(1, "Its converted copy's folder lies outside this server's converted cache; "
                            "deleting it is not built.")
            continue
        part = Part("converted", name=target.name, units=units, target=target)
        for unit in units:
            size, items = collect.tree("converted", "converted", unit)
            part.bytes += size
            part.files += len(items)
        result.parts.append(part)
    sources = _conversion_sources(index)
    # Each of the group's current sources and the model converting from it.
    mine = {sources[model]: model for model in sorted(group) if model in sources}
    mine_repos = {repo for repo, _ in mine}
    current = {target.name for target in index.converted_targets.values()}
    for tree in _older_trees(converted_root, current, index.converted_targets.values(), _is_work):
        source = _marker_source(tree / CONVERSION_MARKER)
        if source is None:
            continue  # without a readable marker it stays "not referenced"
        if source in mine:
            if not _inside(tree, converted_root):
                collect.hold(1, "An older converted copy lies outside this server's converted cache.")
                continue
            size, items = collect.tree("converted-old", "converted", tree)
            result.parts.append(Part("converted-old", name=tree.name, bytes=size, files=len(items), units=[tree],
                                     target=index.converted_targets[mine[source]]))
        elif source[0] in mine_repos:
            owners = {model for model, (repo, _) in sources.items() if repo == source[0]}
            claimed = {sources[model][1] for model in owners}
            if len({entry_of(model) for model in owners}) > 1 and source[1] not in claimed:
                collect.hold(2, f"The older converted copy {tree.name} comes from {source[0]}, which "
                                f"{' and '.join(sorted(_label(entry_of(model)) for model in owners))} share, "
                                "from a file neither uses now; it is not attributed to either.")

    # Partial downloads in what the plan would touch: another process may be
    # writing them (refused by the delete with `model_downloading`).
    for repo in sorted(touched):
        blobs = _repo_folder(hub, repo) / "blobs"
        try:
            partial = sorted(blobs.glob("*.incomplete"))
        except OSError:
            partial = []
        result.downloading.extend(
            {"repo": repo, "file": Path(os.path.relpath(path, hub)).as_posix()} for path in partial)

    # A cache-wide shared Xet store serves the Hub parts' bytes to other
    # repositories: held where the plan takes something from the Hub, never
    # for an entry with nothing there (its converted copy is not the store's).
    if (any(part.kind in ("hub", "lite") for part in result.parts)
            and os.path.lexists(hub / "blobs" / SHARED_STORE_MARKER)):
        collect.hold(0, "This Hugging Face cache keeps a shared Xet store across repositories; "
                        "deleting from it is not built yet.")
    if any(part.shared_repository for part in result.parts) and not SHARED_REPOSITORY_DELETE:
        repos = sorted({part.repo for part in result.parts if part.shared_repository and part.repo})
        others = sorted({_label(entry_of(model)) for repo in repos for model in index.recipe_repos.get(repo, ())}
                        - {entry.name})
        collect.hold(4, f"Its files in {', '.join(repos)} sit in a repository "
                        f"{', '.join(others)} also use{'s' if len(others) == 1 else ''}; deleting files "
                        "inside a shared repository is not built yet.")
    for removal in collect.removals:
        if not _inside(removal.path, collect.roots[removal.root]):
            collect.hold(1, "A planned file lies outside this server's caches; nothing can be deleted.")
            break
    if collect.held:
        result.held = {"code": MODEL_DELETE_HELD, "detail": min(collect.held)[1]}
    result.removals = sorted(collect.removals, key=lambda removal: (removal.root, removal.relative))
    result.pin = _pin(result, kept, targets)
    if not result.parts and not result.kept_shared and not result.downloading and not collect.found:
        raise NotOnDisk(entry_id)
    return result


def annotate_holds(context: PlanContext | None = None) -> Callable[[dict, model_cache_storage.CacheIndex], None]:
    """For `model_cache_storage.snapshot(annotate=...)`: each catalogue
    entry's row gains `delete_held`, its plan's `held` detail on this scan's
    index (a shared Xet store, a link out of its repository, an unrecorded
    revision several entries could own, ...), or null; a lite id's sub-line
    always null.
    So the page holds such a row's Delete… as it reads the cache, not only
    once pressed. The plan's other refusals (selected, busy, a partial
    download) are the page's and the press's. Runs under the snapshot's
    scan reservation, reading no file the plans would not."""
    context = context or PlanContext()

    def annotate(answer: dict, index: model_cache_storage.CacheIndex) -> None:
        for row in answer.get("models", []):
            row["delete_held"] = None
            entry = model_info.catalogue_entry(row["model_id"])
            if entry is None or entry.repo != row["model_id"]:
                continue
            try:
                planned = _plan(entry.repo, context, None, index)
            except NotOnDisk:
                continue
            except (OSError, ValueError):
                # The press reads its plan again and says what it meets.
                log.warning("could not read %s's delete plan with the model cache", entry.name)
                log.debug("plan of %s", entry.repo, exc_info=True)
                continue
            if planned.held is not None:
                row["delete_held"] = planned.held["detail"]

    return annotate


def _present_entries(index: model_cache_storage.CacheIndex) -> set[str]:
    """Entries with something of their own on disk: exclusive Hub bytes,
    other-revision bytes or a converted tree (the last-holder rule)."""
    present = set()
    for model, row in index.rows.items():
        if row.get("exclusive_bytes") or row.get("other_revision_bytes") or row.get("converted_bytes"):
            present.add(entry_of(model))
    return present


def _work_directories(root: Path, target: Path, is_work) -> list[Path]:
    try:
        return sorted(path for path in root.iterdir() if path.name.startswith(target.name + "-") and is_work(path, target))
    except OSError:
        return []


def _older_trees(root: Path, current: set[str], targets: Iterable[Path], is_work) -> list[Path]:
    """Folders under the converted root that are no catalogue entry's
    current target, nor one's work directory, nor a link."""
    targets = list(targets)
    try:
        candidates = sorted(root.iterdir())
    except OSError:
        return []
    return [path for path in candidates
            if path.name not in current and not _is_link(path) and path.is_dir()
            and not any(is_work(path, target) for target in targets)]


def _marker_source(marker: Path) -> tuple[str, str | None] | None:
    """(source repository, file) from a converted tree's marker, or None
    when it is missing, unreadable or names no source repository."""
    try:
        if _is_link(marker) or not marker.is_file():
            return None
        with marker.open("rb") as stream:
            data = stream.read(_MARKER_BYTES + 1)
        if len(data) > _MARKER_BYTES:
            return None
        recipe = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    source = recipe.get("source") if isinstance(recipe, dict) else None
    if not isinstance(source, dict) or not isinstance(source.get("repo_id"), str):
        return None
    filename = source.get("filename")
    return source["repo_id"], filename if isinstance(filename, str) else None


def _conversion_sources(index: model_cache_storage.CacheIndex) -> dict[str, tuple[str, str | None]]:
    """Each converting catalogue id's current source (repository, file)."""
    sources = {}
    for model in index.converted_targets:
        conversion = model_info._conversion_target(model)
        if conversion is None:
            continue
        source = json.loads(conversion[2]).get("source") or {}
        if isinstance(source.get("repo_id"), str):
            filename = source.get("filename")
            sources[model] = (source["repo_id"], filename if isinstance(filename, str) else None)
    return sources


def _pin(result: Plan, kept: dict, targets: dict[str, Path]) -> str:
    """sha256 over every planned removal (kind, path relative to its root,
    size, mtime_ns), the kept and also-taken sets and the converted target's
    name: the delete re-plans and refuses a different one."""
    def kept_stamp(file: model_cache_storage.IndexedFile, sharers: frozenset[str]) -> list:
        try:
            when = os.lstat(file.entry).st_mtime_ns
        except OSError:
            when = -1
        return [file.repo_id, file.revision, file.name, file.size, when, sorted(sharers)]

    payload = {
        "entry": result.entry,
        "remove": [[removal.kind, removal.root, removal.relative, removal.size, removal.mtime_ns]
                   for removal in result.removals],
        "kept": sorted(kept_stamp(file, sharers) for file, sharers in kept.values()),
        "also_taken": sorted([share["bytes"], share["with"]] for share in result.also_taken_shared),
        "converted": sorted(target.name for target in targets.values()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()



# ---------------------------------------------------------------- the delete


class DeleteRefused(Exception):
    """A delete refused with nothing changed: `code` (a `refusal_codes`
    code) and `detail`, the person's sentence. A route raises it as
    `refusals.refusal(409, error.detail, error)`. Raised while renaming
    (`remove()`), it also names the `part` that refused, the `reason` its
    `remaining` row would give and, when not the part's planned bytes, the
    `size` left: what is said if a rename then will not go back."""

    def __init__(self, code: str, detail: str, *, part: Part | None = None, reason: str | None = None,
                 size: int | None = None) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.part = part
        self.reason = reason
        self.size = size


@dataclass
class Outcome:
    """What one delete did: whether anything changed yet, and what stays
    (`remaining`: each part's kind, its bytes left, the reason in words)."""

    label: str
    planned_bytes: int
    changed: bool = False
    freed_bytes: int | None = None
    remaining: list[dict] = field(default_factory=list)

    def stop(self, part: Part, code: str, detail: str, reason: str, size: int | None = None) -> None:
        """Before anything changed, refuse with `code`; after, record what
        stays and go on: something did change, so no refusal code may say
        nothing happened."""
        if not self.changed:
            raise DeleteRefused(code, detail)
        self.remaining.append({"kind": part.kind, "bytes": part.bytes if size is None else size, "reason": reason})

    def answer(self) -> dict:
        return {"label": self.label, "planned_bytes": self.planned_bytes,
                "freed_bytes": self.freed_bytes, "remaining": self.remaining}


def delete(model_id: str, pin: str, context: PlanContext | None = None, cache_dir: Path | None = None,
           *, lock_timeout: float = LOCK_TIMEOUT) -> Outcome:
    """Delete `model_id`'s catalogue entry as the plan with `pin` said, on
    the calling (daemon) thread. Holds the scan reservation for its whole run
    (`model_cache_storage.ScanBusy` while another scan or delete has it) and
    `keep_awake.busy`; re-plans with `context`, refuses a different pin, a
    selected, loaded or saved-for-start entry, a fetch it touches, a partial
    download or a held part, and re-asserts every path inside the configured
    roots, all before the first rename (`DeleteRefused`, nothing changed);
    then `remove()` renames every part before removing any, so a refused
    rename puts the others back and is `DeleteRefused` too.
    Raises `NotOnDisk` as `plan()` does. The model cache facts are
    invalidated on the way out, whatever happened."""
    entry = model_info.catalogue_entry(model_id)
    if entry is None or entry.repo != model_id:
        raise NotOnDisk(model_id)
    if not model_cache_storage._scan_lock.acquire(blocking=False):
        raise model_cache_storage.ScanBusy
    try:
        with keep_awake.busy(DELETING):
            result = _plan(entry.repo, context or PlanContext(), cache_dir)
            _refuse_before_change(result, pin)
            outcome = Outcome(result.label, result.frees_bytes)
            volumes = _volumes(result)
            # What an earlier delete's open file left in the trash goes first,
            # so the free space measured after counts this delete's alone.
            if result.hub is not None:
                sweep_trash(model_cache_storage.trash_dir(result.hub))
            before = _free(volumes)
            try:
                remove(result, outcome, lock_timeout=lock_timeout)
            finally:
                after = _free(volumes)
                if before is not None and after is not None:
                    # Another writer can move it either way: "about".
                    outcome.freed_bytes = max(0, after - before)
            return outcome
    finally:
        model_cache_storage._scan_lock.release()
        model_info.invalidate_cache_facts()


def _refuse_before_change(result: Plan, pin: str) -> None:
    """The worker's own checks on its fresh plan; the route's guards ran
    first (`main._model_delete_refusal`), these hold whatever changed since."""
    label = result.label
    if not hmac.compare_digest(result.pin.encode(), str(pin).encode()):
        raise DeleteRefused(MODEL_CACHE_CHANGED, f"{label}'s files changed since the plan was read; "
                                                 "read it again.")
    context = result.context
    if any(result.names(current) for current in (context.selected, context.loaded, context.saved)):
        raise DeleteRefused(MODEL_SELECTED, f"{label} is the selected checkpoint or the one saved to load "
                                            "on the next start. Choose another checkpoint first.")
    if result.fetching(context.pending):
        raise DeleteRefused(MODEL_DELETE_BUSY, f"Check is reading a tokenizer from its files; delete {label} "
                                               "once it is done.")
    if result.downloading:
        partial = result.downloading[0]
        raise DeleteRefused(MODEL_DOWNLOADING, f"A download is still writing {partial['file']} in "
                                               f"{partial['repo']}; let it finish, or stop it, first.")
    if result.held is not None:
        raise DeleteRefused(MODEL_DELETE_HELD, result.held["detail"])
    if not HUB_DELETE and any(part.kind in ("hub", "lite") for part in result.parts):
        raise DeleteRefused(MODEL_DELETE_HELD, f"Deleting {label}'s downloaded files is not built yet; "
                                               "nothing was deleted.")
    assert_inside_roots(result)


def _same_folder(one: Path, other: Path) -> bool:
    return os.path.normcase(os.path.abspath(one)) == os.path.normcase(os.path.abspath(other))


def assert_inside_roots(result: Plan) -> None:
    """Every path the plan acts on - each removal, each unit, each converted
    target, its lock and the folder its work directories go in - lies inside
    this server's configured roots, read now (the Hub cache as the scan
    resolves it, the converted root), and the plan's roots are those. Raises
    `DeleteRefused` (`model_delete_held`) otherwise; called before the
    first rename, so nothing has changed."""
    hub = model_cache_storage.hub_cache_dir().expanduser().resolve()
    converted = model_cache_storage.hf_home_dir("converted")
    refused = DeleteRefused(MODEL_DELETE_HELD, "A planned file lies outside this server's caches; "
                                               "nothing was deleted.")
    if (result.hub is None or result.converted_root is None or not _same_folder(result.hub, hub)
            or not _same_folder(result.converted_root, converted)):
        raise refused
    roots = {"hub": hub, "converted": converted}
    paths = [(removal.root, removal.path) for removal in result.removals]
    for part in result.parts:
        root = "converted" if part.kind in ("converted", "converted-old") else "hub"
        paths.extend((root, unit) for unit in part.units)
        if root == "converted":
            target = part.target
            # The loaders lock and work beside the target, in the root itself.
            if target is None or not _same_folder(target.parent, converted):
                raise refused
            paths.append((root, target))
            # Its lock, and an older tree's own.
            paths.extend((root, converted / f"{name}.lock") for name in
                         {target.name, *(unit.name for unit in part.units if part.kind == "converted-old")})
    for root, path in paths:
        if not _inside(path, roots[root]):
            raise refused
    # The Hub cache's trash is its sibling, never a link, never a drive root's.
    trash = model_cache_storage.trash_dir(hub)
    if any(part.kind in ("hub", "lite") for part in result.parts) and (
            hub.parent == hub or (os.path.lexists(trash) and (_is_link(trash) or not trash.is_dir()))):
        raise refused


def _volumes(result: Plan) -> list[Path]:
    """The plan's roots, one per volume, for the free-space measure."""
    found: dict[object, Path] = {}
    for root in (result.hub, result.converted_root):
        if root is None:
            continue
        try:
            device: object = os.stat(root).st_dev
        except OSError:
            device = root
        found.setdefault(device, root)
    return list(found.values())


def _free(volumes: list[Path]) -> int | None:
    sizes = [model_cache_storage.free_bytes(volume) for volume in volumes]
    known = [size for size in sizes if size is not None]
    return sum(known) if known and len(known) == len(sizes) else None


_CONVERTED_KINDS = ("converted", "converted-old")
_HUB_KINDS = ("hub", "lite")


@dataclass
class _Move:
    """One rename a delete made before removing anything: `part`'s folder
    from `source` to `moved`, inside `holder`, the empty folder made for it
    (a work directory of the target, or a folder of the Hub cache's trash).
    Each part makes at most one."""

    part: Part
    source: Path
    moved: Path
    holder: Path


def remove(result: Plan, outcome: Outcome, *, lock_timeout: float = LOCK_TIMEOUT,
           kinds: Iterable[str] | None = None) -> None:
    """Remove the plan's parts (those of `kinds`, all by default), renaming
    every one before removing any (docs/maintainer/plan-model-delete.md,
    "How files are removed"):

    1. each converted part under the loaders' lock (the target's `.lock`, an
       older tree's own too, each taken once and held to the end of step 4),
       its tree re-read as planned and renamed into the target's work shape;
    2. each repository the group owns outright renamed into a new folder of
       the Hub cache's trash;
    3. the files inside shared repositories (stage 4) checked as planned,
       then unlinked (`_unlink_files`): refs of each emptied revision, its
       snapshot entries, the blobs no surviving entry resolves to, emptied
       folders bottom-up and `.no_exist/<commit>`;
    4. every work directory of each converted target removed;
    5. the locks released, each repository's trash folder removed.

    A refusal in steps 1-3 before the first unlink (a lock held, a tree,
    repository or file changed, a partial download, a refused rename, a
    first unlink refused) puts every rename back and raises
    `DeleteRefused`: nothing changed, whichever part refused. Only if a
    rename will not go back has something changed; then what was put back,
    or never reached, is `remaining` beside the refused part, and what will
    not go back is removed. A failure after the first unlink, or in steps
    4-5, is `remaining`. The caller holds the scan reservation and has
    checked the plan (`delete()`)."""
    from .backends.single_file import _is_work

    wanted = None if kinds is None else set(kinds)
    parts = [part for part in result.parts if wanted is None or part.kind in wanted]
    converted = sorted((part for part in parts if part.kind in _CONVERTED_KINDS),
                       key=lambda part: part.kind != "converted")
    repositories = [part for part in parts if part.kind in _HUB_KINDS and not part.shared_repository]
    shared = [part for part in parts if part.kind in _HUB_KINDS and part.shared_repository]
    moves: list[_Move] = []
    going = {id(part) for part in parts}
    with ExitStack() as locks:
        try:
            taken: set[str] = set()
            for part in converted:
                _stage_converted(result, part, locks, taken, moves, lock_timeout)
            for part in repositories:
                _stage_repository(result, part, moves)
            for part in shared:
                _check_files(result, part)
            # Raises only when its first unlink is refused: nothing changed.
            _unlink_files(result, shared, outcome)
        except DeleteRefused as refused:
            stranded = _put_back(result, moves)
            if not stranded:
                raise
            # A rename that will not go back: that part is out of the cache
            # (a work directory, the trash), so something changed after all.
            outcome.changed = True
            moves = stranded
            going = {id(move.part) for move in stranded}
            for part in parts:
                if id(part) in going:
                    continue
                if part is refused.part:
                    outcome.remaining.append({"kind": part.kind,
                                              "bytes": part.bytes if refused.size is None else refused.size,
                                              "reason": refused.reason or refused.detail})
                else:
                    outcome.remaining.append({"kind": part.kind, "bytes": part.bytes,
                                              "reason": "It stays: the delete stopped at another part."})
        if moves:
            outcome.changed = True
        for part in converted:
            if id(part) in going:
                assert part.target is not None
                _sweep_work(part.target.parent, part.target, part, outcome, result.label, _is_work)
    _remove_trash(result, [move for move in moves if move.part.kind in _HUB_KINDS], outcome)


def remove_converted(result: Plan, outcome: Outcome, *, lock_timeout: float = LOCK_TIMEOUT) -> None:
    """`remove()` for the plan's converted parts alone."""
    remove(result, outcome, lock_timeout=lock_timeout, kinds=_CONVERTED_KINDS)


def remove_hub(result: Plan, outcome: Outcome) -> None:
    """`remove()` for the plan's Hub and lite parts alone: the repositories
    it takes whole (the entry's own and its lite ones) and its files inside
    shared ones."""
    remove(result, outcome, kinds=_HUB_KINDS)


def _put_back(result: Plan, moves: list[_Move]) -> list[_Move]:
    """Undo `moves`, newest first, each folder renamed back to where it was
    and the empty folder made for it removed; answers the moves that would
    not go back (in their order), left where they were moved to."""
    stranded = []
    for move in reversed(moves):
        try:
            os.rename(move.moved, move.source)
        except OSError:
            log.warning("could not put %s back after a refused delete", move.source.name)
            log.debug("rename back of %s", move.source.name, exc_info=True)
            stranded.append(move)
            continue
        try:
            move.holder.rmdir()
        except OSError:
            pass  # empty: a work directory or trash folder the next load or delete sweeps
    if result.hub is not None and any(move.part.kind in _HUB_KINDS for move in moves):
        try:
            model_cache_storage.trash_dir(result.hub).rmdir()
        except OSError:
            pass  # an earlier leftover, or a stranded one, is still there
    return stranded[::-1]


def _stage_converted(result: Plan, part: Part, locks: ExitStack, taken: set[str], moves: list[_Move],
                     lock_timeout: float) -> None:
    """Steps 1 of `remove()` for one converted part:

    1. the target's `.lock` (an older tree's own too), with a short timeout:
       a timeout is a conversion running; a lock another part took is not
       taken twice (a second `FileLock` on it in this process would wait on
       the first);
    2. the tree as planned, re-read under the lock;
    3. `os.rename(tree, <target>-<8>/tree)`: the tree stops being a hit at
       once, and a refused rename changes nothing.

    Its work directories go in step 3 (`_sweep_work`), a failure midway
    leaving only `_is_work`'s shape, which the next load's `sweep_work` and
    the next delete remove. The `.lock` stays: zero bytes, recreated by the
    loaders, and unlinking a lock another process may take after the
    release would let two holders sweep each other's work directories.
    Raises `DeleteRefused` (with the part) on any refusal."""
    from filelock import Timeout
    from huggingface_hub.utils._fixes import WeakFileLock

    target = part.target
    assert target is not None
    root = target.parent
    label = result.label
    # By what the plan made each unit, not by the disk: a work directory a
    # load swept since is no tree that changed.
    trees = [unit for unit in part.units if unit == target or part.kind == "converted-old"]
    for lock in [root / f"{target.name}.lock", *(root / f"{tree.name}.lock" for tree in trees if tree != target)]:
        key = os.path.normcase(os.path.abspath(lock))
        if key in taken:
            continue
        try:
            locks.enter_context(WeakFileLock(lock, timeout=lock_timeout))
        except Timeout:
            raise DeleteRefused(MODEL_DELETE_BUSY, f"{label} is being converted or loaded by another run; delete "
                                                   "it once that is done.",
                                part=part, reason="Another run was converting or loading it.") from None
        taken.add(key)
    changed = f"{label}'s converted files changed since the plan was read; read it again."
    if part.kind == "converted" and target not in trees and os.path.lexists(target):
        raise DeleteRefused(MODEL_CACHE_CHANGED, changed, part=part, size=0,
                            reason="Its converted copy appeared after the plan was read.")
    for tree in trees:
        if not _as_planned(result, tree):
            raise DeleteRefused(MODEL_CACHE_CHANGED, changed, part=part, reason="It changed after the plan was read.")
    for tree in trees:
        moves.append(_move_aside(tree, target, part, label))


def _as_planned(result: Plan, tree: Path) -> bool:
    """Whether `tree` is still the real folder the plan measured, file for
    file (path, size, mtime_ns)."""
    if _is_link(tree) or not tree.is_dir():
        return False
    assert result.hub is not None and result.converted_root is not None
    collect = _Collector(result.hub, result.converted_root)
    _, items = collect.tree("converted", "converted", tree)
    prefix = tree.name + "/"
    planned = {(removal.relative, removal.size, removal.mtime_ns) for removal in result.removals
               if removal.root == "converted" and removal.relative.startswith(prefix)}
    return not collect.held and {(item.relative, item.size, item.mtime_ns) for item in items} == planned


def _new_work(root: Path, target: Path) -> Path:
    """A fresh, empty `<target>-<8 [a-z0-9]>` folder in `root`."""
    for _ in range(64):
        work = root / f"{target.name}-{''.join(secrets.choice(_WORK_ALPHABET) for _ in range(8))}"
        try:
            work.mkdir()
        except FileExistsError:
            continue
        return work
    raise FileExistsError(f"no free work directory name beside {target.name}")


def _move_aside(tree: Path, target: Path, part: Part, label: str) -> _Move:
    """Rename `tree` into a new work directory of `target`; raises
    `DeleteRefused` (`model_files_in_use`) when it would not go."""
    in_use = (f"A file in {label}'s converted copy is open, perhaps in another program, so it could not "
              "be moved aside; nothing was deleted.")
    try:
        work = _new_work(tree.parent, target)
    except OSError:
        log.warning("could not make a work directory to delete the converted copy %s", tree.name)
        log.debug("work directory for %s", tree.name, exc_info=True)
        raise DeleteRefused(MODEL_FILES_IN_USE, in_use, part=part, reason="It could not be moved aside.") from None
    try:
        os.rename(tree, work / "tree")
    except OSError:
        log.warning("could not move the converted copy %s aside: a file in it is open", tree.name)
        log.debug("rename of %s", tree.name, exc_info=True)
        try:
            work.rmdir()
        except OSError:
            pass  # empty: `_is_work`'s shape, swept by the next load or delete
        raise DeleteRefused(MODEL_FILES_IN_USE, in_use, part=part,
                            reason="A file in it is open, so it could not be moved aside.") from None
    return _Move(part, tree, work / "tree", work)


def _contents(folder: Path) -> tuple[int, int]:
    """(entries, logical bytes) under `folder`, never following a link."""
    entries = size = 0
    for directory, dirs, files in os.walk(folder, followlinks=False):
        for name in [*dirs, *files]:
            entries += 1
            try:
                info = os.lstat(Path(directory) / name)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                size += info.st_size
    return entries, size


def _sweep_work(root: Path, target: Path, part: Part, outcome: Outcome, label: str, is_work) -> None:
    """Remove every work directory of `target`, under its lock: each is
    stale, the renamed tree's included."""
    for work in _work_directories(root, target, is_work):
        if not _inside(work, root):
            continue
        before = _contents(work)
        try:
            shutil.rmtree(work)
        except OSError:
            log.warning("could not remove all of a deleted converted copy yet (%s): a file in it is open",
                        work.name)
            log.debug("rmtree of %s", work.name, exc_info=True)
            left = _contents(work) if os.path.lexists(work) else (0, 0)
            if left != before:
                outcome.changed = True
            outcome.stop(part, MODEL_FILES_IN_USE,
                         f"A file in {label}'s converted copy is open, perhaps in another program; "
                         "nothing was deleted.",
                         "A file in it is still open; the next load or delete removes the rest.", size=left[1])
            continue
        outcome.changed = True


# ------------------------------------------------- repositories, whole


def _is_bin(path: Path) -> bool:
    """One of the trash's own folders: eight hex digits, a real folder."""
    return (len(path.name) == 8 and all(char in "0123456789abcdef" for char in path.name)
            and not _is_link(path) and path.is_dir())


def sweep_trash(trash: Path) -> list[str]:
    """Remove what earlier deletes left in the Hub cache's trash (only its
    own `<8 hex>` folders, never a link), and the trash once empty. Answers
    the names it could not remove; a failure is logged and left for the
    next delete."""
    if not os.path.lexists(trash) or _is_link(trash) or not trash.is_dir():
        return []
    left = []
    try:
        bins = sorted(trash.iterdir())
    except OSError:
        return []
    for bin_ in bins:
        if not _is_bin(bin_):
            left.append(bin_.name)
            continue
        try:
            shutil.rmtree(bin_)
        except OSError:
            log.warning("could not yet remove files an earlier checkpoint delete left (%s): a file is open",
                        bin_.name)
            log.debug("rmtree of %s", bin_.name, exc_info=True)
            left.append(bin_.name)
    if not left:
        try:
            trash.rmdir()
        except OSError:
            pass  # not empty after all: the next delete's
    return left


def _new_bin(trash: Path) -> Path:
    """A fresh, empty `<8 hex>` folder in the trash (made if need be)."""
    trash.mkdir(exist_ok=True)
    if _is_link(trash) or not trash.is_dir():
        raise NotADirectoryError(trash.name)
    for _ in range(64):
        bin_ = trash / secrets.token_hex(4)
        try:
            bin_.mkdir()
        except FileExistsError:
            continue
        return bin_
    raise FileExistsError(f"no free folder name in {trash.name}")


def _partial(folder: Path) -> str | None:
    """A partial download's name (relative to the Hub cache) in `folder`."""
    try:
        found = sorted((folder / "blobs").glob("*.incomplete"))
    except OSError:
        return None
    return f"{folder.name}/blobs/{found[0].name}" if found else None


def _stage_repository(result: Plan, part: Part, moves: list[_Move]) -> None:
    """Step 2 of `remove()` for one whole repository: renamed out of the
    Hub cache into a new folder of its trash. Raises `DeleteRefused` (with
    the part) for a part that is not one whole folder, a partial download or
    a refused rename."""
    label = result.label
    if not part.whole or part.shared_repository or len(part.units) != 1:
        # Never reached: `remove()` takes a shared repository's files one by
        # one (`_unlink_files`), and only a whole folder comes here.
        raise DeleteRefused(MODEL_DELETE_HELD, f"{label}'s files in {part.repo} are not one whole folder of its "
                                               "own; nothing was deleted.",
                            part=part, reason="It sits in a repository another checkpoint uses.")
    folder = part.units[0]
    partial = _partial(folder)
    if partial is not None:
        raise DeleteRefused(MODEL_DOWNLOADING, f"A download is still writing {partial} in {part.repo}; "
                                               "let it finish, or stop it, first.",
                            part=part, reason="A download started writing in it.")
    assert result.hub is not None
    moves.append(_move_to_trash(folder, model_cache_storage.trash_dir(result.hub), part, label))


def _move_to_trash(folder: Path, trash: Path, part: Part, label: str) -> _Move:
    """Rename one repository folder into a new folder of the trash; raises
    `DeleteRefused` when it would not go."""
    try:
        bin_ = _new_bin(trash)
    except OSError:
        try:
            trash.rmdir()  # made for this delete and left empty: nothing changed
        except OSError:
            pass
        log.warning("could not make a folder beside the Hub cache to delete %s", part.repo)
        log.debug("trash folder for %s", part.repo, exc_info=True)
        raise DeleteRefused(MODEL_FILES_IN_USE, f"No folder could be made beside the Hub cache to delete "
                                                f"{label}'s files from; nothing was deleted.",
                            part=part, reason="It could not be moved out of the cache.") from None
    try:
        os.rename(folder, bin_ / folder.name)
    except OSError as error:
        for empty in (bin_, trash):
            try:
                empty.rmdir()  # the trash too, unless another folder is in it
            except OSError:
                break  # empty: the next delete sweeps it
        if isinstance(error, FileNotFoundError):
            log.warning("%s's cache folder went before its delete could move it", part.repo)
            raise DeleteRefused(MODEL_CACHE_CHANGED, f"{label}'s files changed since the plan was read; read it "
                                                     "again.",
                                part=part, size=0, reason="It went from the cache before the delete reached it.") \
                from None
        log.warning("could not move %s out of the Hub cache: a file in it is open, or it is on another drive",
                    part.repo)
        log.debug("rename of %s", part.repo, exc_info=True)
        if getattr(error, "errno", None) == errno.EXDEV:
            raise DeleteRefused(MODEL_FILES_IN_USE, f"{label}'s files ({part.repo}) lie on another drive than the "
                                                    "Hub cache's folder, so they could not be moved out of it; "
                                                    "nothing was deleted.",
                                part=part, reason="It lies on another drive, so it could not be moved out of the "
                                                  "cache.") from None
        raise DeleteRefused(MODEL_FILES_IN_USE,
                            f"A file in {label}'s downloaded files ({part.repo}) is open, perhaps in another "
                            "program, so they could not be moved out of the cache; nothing was deleted.",
                            part=part, reason="A file in it is open, so it could not be moved out of the cache.") \
            from None
    return _Move(part, folder, bin_ / folder.name, bin_)


def _remove_trash(result: Plan, moves: list[_Move], outcome: Outcome) -> None:
    """Step 4 of `remove()`: each moved repository's trash folder removed;
    what an open file keeps there is `remaining` (out of the cache already;
    the next delete sweeps it), and the trash goes once empty."""
    if not moves:
        return
    assert result.hub is not None
    for move in moves:
        try:
            shutil.rmtree(move.holder)
        except OSError:
            log.warning("could not remove all of %s's deleted files yet (%s): a file in them is open",
                        move.part.repo, move.holder.name)
            log.debug("rmtree of %s", move.holder.name, exc_info=True)
            left = _contents(move.holder) if os.path.lexists(move.holder) else (0, 0)
            outcome.stop(move.part, MODEL_FILES_IN_USE, f"A file in {result.label}'s downloaded files is open; "
                                                        "nothing was deleted.",
                         "A file in it is still open; it is out of the cache, and the next delete removes the rest.",
                         size=left[1])
    if not any(os.path.lexists(move.holder) for move in moves):
        try:
            model_cache_storage.trash_dir(result.hub).rmdir()
        except OSError:
            pass  # an earlier leftover is still there: the inventory names it


# ------------------------------------- files inside a shared repository


def _key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _refs_naming(folder: Path, commits: set[str]) -> dict[str, list[Path]]:
    """Each of `commits` -> the ref files under `folder/refs` naming it, read
    now (`refs/main`, `refs/refs/pr/1`, ...). A ref is read as the Hub's scan
    reads it, its text the commit; surrounding whitespace is ignored here, so
    a delete finds more refs, never fewer."""
    found: dict[str, list[Path]] = defaultdict(list)
    refs = folder / "refs"
    if not commits or _is_link(refs) or not refs.is_dir():
        return found
    for directory, dirs, files in os.walk(refs, followlinks=False):
        here = Path(directory)
        dirs[:] = [name for name in dirs if not _is_link(here / name)]
        for name in files:
            path = here / name
            try:
                commit = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if commit in commits:
                found[commit].append(path)
    return found


def _snapshot_items(directory: Path) -> set[str]:
    """Every file and link under `directory`, never following a link."""
    items: set[str] = set()
    for here, dirs, files in os.walk(directory, followlinks=False):
        links = [name for name in dirs if _is_link(Path(here) / name)]
        dirs[:] = [name for name in dirs if name not in links]
        items.update(_key(Path(here) / name) for name in [*files, *links])
    return items


def _check_files(result: Plan, part: Part) -> None:
    """Step 3's check, before the first unlink, for one part inside a shared
    repository: no partial download in the repository, every planned entry,
    blob and ref still as planned (size, mtime_ns), and an emptied revision
    still holding only the planned entries, with only the planned refs
    naming it. Raises `DeleteRefused` (with the part) otherwise."""
    assert result.hub is not None and part.repo is not None and part.revision is not None
    label = result.label
    folder = _repo_folder(result.hub, part.repo)
    changed = DeleteRefused(MODEL_CACHE_CHANGED, f"{label}'s files changed since the plan was read; read it again.",
                            part=part, reason="It changed after the plan was read.")
    if _is_link(folder) or not folder.is_dir():
        raise changed
    partial = _partial(folder)
    if partial is not None:
        raise DeleteRefused(MODEL_DOWNLOADING, f"A download is still writing {partial} in {part.repo}; "
                                               "let it finish, or stop it, first.",
                            part=part, reason="A download started writing in it.")
    planned = {_key(removal.path): removal for removal in result.removals}
    for path in [*part.entries, *part.blobs, *part.refs]:
        removal = planned.get(_key(path))
        try:
            info = os.lstat(path)
        except OSError:
            raise changed from None
        size = info.st_size if stat.S_ISREG(info.st_mode) else 0
        if removal is None or (size, info.st_mtime_ns) != (removal.size, removal.mtime_ns):
            raise changed
    if part.emptied:
        snapshot = folder / "snapshots" / part.revision
        if _snapshot_items(snapshot) != {_key(entry) for entry in part.entries}:
            raise changed
        named = _refs_naming(folder, {part.revision}).get(part.revision, [])
        if {_key(ref) for ref in named} != {_key(ref) for ref in part.refs}:
            raise changed


class _Stopped(Exception):
    """An unlink refused inside a shared repository: the part it stopped,
    and whether at a blob (every snapshot entry naming it already gone)."""

    def __init__(self, part: Part, blob: bool = False) -> None:
        super().__init__(part.repo)
        self.part = part
        self.blob = blob


@dataclass
class _Unlinks:
    """How many files the file-level step has unlinked so far."""

    count: int = 0


def _unlink(path: Path, part: Part, done: _Unlinks, outcome: Outcome, *, blob: bool = False) -> None:
    """One file or link; one gone already is no failure. Anything else
    refused (an open file on Windows, a permission) stops the step at once."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError:
        log.warning("could not remove a file of %s: it is open, or not this server's to remove", part.repo)
        log.debug("unlink in %s", part.repo, exc_info=True)
        raise _Stopped(part, blob) from None
    done.count += 1
    outcome.changed = True


def _unlink_files(result: Plan, parts: list[Part], outcome: Outcome) -> None:
    """Step 3's unlinks for every part inside a shared repository, one
    repository at a time (`_unlink_repository`). A refused unlink stops the
    step at once: refused before anything was unlinked, it raises
    `DeleteRefused` (`model_files_in_use`, with the part), so `remove()`
    puts every rename back; after, the stopped part's bytes left and each
    part not reached are `remaining`. What stays is always files missing
    from a revision, which the next load downloads again, never a ref to a
    missing snapshot."""
    if not parts:
        return
    done = _Unlinks()
    by_repo: dict[str, list[Part]] = defaultdict(list)
    for part in parts:
        assert part.repo is not None
        by_repo[part.repo].append(part)
    order = [part for repo in sorted(by_repo) for part in by_repo[repo]]
    for repo in sorted(by_repo):
        try:
            _unlink_repository(result, repo, by_repo[repo], done, outcome)
        except _Stopped as stopped:
            part = stopped.part
            if done.count == 0:
                raise DeleteRefused(
                    MODEL_FILES_IN_USE,
                    f"A file in {result.label}'s downloaded files ({part.repo}) is open, perhaps in another "
                    "program; nothing was deleted.",
                    part=part, reason="A file in it is open, so it could not be removed.") from None
            # The stopped repository's parts (one before it may still hold
            # blobs, which go after every entry) and every later one.
            _remaining_after_stop(result, order[order.index(by_repo[repo][0]):], stopped, outcome)
            return


def _remaining_after_stop(result: Plan, parts: list[Part], stopped: _Stopped, outcome: Outcome) -> None:
    """What a stop inside a shared repository leaves, as `remaining`: for the
    stopped part and every part after it, the planned entries and blobs
    still on disk. A blob no snapshot entry names now (its entries went
    before the stop, whether it came at an entry or at a blob) is its own
    row: no later plan sees it, since plans read snapshot entries, so only
    removing it by hand frees it. The rest a later delete can take."""
    hub = result.hub
    assert hub is not None
    sizes = {_key(removal.path): removal.size for removal in result.removals}
    alive: dict[str, set[str]] = {}
    open_file = "A file in it is open, perhaps in another program, so "
    for later in parts:
        assert later.repo is not None
        left = [path for path in [*later.entries, *later.blobs] if os.path.lexists(path)]
        blobs = {_key(blob) for blob in later.blobs}
        left_blobs = [path for path in left if _key(path) in blobs]
        if left_blobs and later.repo not in alive:
            alive[later.repo] = _named_blobs(_repo_folder(hub, later.repo))
        unnamed = [path for path in left_blobs if _key(path) not in alive[later.repo]]
        named = [path for path in left if path not in unnamed]
        stopped_here = later is stopped.part
        if named or (stopped_here and not unnamed):
            reason = (f"{open_file}some of its files stay; a later delete can take them." if stopped_here
                      else "It stays: the delete stopped at another part.")
            outcome.remaining.append({"kind": later.kind, "reason": reason,
                                      "bytes": sum(sizes.get(_key(path), 0) for path in named)})
        if unnamed:
            if not stopped_here:
                why = "The delete stopped at another part, so its"
            elif stopped.blob:
                why = f"{open_file}its"
            else:
                why = "The delete stopped partway, so some of its"
            outcome.remaining.append({
                "kind": later.kind, "bytes": sum(sizes.get(_key(path), 0) for path in unnamed),
                "reason": f"{why} stored bytes stay in the repository's blobs folder, which no revision names "
                          "now; remove them by hand."})


def _unlink_repository(result: Plan, repo: str, parts: list[Part], done: _Unlinks, outcome: Outcome) -> None:
    """The plan's files inside one shared repository, in the order that keeps
    the cache readable at every step (docs/maintainer/plan-model-delete.md,
    "Files inside a repository others use"):

    1. each emptied revision's refs, as found now: it becomes detached, never
       a ref to a missing snapshot;
    2. each snapshot entry;
    3. each planned blob no surviving snapshot entry resolves to, checked
       now (a link's first hop, as the Hub's scan reads it): a blob still
       named is kept, never left a broken link;
    4. emptied folders bottom-up (an emptied revision's snapshot folder only
       while no ref names it), then its `.no_exist/<commit>`, and the
       repository's folder once nothing at all is left in it.

    Raises `_Stopped` at the first refused unlink, after step 4 for what it
    had unlinked here (only empty folders go, so it is safe at any point)."""
    assert result.hub is not None
    folder = _repo_folder(result.hub, repo)
    emptied = {part.revision: part for part in parts if part.emptied and part.revision is not None}
    before = done.count
    try:
        for commit, refs in sorted(_refs_naming(folder, set(emptied)).items()):
            for ref in sorted(refs):
                _unlink(ref, emptied[commit], done, outcome)
        for part in parts:
            for entry in part.entries:
                _unlink(entry, part, done, outcome)
        alive = _named_blobs(folder)
        for part in parts:
            for blob in part.blobs:
                if _key(blob) in alive:
                    log.info("kept a blob of %s that another revision's file still names", repo)
                    continue
                _unlink(blob, part, done, outcome, blob=True)
    finally:
        # Nothing unlinked here (a refusal that puts everything back), nothing pruned.
        if done.count > before:
            _prune_repository(folder, parts, set(emptied))


def _prune_repository(folder: Path, parts: list[Part], emptied: set[str]) -> None:
    """Step 4 of `_unlink_repository`."""
    named = _refs_naming(folder, emptied)
    for part in parts:
        assert part.revision is not None
        snapshot = folder / "snapshots" / part.revision
        whole = part.emptied and not named.get(part.revision)
        _prune(snapshot, including_top=whole)
        if whole and not os.path.lexists(snapshot):
            _remove_notes(folder / ".no_exist" / part.revision)
    if emptied:
        # A nested ref's emptied folders (`refs/refs/pr/`); `refs/` stays.
        _prune(folder / "refs", including_top=False)
    if not _snapshot_items(folder):
        # Nothing of any revision is left: its empty folders go, the
        # repository's too. Otherwise `snapshots/` stays: the scan needs it.
        _prune(folder, including_top=True)


def _named_blobs(folder: Path) -> set[str]:
    """The first hop of every link under the repository's snapshots: the
    blobs some snapshot entry still resolves to."""
    named: set[str] = set()
    for here, dirs, files in os.walk(folder / "snapshots", followlinks=False):
        for name in [*files, *dirs]:
            path = Path(here) / name
            if path.is_symlink():
                try:
                    named.add(_key(Path(here) / os.readlink(path)))
                except OSError:
                    continue
        dirs[:] = [name for name in dirs if not _is_link(Path(here) / name)]
    return named


def _prune(top: Path, *, including_top: bool) -> None:
    """Remove the empty folders under `top`, bottom-up (never a link), and
    `top` itself when asked and empty. A folder that is not empty stays."""
    if _is_link(top) or not top.is_dir():
        return
    for here, dirs, _ in os.walk(top, topdown=False, followlinks=False):
        for name in dirs:
            path = Path(here) / name
            if _is_link(path):
                continue
            try:
                os.rmdir(path)
            except OSError:
                pass  # not empty: something stays in it
    if including_top:
        try:
            os.rmdir(top)
        except OSError:
            pass


def _remove_notes(folder: Path) -> None:
    """A revision's `.no_exist/<commit>`: the Hub client's zero-byte notes of
    files the revision lacks. Best effort: a note left is harmless."""
    if _is_link(folder) or not folder.is_dir():
        return
    for here, dirs, files in os.walk(folder, topdown=False, followlinks=False):
        for name in [*files, *(name for name in dirs if _is_link(Path(here) / name))]:
            try:
                os.unlink(Path(here) / name)
            except OSError:
                log.debug("could not remove a .no_exist note", exc_info=True)
        for name in dirs:
            try:
                os.rmdir(Path(here) / name)
            except OSError:
                pass
    try:
        os.rmdir(folder)
    except OSError:
        pass
